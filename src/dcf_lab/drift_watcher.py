"""Automated drift watcher and adaptive response pipeline.

This module builds on the existing drift monitors in ``dcf_lab.drift`` to
provide an orchestration layer that

* ingests recent walk-forward diagnostics (accuracy / coverage per window),
* evaluates rolling error streams with ADWIN and Page-Hinkley,
* logs structured drift events for observability,
* triggers light retuning via ``tools/auto_meta.py`` when drift is detected,
* escalates to partial retraining jobs if degradation persists for multiple
  months, and
* exposes a small CLI so the watcher can be scheduled monthly and/or invoked
  immediately when fresh drift evidence is produced.

The implementation intentionally reuses the existing monitor classes, avoids
retraining logic duplication, and relies on configuration so teams can adjust
thresholds or wire the retraining hook into internal orchestration systems.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    # Reuse diagnostics ingestion helpers from auto_meta to avoid duplication.
    from tools.auto_meta import (
        WindowRecord,
        collect_diagnostic_files,
        load_window_records,
    )
except Exception as exc:  # pragma: no cover - defensive
    raise RuntimeError(
        "tools.auto_meta is required for drift_watcher; ensure the repository root is on PYTHONPATH"
    ) from exc

from dcf_lab.drift.monitors import ADWINMonitor, DriftAlert, PageHinkleyMonitor


LOGGER = logging.getLogger(__name__)

Key = Tuple[str, int, str]  # (symbol, horizon, regime)


@dataclass
class DriftWatcherConfig:
    """Runtime configuration for :class:`DriftWatcher`."""

    runs_root: Path = Path("runs")
    events_log: Path = Path("logs/drift_events.jsonl")
    state_path: Path = Path("config/drift_watcher_state.json")
    auto_meta_script: Path = Path("tools/auto_meta.py")
    easy_launcher_script: Path = Path("easy_launcher.py")
    python_executable: str = sys.executable

    # Monitor sensitivity
    adwin_delta: float = 0.002
    adwin_min_window: int = 30
    adwin_max_window: int = 500
    ph_threshold: float = 40.0
    ph_alpha: float = 0.998
    ph_min_detections: int = 5

    # Retune / retrain policy
    auto_meta_iterations: int = 40
    auto_meta_init_points: int = 12
    auto_meta_window_months: int = 6
    retune_cooldown_days: int = 7
    degradation_months: int = 2
    retrain_years: int = 6
    retrain_cooldown_days: int = 90
    lookback_symbols: Optional[Sequence[str]] = None
    lookback_horizons: Optional[Sequence[int]] = None

    # Retrain hook toggle
    enable_retrain: bool = True

    # Additional CLI args for spawned jobs
    auto_meta_extra_args: Sequence[str] = ()
    retrain_extra_args: Sequence[str] = ()


@dataclass
class DriftEventRecord:
    """Serializable drift event for logging and auditing."""

    timestamp: str
    detector: str
    symbol: str
    horizon: int
    regime: str
    severity: str
    metric_value: float
    threshold: float
    recommendations: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


@dataclass
class _MonitorBundle:
    adwin: ADWINMonitor
    page_hinkley: PageHinkleyMonitor


class DriftWatcher:
    """Evaluate rolling diagnostics streams and trigger adaptive responses."""

    def __init__(self, config: DriftWatcherConfig):
        self.config = config
        self._monitors: Dict[Key, _MonitorBundle] = {}
        self.state = self._load_state()
        self.config.events_log.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def monitor(
        self,
        runs: Optional[Sequence[str]] = None,
        symbols: Optional[Sequence[str]] = None,
        horizons: Optional[Sequence[int]] = None,
        dry_run: bool = False,
    ) -> None:
        """Run drift detection on available diagnostics and trigger actions."""

        records = self._load_records(runs, symbols, horizons)
        if not records:
            LOGGER.info("No diagnostics records found for drift monitoring.")
            return

        grouped: Dict[Key, List[WindowRecord]] = {}
        for record in records:
            key = (record.symbol, int(record.horizon), record.regime)
            grouped.setdefault(key, []).append(record)

        for key, rows in grouped.items():
            self._process_group(key, rows, dry_run)

        self._save_state()

    def _process_group(self, key: Key, rows: Sequence[WindowRecord], dry_run: bool) -> None:
        def _window_ts(record: WindowRecord) -> datetime:
            ts = record.test_end
            if ts is None:
                return datetime.now(timezone.utc)
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        sorted_rows = sorted(rows, key=_window_ts)
        monitors = self._get_monitors(key)
        for row in sorted_rows:
            timestamp = pd.Timestamp(_window_ts(row))
            error = max(0.0, 1.0 - float(row.accuracy))
            adwin_alert = monitors.adwin.add_element(error, timestamp)
            page_alert = monitors.page_hinkley.add_element(error, timestamp)

            for detector, alert in ("adwin", adwin_alert), ("page_hinkley", page_alert):
                if alert is not None:
                    self._handle_alert(key, detector, alert, row, dry_run)

    def run_monthly(self, runs: Optional[Sequence[str]] = None, dry_run: bool = False) -> None:
        """Run monthly light retuning regardless of drift alerts."""

        if not self._should_run_global_retune(dry_run):
            LOGGER.info("Skipping monthly retune – cooldown active.")
            return

        diag_files = self._resolve_runs(runs)
        recent = self._select_recent(diag_files, self.config.auto_meta_window_months)
        if not recent:
            LOGGER.warning("Monthly retune skipped – no recent diagnostics found.")
            return

        self._invoke_auto_meta(recent, dry_run=dry_run)
        self.state.setdefault("global", {})["last_retune"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_monitors(self, key: Key) -> _MonitorBundle:
        if key not in self._monitors:
            self._monitors[key] = _MonitorBundle(
                adwin=ADWINMonitor(
                    delta=self.config.adwin_delta,
                    min_window_size=self.config.adwin_min_window,
                    max_window_size=self.config.adwin_max_window,
                ),
                page_hinkley=PageHinkleyMonitor(
                    threshold=self.config.ph_threshold,
                    alpha=self.config.ph_alpha,
                    min_detections=self.config.ph_min_detections,
                ),
            )
        return self._monitors[key]

    def _handle_alert(
        self,
        key: Key,
        detector: str,
        alert: DriftAlert,
        row: WindowRecord,
        dry_run: bool,
    ) -> None:
        symbol, horizon, regime = key
        LOGGER.warning(
            "Drift detected | detector=%s symbol=%s horizon=%s regime=%s severity=%s metric=%.4f",
            detector,
            symbol,
            horizon,
            regime,
            alert.severity.value,
            alert.metric_value,
        )

        self._log_event(
            DriftEventRecord(
                timestamp=str(alert.timestamp),
                detector=detector,
                symbol=symbol,
                horizon=horizon,
                regime=regime,
                severity=alert.severity.value,
                metric_value=alert.metric_value,
                threshold=alert.threshold,
                recommendations=alert.recommendations,
                metadata={
                    "coverage": row.coverage,
                    "return_weighted_acc": row.return_weighted_acc,
                    "sharpe": row.sharpe,
                    "sortino": row.sortino,
                },
            )
        )

        now = datetime.now(timezone.utc)
        if self._should_trigger_retune(key, now):
            recent = self._select_recent(self._resolve_runs(None), self.config.auto_meta_window_months)
            if recent:
                self._invoke_auto_meta(recent, dry_run=dry_run)
                self._mark_retune(key, now)
            else:
                LOGGER.warning("Retune skipped – no recent diagnostics found for auto_meta.")

        if self.config.enable_retrain:
            self._update_degradation(key, now, dry_run)

    def _should_trigger_retune(self, key: Key, timestamp: datetime) -> bool:
        retune_state = self.state.setdefault("retune", {})
        entry = retune_state.get(self._key_to_str(key))
        if entry is None:
            return True
        last_retune = datetime.fromisoformat(entry["last_retune"])
        cooldown = timedelta(days=self.config.retune_cooldown_days)
        return (timestamp - last_retune) >= cooldown

    def _mark_retune(self, key: Key, timestamp: datetime) -> None:
        retune_state = self.state.setdefault("retune", {})
        retune_state[self._key_to_str(key)] = {"last_retune": timestamp.isoformat()}
        self._save_state()

    def _update_degradation(self, key: Key, timestamp: datetime, dry_run: bool) -> None:
        degradation_state = self.state.setdefault("degradation", {})
        key_str = self._key_to_str(key)
        entry = degradation_state.get(key_str)
        if entry is None:
            degradation_state[key_str] = {
                "first_detected": timestamp.isoformat(),
                "last_seen": timestamp.isoformat(),
                "retrained_at": None,
            }
            self._save_state()
            return

        entry["last_seen"] = timestamp.isoformat()
        self._save_state()

        first = datetime.fromisoformat(entry["first_detected"])
        if entry.get("retrained_at"):
            retrained_at = datetime.fromisoformat(entry["retrained_at"])
            if (timestamp - retrained_at) < timedelta(days=self.config.retrain_cooldown_days):
                return

        if (timestamp - first) >= timedelta(days=30 * self.config.degradation_months):
            LOGGER.warning(
                "Persistent degradation detected for %s – triggering partial retrain.",
                key_str,
            )
            self._spawn_retrain_job(key, dry_run=dry_run)
            entry["retrained_at"] = timestamp.isoformat()
            entry["first_detected"] = timestamp.isoformat()
            self._save_state()

    def _spawn_retrain_job(self, key: Key, dry_run: bool) -> None:
        symbol, horizon, regime = key
        start_years = self.config.retrain_years
        start_date = (datetime.now(timezone.utc) - timedelta(days=365 * start_years)).strftime("%Y-%m-%d")

        cmd = [
            self.config.python_executable,
            str(self.config.easy_launcher_script),
            "--walkforward",
            "--symbols",
            symbol,
            "--horizons",
            str(int(horizon)),
            "--wf-start",
            start_date,
            "--wf-train-years",
            str(start_years),
            "--wf-step-years",
            "1",
        ]
        if self.config.retrain_extra_args:
            cmd.extend(self.config.retrain_extra_args)

        env = os.environ.copy()
        env["DRIFT_TARGET_REGIME"] = regime
        env.setdefault("PYTHONUNBUFFERED", "1")

        if dry_run:
            LOGGER.info("[dry-run] Retrain command: %s", " ".join(cmd))
            return

        LOGGER.info("Spawning partial retrain job: %s", " ".join(cmd))
        try:
            subprocess.Popen(cmd, env=env, cwd=self.config.easy_launcher_script.parent)
        except Exception as exc:
            LOGGER.error("Failed to spawn retrain job: %s", exc)

    def _invoke_auto_meta(self, runs: Sequence[Path], dry_run: bool) -> None:
        cmd = [
            self.config.python_executable,
            str(self.config.auto_meta_script),
            "--runs",
            *[str(path) for path in runs],
            "--iterations",
            str(self.config.auto_meta_iterations),
            "--init-points",
            str(self.config.auto_meta_init_points),
        ]
        if self.config.auto_meta_extra_args:
            cmd.extend(self.config.auto_meta_extra_args)

        if dry_run:
            LOGGER.info("[dry-run] auto_meta command: %s", " ".join(cmd))
            return

        LOGGER.info("Running auto_meta retune: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=self.config.auto_meta_script.parent)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("auto_meta failed with return code %s", exc.returncode)
        except Exception as exc:
            LOGGER.error("Failed to invoke auto_meta: %s", exc)

    # ------------------------------------------------------------------
    # Record ingestion / selection helpers
    # ------------------------------------------------------------------

    def _load_records(
        self,
        runs: Optional[Sequence[str]],
        symbols: Optional[Sequence[str]],
        horizons: Optional[Sequence[int]],
    ) -> List[WindowRecord]:
        diag_files = self._resolve_runs(runs)
        if not diag_files:
            return []
        records = load_window_records(diag_files)
        sym_filter = set(symbols or self.config.lookback_symbols or [])
        hor_filter = {int(h) for h in (horizons or self.config.lookback_horizons or [])}

        filtered: List[WindowRecord] = []
        for record in records:
            if sym_filter and record.symbol not in sym_filter:
                continue
            if hor_filter and int(record.horizon) not in hor_filter:
                continue
            filtered.append(record)
        return filtered

    def _resolve_runs(self, runs: Optional[Sequence[str]]) -> List[Path]:
        if runs:
            candidates = runs
        else:
            candidates = [str(self.config.runs_root)]
        return collect_diagnostic_files(candidates)

    def _select_recent(self, files: Sequence[Path], months: int) -> List[Path]:
        if months <= 0:
            return list(files)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
        recent: List[Path] = []
        for path in files:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except FileNotFoundError:
                continue
            if mtime >= cutoff:
                recent.append(path)
        return recent

    # ------------------------------------------------------------------
    # State & logging utilities
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, Dict[str, object]]:
        if self.config.state_path.exists():
            try:
                return json.loads(self.config.state_path.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.warning("Failed to load drift watcher state; starting fresh.")
        return {}

    def _save_state(self) -> None:
        try:
            self.config.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception as exc:
            LOGGER.error("Failed to persist drift watcher state: %s", exc)

    def _log_event(self, event: DriftEventRecord) -> None:
        try:
            with self.config.events_log.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")
        except Exception as exc:
            LOGGER.error("Failed to append drift event log: %s", exc)

    def _should_run_global_retune(self, dry_run: bool) -> bool:
        if dry_run:
            return True
        global_state = self.state.setdefault("global", {})
        last = global_state.get("last_retune")
        if not last:
            return True
        last_dt = datetime.fromisoformat(last)
        return (datetime.now(timezone.utc) - last_dt) >= timedelta(days=30)

    @staticmethod
    def _key_to_str(key: Key) -> str:
        symbol, horizon, regime = key
        return f"{symbol}|{int(horizon)}|{regime}"


# ----------------------------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor drift diagnostics and trigger adaptive responses")
    parser.add_argument("--runs", nargs="*", help="Optional run directories or diagnostics files")
    parser.add_argument("--symbols", nargs="*", help="Restrict monitoring to specific symbols")
    parser.add_argument("--horizons", nargs="*", type=int, help="Restrict monitoring to specific horizons")
    parser.add_argument("--mode", choices=["drift", "monthly"], default="drift", help="Execution mode")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing subprocesses")
    parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    config = DriftWatcherConfig()
    watcher = DriftWatcher(config)

    if args.mode == "monthly":
        watcher.run_monthly(args.runs, dry_run=args.dry_run)
    else:
        watcher.monitor(args.runs, args.symbols, args.horizons, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
