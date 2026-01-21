#!/usr/bin/env python3
"""Convenience wrapper: materialize Dagster prep-families assets for a (symbol, horizon).

Why this exists:
- Most family assets are partitioned by symbol only.
- A small set (quantile_forecast, calibration, online_learning, plus horizon-bound HF blocks) are
  partitioned by (horizon|symbol).
Dagster cannot materialize assets with mismatched PartitionsDefinitions in the same run, so we launch
(two) runs in the right order.

Important:
- The default (two-run) path materializes family caches only; it does NOT materialize
    the stage-b `prep_families_manifest` asset (which writes the merged parquet under cache/features).
- Use `--single-run` to run `prep_families_manifest` only, or `--with-merged` to run caches first
    and then materialize the merged parquet.

Ordering note:
- The horizon-bound dependency chain (quantile_forecast → calibration → online_learning) must
    complete before other horizon-bound families (HF blocks) to avoid cache-miss cascades.
    This wrapper materializes that chain first.

Example:
  .venv/bin/python tools/run_dagster_prep_families_all.py --symbol AAPL --horizon 63 --background
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    args: list[str]
    log_path: Path | None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "bin" / "python"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(spec: RunSpec, env: dict[str, str], *, wait: bool) -> int:
    if spec.log_path is None:
        print(f"[run] {spec.name}: {' '.join(spec.args)}")
        return subprocess.call(spec.args, env=env)

    _ensure_parent(spec.log_path)
    print(f"[bg]  {spec.name}: {' '.join(spec.args)}")
    print(f"      log={spec.log_path}")
    with spec.log_path.open("wb") as f:
        proc = subprocess.Popen(spec.args, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"      pid={proc.pid}")
    if not wait:
        return 0
    return int(proc.wait())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument(
        "--dagster-home",
        default=None,
        help="Defaults to <repo>/.dagster_home (recommended).",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Path to python executable (defaults to <repo>/.venv/bin/python).",
    )
    parser.add_argument(
        "--module",
        default="dagster_prep_families",
        help="Dagster code location module (default: dagster_prep_families).",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch runs in background and write logs under artifacts/logs/.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "In --background mode, do not wait for each run to finish. "
            "Warning: this disables strict sequencing and can cause cache-miss failures."
        ),
    )
    parser.add_argument(
        "--with-merged",
        action="store_true",
        help=(
            "After materializing family caches (two-run flow), also materialize prep_families_manifest "
            "for this (symbol,horizon) partition to write cache/features/<SYMBOL>_h<H>_merged.parquet."
        ),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=(
            "Run prep_families_manifest (stage-b) for this (symbol,horizon) partition. "
            "This produces cache/features/<SYMBOL>_h<H>_merged.parquet in one Dagster run."
        ),
    )

    args = parser.parse_args(argv)

    repo_root = _repo_root()
    dagster_home = Path(args.dagster_home) if args.dagster_home else (repo_root / ".dagster_home")
    python_exe = Path(args.python) if args.python else _default_python(repo_root)

    if not python_exe.exists():
        raise SystemExit(f"Python not found at: {python_exe}")

    # Use python -m dagster to handle paths with spaces
    dagster_cmd = [str(python_exe), "-m", "dagster"]

    env = os.environ.copy()
    env["DAGSTER_HOME"] = str(dagster_home)

    # Reduce burst rate to avoid transient provider blocks (e.g., 403) during highly parallel runs.
    # Can be overridden by the caller.
    env.setdefault("EODHD_MIN_REQUEST_INTERVAL", "0.25")

    symbol = args.symbol
    horizon = int(args.horizon)
    horizon_pk = f"{horizon}|{symbol}"

    # Ensure the Dagster partitions definitions include this (symbol,horizon).
    # The dagster_prep_families code location reads DAGSTER_SYMBOLS / DAGSTER_HORIZONS
    # to build partition keys; without this, materialize can fail with
    # DagsterInvalidSubsetError for symbols not in the configured universe.
    env["DAGSTER_SYMBOLS"] = str(symbol).upper()
    env["DAGSTER_HORIZONS"] = str(horizon)

    # We default to strict sequencing (wait between runs) to ensure the sequential chain
    # completes before HF blocks and to avoid cross-run cache races.
    wait = not bool(args.detach)

    ts = _timestamp()
    logs_dir = repo_root / "artifacts" / "logs"

    symbol_log = logs_dir / f"dagster_materialize_all_symbol_{symbol}_{ts}.log" if args.background else None
    horizon_log = (
        logs_dir / f"dagster_materialize_all_horizon_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )

    merged_log = (
        logs_dir / f"dagster_materialize_with_merged_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )

    single_log = (
        logs_dir / f"dagster_materialize_single_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )

    if args.single_run:
        single_spec = RunSpec(
            name="prep_families_manifest(single-run)",
            args=dagster_cmd + [
                "asset",
                "materialize",
                "-m",
                args.module,
                "--select",
                "prep_families_manifest",
                "--partition",
                horizon_pk,
            ],
            log_path=single_log,
        )
        return _run(single_spec, env=env, wait=wait)

    if not args.with_merged:
        print(
            "[note] This default mode materializes family caches only (no merged parquet). "
            "Use --with-merged or --single-run to write cache/features/<SYMBOL>_h<H>_merged.parquet."
        )

    # 1) Symbol-only families
    symbol_spec = RunSpec(
        name="families(symbol-only)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "group:families",
            "--partition",
            symbol,
        ],
        log_path=symbol_log,
    )

    # 2) Horizon-bound families
    horizon_chain_spec = RunSpec(
        name="families(horizon-chain)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "family_quantile_forecast,family_calibration,family_online_learning",
            "--partition",
            horizon_pk,
        ],
        log_path=horizon_log,
    )

    horizon_rest_log = (
        logs_dir / f"dagster_materialize_all_horizon_rest_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )
    horizon_rest_spec = RunSpec(
        name="families(horizon-rest)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "family_forecast_hf,family_hf_agg",
            "--partition",
            horizon_pk,
        ],
        log_path=horizon_rest_log,
    )

    # 3) Unified merged parquet (stage-b)
    merged_spec = RunSpec(
        name="prep_families_manifest(merged)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "prep_families_manifest",
            "--partition",
            horizon_pk,
        ],
        log_path=merged_log,
    )

    rc = _run(symbol_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    rc = _run(horizon_chain_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    rc = _run(horizon_rest_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    if args.with_merged:
        return _run(merged_spec, env=env, wait=wait)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
