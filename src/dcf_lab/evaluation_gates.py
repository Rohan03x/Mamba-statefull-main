"""Evaluation gate helpers for enforcing deployment thresholds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "EvaluationGateConfig",
    "GateMetric",
    "GateSummary",
    "evaluate_gates",
]


@dataclass(slots=True)
class EvaluationGateConfig:
    """Thresholds and knobs for evaluation gates."""

    min_precision: float = 0.70
    min_coverage: float = 0.45
    min_accuracy: float = 0.70
    coverage_confidence: float = 0.15
    sharpe_floor: float = 0.0
    sortino_floor: float = 0.0
    drift_lookback_days: int = 30


@dataclass(slots=True)
class GateMetric:
    precision: float
    coverage: float
    accuracy: float
    sharpe: float
    sortino: float
    confident: int
    total: int


@dataclass(slots=True)
class GateSummary:
    passed: bool
    failures: List[str]
    overall: GateMetric
    per_symbol: Dict[str, Dict[str, GateMetric]]
    drift_events: List[Dict[str, Any]]
    rl_metrics: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "overall": asdict(self.overall),
            "per_symbol": {
                sym: {hz: asdict(metric) for hz, metric in horizons.items()}
                for sym, horizons in self.per_symbol.items()
            },
            "drift_events": self.drift_events,
            "rl_metrics": self.rl_metrics,
        }


def evaluate_gates(
    symbol_results: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    *,
    rl_metrics: Optional[Mapping[str, Any]] = None,
    drift_log: Optional[Path] = None,
    config: Optional[EvaluationGateConfig] = None,
) -> GateSummary:
    cfg = config or EvaluationGateConfig()

    per_symbol, metric_pool, total_confident, total_predictions = _collect_symbol_metrics(symbol_results, cfg)
    if not metric_pool:
        default_metric = GateMetric(precision=0.0, coverage=0.0, accuracy=0.0, sharpe=0.0, sortino=0.0, confident=0, total=0)
        return GateSummary(
            passed=False,
            failures=["insufficient_data"],
            overall=default_metric,
            per_symbol=per_symbol,
            drift_events=[],
            rl_metrics=dict(rl_metrics) if rl_metrics else None,
        )

    overall_metric = _aggregate_overall(metric_pool, total_confident, total_predictions)
    drift_events = _recent_drift_events(drift_log, cfg.drift_lookback_days) if drift_log else []
    failures = _determine_failures(overall_metric, rl_metrics, drift_events, cfg)

    return GateSummary(
        passed=len(failures) == 0,
        failures=failures,
        overall=overall_metric,
        per_symbol=per_symbol,
        drift_events=drift_events,
        rl_metrics=dict(rl_metrics) if rl_metrics else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_symbol_metrics(
    symbol_results: Mapping[str, Mapping[Any, Mapping[str, Any]]],
    cfg: EvaluationGateConfig,
) -> tuple[Dict[str, Dict[str, GateMetric]], List[GateMetric], int, int]:
    per_symbol: Dict[str, Dict[str, GateMetric]] = {}
    pool: List[GateMetric] = []
    total_confident = 0
    total_predictions = 0

    for symbol, horizons in symbol_results.items():
        metrics_for_symbol: Dict[str, GateMetric] = {}
        for horizon, payload in horizons.items():
            probs = _to_array(payload.get("probabilities"))
            targets = _to_array(payload.get("targets"))
            returns = _to_array(payload.get("returns"))
            if probs.size == 0 or targets.size == 0:
                continue
            metric = _compute_metric(probs, targets, returns, cfg)
            metrics_for_symbol[str(horizon)] = metric
            pool.append(metric)
            total_confident += metric.confident
            total_predictions += metric.total
        if metrics_for_symbol:
            per_symbol[symbol] = metrics_for_symbol

    return per_symbol, pool, total_confident, total_predictions


def _aggregate_overall(metrics: Sequence[GateMetric], confident: int, total: int) -> GateMetric:
    precision = float(np.mean([m.precision for m in metrics]))
    coverage = float(np.mean([m.coverage for m in metrics]))
    accuracy = float(np.mean([m.accuracy for m in metrics]))
    sharpe = float(np.mean([m.sharpe for m in metrics]))
    sortino = float(np.mean([m.sortino for m in metrics]))
    return GateMetric(
        precision=precision,
        coverage=coverage,
        accuracy=accuracy,
        sharpe=sharpe,
        sortino=sortino,
        confident=int(confident),
        total=int(total),
    )


def _determine_failures(
    overall: GateMetric,
    rl_metrics: Optional[Mapping[str, Any]],
    drift_events: Sequence[Mapping[str, Any]],
    cfg: EvaluationGateConfig,
) -> List[str]:
    failures: List[str] = []
    if overall.precision < cfg.min_precision:
        failures.append("precision")
    if overall.coverage < cfg.min_coverage:
        failures.append("coverage")
    if overall.accuracy < cfg.min_accuracy:
        failures.append("accuracy")
    if overall.sharpe < cfg.sharpe_floor:
        failures.append("sharpe")
    if overall.sortino < cfg.sortino_floor:
        failures.append("sortino")
    if drift_events:
        failures.append("drift")

    if rl_metrics:
        rl_prec = float(rl_metrics.get("precision", 0.0))
        rl_cov = float(rl_metrics.get("coverage", 0.0))
        rl_acc = float(rl_metrics.get("accuracy", 0.0))
        if rl_prec < cfg.min_precision:
            failures.append("rl_precision")
        if rl_cov < cfg.min_coverage:
            failures.append("rl_coverage")
        if rl_acc < cfg.min_accuracy:
            failures.append("rl_accuracy")

    return failures


def _compute_metric(
    probabilities: np.ndarray,
    targets: np.ndarray,
    returns: np.ndarray,
    cfg: EvaluationGateConfig,
) -> GateMetric:
    probs = probabilities.astype(np.float32)
    targs = targets.astype(np.float32)
    if returns.size == 0 or returns.shape[0] != probs.shape[0]:
        pseudo_returns = np.where((probs > 0.5) == (targs > 0.5), 0.01, -0.01)
        rets = pseudo_returns.astype(np.float32)
    else:
        rets = returns.astype(np.float32)

    confidence_mask = np.abs(probs - 0.5) >= cfg.coverage_confidence
    confident = int(confidence_mask.sum())
    total = int(len(probs))
    coverage = float(confident / total) if total > 0 else 0.0

    if confident > 0:
        confident_preds = probs[confidence_mask] > 0.5
        confident_targets = targs[confidence_mask] > 0.5
        precision = float((confident_preds == confident_targets).mean())
    else:
        precision = 0.0

    accuracy = float(((probs > 0.5) == (targs > 0.5)).mean()) if total > 0 else 0.0

    sharpe = _calc_sharpe(rets)
    sortino = _calc_sortino(rets)

    return GateMetric(
        precision=precision,
        coverage=coverage,
        accuracy=accuracy,
        sharpe=sharpe,
        sortino=sortino,
        confident=confident,
        total=total,
    )


def _calc_sharpe(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    if std <= 1e-6:
        std = 1e-6
    return mean / std


def _calc_sortino(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mean = float(np.mean(returns))
    downside = returns[returns < 0]
    denom = float(np.std(downside)) if downside.size else 1e-6
    if denom <= 1e-6:
        denom = 1e-6
    return mean / denom


def _recent_drift_events(path: Path, lookback_days: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    events: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_raw = payload.get("timestamp")
                if not ts_raw:
                    continue
                try:
                    timestamp = datetime.fromisoformat(ts_raw)
                except ValueError:
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if timestamp >= cutoff:
                    events.append(payload)
    except Exception:
        return []
    return events


def _to_array(values: Optional[Sequence[float]]) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float32)
    try:
        return np.asarray(list(values), dtype=np.float32)
    except Exception:
        return np.asarray([], dtype=np.float32)
