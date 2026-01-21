"""Confidence-based coverage utilities for horizon specialists.

Provides helpers to tune trade thresholds that balance precision and coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    """Summary statistics for a tuned decision threshold."""

    threshold: float
    precision: float
    coverage: float
    trade_count: int
    sample_count: int
    meets_constraints: bool
    coverage_shortfall: float
    min_precision: float
    min_coverage: float

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serialisable dictionary view of the metrics."""
        return {
            "threshold": float(self.threshold),
            "precision": float(self.precision),
            "coverage": float(self.coverage),
            "trade_count": int(self.trade_count),
            "sample_count": int(self.sample_count),
            "meets_constraints": bool(self.meets_constraints),
            "coverage_shortfall": float(self.coverage_shortfall),
            "min_precision": float(self.min_precision),
            "min_coverage": float(self.min_coverage),
        }


def _iter_thresholds() -> Iterable[float]:
    """Generate candidate thresholds in the closed interval [0.50, 0.95]."""
    return np.linspace(0.50, 0.95, num=46, dtype=float)


def _evaluate_threshold(
    y_bin: np.ndarray,
    p_arr: np.ndarray,
    threshold: float,
    sample_count: int,
    min_precision: float,
    min_coverage: float,
) -> ThresholdMetrics:
    mask = (p_arr >= threshold) | (p_arr <= 1.0 - threshold)
    trade_count = int(mask.sum())
    coverage = float(trade_count / sample_count) if sample_count > 0 else 0.0

    if trade_count == 0:
        precision = 0.0
    else:
        y_pred = (p_arr >= threshold).astype(int)
        precision = float((y_pred[mask] == y_bin[mask]).mean()) if trade_count > 0 else 0.0

    meets = (precision >= min_precision) and (coverage >= min_coverage)
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision),
        coverage=float(coverage),
        trade_count=trade_count,
        sample_count=sample_count,
        meets_constraints=bool(meets),
        coverage_shortfall=float(max(0.0, min_coverage - coverage)),
        min_precision=float(min_precision),
        min_coverage=float(min_coverage),
    )


def tune_threshold(
    y_true: Iterable[float],
    proba: Iterable[float],
    min_precision: float = 0.70,
    min_coverage: float = 0.35,
) -> Tuple[float, ThresholdMetrics]:
    """Select a probability threshold that maintains precision while maximising coverage.

    Args:
        y_true: Ground-truth binary labels (0 = down/flat, 1 = up).
        proba: Model probability estimates for the positive class.
        min_precision: Minimum directional hit-rate required across executed trades.
        min_coverage: Minimum proportion of observations that should trigger a trade.

    Returns:
        A tuple of (threshold, metrics). ``threshold`` is the chosen probability cut-off
        for the positive class. ``metrics`` captures precision/coverage statistics for
        the selected threshold, including any shortfall relative to ``min_coverage``.

    Notes:
        Coverage is computed using the symmetric rule ``|p - 0.5| >= (t - 0.5)`` which is
        equivalent to ``(p >= t) or (p <= 1 - t)``. Precision is measured on the executed
        trades only (i.e. those samples satisfying the coverage mask).
    """

    y_arr = np.asarray(list(y_true), dtype=float)
    p_arr = np.asarray(list(proba), dtype=float)

    if y_arr.size == 0 or p_arr.size == 0:
        default_metrics = ThresholdMetrics(
            threshold=0.5,
            precision=0.0,
            coverage=0.0,
            trade_count=0,
            sample_count=int(max(len(y_arr), len(p_arr))),
            meets_constraints=False,
            coverage_shortfall=float(max(0.0, min_coverage)),
            min_precision=float(min_precision),
            min_coverage=float(min_coverage),
        )
        return 0.5, default_metrics

    if y_arr.shape != p_arr.shape:
        min_len = min(len(y_arr), len(p_arr))
        y_arr = y_arr[:min_len]
        p_arr = p_arr[:min_len]

    valid_mask = (~np.isnan(y_arr)) & (~np.isnan(p_arr))
    if not np.all(valid_mask):
        y_arr = y_arr[valid_mask]
        p_arr = p_arr[valid_mask]

    y_bin = (y_arr > 0.5).astype(int)
    sample_count = int(y_bin.size)

    if sample_count == 0:
        default_metrics = ThresholdMetrics(
            threshold=0.5,
            precision=0.0,
            coverage=0.0,
            trade_count=0,
            sample_count=0,
            meets_constraints=False,
            coverage_shortfall=float(max(0.0, min_coverage)),
            min_precision=float(min_precision),
            min_coverage=float(min_coverage),
        )
        return 0.5, default_metrics

    metrics_list = [
        _evaluate_threshold(
            y_bin=y_bin,
            p_arr=p_arr,
            threshold=float(threshold),
            sample_count=sample_count,
            min_precision=min_precision,
            min_coverage=min_coverage,
        )
        for threshold in _iter_thresholds()
    ]

    candidate_pool = [m for m in metrics_list if m.meets_constraints]
    if candidate_pool:
        chosen = max(candidate_pool, key=lambda m: (m.coverage, -m.threshold))
        return chosen.threshold, chosen

    fallback = max(metrics_list, key=lambda m: (m.precision, m.coverage, -m.threshold))
    return fallback.threshold, fallback