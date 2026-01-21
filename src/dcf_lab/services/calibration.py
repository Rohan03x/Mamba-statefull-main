from typing import Any, Callable, Dict

import numpy as np
from scipy import stats

from ..metrics.calibration import (
    bhattacharyya_distance,
    brier,
    calculate_overlap,
    crps_from_quantiles,
    ece,
    picp,
    pit_samples,
)

MetricFn = Callable[..., float]

# Define metric registries for different types of calibration tasks
REGRESSION = {
    "picp@80": lambda y, ctx: picp(y, ctx["q10"], ctx["q90"]),
    "crps": lambda y, ctx: crps_from_quantiles(y, ctx["quantiles"]),
    # returns dict; merge later
    "pit": lambda y, ctx: pit_samples(y, ctx["cdf"])
}

CLASSIFICATION = {
    "brier": lambda y, ctx: brier(y, ctx["p"]),
    "ece": lambda y, ctx: ece(y, ctx["p"], bins=10),
}

# Distribution comparison metrics
DIST_COMPARISON = {
    "market_median": lambda d1,
    d2: float(
        np.median(d1)),
    "forecast_median": lambda d1,
    d2: float(
        np.median(d2)),
    "market_std": lambda d1,
    d2: float(
        np.std(d1)),
    "forecast_std": lambda d1,
    d2: float(
        np.std(d2)),
    "percent_diff_median": lambda d1,
    d2: float(
        (np.median(d2) - np.median(d1)) / np.median(d1) * 100),
    "percent_diff_std": lambda d1,
    d2: float(
        (np.std(d2) - np.std(d1)) / np.std(d1) * 100),
    "ks_statistic": lambda d1,
    d2: float(
        stats.ks_2samp(
            d1,
            d2).statistic),
    "p_value": lambda d1,
    d2: float(
        stats.ks_2samp(
            d1,
            d2).pvalue),
    "forecast_upside": lambda d1,
    d2: float(
        np.median(d2) / np.median(d1) - 1),
    "overlap_coefficient": lambda d1,
    d2: calculate_overlap(
        d1,
        d2)}

# Parameter distribution metrics for normal distributions
NORMAL_DIST_COMPARISON = {
    "market_mean": lambda m1, s1, m2, s2: m1,
    "market_std": lambda m1, s1, m2, s2: s1,
    "forecast_mean": lambda m1, s1, m2, s2: m2,
    "forecast_std": lambda m1, s1, m2, s2: s2,
    "percent_diff_mean": lambda m1, s1, m2, s2:
        float((m2 - m1) / m1 * 100) if abs(m1) > 1e-10 else float('inf'),
    "percent_diff_std": lambda m1, s1, m2, s2:
        float((s2 - s1) / s1 * 100) if abs(s1) > 1e-10 else float('inf'),
    "z_score_diff": lambda m1, s1, m2, s2:
        float(abs(m2 - m1) / s1) if abs(s1) > 1e-10 else float('inf'),
    "bhattacharyya_distance": lambda m1, s1, m2, s2:
        bhattacharyya_distance(m1, s1, m2, s2)
}


def calculate_calibration_metrics(
    y_true: np.ndarray,
    context: Dict[str, Any],
    task_type: str = "regression"
) -> Dict[str, float]:
    """
    Calculate calibration metrics for a given task type.

    Args:
        y_true: True values
        context: Dictionary with inputs required by the metrics
        task_type: Type of task ('regression' or 'classification')

    Returns:
        Dictionary of calibration metrics
    """
    metrics = REGRESSION if task_type == "regression" else CLASSIFICATION
    out = {}

    for name, fn in metrics.items():
        val = fn(y_true, context)
        if isinstance(val, dict):  # e.g., PIT summary
            out.update({f"{name}_{k}": v for k, v in val.items()})
        else:
            out[name] = val

    return out


def calculate_distribution_comparison(
    market_values: np.ndarray,
    forecast_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comparison metrics between two value distributions.

    Args:
        market_values: Market-implied values
        forecast_values: Forecast model values

    Returns:
        Dictionary of comparison metrics
    """
    metrics = {}

    for name, fn in DIST_COMPARISON.items():
        metrics[name] = fn(market_values, forecast_values)

    return metrics


def calculate_normal_distribution_metrics(
    market_mean: float,
    market_std: float,
    forecast_mean: float,
    forecast_std: float
) -> Dict[str, float]:
    """
    Calculate metrics comparing two normal distributions.

    Args:
        market_mean: Mean of market-implied distribution
        market_std: Standard deviation of market-implied distribution
        forecast_mean: Mean of forecast distribution
        forecast_std: Standard deviation of forecast distribution

    Returns:
        Dictionary of comparison metrics
    """
    metrics = {}

    for name, fn in NORMAL_DIST_COMPARISON.items():
        metrics[name] = fn(
            market_mean,
            market_std,
            forecast_mean,
            forecast_std)

    return metrics
