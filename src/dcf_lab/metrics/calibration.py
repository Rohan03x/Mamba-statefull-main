from typing import Dict, Mapping

import numpy as np


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate the Brier score for probability predictions.

    Args:
        y: Actual values (0 or 1)
        p: Predicted probabilities

    Returns:
        Brier score (mean squared error of probabilities)
    """
    return float(np.mean((p - y) ** 2))


def picp(y: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """
    Calculate the Prediction Interval Coverage Probability (PICP).

    Args:
        y: Actual values
        q_low: Lower quantile predictions
        q_high: Upper quantile predictions

    Returns:
        PICP score (fraction of observations within the interval)
    """
    covered = (y >= q_low) & (y <= q_high)
    return float(np.mean(covered))


def crps_from_quantiles(y: np.ndarray, q: Mapping[float, np.ndarray]) -> float:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) from quantiles.

    Args:
        y: Actual values
        q: Dictionary mapping quantile levels to predicted quantiles

    Returns:
        CRPS score (integrated absolute error between CDFs)
    """
    # pinball approximation over supplied quantiles
    taus = np.array(sorted(q.keys()))
    errs = [
        np.mean(
            np.maximum(
                taus[i] * (y - q[taus[i]]),
                (taus[i] - 1) * (y - q[taus[i]]))) for i in range(len(taus))]
    return float(np.trapz(errs, taus))


def pit_samples(y: np.ndarray, cdf_vals: np.ndarray) -> Dict[str, float]:
    """
    Calculate Probability Integral Transform (PIT) statistics.

    Args:
        y: Actual values
        cdf_vals: Values from the CDF at points corresponding to y

    Returns:
        Dictionary of PIT statistics
    """
    # return summary stats; KS test can be added if already in deps
    return {
        "pit_mean": float(
            np.mean(cdf_vals)), "pit_std": float(
            np.std(cdf_vals))}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        y: Actual values (0 or 1)
        p: Predicted probabilities
        bins: Number of bins to use for calibration assessment

    Returns:
        ECE score (weighted average of calibration errors across bins)
    """
    # expected calibration error for classification-like probs
    bin_edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, bin_edges) - 1
    e = 0.0
    n = len(y)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        e += (m.sum()/n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def bhattacharyya_distance(
        m1: float,
        s1: float,
        m2: float,
        s2: float) -> float:
    """
    Calculate the Bhattacharyya distance between two normal distributions.

    Args:
        m1: Mean of first distribution
        s1: Standard deviation of first distribution
        m2: Mean of second distribution
        s2: Standard deviation of second distribution

    Returns:
        Bhattacharyya distance
    """
    term1 = 0.25 * np.log(0.25 * ((s1**2 / s2**2) + (s2**2 / s1**2) + 2))
    term2 = 0.25 * ((m1 - m2)**2 / (s1**2 + s2**2))
    return float(term1 + term2)


def calculate_overlap(
        dist1: np.ndarray,
        dist2: np.ndarray,
        bins: int = 100) -> float:
    """
    Calculate the overlap coefficient between two empirical distributions.

    Args:
        dist1: First distribution samples
        dist2: Second distribution samples
        bins: Number of bins for histogram

    Returns:
        Overlap coefficient (0-1)
    """
    # Calculate histograms with the same bins
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Calculate normalized histograms
    hist1, _ = np.histogram(dist1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)

    # Calculate the overlap as the sum of the minimum values at each bin
    bin_width = bin_edges[1] - bin_edges[0]
    overlap = np.sum(np.minimum(hist1, hist2)) * bin_width

    return float(overlap)
