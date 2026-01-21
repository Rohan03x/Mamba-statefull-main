"""
Distribution Helper Functions

This module provides helper functions for creating various distribution specifications
for use in Monte Carlo DCF and other valuation models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from .dcf_mc import DistributionSpec, constant_value


def _derive_bounds(
    base_case: float,
    low_case: Optional[float],
    high_case: Optional[float],
    distribution_type: str,
    std_dev: Optional[float]
) -> tuple[float, float]:
    """
    Derive lower and upper bounds for the distribution.

    Args:
        base_case: Base case value
        low_case: Low case value (if provided)
        high_case: High case value (if provided)
        distribution_type: Type of distribution
        std_dev: Standard deviation (for normal distribution)

    Returns:
        Tuple of (low_case, high_case)
    """
    # Derive low case if not provided
    if low_case is None:
        if std_dev is not None and distribution_type == "normal":
            low_case = base_case - 2 * std_dev
        else:
            low_case = base_case * 0.8

    # Derive high case if not provided
    if high_case is None:
        if std_dev is not None and distribution_type == "normal":
            high_case = base_case + 2 * std_dev
        else:
            high_case = base_case * 1.2

    # Ensure low <= base <= high
    low_case = min(low_case, base_case)
    high_case = max(high_case, base_case)

    return low_case, high_case


def _apply_skew(
    distribution_type: str,
    skew: float,
    low_case: float,
    high_case: float,
    base_case: float
) -> float:
    """
    Apply skew to determine the mode of the distribution.

    Args:
        distribution_type: Type of distribution
        skew: Skew parameter (-1 to 1)
        low_case: Lower bound
        high_case: Upper bound
        base_case: Base case value

    Returns:
        Mode value for the distribution
    """
    if distribution_type == "triangular" and skew != 0:
        # Skew the mode based on the skew parameter
        skew_range = high_case - low_case
        # Map skew from [-1, 1] to [0, 1] (position within the range)
        skew_position = 0.5 + (skew / 2)
        mode = low_case + skew_position * skew_range
        # Use the skewed mode instead of base_case
        return np.clip(mode, low_case, high_case)

    return base_case


def _create_triangular_distribution(
        low_case: float,
        high_case: float,
        mode: float) -> DistributionSpec:
    """Create triangular distribution specification."""
    return lambda size=None: stats.triang.rvs(
        c=(mode - low_case) / (high_case - low_case),
        loc=low_case,
        scale=high_case - low_case,
        size=size
    )


def _create_normal_distribution(
        base_case: float,
        low_case: float,
        high_case: float,
        std_dev: Optional[float]) -> DistributionSpec:
    """Create normal distribution specification."""
    if std_dev is None:
        # Estimate std_dev from the range (assuming ~95% confidence interval)
        std_dev = (high_case - low_case) / 4

    # Define a truncated normal distribution
    return lambda size=None: np.clip(
        stats.norm.rvs(loc=base_case, scale=std_dev, size=size),
        low_case, high_case
    )


def _create_uniform_distribution(
        low_case: float,
        high_case: float) -> DistributionSpec:
    """Create uniform distribution specification."""
    return lambda size=None: stats.uniform.rvs(
        loc=low_case,
        scale=high_case - low_case,
        size=size
    )


def create_driver_distribution(
    base_case: float,
    low_case: Optional[float] = None,
    high_case: Optional[float] = None,
    distribution_type: str = "triangular",
    std_dev: Optional[float] = None,
    skew: float = 0.0,
) -> DistributionSpec:
    """
    Create a distribution specification for a financial driver.

    Args:
        base_case: Base case value (mean or mode, depending on distribution)
        low_case: Low case value (if None, will be derived from std_dev or as 0.8 * base_case)
        high_case: High case value (if None, will be derived from std_dev or as 1.2 * base_case)
        distribution_type: Type of distribution ('triangular', 'normal', 'uniform', 'custom')
        std_dev: Standard deviation (used for normal distribution if provided)
        skew: Skewness parameter (between -1 and 1, with 0 being symmetric)

    Returns:
        DistributionSpec object representing the specified distribution
    """
    # If constant value, return it
    if distribution_type == "constant":
        return constant_value(base_case)

    # Derive bounds
    low_case, high_case = _derive_bounds(
        base_case, low_case, high_case, distribution_type, std_dev)

    # Apply skew to determine mode
    mode = _apply_skew(distribution_type, skew, low_case, high_case, base_case)

    # Create distribution based on type
    if distribution_type == "triangular":
        return _create_triangular_distribution(low_case, high_case, mode)
    elif distribution_type == "normal":
        return _create_normal_distribution(
            base_case, low_case, high_case, std_dev)
    elif distribution_type == "uniform":
        return _create_uniform_distribution(low_case, high_case)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")


def create_wacc_distribution(
    base_wacc: float,
    low_wacc: Optional[float] = None,
    high_wacc: Optional[float] = None,
    distribution_type: str = "triangular",
    skew: float = 0.0
) -> DistributionSpec:
    """
    Create a distribution specification for WACC.

    Args:
        base_wacc: Base case WACC value
        low_wacc: Low case WACC (if None, will be derived as base_wacc - 1%)
        high_wacc: High case WACC (if None, will be derived as base_wacc + 1.5%)
        distribution_type: Type of distribution ('triangular', 'normal', 'uniform')
        skew: Skewness parameter (between -1 and 1, with 0 being symmetric)

    Returns:
        DistributionSpec object representing the specified WACC distribution
    """
    # Default low and high cases for WACC if not provided
    if low_wacc is None:
        low_wacc = max(base_wacc - 0.01, 0.01)  # Ensure WACC is at least 1%

    if high_wacc is None:
        high_wacc = base_wacc + 0.015

    # Use the general function to create the distribution
    return create_driver_distribution(
        base_case=base_wacc,
        low_case=low_wacc,
        high_case=high_wacc,
        distribution_type=distribution_type,
        skew=skew
    )


def create_terminal_value_distribution(
    base_multiple: float,
    low_multiple: Optional[float] = None,
    high_multiple: Optional[float] = None,
    distribution_type: str = "triangular",
    skew: float = 0.0
) -> DistributionSpec:
    """
    Create a distribution specification for terminal value multiple.

    Args:
        base_multiple: Base case multiple (e.g., EV/EBITDA)
        low_multiple: Low case multiple (if None, will be derived as 0.8 * base_multiple)
        high_multiple: High case multiple (if None, will be derived as 1.2 * base_multiple)
        distribution_type: Type of distribution ('triangular', 'normal', 'uniform')
        skew: Skewness parameter (between -1 and 1, with 0 being symmetric)

    Returns:
        DistributionSpec object representing the specified terminal value distribution
    """
    # Default low and high cases if not provided
    if low_multiple is None:
        low_multiple = base_multiple * 0.8

    if high_multiple is None:
        high_multiple = base_multiple * 1.2

    # Use the general function to create the distribution
    return create_driver_distribution(
        base_case=base_multiple,
        low_case=low_multiple,
        high_case=high_multiple,
        distribution_type=distribution_type,
        skew=skew
    )


def create_triangular_distribution(
    min_value: float,
    most_likely_value: float,
    max_value: float
) -> DistributionSpec:
    """
    Create a triangular distribution for Monte Carlo simulation.

    Args:
        min_value: Minimum value of the distribution
        most_likely_value: Mode (peak) of the distribution
        max_value: Maximum value of the distribution

    Returns:
        DistributionSpec object representing a triangular distribution
    """
    # Ensure min <= most_likely <= max
    min_value = min(min_value, most_likely_value)
    max_value = max(max_value, most_likely_value)

    # Calculate the 'c' parameter for scipy's triangular distribution
    # which is the fraction of the way from min to max where the peak occurs
    if max_value > min_value:
        c = (most_likely_value - min_value) / (max_value - min_value)
    else:
        c = 0.5  # Default to center if min == max

    return lambda size=None: stats.triang.rvs(
        c=c,
        loc=min_value,
        scale=max_value - min_value,
        size=size
    )


def summarize_valuation_results(
    results: np.ndarray,
    confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[str, Any]:
    """
    Summarize Monte Carlo valuation results

    Args:
        results: Array of valuation results
        confidence_levels: List of confidence levels to report

    Returns:
        Dictionary with summary statistics
    """
    # Calculate basic statistics
    mean_value = np.mean(results)
    median_value = np.median(results)
    std_dev = np.std(results)
    min_value = np.min(results)
    max_value = np.max(results)

    # Calculate percentiles
    percentiles = {}
    for cl in confidence_levels:
        percentiles[f"p{int(cl*100)}"] = np.percentile(results, cl * 100)

    # Calculate upside/downside from median
    p25 = percentiles.get("p25", np.percentile(results, 25))
    p75 = percentiles.get("p75", np.percentile(results, 75))
    downside = (median_value - p25) / median_value
    upside = (p75 - median_value) / median_value

    return {
        "mean": mean_value,
        "median": median_value,
        "std_dev": std_dev,
        "min": min_value,
        "max": max_value,
        "percentiles": percentiles,
        "downside": downside,
        "upside": upside
    }
