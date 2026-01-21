"""
Data preparation utilities for calibration visualizations
"""
from typing import Any, Dict

import numpy as np
from scipy import stats


def prepare_value_distribution_data(
    market_values: np.ndarray,
    forecast_values: np.ndarray
) -> Dict[str, Any]:
    """
    Prepare data for value distribution plots

    Args:
        market_values: Array of market values
        forecast_values: Array of forecast values

    Returns:
        Dictionary with prepared data
    """
    # Create KDE for both distributions
    kde_market = stats.gaussian_kde(market_values)
    kde_forecast = stats.gaussian_kde(forecast_values)

    # Create x range for plotting
    x = np.linspace(
        min(np.min(market_values), np.min(forecast_values)),
        max(np.max(market_values), np.max(forecast_values)),
        1000
    )

    return {
        'x': x,
        'market_density': kde_market(x),
        'forecast_density': kde_forecast(x),
        'market_median': np.median(market_values),
        'forecast_median': np.median(forecast_values)
    }


def prepare_parameter_distribution_data(
    market_spec: Dict,
    forecast_spec: Dict
) -> Dict[str, Any]:
    """
    Prepare data for parameter distribution plots

    Args:
        market_spec: Market distribution specification
        forecast_spec: Forecast distribution specification

    Returns:
        Dictionary with prepared data or None if not supported
    """
    # Only handle normal distributions
    if market_spec.dist_type != 'normal' or forecast_spec.dist_type != 'normal':
        return None

    market_mean = market_spec.params.get('mean')
    market_std = market_spec.params.get('std')
    forecast_mean = forecast_spec.params.get('mean')
    forecast_std = forecast_spec.params.get('std')

    # Create x range for plotting
    x = np.linspace(
        min(market_mean - 3*market_std, forecast_mean - 3*forecast_std),
        max(market_mean + 3*market_std, forecast_mean + 3*forecast_std),
        1000
    )

    # Calculate PDFs
    market_pdf = stats.norm.pdf(x, market_mean, market_std)
    forecast_pdf = stats.norm.pdf(x, forecast_mean, forecast_std)

    return {
        'x': x,
        'market_pd': market_pdf,
        'forecast_pd': forecast_pdf,
        'market_mean': market_mean,
        'forecast_mean': forecast_mean
    }
