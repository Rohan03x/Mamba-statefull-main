"""
Visualization helpers for calibration plots
"""
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_value_distribution(
    ax: plt.Axes,
    market_values: np.ndarray,
    forecast_values: np.ndarray,
    metrics: Dict[str, Any],
    value_per_share_label: str
) -> None:
    """
    Plot value distributions comparison

    Args:
        ax: Matplotlib axes to plot on
        market_values: Array of market values
        forecast_values: Array of forecast values
        metrics: Metrics dictionary for the parameter
        value_per_share_label: Label for x-axis
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

    # Plot the distributions
    ax.plot(
        x,
        kde_market(x),
        'b-',
        label=f'Market (${
            np.median(market_values):.2f})')
    ax.plot(
        x,
        kde_forecast(x),
        'r-',
        label=f'Forecast (${
            np.median(forecast_values):.2f})')

    # Add metrics to plot
    metrics_text = [
        f"Upside: {metrics['forecast_upside']:.1%}",
        f"Overlap: {metrics['overlap_coefficient']:.2f}",
        f"KS-stat: {metrics['ks_statistic']:.3f}"
    ]

    add_metrics_text(ax, metrics_text)

    ax.set_title('Value Distribution Comparison')
    ax.set_xlabel(value_per_share_label)
    ax.set_ylabel('Probability Density')

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_parameter_distribution(
    ax: plt.Axes,
    market_spec: Dict,
    forecast_spec: Dict,
    param_name: str,
    metrics: Dict[str, Any]
) -> None:
    """
    Plot parameter distribution comparison for normal distributions

    Args:
        ax: Matplotlib axes to plot on
        market_spec: Market distribution specification
        forecast_spec: Forecast distribution specification
        param_name: Parameter name
        metrics: Metrics dictionary for the parameter
    """
    # Extract distribution parameters
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

    # Plot the distributions
    ax.plot(x, market_pdf, 'b-', label=f'Market ({market_mean:.2%})')
    ax.plot(x, forecast_pdf, 'r-', label=f'Forecast ({forecast_mean:.2%})')

    # Add metrics to plot
    metrics_text = [
        f"Diff: {metrics['percent_diff_mean']:.1f}%",
        f"Z-diff: {metrics['z_score_diff']:.2f}",
        f"Bhatt: {metrics['bhattacharyya_distance']:.3f}"
    ]

    add_metrics_text(ax, metrics_text)

    # Make title more readable
    title = param_name.replace('_', ' ').title()
    ax.set_title(title)

    # Handle x-axis labels based on parameter
    if param_name in [
        'wacc',
        'revenue_growth',
        'ebit_margin',
            'terminal_growth']:
        ax.set_xlabel('Rate (%)')
        # Format x ticks as percentages
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: f'{
                    x:.1%}'))

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()


def add_metrics_text(
        ax: plt.Axes,
        metrics_text: List[str],
        y_start: float = 0.9,
        step: float = 0.05) -> None:
    """
    Add metrics text to the axes

    Args:
        ax: Matplotlib axes to add text to
        metrics_text: List of text strings to add
        y_start: Starting y position (in axes coordinates)
        step: Step size for each line
    """
    y_pos = y_start
    for text in metrics_text:
        ax.text(0.05, y_pos, text, transform=ax.transAxes)
        y_pos -= step


def create_comparison_figure(
    parameters: List[str],
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create a figure for comparison plots

    Args:
        parameters: List of parameters to plot
        figsize: Figure size

    Returns:
        Figure and axes
    """
    n_plots = len(parameters)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Make axes always iterable
    if n_plots == 1:
        axes = [axes]

    return fig, axes
