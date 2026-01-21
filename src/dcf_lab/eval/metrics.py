"""
Evaluation metrics for forecasting and trading performance.
GPU-ACCELERATED VERSION with CuPy/RAPIDS for walk-forward optimization speedup
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# GPU Acceleration Imports and Setup
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
    print("🚀 GPU METRICS: CuPy/cuDF available - enabling GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU METRICS: CuPy/cuDF not available - using CPU fallback")
    # Create dummy cp namespace for compatibility
    class _DummyCuPy:
        def __getattr__(self, name):
            return getattr(np, name)
    cp = _DummyCuPy()

logger = logging.getLogger(__name__)


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_lower: Optional[np.ndarray] = None,
    y_pred_upper: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_pred_lower: Lower confidence interval (optional)
        y_pred_upper: Upper confidence interval (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["mae"] = mean_absolute_error(y_true, y_pred)

    # MAPE
    mask = y_true != 0
    metrics["mape"] = np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Directional accuracy
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    metrics["dir_acc"] = np.mean(direction_true == direction_pred) * 100

    # Hit rate at different thresholds
    for threshold in [0.0, 0.001, 0.002, 0.005]:
        pred_up = y_pred > threshold
        true_up = y_true > threshold
        metrics[f"hit_rate_{threshold:.3f}"] = np.mean(
            pred_up == true_up) * 100

    # Prediction interval coverage (if intervals provided)
    if y_pred_lower is not None and y_pred_upper is not None:
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        metrics["picp"] = np.mean(in_interval) * 100

        # Mean prediction interval width
        metrics["mpiw"] = np.mean(y_pred_upper - y_pred_lower)

    return metrics


def calculate_crps(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray
) -> float:
    """
    Calculate Continuous Ranked Probability Score for probabilistic forecasts.

    Args:
        y_true: True values
        y_pred_samples: Predicted samples (shape: [n_samples, n_forecasts])

    Returns:
        CRPS value
    """
    n_samples = y_pred_samples.shape[0]
    n_forecasts = y_pred_samples.shape[1]
    crps = np.zeros(n_forecasts)

    for i in range(n_forecasts):
        y_samples = y_pred_samples[:, i]
        y_t = y_true[i]

        # Sort samples
        y_samples.sort()

        # Calculate CRPS
        crps[i] = np.sum(np.abs(y_samples - y_t)) / n_samples

    return np.mean(crps)


def calculate_trading_metrics(
    returns: pd.Series,
    positions: pd.Series,
    risk_free_rate: float = 0.0,
    transaction_costs_bps: float = 3.0,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate trading strategy performance metrics.
    🚀 GPU-ACCELERATED VERSION - Uses CuPy for vectorized calculations
    
    Args:
        returns: Asset returns
        positions: Strategy positions
        risk_free_rate: Risk-free rate (annualized)
        transaction_costs_bps: Transaction costs in basis points
        annualization_factor: Number of periods in a year

    Returns:
        Dictionary of metrics
    """
    
    # Use GPU acceleration if available and data is large enough
    use_gpu = GPU_AVAILABLE and len(returns) > 1000
    
    if use_gpu:
        # Convert to GPU arrays for vectorized computation
        returns_gpu = cp.asarray(returns.values)
        positions_gpu = cp.asarray(positions.values)
        
        # GPU-vectorized position changes and costs
        position_changes_gpu = cp.diff(positions_gpu, prepend=0)
        transaction_costs_gpu = cp.abs(position_changes_gpu) * (transaction_costs_bps / 10000)
        
        # GPU-vectorized strategy returns
        positions_shifted_gpu = cp.roll(positions_gpu, 1)
        positions_shifted_gpu[0] = 0  # First position has no lag
        strategy_returns_gpu = positions_shifted_gpu * returns_gpu - transaction_costs_gpu
        
        # GPU-vectorized cumulative returns (using scan operation)
        cum_returns_gpu = cp.cumprod(1 + strategy_returns_gpu)
        
        # Convert back to numpy for final calculations (minimal cost)
        strategy_returns = pd.Series(cp.asnumpy(strategy_returns_gpu), index=returns.index)
        cum_returns = pd.Series(cp.asnumpy(cum_returns_gpu), index=returns.index)
        position_changes = pd.Series(cp.asnumpy(position_changes_gpu), index=positions.index)
        
        print(f"⚡ GPU METRICS: Accelerated calculation for {len(returns)} data points")
        
    else:
        # CPU fallback (original code)
        position_changes = positions.diff().fillna(0)
        transaction_costs = np.abs(position_changes) * (transaction_costs_bps / 10000)
        strategy_returns = positions.shift(1) * returns - transaction_costs
        cum_returns = (1 + strategy_returns).cumprod()

    metrics = {}

    # Vectorized calculations (same for both GPU and CPU paths)
    metrics["total_return"] = float(cum_returns.iloc[-1] - 1)
    
    n_years = len(returns) / annualization_factor
    metrics["cagr"] = float((1 + metrics["total_return"]) ** (1 / n_years) - 1)
    
    # GPU-accelerated or standard volatility
    if use_gpu:
        volatility_gpu = cp.std(cp.asarray(strategy_returns.values)) * cp.sqrt(annualization_factor)
        metrics["volatility"] = float(cp.asnumpy(volatility_gpu))
    else:
        metrics["volatility"] = float(strategy_returns.std() * np.sqrt(annualization_factor))

    # GPU-accelerated Sharpe ratio
    excess_returns = strategy_returns - risk_free_rate / annualization_factor
    if use_gpu:
        excess_gpu = cp.asarray(excess_returns.values)
        returns_gpu = cp.asarray(strategy_returns.values)
        sharpe_gpu = cp.sqrt(annualization_factor) * cp.mean(excess_gpu) / cp.std(returns_gpu)
        metrics["sharpe"] = float(cp.asnumpy(sharpe_gpu))
    else:
        metrics["sharpe"] = float(np.sqrt(annualization_factor) * excess_returns.mean() / strategy_returns.std())

    # GPU-accelerated Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0:
        if use_gpu and len(downside_returns) > 100:
            downside_gpu = cp.asarray(downside_returns.values)
            excess_gpu = cp.asarray(excess_returns.values)
            downside_std_gpu = cp.std(downside_gpu) * cp.sqrt(annualization_factor)
            sortino_gpu = cp.sqrt(annualization_factor) * cp.mean(excess_gpu) / downside_std_gpu
            metrics["sortino"] = float(cp.asnumpy(sortino_gpu))
        else:
            downside_std = downside_returns.std() * np.sqrt(annualization_factor)
            metrics["sortino"] = float(np.sqrt(annualization_factor) * excess_returns.mean() / downside_std)
    else:
        metrics["sortino"] = float(np.inf)

    # GPU-accelerated maximum drawdown using scan operations
    if use_gpu:
        cum_returns_gpu = cp.asarray(cum_returns.values)
        # Rolling maximum using cumulative max (scan operation)
        rolling_max_gpu = cp.maximum.accumulate(cum_returns_gpu)
        drawdowns_gpu = (cum_returns_gpu - rolling_max_gpu) / rolling_max_gpu
        metrics["max_drawdown"] = float(cp.asnumpy(cp.min(drawdowns_gpu)))
    else:
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        metrics["max_drawdown"] = float(drawdowns.min())

    # Calmar ratio
    metrics["calmar"] = float(metrics["cagr"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] != 0 else 0)

    # Trading statistics (vectorized where possible)
    trades = position_changes[position_changes != 0]
    metrics["n_trades"] = len(trades)
    
    if use_gpu:
        position_changes_gpu = cp.asarray(position_changes.values)
        positions_gpu = cp.asarray(positions.values)
        metrics["turnover"] = float(cp.asnumpy(cp.sum(cp.abs(position_changes_gpu))))
        metrics["avg_position"] = float(cp.asnumpy(cp.mean(cp.abs(positions_gpu))))
    else:
        metrics["turnover"] = float(abs(position_changes).sum())
        metrics["avg_position"] = float(abs(positions).mean())

    # Win rate and trade statistics
    trade_returns = strategy_returns[position_changes != 0]
    if len(trade_returns) > 0:
        if use_gpu and len(trade_returns) > 100:
            trade_returns_gpu = cp.asarray(trade_returns.values)
            metrics["win_rate"] = float(cp.asnumpy(cp.mean(trade_returns_gpu > 0) * 100))
            metrics["avg_trade_return"] = float(cp.asnumpy(cp.mean(trade_returns_gpu)))
            
            wins_gpu = trade_returns_gpu[trade_returns_gpu > 0]
            losses_gpu = trade_returns_gpu[trade_returns_gpu < 0]
            
            metrics["avg_win"] = float(cp.asnumpy(cp.mean(wins_gpu))) if len(wins_gpu) > 0 else 0.0
            metrics["avg_loss"] = float(cp.asnumpy(cp.mean(losses_gpu))) if len(losses_gpu) > 0 else 0.0
        else:
            metrics["win_rate"] = float((trade_returns > 0).mean() * 100)
            metrics["avg_trade_return"] = float(trade_returns.mean())
            metrics["avg_win"] = float(trade_returns[trade_returns > 0].mean()) if len(trade_returns[trade_returns > 0]) > 0 else 0.0
            metrics["avg_loss"] = float(trade_returns[trade_returns < 0].mean()) if len(trade_returns[trade_returns < 0]) > 0 else 0.0
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_trade_return"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0

    # Market exposure
    if use_gpu:
        positions_gpu = cp.asarray(positions.values)
        metrics["exposure"] = float(cp.asnumpy(cp.mean(positions_gpu != 0) * 100))
    else:
        metrics["exposure"] = float((positions != 0).mean() * 100)

    return metrics
