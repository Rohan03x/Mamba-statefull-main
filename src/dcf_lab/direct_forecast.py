"""
Functions to prepare datasets for direct multi-horizon log returns forecasting
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def prepare_log_returns_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    horizon: int = 10,
    sequence_length: int = 60
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare log returns data for training multi-horizon models

    Args:
        df: DataFrame with features and prices
        feature_columns: List of feature columns to use
        horizon: Number of days to forecast
        sequence_length: Length of input sequence

    Returns:
        Tuple of (dataframe with log returns, features array, multi-horizon targets array)
    """
    # Calculate log returns
    df = df.copy()
    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))

    # Drop NaN values at the beginning
    df = df.dropna()

    # Create sequences for direct multi-horizon forecasting
    sequences = []
    targets = []

    # Leave out the last 'horizon' periods so we can create multi-horizon
    # targets
    for i in range(sequence_length, len(df) - horizon):
        # Input sequence
        seq = df[feature_columns].values[i - sequence_length:i]

        # Multi-horizon targets (next 'horizon' log returns)
        target = df['log_return'].values[i:i+horizon]

        sequences.append(seq)
        targets.append(target)

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    return df, X, y


def prepare_torch_datasets(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare PyTorch datasets with train/validation split

    Args:
        X: Input features array
        y: Target array
        val_split: Validation split ratio

    Returns:
        Tuple of (X_train, y_train, X_val, y_val) tensors
    """
    # Calculate split point
    split_idx = int(len(X) * (1 - val_split))

    # Split data
    x_train, x_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor


def evaluate_model_metrics(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluate model performance metrics

    Args:
        model: Trained model
        X_val: Validation input tensor
        y_val: Validation target tensor

    Returns:
        Dictionary with performance metrics
    """
    model.eval()
    metrics = {}

    with torch.no_grad():
        # Get predictions
        predictions = model(x_val)

        # Check model output format
        if hasattr(model, 'quantiles') and len(model.quantiles) > 0:
            # Get median predictions (quantile=0.5 or middle index)
            median_idx = [
                i for i,
                q in enumerate(
                    model.quantiles) if abs(
                    q -
                    0.5) < 1e-6]
            if median_idx:
                y_pred = predictions[:, median_idx[0], :]
            else:
                # Use middle quantile as approximation
                y_pred = predictions[:, len(model.quantiles) // 2, :]
        else:
            # Single output model
            y_pred = predictions

        # MSE and RMSE
        mse = torch.mean((y_pred - y_val) ** 2).item()
        rmse = np.sqrt(mse)
        metrics['mse'] = mse
        metrics['rmse'] = rmse

        # MAE
        mae = torch.mean(torch.abs(y_pred - y_val)).item()
        metrics['mae'] = mae

        # Direction accuracy (hit rate)
        y_sign = torch.sign(y_val)
        pred_sign = torch.sign(y_pred)
        correct = (y_sign == pred_sign).float().mean().item()
        metrics['direction_accuracy'] = correct

        # Calculate metrics per horizon
        for h in range(y_val.shape[1]):
            metrics[f'h{h +
                        1}_mse'] = torch.mean((y_pred[:, h] -
                                               y_val[:, h]) ** 2).item()
            metrics[f'h{h +
                        1}_mae'] = torch.mean(torch.abs(y_pred[:, h] -
                                                        y_val[:, h])).item()
            metrics[f'h{h+1}_dir_acc'] = (torch.sign(y_pred[:, h])
                                          == torch.sign(y_val[:, h])).float().mean().item()

    return metrics
