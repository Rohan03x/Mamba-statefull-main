"""
Helper functions for AI price forecasting to reduce cognitive complexity
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


def create_model_config(use_quantiles: bool) -> Dict:
    """Create model configuration dictionary"""
    return {
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 3,
        'dim_feedforward': 128,
        'dropout': 0.3,
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9] if use_quantiles else [0.5]
    }


def train_multihorizon_model(
    df_train: pd.DataFrame,
    feature_columns: List[str],
    horizon: int,
    model_config: Dict
) -> Tuple[torch.nn.Module, Dict, Dict]:
    """Train multi-horizon forecasting model"""
    from .multihorizon_trainer import train_multihorizon_forecast_model

    return train_multihorizon_forecast_model(
        df=df_train,
        feature_columns=feature_columns,
        horizon=horizon,
        sequence_length=60,
        epochs=50,
        batch_size=32,
        model_config=model_config
    )


def prepare_legacy_dataset(
    df_train: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare dataset for legacy single-step forecasting"""
    from .forecast_helpers import prepare_transformer_dataset

    sequence_length = 60
    X, y = prepare_transformer_dataset(
        df=df_train,
        feature_columns=feature_columns,
        target_column='adj_close',
        sequence_length=sequence_length,
        forecast_horizon=1
    )

    if len(X) < 10:
        raise ValueError(f"Not enough data for training: {len(X)}")

    # Check for NaNs in tensors
    if torch.isnan(X).any():
        print("Warning: NaN values found in input tensor X")
        X = torch.nan_to_num(X, nan=0.0)

    if torch.isnan(y).any():
        print("Warning: NaN values found in target tensor y")
        y = torch.nan_to_num(y, nan=0.0)

    return X, y


def create_legacy_model(model_type: str, input_dim: int) -> torch.nn.Module:
    """Create legacy model for single-step forecasting"""
    # Simple placeholder models to avoid circular imports
    if model_type == 'transformer':
        # Create a simple transformer-like model
        class SimpleTransformer(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim, 1)

            def forward(self, x):
                return self.linear(x)

        return SimpleTransformer(input_dim)
    else:  # lstm
        # Create a simple LSTM-like model
        class SimpleLSTM(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_dim, 64, batch_first=True)
                self.linear = torch.nn.Linear(64, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.linear(lstm_out[:, -1, :])

        return SimpleLSTM(input_dim)


def train_legacy_model(
    X: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module
) -> Tuple[torch.nn.Module, Dict]:
    """Train legacy model with validation"""
    # For legacy compatibility, create a simple training approach
    if hasattr(model, 'fit'):
        # For sklearn-style models
        model.fit(X, y)
        history = {'loss': [0.1]}  # Placeholder history
        return model, history
    else:
        # For PyTorch models - use direct training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=1e-5)

        history = {'loss': []}

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(
                outputs, y.unsqueeze(1) if len(
                    y.shape) == 1 else y)

            loss.backward()
            optimizer.step()

            history['loss'].append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        return model, history


def validate_forecasts(
    forecast_series: pd.Series,
    df: pd.DataFrame,
    quantile_dict: Optional[Dict] = None
) -> pd.Series:
    """Validate and clean forecast results"""
    # Check if forecasts contain NaN values
    if forecast_series.isna().any():
        print("Warning: NaN values in forecast results")
        forecast_series = forecast_series.ffill().fillna(
            df['adj_close'].iloc[-1])
        print("NaN values in forecasts replaced with last known price")

    # Check for potential issues
    if quantile_dict:
        if quantile_dict.get("is_monotone", False):
            print(
                "Warning: Monotone forecast pattern detected without significant news events.")
            print("This could indicate model issues or unrealistic predictions.")

        if quantile_dict.get("narrow_forecast", False):
            print("Warning: Forecast uncertainty is very narrow.")
            print(
                "Consider blending with options-implied volatility for more realistic uncertainty.")

    return forecast_series


def create_model_info(
    model_type: str,
    feature_columns: List[str],
    metrics: Optional[Dict] = None,
    quantile_dict: Optional[Dict] = None,
    horizon: int = 10,
    training_samples: int = 0,
    final_loss: float = 0.0
) -> Dict:
    """Create model information dictionary"""
    if metrics and quantile_dict:
        # Multi-horizon model info
        return {
            "model_type": "multihorizon_transformer",
            "feature_columns": feature_columns,
            "direction_accuracy": metrics.get("direction_accuracy", 0.0),
            "horizon": horizon,
            "quantile_predictions": quantile_dict,
            "training_metrics": metrics
        }
    else:
        # Legacy model info
        return {
            "model_type": model_type,
            "features": feature_columns,
            "training_samples": training_samples,
            "final_loss": final_loss
        }
