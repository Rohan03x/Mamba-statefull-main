"""
Configuration management using Pydantic models.
"""
import logging
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    transaction_cost_bps: float = Field(
        3.0,
        description="Transaction cost in basis points"
    )
    position_limit: float = Field(
        1.0,
        description="Maximum absolute position size"
    )
    stop_loss_pct: Optional[float] = Field(
        None,
        description="Stop loss percentage (if None, no stop loss)"
    )
    take_profit_pct: Optional[float] = Field(
        None,
        description="Take profit percentage (if None, no take profit)"
    )
    kelly_fraction: Optional[float] = Field(
        0.5,
        description="Kelly fraction for position sizing (if None, use fixed size)"
    )
    use_classification: bool = Field(
        False,
        description="Whether to use classification probabilities for signals"
    )
    threshold: float = Field(
        0.55,
        description="Probability threshold for classification signals"
    )


class ModelConfig(BaseModel):
    """Configuration for model training."""

    # Common parameters
    window: int = Field(
        ...,
        description="Number of lookback periods"
    )
    horizon: int = Field(
        ...,
        description="Number of periods to forecast"
    )
    target: str = Field(
        "log_ret_1d",
        description="Target variable to predict"
    )
    cv_folds: int = Field(
        5,
        description="Number of cross-validation folds"
    )

    # LSTM parameters
    lstm_hidden_size: int = Field(
        128,
        description="Hidden size for LSTM"
    )
    lstm_layers: int = Field(
        2,
        description="Number of LSTM layers"
    )
    lstm_dropout: float = Field(
        0.2,
        description="Dropout rate for LSTM"
    )
    batch_size: int = Field(
        32,
        description="Batch size for training"
    )
    max_epochs: int = Field(
        100,
        description="Maximum number of epochs"
    )
    learning_rate: float = Field(
        1e-3,
        description="Learning rate"
    )

    # ARIMA parameters
    seasonal: bool = Field(
        False,
        description="Whether to use seasonal ARIMA"
    )
    seasonal_period: Optional[int] = Field(
        None,
        description="Seasonal period for ARIMA"
    )

    # XGBoost parameters
    max_depth: int = Field(
        6,
        description="Maximum tree depth"
    )
    n_estimators: int = Field(
        100,
        description="Number of trees"
    )
    learning_rate_xgb: float = Field(
        0.1,
        description="Learning rate for XGBoost"
    )

    # Feature parameters
    use_ta_features: bool = Field(
        True,
        description="Whether to use technical analysis features"
    )
    use_price_features: bool = Field(
        True,
        description="Whether to use price-derived features"
    )
    use_volume_features: bool = Field(
        True,
        description="Whether to use volume features"
    )


class TradingConfig(BaseModel):
    """Main configuration for trading system."""

    # Data parameters
    ticker: str = Field(
        ...,
        description="Stock ticker symbol"
    )
    start_date: str = Field(
        ...,
        description="Start date for data (YYYY-MM-DD)"
    )
    end_date: str = Field(
        ...,
        description="End date for data (YYYY-MM-DD)"
    )
    interval: str = Field(
        "1d",
        description="Data interval (1m|5m|1h|1d|1wk|1mo)"
    )

    # Model configuration
    model: ModelConfig = Field(
        ...,
        description="Model configuration"
    )

    # Backtest configuration
    backtest: BacktestConfig = Field(
        default_factory=BacktestConfig,
        description="Backtest configuration"
    )

    # Feature selection
    features: List[str] = Field(
        [],
        description="List of features to use (empty for all)"
    )

    # Additional parameters
    cache_data: bool = Field(
        True,
        description="Whether to cache downloaded data"
    )
    random_seed: int = Field(
        42,
        description="Random seed for reproducibility"
    )

    @validator("interval")
    def validate_interval(cls, v):
        """Validate data interval."""
        valid_intervals = ["1m", "5m", "1h", "1d", "1wk", "1mo"]
        if v not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")
        return v

    @validator("features")
    def validate_features(cls, v):
        """Validate feature list."""
        if not v:
            logger.info(
                "No features specified, will use all available features")
        return v


def load_config(config_path: Path) -> TradingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return TradingConfig(**config_dict)


# Default configurations
SHORT_TERM_CONFIG = {
    "ticker": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2025-09-16",
    "interval": "1d",
    "model": {
        "window": 60,
        "horizon": 5,
        "cv_folds": 5,
        "lstm_hidden_size": 128,
        "lstm_layers": 2,
        "lstm_dropout": 0.2,
        "batch_size": 32,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "seasonal": False,
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate_xgb": 0.1,
        "use_ta_features": True,
        "use_price_features": True,
        "use_volume_features": True
    },
    "backtest": {
        "transaction_cost_bps": 3.0,
        "position_limit": 1.0,
        "kelly_fraction": 0.5,
        "threshold": 0.55
    }
}

LONG_TERM_CONFIG = {
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2025-09-16",
    "interval": "1mo",
    "model": {
        "window": 24,
        "horizon": 12,
        "cv_folds": 5,
        "lstm_hidden_size": 64,
        "lstm_layers": 2,
        "lstm_dropout": 0.2,
        "batch_size": 16,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "seasonal": True,
        "seasonal_period": 12,
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate_xgb": 0.1,
        "use_ta_features": True,
        "use_price_features": True,
        "use_volume_features": True
    },
    "backtest": {
        "transaction_cost_bps": 3.0,
        "position_limit": 1.0,
        "kelly_fraction": 0.3,
        "threshold": 0.6
    }
}


def save_default_configs(config_dir: Path) -> None:
    """
    Save default configuration files.

    Args:
        config_dir: Directory to save configurations
    """
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save short-term config
    with open(config_dir / "short_term.yaml", "w") as f:
        yaml.dump(SHORT_TERM_CONFIG, f, sort_keys=False)

    # Save long-term config
    with open(config_dir / "long_term.yaml", "w") as f:
        yaml.dump(LONG_TERM_CONFIG, f, sort_keys=False)
