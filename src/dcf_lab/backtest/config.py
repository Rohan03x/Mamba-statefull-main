"""
Walk-Forward Backtest Framework

This module implements sophisticated walk-forward backtesting with:
1. Multiple cadences: rebalancing, re-training, hyperparameter refresh
2. Rolling origin evaluation (true "live" predictions)
3. Comprehensive artifact storage for auditability
4. Backtest overfitting detection (PBO and other safeguards)

Following best practices from:
- David H. Bailey (rolling origin evaluation)
- López de Prado (financial ML methodologies)
- Academic literature on backtest overfitting

Key Features:
- Weekly rebalancing with monthly re-training
- Quarterly hyperparameter refresh with Optuna
- Expanding/rolling data windows (2-5 years)
- Complete artifact trail for audit purposes
- PBO detection to prevent false discoveries
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataWindowType(Enum):
    """Data window policies for training"""
    EXPANDING = "expanding"  # Use all available historical data
    ROLLING_2Y = "rolling_2y"  # Rolling 2-year window
    ROLLING_3Y = "rolling_3y"  # Rolling 3-year window
    ROLLING_5Y = "rolling_5y"  # Rolling 5-year window


class CadenceType(Enum):
    """Different cadence types for backtest operations"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


@dataclass
class BacktestConfig:
    """
    Comprehensive configuration for walk-forward backtesting
    
    This class manages all aspects of backtest timing and data policies:
    - Rebalancing frequency (portfolio updates)
    - Re-training frequency (model updates)
    - Hyperparameter refresh frequency
    - Data window policies
    - Artifact storage settings
    """
    
    # Core timing configuration
    rebalance_cadence: CadenceType = CadenceType.WEEKLY
    retrain_cadence: CadenceType = CadenceType.MONTHLY
    hyperparam_refresh_cadence: CadenceType = CadenceType.QUARTERLY
    
    # Data window configuration
    data_window_type: DataWindowType = DataWindowType.EXPANDING
    min_training_days: int = 252  # Minimum 1 year of data
    max_training_days: Optional[int] = None  # No limit for expanding
    
    # Validation configuration
    validation_split: float = 0.2  # 20% of training data for validation
    validation_days: Optional[int] = None  # Override validation_split if set
    
    # Hyperparameter optimization
    use_optuna: bool = True
    optuna_trials: int = 100
    optuna_timeout: Optional[int] = 3600  # 1 hour timeout
    optuna_pruning: bool = True
    
    # Artifact storage
    store_artifacts: bool = True
    artifact_base_path: str = "backtest_artifacts"
    compress_artifacts: bool = True
    store_predictions: bool = True
    store_features: bool = True
    store_models: bool = False  # Models can be large
    
    # Performance and safety
    max_concurrent_jobs: int = 4
    random_seed: int = 42
    enable_pbo_detection: bool = True
    pbo_trials: int = 1000
    continue_on_error: bool = False  # Whether to continue backtest on step failures
    
    # Advanced settings
    allow_partial_retraining: bool = True  # Allow training on new data only
    force_retrain_threshold: float = 0.1  # Retrain if performance drops 10%
    market_data_lag_days: int = 1  # Account for data availability lag
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_cadences()
        self._set_window_limits()
        self._create_artifact_path()
    
    def _validate_cadences(self):
        """Ensure cadence hierarchy makes sense"""
        cadence_order = {
            CadenceType.DAILY: 1,
            CadenceType.WEEKLY: 7,
            CadenceType.MONTHLY: 30,
            CadenceType.QUARTERLY: 90,
            CadenceType.ANNUALLY: 365
        }
        
        rebal_days = cadence_order[self.rebalance_cadence]
        retrain_days = cadence_order[self.retrain_cadence]
        hyperparam_days = cadence_order[self.hyperparam_refresh_cadence]
        
        if retrain_days < rebal_days:
            logger.warning(
                f"Re-training cadence ({self.retrain_cadence}) is more frequent "
                f"than rebalancing ({self.rebalance_cadence}). This may be inefficient."
            )
        
        if hyperparam_days < retrain_days:
            logger.warning(
                f"Hyperparameter refresh ({self.hyperparam_refresh_cadence}) is more frequent "
                f"than re-training ({self.retrain_cadence}). This may cause overfitting."
            )
    
    def _set_window_limits(self):
        """Set data window limits based on window type"""
        if self.data_window_type == DataWindowType.ROLLING_2Y:
            self.max_training_days = 2 * 252  # 2 years
        elif self.data_window_type == DataWindowType.ROLLING_3Y:
            self.max_training_days = 3 * 252  # 3 years
        elif self.data_window_type == DataWindowType.ROLLING_5Y:
            self.max_training_days = 5 * 252  # 5 years
        # EXPANDING keeps max_training_days as None
    
    def _create_artifact_path(self):
        """Create artifact storage directory"""
        if self.store_artifacts:
            artifact_path = Path(self.artifact_base_path)
            artifact_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created artifact storage at: {artifact_path.absolute()}")
    
    def get_cadence_days(self, cadence: CadenceType) -> int:
        """Convert cadence to approximate days"""
        cadence_mapping = {
            CadenceType.DAILY: 1,
            CadenceType.WEEKLY: 7,
            CadenceType.MONTHLY: 30,
            CadenceType.QUARTERLY: 90,
            CadenceType.ANNUALLY: 365
        }
        return cadence_mapping[cadence]
    
    def validate_timing(self) -> bool:
        """
        Validate that timing configuration makes sense
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self._validate_cadences()
            return True
        except ValueError:
            return False
    
    def should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """Check if portfolio should be rebalanced"""
        days_diff = (current_date - last_rebalance).days
        return days_diff >= self.get_cadence_days(self.rebalance_cadence)
    
    def should_retrain(self, current_date: datetime, last_retrain: datetime) -> bool:
        """Check if models should be retrained"""
        days_diff = (current_date - last_retrain).days
        return days_diff >= self.get_cadence_days(self.retrain_cadence)
    
    def should_refresh_hyperparams(self, current_date: datetime, last_refresh: datetime) -> bool:
        """Check if hyperparameters should be refreshed"""
        days_diff = (current_date - last_refresh).days
        return days_diff >= self.get_cadence_days(self.hyperparam_refresh_cadence)
    
    def get_training_window(self, current_date: datetime, 
                          available_data_start: datetime) -> Tuple[datetime, datetime]:
        """
        Get training data window based on configuration
        
        Returns:
            (start_date, end_date) for training data
        """
        # Account for market data lag
        effective_current = current_date - timedelta(days=self.market_data_lag_days)
        
        if self.data_window_type == DataWindowType.EXPANDING:
            # Use all available data
            start_date = available_data_start
        else:
            # Use rolling window
            window_days = self.max_training_days
            start_date = max(
                available_data_start,
                effective_current - timedelta(days=window_days)
            )
        
        # Ensure minimum training period
        min_start = effective_current - timedelta(days=self.min_training_days)
        start_date = min(start_date, min_start)
        
        return start_date, effective_current
    
    def get_validation_window(self, train_start: datetime, 
                            train_end: datetime) -> Tuple[datetime, datetime]:
        """
        Get validation data window from training period
        
        Uses time-ordered split to prevent lookahead bias
        """
        total_days = (train_end - train_start).days
        
        if self.validation_days is not None:
            val_days = min(self.validation_days, int(total_days * 0.5))
        else:
            val_days = int(total_days * self.validation_split)
        
        # Validation is the last portion of training data
        val_start = train_end - timedelta(days=val_days)
        val_end = train_end
        
        # Adjust training end to not overlap with validation
        adjusted_train_end = val_start - timedelta(days=1)
        
        return val_start, val_end, adjusted_train_end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            'rebalance_cadence': self.rebalance_cadence.value,
            'retrain_cadence': self.retrain_cadence.value,
            'hyperparam_refresh_cadence': self.hyperparam_refresh_cadence.value,
            'data_window_type': self.data_window_type.value,
            'min_training_days': self.min_training_days,
            'max_training_days': self.max_training_days,
            'validation_split': self.validation_split,
            'validation_days': self.validation_days,
            'use_optuna': self.use_optuna,
            'optuna_trials': self.optuna_trials,
            'optuna_timeout': self.optuna_timeout,
            'optuna_pruning': self.optuna_pruning,
            'store_artifacts': self.store_artifacts,
            'artifact_base_path': self.artifact_base_path,
            'compress_artifacts': self.compress_artifacts,
            'store_predictions': self.store_predictions,
            'store_features': self.store_features,
            'store_models': self.store_models,
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'random_seed': self.random_seed,
            'enable_pbo_detection': self.enable_pbo_detection,
            'pbo_trials': self.pbo_trials,
            'allow_partial_retraining': self.allow_partial_retraining,
            'force_retrain_threshold': self.force_retrain_threshold,
            'market_data_lag_days': self.market_data_lag_days
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Create configuration from dictionary"""
        # Convert string enums back to enum objects
        config_dict['rebalance_cadence'] = CadenceType(config_dict['rebalance_cadence'])
        config_dict['retrain_cadence'] = CadenceType(config_dict['retrain_cadence'])
        config_dict['hyperparam_refresh_cadence'] = CadenceType(config_dict['hyperparam_refresh_cadence'])
        config_dict['data_window_type'] = DataWindowType(config_dict['data_window_type'])
        
        return cls(**config_dict)
    
    def get_config_hash(self) -> str:
        """Get hash of configuration for artifact tracking"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class BacktestState:
    """
    Track the current state of a walk-forward backtest
    
    This maintains all timing information and triggers for the backtest process
    """
    
    current_date: datetime
    backtest_start: datetime
    backtest_end: datetime
    
    # Last action dates
    last_rebalance: Optional[datetime] = None
    last_retrain: Optional[datetime] = None
    last_hyperparam_refresh: Optional[datetime] = None
    
    # Performance tracking
    recent_performance: List[float] = field(default_factory=list)
    performance_window: int = 10  # Track last 10 periods
    
    # Artifact tracking
    total_rebalances: int = 0
    total_retrains: int = 0
    total_hyperparam_refreshes: int = 0
    
    def __post_init__(self):
        """Initialize state"""
        if self.last_rebalance is None:
            self.last_rebalance = self.backtest_start
        if self.last_retrain is None:
            self.last_retrain = self.backtest_start
        if self.last_hyperparam_refresh is None:
            self.last_hyperparam_refresh = self.backtest_start
    
    def update_performance(self, performance: float):
        """Update recent performance tracking"""
        self.recent_performance.append(performance)
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance.pop(0)
    
    def get_avg_recent_performance(self) -> float:
        """Get average recent performance"""
        if not self.recent_performance:
            return 0.0
        return np.mean(self.recent_performance)
    
    def should_force_retrain(self, threshold: float) -> bool:
        """Check if performance has degraded enough to force retraining"""
        if len(self.recent_performance) < 3:
            return False
        
        recent_avg = np.mean(self.recent_performance[-3:])
        overall_avg = self.get_avg_recent_performance()
        
        if overall_avg <= 0:
            return False
        
        degradation = (overall_avg - recent_avg) / overall_avg
        return degradation > threshold
    
    def record_rebalance(self, date: datetime):
        """Record a rebalancing event"""
        self.last_rebalance = date
        self.total_rebalances += 1
    
    def record_retrain(self, date: datetime):
        """Record a retraining event"""
        self.last_retrain = date
        self.total_retrains += 1
    
    def record_hyperparam_refresh(self, date: datetime):
        """Record a hyperparameter refresh event"""
        self.last_hyperparam_refresh = date
        self.total_hyperparam_refreshes += 1
    
    def get_progress_pct(self) -> float:
        """Get backtest progress percentage"""
        total_days = (self.backtest_end - self.backtest_start).days
        elapsed_days = (self.current_date - self.backtest_start).days
        return min(100.0, max(0.0, elapsed_days / total_days * 100))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            'current_date': self.current_date.isoformat(),
            'backtest_start': self.backtest_start.isoformat(),
            'backtest_end': self.backtest_end.isoformat(),
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'last_hyperparam_refresh': self.last_hyperparam_refresh.isoformat() if self.last_hyperparam_refresh else None,
            'recent_performance': self.recent_performance,
            'performance_window': self.performance_window,
            'total_rebalances': self.total_rebalances,
            'total_retrains': self.total_retrains,
            'total_hyperparam_refreshes': self.total_hyperparam_refreshes
        }


def create_default_backtest_config() -> BacktestConfig:
    """Create a default backtest configuration following best practices"""
    return BacktestConfig(
        rebalance_cadence=CadenceType.WEEKLY,
        retrain_cadence=CadenceType.MONTHLY,
        hyperparam_refresh_cadence=CadenceType.QUARTERLY,
        data_window_type=DataWindowType.EXPANDING,
        min_training_days=252,  # 1 year minimum
        validation_split=0.2,
        use_optuna=True,
        optuna_trials=100,
        optuna_timeout=3600,
        store_artifacts=True,
        enable_pbo_detection=True,
        market_data_lag_days=1
    )


def create_conservative_backtest_config() -> BacktestConfig:
    """Create a conservative backtest configuration to minimize overfitting"""
    return BacktestConfig(
        rebalance_cadence=CadenceType.MONTHLY,  # Less frequent rebalancing
        retrain_cadence=CadenceType.QUARTERLY,  # Less frequent retraining
        hyperparam_refresh_cadence=CadenceType.ANNUALLY,  # Very conservative hyperparam refresh
        data_window_type=DataWindowType.ROLLING_3Y,  # Fixed 3-year window
        min_training_days=504,  # 2 years minimum
        validation_split=0.3,  # Larger validation set
        use_optuna=True,
        optuna_trials=50,  # Fewer trials
        optuna_timeout=1800,  # Shorter timeout
        store_artifacts=True,
        enable_pbo_detection=True,
        pbo_trials=2000,  # More PBO trials
        force_retrain_threshold=0.15,  # Higher threshold for forced retraining
        market_data_lag_days=2  # More conservative data lag
    )


def create_aggressive_backtest_config() -> BacktestConfig:
    """Create an aggressive backtest configuration for maximum adaptability"""
    return BacktestConfig(
        rebalance_cadence=CadenceType.DAILY,  # Daily rebalancing
        retrain_cadence=CadenceType.WEEKLY,  # Weekly retraining
        hyperparam_refresh_cadence=CadenceType.MONTHLY,  # Monthly hyperparam refresh
        data_window_type=DataWindowType.ROLLING_2Y,  # Shorter, more adaptive window
        min_training_days=126,  # 6 months minimum
        validation_split=0.15,  # Smaller validation for more training data
        use_optuna=True,
        optuna_trials=200,  # More trials
        optuna_timeout=7200,  # Longer timeout
        store_artifacts=True,
        enable_pbo_detection=True,
        allow_partial_retraining=True,
        force_retrain_threshold=0.05,  # Lower threshold for forced retraining
        market_data_lag_days=1
    )


if __name__ == "__main__":
    # Example usage
    print("🏗️ Backtest Configuration Examples")
    print("=" * 50)
    
    # Default configuration
    config = create_default_backtest_config()
    print("📊 Default Config:")
    print(f"  Rebalance: {config.rebalance_cadence.value}")
    print(f"  Retrain: {config.retrain_cadence.value}")
    print(f"  Hyperparam: {config.hyperparam_refresh_cadence.value}")
    print(f"  Data Window: {config.data_window_type.value}")
    print(f"  Config Hash: {config.get_config_hash()}")
    
    # Test timing logic
    print("\n📅 Timing Logic Test:")
    start_date = datetime(2020, 1, 1)
    state = BacktestState(
        current_date=start_date,
        backtest_start=start_date,
        backtest_end=datetime(2023, 1, 1)
    )
    
    # Simulate 35 days later
    test_date = start_date + timedelta(days=35)
    print(f"  After 35 days ({test_date.strftime('%Y-%m-%d')}):")
    print(f"    Should rebalance: {config.should_rebalance(test_date, state.last_rebalance)}")
    print(f"    Should retrain: {config.should_retrain(test_date, state.last_retrain)}")
    print(f"    Should refresh hyperparams: {config.should_refresh_hyperparams(test_date, state.last_hyperparam_refresh)}")
    
    # Test data windows
    train_start, train_end = config.get_training_window(test_date, start_date)
    print(f"  Training window: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    
    val_start, val_end, adj_train_end = config.get_validation_window(train_start, train_end)
    print(f"  Validation window: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")