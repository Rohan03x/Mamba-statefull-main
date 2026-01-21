"""
Configuration classes to reduce parameter complexity in the DCF Suite
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ForecastConfig:
    """Configuration for AI forecasting models"""
    forecast_horizon: int = 5
    quantiles: List[float] = None
    sequence_length: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    dropout: float = 0.1
    hidden_size: int = 64
    num_layers: int = 2
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forecasting"""
    num_experts: int = 3
    gating_hidden_size: int = 32
    expert_weights: Dict[str, float] = None
    min_expert_weight: float = 0.01
    weight_temperature: float = 1.0
    use_dynamic_weighting: bool = True
    
    def __post_init__(self):
        if self.expert_weights is None:
            # Equal weights for all experts
            weight = 1.0 / self.num_experts
            self.expert_weights = {
                'transformer': weight,
                'lightgbm': weight,
                'ar': weight
            }


@dataclass
class ValidationConfig:
    """Configuration for model validation and cross-validation"""
    cv_method: str = 'time_series'
    cv_folds: int = 5
    cv_test_size: int = 50
    cv_gap: int = 0
    min_train_size: int = 100
    primary_metric: str = 'mae'
    include_stability_penalty: bool = False
    stability_weight: float = 0.1
    include_robustness_penalty: bool = False
    robustness_weight: float = 0.05


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    n_trials: int = 100
    optimization_method: str = 'optuna'  # 'optuna', 'random', 'grid', 'bayesian'
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1
    random_seed: int = 42
    
    # Parameter space configuration
    optimize_forecast_horizon: bool = True
    horizon_min: int = 1
    horizon_max: int = 10
    horizon_step: int = 1
    
    optimize_quantiles: bool = True
    min_quantiles: int = 3
    max_quantiles: int = 9
    
    optimize_ensemble_weights: bool = True
    
    # Caching
    cache_results: bool = False
    cache_directory: str = "./cache"


@dataclass
class PrecisionConfig:
    """Configuration for numerical precision settings"""
    decimal_precision: int = 50
    tolerance_strict: float = 1e-15
    tolerance_standard: float = 1e-10
    tolerance_relaxed: float = 1e-6
    use_high_precision: bool = False
    torch_dtype: str = 'float32'  # 'float32' or 'float64'
    numpy_dtype: str = 'float64'


@dataclass
class AnchorsConfig:
    """Configuration for options anchoring"""
    anchor_weight: float = 0.3
    max_days_difference: int = 7  # Max difference in days for expiry matching
    min_option_volume: int = 10
    min_open_interest: int = 100
    use_implied_volatility: bool = True
    blend_method: str = 'weighted_average'  # 'weighted_average', 'bayesian'


@dataclass
class VisualizationConfig:
    """Configuration for plots and visualizations"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn'
    color_palette: str = 'viridis'
    save_plots: bool = False
    output_directory: str = "./plots"
    plot_format: str = 'png'
    show_confidence_intervals: bool = True
    confidence_level: float = 0.95


@dataclass
class DataConfig:
    """Configuration for data handling and preprocessing"""
    max_lookback_days: int = 1000
    min_data_points: int = 50
    handle_missing_data: str = 'interpolate'  # 'drop', 'interpolate', 'forward_fill'
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'none'
    outlier_threshold: float = 3.0
    normalize_features: bool = True
    feature_engineering: bool = True


@dataclass
class RiskConfig:
    """Configuration for risk calculations and metrics"""
    confidence_levels: List[float] = None
    var_method: str = 'historical'  # 'historical', 'parametric', 'monte_carlo'
    expected_shortfall_alpha: float = 0.05
    max_drawdown_lookback: int = 252
    volatility_window: int = 30
    correlation_window: int = 60
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


@dataclass
class BacktestConfig:
    """Configuration for backtesting and walk-forward validation"""
    initial_window_size: int = 252
    step_size: int = 21  # Monthly steps
    min_window_size: int = 100
    max_window_size: int = 1000
    refit_frequency: int = 63  # Quarterly refitting
    performance_metrics: List[str] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = [
                'mae', 'rmse', 'direction_accuracy', 'sharpe_ratio'
            ]


class ConfigManager:
    """Manager class to handle all configuration objects"""
    
    def __init__(self):
        self.forecast = ForecastConfig()
        self.ensemble = EnsembleConfig()
        self.validation = ValidationConfig()
        self.optimization = OptimizationConfig()
        self.precision = PrecisionConfig()
        self.anchors = AnchorsConfig()
        self.visualization = VisualizationConfig()
        self.data = DataConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
    
    def update_config(self, config_name: str, **kwargs):
        """Update a specific configuration with new values"""
        if hasattr(self, config_name):
            config = getattr(self, config_name)
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"Unknown parameter {key} for {config_name}")
        else:
            raise ValueError(f"Unknown configuration {config_name}")
    
    def get_config(self, config_name: str):
        """Get a specific configuration object"""
        if hasattr(self, config_name):
            return getattr(self, config_name)
        else:
            raise ValueError(f"Unknown configuration {config_name}")
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all configurations to a dictionary"""
        return {
            'forecast': self.forecast.__dict__,
            'ensemble': self.ensemble.__dict__,
            'validation': self.validation.__dict__,
            'optimization': self.optimization.__dict__,
            'precision': self.precision.__dict__,
            'anchors': self.anchors.__dict__,
            'visualization': self.visualization.__dict__,
            'data': self.data.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__
        }
    
    def from_dict(self, config_dict: Dict[str, Dict[str, Any]]):
        """Load configurations from a dictionary"""
        for config_name, config_values in config_dict.items():
            if hasattr(self, config_name):
                config = getattr(self, config_name)
                for key, value in config_values.items():
                    if hasattr(config, key):
                        setattr(config, key, value)


# Factory functions for common configurations
def create_development_config() -> ConfigManager:
    """Create configuration optimized for development/testing"""
    config = ConfigManager()
    
    # Faster training for development
    config.forecast.num_epochs = 10
    config.forecast.sequence_length = 10
    config.optimization.n_trials = 20
    config.validation.cv_folds = 3
    
    return config


def create_production_config() -> ConfigManager:
    """Create configuration optimized for production use"""
    config = ConfigManager()
    
    # More robust settings for production
    config.forecast.num_epochs = 100
    config.forecast.sequence_length = 30
    config.optimization.n_trials = 500
    config.validation.cv_folds = 10
    config.precision.use_high_precision = True
    
    return config


def create_fast_config() -> ConfigManager:
    """Create configuration for quick experiments"""
    config = ConfigManager()
    
    # Minimal settings for speed
    config.forecast.num_epochs = 5
    config.forecast.sequence_length = 5
    config.optimization.n_trials = 10
    config.validation.cv_folds = 2
    config.ensemble.num_experts = 2
    
    return config