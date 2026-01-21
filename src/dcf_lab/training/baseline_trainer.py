"""
Baseline Training Pipeline for Financial ML Models

Implements comprehensive model training and evaluation pipeline with:
- Per-fold training for all model families (TFT, LightGBM, ARIMA, etc.)
- Rich metrics: RMSE/MAE, directional accuracy, calibration, interval coverage
- Calibration utilities and reliability diagrams
- Integration with existing dcf_lab models
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator
import lightgbm as lgb
from scipy import stats

# Import our CV framework
from ..cv import CVConfig, WalkForwardValidator, create_target_variable

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline"""
    # Cross-validation
    cv_config: CVConfig = field(default_factory=CVConfig)
    
    # Model families to train
    model_families: List[str] = field(default_factory=lambda: [
        'lightgbm', 'random_forest', 'ridge', 'tft', 'arima'
    ])
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'mae', 'mape', 'directional_accuracy',
        'information_coefficient', 'hit_rate', 'sharpe_ratio', 'max_drawdown'
    ])
    
    # Calibration settings
    enable_calibration: bool = True
    calibration_bins: int = 10
    
    # Interval coverage levels
    coverage_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    
    # Output settings
    output_dir: Optional[str] = None
    save_predictions: bool = True
    save_models: bool = False
    plot_results: bool = True


@dataclass
class ModelResult:
    """Results for a single model"""
    model_name: str
    fold_results: List[Dict[str, float]]
    aggregated_metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    fold_indices: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    calibration_data: Optional[Dict] = None
    interval_coverage: Optional[Dict[str, float]] = None


class MetricsCalculator:
    """Calculate comprehensive financial ML metrics"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle division by zero)
        non_zero_mask = np.abs(y_true) > 1e-8
        if np.any(non_zero_mask):
            mape_values = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
            metrics['mape'] = np.mean(mape_values) * 100
        else:
            metrics['mape'] = np.inf
            
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
        
        return metrics
    
    @staticmethod
    def calculate_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate directional prediction metrics"""
        metrics = {}
        
        # Direction accuracy
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(y_true_sign == y_pred_sign)
        
        # Hit rate (percentage of correct direction predictions)
        metrics['hit_rate'] = metrics['directional_accuracy']
        
        # Up/down capture
        up_mask = y_true > 0
        down_mask = y_true < 0
        
        if np.any(up_mask):
            metrics['up_capture'] = np.mean(y_pred_sign[up_mask] == 1)
        else:
            metrics['up_capture'] = 0.0
            
        if np.any(down_mask):
            metrics['down_capture'] = np.mean(y_pred_sign[down_mask] == -1)
        else:
            metrics['down_capture'] = 0.0
            
        return metrics
    
    @staticmethod
    def calculate_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate Information Coefficient (Spearman correlation)"""
        try:
            ic, p_value = stats.spearmanr(y_true, y_pred)
            return {
                'information_coefficient': ic if not np.isnan(ic) else 0.0,
                'ic_p_value': p_value if not np.isnan(p_value) else 1.0
            }
        except Exception:
            return {'information_coefficient': 0.0, 'ic_p_value': 1.0}
    
    @staticmethod
    def calculate_financial_metrics(returns: np.ndarray) -> Dict[str, float]:
        """Calculate financial performance metrics from returns"""
        metrics = {}
        
        if len(returns) == 0:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0}
        
        # Annualized Sharpe ratio (assuming daily returns)
        metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Volatility (annualized)
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Win rate
        metrics['win_rate'] = np.mean(returns > 0)
        
        return metrics


class CalibrationAnalyzer:
    """Analyze and visualize model calibration"""
    
    @staticmethod
    def analyze_calibration(y_true: np.ndarray, y_prob: np.ndarray, 
                          n_bins: int = 10) -> Dict[str, Any]:
        """Analyze probability calibration"""
        # Convert to binary classification if needed
        if len(np.unique(y_true)) > 2:
            y_true_binary = (y_true > np.median(y_true)).astype(int)
        else:
            y_true_binary = y_true.astype(int)
            
        # Ensure probabilities are in [0, 1]
        y_prob_norm = (y_prob - np.min(y_prob)) / (np.max(y_prob) - np.min(y_prob) + 1e-8)
        
        try:
            # Calibration curve
            fraction_pos, mean_pred = calibration_curve(
                y_true_binary, y_prob_norm, n_bins=n_bins, strategy='uniform'
            )
            
            # Brier score
            brier_score = np.mean((y_prob_norm - y_true_binary) ** 2)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_norm > bin_lower) & (y_prob_norm <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_binary[in_bin].mean()
                    avg_confidence_in_bin = y_prob_norm[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
            return {
                'fraction_positive': fraction_pos,
                'mean_predicted': mean_pred,
                'brier_score': brier_score,
                'expected_calibration_error': ece,
                'bin_boundaries': bin_boundaries
            }
            
        except Exception as e:
            logger.warning(f"Calibration analysis failed: {e}")
            return {
                'fraction_positive': np.array([]),
                'mean_predicted': np.array([]),
                'brier_score': np.inf,
                'expected_calibration_error': np.inf,
                'bin_boundaries': np.array([])
            }
    
    @staticmethod
    def plot_calibration_curve(calibration_data: Dict[str, Any], 
                             model_name: str, output_dir: Optional[str] = None):
        """Plot calibration curve"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            
            # Model calibration
            if len(calibration_data['fraction_positive']) > 0:
                ax.plot(calibration_data['mean_predicted'], 
                       calibration_data['fraction_positive'],
                       'o-', label=f'{model_name}')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add ECE in text box
            ece = calibration_data.get('expected_calibration_error', np.inf)
            brier = calibration_data.get('brier_score', np.inf)
            textstr = f'ECE: {ece:.3f}\nBrier Score: {brier:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            if output_dir:
                output_path = Path(output_dir) / f'{model_name}_calibration.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved calibration plot: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot calibration curve: {e}")


class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseEstimator:
        """Create model instance by name"""
        if model_name == 'lightgbm':
            import os
            params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
            }
            # Add GPU support via environment variable
            device = os.environ.get('LIGHTGBM_DEVICE', 'cpu')
            if device == 'gpu':
                params['device_type'] = 'cuda'
                params['gpu_platform_id'] = int(os.environ.get('LIGHTGBM_GPU_PLATFORM_ID', 0))
                params['gpu_device_id'] = int(os.environ.get('LIGHTGBM_GPU_DEVICE_ID', 0))
            params.update(kwargs)
            return lgb.LGBMRegressor(**params)
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                **kwargs
            )
        elif model_name == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, **kwargs)
        elif model_name == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(alpha=0.1, l1_ratio=0.5, **kwargs)
        elif model_name == 'xgboost':
            try:
                import os
                import xgboost as xgb
                params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                }
                # Add GPU support via environment variable
                device = os.environ.get('XGB_DEVICE', 'cpu')
                if device == 'gpu':
                    params['device'] = 'cuda'
                    params['tree_method'] = 'hist'
                else:
                    params['tree_method'] = os.environ.get('XGB_TREE_METHOD', 'hist')
                params.update(kwargs)
                return xgb.XGBRegressor(**params)
            except ImportError:
                logger.warning("XGBoost not available, using LightGBM instead")
                return ModelFactory.create_model('lightgbm', **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_name}")


class BaselineTrainer:
    """Main training pipeline for baseline models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.validator = WalkForwardValidator(config.cv_config)
        self.metrics_calc = MetricsCalculator()
        self.calibration_analyzer = CalibrationAnalyzer()
        
        # Setup output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelResult]:
        """Train all model families and return results"""
        results = {}
        
        logger.info(f"Training {len(self.config.model_families)} model families")
        logger.info(f"Data shape: {X.shape}, Target shape: {y.shape}")
        
        for model_name in self.config.model_families:
            try:
                logger.info(f"Training {model_name}...")
                result = self._train_single_model(model_name, X, y)
                results[model_name] = result
                
                # Log summary metrics
                agg_metrics = result.aggregated_metrics
                rmse = agg_metrics.get('rmse_mean', 'N/A')
                dir_acc = agg_metrics.get('directional_accuracy_mean', 'N/A')
                ic = agg_metrics.get('information_coefficient_mean', 'N/A')
                
                logger.info(f"{model_name} - RMSE: {rmse:.4f}, "
                          f"Dir Acc: {dir_acc:.3f}, IC: {ic:.3f}")
                          
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
                
        # Generate comparison report
        if results:
            self._generate_comparison_report(results)
            
        return results
    
    def _train_single_model(self, model_name: str, X: pd.DataFrame, 
                           y: pd.Series) -> ModelResult:
        """Train a single model family"""
        # Create model instance
        model = ModelFactory.create_model(model_name)
        
        # Run walk-forward validation
        cv_results = self.validator.validate_model(model, X, y)
        
        # Calculate additional metrics
        predictions = np.array(cv_results['predictions'])
        actuals = np.array(cv_results['actuals'])
        fold_indices = np.array(cv_results['fold_indices'])
        
        # Comprehensive metrics
        all_metrics = {}
        all_metrics.update(self.metrics_calc.calculate_regression_metrics(actuals, predictions))
        all_metrics.update(self.metrics_calc.calculate_directional_metrics(actuals, predictions))
        all_metrics.update(self.metrics_calc.calculate_information_coefficient(actuals, predictions))
        
        # Financial metrics (if returns)
        if self.config.cv_config.target_type in ['log_returns', 'abs_returns']:
            financial_metrics = self.metrics_calc.calculate_financial_metrics(predictions)
            all_metrics.update(financial_metrics)
            
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            
        # Calibration analysis
        calibration_data = None
        if self.config.enable_calibration:
            try:
                # Use predictions as proxy probabilities for regression
                calibration_data = self.calibration_analyzer.analyze_calibration(
                    actuals, predictions, self.config.calibration_bins
                )
                
                if self.config.plot_results and self.output_dir:
                    self.calibration_analyzer.plot_calibration_curve(
                        calibration_data, model_name, str(self.output_dir)
                    )
            except Exception as e:
                logger.warning(f"Calibration analysis failed for {model_name}: {e}")
        
        # Interval coverage (simple version using quantiles)
        interval_coverage = self._calculate_interval_coverage(actuals, predictions)
        
        return ModelResult(
            model_name=model_name,
            fold_results=cv_results['fold_results'],
            aggregated_metrics={**cv_results['aggregated_metrics'], **all_metrics},
            predictions=predictions,
            actuals=actuals,
            fold_indices=fold_indices,
            feature_importance=feature_importance,
            calibration_data=calibration_data,
            interval_coverage=interval_coverage
        )
    
    def _calculate_interval_coverage(self, actuals: np.ndarray, 
                                   predictions: np.ndarray) -> Dict[str, float]:
        """Calculate empirical interval coverage"""
        residuals = actuals - predictions
        coverage = {}
        
        for level in self.config.coverage_levels:
            # Simple quantile-based intervals
            alpha = 1 - level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            # Calculate interval bounds from residual distribution
            lower_bound = np.quantile(residuals, lower_q)
            upper_bound = np.quantile(residuals, upper_q)
            
            # Check coverage
            in_interval = (residuals >= lower_bound) & (residuals <= upper_bound)
            empirical_coverage = np.mean(in_interval)
            
            coverage[f'coverage_{level:.0%}'] = empirical_coverage
            
        return coverage
    
    def _generate_comparison_report(self, results: Dict[str, ModelResult]):
        """Generate comparison report across models"""
        try:
            # Create comparison DataFrame
            comparison_data = []
            
            for model_name, result in results.items():
                row = {'Model': model_name}
                
                # Add key metrics
                metrics = result.aggregated_metrics
                for metric in ['rmse_mean', 'mae_mean', 'directional_accuracy_mean', 
                              'information_coefficient_mean', 'sharpe_ratio', 'max_drawdown']:
                    row[metric] = metrics.get(metric, np.nan)
                    
                comparison_data.append(row)
                
            comparison_df = pd.DataFrame(comparison_data)
            
            logger.info("\n" + "="*80)
            logger.info("MODEL COMPARISON REPORT")
            logger.info("="*80)
            logger.info(comparison_df.to_string(index=False, float_format='%.4f'))
            
            # Save to file
            if self.output_dir:
                comparison_path = self.output_dir / 'model_comparison.csv'
                comparison_df.to_csv(comparison_path, index=False)
                logger.info(f"Saved comparison report: {comparison_path}")
                
                # Save detailed results
                results_path = self.output_dir / 'detailed_results.json'
                self._save_results_json(results, results_path)
                
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
    
    def _save_results_json(self, results: Dict[str, ModelResult], output_path: Path):
        """Save results to JSON (serializable parts only)"""
        try:
            json_results = {}
            
            for model_name, result in results.items():
                json_results[model_name] = {
                    'aggregated_metrics': result.aggregated_metrics,
                    'feature_importance': result.feature_importance,
                    'interval_coverage': result.interval_coverage,
                    'n_folds': len(result.fold_results),
                    'n_predictions': len(result.predictions)
                }
                
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
                
            logger.info(f"Saved detailed results: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results JSON: {e}")


# Example usage and testing
def example_baseline_training():
    """Demonstrate baseline training pipeline"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Synthetic price data
    rng = np.random.default_rng(42)
    trend = np.linspace(100, 200, n_samples)
    noise = rng.normal(0, 5, n_samples)
    prices = trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'volume': rng.lognormal(10, 1, n_samples),
        'high': prices * (1 + rng.uniform(0, 0.02, n_samples)),
        'low': prices * (1 - rng.uniform(0, 0.02, n_samples))
    }, index=dates)
    
    # Add features
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    df['rsi'] = 50  # Placeholder
    df = df.dropna()
    
    # Configuration
    cv_config = CVConfig(
        n_splits=3,
        test_size=30,
        gap=1,
        horizon=1,
        target_type='log_returns'
    )
    
    training_config = TrainingConfig(
        cv_config=cv_config,
        model_families=['ridge', 'lightgbm', 'random_forest'],
        output_dir='baseline_results'
    )
    
    # Create target
    target = create_target_variable(df, cv_config)
    
    # Align data
    common_index = df.index.intersection(target.index)
    X = df.loc[common_index, ['returns', 'sma_20', 'volatility', 'rsi']]
    y = target.loc[common_index]
    
    # Train models
    trainer = BaselineTrainer(training_config)
    results = trainer.train_all_models(X, y)
    
    print(f"\nTrained {len(results)} models successfully!")
    return results


if __name__ == "__main__":
    example_baseline_training()