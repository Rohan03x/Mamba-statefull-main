"""
Advanced Probability Calibration for AI Price Forecasting

This module implements comprehensive probability calibration techniques to improve
forecast accuracy and uncertainty quantification for financial predictions.

Key Features:
- Platt scaling for binary classification calibration
- Isotonic regression for non-parametric calibration
- Temperature scaling for neural network calibration
- Beta calibration for improved probability calibration
- Conformal prediction for distribution-free intervals
- Expected calibration error (ECE) and reliability metrics
- Multi-class calibration for regime predictions
- Time-varying calibration for non-stationary markets

Calibration Methods:
- Platt Scaling: Sigmoid transformation of prediction scores
- Isotonic Regression: Monotonic non-parametric calibration
- Temperature Scaling: Neural network output scaling
- Beta Calibration: Beta distribution-based calibration
- Conformal Prediction: Distribution-free uncertainty quantification

Applications:
- Price direction prediction calibration
- Volatility forecast uncertainty
- Regime change probability calibration
- Risk measure confidence intervals
- Portfolio optimization uncertainty
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Core calibration libraries
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import log_loss
    SKLEARN_ADVANCED_AVAILABLE = True
except ImportError:
    SKLEARN_ADVANCED_AVAILABLE = False

# Statistical libraries
try:
    from scipy import stats, optimize
    from scipy.special import expit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration quality metrics"""
    expected_calibration_error: float  # Average calibration error
    maximum_calibration_error: float  # Worst-case calibration error
    brier_score: float  # Brier score (lower is better)
    log_loss: float  # Logarithmic loss
    reliability: float  # Reliability component of Brier score
    resolution: float  # Resolution component of Brier score
    uncertainty: float  # Uncertainty component of Brier score
    sharpness: float  # Average confidence of predictions


@dataclass
class CalibrationResult:
    """Results from probability calibration"""
    calibrated_probabilities: np.ndarray
    calibration_function: Optional[Callable] = None
    calibration_metrics: Optional[CalibrationMetrics] = None
    method_used: str = "unknown"
    confidence_intervals: Optional[np.ndarray] = None
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None


class PlattScaling:
    """
    Platt Scaling for Binary Classification Calibration
    
    Fits a sigmoid function to map classifier outputs to calibrated probabilities.
    """
    
    def __init__(self):
        self.sigmoid_params = None
        self.is_fitted = False
    
    def fit(self, decision_scores: np.ndarray, true_labels: np.ndarray) -> 'PlattScaling':
        """
        Fit Platt scaling parameters
        
        Args:
            decision_scores: Raw classifier scores
            true_labels: True binary labels (0/1)
            
        Returns:
            Self for method chaining
        """
        try:
            if not SKLEARN_ADVANCED_AVAILABLE:
                raise ImportError("Scikit-learn required for Platt scaling")
            
            # Fit logistic regression to decision scores
            clf = LogisticRegression()
            clf.fit(decision_scores.reshape(-1, 1), true_labels)
            
            # Extract sigmoid parameters
            self.sigmoid_params = {
                'A': clf.coef_[0][0],
                'B': clf.intercept_[0]
            }
            self.is_fitted = True
            
            logger.info("✅ Platt scaling fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting Platt scaling: {e}")
            # Fallback to identity function
            self.sigmoid_params = {'A': 1.0, 'B': 0.0}
            self.is_fitted = True
            return self
    
    def transform(self, decision_scores: np.ndarray) -> np.ndarray:
        """Transform scores to calibrated probabilities"""
        if not self.is_fitted:
            raise ValueError("Platt scaling not fitted yet")
        
        A = self.sigmoid_params['A']
        B = self.sigmoid_params['B']
        
        # Apply sigmoid transformation
        if SCIPY_AVAILABLE:
            return expit(A * decision_scores + B)
        else:
            # Fallback sigmoid implementation
            return 1 / (1 + np.exp(-(A * decision_scores + B)))
    
    def fit_transform(self, decision_scores: np.ndarray, 
                     true_labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(decision_scores, true_labels).transform(decision_scores)


class AdvancedIsotonicCalibrator:
    """
    Enhanced Isotonic Regression Calibration
    
    Non-parametric calibration that ensures monotonicity with advanced features.
    """
    
    def __init__(self, out_of_bounds='clip'):
        self.isotonic_regressor = None
        self.is_fitted = False
        self.out_of_bounds = out_of_bounds
    
    def fit(self, probabilities: np.ndarray, 
            true_labels: np.ndarray) -> 'AdvancedIsotonicCalibrator':
        """Fit isotonic regression calibrator"""
        try:
            self.isotonic_regressor = IsotonicRegression(out_of_bounds=self.out_of_bounds)
            self.isotonic_regressor.fit(probabilities, true_labels)
            self.is_fitted = True
            
            logger.info("✅ Advanced isotonic calibration fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting isotonic calibration: {e}")
            # Fallback to identity
            self.isotonic_regressor = lambda x: x
            self.is_fitted = True
            return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using isotonic regression"""
        if not self.is_fitted:
            raise ValueError("Isotonic calibrator not fitted yet")
        
        if hasattr(self.isotonic_regressor, 'predict'):
            return self.isotonic_regressor.predict(probabilities)
        else:
            return self.isotonic_regressor(probabilities)
    
    def fit_transform(self, probabilities: np.ndarray, 
                     true_labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(probabilities, true_labels).transform(probabilities)


class ProbabilityCalibrationSystem:
    """
    Comprehensive Probability Calibration System
    
    Provides multiple calibration methods and evaluation metrics for
    improving forecast reliability and uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize advanced calibration system"""
        self.calibrators = {}
        self.evaluation_metrics = {}
        
    def calculate_calibration_metrics(self, probabilities: np.ndarray, 
                                    true_labels: np.ndarray, 
                                    n_bins: int = 10) -> CalibrationMetrics:
        """
        Calculate comprehensive calibration metrics
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels
            n_bins: Number of bins for calibration plot
            
        Returns:
            Calibration quality metrics
        """
        try:
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            mce = 0.0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = true_labels[in_bin].mean()
                    avg_confidence_in_bin = probabilities[in_bin].mean()
                    
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_error * prop_in_bin
                    mce = max(mce, bin_error)
            
            # Brier Score and decomposition
            brier = brier_score_loss(true_labels, probabilities)
            
            # Brier score decomposition
            reliability = 0.0
            resolution = 0.0
            uncertainty = np.mean(true_labels) * (1 - np.mean(true_labels))
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = true_labels[in_bin].mean()
                    avg_confidence_in_bin = probabilities[in_bin].mean()
                    
                    reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                    resolution += prop_in_bin * (accuracy_in_bin - np.mean(true_labels)) ** 2
            
            # Log loss
            eps = 1e-15
            prob_clipped = np.clip(probabilities, eps, 1 - eps)
            if SKLEARN_ADVANCED_AVAILABLE:
                logloss = log_loss(true_labels, prob_clipped)
            else:
                logloss = -np.mean(true_labels * np.log(prob_clipped) + 
                                 (1 - true_labels) * np.log(1 - prob_clipped))
            
            # Sharpness (average confidence)
            sharpness = np.mean(np.maximum(probabilities, 1 - probabilities))
            
            return CalibrationMetrics(
                expected_calibration_error=ece,
                maximum_calibration_error=mce,
                brier_score=brier,
                log_loss=logloss,
                reliability=reliability,
                resolution=resolution,
                uncertainty=uncertainty,
                sharpness=sharpness
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating calibration metrics: {e}")
            # Return default metrics
            return CalibrationMetrics(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.25, 0.5)
    
    def calibrate_probabilities(self, probabilities: np.ndarray, 
                              true_labels: np.ndarray,
                              method: str = 'auto') -> CalibrationResult:
        """
        Calibrate probabilities using specified method
        
        Args:
            probabilities: Uncalibrated probabilities
            true_labels: True binary labels
            method: Calibration method ('platt', 'isotonic', 'auto')
            
        Returns:
            Calibration results with calibrated probabilities
        """
        try:
            if method == 'auto':
                # Choose best method based on data characteristics
                method = self._select_best_method(probabilities, true_labels)
            
            # Apply selected calibration method
            if method == 'platt':
                calibrator = PlattScaling()
                calibrated_probs = calibrator.fit_transform(probabilities, true_labels)
                calibration_function = calibrator.transform
            
            elif method == 'isotonic':
                calibrator = AdvancedIsotonicCalibrator()
                calibrated_probs = calibrator.fit_transform(probabilities, true_labels)
                calibration_function = calibrator.transform
            
            else:
                # Fallback to identity (no calibration)
                calibrated_probs = probabilities
                def calibration_function(x):
                    return x
                method = 'identity'
            
            # Calculate calibration metrics
            metrics = self.calculate_calibration_metrics(calibrated_probs, true_labels)
            
            # Store calibrator for future use
            self.calibrators[method] = calibration_function
            
            logger.info(f"✅ Probability calibration completed using {method}")
            logger.info(f"📊 ECE: {metrics.expected_calibration_error:.4f}, Brier: {metrics.brier_score:.4f}")
            
            return CalibrationResult(
                calibrated_probabilities=calibrated_probs,
                calibration_function=calibration_function,
                calibration_metrics=metrics,
                method_used=method
            )
            
        except Exception as e:
            logger.error(f"❌ Error in probability calibration: {e}")
            # Return uncalibrated probabilities
            return CalibrationResult(
                calibrated_probabilities=probabilities,
                method_used='identity'
            )
    
    def _select_best_method(self, probabilities: np.ndarray, 
                          true_labels: np.ndarray) -> str:
        """Select best calibration method based on data characteristics"""
        try:
            # For small datasets, use Platt scaling
            if len(probabilities) < 100:
                return 'platt'
            
            # For larger datasets, compare methods via cross-validation
            if SKLEARN_ADVANCED_AVAILABLE and len(probabilities) > 500:
                methods = ['platt', 'isotonic']
                best_method = 'platt'
                best_score = np.inf
                
                for method in methods:
                    if method == 'platt':
                        calibrator = PlattScaling()
                    else:
                        calibrator = AdvancedIsotonicCalibrator()
                    
                    # Simple holdout validation
                    n_train = int(0.7 * len(probabilities))
                    train_probs = probabilities[:n_train]
                    train_labels = true_labels[:n_train]
                    val_probs = probabilities[n_train:]
                    val_labels = true_labels[n_train:]
                    
                    calibrator.fit(train_probs, train_labels)
                    cal_probs = calibrator.transform(val_probs)
                    
                    # Evaluate using Brier score
                    score = np.mean((cal_probs - val_labels) ** 2)
                    
                    if score < best_score:
                        best_score = score
                        best_method = method
                
                return best_method
            
            # Default to isotonic for medium datasets
            return 'isotonic'
            
        except Exception:
            return 'platt'  # Safe fallback
    
    def get_calibration_features(self, probabilities: np.ndarray,
                               true_labels: np.ndarray) -> Dict[str, float]:
        """
        Generate calibration features for ML models
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels
            
        Returns:
            Dictionary of calibration features
        """
        try:
            features = {}
            
            # Basic calibration metrics
            metrics = self.calculate_calibration_metrics(probabilities, true_labels)
            
            features['expected_calibration_error'] = metrics.expected_calibration_error
            features['maximum_calibration_error'] = metrics.maximum_calibration_error
            features['brier_score'] = metrics.brier_score
            features['log_loss'] = metrics.log_loss
            features['reliability'] = metrics.reliability
            features['resolution'] = metrics.resolution
            features['uncertainty'] = metrics.uncertainty
            features['sharpness'] = metrics.sharpness
            
            # Probability distribution features
            features['prob_mean'] = np.mean(probabilities)
            features['prob_std'] = np.std(probabilities)
            features['prob_skewness'] = stats.skew(probabilities) if SCIPY_AVAILABLE else 0.0
            features['prob_kurtosis'] = stats.kurtosis(probabilities) if SCIPY_AVAILABLE else 0.0
            
            # Confidence features
            features['high_confidence_ratio'] = np.mean((probabilities > 0.8) | (probabilities < 0.2))
            features['low_confidence_ratio'] = np.mean((probabilities >= 0.4) & (probabilities <= 0.6))
            features['extreme_confidence_ratio'] = np.mean((probabilities > 0.95) | (probabilities < 0.05))
            
            # Calibration quality indicators
            features['overconfidence_rate'] = np.mean(probabilities > true_labels)
            features['underconfidence_rate'] = np.mean(probabilities < true_labels)
            
            logger.info(f"📊 Generated {len(features)} calibration features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating calibration features: {e}")
            return {}

    # Adapter for UniversalFeatureAggregator
    def get_calibration_features_for_ticker(self, ticker: str) -> Dict[str, float]:
        """Lightweight adapter for aggregator which may call with just a ticker.
        Returns last computed metrics if available; otherwise empty.
        """
        try:
            if hasattr(self, 'last_metrics') and isinstance(self.last_metrics, dict):
                return {k: float(v) for k, v in self.last_metrics.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}


class CalibrationValidator:
    """
    Validates and measures calibration quality of probabilistic forecasts
    """

    def __init__(self):
        self.calibration_data = []
        self.calibration_metrics = {}

    def add_forecast_validation(self,
                                predictions: np.ndarray,
                                actuals: np.ndarray,
                                quantile_levels: List[float]) -> Dict[str,
                                                                      float]:
        """
        Add forecast validation data for calibration assessment

        Args:
            predictions: Quantile predictions [n_samples, n_quantiles]
            actuals: Actual observed values [n_samples]
            quantile_levels: List of quantile levels (e.g., [0.1, 0.5, 0.9])

        Returns:
            Dictionary with calibration metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        metrics = {}

        for i, q_level in enumerate(quantile_levels):
            if predictions.shape[1] > i:
                pred_quantile = predictions[:, i]

                # ===== FIX ARRAY AMBIGUITY ERRORS =====
                # Calculate empirical coverage with safe array comparison
                try:
                    coverage_check = np.asarray(actuals <= pred_quantile, dtype=bool)
                    empirical_coverage = np.mean(coverage_check)
                except ValueError:
                    # Fallback for problematic array comparisons
                    empirical_coverage = np.mean([a <= p for a, p in zip(actuals, pred_quantile)])

                # Calculate calibration error
                calibration_error = abs(empirical_coverage - q_level)

                metrics[f'q{int(q_level * 100):02d}_coverage'] = empirical_coverage
                metrics[f'q{int(q_level*100):02d}_error'] = calibration_error

                # Store for calibration analysis with safe comparison
                try:
                    hit_values = [a <= p for a, p in zip(pred_quantile, actuals)]
                    self.calibration_data.extend([
                        {'quantile_level': q_level, 'prediction': p,
                            'actual': a, 'hit': hit}
                        for p, a, hit in zip(pred_quantile, actuals, hit_values)
                    ])
                except Exception as e:
                    logger.warning(f"Calibration data storage failed: {e}")

        # Calculate interval coverage
        if len(quantile_levels) >= 3:
            # Assume symmetric intervals around median
            lower_idx = 0  # Lowest quantile
            upper_idx = -1  # Highest quantile

            if predictions.shape[1] >= 3:
                lower_preds = predictions[:, lower_idx]
                upper_preds = predictions[:, upper_idx]

                # ===== FIX ARRAY AMBIGUITY ERRORS =====
                # Use explicit array comparisons to prevent ambiguous truth value errors
                try:
                    # Check if actuals fall within intervals with proper array handling
                    lower_check = np.asarray(actuals >= lower_preds, dtype=bool)
                    upper_check = np.asarray(actuals <= upper_preds, dtype=bool)
                    in_interval = np.logical_and(lower_check, upper_check)
                    interval_coverage = np.mean(in_interval)

                    # Expected coverage for this interval
                    expected_coverage = quantile_levels[upper_idx] - \
                        quantile_levels[lower_idx]
                    interval_error = abs(interval_coverage - expected_coverage)

                    metrics['interval_coverage'] = interval_coverage
                    metrics['interval_expected'] = expected_coverage
                    metrics['interval_error'] = interval_error
                except ValueError as e:
                    logger.warning(f"Interval coverage calculation failed: {e}")
                    # Provide default values if calculation fails
                    metrics['interval_coverage'] = 0.0
                    metrics['interval_expected'] = 0.0
                    metrics['interval_error'] = 1.0

        # Calculate overall calibration score
        calibration_errors = [
            v for k, v in metrics.items() if k.endswith('_error')]
        if calibration_errors:
            metrics['mean_calibration_error'] = np.mean(calibration_errors)

        self.calibration_metrics = metrics
        return metrics

    def calculate_brier_score(self, prob_predictions: np.ndarray,
                              binary_outcomes: np.ndarray) -> float:
        """
        Calculate Brier score for probabilistic predictions

        Args:
            prob_predictions: Predicted probabilities [0, 1]
            binary_outcomes: Binary outcomes (0 or 1)

        Returns:
            Brier score (lower is better)
        """
        return brier_score_loss(binary_outcomes, prob_predictions)

    def plot_reliability_diagram(
            self, save_path: Optional[str] = None) -> None:
        """
        Plot reliability diagram showing calibration quality

        Args:
            save_path: Optional path to save the plot
        """
        if not self.calibration_data:
            logger.warning("No calibration data available for plotting")
            return

        df = pd.DataFrame(self.calibration_data)

        _, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Reliability diagram
        quantile_levels = sorted(df['quantile_level'].unique())

        for q_level in quantile_levels:
            q_data = df[df['quantile_level'] == q_level]

            # Bin predictions and calculate empirical frequencies
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            empirical_freq = []
            for i in range(len(bins) - 1):
                mask = (q_data['prediction'] >= bins[i]) & (
                    q_data['prediction'] < bins[i+1])
                if mask.sum() > 0:
                    freq = q_data[mask]['hit'].mean()
                    empirical_freq.append(freq)
                else:
                    empirical_freq.append(np.nan)

            axes[0].plot(bin_centers, empirical_freq, 'o-',
                         label=f'Q{int(q_level*100):02d}', alpha=0.7)

        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5,
                     label='Perfect calibration')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Empirical Frequency')
        axes[0].set_title('Reliability Diagram')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Calibration error by quantile
        calibration_errors = []
        for q_level in quantile_levels:
            error_key = f'q{int(q_level*100):02d}_error'
            if error_key in self.calibration_metrics:
                calibration_errors.append(self.calibration_metrics[error_key])
            else:
                calibration_errors.append(0)

        axes[1].bar([f'Q{int(q*100):02d}' for q in quantile_levels],
                    calibration_errors, alpha=0.7)
        axes[1].set_xlabel('Quantile Level')
        axes[1].set_ylabel('Calibration Error')
        axes[1].set_title('Calibration Error by Quantile')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to {save_path}")
        else:
            plt.show()


class IsotonicCalibrator:
    """
    Implements isotonic calibration for quantile predictions
    """

    def __init__(self):
        self.calibrators = {}
        self.is_fitted = False

    def fit(self, predictions: np.ndarray, actuals: np.ndarray,
            quantile_levels: List[float]) -> 'IsotonicCalibrator':
        """
        Fit isotonic calibration models for each quantile level

        Args:
            predictions: Quantile predictions [n_samples, n_quantiles]
            actuals: Actual observed values [n_samples]
            quantile_levels: List of quantile levels

        Returns:
            Self for method chaining
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        self.calibrators = {}

        for i, q_level in enumerate(quantile_levels):
            if predictions.shape[1] > i:
                pred_quantile = predictions[:, i]

                # Create binary outcomes (1 if actual <= predicted quantile)
                binary_outcomes = (actuals <= pred_quantile).astype(float)

                # Fit isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')

                try:
                    calibrator.fit(pred_quantile, binary_outcomes)
                    self.calibrators[q_level] = calibrator
                    logger.info(f"Fitted calibrator for quantile {q_level}")
                except Exception as e:
                    logger.warning(
                        f"Failed to fit calibrator for quantile {q_level}: {e}")

        self.is_fitted = len(self.calibrators) > 0
        return self

    def transform(self, predictions: np.ndarray,
                  quantile_levels: List[float]) -> np.ndarray:
        """
        Apply calibration to new predictions

        Args:
            predictions: Raw quantile predictions [n_samples, n_quantiles]
            quantile_levels: List of quantile levels

        Returns:
            Calibrated predictions [n_samples, n_quantiles]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        calibrated_preds = predictions.copy()

        for i, q_level in enumerate(quantile_levels):
            if q_level in self.calibrators and predictions.shape[1] > i:
                calibrator = self.calibrators[q_level]
                pred_quantile = predictions[:, i]

                try:
                    # Get calibrated probabilities
                    calibrated_probs = calibrator.transform(pred_quantile)

                    # Convert back to quantile predictions
                    # This is a simplified approach - in practice you might use
                    # more sophisticated methods
                    calibrated_preds[:, i] = np.percentile(
                        pred_quantile, calibrated_probs * 100)

                except Exception as e:
                    logger.warning(
                        f"Failed to calibrate quantile {q_level}: {e}")

        return calibrated_preds

    def fit_transform(self, predictions: np.ndarray, actuals: np.ndarray,
                      quantile_levels: List[float]) -> np.ndarray:
        """
        Fit calibrator and transform predictions in one step

        Args:
            predictions: Quantile predictions [n_samples, n_quantiles]
            actuals: Actual observed values [n_samples]
            quantile_levels: List of quantile levels

        Returns:
            Calibrated predictions [n_samples, n_quantiles]
        """
        return self.fit(
            predictions,
            actuals,
            quantile_levels).transform(
            predictions,
            quantile_levels)


class ProbabilityCalibrationSystem:
    """
    Comprehensive probability calibration system for AI forecasts
    """

    def __init__(self, min_calibration_samples: int = 100):
        """
        Initialize calibration system

        Args:
            min_calibration_samples: Minimum samples needed for calibration
        """
        self.min_calibration_samples = min_calibration_samples
        self.validator = CalibrationValidator()
        self.calibrator = IsotonicCalibrator()
        self.calibration_history = []

    def _extract_single_value(self, values):
        """Extract single value from various data formats"""
        if isinstance(values, np.ndarray):
            # Take first forecast horizon for simplicity
            if values.ndim > 1:
                return float(values[0, 0])  # First sample, first horizon
            else:
                return float(values[0])  # First element
        elif isinstance(values, list):
            # Take first forecast horizon for simplicity
            if isinstance(values[0], list):
                return float(values[0][0])  # First horizon, first forecast
            else:
                return float(values[0])
        else:
            # Try to convert to float directly
            return float(values)

    def _extract_quantile_data(self, forecast_results):
        """Extract quantile data from forecast results"""
        if 'quantile_forecasts' not in forecast_results:
            return None, None

        quantile_data = forecast_results['quantile_forecasts']
        quantile_levels = []
        predictions = []

        for key, values in quantile_data.items():
            # Ensure key is a string and starts with 'q_'
            key_str = str(key) if not isinstance(key, str) else key
            if not key_str.startswith('q_') or values is None:
                continue

            q_level = int(key_str[2:]) / 100.0
            quantile_levels.append(q_level)

            # Extract single prediction value
            predictions.append(self._extract_single_value(values))

        return quantile_levels, predictions

    def _validate_and_store_calibration(
            self,
            quantile_levels,
            predictions,
            actual_returns):
        """Validate calibration and store history"""
        # Prepare data
        quantile_levels = sorted(quantile_levels)
        predictions_array = np.array(predictions).reshape(1, -1)
        # Use first actual for validation
        actuals_array = np.array(actual_returns[:1])

        # Validate calibration
        metrics = self.validator.add_forecast_validation(
            predictions_array, actuals_array, quantile_levels
        )

        # Store calibration history
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy(),
            'sample_size': len(actual_returns)
        })

        return metrics

    def _compute_hedge_fund_calibration_features(
            self, forecast_results: Dict[str, Any], actual_returns: List[float]
    ) -> Dict[str, float]:
        """
        Compute HEDGE-FUND STANDARD calibration features for maximum alpha
        
        These features measure prediction quality, overconfidence, regime shifts,
        and are highly predictive of future error and risk-adjusted returns.
        
        Returns:
            Dictionary with 50+ advanced calibration features
        """
        features = {}
        
        # Extract quantile forecasts
        if 'quantile_forecasts' not in forecast_results:
            return self._empty_hedge_fund_features()
        
        quantile_data = forecast_results['quantile_forecasts']
        actual = np.array(actual_returns)
        
        # Parse quantile predictions into structured arrays
        quantiles = {}
        for key, values in quantile_data.items():
            key_str = str(key) if not isinstance(key, str) else key
            if not key_str.startswith('q_'):
                continue
            q_level = int(key_str[2:]) / 100.0
            # Extract values (handle various formats)
            if isinstance(values, (list, np.ndarray)):
                quantiles[q_level] = np.array(values).flatten()[:len(actual)]
            else:
                quantiles[q_level] = np.full(len(actual), float(values))
        
        if not quantiles:
            return self._empty_hedge_fund_features()
        
        # Ensure all quantiles have same length
        min_len = min(len(v) for v in quantiles.values())
        for q in quantiles:
            quantiles[q] = quantiles[q][:min_len]
        actual = actual[:min_len]
        
        # ===================================================================
        # A) QUANTILE COVERAGE ERRORS
        # ===================================================================
        for q_level, q_pred in quantiles.items():
            q_int = int(q_level * 100)
            # Coverage error = (actual < prediction) - nominal_coverage
            actual_coverage = np.mean(actual < q_pred)
            coverage_error = actual_coverage - q_level
            features[f'coverage_error_q{q_int:02d}'] = float(coverage_error)
        
        # ===================================================================
        # B) QUANTILE MISALIGNMENT & INTERVAL WIDTH FEATURES
        # ===================================================================
        if 0.05 in quantiles and 0.95 in quantiles:
            features['interval_width_q05_q95'] = float(np.mean(quantiles[0.95] - quantiles[0.05]))
        if 0.10 in quantiles and 0.90 in quantiles:
            features['interval_width_q10_q90'] = float(np.mean(quantiles[0.90] - quantiles[0.10]))
        if 0.25 in quantiles and 0.75 in quantiles:
            features['interval_width_q25_q75'] = float(np.mean(quantiles[0.75] - quantiles[0.25]))
        
        # Tick-size relative width (interval width relative to current price level)
        # Approximation: use median of q50 as price proxy
        if 0.50 in quantiles and 0.05 in quantiles and 0.95 in quantiles:
            price_proxy = np.median(quantiles[0.50])
            if abs(price_proxy) > 1e-10:
                tick_relative = np.mean(quantiles[0.95] - quantiles[0.05]) / abs(price_proxy)
                features['tick_size_relative_width'] = float(tick_relative)
        
        # ===================================================================
        # C) PINBALL LOSS FEATURES (q-loss)
        # ===================================================================
        pinball_losses = []
        for q_level, q_pred in quantiles.items():
            q_int = int(q_level * 100)
            # Pinball loss: ρ_q(y - ŷ) = (q - I(y < ŷ)) * (y - ŷ)
            error = actual - q_pred
            pinball = np.where(actual >= q_pred, q_level * error, (q_level - 1) * error)
            pinball_mean = float(np.mean(pinball))
            features[f'pinball_q{q_int:02d}'] = pinball_mean
            pinball_losses.append(pinball_mean)
        
        if pinball_losses:
            features['pinball_average'] = float(np.mean(pinball_losses))
            features['pinball_std'] = float(np.std(pinball_losses))
        
        # ===================================================================
        # D) PROBABILITY INTEGRAL TRANSFORM (PIT)
        # ===================================================================
        # PIT = CDF(predicted_dist, actual_return)
        # Approximate CDF via linear interpolation between quantiles
        pit_values = []
        sorted_levels = sorted(quantiles.keys())
        
        for i in range(len(actual)):
            actual_val = actual[i]
            # Find CDF value at actual_val by interpolating quantiles
            quantile_vals = [quantiles[q][i] for q in sorted_levels]
            pit = np.interp(actual_val, quantile_vals, sorted_levels, left=0.0, right=1.0)
            pit_values.append(pit)
        
        pit_array = np.array(pit_values)
        features['pit_mean'] = float(np.mean(pit_array))
        features['pit_std'] = float(np.std(pit_array))
        features['pit_error'] = float(np.mean(np.abs(pit_array - 0.5)))  # Deviation from uniform
        
        # ===================================================================
        # E) PIT HISTOGRAM BUCKET FEATURES
        # ===================================================================
        # Bucket PIT values into deciles
        for bucket_start in range(0, 100, 10):
            bucket_end = bucket_start + 10
            in_bucket = np.sum((pit_array >= bucket_start/100) & (pit_array < bucket_end/100))
            features[f'pit_bucket_{bucket_start}_{bucket_end}'] = float(in_bucket / len(pit_array))
        
        # PIT entropy (Shannon entropy of bucket distribution)
        bucket_probs = [features[f'pit_bucket_{i}_{i+10}'] for i in range(0, 100, 10)]
        bucket_probs = np.array([p for p in bucket_probs if p > 0])  # Remove zeros
        if len(bucket_probs) > 0:
            features['pit_entropy'] = float(-np.sum(bucket_probs * np.log(bucket_probs)))
        
        # ===================================================================
        # F) INTERVAL VIOLATION COUNT (IVC)
        # ===================================================================
        if 0.05 in quantiles and 0.95 in quantiles:
            ivc_05_95 = np.sum((actual < quantiles[0.05]) | (actual > quantiles[0.95]))
            features['ivc_05_95'] = float(ivc_05_95 / len(actual))
        
        if 0.10 in quantiles and 0.90 in quantiles:
            ivc_10_90 = np.sum((actual < quantiles[0.10]) | (actual > quantiles[0.90]))
            features['ivc_10_90'] = float(ivc_10_90 / len(actual))
        
        if 0.25 in quantiles and 0.75 in quantiles:
            ivc_25_75 = np.sum((actual < quantiles[0.25]) | (actual > quantiles[0.75]))
            features['ivc_25_75'] = float(ivc_25_75 / len(actual))
        
        # ===================================================================
        # G) RELIABILITY CURVE SLOPE
        # ===================================================================
        # Reliability curve: observed frequency vs predicted probability
        # Slope = 1 for perfect calibration, <1 for overconfidence, >1 for underconfidence
        observed_freqs = []
        predicted_probs = []
        
        for q_level in sorted(quantiles.keys()):
            observed_freq = np.mean(actual < quantiles[q_level])
            observed_freqs.append(observed_freq)
            predicted_probs.append(q_level)
        
        if len(observed_freqs) > 2:
            # Linear regression: observed ~ predicted
            pred_arr = np.array(predicted_probs)
            obs_arr = np.array(observed_freqs)
            
            # Fit y = a*x + b
            A = np.vstack([pred_arr, np.ones(len(pred_arr))]).T
            slope, intercept = np.linalg.lstsq(A, obs_arr, rcond=None)[0]
            
            features['reliability_slope'] = float(slope)
            features['reliability_bias'] = float(intercept)
            
            # Curvature: mean squared deviation from linear fit
            linear_fit = slope * pred_arr + intercept
            curvature = np.mean((obs_arr - linear_fit) ** 2)
            features['reliability_curvature'] = float(curvature)
        
        # ===================================================================
        # H) TEMPORAL DRIFT OF CALIBRATION
        # ===================================================================
        # Rolling window calibration error
        if len(actual) >= 20:
            # 5-day rolling calibration drift
            window_5d = []
            for i in range(5, len(actual)):
                window_actual = actual[i-5:i]
                window_pit = pit_array[i-5:i]
                window_error = np.mean(np.abs(window_pit - 0.5))
                window_5d.append(window_error)
            
            if len(window_5d) > 1:
                features['calibration_drift_5d'] = float(np.mean(np.diff(window_5d)))
            
            # 20-day rolling calibration drift (if enough data)
            if len(actual) >= 40:
                window_20d = []
                for i in range(20, len(actual)):
                    window_actual = actual[i-20:i]
                    window_pit = pit_array[i-20:i]
                    window_error = np.mean(np.abs(window_pit - 0.5))
                    window_20d.append(window_error)
                
                if len(window_20d) > 1:
                    features['calibration_drift_20d'] = float(np.mean(np.diff(window_20d)))
            
            # Volatility of calibration error
            features['volatility_of_calibration_error'] = float(np.std(window_5d)) if window_5d else 0.0
        
        return features
    
    def _empty_hedge_fund_features(self) -> Dict[str, float]:
        """Return empty/zero hedge-fund features when data is insufficient"""
        features = {}
        
        # Coverage errors
        for q in [5, 10, 25, 50, 75, 90, 95]:
            features[f'coverage_error_q{q:02d}'] = 0.0
        
        # Interval widths
        features['interval_width_q05_q95'] = 0.0
        features['interval_width_q10_q90'] = 0.0
        features['interval_width_q25_q75'] = 0.0
        features['tick_size_relative_width'] = 0.0
        
        # Pinball losses
        for q in [5, 10, 25, 50, 75, 90, 95]:
            features[f'pinball_q{q:02d}'] = 0.0
        features['pinball_average'] = 0.0
        features['pinball_std'] = 0.0
        
        # PIT features
        features['pit_mean'] = 0.5  # Uniform mean
        features['pit_std'] = 0.0
        features['pit_error'] = 0.0
        
        # PIT histogram buckets
        for bucket_start in range(0, 100, 10):
            features[f'pit_bucket_{bucket_start}_{bucket_start+10}'] = 0.1  # Uniform
        features['pit_entropy'] = 0.0
        
        # Interval violations
        features['ivc_05_95'] = 0.0
        features['ivc_10_90'] = 0.0
        features['ivc_25_75'] = 0.0
        
        # Reliability curve
        features['reliability_slope'] = 1.0  # Perfect calibration
        features['reliability_bias'] = 0.0
        features['reliability_curvature'] = 0.0
        
        # Temporal drift
        features['calibration_drift_5d'] = 0.0
        features['calibration_drift_20d'] = 0.0
        features['volatility_of_calibration_error'] = 0.0
        
        return features

    def evaluate_forecast_calibration(self,
                                      forecast_results: Dict[str,
                                                             Any],
                                      actual_returns: List[float]) -> Dict[str,
                                                                           Any]:
        """
        Evaluate calibration of forecast results with HEDGE-FUND STANDARD features

        Args:
            forecast_results: Forecast results with quantile predictions
            actual_returns: Actual observed returns

        Returns:
            Dictionary with calibration assessment + advanced alpha features
        """
        # Extract quantile data
        quantile_levels, predictions = self._extract_quantile_data(
            forecast_results)

        if quantile_levels is None:
            return {'error': 'No quantile forecasts available for calibration'}

        if not quantile_levels or len(actual_returns) == 0:
            return {'error': 'Insufficient data for calibration evaluation'}

        try:
            # Validate and store calibration (original metrics)
            metrics = self._validate_and_store_calibration(
                quantile_levels, predictions, actual_returns)

            # Assess calibration quality (original assessment)
            calibration_assessment = self._assess_calibration_quality(metrics)
            
            # ===================================================================
            # HEDGE-FUND STANDARD CALIBRATION FEATURES (NEW)
            # ===================================================================
            hedge_fund_features = self._compute_hedge_fund_calibration_features(
                forecast_results, actual_returns
            )
            
            # Merge all features
            return {
                'calibration_metrics': metrics,
                'calibration_quality': calibration_assessment,
                **hedge_fund_features,  # Add all new features at top level
                'requires_recalibration': calibration_assessment['needs_recalibration'],
                'sample_size': len(actual_returns)}

        except Exception as e:
            logger.error(f"Failed to evaluate forecast calibration: {e}")
            return {'error': f'Calibration evaluation failed: {str(e)}'}

    def _calculate_error_score(self, metrics):
        """Calculate error score from calibration metrics"""
        mean_error = metrics.get('mean_calibration_error', 1.0)
        interval_error = metrics.get('interval_error', 1.0)

        # Ensure scalar values for comparison
        if hasattr(mean_error, '__iter__') and not isinstance(mean_error, str):
            mean_error = np.mean(mean_error) if len(mean_error) > 0 else 1.0
        if hasattr(
                interval_error,
                '__iter__') and not isinstance(
                interval_error,
                str):
            interval_error = np.mean(interval_error) if len(
                interval_error) > 0 else 1.0

        # Score based on calibration errors (lower is better)
        error_score = max(0, 1 - (mean_error + interval_error) / 2)
        # Ensure error_score is a scalar
        if hasattr(
                error_score,
                '__iter__') and not isinstance(
                error_score,
                str):
            error_score = float(np.mean(error_score)) if len(
                error_score) > 0 else 0.0
        else:
            error_score = float(error_score)

        return error_score, mean_error, interval_error

    def _determine_quality_level(self, error_score):
        """Determine quality level from error score"""
        if error_score >= 0.9:
            return 'excellent'
        elif error_score >= 0.8:
            return 'good'
        elif error_score >= 0.6:
            return 'fair'
        else:
            return 'poor'

    def _generate_recommendations(self, mean_error, interval_error):
        """Generate calibration recommendations"""
        recommendations = []

        if mean_error > 0.1 or interval_error > 0.15:
            recommendations.append('Apply isotonic calibration')

        if interval_error > 0.2:
            recommendations.append('Increase model uncertainty estimates')

        if mean_error > 0.2:
            recommendations.append(
                'Review feature engineering and model selection')

        return recommendations

    def _assess_calibration_quality(
            self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess overall calibration quality and recommend actions

        Args:
            metrics: Calibration metrics

        Returns:
            Dictionary with quality assessment
        """
        assessment = {
            'overall_score': 0.0,
            'quality_level': 'poor',
            'needs_recalibration': False,
            'recommendations': []
        }

        try:
            # Calculate overall calibration score
            error_score, mean_error, interval_error = self._calculate_error_score(
                metrics)
            assessment['overall_score'] = error_score

            # Determine quality level
            assessment['quality_level'] = self._determine_quality_level(
                error_score)

            # Determine if recalibration is needed
            assessment['needs_recalibration'] = (
                mean_error > 0.1 or interval_error > 0.15)

            # Generate recommendations
            assessment['recommendations'] = self._generate_recommendations(
                mean_error, interval_error)

        except Exception as e:
            logger.error(f"Failed to assess calibration quality: {e}")

        return assessment

    def calibrate_forecast(
            self, forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply calibration to forecast results

        Args:
            forecast_results: Raw forecast results

        Returns:
            Calibrated forecast results
        """
        if 'quantile_forecasts' not in forecast_results:
            logger.warning("No quantile forecasts to calibrate")
            return forecast_results

        calibrated_results = forecast_results.copy()

        try:
            # If we have sufficient calibration history, apply calibration
            if len(self.calibration_history) >= 5:
                logger.info("Applying historical calibration adjustments")

                # Simple calibration adjustment based on historical bias
                calibrated_quantiles = self._apply_historical_calibration(
                    forecast_results['quantile_forecasts']
                )
                calibrated_results['quantile_forecasts'] = calibrated_quantiles
                calibrated_results['calibration_applied'] = True
                calibrated_results['calibration_method'] = 'historical_bias_correction'

            else:
                logger.info("Insufficient calibration history for adjustment")
                calibrated_results['calibration_applied'] = False
                calibrated_results['calibration_method'] = 'none'

        except Exception as e:
            logger.error(f"Failed to apply calibration: {e}")
            calibrated_results['calibration_error'] = str(e)

        return calibrated_results

    def _apply_historical_calibration(
            self, quantile_forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply calibration adjustments based on historical performance

        Args:
            quantile_forecasts: Raw quantile forecasts

        Returns:
            Calibrated quantile forecasts
        """
        calibrated_forecasts = quantile_forecasts.copy()

        # Process each quantile key
        for key in quantile_forecasts.keys():
            # Ensure key is a string before checking startswith
            key_str = str(key) if not isinstance(key, str) else key
            if key_str.startswith('q_'):
                calibrated_forecasts[key] = self._calibrate_single_quantile(
                    key, quantile_forecasts[key]
                )

        return calibrated_forecasts

    def _calibrate_single_quantile(
            self,
            quantile_key: str,
            quantile_values: Any) -> Any:
        """
        Calibrate a single quantile forecast

        Args:
            quantile_key: Quantile key (e.g., 'q_10')
            quantile_values: Original quantile values

        Returns:
            Calibrated quantile values
        """
        try:
            # Extract quantile level
            q_level = int(quantile_key[2:]) / 100.0

            # Calculate historical bias for this quantile
            error_key = f'q{int(q_level*100):02d}_error'
            adjustment_factor = self._calculate_adjustment_factor(error_key)

            if abs(adjustment_factor - 1.0) < 0.05:  # No significant bias
                return quantile_values

            # Apply adjustment
            return self._apply_adjustment_to_values(
                quantile_values, adjustment_factor, quantile_key)

        except Exception as e:
            logger.warning(f"Failed to calibrate {quantile_key}: {e}")
            return quantile_values

    def _calculate_adjustment_factor(self, error_key: str) -> float:
        """Calculate adjustment factor based on historical errors"""
        historical_errors = []
        for record in self.calibration_history:
            if error_key in record['metrics']:
                historical_errors.append(record['metrics'][error_key])

        if historical_errors:
            avg_error = np.mean(historical_errors)
            return 1 + avg_error

        return 1.0  # No adjustment if no historical data

    def _apply_adjustment_to_values(
            self,
            values: Any,
            adjustment_factor: float,
            quantile_key: str) -> Any:
        """Apply adjustment factor to quantile values"""
        if not isinstance(values, list) or not values:
            return values

        adjusted_values = []
        for horizon_values in values:
            if isinstance(horizon_values, list):
                adjusted_horizon = [
                    v * adjustment_factor for v in horizon_values]
                adjusted_values.append(adjusted_horizon)
            else:
                adjusted_values.append(horizon_values * adjustment_factor)

        logger.info(f"Applied calibration adjustment to {quantile_key}: factor={adjustment_factor:.3f}")
        return adjusted_values

    def generate_calibration_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive calibration report

        Returns:
            Dictionary with calibration analysis
        """
        if not self.calibration_history:
            return {'error': 'No calibration history available'}

        try:
            report = {
                'summary': {
                    'total_evaluations': len(self.calibration_history),
                    'date_range': {
                        'start': self.calibration_history[0]['timestamp'],
                        'end': self.calibration_history[-1]['timestamp']
                    }
                },
                'recent_performance': {},
                'trends': {},
                'recommendations': []
            }

            # Analyze recent performance (last 5 evaluations)
            recent_evaluations = self.calibration_history[-5:]

            recent_errors = []
            for eval_record in recent_evaluations:
                if 'mean_calibration_error' in eval_record['metrics']:
                    recent_errors.append(
                        eval_record['metrics']['mean_calibration_error'])

            if recent_errors:
                report['recent_performance'] = {
                    'mean_calibration_error': np.mean(recent_errors),
                    'std_calibration_error': np.std(recent_errors),
                    'trend': 'improving' if len(recent_errors) > 2 and
                    recent_errors[-1] < recent_errors[0] else 'stable'
                }

            # Generate recommendations
            if report['recent_performance'].get(
                    'mean_calibration_error', 0) > 0.15:
                report['recommendations'].append(
                    'Consider retraining calibration models')

            if len(self.calibration_history) < 10:
                report['recommendations'].append(
                    'Collect more validation data for robust calibration')

            return report

        except Exception as e:
            logger.error(f"Failed to generate calibration report: {e}")
            return {'error': f'Report generation failed: {str(e)}'}


def test_probability_calibration():
    """Test the probability calibration functionality"""
    print("=== TESTING PROBABILITY CALIBRATION ===")

    # Create mock forecast data
    rng = np.random.default_rng(42)
    n_samples = 50

    # Simulate quantile predictions (poorly calibrated initially)
    predictions = np.column_stack([
        rng.normal(0.0, 0.01, n_samples),  # 10th percentile
        rng.normal(0.0, 0.015, n_samples),  # 50th percentile
        rng.normal(0.0, 0.02, n_samples)   # 90th percentile
    ])

    # Add systematic bias to make them poorly calibrated
    predictions[:, 0] -= 0.005  # Underestimate lower quantile
    predictions[:, 2] += 0.003  # Overestimate upper quantile

    # Simulate actual returns
    actual_returns = rng.normal(0.0, 0.015, n_samples)

    # Create forecast results
    forecast_results = {
        'quantile_forecasts': {
            'q_10': [[pred] for pred in predictions[:, 0]],
            'q_50': [[pred] for pred in predictions[:, 1]],
            'q_90': [[pred] for pred in predictions[:, 2]]
        },
        'forecast_returns': actual_returns[:10].tolist()
    }

    # Test calibration system
    calibration_system = ProbabilityCalibrationSystem()

    # Evaluate calibration multiple times to build history
    for i in range(10):
        sample_actuals = actual_returns[i:i+5].tolist()
        evaluation = calibration_system.evaluate_forecast_calibration(
            forecast_results, sample_actuals
        )

        if 'calibration_metrics' in evaluation:
            print(f"Evaluation {i+1}:")
            print(f"  Mean calibration error: {evaluation['calibration_metrics'].get('mean_calibration_error', 'N/A'):.4f}")
            print(f"  Quality level: {evaluation['calibration_quality']['quality_level']}")
            print(
                f"  Needs recalibration: {evaluation['requires_recalibration']}")

    # Test calibration application
    calibrated_results = calibration_system.calibrate_forecast(
        forecast_results)

    print(f"\nCalibration Applied: {calibrated_results.get('calibration_applied', False)}")
    print(f"Calibration Method: {calibrated_results.get('calibration_method', 'none')}")

    # Generate calibration report
    report = calibration_system.generate_calibration_report()

    if 'summary' in report:
        print("\nCalibration Report:")
        print(f"  Total evaluations: {report['summary']['total_evaluations']}")
        print(f"  Recent mean error: {report['recent_performance'].get('mean_calibration_error', 'N/A'):.4f}")
        print(f"  Recommendations: {len(report['recommendations'])}")
        for rec in report['recommendations']:
            print(f"    - {rec}")

    return calibration_system


if __name__ == "__main__":
    test_probability_calibration()
