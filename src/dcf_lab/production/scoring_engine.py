"""
Production Scoring Engine with Calibration and Conformal Intervals

This module implements comprehensive model scoring for the production trading system:
- Real-time prediction scoring with multi-horizon support
- Probability calibration using Platt scaling and isotonic regression
- Conformal prediction intervals with adaptive coverage
- Quantile predictions with proper scoring rules
- Performance monitoring and scoring quality assessment

Key features:
- Multi-model ensemble scoring with uncertainty quantification
- Adaptive conformal prediction for time-series data
- Real-time calibration monitoring and adjustment
- Confidence intervals with guaranteed coverage properties
- Integration with monitoring system for scoring validation
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field

# ML imports
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

@dataclass
class ScoringConfig:
    """Configuration for production scoring system"""
    
    # Prediction horizons
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])  # Days
    
    # Calibration parameters
    calibration_method: str = "isotonic"  # "platt", "isotonic"
    calibration_window: int = 1000  # Samples for calibration fitting
    recalibration_frequency: int = 100  # Recalibrate every N predictions
    
    # Conformal prediction parameters
    conformal_alpha: float = 0.1  # 1-alpha = coverage probability (90%)
    conformal_window: int = 500  # Window for conformity scores
    adaptive_conformal: bool = True  # Use adaptive conformal prediction
    
    # Quantile prediction parameters
    quantile_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    
    # Scoring validation
    min_coverage_threshold: float = 0.85  # Minimum coverage for intervals
    max_interval_width_ratio: float = 2.0  # Maximum width relative to prediction
    
    # Performance monitoring
    scoring_window: int = 200  # Window for scoring performance tracking
    alert_on_miscalibration: bool = True

@dataclass
class PredictionResult:
    """Complete prediction result with uncertainty quantification"""
    
    timestamp: str
    horizon: int  # Prediction horizon in days
    point_prediction: float
    prediction_std: float
    
    # Calibrated probabilities (for classification/direction)
    prob_positive: Optional[float] = None
    prob_negative: Optional[float] = None
    calibrated: bool = False
    
    # Quantile predictions
    quantiles: Dict[float, float] = field(default_factory=dict)
    
    # Conformal intervals
    conformal_lower: Optional[float] = None
    conformal_upper: Optional[float] = None
    conformal_coverage: float = 0.9
    
    # Model information
    model_version: str = ""
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    
    # Uncertainty measures
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    uncertainty_score: float = 0.0
    
    # Metadata
    features_used: List[str] = field(default_factory=list)
    scoring_time_ms: float = 0.0

class CalibrationEngine:
    """Handles probability calibration for predictions"""
    
    def __init__(self, method: str = "isotonic", window_size: int = 1000):
        self.method = method
        self.window_size = window_size
        
        # Calibration models per horizon
        self.calibrators = {}
        
        # Calibration data storage
        self.calibration_data = {}
        
        # Performance tracking
        self.calibration_scores = {}
        
    def fit_calibrator(self, horizon: int, predictions: np.ndarray, 
                      true_outcomes: np.ndarray) -> bool:
        """
        Fit calibration model for a specific horizon
        
        Args:
            horizon: Prediction horizon
            predictions: Raw model predictions
            true_outcomes: Binary outcomes (1 for positive move, 0 for negative)
            
        Returns:
            Success status
        """
        
        try:
            if len(predictions) < 50:  # Need minimum data
                logger.warning(f"Insufficient data for calibration at horizon {horizon}")
                return False
            
            # Convert predictions to probabilities if needed
            if np.min(predictions) < 0 or np.max(predictions) > 1:
                # Assume raw predictions, convert using sigmoid
                probs = 1 / (1 + np.exp(-predictions))
            else:
                probs = predictions
            
            # Fit calibration model
            if self.method == "platt":
                calibrator = LogisticRegression()
                calibrator.fit(probs.reshape(-1, 1), true_outcomes)
            elif self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probs, true_outcomes)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            self.calibrators[horizon] = calibrator
            
            # Store calibration data
            self.calibration_data[horizon] = {
                'predictions': probs[-self.window_size:],
                'outcomes': true_outcomes[-self.window_size:],
                'fitted_time': datetime.now().isoformat()
            }
            
            # Calculate calibration score
            calibrated_probs = self.predict_calibrated(horizon, probs)
            brier_score = brier_score_loss(true_outcomes, calibrated_probs)
            self.calibration_scores[horizon] = brier_score
            
            logger.info(f"Fitted calibrator for horizon {horizon}, Brier score: {brier_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Calibration fitting failed for horizon {horizon}: {e}")
            return False
    
    def predict_calibrated(self, horizon: int, predictions: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities"""
        
        if horizon not in self.calibrators:
            logger.warning(f"No calibrator available for horizon {horizon}")
            return predictions  # Return uncalibrated
        
        calibrator = self.calibrators[horizon]
        
        try:
            if self.method == "platt":
                return calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
            elif self.method == "isotonic":
                return calibrator.predict(predictions)
        except Exception as e:
            logger.error(f"Calibration prediction failed: {e}")
            return predictions
    
    def update_calibration(self, horizon: int, new_predictions: np.ndarray, 
                          new_outcomes: np.ndarray):
        """Update calibration with new data"""
        
        if horizon not in self.calibration_data:
            self.calibration_data[horizon] = {
                'predictions': [],
                'outcomes': [],
                'fitted_time': datetime.now().isoformat()
            }
        
        # Add new data
        data = self.calibration_data[horizon]
        data['predictions'] = np.concatenate([data['predictions'], new_predictions])[-self.window_size:]
        data['outcomes'] = np.concatenate([data['outcomes'], new_outcomes])[-self.window_size:]
        
        # Refit if enough new data
        if len(new_predictions) >= 20:  # Refit threshold
            self.fit_calibrator(horizon, data['predictions'], data['outcomes'])
    
    def get_calibration_quality(self, horizon: int) -> Dict[str, float]:
        """Assess calibration quality"""
        
        if horizon not in self.calibration_data or horizon not in self.calibrators:
            return {'calibration_available': False}
        
        data = self.calibration_data[horizon]
        predictions = data['predictions']
        outcomes = data['outcomes']
        
        if len(predictions) < 20:
            return {'calibration_available': False}
        
        # Get calibrated probabilities
        calibrated_probs = self.predict_calibrated(horizon, predictions)
        
        # Calculate reliability (calibration curve)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = outcomes[in_bin].mean()
                avg_confidence_in_bin = calibrated_probs[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        if bin_accuracies:
            for i, (acc, conf) in enumerate(zip(bin_accuracies, bin_confidences)):
                bin_weight = ((calibrated_probs > bin_boundaries[i]) & 
                            (calibrated_probs <= bin_boundaries[i+1])).mean()
                ece += bin_weight * abs(acc - conf)
        
        return {
            'calibration_available': True,
            'brier_score': brier_score_loss(outcomes, calibrated_probs),
            'expected_calibration_error': ece,
            'num_samples': len(predictions),
            'mean_predicted_prob': calibrated_probs.mean(),
            'mean_actual_rate': outcomes.mean()
        }

class ConformalPredictor:
    """Implements conformal prediction for uncertainty quantification"""
    
    def __init__(self, alpha: float = 0.1, window_size: int = 500, adaptive: bool = True):
        self.alpha = alpha  # 1-alpha = coverage probability
        self.window_size = window_size
        self.adaptive = adaptive
        
        # Store conformity scores per horizon
        self.conformity_scores = {}
        
        # Adaptive parameters
        self.coverage_errors = {}
        self.adaptive_alpha = {}
        
    def compute_conformity_score(self, prediction: float, actual: float) -> float:
        """Compute conformity score (absolute residual)"""
        return abs(prediction - actual)
    
    def fit_conformal_model(self, horizon: int, predictions: np.ndarray, 
                           actuals: np.ndarray) -> bool:
        """
        Fit conformal prediction model
        
        Args:
            horizon: Prediction horizon
            predictions: Model predictions
            actuals: Actual outcomes
            
        Returns:
            Success status
        """
        
        try:
            if len(predictions) < 50:
                logger.warning(f"Insufficient data for conformal prediction at horizon {horizon}")
                return False
            
            # Compute conformity scores
            scores = np.array([
                self.compute_conformity_score(pred, actual)
                for pred, actual in zip(predictions, actuals)
            ])
            
            # Store recent scores
            self.conformity_scores[horizon] = scores[-self.window_size:]
            
            # Initialize adaptive parameters
            if self.adaptive:
                self.coverage_errors[horizon] = []
                self.adaptive_alpha[horizon] = self.alpha
            
            logger.info(f"Fitted conformal model for horizon {horizon}")
            return True
            
        except Exception as e:
            logger.error(f"Conformal fitting failed for horizon {horizon}: {e}")
            return False
    
    def predict_interval(self, horizon: int, prediction: float) -> Tuple[float, float]:
        """
        Predict conformal interval
        
        Args:
            horizon: Prediction horizon
            prediction: Point prediction
            
        Returns:
            (lower_bound, upper_bound)
        """
        
        if horizon not in self.conformity_scores:
            logger.warning(f"No conformal model for horizon {horizon}")
            # Return naive interval
            std_estimate = abs(prediction) * 0.1  # 10% of prediction
            return (prediction - 2*std_estimate, prediction + 2*std_estimate)
        
        scores = self.conformity_scores[horizon]
        current_alpha = self._get_current_alpha(horizon)
        
        # Calculate quantile of conformity scores
        quantile_level = 1.0 - current_alpha
        if len(scores) == 0:
            threshold = 0.0
        else:
            # Use (n+1)/n * quantile for finite sample correction
            corrected_level = (len(scores) + 1) / len(scores) * quantile_level
            corrected_level = min(corrected_level, 1.0)
            threshold = np.quantile(scores, corrected_level)
        
        # Construct interval
        lower_bound = prediction - threshold
        upper_bound = prediction + threshold
        
        return (lower_bound, upper_bound)
    
    def update_conformal_model(self, horizon: int, new_prediction: float, new_actual: float):
        """Update conformal model with new observation"""
        
        if horizon not in self.conformity_scores:
            self.conformity_scores[horizon] = []
        
        # Compute new conformity score
        new_score = self.compute_conformity_score(new_prediction, new_actual)
        
        # Add to scores (with window limit)
        scores = self.conformity_scores[horizon]
        scores.append(new_score)
        self.conformity_scores[horizon] = scores[-self.window_size:]
        
        # Update adaptive alpha if enabled
        if self.adaptive:
            self._update_adaptive_alpha(horizon, new_prediction, new_actual)
    
    def _get_current_alpha(self, horizon: int) -> float:
        """Get current alpha (potentially adapted)"""
        
        if self.adaptive and horizon in self.adaptive_alpha:
            return self.adaptive_alpha[horizon]
        return self.alpha
    
    def _update_adaptive_alpha(self, horizon: int, prediction: float, actual: float):
        """Update adaptive alpha based on coverage performance"""
        
        if horizon not in self.coverage_errors:
            self.coverage_errors[horizon] = []
            self.adaptive_alpha[horizon] = self.alpha
        
        # Check if previous prediction interval covered actual
        if len(self.conformity_scores[horizon]) > 1:
            # Get interval using previous conformity scores
            prev_scores = self.conformity_scores[horizon][:-1]
            if len(prev_scores) > 0:
                prev_threshold = np.quantile(prev_scores, 1.0 - self.adaptive_alpha[horizon])
                prev_lower = prediction - prev_threshold
                prev_upper = prediction + prev_threshold
                
                # Coverage error: 1 if covered, 0 if not
                coverage = 1.0 if prev_lower <= actual <= prev_upper else 0.0
                target_coverage = 1.0 - self.alpha
                
                # Error: positive if under-covering, negative if over-covering
                error = target_coverage - coverage
                self.coverage_errors[horizon].append(error)
                
                # Keep recent errors only
                self.coverage_errors[horizon] = self.coverage_errors[horizon][-100:]
                
                # Adapt alpha based on average coverage error
                if len(self.coverage_errors[horizon]) >= 10:
                    avg_error = np.mean(self.coverage_errors[horizon][-20:])
                    
                    # Adjust alpha: if under-covering (positive error), decrease alpha (wider intervals)
                    # If over-covering (negative error), increase alpha (narrower intervals)
                    learning_rate = 0.01
                    adjustment = -learning_rate * avg_error
                    
                    new_alpha = self.adaptive_alpha[horizon] + adjustment
                    new_alpha = np.clip(new_alpha, 0.01, 0.5)  # Keep reasonable bounds
                    
                    self.adaptive_alpha[horizon] = new_alpha
    
    def get_coverage_statistics(self, horizon: int) -> Dict[str, float]:
        """Get coverage statistics for quality assessment"""
        
        if horizon not in self.coverage_errors or not self.coverage_errors[horizon]:
            return {'coverage_available': False}
        
        errors = self.coverage_errors[horizon]
        target_coverage = 1.0 - self.alpha
        actual_coverage = target_coverage - np.mean(errors)
        
        return {
            'coverage_available': True,
            'target_coverage': target_coverage,
            'actual_coverage': actual_coverage,
            'coverage_error': np.mean(errors),
            'coverage_std': np.std(errors),
            'num_observations': len(errors),
            'current_alpha': self._get_current_alpha(horizon)
        }

class QuantilePredictor:
    """Generates quantile predictions with proper scoring"""
    
    def __init__(self, quantile_levels: List[float]):
        self.quantile_levels = sorted(quantile_levels)
        self.quantile_models = {}  # Per horizon quantile models
        
    def fit_quantile_models(self, horizon: int, features: pd.DataFrame, 
                           targets: np.ndarray) -> bool:
        """Fit quantile regression models for each quantile level"""
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            models = {}
            
            for quantile in self.quantile_levels:
                # Use Gradient Boosting with quantile loss
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=quantile,
                    n_estimators=100,
                    max_depth=3,
                    random_state=42
                )
                
                model.fit(features, targets)
                models[quantile] = model
            
            self.quantile_models[horizon] = models
            
            logger.info(f"Fitted quantile models for horizon {horizon}")
            return True
            
        except Exception as e:
            logger.error(f"Quantile model fitting failed for horizon {horizon}: {e}")
            return False
    
    def predict_quantiles(self, horizon: int, features: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles for given features"""
        
        if horizon not in self.quantile_models:
            logger.warning(f"No quantile models for horizon {horizon}")
            return {}
        
        models = self.quantile_models[horizon]
        predictions = {}
        
        try:
            for quantile, model in models.items():
                pred = model.predict(features)
                predictions[quantile] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantile prediction failed: {e}")
            return {}
    
    def calculate_quantile_score(self, quantile: float, predictions: np.ndarray, 
                                actuals: np.ndarray) -> float:
        """Calculate quantile score (pinball loss)"""
        
        errors = actuals - predictions
        loss = np.maximum((quantile - 1) * errors, quantile * errors)
        return np.mean(loss)
    
    def validate_quantile_predictions(self, horizon: int, features: pd.DataFrame, 
                                    actuals: np.ndarray) -> Dict[str, float]:
        """Validate quantile prediction quality"""
        
        if horizon not in self.quantile_models:
            return {'quantile_validation_available': False}
        
        # Get quantile predictions
        quantile_preds = self.predict_quantiles(horizon, features)
        
        if not quantile_preds:
            return {'quantile_validation_available': False}
        
        # Calculate scores for each quantile
        quantile_scores = {}
        coverage_stats = {}
        
        for quantile in self.quantile_levels:
            if quantile in quantile_preds:
                preds = quantile_preds[quantile]
                
                # Quantile score (pinball loss)
                score = self.calculate_quantile_score(quantile, preds, actuals)
                quantile_scores[f'quantile_{quantile}_score'] = score
                
                # Coverage statistics
                coverage = np.mean(actuals <= preds)
                coverage_stats[f'quantile_{quantile}_coverage'] = coverage
        
        # Interval coverage (between quantiles)
        if 0.05 in quantile_preds and 0.95 in quantile_preds:
            lower = quantile_preds[0.05]
            upper = quantile_preds[0.95]
            interval_coverage = np.mean((actuals >= lower) & (actuals <= upper))
            coverage_stats['interval_90_coverage'] = interval_coverage
        
        return {
            'quantile_validation_available': True,
            **quantile_scores,
            **coverage_stats,
            'num_samples': len(actuals)
        }

class ProductionScoringEngine:
    """Main scoring engine for production predictions"""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        
        # Initialize components
        self.calibration_engine = CalibrationEngine(
            method=self.config.calibration_method,
            window_size=self.config.calibration_window
        )
        
        self.conformal_predictor = ConformalPredictor(
            alpha=self.config.conformal_alpha,
            window_size=self.config.conformal_window,
            adaptive=self.config.adaptive_conformal
        )
        
        self.quantile_predictor = QuantilePredictor(self.config.quantile_levels)
        
        # Scoring history
        self.scoring_history = []
        self.performance_metrics = {}
        
        # Model registry
        self.active_models = {}  # horizon -> model
        
    def register_model(self, horizon: int, model: BaseEstimator, version: str):
        """Register a model for specific horizon"""
        
        self.active_models[horizon] = {
            'model': model,
            'version': version,
            'registered_time': datetime.now().isoformat()
        }
        
        logger.info(f"Registered model {version} for horizon {horizon}")
    
    def score_prediction(self, horizon: int, features: pd.DataFrame, 
                        timestamp: Optional[str] = None) -> PredictionResult:
        """
        Generate comprehensive prediction with uncertainty quantification
        
        Args:
            horizon: Prediction horizon in days
            features: Input features for prediction
            timestamp: Prediction timestamp (defaults to now)
            
        Returns:
            Complete prediction result
        """
        
        start_time = datetime.now()
        timestamp = timestamp or start_time.isoformat()
        
        try:
            # Get base model prediction
            if horizon not in self.active_models:
                raise ValueError(f"No model registered for horizon {horizon}")
            
            model_info = self.active_models[horizon]
            model = model_info['model']
            
            # Generate point prediction
            point_pred = model.predict(features)[0]
            
            # Estimate prediction uncertainty
            pred_std = self._estimate_prediction_uncertainty(model, features, horizon)
            
            # Get calibrated probabilities (for direction)
            prob_positive, prob_negative = self._get_calibrated_probabilities(
                horizon, point_pred, features
            )
            
            # Get quantile predictions
            quantiles = self._get_quantile_predictions(horizon, features)
            
            # Get conformal intervals
            conformal_lower, conformal_upper = self.conformal_predictor.predict_interval(
                horizon, point_pred
            )
            
            # Calculate prediction intervals (parametric)
            pred_lower = point_pred - 1.96 * pred_std
            pred_upper = point_pred + 1.96 * pred_std
            
            # Uncertainty score (combination of different uncertainty measures)
            uncertainty_score = self._calculate_uncertainty_score(
                pred_std, conformal_upper - conformal_lower, quantiles
            )
            
            # Create prediction result
            result = PredictionResult(
                timestamp=timestamp,
                horizon=horizon,
                point_prediction=point_pred,
                prediction_std=pred_std,
                prob_positive=prob_positive,
                prob_negative=prob_negative,
                calibrated=horizon in self.calibration_engine.calibrators,
                quantiles=quantiles,
                conformal_lower=conformal_lower,
                conformal_upper=conformal_upper,
                conformal_coverage=1.0 - self.conformal_predictor._get_current_alpha(horizon),
                model_version=model_info['version'],
                prediction_interval=(pred_lower, pred_upper),
                uncertainty_score=uncertainty_score,
                features_used=list(features.columns),
                scoring_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            # Store for monitoring
            self.scoring_history.append(result)
            
            # Limit history size
            if len(self.scoring_history) > self.config.scoring_window * len(self.config.prediction_horizons):
                self.scoring_history = self.scoring_history[-self.config.scoring_window:]
            
            return result
            
        except Exception as e:
            logger.error(f"Scoring failed for horizon {horizon}: {e}")
            
            # Return minimal result
            return PredictionResult(
                timestamp=timestamp,
                horizon=horizon,
                point_prediction=0.0,
                prediction_std=1.0,
                scoring_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def update_with_actuals(self, predictions: List[PredictionResult], 
                           actuals: List[float]) -> Dict[str, Any]:
        """
        Update models with actual outcomes
        
        Args:
            predictions: List of previous predictions
            actuals: Corresponding actual outcomes
            
        Returns:
            Update summary
        """
        
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        updates = {}
        
        # Group by horizon
        horizon_data = {}
        for pred, actual in zip(predictions, actuals):
            horizon = pred.horizon
            if horizon not in horizon_data:
                horizon_data[horizon] = {'predictions': [], 'actuals': []}
            
            horizon_data[horizon]['predictions'].append(pred)
            horizon_data[horizon]['actuals'].append(actual)
        
        # Update each horizon
        for horizon, data in horizon_data.items():
            preds = data['predictions']
            acts = data['actuals']
            
            # Update conformal predictor
            for pred, actual in zip(preds, acts):
                self.conformal_predictor.update_conformal_model(
                    horizon, pred.point_prediction, actual
                )
            
            # Update calibration (for binary outcomes)
            if len(acts) >= 10:  # Need minimum data
                binary_outcomes = [1.0 if a > 0 else 0.0 for a in acts]
                point_preds = [p.point_prediction for p in preds]
                
                self.calibration_engine.update_calibration(
                    horizon, np.array(point_preds), np.array(binary_outcomes)
                )
            
            # Calculate performance metrics
            point_preds = np.array([p.point_prediction for p in preds])
            actual_vals = np.array(acts)
            
            rmse = np.sqrt(np.mean((point_preds - actual_vals) ** 2))
            mae = np.mean(np.abs(point_preds - actual_vals))
            
            # Interval coverage
            coverage_90 = 0.0
            coverage_conformal = 0.0
            
            for pred, actual in zip(preds, acts):
                # Parametric interval coverage
                if pred.prediction_interval[0] <= actual <= pred.prediction_interval[1]:
                    coverage_90 += 1.0
                
                # Conformal interval coverage
                if (pred.conformal_lower is not None and pred.conformal_upper is not None and
                    pred.conformal_lower <= actual <= pred.conformal_upper):
                    coverage_conformal += 1.0
            
            coverage_90 /= len(preds)
            coverage_conformal /= len(preds)
            
            horizon_metrics = {
                'rmse': rmse,
                'mae': mae,
                'coverage_90': coverage_90,
                'coverage_conformal': coverage_conformal,
                'num_samples': len(preds)
            }
            
            updates[f'horizon_{horizon}'] = horizon_metrics
            
            # Store in performance metrics
            if horizon not in self.performance_metrics:
                self.performance_metrics[horizon] = []
            
            self.performance_metrics[horizon].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': horizon_metrics
            })
        
        logger.info(f"Updated scoring models with {len(actuals)} observations")
        
        return {
            'update_timestamp': datetime.now().isoformat(),
            'total_samples': len(actuals),
            'horizons_updated': list(horizon_data.keys()),
            'metrics_by_horizon': updates
        }
    
    def _estimate_prediction_uncertainty(self, model: BaseEstimator, 
                                       features: pd.DataFrame, horizon: int) -> float:
        """Estimate prediction uncertainty"""
        
        try:
            # For ensemble models, use prediction variance
            if hasattr(model, 'estimators_'):
                # Get predictions from individual estimators
                individual_preds = []
                for estimator in model.estimators_:
                    pred = estimator.predict(features)[0]
                    individual_preds.append(pred)
                
                return np.std(individual_preds)
            
            # For models with prediction intervals
            elif hasattr(model, 'predict_proba'):
                # Use entropy of probability distribution
                probs = model.predict_proba(features)[0]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                return entropy  # Higher entropy = higher uncertainty
            
            # Default: use historical residual standard deviation
            else:
                if horizon in self.performance_metrics and self.performance_metrics[horizon]:
                    recent_metrics = self.performance_metrics[horizon][-10:]  # Last 10 observations
                    avg_rmse = np.mean([m['metrics']['rmse'] for m in recent_metrics])
                    return avg_rmse
                else:
                    return 1.0  # Default uncertainty
                    
        except Exception:
            return 1.0  # Default uncertainty
    
    def _get_calibrated_probabilities(self, horizon: int, prediction: float, 
                                    features: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Get calibrated probabilities for direction"""
        
        try:
            # Convert prediction to probability of positive move
            prob_raw = 1.0 / (1.0 + np.exp(-prediction))  # Sigmoid
            
            # Apply calibration if available
            prob_calibrated = self.calibration_engine.predict_calibrated(
                horizon, np.array([prob_raw])
            )[0]
            
            return prob_calibrated, 1.0 - prob_calibrated
            
        except Exception:
            return None, None
    
    def _get_quantile_predictions(self, horizon: int, features: pd.DataFrame) -> Dict[float, float]:
        """Get quantile predictions"""
        
        try:
            quantile_preds = self.quantile_predictor.predict_quantiles(horizon, features)
            
            # Convert to single values (assuming single sample)
            quantiles = {}
            for quantile, preds in quantile_preds.items():
                if len(preds) > 0:
                    quantiles[quantile] = preds[0]
            
            return quantiles
            
        except Exception:
            return {}
    
    def _calculate_uncertainty_score(self, pred_std: float, conformal_width: float, 
                                   quantiles: Dict[float, float]) -> float:
        """Calculate composite uncertainty score"""
        
        uncertainty_components = []
        
        # Standard deviation component
        uncertainty_components.append(pred_std)
        
        # Conformal interval width component
        uncertainty_components.append(conformal_width / 4.0)  # Normalize roughly to std scale
        
        # Quantile spread component
        if 0.05 in quantiles and 0.95 in quantiles:
            quantile_width = quantiles[0.95] - quantiles[0.05]
            uncertainty_components.append(quantile_width / 3.29)  # Normalize to std scale
        
        # Return average uncertainty
        if uncertainty_components:
            return np.mean(uncertainty_components)
        else:
            return 1.0
    
    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive scoring system summary"""
        
        summary = {
            'active_models': {
                horizon: info['version'] 
                for horizon, info in self.active_models.items()
            },
            'scoring_history_length': len(self.scoring_history),
            'horizons_configured': self.config.prediction_horizons,
            'calibration_status': {},
            'conformal_status': {},
            'recent_performance': {}
        }
        
        # Calibration status per horizon
        for horizon in self.config.prediction_horizons:
            cal_quality = self.calibration_engine.get_calibration_quality(horizon)
            summary['calibration_status'][f'horizon_{horizon}'] = cal_quality
        
        # Conformal status per horizon
        for horizon in self.config.prediction_horizons:
            conf_stats = self.conformal_predictor.get_coverage_statistics(horizon)
            summary['conformal_status'][f'horizon_{horizon}'] = conf_stats
        
        # Recent performance
        for horizon in self.config.prediction_horizons:
            if horizon in self.performance_metrics and self.performance_metrics[horizon]:
                recent = self.performance_metrics[horizon][-5:]  # Last 5 observations
                avg_metrics = {}
                
                for metric_name in ['rmse', 'mae', 'coverage_90', 'coverage_conformal']:
                    values = [obs['metrics'].get(metric_name, 0) for obs in recent]
                    avg_metrics[metric_name] = np.mean(values) if values else 0.0
                
                summary['recent_performance'][f'horizon_{horizon}'] = avg_metrics
        
        return summary
    
    def validate_scoring_quality(self) -> Dict[str, Any]:
        """Validate overall scoring system quality"""
        
        validation_results = {
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check calibration quality
        for horizon in self.config.prediction_horizons:
            cal_quality = self.calibration_engine.get_calibration_quality(horizon)
            
            if cal_quality.get('calibration_available', False):
                ece = cal_quality.get('expected_calibration_error', 1.0)
                if ece > 0.1:  # Poor calibration
                    validation_results['issues'].append(f"Poor calibration at horizon {horizon}: ECE={ece:.3f}")
                    validation_results['recommendations'].append(f"Recalibrate model for horizon {horizon}")
        
        # Check conformal coverage
        for horizon in self.config.prediction_horizons:
            conf_stats = self.conformal_predictor.get_coverage_statistics(horizon)
            
            if conf_stats.get('coverage_available', False):
                actual_coverage = conf_stats.get('actual_coverage', 0.0)
                target_coverage = conf_stats.get('target_coverage', 0.9)
                
                coverage_error = abs(actual_coverage - target_coverage)
                if coverage_error > 0.05:  # 5% deviation
                    validation_results['issues'].append(f"Coverage deviation at horizon {horizon}: {coverage_error:.3f}")
        
        # Check recent performance
        for horizon in self.config.prediction_horizons:
            if horizon in self.performance_metrics and self.performance_metrics[horizon]:
                recent = self.performance_metrics[horizon][-10:]
                
                if len(recent) >= 5:
                    coverage_90_values = [obs['metrics'].get('coverage_90', 0) for obs in recent]
                    avg_coverage = np.mean(coverage_90_values)
                    
                    if avg_coverage < self.config.min_coverage_threshold:
                        validation_results['issues'].append(f"Low interval coverage at horizon {horizon}: {avg_coverage:.3f}")
        
        # Set overall status
        if validation_results['issues']:
            validation_results['overall_status'] = 'degraded' if len(validation_results['issues']) < 3 else 'critical'
        
        return validation_results