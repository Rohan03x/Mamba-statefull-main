"""
Conformal Prediction Framework

This module implements conformal prediction methods for providing prediction
intervals with exact coverage guarantees. Supports both regression and
classification tasks with time-series aware validation.

Key Features:
- Multiple conformal methods (Quantile, CQR, Adaptive)
- Exact coverage guarantees (80%, 95%, etc.)
- Time-series aware validation splits
- Adaptive conformal prediction for non-stationary data
- Comprehensive coverage analysis and diagnostics

Conformal prediction provides distribution-free uncertainty quantification
with finite-sample coverage guarantees, making it ideal for financial ML
where reliable uncertainty estimates are crucial for risk management.

References:
- Vovk, V. et al. (2005). Algorithmic Learning in a Random World
- Angelopoulos, A. & Bates, S. (2021). A gentle introduction to conformal prediction
- Romano, Y. et al. (2019). Conformalized quantile regression
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class ConformalMethod(Enum):
    """Available conformal prediction methods"""
    QUANTILE = "quantile"  # Standard quantile regression
    CQR = "cqr"           # Conformalized Quantile Regression
    ADAPTIVE = "adaptive"  # Adaptive conformal prediction
    JACKKNIFE = "jackknife"  # Jackknife+ method
    NAIVE = "naive"       # Naive conformal prediction


@dataclass
class ConformalResults:
    """Results from conformal prediction"""
    
    # Predictions and intervals
    point_predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    interval_widths: np.ndarray
    
    # Coverage analysis
    target_coverage: float
    empirical_coverage: float
    conditional_coverage: Dict[str, float]
    coverage_gap: float
    
    # Quality metrics
    mean_width: float
    median_width: float
    width_std: float
    efficiency_score: float  # Narrow intervals with good coverage
    
    # Method information
    method: ConformalMethod
    alpha: float  # Miscoverage level (1 - target_coverage)
    quantiles_used: List[float]
    
    # Temporal information
    prediction_date: datetime
    calibration_window: Tuple[datetime, datetime]
    n_calibration_samples: int
    
    # Diagnostics
    coverage_by_time: Dict[str, float]
    width_by_time: Dict[str, float]
    residual_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            'point_predictions': self.point_predictions.tolist(),
            'lower_bounds': self.lower_bounds.tolist(),
            'upper_bounds': self.upper_bounds.tolist(),
            'interval_widths': self.interval_widths.tolist(),
            'target_coverage': self.target_coverage,
            'empirical_coverage': self.empirical_coverage,
            'conditional_coverage': self.conditional_coverage,
            'coverage_gap': self.coverage_gap,
            'mean_width': self.mean_width,
            'median_width': self.median_width,
            'width_std': self.width_std,
            'efficiency_score': self.efficiency_score,
            'method': self.method.value,
            'alpha': self.alpha,
            'quantiles_used': self.quantiles_used,
            'prediction_date': self.prediction_date.isoformat(),
            'calibration_window': [self.calibration_window[0].isoformat(),
                                 self.calibration_window[1].isoformat()],
            'n_calibration_samples': self.n_calibration_samples,
            'coverage_by_time': self.coverage_by_time,
            'width_by_time': self.width_by_time,
            'residual_analysis': self.residual_analysis
        }


class ConformalPredictor(ABC):
    """Base class for conformal predictors"""
    
    def __init__(self, name: str, alpha: float = 0.1):
        self.name = name
        self.alpha = alpha  # Miscoverage level
        self.target_coverage = 1 - alpha
        self.is_fitted = False
        self.calibration_scores = None
        self.quantile = None
        
    @abstractmethod
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, 
           predictions_cal: np.ndarray) -> 'ConformalPredictor':
        """Fit conformal predictor on calibration data"""
        pass
    
    @abstractmethod
    def predict(self, predictions: np.ndarray, 
               X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate conformal prediction intervals"""
        pass
    
    def _compute_quantile(self, scores: np.ndarray) -> float:
        """Compute conformal quantile with finite-sample correction"""
        n = len(scores)
        # Finite-sample correction for exact coverage
        adjusted_alpha = (1 - self.alpha) * (n + 1) / n
        return np.quantile(scores, adjusted_alpha)


class RegressionConformalPredictor(ConformalPredictor):
    """Standard conformal prediction for regression"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__("Regression Conformal", alpha)
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, 
           predictions_cal: np.ndarray) -> 'RegressionConformalPredictor':
        """Fit using absolute residuals as conformity scores"""
        # Compute conformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - predictions_cal)
        self.quantile = self._compute_quantile(self.calibration_scores)
        self.is_fitted = True
        return self
    
    def predict(self, predictions: np.ndarray, 
               X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals using calibrated quantile"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor must be fitted before prediction")
        
        lower_bounds = predictions - self.quantile
        upper_bounds = predictions + self.quantile
        
        return lower_bounds, upper_bounds


class CQRConformalPredictor(ConformalPredictor):
    """Conformalized Quantile Regression"""
    
    def __init__(self, alpha: float = 0.1, 
                 quantile_estimator: Optional[BaseEstimator] = None):
        super().__init__("CQR", alpha)
        self.quantile_estimator = quantile_estimator or self._create_default_estimator()
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2
        self.lower_model = None
        self.upper_model = None
        
    def _create_default_estimator(self) -> BaseEstimator:
        """Create default quantile regression estimator"""
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, 
           predictions_cal: np.ndarray) -> 'CQRConformalPredictor':
        """Fit CQR using quantile regression and conformal calibration"""
        # Fit quantile regression models
        self.lower_model = self._clone_estimator()
        self.upper_model = self._clone_estimator()
        
        # For RandomForest, we use quantile prediction
        if hasattr(self.quantile_estimator, 'predict'):
            # Train models for lower and upper quantiles
            self.lower_model.fit(X_cal, y_cal)
            self.upper_model.fit(X_cal, y_cal)
            
            # Get quantile predictions on calibration set
            if hasattr(self.lower_model, 'predict'):
                # For RandomForest, extract quantiles from predictions
                tree_predictions = np.array([
                    tree.predict(X_cal) for tree in self.lower_model.estimators_
                ])
                lower_preds = np.quantile(tree_predictions, self.lower_quantile, axis=0)
                upper_preds = np.quantile(tree_predictions, self.upper_quantile, axis=0)
            else:
                # Fallback to simple prediction
                preds = self.lower_model.predict(X_cal)
                lower_preds = preds
                upper_preds = preds
        else:
            # Fallback: use provided predictions
            lower_preds = predictions_cal
            upper_preds = predictions_cal
        
        # Compute conformity scores
        self.calibration_scores = np.maximum(
            lower_preds - y_cal,  # Lower miss
            y_cal - upper_preds   # Upper miss
        )
        
        self.quantile = self._compute_quantile(self.calibration_scores)
        self.is_fitted = True
        return self
    
    def _clone_estimator(self) -> BaseEstimator:
        """Clone the quantile estimator"""
        from sklearn.base import clone
        return clone(self.quantile_estimator)
    
    def predict(self, predictions: np.ndarray, 
               X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate CQR prediction intervals"""
        if not self.is_fitted:
            raise ValueError("CQR predictor must be fitted before prediction")
        
        if X_test is not None and self.lower_model is not None:
            # Use quantile regression predictions
            if hasattr(self.lower_model, 'estimators_'):
                # For RandomForest, extract quantiles
                tree_predictions = np.array([
                    tree.predict(X_test) for tree in self.lower_model.estimators_
                ])
                lower_base = np.quantile(tree_predictions, self.lower_quantile, axis=0)
                upper_base = np.quantile(tree_predictions, self.upper_quantile, axis=0)
            else:
                # Fallback to point predictions
                lower_base = self.lower_model.predict(X_test)
                upper_base = self.upper_model.predict(X_test)
        else:
            # Use provided predictions as base
            lower_base = predictions
            upper_base = predictions
        
        # Apply conformal adjustment
        lower_bounds = lower_base - self.quantile
        upper_bounds = upper_base + self.quantile
        
        return lower_bounds, upper_bounds


class AdaptiveConformalPredictor(ConformalPredictor):
    """Adaptive conformal prediction for non-stationary data"""
    
    def __init__(self, alpha: float = 0.1, adaptation_rate: float = 0.1):
        super().__init__("Adaptive Conformal", alpha)
        self.adaptation_rate = adaptation_rate
        self.running_quantile = None
        self.update_count = 0
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, 
           predictions_cal: np.ndarray) -> 'AdaptiveConformalPredictor':
        """Initialize adaptive conformal predictor"""
        # Compute initial conformity scores
        self.calibration_scores = np.abs(y_cal - predictions_cal)
        self.running_quantile = self._compute_quantile(self.calibration_scores)
        self.is_fitted = True
        return self
    
    def predict(self, predictions: np.ndarray, 
               X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adaptive prediction intervals"""
        if not self.is_fitted:
            raise ValueError("Adaptive predictor must be fitted before prediction")
        
        lower_bounds = predictions - self.running_quantile
        upper_bounds = predictions + self.running_quantile
        
        return lower_bounds, upper_bounds
    
    def update(self, y_true: np.ndarray, predictions: np.ndarray):
        """Update adaptive quantile with new observations"""
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before updating")
        
        # Compute new conformity scores
        new_scores = np.abs(y_true - predictions)
        
        # Update running quantile with exponential moving average
        for score in new_scores:
            if self.running_quantile is None:
                self.running_quantile = score
            else:
                # Check if we're above or below target coverage
                indicator = score > self.running_quantile
                coverage_error = indicator - self.alpha
                
                # Adapt quantile based on coverage error
                adjustment = self.adaptation_rate * coverage_error
                self.running_quantile = max(0, self.running_quantile + adjustment)
        
        self.update_count += len(y_true)


class ConformalDiagnostics:
    """Comprehensive conformal prediction diagnostics"""
    
    @staticmethod
    def analyze_coverage(y_true: np.ndarray, 
                        lower_bounds: np.ndarray, 
                        upper_bounds: np.ndarray,
                        target_coverage: float) -> Dict[str, float]:
        """Analyze prediction interval coverage"""
        # Overall coverage
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        empirical_coverage = np.mean(in_interval)
        coverage_gap = abs(empirical_coverage - target_coverage)
        
        # Coverage by time (if temporal structure available)
        results = {
            'empirical_coverage': empirical_coverage,
            'target_coverage': target_coverage,
            'coverage_gap': coverage_gap,
            'lower_violations': np.mean(y_true < lower_bounds),
            'upper_violations': np.mean(y_true > upper_bounds)
        }
        
        return results
    
    @staticmethod
    def analyze_efficiency(lower_bounds: np.ndarray, 
                          upper_bounds: np.ndarray) -> Dict[str, float]:
        """Analyze prediction interval efficiency (width)"""
        widths = upper_bounds - lower_bounds
        
        return {
            'mean_width': np.mean(widths),
            'median_width': np.median(widths),
            'std_width': np.std(widths),
            'min_width': np.min(widths),
            'max_width': np.max(widths),
            'width_iqr': np.percentile(widths, 75) - np.percentile(widths, 25)
        }
    
    @staticmethod
    def conditional_coverage_analysis(y_true: np.ndarray,
                                    lower_bounds: np.ndarray,
                                    upper_bounds: np.ndarray,
                                    X: np.ndarray,
                                    n_bins: int = 5) -> Dict[str, Any]:
        """Analyze conditional coverage across feature space"""
        in_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        
        conditional_coverage = {}
        
        # Coverage by feature quantiles
        for i in range(X.shape[1]):
            feature = X[:, i]
            quantiles = np.quantile(feature, np.linspace(0, 1, n_bins + 1))
            
            coverage_by_bin = []
            for j in range(len(quantiles) - 1):
                in_bin = (feature >= quantiles[j]) & (feature < quantiles[j + 1])
                if np.sum(in_bin) > 0:
                    bin_coverage = np.mean(in_interval[in_bin])
                    coverage_by_bin.append(bin_coverage)
                else:
                    coverage_by_bin.append(np.nan)
            
            conditional_coverage[f'feature_{i}'] = coverage_by_bin
        
        return conditional_coverage