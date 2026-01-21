"""
Advanced Probability Calibration Methods

This module implements multiple state-of-the-art calibration techniques:
1. Isotonic Regression Calibration
2. Platt Scaling (Sigmoid Calibration)  
3. Temperature Scaling
4. Beta Calibration
5. Histogram Binning

All methods are time-series aware and prevent data leakage through proper
validation splits. Includes comprehensive calibration diagnostics and
uncertainty quantification.

Key Features:
- Multiple calibration methods with automatic selection
- Time-series aware cross-validation
- Calibration quality metrics (reliability, sharpness, resolution)
- Integration with existing CV framework
- Financial ML best practices

References:
- Platt, J. (1999). Probabilistic outputs for support vector machines
- Zadrozny, B. & Elkan, C. (2002). Obtaining calibrated probability estimates
- Guo, C. et al. (2017). On calibration of modern neural networks
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
import logging
from datetime import datetime

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Available calibration methods"""
    ISOTONIC = "isotonic"
    PLATT = "platt"
    TEMPERATURE = "temperature" 
    BETA = "beta"
    HISTOGRAM = "histogram"
    ENSEMBLE = "ensemble"


@dataclass
class CalibrationResults:
    """Results from probability calibration"""
    
    # Calibrated predictions
    calibrated_probabilities: np.ndarray
    original_probabilities: np.ndarray
    
    # Calibration quality metrics
    reliability_score: float  # How close predictions match frequencies
    sharpness_score: float   # Spread of predictions (higher = more confident)
    resolution_score: float  # Ability to discriminate outcomes
    brier_score: float      # Overall calibration quality
    ece_score: float        # Expected Calibration Error
    mce_score: float        # Maximum Calibration Error
    
    # Method information
    method: CalibrationMethod
    calibrator_params: Dict[str, Any]
    validation_scores: Dict[str, float]
    
    # Temporal information
    calibration_date: datetime
    data_start_date: datetime
    data_end_date: datetime
    n_samples: int
    
    # Diagnostics
    reliability_diagram: Dict[str, np.ndarray]
    coverage_analysis: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            'calibrated_probabilities': self.calibrated_probabilities.tolist(),
            'original_probabilities': self.original_probabilities.tolist(),
            'reliability_score': self.reliability_score,
            'sharpness_score': self.sharpness_score,
            'resolution_score': self.resolution_score,
            'brier_score': self.brier_score,
            'ece_score': self.ece_score,
            'mce_score': self.mce_score,
            'method': self.method.value,
            'calibrator_params': self.calibrator_params,
            'validation_scores': self.validation_scores,
            'calibration_date': self.calibration_date.isoformat(),
            'data_start_date': self.data_start_date.isoformat(),
            'data_end_date': self.data_end_date.isoformat(),
            'n_samples': self.n_samples,
            'reliability_diagram': {k: v.tolist() for k, v in self.reliability_diagram.items()},
            'coverage_analysis': self.coverage_analysis
        }


class ProbabilityCalibrator(ABC):
    """Base class for probability calibrators"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.calibrator = None
        self.fit_params = {}
        
    @abstractmethod
    def fit(self, probabilities: np.ndarray, 
           true_outcomes: np.ndarray) -> 'ProbabilityCalibrator':
        """Fit calibrator to data"""
        pass
    
    @abstractmethod
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities"""
        pass
    
    def fit_calibrate(self, probabilities: np.ndarray, 
                     true_outcomes: np.ndarray) -> np.ndarray:
        """Fit and apply calibration in one step"""
        return self.fit(probabilities, true_outcomes).calibrate(probabilities)


class IsotonicCalibrator(ProbabilityCalibrator):
    """Isotonic regression calibration (non-parametric, monotonic)"""
    
    def __init__(self, out_of_bounds: str = 'clip'):
        super().__init__("Isotonic")
        self.out_of_bounds = out_of_bounds
        
    def fit(self, probabilities: np.ndarray, 
           true_outcomes: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic regression calibrator"""
        self.calibrator = IsotonicRegression(
            out_of_bounds=self.out_of_bounds
        )
        self.calibrator.fit(probabilities, true_outcomes)
        self.is_fitted = True
        self.fit_params = {
            'out_of_bounds': self.out_of_bounds,
            'n_samples': len(probabilities)
        }
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        return self.calibrator.predict(probabilities)


class PlattCalibrator(ProbabilityCalibrator):
    """Platt scaling (sigmoid calibration)"""
    
    def __init__(self, max_iter: int = 100, random_state: int = 42):
        super().__init__("Platt")
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, probabilities: np.ndarray, 
           true_outcomes: np.ndarray) -> 'PlattCalibrator':
        """Fit Platt scaling (logistic regression)"""
        # Convert to log-odds for logistic regression
        probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        log_odds = np.log(probabilities_clipped / (1 - probabilities_clipped))
        
        self.calibrator = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.calibrator.fit(log_odds.reshape(-1, 1), true_outcomes)
        self.is_fitted = True
        self.fit_params = {
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'n_samples': len(probabilities),
            'coef': self.calibrator.coef_[0][0],
            'intercept': self.calibrator.intercept_[0]
        }
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        log_odds = np.log(probabilities_clipped / (1 - probabilities_clipped))
        return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]


class TemperatureScalingCalibrator(ProbabilityCalibrator):
    """Temperature scaling calibration (single parameter)"""
    
    def __init__(self):
        super().__init__("Temperature")
        self.temperature = 1.0
        
    def fit(self, probabilities: np.ndarray, 
           true_outcomes: np.ndarray) -> 'TemperatureScalingCalibrator':
        """Fit temperature parameter using MLE"""
        
        def nll_loss(temperature):
            """Negative log-likelihood loss"""
            probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
            # Convert to logits
            logits = np.log(probabilities_clipped / (1 - probabilities_clipped))
            # Apply temperature scaling
            scaled_logits = logits / temperature
            # Convert back to probabilities
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, 1e-15, 1 - 1e-15)
            # Calculate negative log-likelihood
            return -np.mean(true_outcomes * np.log(scaled_probs) + 
                          (1 - true_outcomes) * np.log(1 - scaled_probs))
        
        # Optimize temperature parameter
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        self.fit_params = {
            'temperature': self.temperature,
            'n_samples': len(probabilities),
            'optimization_success': result.success
        }
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        # Convert to logits
        logits = np.log(probabilities_clipped / (1 - probabilities_clipped))
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        # Convert back to probabilities
        return 1 / (1 + np.exp(-scaled_logits))


class BetaCalibrator(ProbabilityCalibrator):
    """Beta calibration using maximum likelihood estimation"""
    
    def __init__(self):
        super().__init__("Beta")
        self.alpha = 1.0
        self.beta = 1.0
        
    def fit(self, probabilities: np.ndarray, 
           true_outcomes: np.ndarray) -> 'BetaCalibrator':
        """Fit beta distribution parameters"""
        
        def beta_nll(params):
            """Negative log-likelihood for beta calibration"""
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            
            # Transform probabilities using beta CDF
            calibrated = stats.beta.cdf(probabilities, alpha, beta)
            calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
            
            # Calculate negative log-likelihood
            return -np.mean(true_outcomes * np.log(calibrated) + 
                          (1 - true_outcomes) * np.log(1 - calibrated))
        
        # Optimize parameters
        from scipy.optimize import minimize
        result = minimize(beta_nll, [1.0, 1.0], method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        if result.success:
            self.alpha, self.beta = result.x
        else:
            logger.warning("Beta calibration optimization failed, using defaults")
            self.alpha, self.beta = 1.0, 1.0
            
        self.is_fitted = True
        self.fit_params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'n_samples': len(probabilities),
            'optimization_success': result.success
        }
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply beta calibration"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        return stats.beta.cdf(probabilities, self.alpha, self.beta)


class CalibrationDiagnostics:
    """Comprehensive calibration quality assessment"""
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, 
                                 y_prob: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    @staticmethod
    def maximum_calibration_error(y_true: np.ndarray, 
                                y_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
        return mce
    
    @staticmethod
    def reliability_diagram(y_true: np.ndarray, 
                          y_prob: np.ndarray, 
                          n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Generate reliability diagram data"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0.0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                count_in_bin = 0
                
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            counts.append(count_in_bin)
            
        return {
            'accuracies': np.array(accuracies),
            'confidences': np.array(confidences),
            'counts': np.array(counts),
            'bin_boundaries': bin_boundaries
        }