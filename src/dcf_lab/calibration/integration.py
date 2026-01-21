"""
Calibration Integration Framework

This module integrates probability calibration and conformal prediction with
the existing ML framework, providing seamless calibration for CV, ensemble,
and backtest workflows.

Key Features:
- Integration with walk-forward cross-validation
- Calibrated ensemble predictions
- Backtest-aware calibration pipelines
- Unified configuration and validation
- Production-ready calibration workflows

The integration ensures that calibration is properly validated using
time-series aware splits to prevent data leakage while maintaining
the temporal structure essential for financial ML.

Components:
- CalibratedCVFramework: CV integration
- CalibratedEnsemble: Ensemble calibration  
- CalibratedBacktester: Backtest integration
- CalibrationConfig: Unified configuration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit

from .calibrators import (
    ProbabilityCalibrator, CalibrationMethod, IsotonicCalibrator, PlattCalibrator, TemperatureScalingCalibrator
)
from .conformal import (
    ConformalPredictor, ConformalMethod, CQRConformalPredictor, AdaptiveConformalPredictor
)
from .uncertainty import UncertaintyQuantifier, CalibrationDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Comprehensive calibration configuration"""
    
    # Calibration methods
    calibration_method: CalibrationMethod = CalibrationMethod.ISOTONIC
    conformal_method: ConformalMethod = ConformalMethod.CQR
    
    # Coverage targets
    coverage_levels: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    
    # Validation settings
    use_time_series_split: bool = True
    n_calibration_splits: int = 5
    calibration_test_size: float = 0.2
    
    # Recalibration settings
    recalibration_frequency: str = "weekly"  # daily, weekly, monthly
    min_calibration_samples: int = 100
    
    # Quality thresholds
    max_calibration_error: float = 0.1  # ECE threshold
    min_coverage_probability: float = 0.85  # Minimum acceptable coverage
    
    # Advanced settings
    enable_adaptive_conformal: bool = True
    adaptation_rate: float = 0.1
    store_calibration_history: bool = True
    
    # Integration settings
    integrate_with_ensemble: bool = True
    integrate_with_backtest: bool = True
    validate_on_test_set: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        valid = True
        
        if not (0 < self.calibration_test_size < 1):
            logger.error("calibration_test_size must be between 0 and 1")
            valid = False
            
        if self.n_calibration_splits < 2:
            logger.error("n_calibration_splits must be at least 2")
            valid = False
            
        for coverage in self.coverage_levels:
            if not (0 < coverage < 1):
                logger.error(f"Coverage level {coverage} must be between 0 and 1")
                valid = False
        
        if not (0 < self.max_calibration_error < 1):
            logger.error("max_calibration_error must be between 0 and 1")
            valid = False
        
        return valid


class CalibratedCVFramework:
    """Cross-validation framework with integrated calibration"""
    
    def __init__(self, 
                 base_estimator: BaseEstimator,
                 config: CalibrationConfig):
        self.base_estimator = base_estimator
        self.config = config
        self.calibrators = {}
        self.conformal_predictors = {}
        self.calibration_history = []
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid calibration configuration")
        
        logger.info(f"Initialized CalibratedCVFramework with {config.calibration_method.value} calibration")
    
    def fit_predict_calibrated(self, 
                              x_data: np.ndarray,
                              y_data: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              return_intervals: bool = True) -> Dict[str, Any]:
        """Fit model with calibrated predictions using time-series CV"""
        
        if self.config.use_time_series_split:
            cv_splitter = TimeSeriesSplit(n_splits=self.config.n_calibration_splits)
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(n_splits=self.config.n_calibration_splits, shuffle=False)
        
        all_predictions = []
        all_calibrated_predictions = []
        all_intervals = {}
        all_true_values = []
        all_indices = []
        
        # Initialize interval storage for each coverage level
        for coverage in self.config.coverage_levels:
            all_intervals[coverage] = {'lower': [], 'upper': []}
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(x_data)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_calibration_splits}")
            
            # Split data for this fold
            x_train, x_test = x_data[train_idx], x_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]
            
            # Further split training data for calibration
            cal_split_idx = int(len(train_idx) * (1 - self.config.calibration_test_size))
            
            x_fit = x_train[:cal_split_idx]
            y_fit = y_train[:cal_split_idx]
            x_cal = x_train[cal_split_idx:]
            y_cal = y_train[cal_split_idx:]
            
            # Fit base model
            model = clone(self.base_estimator)
            model.fit(x_fit, y_fit)
            
            # Get predictions for calibration and test
            cal_predictions = model.predict(x_cal)
            test_predictions = model.predict(x_test)
            
            # Fit calibrator
            calibrator = self._create_calibrator()
            calibrator.fit(cal_predictions, y_cal)
            
            # Fit conformal predictors for each coverage level
            conformal_predictors = {}
            for coverage in self.config.coverage_levels:
                alpha = 1 - coverage
                conformal_pred = self._create_conformal_predictor(alpha)
                conformal_pred.fit(x_cal, y_cal, cal_predictions)
                conformal_predictors[coverage] = conformal_pred
            
            # Apply calibration to test predictions
            if hasattr(calibrator, 'calibrate'):
                calibrated_test_predictions = calibrator.calibrate(test_predictions)
            else:
                calibrated_test_predictions = test_predictions
            
            # Generate prediction intervals
            fold_intervals = {}
            for coverage in self.config.coverage_levels:
                conformal_pred = conformal_predictors[coverage]
                lower, upper = conformal_pred.predict(test_predictions, x_test)
                fold_intervals[coverage] = {'lower': lower, 'upper': upper}
            
            # Store results
            all_predictions.extend(test_predictions)
            all_calibrated_predictions.extend(calibrated_test_predictions)
            all_true_values.extend(y_test)
            all_indices.extend(test_idx)
            
            for coverage in self.config.coverage_levels:
                all_intervals[coverage]['lower'].extend(fold_intervals[coverage]['lower'])
                all_intervals[coverage]['upper'].extend(fold_intervals[coverage]['upper'])
            
            # Store fold-specific results
            fold_result = {
                'fold_idx': fold_idx,
                'test_indices': test_idx,
                'calibrator': calibrator,
                'conformal_predictors': conformal_predictors,
                'n_train': len(x_fit),
                'n_cal': len(x_cal),
                'n_test': len(x_test)
            }
            fold_results.append(fold_result)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_calibrated_predictions = np.array(all_calibrated_predictions)
        all_true_values = np.array(all_true_values)
        all_indices = np.array(all_indices)
        
        # Convert intervals to numpy arrays
        for coverage in self.config.coverage_levels:
            all_intervals[coverage]['lower'] = np.array(all_intervals[coverage]['lower'])
            all_intervals[coverage]['upper'] = np.array(all_intervals[coverage]['upper'])
        
        # Analyze calibration quality
        calibration_analysis = self._analyze_calibration_quality(
            all_true_values, all_calibrated_predictions, all_intervals
        )
        
        # Prepare results
        results = {
            'predictions': all_predictions,
            'calibrated_predictions': all_calibrated_predictions,
            'true_values': all_true_values,
            'test_indices': all_indices,
            'prediction_intervals': all_intervals,
            'calibration_analysis': calibration_analysis,
            'fold_results': fold_results,
            'config': self.config
        }
        
        # Store in history
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'results': results,
            'data_shape': x_data.shape
        })
        
        return results
    
    def _create_calibrator(self) -> ProbabilityCalibrator:
        """Create calibrator based on configuration"""
        if self.config.calibration_method == CalibrationMethod.ISOTONIC:
            return IsotonicCalibrator()
        elif self.config.calibration_method == CalibrationMethod.PLATT:
            return PlattCalibrator()
        elif self.config.calibration_method == CalibrationMethod.TEMPERATURE:
            return TemperatureScalingCalibrator()
        else:
            logger.warning(f"Unknown calibration method {self.config.calibration_method}, using isotonic")
            return IsotonicCalibrator()
    
    def _create_conformal_predictor(self, alpha: float) -> ConformalPredictor:
        """Create conformal predictor based on configuration"""
        if self.config.conformal_method == ConformalMethod.CQR:
            return CQRConformalPredictor(alpha=alpha)
        elif self.config.conformal_method == ConformalMethod.ADAPTIVE:
            return AdaptiveConformalPredictor(
                alpha=alpha, 
                adaptation_rate=self.config.adaptation_rate
            )
        else:
            logger.warning(f"Unknown conformal method {self.config.conformal_method}, using CQR")
            return CQRConformalPredictor(alpha=alpha)
    
    def _analyze_calibration_quality(self, 
                                   y_true: np.ndarray,
                                   y_calibrated: np.ndarray,
                                   intervals: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze calibration and coverage quality"""
        
        # Initialize uncertainty quantifier
        uncertainty_quantifier = UncertaintyQuantifier()
        
        # Analyze calibration (for classification tasks)
        if np.all((y_calibrated >= 0) & (y_calibrated <= 1)):
            # Convert regression targets to binary for calibration analysis
            y_binary = (y_true > np.median(y_true)).astype(int)
            calibration_metrics = uncertainty_quantifier.analyze_calibration(y_binary, y_calibrated)
        else:
            calibration_metrics = None
        
        # Analyze coverage for each level
        coverage_analysis = {}
        for coverage_level in self.config.coverage_levels:
            lower_bounds = intervals[coverage_level]['lower']
            upper_bounds = intervals[coverage_level]['upper']
            
            coverage_metrics = uncertainty_quantifier.analyze_coverage(
                y_true, lower_bounds, upper_bounds, coverage_level
            )
            coverage_analysis[coverage_level] = coverage_metrics
        
        return {
            'calibration_metrics': calibration_metrics.to_dict() if calibration_metrics else None,
            'coverage_analysis': coverage_analysis,
            'overall_quality': self._compute_overall_quality_score(coverage_analysis)
        }
    
    def _compute_overall_quality_score(self, coverage_analysis: Dict[str, Any]) -> float:
        """Compute overall quality score for calibration"""
        scores = []
        
        for coverage_level, metrics in coverage_analysis.items():
            # Coverage quality (closer to target = better)
            coverage_quality = 1 - abs(metrics['coverage_gap'])
            
            # Efficiency (narrower intervals = better, but penalize if coverage too low)
            efficiency = 1 / (1 + metrics['average_width'])
            if metrics['empirical_coverage'] < self.config.min_coverage_probability:
                efficiency *= 0.5  # Penalize poor coverage
            
            # Combined score for this coverage level
            level_score = 0.7 * coverage_quality + 0.3 * efficiency
            scores.append(level_score)
        
        return np.mean(scores)


class CalibrationValidator:
    """Comprehensive validation for calibrated models"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.diagnostics = CalibrationDiagnostics()
    
    def validate_calibration(self, 
                           results: Dict[str, Any],
                           generate_plots: bool = True) -> Dict[str, Any]:
        """Comprehensive calibration validation"""
        
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Extract data
        results['true_values']
        results['calibrated_predictions']
        results['prediction_intervals']
        
        # Validate calibration quality
        if results['calibration_analysis']['calibration_metrics'] is not None:
            cal_metrics = results['calibration_analysis']['calibration_metrics']
            
            # Check ECE threshold
            if cal_metrics['calibration_error'] > self.config.max_calibration_error:
                validation_results['errors'].append(
                    f"Calibration error {cal_metrics['calibration_error']:.4f} exceeds threshold {self.config.max_calibration_error}"
                )
                validation_results['passed'] = False
            
            validation_results['metrics']['calibration'] = cal_metrics
        
        # Validate coverage
        coverage_results = {}
        for coverage_level in self.config.coverage_levels:
            coverage_metrics = results['calibration_analysis']['coverage_analysis'][coverage_level]
            
            # Check coverage adequacy
            if coverage_metrics['empirical_coverage'] < self.config.min_coverage_probability:
                validation_results['warnings'].append(
                    f"Coverage {coverage_metrics['empirical_coverage']:.3f} below minimum {self.config.min_coverage_probability} for level {coverage_level}"
                )
            
            coverage_results[coverage_level] = coverage_metrics
        
        validation_results['metrics']['coverage'] = coverage_results
        
        # Overall quality assessment
        overall_quality = results['calibration_analysis']['overall_quality']
        validation_results['metrics']['overall_quality'] = overall_quality
        
        if overall_quality < 0.7:
            validation_results['warnings'].append(
                f"Overall quality score {overall_quality:.3f} is below recommended threshold 0.7"
            )
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        return validation_results
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if validation_results['errors']:
            recommendations.append("Consider recalibrating with a different method")
            recommendations.append("Increase calibration sample size if possible")
        
        if validation_results['warnings']:
            recommendations.append("Monitor calibration performance over time")
            recommendations.append("Consider adaptive conformal prediction for non-stationary data")
        
        # Quality-based recommendations
        quality = validation_results['metrics'].get('overall_quality', 1.0)
        if quality < 0.8:
            recommendations.append("Experiment with ensemble calibration methods")
            recommendations.append("Validate feature engineering for better base predictions")
        
        return recommendations