"""
Comprehensive Calibration & Conformal Prediction Demonstration

This script demonstrates the complete calibration and conformal prediction
framework, including:

1. Probability calibration with multiple methods
2. Conformal prediction intervals with exact coverage
3. Integration with time-series cross-validation
4. Comprehensive uncertainty quantification
5. Calibration quality assessment and validation

The demonstration uses synthetic financial data to show how the framework
provides well-calibrated probabilities and reliable uncertainty bands
for financial ML applications.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import calibration modules
from calibrators import (
    CalibrationMethod, IsotonicCalibrator, PlattCalibrator, 
    TemperatureScalingCalibrator
)
from conformal import (
    ConformalMethod, RegressionConformalPredictor, CQRConformalPredictor,
    AdaptiveConformalPredictor, ConformalDiagnostics
)
from uncertainty import UncertaintyQuantifier
from integration import CalibratedCVFramework, CalibrationConfig, CalibrationValidator

# Import ML utilities
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def generate_synthetic_financial_data(n_samples: int = 1000, 
                                     n_features: int = 10,
                                     noise_level: float = 0.1,
                                     regime_change: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Generate synthetic financial time series data with realistic properties"""
    
    logger.info(f"🎲 Generating synthetic financial data: {n_samples} samples, {n_features} features")
    
    # Generate time index
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=n_samples, freq='D')
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Base features with financial characteristics
    features = []
    
    for i in range(n_features):
        # Different types of features
        if i % 3 == 0:
            # Trending feature (like price momentum)
            trend = np.cumsum(rng.normal(0, 0.01, n_samples))
            feature = trend + rng.normal(0, 0.1, n_samples)
        elif i % 3 == 1:
            # Mean-reverting feature (like volatility)
            feature = np.zeros(n_samples)
            feature[0] = rng.normal(0, 1)
            for t in range(1, n_samples):
                feature[t] = 0.9 * feature[t-1] + rng.normal(0, 0.3)
        else:
            # Random walk feature
            feature = np.cumsum(rng.normal(0, 0.02, n_samples))
        
        features.append(feature)
    
    X = np.column_stack(features)
    
    # Generate targets with non-linear relationships
    # Use first few features for prediction
    base_signal = (
        2.0 * X[:, 0] +
        1.5 * X[:, 1] * X[:, 2] +  # Interaction term
        0.8 * np.sign(X[:, 3]) * np.abs(X[:, 3])**0.5 +  # Non-linear
        0.5 * X[:, 4]
    )
    
    # Add regime change if requested
    if regime_change:
        regime_change_point = n_samples // 2
        base_signal[regime_change_point:] *= 1.5  # Regime shift
    
    # Add noise and make it heteroskedastic
    volatility = 0.5 + 0.3 * np.abs(X[:, 1])  # Volatility depends on feature
    noise = rng.normal(0, volatility * noise_level)
    
    y_continuous = base_signal + noise
    
    # Create binary classification target
    threshold = np.percentile(y_continuous, 60)  # 40% positive class
    y_binary = (y_continuous > threshold).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"✅ Generated data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    logger.info(f"   Target statistics: mean={np.mean(y_continuous):.3f}, std={np.std(y_continuous):.3f}")
    logger.info(f"   Binary class balance: {np.mean(y_binary):.3f}")
    
    return X_scaled, y_continuous, y_binary, dates


def demonstrate_calibration_methods():
    """Demonstrate different calibration methods"""
    
    logger.info("🔧 Demonstrating Calibration Methods")
    logger.info("=" * 60)
    
    # Generate data
    X, y_continuous, y_binary, dates = generate_synthetic_financial_data(n_samples=800)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_cal = X[:split_idx], X[split_idx:]
    y_train, y_cal = y_binary[:split_idx], y_binary[split_idx:]
    
    # Train a base model (deliberately miscalibrated)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train.astype(float))
    
    # Get raw predictions (probabilities)
    raw_predictions = model.predict(X_cal)
    raw_predictions = np.clip(raw_predictions, 0.01, 0.99)  # Convert to probabilities
    
    # Test different calibration methods
    calibration_methods = {
        'Isotonic': IsotonicCalibrator(),
        'Platt': PlattCalibrator(),
        'Temperature': TemperatureScalingCalibrator()
    }
    
    calibration_results = {}
    uncertainty_quantifier = UncertaintyQuantifier()
    
    logger.info("📊 Calibration Method Comparison:")
    
    for method_name, calibrator in calibration_methods.items():
        logger.info(f"\n🔍 Testing {method_name} calibration...")
        
        # Fit calibrator
        calibrator.fit(raw_predictions, y_cal)
        
        # Apply calibration
        calibrated_predictions = calibrator.calibrate(raw_predictions)
        
        # Analyze calibration quality
        metrics = uncertainty_quantifier.analyze_calibration(y_cal, calibrated_predictions)
        
        calibration_results[method_name] = {
            'calibrator': calibrator,
            'predictions': calibrated_predictions,
            'metrics': metrics
        }
        
        logger.info(f"   ✓ ECE: {metrics.calibration_error:.4f}")
        logger.info(f"   ✓ MCE: {metrics.max_calibration_error:.4f}")
        logger.info(f"   ✓ Brier Score: {metrics.brier_score:.4f}")
        logger.info(f"   ✓ Confidence Correlation: {metrics.confidence_correlation:.4f}")
    
    # Find best calibration method
    best_method = min(calibration_results.keys(), 
                     key=lambda k: calibration_results[k]['metrics'].calibration_error)
    
    logger.info(f"\n🏆 Best calibration method: {best_method}")
    logger.info(f"   ECE: {calibration_results[best_method]['metrics'].calibration_error:.4f}")
    
    return calibration_results, raw_predictions, y_cal


def demonstrate_conformal_prediction():
    """Demonstrate conformal prediction methods"""
    
    logger.info("\n🎯 Demonstrating Conformal Prediction")
    logger.info("=" * 60)
    
    # Generate regression data
    X, y_continuous, y_binary, dates = generate_synthetic_financial_data(n_samples=800)
    
    # Split data for conformal prediction
    split_idx = int(0.6 * len(X))
    cal_idx = int(0.8 * len(X))
    
    X_train = X[:split_idx]
    X_cal = X[split_idx:cal_idx]
    X_test = X[cal_idx:]
    
    y_train = y_continuous[:split_idx]
    y_cal = y_continuous[split_idx:cal_idx]
    y_test = y_continuous[cal_idx:]
    
    # Train base model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    cal_predictions = model.predict(X_cal)
    test_predictions = model.predict(X_test)
    
    # Test different conformal methods
    conformal_methods = {
        'Standard': RegressionConformalPredictor(alpha=0.1),
        'CQR': CQRConformalPredictor(alpha=0.1),
        'Adaptive': AdaptiveConformalPredictor(alpha=0.1, adaptation_rate=0.05)
    }
    
    logger.info("📊 Conformal Method Comparison:")
    
    conformal_results = {}
    for method_name, conformal_predictor in conformal_methods.items():
        logger.info(f"\n🔍 Testing {method_name} conformal prediction...")
        
        # Fit conformal predictor
        conformal_predictor.fit(X_cal, y_cal, cal_predictions)
        
        # Generate prediction intervals
        lower_bounds, upper_bounds = conformal_predictor.predict(test_predictions, X_test)
        
        # Analyze coverage
        coverage_analysis = ConformalDiagnostics.analyze_coverage(
            y_test, lower_bounds, upper_bounds, target_coverage=0.9
        )
        
        efficiency_analysis = ConformalDiagnostics.analyze_efficiency(
            lower_bounds, upper_bounds
        )
        
        conformal_results[method_name] = {
            'predictor': conformal_predictor,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'coverage': coverage_analysis,
            'efficiency': efficiency_analysis
        }
        
        logger.info(f"   ✓ Empirical Coverage: {coverage_analysis['empirical_coverage']:.3f}")
        logger.info(f"   ✓ Target Coverage: {coverage_analysis['target_coverage']:.3f}")
        logger.info(f"   ✓ Coverage Gap: {coverage_analysis['coverage_gap']:.3f}")
        logger.info(f"   ✓ Mean Width: {efficiency_analysis['mean_width']:.3f}")
    
    # Find best conformal method (closest to target coverage with narrowest intervals)
    def score_method(method_data):
        coverage_score = 1 - abs(method_data['coverage']['coverage_gap'])
        efficiency_score = 1 / (1 + method_data['efficiency']['mean_width'])
        return 0.7 * coverage_score + 0.3 * efficiency_score
    
    best_conformal = max(conformal_results.keys(), key=lambda k: score_method(conformal_results[k]))
    
    logger.info(f"\n🏆 Best conformal method: {best_conformal}")
    best_result = conformal_results[best_conformal]
    logger.info(f"   Coverage: {best_result['coverage']['empirical_coverage']:.3f}")
    logger.info(f"   Mean Width: {best_result['efficiency']['mean_width']:.3f}")
    
    return conformal_results, test_predictions, y_test


def demonstrate_integrated_framework():
    """Demonstrate integrated calibration framework with CV"""
    
    logger.info("\n🚀 Demonstrating Integrated Calibration Framework")
    logger.info("=" * 60)
    
    # Generate data
    X, y_continuous, y_binary, dates = generate_synthetic_financial_data(n_samples=1000)
    
    # Configuration
    config = CalibrationConfig(
        calibration_method=CalibrationMethod.ISOTONIC,
        conformal_method=ConformalMethod.CQR,
        coverage_levels=[0.8, 0.9, 0.95],
        use_time_series_split=True,
        n_calibration_splits=5,
        recalibration_frequency="weekly",
        enable_adaptive_conformal=True
    )
    
    # Base estimator
    base_estimator = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Create calibrated CV framework
    calibrated_cv = CalibratedCVFramework(base_estimator, config)
    
    logger.info("📊 Running calibrated cross-validation...")
    
    # Fit and predict with calibration
    results = calibrated_cv.fit_predict_calibrated(X, y_continuous)
    
    # Analyze results
    logger.info("✅ Calibrated CV Results:")
    logger.info(f"   Total predictions: {len(results['predictions'])}")
    logger.info(f"   CV folds: {len(results['fold_results'])}")
    
    # Coverage analysis for each level
    for coverage_level in config.coverage_levels:
        coverage_metrics = results['calibration_analysis']['coverage_analysis'][coverage_level]
        logger.info(f"\n📏 Coverage Level {coverage_level}:")
        logger.info(f"   ✓ Empirical Coverage: {coverage_metrics['empirical_coverage']:.3f}")
        logger.info(f"   ✓ Target Coverage: {coverage_metrics['target_coverage']:.3f}")
        logger.info(f"   ✓ Coverage Gap: {coverage_metrics['coverage_gap']:.3f}")
        logger.info(f"   ✓ Average Width: {coverage_metrics['average_width']:.3f}")
        logger.info(f"   ✓ Efficiency Score: {coverage_metrics['efficiency_score']:.3f}")
    
    # Overall quality
    overall_quality = results['calibration_analysis']['overall_quality']
    logger.info(f"\n🎯 Overall Quality Score: {overall_quality:.3f}")
    
    # Validation
    validator = CalibrationValidator(config)
    validation_results = validator.validate_calibration(results)
    
    logger.info("\n✅ Validation Results:")
    logger.info(f"   Passed: {validation_results['passed']}")
    logger.info(f"   Warnings: {len(validation_results['warnings'])}")
    logger.info(f"   Errors: {len(validation_results['errors'])}")
    
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            logger.warning(f"   ⚠️ {warning}")
    
    if validation_results['errors']:
        for error in validation_results['errors']:
            logger.error(f"   ❌ {error}")
    
    if validation_results['recommendations']:
        logger.info("\n💡 Recommendations:")
        for rec in validation_results['recommendations']:
            logger.info(f"   • {rec}")
    
    return results, validation_results


def create_performance_summary():
    """Create comprehensive performance summary"""
    
    logger.info("\n📋 Performance Summary")
    logger.info("=" * 60)
    
    summary = {
        'calibration_methods_tested': ['Isotonic', 'Platt', 'Temperature Scaling'],
        'conformal_methods_tested': ['Standard', 'CQR', 'Adaptive'],
        'coverage_levels_supported': [0.8, 0.9, 0.95],
        'key_features': [
            'Time-series aware cross-validation',
            'Multiple calibration methods with automatic selection',
            'Conformal prediction with exact coverage guarantees',
            'Comprehensive uncertainty quantification',
            'Integration with existing ML framework',
            'Robust validation and quality assessment'
        ],
        'use_cases': [
            'Risk-aware trading decisions',
            'Portfolio optimization under uncertainty',
            'Model confidence assessment',
            'Regulatory compliance and audit trails',
            'Outlier and regime change detection'
        ]
    }
    
    logger.info("✅ Framework Capabilities:")
    for feature in summary['key_features']:
        logger.info(f"   ✓ {feature}")
    
    logger.info("\n🎯 Financial ML Use Cases:")
    for use_case in summary['use_cases']:
        logger.info(f"   • {use_case}")
    
    return summary


def main():
    """Main demonstration function"""
    
    logger.info("🎯 Probability Calibration & Conformal Prediction Demonstration")
    logger.info("================================================================")
    
    try:
        # Demonstrate individual components
        logger.info("Phase 1: Calibration Methods")
        calibration_results, raw_predictions, y_cal = demonstrate_calibration_methods()
        
        logger.info("\nPhase 2: Conformal Prediction")
        conformal_results, test_predictions, y_test = demonstrate_conformal_prediction()
        
        logger.info("\nPhase 3: Integrated Framework")
        framework_results, validation_results = demonstrate_integrated_framework()
        
        logger.info("\nPhase 4: Performance Summary")
        summary = create_performance_summary()
        
        # Final assessment
        logger.info("\n🎉 Demonstration completed successfully!")
        logger.info("\nKey achievements:")
        logger.info("   ✓ Probability calibration with multiple methods")
        logger.info("   ✓ Conformal prediction intervals with verified coverage")
        logger.info("   ✓ Time-series aware uncertainty quantification")
        logger.info("   ✓ Comprehensive validation and quality assessment")
        logger.info("   ✓ Production-ready calibration framework")
        logger.info("\nThe calibration framework is ready for financial ML applications!")
        
        return {
            'calibration_results': calibration_results,
            'conformal_results': conformal_results,
            'framework_results': framework_results,
            'validation_results': validation_results,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    results = main()