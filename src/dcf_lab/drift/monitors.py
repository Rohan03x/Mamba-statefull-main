"""
Stream Monitors for Concept Drift Detection

This module implements sophisticated drift detection algorithms optimized
for financial time series, including:

1. ADWIN (Adaptive Sliding Window): Automatically adjusts window size when
   distribution changes are detected
2. Page-Hinkley Test: Detects persistent mean shifts with configurable sensitivity
3. Feature Drift Monitor: Tracks changes in feature distributions and correlations
4. Residual Drift Monitor: Monitors model prediction quality degradation

All monitors are designed for real-time streaming data and provide actionable
drift alerts with severity levels for adaptive learning systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels for adaptive actions"""
    NONE = "none"
    MINOR = "minor"      # Small adjustments needed
    MODERATE = "moderate"  # Light retraining recommended
    MAJOR = "major"      # Hard reset/full retrain required
    CRITICAL = "critical"  # Emergency intervention needed


@dataclass
class DriftAlert:
    """Drift detection alert with metadata"""
    timestamp: pd.Timestamp
    monitor_type: str
    severity: DriftSeverity
    metric_value: float
    threshold: float
    confidence: float
    description: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ADWINMonitor:
    """
    ADWIN (Adaptive Windowing) algorithm for concept drift detection.
    
    Automatically adjusts window size when distribution changes are detected.
    Based on the paper: "Learning from Time-Changing Data with Adaptive Windowing"
    by Bifet & Gavalda (2007).
    
    Key Features:
    - Automatic window size adjustment
    - Distribution change detection
    - Configurable confidence levels
    - Financial time series optimizations
    """
    
    def __init__(self, 
                 delta: float = 0.002,
                 min_window_size: int = 30,
                 max_window_size: int = 1000,
                 enable_financial_features: bool = True):
        """
        Initialize ADWIN monitor
        
        Args:
            delta: Confidence parameter (smaller = more sensitive)
            min_window_size: Minimum window size to maintain
            max_window_size: Maximum window size limit
            enable_financial_features: Enable financial-specific optimizations
        """
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.enable_financial_features = enable_financial_features
        
        # Internal state
        self.window = deque()
        self.n_detections = 0
        self.total_variance = 0.0
        self.window_variance = 0.0
        self.last_detection_time = None
        
        # Statistics tracking
        self.detection_history = []
        self.window_sizes = []
        self.variance_history = []
        
        logger.info(f"🔍 Initialized ADWIN monitor: delta={delta}, window=[{min_window_size}, {max_window_size}]")
    
    def add_element(self, value: float, timestamp: Optional[pd.Timestamp] = None) -> Optional[DriftAlert]:
        """
        Add new element and check for drift
        
        Args:
            value: New observation value
            timestamp: Optional timestamp for the observation
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Add to window
        self.window.append((value, timestamp))
        
        # Maintain max window size
        if len(self.window) > self.max_window_size:
            self.window.popleft()
        
        # Need minimum samples for detection
        if len(self.window) < self.min_window_size:
            return None
        
        # Check for drift using ADWIN algorithm
        drift_detected, cut_point = self._detect_drift()
        
        if drift_detected:
            return self._create_drift_alert(cut_point, timestamp)
        
        return None
    
    def _detect_drift(self) -> Tuple[bool, Optional[int]]:
        """
        Core ADWIN drift detection algorithm
        
        Returns:
            Tuple of (drift_detected, cut_point_index)
        """
        window_array = np.array([x[0] for x in self.window])
        n = len(window_array)
        
        # Try different cut points
        for i in range(self.min_window_size, n - self.min_window_size):
            # Split window at cut point
            W0 = window_array[:i]
            W1 = window_array[i:]
            
            # Calculate means and variances
            mu0, mu1 = np.mean(W0), np.mean(W1)
            n0, n1 = len(W0), len(W1)
            
            # Skip if insufficient samples
            if n0 < self.min_window_size or n1 < self.min_window_size:
                continue
            
            # Calculate ADWIN bound
            if self._adwin_bound_exceeded(mu0, mu1, n0, n1):
                # Financial-specific validation
                if self.enable_financial_features:
                    if self._validate_financial_drift(W0, W1):
                        return True, i
                else:
                    return True, i
        
        return False, None
    
    def _adwin_bound_exceeded(self, mu0: float, mu1: float, n0: int, n1: int) -> bool:
        """
        Check if ADWIN bound is exceeded
        
        Args:
            mu0, mu1: Means of the two sub-windows
            n0, n1: Sizes of the two sub-windows
            
        Returns:
            True if bound exceeded (drift detected)
        """
        # Harmonic mean of window sizes
        m = 1.0 / ((1.0 / n0) + (1.0 / n1))
        
        # ADWIN bound calculation
        delta_prime = self.delta / len(self.window)
        epsilon = np.sqrt((2.0 / m) * np.log(2.0 / delta_prime))
        
        # Check if difference exceeds bound
        return abs(mu0 - mu1) > epsilon
    
    def _validate_financial_drift(self, W0: np.ndarray, W1: np.ndarray) -> bool:
        """
        Additional validation for financial time series
        
        Args:
            W0, W1: The two sub-windows
            
        Returns:
            True if financial drift criteria are met
        """
        # Volatility change check
        vol0, vol1 = np.std(W0), np.std(W1)
        vol_ratio = max(vol0, vol1) / (min(vol0, vol1) + 1e-8)
        
        if vol_ratio > 2.0:  # Significant volatility change
            return True
        
        # Trend change check
        trend0 = np.polyfit(range(len(W0)), W0, 1)[0] if len(W0) > 1 else 0
        trend1 = np.polyfit(range(len(W1)), W1, 1)[0] if len(W1) > 1 else 0
        
        if abs(trend0 - trend1) > 0.1:  # Significant trend change
            return True
        
        return False
    
    def _create_drift_alert(self, cut_point: int, timestamp: pd.Timestamp) -> DriftAlert:
        """Create drift alert with metadata"""
        
        window_array = np.array([x[0] for x in self.window])
        W0 = window_array[:cut_point]
        W1 = window_array[cut_point:]
        
        # Calculate severity
        mean_diff = abs(np.mean(W0) - np.mean(W1))
        vol_diff = abs(np.std(W0) - np.std(W1))
        
        if mean_diff > 0.5 or vol_diff > 0.3:
            severity = DriftSeverity.MAJOR
        elif mean_diff > 0.3 or vol_diff > 0.2:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.MINOR
        
        # Update internal state
        self.n_detections += 1
        self.last_detection_time = timestamp
        self.detection_history.append(timestamp)
        
        # Trim window at cut point
        for _ in range(cut_point):
            if self.window:
                self.window.popleft()
        
        alert = DriftAlert(
            timestamp=timestamp,
            monitor_type="ADWIN",
            severity=severity,
            metric_value=mean_diff,
            threshold=self.delta,
            confidence=1.0 - self.delta,
            description=f"ADWIN detected distribution change at cut point {cut_point}",
            recommendations=self._get_recommendations(severity),
            metadata={
                'cut_point': cut_point,
                'mean_difference': mean_diff,
                'volatility_difference': vol_diff,
                'window_size_before': len(window_array),
                'window_size_after': len(self.window)
            }
        )
        
        logger.warning(f"🚨 ADWIN drift detected: {severity.value} severity at {timestamp}")
        return alert
    
    def _get_recommendations(self, severity: DriftSeverity) -> List[str]:
        """Get recommended actions based on drift severity"""
        
        if severity == DriftSeverity.MINOR:
            return [
                "Monitor closely for additional drift signals",
                "Consider light model adjustment",
                "Update feature importance weights"
            ]
        elif severity == DriftSeverity.MODERATE:
            return [
                "Perform light retraining with recent data",
                "Adjust ensemble weights",
                "Increase monitoring frequency"
            ]
        elif severity == DriftSeverity.MAJOR:
            return [
                "Full model retraining recommended",
                "Reset regime detection models",
                "Review feature engineering pipeline"
            ]
        else:
            return ["Continue monitoring"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        
        return {
            'total_detections': self.n_detections,
            'current_window_size': len(self.window),
            'detection_rate': self.n_detections / max(1, len(self.detection_history)),
            'last_detection': self.last_detection_time,
            'average_window_size': np.mean(self.window_sizes) if self.window_sizes else 0,
            'detection_frequency': len(self.detection_history)
        }


class PageHinkleyMonitor:
    """
    Page-Hinkley test for detecting persistent mean shifts.
    
    Detects gradual changes in the mean of a signal over time.
    Particularly effective for detecting trending behavior in financial data.
    
    Key Features:
    - Persistent mean shift detection
    - Configurable sensitivity
    - Two-sided testing (increase/decrease)
    - Financial market optimizations
    """
    
    def __init__(self,
                 threshold: float = 50.0,
                 alpha: float = 0.9999,
                 enable_two_sided: bool = True,
                 min_detections: int = 5):
        """
        Initialize Page-Hinkley monitor
        
        Args:
            threshold: Detection threshold (higher = less sensitive)
            alpha: Forgetting factor for exponential averaging
            enable_two_sided: Enable detection of both increases and decreases
            min_detections: Minimum detections before alerting
        """
        self.threshold = threshold
        self.alpha = alpha
        self.enable_two_sided = enable_two_sided
        self.min_detections = min_detections
        
        # Test statistics
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.n_pos = 0
        self.n_neg = 0
        
        # Running statistics
        self.running_mean = None
        self.n_samples = 0
        
        # Detection tracking
        self.detection_history = []
        self.last_detection_time = None
        
        logger.info(f"📈 Initialized Page-Hinkley monitor: threshold={threshold}, alpha={alpha}")
    
    def add_element(self, value: float, timestamp: Optional[pd.Timestamp] = None) -> Optional[DriftAlert]:
        """
        Add new element and check for drift
        
        Args:
            value: New observation value
            timestamp: Optional timestamp for the observation
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Update running mean
        if self.running_mean is None:
            self.running_mean = value
        else:
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * value
        
        self.n_samples += 1
        
        # Calculate deviation from mean
        deviation = value - self.running_mean
        
        # Update cumulative sums
        if deviation > 0:
            self.sum_pos = max(0, self.sum_pos + deviation)
            self.n_pos += 1
        else:
            self.sum_neg = max(0, self.sum_neg - deviation)
            self.n_neg += 1
        
        # Check for drift
        drift_type = None
        test_statistic = None
        
        if self.sum_pos > self.threshold and self.n_pos >= self.min_detections:
            drift_type = "upward"
            test_statistic = self.sum_pos
            self.sum_pos = 0.0  # Reset after detection
            self.n_pos = 0
        
        elif self.enable_two_sided and self.sum_neg > self.threshold and self.n_neg >= self.min_detections:
            drift_type = "downward"
            test_statistic = self.sum_neg
            self.sum_neg = 0.0  # Reset after detection
            self.n_neg = 0
        
        if drift_type:
            return self._create_drift_alert(drift_type, test_statistic, timestamp)
        
        return None
    
    def _create_drift_alert(self, drift_type: str, test_statistic: float, timestamp: pd.Timestamp) -> DriftAlert:
        """Create drift alert for Page-Hinkley detection"""
        
        # Determine severity based on test statistic
        excess = test_statistic - self.threshold
        normalized_excess = excess / self.threshold
        
        if normalized_excess > 2.0:
            severity = DriftSeverity.MAJOR
        elif normalized_excess > 1.0:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.MINOR
        
        # Update tracking
        self.detection_history.append((timestamp, drift_type, test_statistic))
        self.last_detection_time = timestamp
        
        alert = DriftAlert(
            timestamp=timestamp,
            monitor_type="Page-Hinkley",
            severity=severity,
            metric_value=test_statistic,
            threshold=self.threshold,
            confidence=test_statistic / (test_statistic + self.threshold),
            description=f"Page-Hinkley detected {drift_type} mean shift",
            recommendations=self._get_recommendations(severity, drift_type),
            metadata={
                'drift_direction': drift_type,
                'test_statistic': test_statistic,
                'threshold_excess': excess,
                'running_mean': self.running_mean,
                'n_samples': self.n_samples
            }
        )
        
        logger.warning(f"📊 Page-Hinkley drift detected: {drift_type} {severity.value} at {timestamp}")
        return alert
    
    def _get_recommendations(self, severity: DriftSeverity, drift_type: str) -> List[str]:
        """Get recommended actions based on drift type and severity"""
        
        base_recommendations = []
        
        if drift_type == "upward":
            base_recommendations.extend([
                "Market regime may be shifting upward",
                "Consider adjusting position sizing upward",
                "Review risk management parameters"
            ])
        else:
            base_recommendations.extend([
                "Market regime may be shifting downward", 
                "Consider defensive positioning",
                "Increase hedging allocation"
            ])
        
        if severity == DriftSeverity.MAJOR:
            base_recommendations.extend([
                "Immediate model retraining recommended",
                "Review all model assumptions",
                "Consider regime-switching models"
            ])
        elif severity == DriftSeverity.MODERATE:
            base_recommendations.extend([
                "Light retraining with recent data",
                "Update feature importance",
                "Increase monitoring frequency"
            ])
        
        return base_recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        
        recent_detections = [d for d in self.detection_history if len(self.detection_history) <= 10]
        
        return {
            'total_detections': len(self.detection_history),
            'current_sum_pos': self.sum_pos,
            'current_sum_neg': self.sum_neg,
            'running_mean': self.running_mean,
            'n_samples': self.n_samples,
            'last_detection': self.last_detection_time,
            'recent_detections': recent_detections,
            'detection_rate': len(self.detection_history) / max(1, self.n_samples)
        }


class FeatureDriftMonitor:
    """
    Monitor for feature distribution and correlation changes.
    
    Tracks changes in:
    - Individual feature distributions
    - Feature correlations
    - Feature importance shifts
    - Statistical properties (mean, variance, skewness)
    """
    
    def __init__(self,
                 feature_names: List[str],
                 correlation_threshold: float = 0.3,
                 distribution_threshold: float = 0.1,
                 window_size: int = 100):
        """
        Initialize feature drift monitor
        
        Args:
            feature_names: Names of features to monitor
            correlation_threshold: Threshold for correlation change alerts
            distribution_threshold: Threshold for distribution change alerts
            window_size: Window size for statistical calculations
        """
        self.feature_names = feature_names
        self.correlation_threshold = correlation_threshold
        self.distribution_threshold = distribution_threshold
        self.window_size = window_size
        
        # Feature history
        self.feature_history = {name: deque(maxlen=window_size) for name in feature_names}
        self.baseline_stats = {}
        self.baseline_correlations = None
        
        # Drift tracking
        self.drift_history = []
        self.last_correlation_matrix = None
        
        logger.info(f"🔍 Initialized feature drift monitor: {len(feature_names)} features, window={window_size}")
    
    def add_features(self, features: Dict[str, float], timestamp: Optional[pd.Timestamp] = None) -> List[DriftAlert]:
        """
        Add new feature values and check for drift
        
        Args:
            features: Dictionary of feature_name -> value
            timestamp: Optional timestamp
            
        Returns:
            List of drift alerts (can be multiple)
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        alerts = []
        
        # Add features to history
        for name, value in features.items():
            if name in self.feature_history:
                self.feature_history[name].append(value)
        
        # Need sufficient history for analysis
        min_history = min(len(hist) for hist in self.feature_history.values())
        if min_history < self.window_size // 2:
            return alerts
        
        # Check for feature distribution drift
        distribution_alerts = self._check_distribution_drift(timestamp)
        alerts.extend(distribution_alerts)
        
        # Check for correlation drift
        if min_history >= self.window_size:
            correlation_alerts = self._check_correlation_drift(timestamp)
            alerts.extend(correlation_alerts)
        
        return alerts
    
    def _check_distribution_drift(self, timestamp: pd.Timestamp) -> List[DriftAlert]:
        """Check for changes in individual feature distributions"""
        
        alerts = []
        
        for feature_name, history in self.feature_history.items():
            if len(history) < 20:  # Need minimum samples
                continue
            
            # Calculate current statistics
            current_data = np.array(list(history))
            current_mean = np.mean(current_data)
            current_std = np.std(current_data)
            current_skew = self._calculate_skewness(current_data)
            
            # Initialize baseline if not exists
            if feature_name not in self.baseline_stats:
                self.baseline_stats[feature_name] = {
                    'mean': current_mean,
                    'std': current_std,
                    'skew': current_skew
                }
                continue
            
            # Compare with baseline
            baseline = self.baseline_stats[feature_name]
            
            # Calculate normalized differences
            mean_diff = abs(current_mean - baseline['mean']) / (baseline['std'] + 1e-8)
            std_ratio = current_std / (baseline['std'] + 1e-8)
            skew_diff = abs(current_skew - baseline['skew'])
            
            # Check thresholds
            drift_detected = False
            drift_components = []
            
            if mean_diff > self.distribution_threshold:
                drift_detected = True
                drift_components.append(f"mean shift: {mean_diff:.3f}")
            
            if abs(np.log(std_ratio)) > self.distribution_threshold:
                drift_detected = True
                drift_components.append(f"volatility change: {std_ratio:.3f}x")
            
            if skew_diff > 0.5:
                drift_detected = True
                drift_components.append(f"skewness change: {skew_diff:.3f}")
            
            if drift_detected:
                # Determine severity
                max_change = max(mean_diff, abs(np.log(std_ratio)), skew_diff)
                if max_change > 1.0:
                    severity = DriftSeverity.MAJOR
                elif max_change > 0.5:
                    severity = DriftSeverity.MODERATE
                else:
                    severity = DriftSeverity.MINOR
                
                alert = DriftAlert(
                    timestamp=timestamp,
                    monitor_type="Feature Distribution",
                    severity=severity,
                    metric_value=max_change,
                    threshold=self.distribution_threshold,
                    confidence=min(1.0, max_change / self.distribution_threshold),
                    description=f"Feature '{feature_name}' distribution drift: {', '.join(drift_components)}",
                    recommendations=self._get_feature_recommendations(feature_name, severity),
                    metadata={
                        'feature_name': feature_name,
                        'mean_difference': mean_diff,
                        'std_ratio': std_ratio,
                        'skew_difference': skew_diff,
                        'drift_components': drift_components
                    }
                )
                
                alerts.append(alert)
                
                # Update baseline (adaptive)
                self.baseline_stats[feature_name] = {
                    'mean': 0.9 * baseline['mean'] + 0.1 * current_mean,
                    'std': 0.9 * baseline['std'] + 0.1 * current_std,
                    'skew': 0.9 * baseline['skew'] + 0.1 * current_skew
                }
        
        return alerts
    
    def _check_correlation_drift(self, timestamp: pd.Timestamp) -> List[DriftAlert]:
        """Check for changes in feature correlations"""
        
        alerts = []
        
        # Create feature matrix
        feature_matrix = []
        for name in self.feature_names:
            if name in self.feature_history and len(self.feature_history[name]) >= self.window_size:
                feature_matrix.append(list(self.feature_history[name]))
        
        if len(feature_matrix) < 2:
            return alerts
        
        # Calculate correlation matrix
        feature_df = pd.DataFrame(np.array(feature_matrix).T, columns=self.feature_names[:len(feature_matrix)])
        current_corr = feature_df.corr().values
        
        # Initialize baseline
        if self.baseline_correlations is None:
            self.baseline_correlations = current_corr.copy()
            return alerts
        
        # Compare correlations
        corr_diff = np.abs(current_corr - self.baseline_correlations)
        max_change = np.max(corr_diff[np.triu_indices(len(current_corr), k=1)])
        
        if max_change > self.correlation_threshold:
            # Find most changed correlations
            changed_pairs = []
            for i in range(len(current_corr)):
                for j in range(i + 1, len(current_corr)):
                    if corr_diff[i, j] > self.correlation_threshold:
                        feature_i = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                        feature_j = self.feature_names[j] if j < len(self.feature_names) else f"feature_{j}"
                        changed_pairs.append((feature_i, feature_j, corr_diff[i, j]))
            
            # Determine severity
            if max_change > 0.6:
                severity = DriftSeverity.MAJOR
            elif max_change > 0.4:
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.MINOR
            
            alert = DriftAlert(
                timestamp=timestamp,
                monitor_type="Feature Correlation",
                severity=severity,
                metric_value=max_change,
                threshold=self.correlation_threshold,
                confidence=min(1.0, max_change / self.correlation_threshold),
                description=f"Feature correlation drift detected: max change {max_change:.3f}",
                recommendations=self._get_correlation_recommendations(severity),
                metadata={
                    'max_correlation_change': max_change,
                    'changed_pairs': changed_pairs[:5],  # Top 5 changes
                    'n_changed_pairs': len(changed_pairs)
                }
            )
            
            alerts.append(alert)
            
            # Update baseline (adaptive)
            self.baseline_correlations = 0.9 * self.baseline_correlations + 0.1 * current_corr
        
        self.last_correlation_matrix = current_corr
        return alerts
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    def _get_feature_recommendations(self, feature_name: str, severity: DriftSeverity) -> List[str]:
        """Get recommendations for feature drift"""
        
        recommendations = [
            f"Feature '{feature_name}' showing distribution changes",
            "Review feature engineering pipeline",
            "Check data source quality"
        ]
        
        if severity == DriftSeverity.MAJOR:
            recommendations.extend([
                "Consider feature transformation or replacement",
                "Retrain feature selection models",
                "Investigate root cause of change"
            ])
        elif severity == DriftSeverity.MODERATE:
            recommendations.extend([
                "Update feature normalization parameters",
                "Consider adaptive feature scaling"
            ])
        
        return recommendations
    
    def _get_correlation_recommendations(self, severity: DriftSeverity) -> List[str]:
        """Get recommendations for correlation drift"""
        
        recommendations = [
            "Feature correlation structure has changed",
            "Review feature interactions",
            "Consider dimensionality reduction update"
        ]
        
        if severity == DriftSeverity.MAJOR:
            recommendations.extend([
                "Full feature engineering review needed",
                "Retrain ensemble models",
                "Update feature importance calculations"
            ])
        
        return recommendations


class ResidualDriftMonitor:
    """
    Monitor model prediction quality through residual analysis.
    
    Tracks:
    - Residual distribution changes
    - Prediction accuracy degradation
    - Systematic bias emergence
    - Error pattern shifts
    """
    
    def __init__(self,
                 window_size: int = 100,
                 accuracy_threshold: float = 0.1,
                 bias_threshold: float = 0.05):
        """
        Initialize residual drift monitor
        
        Args:
            window_size: Window size for residual analysis
            accuracy_threshold: Threshold for accuracy degradation
            bias_threshold: Threshold for systematic bias detection
        """
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.bias_threshold = bias_threshold
        
        # Residual tracking
        self.residuals = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.targets = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Baseline metrics
        self.baseline_mse = None
        self.baseline_bias = None
        self.baseline_set = False
        
        # Drift tracking
        self.drift_alerts = []
        
        logger.info(f"📊 Initialized residual drift monitor: window={window_size}")
    
    def add_prediction(self, 
                      prediction: float, 
                      target: float, 
                      timestamp: Optional[pd.Timestamp] = None) -> Optional[DriftAlert]:
        """
        Add new prediction and target, check for residual drift
        
        Args:
            prediction: Model prediction
            target: True target value
            timestamp: Optional timestamp
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        residual = target - prediction
        
        # Add to tracking
        self.residuals.append(residual)
        self.predictions.append(prediction)
        self.targets.append(target)
        self.timestamps.append(timestamp)
        
        # Need sufficient data for analysis
        if len(self.residuals) < self.window_size // 2:
            return None
        
        # Set baseline if not established
        if not self.baseline_set and len(self.residuals) >= self.window_size:
            self._set_baseline()
            return None
        
        # Check for drift
        return self._check_residual_drift(timestamp)
    
    def _set_baseline(self):
        """Set baseline metrics from current window"""
        
        residuals_array = np.array(list(self.residuals))
        self.baseline_mse = np.mean(residuals_array ** 2)
        self.baseline_bias = np.mean(residuals_array)
        self.baseline_set = True
        
        logger.info(f"📊 Set residual baseline: MSE={self.baseline_mse:.4f}, bias={self.baseline_bias:.4f}")
    
    def _check_residual_drift(self, timestamp: pd.Timestamp) -> Optional[DriftAlert]:
        """Check for residual drift patterns"""
        
        if not self.baseline_set:
            return None
        
        residuals_array = np.array(list(self.residuals))
        current_mse = np.mean(residuals_array ** 2)
        current_bias = np.mean(residuals_array)
        
        # Check accuracy degradation
        mse_ratio = current_mse / (self.baseline_mse + 1e-8)
        bias_change = abs(current_bias - self.baseline_bias)
        
        drift_detected = False
        drift_type = None
        main_metric = None
        
        if mse_ratio > (1 + self.accuracy_threshold):
            drift_detected = True
            drift_type = "accuracy_degradation"
            main_metric = mse_ratio
        
        elif bias_change > self.bias_threshold:
            drift_detected = True
            drift_type = "systematic_bias"
            main_metric = bias_change
        
        if drift_detected:
            return self._create_residual_alert(drift_type, main_metric, timestamp, {
                'current_mse': current_mse,
                'baseline_mse': self.baseline_mse,
                'mse_ratio': mse_ratio,
                'current_bias': current_bias,
                'baseline_bias': self.baseline_bias,
                'bias_change': bias_change
            })
        
        return None
    
    def _create_residual_alert(self, 
                             drift_type: str, 
                             metric_value: float, 
                             timestamp: pd.Timestamp,
                             metadata: Dict[str, Any]) -> DriftAlert:
        """Create residual drift alert"""
        
        # Determine severity
        if drift_type == "accuracy_degradation":
            if metric_value > 2.0:
                severity = DriftSeverity.MAJOR
            elif metric_value > 1.5:
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.MINOR
        else:  # systematic_bias
            if metric_value > 0.2:
                severity = DriftSeverity.MAJOR
            elif metric_value > 0.1:
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.MINOR
        
        alert = DriftAlert(
            timestamp=timestamp,
            monitor_type="Residual Drift",
            severity=severity,
            metric_value=metric_value,
            threshold=self.accuracy_threshold if drift_type == "accuracy_degradation" else self.bias_threshold,
            confidence=min(1.0, metric_value / (self.accuracy_threshold if drift_type == "accuracy_degradation" else self.bias_threshold)),
            description=f"Model {drift_type.replace('_', ' ')} detected",
            recommendations=self._get_residual_recommendations(drift_type, severity),
            metadata=metadata
        )
        
        self.drift_alerts.append(alert)
        logger.warning(f"🎯 Residual drift detected: {drift_type} {severity.value} at {timestamp}")
        
        return alert
    
    def _get_residual_recommendations(self, drift_type: str, severity: DriftSeverity) -> List[str]:
        """Get recommendations for residual drift"""
        
        recommendations = []
        
        if drift_type == "accuracy_degradation":
            recommendations.extend([
                "Model prediction accuracy has degraded",
                "Consider model retraining",
                "Review feature quality"
            ])
        else:
            recommendations.extend([
                "Systematic bias detected in predictions",
                "Check for data distribution shift",
                "Review model calibration"
            ])
        
        if severity == DriftSeverity.MAJOR:
            recommendations.extend([
                "Immediate model intervention required",
                "Stop automated trading if applicable",
                "Full model diagnostic needed"
            ])
        elif severity == DriftSeverity.MODERATE:
            recommendations.extend([
                "Schedule model retraining",
                "Increase prediction confidence thresholds",
                "Enhanced monitoring recommended"
            ])
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get residual monitoring statistics"""
        
        if len(self.residuals) == 0:
            return {'status': 'insufficient_data'}
        
        residuals_array = np.array(list(self.residuals))
        
        return {
            'n_predictions': len(self.residuals),
            'current_mse': np.mean(residuals_array ** 2),
            'current_bias': np.mean(residuals_array),
            'current_std': np.std(residuals_array),
            'baseline_mse': self.baseline_mse,
            'baseline_bias': self.baseline_bias,
            'baseline_set': self.baseline_set,
            'n_alerts': len(self.drift_alerts),
            'last_alert': self.drift_alerts[-1].timestamp if self.drift_alerts else None
        }