"""
Production Drift Monitoring System

This module implements real-time drift detection for the production trading system:
- ADWIN (Adaptive Sliding Window) and Page-Hinkley drift detection
- Continuous monitoring of residuals and feature distributions
- Automated drift response with light updates vs full retraining
- Alert system with configurable thresholds and actions

Key features:
- Real-time stream monitoring with adaptive algorithms
- Feature drift detection using statistical tests
- Residual monitoring for model performance drift
- Automated response system with escalation policies
- Integration with model updater for adaptive learning
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection algorithms"""
    
    # ADWIN parameters
    adwin_delta: float = 0.002  # Confidence level (smaller = more sensitive)
    adwin_min_window_length: int = 10
    adwin_max_window_length: int = 1000
    
    # Page-Hinkley parameters
    ph_threshold: float = 50.0  # Detection threshold
    ph_alpha: float = 0.9999  # Forgetting factor
    ph_lambda: float = 50.0  # Magnitude threshold
    
    # Feature drift parameters
    feature_drift_window: int = 100  # Window size for feature distribution comparison
    feature_drift_alpha: float = 0.05  # Significance level for statistical tests
    
    # Monitoring windows
    residual_window: int = 200  # Window for residual monitoring
    performance_window: int = 50  # Window for performance monitoring
    
    # Alert thresholds
    drift_alert_threshold: float = 0.8  # Probability threshold for drift alerts
    performance_degradation_threshold: float = 0.2  # Relative performance loss threshold

@dataclass
class DriftEvent:
    """Represents a detected drift event"""
    
    timestamp: str
    drift_type: str  # 'feature', 'residual', 'performance'
    metric_name: str
    drift_score: float
    confidence: float
    affected_features: List[str] = field(default_factory=list)
    suggested_action: str = "monitor"  # 'monitor', 'light_update', 'full_retrain'
    details: Dict[str, Any] = field(default_factory=dict)

class DriftDetector(ABC):
    """Abstract base class for drift detection algorithms"""
    
    @abstractmethod
    def add_observation(self, value: float) -> bool:
        """Add new observation and return True if drift detected"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset detector state"""
        pass
    
    @abstractmethod
    def get_drift_score(self) -> float:
        """Get current drift score (0-1)"""
        pass

class ADWINDetector(DriftDetector):
    """Adaptive Sliding Window drift detector"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.window_sum = 0.0
        self.window_sum_squared = 0.0
        self.drift_detected = False
        
    def add_observation(self, value: float) -> bool:
        """Add observation and check for drift"""
        
        self.window.append(value)
        self.window_sum += value
        self.window_sum_squared += value * value
        
        # Check for drift using ADWIN algorithm
        drift_detected = self._detect_drift()
        
        if drift_detected:
            self._handle_drift()
            return True
        
        return False
    
    def _detect_drift(self) -> bool:
        """Detect drift using ADWIN algorithm"""
        
        window_length = len(self.window)
        
        if window_length < 10:  # Need minimum window size
            return False
        
        # Check all possible cut points
        for i in range(5, window_length - 5):  # Leave margin on both sides
            
            # Split window into two sub-windows
            window1 = list(self.window)[:i]
            window2 = list(self.window)[i:]
            
            # Calculate means
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)
            
            # Calculate bound for significant difference
            n1, n2 = len(window1), len(window2)
            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            
            # Estimate variance
            var1 = np.var(window1) if n1 > 1 else 0
            var2 = np.var(window2) if n2 > 1 else 0
            var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) if n1 + n2 > 2 else 1.0
            
            # ADWIN bound
            epsilon = np.sqrt((2.0 * var_pooled * np.log(2.0 / self.delta)) / m)
            
            # Check if difference is significant
            if abs(mean1 - mean2) > epsilon:
                return True
        
        return False
    
    def _handle_drift(self):
        """Handle detected drift by trimming window"""
        
        # Remove older observations to adapt to new distribution
        trim_size = max(1, len(self.window) // 4)
        
        for _ in range(trim_size):
            if self.window:
                removed_value = self.window.popleft()
                self.window_sum -= removed_value
                self.window_sum_squared -= removed_value * removed_value
        
        self.drift_detected = True
    
    def reset(self):
        """Reset detector state"""
        self.window.clear()
        self.window_sum = 0.0
        self.window_sum_squared = 0.0
        self.drift_detected = False
    
    def get_drift_score(self) -> float:
        """Get normalized drift score"""
        
        if len(self.window) < 10:
            return 0.0
        
        # Calculate instability as a drift score
        recent_variance = np.var(list(self.window)[-10:]) if len(self.window) >= 10 else 0
        total_variance = np.var(list(self.window)) if len(self.window) > 1 else 0
        
        if total_variance == 0:
            return 0.0
        
        # Normalize to 0-1 range
        instability = min(recent_variance / total_variance, 1.0)
        return instability

class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley change detection algorithm"""
    
    def __init__(self, threshold: float = 50.0, alpha: float = 0.9999, lambda_param: float = 50.0):
        self.threshold = threshold
        self.alpha = alpha
        self.lambda_param = lambda_param
        
        self.sum_positive = 0.0
        self.sum_negative = 0.0
        self.x_mean = 0.0
        self.n_observations = 0
        
    def add_observation(self, value: float) -> bool:
        """Add observation and check for drift"""
        
        self.n_observations += 1
        
        # Update running mean
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value
        
        # Calculate deviation from mean
        deviation = value - self.x_mean - self.lambda_param
        
        # Update cumulative sums
        self.sum_positive = max(0, self.sum_positive + deviation)
        self.sum_negative = max(0, self.sum_negative - deviation)
        
        # Check for drift
        if self.sum_positive > self.threshold or self.sum_negative > self.threshold:
            self.reset()
            return True
        
        return False
    
    def reset(self):
        """Reset detector state"""
        self.sum_positive = 0.0
        self.sum_negative = 0.0
    
    def get_drift_score(self) -> float:
        """Get normalized drift score"""
        
        max_sum = max(self.sum_positive, self.sum_negative)
        return min(max_sum / self.threshold, 1.0)

class FeatureDriftDetector:
    """Detects drift in feature distributions using statistical tests"""
    
    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        self.window_size = window_size
        self.alpha = alpha
        self.reference_windows = {}  # Store reference distributions per feature
        self.current_windows = {}    # Store current observations per feature
        
    def add_observations(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Add feature observations and return drift scores
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary of feature name -> drift score (0-1)
        """
        
        drift_scores = {}
        
        for feature_name, value in features.items():
            if not np.isfinite(value):
                continue
                
            # Initialize windows if needed
            if feature_name not in self.current_windows:
                self.current_windows[feature_name] = deque(maxlen=self.window_size)
                self.reference_windows[feature_name] = None
            
            # Add observation to current window
            self.current_windows[feature_name].append(value)
            
            # Calculate drift score if we have enough data
            if len(self.current_windows[feature_name]) >= self.window_size // 2:
                drift_score = self._calculate_drift_score(feature_name)
                drift_scores[feature_name] = drift_score
        
        return drift_scores
    
    def _calculate_drift_score(self, feature_name: str) -> float:
        """Calculate drift score for a specific feature"""
        
        current_data = list(self.current_windows[feature_name])
        
        # Use first half of data as reference if no reference set
        if self.reference_windows[feature_name] is None:
            if len(current_data) >= self.window_size:
                mid_point = len(current_data) // 2
                self.reference_windows[feature_name] = current_data[:mid_point]
                current_data = current_data[mid_point:]
            else:
                return 0.0  # Not enough data
        
        reference_data = self.reference_windows[feature_name]
        
        if len(current_data) < 10 or len(reference_data) < 10:
            return 0.0
        
        # Perform Kolmogorov-Smirnov test
        try:
            ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
            
            # Convert p-value to drift score (lower p-value = higher drift)
            drift_score = 1.0 - p_value
            
            return min(drift_score, 1.0)
            
        except Exception:
            return 0.0
    
    def update_reference(self, feature_name: str):
        """Update reference distribution for a feature"""
        
        if feature_name in self.current_windows and len(self.current_windows[feature_name]) >= self.window_size:
            self.reference_windows[feature_name] = list(self.current_windows[feature_name])
            self.current_windows[feature_name].clear()

class ProductionDriftMonitor:
    """Main drift monitoring system for production"""
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        
        # Initialize detectors
        self.residual_detector = ADWINDetector(self.config.adwin_delta)
        self.performance_detector = PageHinkleyDetector(
            self.config.ph_threshold, 
            self.config.ph_alpha, 
            self.config.ph_lambda
        )
        self.feature_detector = FeatureDriftDetector(
            self.config.feature_drift_window,
            self.config.feature_drift_alpha
        )
        
        # Monitoring data
        self.drift_events = []
        self.monitoring_metrics = {
            'residuals': deque(maxlen=self.config.residual_window),
            'performance': deque(maxlen=self.config.performance_window),
            'feature_drift_scores': {}
        }
        
        # State tracking
        self.last_drift_time = None
        self.drift_alert_active = False
        
    def monitor_residuals(self, residuals: np.ndarray) -> List[DriftEvent]:
        """
        Monitor model residuals for drift
        
        Args:
            residuals: Array of model residuals
            
        Returns:
            List of detected drift events
        """
        
        events = []
        
        for residual in residuals:
            if not np.isfinite(residual):
                continue
                
            self.monitoring_metrics['residuals'].append(residual)
            
            # Check for drift using ADWIN
            drift_detected = self.residual_detector.add_observation(abs(residual))
            
            if drift_detected:
                drift_score = self.residual_detector.get_drift_score()
                
                event = DriftEvent(
                    timestamp=datetime.now().isoformat(),
                    drift_type='residual',
                    metric_name='absolute_residual',
                    drift_score=drift_score,
                    confidence=1.0 - self.config.adwin_delta,
                    suggested_action=self._suggest_action(drift_score, 'residual'),
                    details={
                        'recent_residual_std': np.std(list(self.monitoring_metrics['residuals'])[-20:]),
                        'total_residual_std': np.std(list(self.monitoring_metrics['residuals']))
                    }
                )
                
                events.append(event)
                self.drift_events.append(event)
                
                logger.warning(f"Residual drift detected: score={drift_score:.3f}")
        
        return events
    
    def monitor_performance(self, metric_values: Dict[str, float]) -> List[DriftEvent]:
        """
        Monitor model performance metrics for drift
        
        Args:
            metric_values: Dictionary of metric name -> value
            
        Returns:
            List of detected drift events
        """
        
        events = []
        
        for metric_name, value in metric_values.items():
            if not np.isfinite(value):
                continue
            
            # Store performance metric
            if metric_name not in self.monitoring_metrics:
                self.monitoring_metrics[metric_name] = deque(maxlen=self.config.performance_window)
            
            self.monitoring_metrics[metric_name].append(value)
            
            # Check for performance degradation
            if len(self.monitoring_metrics[metric_name]) >= 20:
                recent_performance = np.mean(list(self.monitoring_metrics[metric_name])[-10:])
                baseline_performance = np.mean(list(self.monitoring_metrics[metric_name])[:10])
                
                if baseline_performance != 0:
                    relative_change = abs(recent_performance - baseline_performance) / abs(baseline_performance)
                    
                    if relative_change > self.config.performance_degradation_threshold:
                        
                        event = DriftEvent(
                            timestamp=datetime.now().isoformat(),
                            drift_type='performance',
                            metric_name=metric_name,
                            drift_score=min(relative_change, 1.0),
                            confidence=0.8,
                            suggested_action=self._suggest_action(relative_change, 'performance'),
                            details={
                                'recent_value': recent_performance,
                                'baseline_value': baseline_performance,
                                'relative_change': relative_change
                            }
                        )
                        
                        events.append(event)
                        self.drift_events.append(event)
                        
                        logger.warning(f"Performance drift detected in {metric_name}: {relative_change:.3f}")
        
        return events
    
    def monitor_features(self, features: Dict[str, float]) -> List[DriftEvent]:
        """
        Monitor feature distributions for drift
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            List of detected drift events
        """
        
        events = []
        
        # Get drift scores for all features
        drift_scores = self.feature_detector.add_observations(features)
        
        # Update stored drift scores
        self.monitoring_metrics['feature_drift_scores'].update(drift_scores)
        
        # Check for significant drift
        for feature_name, drift_score in drift_scores.items():
            if drift_score > self.config.drift_alert_threshold:
                
                event = DriftEvent(
                    timestamp=datetime.now().isoformat(),
                    drift_type='feature',
                    metric_name=feature_name,
                    drift_score=drift_score,
                    confidence=drift_score,
                    affected_features=[feature_name],
                    suggested_action=self._suggest_action(drift_score, 'feature'),
                    details={
                        'ks_statistic': drift_score,
                        'feature_type': 'continuous'
                    }
                )
                
                events.append(event)
                self.drift_events.append(event)
                
                logger.warning(f"Feature drift detected in {feature_name}: score={drift_score:.3f}")
        
        return events
    
    def _suggest_action(self, drift_score: float, drift_type: str) -> str:
        """Suggest appropriate action based on drift severity"""
        
        if drift_type == 'feature':
            if drift_score > 0.9:
                return 'full_retrain'
            elif drift_score > 0.7:
                return 'light_update'
            else:
                return 'monitor'
        
        elif drift_type == 'residual':
            if drift_score > 0.8:
                return 'light_update'
            elif drift_score > 0.6:
                return 'monitor'
            else:
                return 'monitor'
        
        elif drift_type == 'performance':
            if drift_score > 0.5:
                return 'full_retrain'
            elif drift_score > 0.3:
                return 'light_update'
            else:
                return 'monitor'
        
        return 'monitor'
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        
        recent_events = [e for e in self.drift_events[-10:]]  # Last 10 events
        
        # Calculate average drift scores
        avg_feature_drift = np.mean(list(self.monitoring_metrics['feature_drift_scores'].values())) if self.monitoring_metrics['feature_drift_scores'] else 0.0
        
        # Get current detector states
        residual_drift_score = self.residual_detector.get_drift_score()
        performance_drift_score = self.performance_detector.get_drift_score()
        
        return {
            'system_status': self._get_system_status(),
            'recent_events': [
                {
                    'timestamp': e.timestamp,
                    'type': e.drift_type,
                    'metric': e.metric_name,
                    'score': e.drift_score,
                    'action': e.suggested_action
                } for e in recent_events
            ],
            'drift_scores': {
                'features': avg_feature_drift,
                'residuals': residual_drift_score,
                'performance': performance_drift_score
            },
            'monitoring_windows': {
                'residuals': len(self.monitoring_metrics['residuals']),
                'features': len(self.monitoring_metrics['feature_drift_scores'])
            },
            'last_drift_time': self.last_drift_time
        }
    
    def _get_system_status(self) -> str:
        """Determine overall system status"""
        
        recent_events = self.drift_events[-5:] if len(self.drift_events) >= 5 else self.drift_events
        
        if not recent_events:
            return 'stable'
        
        # Check for critical drift events
        critical_events = [e for e in recent_events if e.suggested_action == 'full_retrain']
        if critical_events:
            return 'critical_drift'
        
        # Check for moderate drift events
        moderate_events = [e for e in recent_events if e.suggested_action == 'light_update']
        if moderate_events:
            return 'moderate_drift'
        
        return 'stable'
    
    def reset_detectors(self):
        """Reset all drift detectors"""
        
        self.residual_detector.reset()
        self.performance_detector.reset()
        self.feature_detector = FeatureDriftDetector(
            self.config.feature_drift_window,
            self.config.feature_drift_alpha
        )
        
        logger.info("Drift detectors reset")

class DriftResponseManager:
    """Manages automated responses to detected drift"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.response_history = []
        
        # Response strategies
        self.response_strategies = {
            'monitor': self._monitor_strategy,
            'light_update': self._light_update_strategy,
            'full_retrain': self._full_retrain_strategy
        }
    
    def handle_drift_event(self, event: DriftEvent) -> Dict[str, Any]:
        """
        Handle a detected drift event
        
        Args:
            event: Drift event to handle
            
        Returns:
            Response result
        """
        
        logger.info(f"Handling drift event: {event.drift_type} - {event.suggested_action}")
        
        # Execute appropriate response strategy
        if event.suggested_action in self.response_strategies:
            response_result = self.response_strategies[event.suggested_action](event)
        else:
            response_result = self._monitor_strategy(event)
        
        # Record response
        response_record = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'action_taken': event.suggested_action,
            'result': response_result
        }
        
        self.response_history.append(response_record)
        
        return response_result
    
    def _monitor_strategy(self, event: DriftEvent) -> Dict[str, Any]:
        """Monitor strategy - no immediate action"""
        
        return {
            'action': 'monitor',
            'success': True,
            'message': f"Monitoring {event.drift_type} drift in {event.metric_name}",
            'details': {
                'drift_score': event.drift_score,
                'monitoring_enabled': True
            }
        }
    
    def _light_update_strategy(self, event: DriftEvent) -> Dict[str, Any]:
        """Light update strategy - incremental model update"""
        
        # This would trigger the model updater for a light update
        logger.info(f"Triggering light update for {event.drift_type} drift")
        
        return {
            'action': 'light_update',
            'success': True,
            'message': f"Light update triggered for {event.drift_type} drift",
            'details': {
                'update_type': 'incremental',
                'affected_components': event.affected_features,
                'scheduled': True
            }
        }
    
    def _full_retrain_strategy(self, event: DriftEvent) -> Dict[str, Any]:
        """Full retrain strategy - complete model retraining"""
        
        # This would trigger the model updater for a full retrain
        logger.warning(f"Triggering full retrain for {event.drift_type} drift")
        
        return {
            'action': 'full_retrain',
            'success': True,
            'message': f"Full retrain scheduled for {event.drift_type} drift",
            'details': {
                'retrain_type': 'complete',
                'priority': 'high',
                'estimated_duration': '2-4 hours'
            }
        }

class StreamingMonitor:
    """Simplified streaming interface for real-time monitoring"""
    
    def __init__(self, drift_monitor: ProductionDriftMonitor, 
                 response_manager: DriftResponseManager):
        self.drift_monitor = drift_monitor
        self.response_manager = response_manager
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_monitoring = True
        logger.info("Started streaming drift monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        logger.info("Stopped streaming drift monitoring")
    
    def process_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of monitoring data
        
        Args:
            data: Dictionary containing residuals, features, and performance metrics
            
        Returns:
            Processing summary
        """
        
        if not self.is_monitoring:
            return {'status': 'monitoring_disabled'}
        
        all_events = []
        
        # Monitor residuals if provided
        if 'residuals' in data:
            residual_events = self.drift_monitor.monitor_residuals(data['residuals'])
            all_events.extend(residual_events)
        
        # Monitor features if provided
        if 'features' in data:
            feature_events = self.drift_monitor.monitor_features(data['features'])
            all_events.extend(feature_events)
        
        # Monitor performance if provided
        if 'performance' in data:
            perf_events = self.drift_monitor.monitor_performance(data['performance'])
            all_events.extend(perf_events)
        
        # Handle any detected events
        response_results = []
        for event in all_events:
            response = self.response_manager.handle_drift_event(event)
            response_results.append(response)
        
        return {
            'status': 'processed',
            'events_detected': len(all_events),
            'responses_triggered': len(response_results),
            'events': all_events,
            'responses': response_results,
            'monitoring_summary': self.drift_monitor.get_monitoring_summary()
        }

class DriftAlertSystem:
    """Alert system for drift detection with notification capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_history = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    def process_alert(self, event: DriftEvent) -> Dict[str, Any]:
        """Process drift event and generate appropriate alerts"""
        
        alert_level = self._determine_alert_level(event.drift_score)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': alert_level,
            'event': event,
            'message': self._generate_alert_message(event, alert_level),
            'requires_attention': alert_level in ['high', 'critical']
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        if alert_level == 'critical':
            logger.critical(alert['message'])
        elif alert_level == 'high':
            logger.error(alert['message'])
        elif alert_level == 'medium':
            logger.warning(alert['message'])
        else:
            logger.info(alert['message'])
        
        return alert
    
    def _determine_alert_level(self, drift_score: float) -> str:
        """Determine alert level based on drift score"""
        
        if drift_score >= self.alert_thresholds['critical']:
            return 'critical'
        elif drift_score >= self.alert_thresholds['high']:
            return 'high'
        elif drift_score >= self.alert_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alert_message(self, event: DriftEvent, level: str) -> str:
        """Generate human-readable alert message"""
        
        return (f"[{level.upper()}] {event.drift_type.title()} drift detected in {event.metric_name}. "
                f"Score: {event.drift_score:.3f}, Action: {event.suggested_action}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]