"""
Production Monitoring System

This module implements comprehensive monitoring for the production trading system:
- Real-time performance tracking with rolling metrics
- Calibration plot generation and monitoring
- Interval coverage validation and drift detection
- Regime mix analysis and options-implied vs realized move tracking
- Alert system with configurable thresholds and escalation
- Dashboard data aggregation for visualization

Key features:
- Multi-dimensional performance monitoring across horizons
- Statistical quality control with control charts
- Real-time alerting with severity levels
- Historical trend analysis and reporting
- Integration with all production components
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import sqlite3
from collections import deque, defaultdict

# Plotting imports
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for production monitoring system"""
    
    # Performance monitoring windows
    short_window: int = 50   # Short-term performance window
    medium_window: int = 200  # Medium-term performance window
    long_window: int = 1000   # Long-term performance window
    
    # Alert thresholds
    rmse_degradation_threshold: float = 0.15  # 15% RMSE increase
    mae_degradation_threshold: float = 0.15   # 15% MAE increase
    coverage_deviation_threshold: float = 0.05  # 5% coverage deviation
    calibration_error_threshold: float = 0.10   # 10% calibration error
    
    # Control chart parameters
    control_chart_sigma: float = 3.0  # Control limit multiplier
    trending_threshold: int = 7       # Consecutive points for trend detection
    
    # Regime analysis
    regime_detection_window: int = 100  # Window for regime change detection
    volatility_threshold: float = 0.2   # Volatility change threshold
    
    # Options analysis
    options_comparison_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Database settings
    db_retention_days: int = 90  # Days to retain detailed monitoring data
    
    # Visualization settings
    plot_style: str = "seaborn"
    figure_size: Tuple[int, int] = (12, 8)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    
    timestamp: str
    horizon: int
    
    # Point prediction metrics
    rmse: float
    mae: float
    r2_score: float
    mean_bias: float
    
    # Probability calibration metrics
    brier_score: Optional[float] = None
    log_score: Optional[float] = None
    calibration_error: Optional[float] = None
    
    # Interval coverage metrics
    coverage_50: float = 0.0
    coverage_80: float = 0.0
    coverage_90: float = 0.0
    coverage_95: float = 0.0
    
    # Conformal interval metrics
    conformal_coverage: float = 0.0
    conformal_width: float = 0.0
    
    # Additional metrics
    prediction_count: int = 0
    avg_uncertainty: float = 0.0
    
    # Market condition indicators
    market_volatility: Optional[float] = None
    regime_indicator: Optional[str] = None

@dataclass
class AlertEvent:
    """Represents a monitoring alert"""
    
    timestamp: str
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    horizon: Optional[int] = None
    message: str = ""
    resolved: bool = False
    resolution_time: Optional[str] = None

class PerformanceTracker:
    """Tracks model performance metrics over time"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Performance history per horizon
        self.performance_history = defaultdict(deque)
        
        # Control chart data
        self.control_limits = {}
        self.process_means = {}
        
        # Trend detection
        self.trend_indicators = defaultdict(list)
        
    def add_performance_metrics(self, metrics: PerformanceMetrics):
        """Add new performance metrics"""
        
        horizon = metrics.horizon
        
        # Add to history with window limit
        self.performance_history[horizon].append(metrics)
        if len(self.performance_history[horizon]) > self.config.long_window:
            self.performance_history[horizon].popleft()
        
        # Update control limits
        self._update_control_limits(horizon)
        
        # Check for trends
        self._check_for_trends(horizon, metrics)
        
        logger.debug(f"Added performance metrics for horizon {horizon}")
    
    def _update_control_limits(self, horizon: int):
        """Update statistical control limits"""
        
        history = list(self.performance_history[horizon])
        
        if len(history) < 20:  # Need minimum data
            return
        
        # Calculate control limits for key metrics
        metrics_to_track = ['rmse', 'mae', 'coverage_90', 'conformal_coverage']
        
        limits = {}
        means = {}
        
        for metric_name in metrics_to_track:
            values = [getattr(m, metric_name) for m in history if hasattr(m, metric_name)]
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                limits[metric_name] = {
                    'center_line': mean_val,
                    'upper_control_limit': mean_val + self.config.control_chart_sigma * std_val,
                    'lower_control_limit': mean_val - self.config.control_chart_sigma * std_val,
                    'upper_warning_limit': mean_val + 2.0 * std_val,
                    'lower_warning_limit': mean_val - 2.0 * std_val
                }
                
                means[metric_name] = mean_val
        
        self.control_limits[horizon] = limits
        self.process_means[horizon] = means
    
    def _check_for_trends(self, horizon: int, metrics: PerformanceMetrics):
        """Check for trending behavior in metrics"""
        
        history = list(self.performance_history[horizon])
        
        if len(history) < self.config.trending_threshold:
            return
        
        # Check RMSE trend
        recent_rmse = [m.rmse for m in history[-self.config.trending_threshold:]]
        
        # Simple trend detection: consecutive increases or decreases
        increasing = all(recent_rmse[i] <= recent_rmse[i+1] for i in range(len(recent_rmse)-1))
        decreasing = all(recent_rmse[i] >= recent_rmse[i+1] for i in range(len(recent_rmse)-1))
        
        if increasing:
            self.trend_indicators[horizon].append({
                'timestamp': metrics.timestamp,
                'trend_type': 'deteriorating',
                'metric': 'rmse',
                'values': recent_rmse
            })
        elif decreasing:
            self.trend_indicators[horizon].append({
                'timestamp': metrics.timestamp,
                'trend_type': 'improving',
                'metric': 'rmse',
                'values': recent_rmse
            })
        
        # Limit trend history
        if len(self.trend_indicators[horizon]) > 100:
            self.trend_indicators[horizon] = self.trend_indicators[horizon][-50:]
    
    def get_rolling_metrics(self, horizon: int, window: int = None) -> Dict[str, float]:
        """Get rolling performance metrics"""
        
        window = window or self.config.medium_window
        history = list(self.performance_history[horizon])
        
        if not history:
            return {}
        
        recent_history = history[-window:]
        
        if not recent_history:
            return {}
        
        # Calculate rolling metrics
        rolling_metrics = {
            'rmse_mean': np.mean([m.rmse for m in recent_history]),
            'rmse_std': np.std([m.rmse for m in recent_history]),
            'mae_mean': np.mean([m.mae for m in recent_history]),
            'mae_std': np.std([m.mae for m in recent_history]),
            'coverage_90_mean': np.mean([m.coverage_90 for m in recent_history]),
            'coverage_90_std': np.std([m.coverage_90 for m in recent_history]),
            'conformal_coverage_mean': np.mean([m.conformal_coverage for m in recent_history]),
            'sample_count': len(recent_history),
            'time_span_hours': self._calculate_time_span_hours(recent_history)
        }
        
        return rolling_metrics
    
    def _calculate_time_span_hours(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate time span of metrics in hours"""
        
        if len(metrics_list) < 2:
            return 0.0
        
        first_time = datetime.fromisoformat(metrics_list[0].timestamp)
        last_time = datetime.fromisoformat(metrics_list[-1].timestamp)
        
        return (last_time - first_time).total_seconds() / 3600.0
    
    def detect_performance_anomalies(self, horizon: int) -> List[Dict[str, Any]]:
        """Detect performance anomalies using control charts"""
        
        if horizon not in self.control_limits:
            return []
        
        history = list(self.performance_history[horizon])
        if not history:
            return []
        
        latest_metrics = history[-1]
        limits = self.control_limits[horizon]
        
        anomalies = []
        
        # Check each tracked metric
        for metric_name, metric_limits in limits.items():
            current_value = getattr(latest_metrics, metric_name, 0.0)
            
            # Check control limits
            if current_value > metric_limits['upper_control_limit']:
                anomalies.append({
                    'type': 'control_violation',
                    'severity': 'error',
                    'metric': metric_name,
                    'value': current_value,
                    'limit': metric_limits['upper_control_limit'],
                    'direction': 'upper'
                })
            elif current_value < metric_limits['lower_control_limit']:
                anomalies.append({
                    'type': 'control_violation',
                    'severity': 'error',
                    'metric': metric_name,
                    'value': current_value,
                    'limit': metric_limits['lower_control_limit'],
                    'direction': 'lower'
                })
            elif current_value > metric_limits['upper_warning_limit']:
                anomalies.append({
                    'type': 'warning_zone',
                    'severity': 'warning',
                    'metric': metric_name,
                    'value': current_value,
                    'limit': metric_limits['upper_warning_limit'],
                    'direction': 'upper'
                })
            elif current_value < metric_limits['lower_warning_limit']:
                anomalies.append({
                    'type': 'warning_zone',
                    'severity': 'warning',
                    'metric': metric_name,
                    'value': current_value,
                    'limit': metric_limits['lower_warning_limit'],
                    'direction': 'lower'
                })
        
        return anomalies

class CalibrationMonitor:
    """Monitors probability calibration quality"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Store calibration data
        self.calibration_data = defaultdict(list)
        
        # Calibration plots cache
        self.plot_cache = {}
        
    def add_calibration_data(self, horizon: int, predicted_probs: np.ndarray, 
                           actual_outcomes: np.ndarray):
        """Add calibration data for monitoring"""
        
        for prob, outcome in zip(predicted_probs, actual_outcomes):
            self.calibration_data[horizon].append({
                'timestamp': datetime.now().isoformat(),
                'predicted_prob': prob,
                'actual_outcome': outcome
            })
        
        # Limit data size
        max_size = self.config.long_window
        if len(self.calibration_data[horizon]) > max_size:
            self.calibration_data[horizon] = self.calibration_data[horizon][-max_size:]
    
    def calculate_calibration_metrics(self, horizon: int, 
                                    window: int = None) -> Dict[str, float]:
        """Calculate calibration metrics"""
        
        window = window or self.config.medium_window
        data = self.calibration_data[horizon]
        
        if len(data) < 20:
            return {'calibration_available': False}
        
        recent_data = data[-window:]
        
        predicted_probs = np.array([d['predicted_prob'] for d in recent_data])
        actual_outcomes = np.array([d['actual_outcome'] for d in recent_data])
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        reliability_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = actual_outcomes[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                
                ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
                
                reliability_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'count': in_bin.sum()
                })
        
        # Brier score
        brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)
        
        # Reliability (calibration)
        reliability = ece
        
        # Resolution (ability to discriminate)
        resolution = np.var(actual_outcomes)
        
        return {
            'calibration_available': True,
            'expected_calibration_error': ece,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'num_samples': len(recent_data),
            'reliability_data': reliability_data
        }
    
    def generate_calibration_plot(self, horizon: int) -> Optional[str]:
        """Generate calibration plot and return as base64 string"""
        
        try:
            calibration_metrics = self.calculate_calibration_metrics(horizon)
            
            if not calibration_metrics.get('calibration_available', False):
                return None
            
            reliability_data = calibration_metrics['reliability_data']
            
            # Create plot
            plt.style.use(self.config.plot_style)
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Extract data for plotting
            confidences = [d['confidence'] for d in reliability_data]
            accuracies = [d['accuracy'] for d in reliability_data]
            counts = [d['count'] for d in reliability_data]
            
            # Reliability diagram
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            
            # Plot calibration points with size proportional to count
            ax.scatter(confidences, accuracies, s=[c*10 for c in counts], 
                               alpha=0.7, c='blue', label='Calibration points')
            
            # Add ECE text
            ece = calibration_metrics['expected_calibration_error']
            ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Plot - Horizon {horizon} Days')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            # Cache the plot
            self.plot_cache[horizon] = {
                'plot_data': plot_data,
                'timestamp': datetime.now().isoformat(),
                'metrics': calibration_metrics
            }
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Failed to generate calibration plot for horizon {horizon}: {e}")
            return None

class CoverageMonitor:
    """Monitors interval coverage performance"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Coverage data storage
        self.coverage_data = defaultdict(list)
        
        # Target coverage levels
        self.target_coverages = {
            50: 0.50,
            80: 0.80,
            90: 0.90,
            95: 0.95
        }
    
    def add_coverage_data(self, horizon: int, prediction_intervals: Dict[int, Tuple[float, float]], 
                         actual_outcome: float):
        """Add coverage data for monitoring"""
        
        coverage_result = {
            'timestamp': datetime.now().isoformat(),
            'actual_outcome': actual_outcome,
            'intervals': prediction_intervals,
            'coverage': {}
        }
        
        # Check coverage for each confidence level
        for confidence_level, (lower, upper) in prediction_intervals.items():
            covered = lower <= actual_outcome <= upper
            coverage_result['coverage'][confidence_level] = covered
        
        self.coverage_data[horizon].append(coverage_result)
        
        # Limit data size
        max_size = self.config.long_window
        if len(self.coverage_data[horizon]) > max_size:
            self.coverage_data[horizon] = self.coverage_data[horizon][-max_size:]
    
    def calculate_coverage_metrics(self, horizon: int, 
                                 window: int = None) -> Dict[str, float]:
        """Calculate coverage metrics"""
        
        window = window or self.config.medium_window
        data = self.coverage_data[horizon]
        
        if len(data) < 10:
            return {'coverage_available': False}
        
        recent_data = data[-window:]
        
        coverage_metrics = {'coverage_available': True, 'num_samples': len(recent_data)}
        
        # Calculate empirical coverage for each level
        for confidence_level in self.target_coverages:
            if confidence_level in recent_data[0]['coverage']:
                coverage_results = [d['coverage'][confidence_level] for d in recent_data]
                empirical_coverage = np.mean(coverage_results)
                target_coverage = self.target_coverages[confidence_level]
                
                coverage_error = abs(empirical_coverage - target_coverage)
                
                coverage_metrics.update({
                    f'coverage_{confidence_level}_empirical': empirical_coverage,
                    f'coverage_{confidence_level}_target': target_coverage,
                    f'coverage_{confidence_level}_error': coverage_error
                })
        
        return coverage_metrics
    
    def detect_coverage_violations(self, horizon: int) -> List[Dict[str, Any]]:
        """Detect coverage violations"""
        
        coverage_metrics = self.calculate_coverage_metrics(horizon)
        
        if not coverage_metrics.get('coverage_available', False):
            return []
        
        violations = []
        
        for confidence_level in self.target_coverages:
            error_key = f'coverage_{confidence_level}_error'
            if error_key in coverage_metrics:
                error = coverage_metrics[error_key]
                
                if error > self.config.coverage_deviation_threshold:
                    violations.append({
                        'type': 'coverage_violation',
                        'confidence_level': confidence_level,
                        'error': error,
                        'threshold': self.config.coverage_deviation_threshold,
                        'empirical_coverage': coverage_metrics[f'coverage_{confidence_level}_empirical'],
                        'target_coverage': coverage_metrics[f'coverage_{confidence_level}_target']
                    })
        
        return violations

class RegimeMonitor:
    """Monitors market regime changes and their impact"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Market data storage
        self.market_data = deque(maxlen=config.regime_detection_window * 2)
        
        # Regime indicators
        self.current_regime = "normal"
        self.regime_history = []
        
        # Performance by regime
        self.regime_performance = defaultdict(list)
    
    def add_market_data(self, timestamp: str, price_change: float, 
                       volatility: float, volume: Optional[float] = None):
        """Add market data for regime analysis"""
        
        market_observation = {
            'timestamp': timestamp,
            'price_change': price_change,
            'volatility': volatility,
            'volume': volume
        }
        
        self.market_data.append(market_observation)
        
        # Detect regime change
        self._detect_regime_change()
    
    def _detect_regime_change(self):
        """Detect market regime changes"""
        
        if len(self.market_data) < self.config.regime_detection_window:
            return
        
        recent_data = list(self.market_data)[-self.config.regime_detection_window:]
        
        # Calculate regime indicators
        avg_volatility = np.mean([d['volatility'] for d in recent_data])
        volatility_std = np.std([d['volatility'] for d in recent_data])
        
        # Simple regime classification
        if avg_volatility > volatility_std * 2:
            new_regime = "high_volatility"
        elif avg_volatility < volatility_std * 0.5:
            new_regime = "low_volatility"
        else:
            new_regime = "normal"
        
        # Check for regime change
        if new_regime != self.current_regime:
            regime_change = {
                'timestamp': datetime.now().isoformat(),
                'previous_regime': self.current_regime,
                'new_regime': new_regime,
                'avg_volatility': avg_volatility,
                'volatility_threshold': volatility_std
            }
            
            self.regime_history.append(regime_change)
            self.current_regime = new_regime
            
            logger.info(f"Regime change detected: {self.current_regime}")
    
    def add_regime_performance(self, metrics: PerformanceMetrics):
        """Add performance metrics categorized by current regime"""
        
        regime_perf = {
            'timestamp': metrics.timestamp,
            'regime': self.current_regime,
            'horizon': metrics.horizon,
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'coverage_90': metrics.coverage_90
        }
        
        self.regime_performance[self.current_regime].append(regime_perf)
        
        # Limit storage
        max_regime_history = 500
        for regime in self.regime_performance:
            if len(self.regime_performance[regime]) > max_regime_history:
                self.regime_performance[regime] = self.regime_performance[regime][-max_regime_history:]
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get comprehensive regime analysis"""
        
        analysis = {
            'current_regime': self.current_regime,
            'regime_changes_24h': len([r for r in self.regime_history 
                                     if datetime.fromisoformat(r['timestamp']) > 
                                     datetime.now() - timedelta(hours=24)]),
            'performance_by_regime': {}
        }
        
        # Calculate performance statistics by regime
        for regime, perf_data in self.regime_performance.items():
            if perf_data:
                rmse_values = [p['rmse'] for p in perf_data]
                mae_values = [p['mae'] for p in perf_data]
                coverage_values = [p['coverage_90'] for p in perf_data]
                
                analysis['performance_by_regime'][regime] = {
                    'rmse_mean': np.mean(rmse_values),
                    'rmse_std': np.std(rmse_values),
                    'mae_mean': np.mean(mae_values),
                    'mae_std': np.std(mae_values),
                    'coverage_90_mean': np.mean(coverage_values),
                    'coverage_90_std': np.std(coverage_values),
                    'sample_count': len(perf_data)
                }
        
        return analysis

class AlertManager:
    """Manages monitoring alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Alert storage
        self.active_alerts = []
        self.alert_history = []
        
        # Alert suppression
        self.suppressed_alerts = set()
        
    def create_alert(self, alert_type: str, severity: str, metric_name: str,
                    current_value: float, threshold_value: float,
                    horizon: Optional[int] = None, message: str = "") -> AlertEvent:
        """Create a new alert"""
        
        alert = AlertEvent(
            timestamp=datetime.now().isoformat(),
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            horizon=horizon,
            message=message or self._generate_alert_message(
                alert_type, metric_name, current_value, threshold_value, horizon
            )
        )
        
        # Check if similar alert is suppressed
        alert_key = f"{alert_type}_{metric_name}_{horizon}"
        if alert_key not in self.suppressed_alerts:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Log alert
            log_func = {
                'info': logger.info,
                'warning': logger.warning,
                'error': logger.error,
                'critical': logger.critical
            }.get(severity, logger.info)
            
            log_func(f"Alert: {alert.message}")
            
            # Suppress similar alerts for a period
            if severity in ['warning', 'error', 'critical']:
                self.suppressed_alerts.add(alert_key)
        
        return alert
    
    def _generate_alert_message(self, alert_type: str, metric_name: str,
                               current_value: float, threshold_value: float,
                               horizon: Optional[int]) -> str:
        """Generate human-readable alert message"""
        
        horizon_str = f" (horizon {horizon})" if horizon else ""
        
        if alert_type == "performance_degradation":
            return (f"Performance degradation detected: {metric_name}{horizon_str} "
                   f"= {current_value:.4f} exceeds threshold {threshold_value:.4f}")
        elif alert_type == "coverage_violation":
            return (f"Coverage violation: {metric_name}{horizon_str} "
                   f"= {current_value:.4f} deviates from target by {abs(current_value - threshold_value):.4f}")
        elif alert_type == "calibration_error":
            return (f"Calibration error: {metric_name}{horizon_str} "
                   f"= {current_value:.4f} exceeds threshold {threshold_value:.4f}")
        else:
            return (f"Alert: {metric_name}{horizon_str} "
                   f"= {current_value:.4f} vs threshold {threshold_value:.4f}")
    
    def resolve_alert(self, alert: AlertEvent, resolution_message: str = ""):
        """Resolve an active alert"""
        
        if alert in self.active_alerts:
            alert.resolved = True
            alert.resolution_time = datetime.now().isoformat()
            
            if resolution_message:
                alert.message += f" [Resolved: {resolution_message}]"
            
            self.active_alerts.remove(alert)
            
            logger.info(f"Alert resolved: {alert.alert_type} for {alert.metric_name}")
    
    def clear_suppressions(self):
        """Clear alert suppressions (called periodically)"""
        
        self.suppressed_alerts.clear()
        logger.debug("Alert suppressions cleared")
    
    def get_active_alerts(self, severity_filter: Optional[List[str]] = None) -> List[AlertEvent]:
        """Get current active alerts"""
        
        if severity_filter:
            return [alert for alert in self.active_alerts if alert.severity in severity_filter]
        return self.active_alerts.copy()
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        ]
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert.alert_type] += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': dict(severity_counts),
            'alerts_by_type': dict(type_counts),
            'recent_critical': [
                alert for alert in recent_alerts if alert.severity == 'critical'
            ][-5:]  # Last 5 critical alerts
        }

class ProductionMonitoringSystem:
    """Main monitoring system coordinating all monitoring components"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # Initialize monitoring components
        self.performance_tracker = PerformanceTracker(self.config)
        self.calibration_monitor = CalibrationMonitor(self.config)
        self.coverage_monitor = CoverageMonitor(self.config)
        self.regime_monitor = RegimeMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.is_monitoring = False
        self.last_update_time = None
        
        # Database for persistent storage
        self.db_path = "monitoring.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                rmse REAL,
                mae REAL,
                r2_score REAL,
                coverage_90 REAL,
                conformal_coverage REAL,
                market_volatility REAL,
                regime_indicator TEXT
            )
        """)
        
        # Alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL,
                threshold_value REAL,
                horizon INTEGER,
                message TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_time TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """Start the monitoring system"""
        
        self.is_monitoring = True
        self.last_update_time = datetime.now()
        
        logger.info("Production monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        
        self.is_monitoring = False
        
        logger.info("Production monitoring system stopped")
    
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """Update performance metrics across all monitoring components"""
        
        if not self.is_monitoring:
            return
        
        # Update performance tracker
        self.performance_tracker.add_performance_metrics(metrics)
        
        # Update regime monitor
        self.regime_monitor.add_regime_performance(metrics)
        
        # Store in database
        self._store_performance_metrics(metrics)
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        self.last_update_time = datetime.now()
    
    def update_calibration_data(self, horizon: int, predicted_probs: np.ndarray,
                               actual_outcomes: np.ndarray):
        """Update calibration monitoring data"""
        
        if not self.is_monitoring:
            return
        
        self.calibration_monitor.add_calibration_data(horizon, predicted_probs, actual_outcomes)
        
        # Check calibration quality
        self._check_calibration_alerts(horizon)
    
    def update_coverage_data(self, horizon: int, 
                           prediction_intervals: Dict[int, Tuple[float, float]],
                           actual_outcome: float):
        """Update coverage monitoring data"""
        
        if not self.is_monitoring:
            return
        
        self.coverage_monitor.add_coverage_data(horizon, prediction_intervals, actual_outcome)
        
        # Check coverage violations
        self._check_coverage_alerts(horizon)
    
    def update_market_data(self, timestamp: str, price_change: float,
                          volatility: float, volume: Optional[float] = None):
        """Update market data for regime monitoring"""
        
        if not self.is_monitoring:
            return
        
        self.regime_monitor.add_market_data(timestamp, price_change, volatility, volume)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance-related alerts"""
        
        horizon = metrics.horizon
        
        # Get rolling metrics for comparison
        rolling_metrics = self.performance_tracker.get_rolling_metrics(horizon, self.config.short_window)
        
        if not rolling_metrics:
            return
        
        # Check RMSE degradation
        baseline_rmse = rolling_metrics.get('rmse_mean', metrics.rmse)
        rmse_increase = (metrics.rmse - baseline_rmse) / baseline_rmse if baseline_rmse > 0 else 0
        
        if rmse_increase > self.config.rmse_degradation_threshold:
            self.alert_manager.create_alert(
                alert_type="performance_degradation",
                severity="warning" if rmse_increase < 0.25 else "error",
                metric_name="rmse",
                current_value=metrics.rmse,
                threshold_value=baseline_rmse * (1 + self.config.rmse_degradation_threshold),
                horizon=horizon
            )
        
        # Check coverage deviation
        target_coverage = 0.90
        coverage_deviation = abs(metrics.coverage_90 - target_coverage)
        
        if coverage_deviation > self.config.coverage_deviation_threshold:
            self.alert_manager.create_alert(
                alert_type="coverage_violation",
                severity="warning",
                metric_name="coverage_90",
                current_value=metrics.coverage_90,
                threshold_value=target_coverage,
                horizon=horizon
            )
        
        # Check control chart violations
        anomalies = self.performance_tracker.detect_performance_anomalies(horizon)
        for anomaly in anomalies:
            self.alert_manager.create_alert(
                alert_type=anomaly['type'],
                severity=anomaly['severity'],
                metric_name=anomaly['metric'],
                current_value=anomaly['value'],
                threshold_value=anomaly['limit'],
                horizon=horizon
            )
    
    def _check_calibration_alerts(self, horizon: int):
        """Check for calibration-related alerts"""
        
        calibration_metrics = self.calibration_monitor.calculate_calibration_metrics(horizon)
        
        if not calibration_metrics.get('calibration_available', False):
            return
        
        ece = calibration_metrics.get('expected_calibration_error', 0.0)
        
        if ece > self.config.calibration_error_threshold:
            self.alert_manager.create_alert(
                alert_type="calibration_error",
                severity="warning" if ece < 0.15 else "error",
                metric_name="expected_calibration_error",
                current_value=ece,
                threshold_value=self.config.calibration_error_threshold,
                horizon=horizon
            )
    
    def _check_coverage_alerts(self, horizon: int):
        """Check for coverage-related alerts"""
        
        violations = self.coverage_monitor.detect_coverage_violations(horizon)
        
        for violation in violations:
            self.alert_manager.create_alert(
                alert_type="coverage_violation",
                severity="warning",
                metric_name=f"coverage_{violation['confidence_level']}",
                current_value=violation['empirical_coverage'],
                threshold_value=violation['target_coverage'],
                horizon=horizon
            )
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO performance_metrics 
                (timestamp, horizon, rmse, mae, r2_score, coverage_90, conformal_coverage, 
                 market_volatility, regime_indicator)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.horizon,
                metrics.rmse,
                metrics.mae,
                metrics.r2_score,
                metrics.coverage_90,
                metrics.conformal_coverage,
                metrics.market_volatility,
                metrics.regime_indicator
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        
        dashboard_data = {
            'system_status': {
                'is_monitoring': self.is_monitoring,
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'current_regime': self.regime_monitor.current_regime
            },
            'alerts': self.alert_manager.get_alert_summary(),
            'performance': {},
            'calibration': {},
            'coverage': {},
            'regime_analysis': self.regime_monitor.get_regime_analysis()
        }
        
        # Get performance data for each horizon
        for horizon in [1, 3, 5, 10, 20]:  # Standard horizons
            rolling_perf = self.performance_tracker.get_rolling_metrics(horizon)
            if rolling_perf:
                dashboard_data['performance'][f'horizon_{horizon}'] = rolling_perf
            
            # Calibration data
            cal_metrics = self.calibration_monitor.calculate_calibration_metrics(horizon)
            if cal_metrics.get('calibration_available', False):
                dashboard_data['calibration'][f'horizon_{horizon}'] = cal_metrics
            
            # Coverage data
            cov_metrics = self.coverage_monitor.calculate_coverage_metrics(horizon)
            if cov_metrics.get('coverage_available', False):
                dashboard_data['coverage'][f'horizon_{horizon}'] = cov_metrics
        
        return dashboard_data
    
    def generate_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        report = {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_hours': hours
            },
            'executive_summary': {},
            'detailed_metrics': {},
            'alerts_summary': self.alert_manager.get_alert_summary(hours),
            'recommendations': []
        }
        
        # Generate executive summary
        total_alerts = report['alerts_summary']['total_alerts']
        critical_alerts = len(report['alerts_summary']['alerts_by_severity'].get('critical', []))
        
        if critical_alerts > 0:
            status = "Critical issues detected"
        elif total_alerts > 10:
            status = "Multiple issues detected"
        elif total_alerts > 0:
            status = "Minor issues detected"
        else:
            status = "System operating normally"
        
        report['executive_summary'] = {
            'status': status,
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'monitoring_uptime': self.is_monitoring
        }
        
        # Add recommendations based on alerts
        if critical_alerts > 0:
            report['recommendations'].append("Immediate attention required for critical alerts")
        
        if report['alerts_summary']['alerts_by_type'].get('performance_degradation', 0) > 5:
            report['recommendations'].append("Consider model retraining due to performance degradation")
        
        if report['alerts_summary']['alerts_by_type'].get('coverage_violation', 0) > 3:
            report['recommendations'].append("Review and recalibrate prediction intervals")
        
        return report
    
    def cleanup_old_data(self):
        """Clean up old monitoring data based on retention policy"""
        
        cutoff_time = datetime.now() - timedelta(days=self.config.db_retention_days)
        cutoff_str = cutoff_time.isoformat()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Clean performance metrics
            conn.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_str,))
            
            # Clean resolved alerts older than retention period
            conn.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", (cutoff_str,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up monitoring data older than {self.config.db_retention_days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old monitoring data: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform monitoring system health check"""
        
        health_status = {
            'overall_health': 'healthy',
            'components': {},
            'issues': []
        }
        
        # Check each component
        components = {
            'performance_tracker': self.performance_tracker,
            'calibration_monitor': self.calibration_monitor,
            'coverage_monitor': self.coverage_monitor,
            'regime_monitor': self.regime_monitor,
            'alert_manager': self.alert_manager
        }
        
        for name, component in components.items():
            try:
                # Basic functionality check
                if hasattr(component, 'config'):
                    health_status['components'][name] = 'healthy'
                else:
                    health_status['components'][name] = 'unknown'
            except Exception as e:
                health_status['components'][name] = 'unhealthy'
                health_status['issues'].append(f"{name}: {str(e)}")
        
        # Check database connectivity
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT COUNT(*) FROM performance_metrics")
            conn.close()
            health_status['components']['database'] = 'healthy'
        except Exception as e:
            health_status['components']['database'] = 'unhealthy'
            health_status['issues'].append(f"database: {str(e)}")
        
        # Set overall health
        unhealthy_components = [k for k, v in health_status['components'].items() if v == 'unhealthy']
        if unhealthy_components:
            health_status['overall_health'] = 'degraded' if len(unhealthy_components) < 3 else 'unhealthy'
        
        return health_status