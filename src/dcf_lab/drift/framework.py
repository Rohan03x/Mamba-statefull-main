"""
Concept Drift Detection & Adaptive Learning Framework

This module provides the main framework that integrates all drift detection
and adaptive learning components into a cohesive system for financial ML.

Key Components:
1. ConceptDriftFramework: Main orchestrator
2. DriftConfig: Configuration management
3. AdaptiveMLPipeline: Complete ML pipeline with drift adaptation
4. DriftValidator: Validation and quality assurance

The framework automatically detects concept drift and applies appropriate
adaptive learning strategies for production financial ML systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

# Import drift components
from .monitors import (
    ADWINMonitor, PageHinkleyMonitor, FeatureDriftMonitor, 
    ResidualDriftMonitor, DriftAlert, DriftSeverity
)
from .adaptive import (
    LightRetrainer, OnlineUpdater, 
    RegimeResetter, AdaptationStrategy, AdaptationResult
)
from .streaming import (
    StreamingFramework, StreamingConfig, IncrementalUpdater, 
    StreamingMode
)

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for concept drift detection and adaptation"""
    
    # Monitor settings
    enable_adwin: bool = True
    enable_page_hinkley: bool = True
    enable_feature_drift: bool = True
    enable_residual_drift: bool = True
    
    # ADWIN settings
    adwin_delta: float = 0.002
    adwin_min_window: int = 30
    adwin_max_window: int = 1000
    
    # Page-Hinkley settings
    ph_threshold: float = 50.0
    ph_alpha: float = 0.9999
    
    # Feature drift settings
    feature_correlation_threshold: float = 0.3
    feature_distribution_threshold: float = 0.1
    feature_window_size: int = 100
    
    # Residual drift settings
    residual_window_size: int = 100
    residual_accuracy_threshold: float = 0.1
    residual_bias_threshold: float = 0.05
    
    # Adaptation settings
    enable_light_retrain: bool = True
    enable_online_updates: bool = True
    enable_regime_reset: bool = True
    
    light_retrain_epochs: int = 5
    light_retrain_min_samples: int = 100
    
    online_update_frequency: int = 10
    online_buffer_size: int = 100
    
    regime_reset_threshold: str = "major"
    regime_reset_cooldown: int = 3600
    
    # Streaming settings
    streaming_mode: StreamingMode = StreamingMode.MICRO_BATCH
    streaming_batch_size: int = 10
    streaming_buffer_size: int = 1000
    
    # Performance settings
    performance_monitoring: bool = True
    validation_frequency: int = 100
    alert_threshold: DriftSeverity = DriftSeverity.MODERATE


class ConceptDriftFramework:
    """
    Main framework for concept drift detection and adaptive learning
    
    Orchestrates all drift monitors and adaptive learning components
    to provide automatic drift detection and model adaptation.
    """
    
    def __init__(self, config: DriftConfig):
        """
        Initialize concept drift framework
        
        Args:
            config: Drift detection and adaptation configuration
        """
        self.config = config
        
        # Initialize monitors
        self.monitors = {}
        self._initialize_monitors()
        
        # Initialize adaptive learners
        self.adaptive_learners = {}
        self._initialize_adaptive_learners()
        
        # Initialize streaming if enabled
        self.streaming_framework = None
        self.incremental_updater = None
        self._initialize_streaming()
        
        # Framework state
        self.is_active = False
        self.registered_models = {}
        self.drift_history = []
        self.adaptation_history = []
        self.performance_history = []
        
        # Statistics
        self.total_samples_processed = 0
        self.total_drift_alerts = 0
        self.total_adaptations = 0
        
        logger.info("🎯 Initialized ConceptDriftFramework")
        logger.info(f"   Monitors: {list(self.monitors.keys())}")
        logger.info(f"   Adaptive learners: {list(self.adaptive_learners.keys())}")
    
    def _initialize_monitors(self):
        """Initialize drift monitors based on configuration"""
        
        if self.config.enable_adwin:
            self.monitors['adwin'] = ADWINMonitor(
                delta=self.config.adwin_delta,
                min_window_size=self.config.adwin_min_window,
                max_window_size=self.config.adwin_max_window
            )
        
        if self.config.enable_page_hinkley:
            self.monitors['page_hinkley'] = PageHinkleyMonitor(
                threshold=self.config.ph_threshold,
                alpha=self.config.ph_alpha
            )
        
        if self.config.enable_residual_drift:
            self.monitors['residual'] = ResidualDriftMonitor(
                window_size=self.config.residual_window_size,
                accuracy_threshold=self.config.residual_accuracy_threshold,
                bias_threshold=self.config.residual_bias_threshold
            )
    
    def _initialize_adaptive_learners(self):
        """Initialize adaptive learning components"""
        
        if self.config.enable_light_retrain:
            self.adaptive_learners['light_retrain'] = LightRetrainer(
                max_epochs=self.config.light_retrain_epochs,
                min_samples=self.config.light_retrain_min_samples
            )
        
        if self.config.enable_online_updates:
            self.adaptive_learners['online_update'] = OnlineUpdater(
                update_frequency=self.config.online_update_frequency,
                buffer_size=self.config.online_buffer_size
            )
        
        if self.config.enable_regime_reset:
            self.adaptive_learners['regime_reset'] = RegimeResetter(
                reset_threshold_severity=self.config.regime_reset_threshold,
                cooldown_period=self.config.regime_reset_cooldown
            )
    
    def _initialize_streaming(self):
        """Initialize streaming framework if needed"""
        
        streaming_config = StreamingConfig(
            mode=self.config.streaming_mode,
            batch_size=self.config.streaming_batch_size,
            max_buffer_size=self.config.streaming_buffer_size,
            performance_tracking=self.config.performance_monitoring
        )
        
        self.streaming_framework = StreamingFramework(streaming_config)
        self.incremental_updater = IncrementalUpdater(self.streaming_framework)
    
    def register_model(self, model_name: str, model: Any, feature_names: List[str]):
        """
        Register a model for drift monitoring and adaptation
        
        Args:
            model_name: Unique identifier for the model
            model: Model object
            feature_names: List of feature names
        """
        self.registered_models[model_name] = {
            'model': model,
            'feature_names': feature_names,
            'last_update': pd.Timestamp.now(),
            'drift_alerts': [],
            'adaptations': [],
            'performance_metrics': []
        }
        
        # Initialize feature drift monitor for this model
        if self.config.enable_feature_drift:
            monitor_key = f'feature_drift_{model_name}'
            self.monitors[monitor_key] = FeatureDriftMonitor(
                feature_names=feature_names,
                correlation_threshold=self.config.feature_correlation_threshold,
                distribution_threshold=self.config.feature_distribution_threshold,
                window_size=self.config.feature_window_size
            )
        
        # Register with streaming framework if available
        if self.incremental_updater:
            self.incremental_updater.register_model(model_name, model)
        
        logger.info(f"📝 Registered model '{model_name}' with {len(feature_names)} features")
    
    def start_monitoring(self):
        """Start drift monitoring"""
        self.is_active = True
        if self.streaming_framework:
            self.streaming_framework.start_stream()
        logger.info("🚀 Drift monitoring started")
    
    def stop_monitoring(self):
        """Stop drift monitoring"""
        self.is_active = False
        if self.streaming_framework:
            self.streaming_framework.stop_stream()
        logger.info("⏹️ Drift monitoring stopped")
    
    def process_sample(self, 
                      model_name: str,
                      features: Dict[str, float],
                      target: Optional[float] = None,
                      prediction: Optional[float] = None,
                      timestamp: Optional[pd.Timestamp] = None) -> List[DriftAlert]:
        """
        Process a new sample and check for drift
        
        Args:
            model_name: Name of the model
            features: Feature values
            target: True target value (if available)
            prediction: Model prediction (if available)
            timestamp: Sample timestamp
            
        Returns:
            List of drift alerts
        """
        if not self.is_active:
            return []
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        alerts = []
        self.total_samples_processed += 1
        
        # Process with general monitors (ADWIN, Page-Hinkley)
        if target is not None:
            # ADWIN monitoring
            if 'adwin' in self.monitors:
                adwin_alert = self.monitors['adwin'].add_element(target, timestamp)
                if adwin_alert:
                    alerts.append(adwin_alert)
            
            # Page-Hinkley monitoring
            if 'page_hinkley' in self.monitors:
                ph_alert = self.monitors['page_hinkley'].add_element(target, timestamp)
                if ph_alert:
                    alerts.append(ph_alert)
        
        # Process with model-specific monitors
        if model_name in self.registered_models:
            # Feature drift monitoring
            feature_monitor_key = f'feature_drift_{model_name}'
            if feature_monitor_key in self.monitors:
                feature_alerts = self.monitors[feature_monitor_key].add_features(features, timestamp)
                alerts.extend(feature_alerts)
            
            # Residual drift monitoring
            if 'residual' in self.monitors and target is not None and prediction is not None:
                residual_alert = self.monitors['residual'].add_prediction(prediction, target, timestamp)
                if residual_alert:
                    alerts.append(residual_alert)
        
        # Process alerts and trigger adaptations
        if alerts:
            self._process_alerts(alerts, model_name, features, target, timestamp)
        
        # Add to streaming framework if available
        if self.streaming_framework and target is not None:
            self.streaming_framework.add_sample(features, target, timestamp)
        
        return alerts
    
    def _process_alerts(self, 
                       alerts: List[DriftAlert], 
                       model_name: str,
                       features: Dict[str, float],
                       target: Optional[float],
                       timestamp: pd.Timestamp):
        """Process drift alerts and trigger adaptations"""
        
        for alert in alerts:
            # Store alert
            self.drift_history.append(alert)
            self.total_drift_alerts += 1
            
            if model_name in self.registered_models:
                self.registered_models[model_name]['drift_alerts'].append(alert)
            
            # Check if adaptation is needed
            if alert.severity.value in ['moderate', 'major', 'critical']:
                self._trigger_adaptation(alert, model_name, timestamp)
    
    def _trigger_adaptation(self, alert: DriftAlert, model_name: str, timestamp: pd.Timestamp):
        """Trigger appropriate adaptation based on drift alert"""
        
        if model_name not in self.registered_models:
            return
        
        model_info = self.registered_models[model_name]
        model = model_info['model']
        
        # Prepare drift information
        drift_info = {
            'severity': alert.severity.value,
            'monitor_type': alert.monitor_type,
            'metric_value': alert.metric_value,
            'timestamp': timestamp,
            'alert': alert
        }
        
        # Get recent data for adaptation
        adaptation_data = self._prepare_adaptation_data(model_name)
        
        # Try each adaptive learner in order of preference
        adaptation_results = []
        
        for learner_name, learner in self.adaptive_learners.items():
            if learner.should_adapt(drift_info):
                try:
                    result = learner.adapt(model, adaptation_data, drift_info)
                    adaptation_results.append(result)
                    
                    if result.success:
                        logger.info(f"✅ Successful adaptation: {learner_name} for {model_name}")
                        self.total_adaptations += 1
                        model_info['adaptations'].append(result)
                        break  # Stop after successful adaptation
                    else:
                        logger.warning(f"⚠️ Failed adaptation: {learner_name} for {model_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Adaptation error ({learner_name}): {e}")
        
        # Store adaptation history
        self.adaptation_history.extend(adaptation_results)
    
    def _prepare_adaptation_data(self, model_name: str) -> Dict[str, Any]:
        """Prepare data for model adaptation"""
        
        # Get data from streaming framework if available
        if self.streaming_framework:
            return self.streaming_framework.get_buffer_data(n_samples=500)
        
        # Fallback: return empty data structure
        return {
            'X': np.array([]),
            'y': np.array([]),
            'timestamps': []
        }
    
    def force_adaptation(self, model_name: str, strategy: AdaptationStrategy) -> Optional[AdaptationResult]:
        """
        Force adaptation with specific strategy
        
        Args:
            model_name: Name of the model to adapt
            strategy: Adaptation strategy to use
            
        Returns:
            AdaptationResult or None if failed
        """
        if model_name not in self.registered_models:
            logger.error(f"Model '{model_name}' not registered")
            return None
        
        # Find appropriate adaptive learner
        learner = None
        if strategy == AdaptationStrategy.LIGHT_RETRAIN and 'light_retrain' in self.adaptive_learners:
            learner = self.adaptive_learners['light_retrain']
        elif strategy == AdaptationStrategy.INCREMENTAL_UPDATE and 'online_update' in self.adaptive_learners:
            learner = self.adaptive_learners['online_update']
        elif strategy == AdaptationStrategy.REGIME_RESET and 'regime_reset' in self.adaptive_learners:
            learner = self.adaptive_learners['regime_reset']
        
        if not learner:
            logger.error(f"No learner available for strategy {strategy}")
            return None
        
        # Prepare adaptation data
        model = self.registered_models[model_name]['model']
        adaptation_data = self._prepare_adaptation_data(model_name)
        drift_info = {
            'severity': 'major',
            'monitor_type': 'manual',
            'timestamp': pd.Timestamp.now()
        }
        
        try:
            result = learner.adapt(model, adaptation_data, drift_info)
            if result.success:
                self.total_adaptations += 1
                self.registered_models[model_name]['adaptations'].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Forced adaptation failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics"""
        
        # Monitor statistics
        monitor_stats = {}
        for name, monitor in self.monitors.items():
            if hasattr(monitor, 'get_statistics'):
                monitor_stats[name] = monitor.get_statistics()
        
        # Model statistics
        model_stats = {}
        for model_name, model_info in self.registered_models.items():
            model_stats[model_name] = {
                'total_alerts': len(model_info['drift_alerts']),
                'total_adaptations': len(model_info['adaptations']),
                'last_update': model_info['last_update']
            }
        
        # Recent performance
        recent_alerts = [a for a in self.drift_history if 
                        (pd.Timestamp.now() - a.timestamp).total_seconds() < 3600]
        
        return {
            'framework_active': self.is_active,
            'total_samples_processed': self.total_samples_processed,
            'total_drift_alerts': self.total_drift_alerts,
            'total_adaptations': self.total_adaptations,
            'recent_alerts_1h': len(recent_alerts),
            'registered_models': list(self.registered_models.keys()),
            'monitors_active': list(self.monitors.keys()),
            'adaptive_learners': list(self.adaptive_learners.keys()),
            'monitor_statistics': monitor_stats,
            'model_statistics': model_stats,
            'streaming_active': self.streaming_framework.is_running if self.streaming_framework else False
        }


class AdaptiveMLPipeline:
    """
    Complete ML pipeline with integrated drift detection and adaptation
    
    Provides a high-level interface for building adaptive ML systems
    """
    
    def __init__(self, config: DriftConfig):
        """Initialize adaptive ML pipeline"""
        self.config = config
        self.drift_framework = ConceptDriftFramework(config)
        self.pipeline_models = {}
        self.feature_transformers = {}
        
        logger.info("🚀 Initialized AdaptiveMLPipeline")
    
    def add_model(self, name: str, model: Any, feature_names: List[str]):
        """Add a model to the adaptive pipeline"""
        self.pipeline_models[name] = model
        self.drift_framework.register_model(name, model, feature_names)
        logger.info(f"➕ Added model '{name}' to adaptive pipeline")
    
    def start_pipeline(self):
        """Start the adaptive ML pipeline"""
        self.drift_framework.start_monitoring()
        logger.info("▶️ Adaptive ML pipeline started")
    
    def stop_pipeline(self):
        """Stop the adaptive ML pipeline"""
        self.drift_framework.stop_monitoring()
        logger.info("⏹️ Adaptive ML pipeline stopped")
    
    def predict_with_monitoring(self, 
                               model_name: str,
                               features: Dict[str, float],
                               true_target: Optional[float] = None,
                               timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Make prediction with drift monitoring
        
        Args:
            model_name: Name of the model
            features: Feature values
            true_target: True target (for monitoring)
            timestamp: Optional timestamp
            
        Returns:
            Prediction results with drift information
        """
        if model_name not in self.pipeline_models:
            raise ValueError(f"Model '{model_name}' not found in pipeline")
        
        model = self.pipeline_models[model_name]
        
        # Make prediction
        feature_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        
        # Monitor for drift
        drift_alerts = self.drift_framework.process_sample(
            model_name=model_name,
            features=features,
            target=true_target,
            prediction=prediction,
            timestamp=timestamp
        )
        
        return {
            'prediction': prediction,
            'model_name': model_name,
            'timestamp': timestamp or pd.Timestamp.now(),
            'drift_alerts': drift_alerts,
            'alert_count': len(drift_alerts),
            'max_severity': max([alert.severity for alert in drift_alerts], 
                               default=DriftSeverity.NONE)
        }


class DriftValidator:
    """
    Validation and quality assurance for drift detection framework
    """
    
    def __init__(self, framework: ConceptDriftFramework):
        """Initialize drift validator"""
        self.framework = framework
        self.validation_history = []
        
        logger.info("✅ Initialized DriftValidator")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate framework configuration"""
        
        issues = []
        warnings = []
        
        config = self.framework.config
        
        # Check monitor settings
        if not any([config.enable_adwin, config.enable_page_hinkley, 
                   config.enable_feature_drift, config.enable_residual_drift]):
            issues.append("No drift monitors enabled")
        
        # Check adaptation settings
        if not any([config.enable_light_retrain, config.enable_online_updates,
                   config.enable_regime_reset]):
            warnings.append("No adaptive learners enabled")
        
        # Check thresholds
        if config.adwin_delta <= 0 or config.adwin_delta >= 1:
            issues.append(f"Invalid ADWIN delta: {config.adwin_delta}")
        
        if config.ph_threshold <= 0:
            issues.append(f"Invalid Page-Hinkley threshold: {config.ph_threshold}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'timestamp': pd.Timestamp.now()
        }
    
    def validate_framework_health(self) -> Dict[str, Any]:
        """Validate framework operational health"""
        
        stats = self.framework.get_statistics()
        health_issues = []
        
        # Check if framework is active
        if not stats['framework_active']:
            health_issues.append("Framework not active")
        
        # Check monitor responsiveness
        if stats['total_samples_processed'] > 1000 and stats['total_drift_alerts'] == 0:
            health_issues.append("No drift alerts despite high sample volume - monitors may be insensitive")
        
        # Check adaptation effectiveness
        if stats['total_drift_alerts'] > 0 and stats['total_adaptations'] == 0:
            health_issues.append("Drift detected but no adaptations performed")
        
        # Check recent activity
        if stats['recent_alerts_1h'] > 50:
            health_issues.append("Excessive drift alerts in past hour - may indicate system instability")
        
        return {
            'healthy': len(health_issues) == 0,
            'issues': health_issues,
            'statistics': stats,
            'timestamp': pd.Timestamp.now()
        }