"""
Adaptive Learning Components for Concept Drift Response

This module implements adaptive learning strategies that respond to concept drift
detection by automatically adjusting models and learning parameters:

1. Light Retrainer: Performs minimal retraining with recency weighting
2. Online Updater: Incremental model updates for streaming data
3. Regime Resetter: Hard reset of regime/HMM models for major drift
4. Recency Weighter: Exponential decay weighting for temporal importance

All components integrate with drift monitors to provide automatic adaptive
responses for financial time series modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Types of adaptation strategies"""
    LIGHT_RETRAIN = "light_retrain"
    INCREMENTAL_UPDATE = "incremental_update"
    REGIME_RESET = "regime_reset"
    WEIGHT_ADJUSTMENT = "weight_adjustment"
    FULL_RETRAIN = "full_retrain"


@dataclass
class AdaptationResult:
    """Result of an adaptation action"""
    strategy: AdaptationStrategy
    success: bool
    execution_time: float
    performance_before: float
    performance_after: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class AdaptiveLearner(ABC):
    """Base class for adaptive learning components"""
    
    @abstractmethod
    def adapt(self, model: Any, data: Dict[str, Any], drift_info: Dict[str, Any]) -> AdaptationResult:
        """Adapt model based on drift information"""
        pass
    
    @abstractmethod
    def should_adapt(self, drift_info: Dict[str, Any]) -> bool:
        """Determine if adaptation is needed"""
        pass


class RecencyWeighter:
    """
    Exponential decay weighting for temporal importance.
    
    Assigns higher weights to recent observations, with configurable
    decay rates for different types of drift.
    """
    
    def __init__(self,
                 base_decay: float = 0.95,
                 drift_decay: float = 0.85,
                 min_weight: float = 0.01,
                 window_size: int = 1000):
        """
        Initialize recency weighter
        
        Args:
            base_decay: Normal decay rate (no drift)
            drift_decay: Accelerated decay rate during drift
            min_weight: Minimum weight to assign
            window_size: Maximum number of samples to weight
        """
        self.base_decay = base_decay
        self.drift_decay = drift_decay
        self.min_weight = min_weight
        self.window_size = window_size
        
        # State tracking
        self.drift_detected = False
        self.drift_start_time = None
        
        logger.info(f"⚖️ Initialized recency weighter: decay={base_decay}/{drift_decay}, window={window_size}")
    
    def set_drift_state(self, drift_detected: bool, timestamp: Optional[pd.Timestamp] = None):
        """Update drift state for adaptive weighting"""
        
        if drift_detected and not self.drift_detected:
            self.drift_start_time = timestamp or pd.Timestamp.now()
            logger.info(f"📉 Recency weighter: drift mode activated at {self.drift_start_time}")
        elif not drift_detected and self.drift_detected:
            logger.info("📈 Recency weighter: drift mode deactivated")
        
        self.drift_detected = drift_detected
    
    def calculate_weights(self, n_samples: int, timestamps: Optional[List[pd.Timestamp]] = None) -> np.ndarray:
        """
        Calculate exponential decay weights
        
        Args:
            n_samples: Number of samples to weight
            timestamps: Optional timestamps for adaptive weighting
            
        Returns:
            Array of weights (most recent = highest weight)
        """
        # Limit to window size
        actual_samples = min(n_samples, self.window_size)
        
        # Choose decay rate based on drift state
        decay_rate = self.drift_decay if self.drift_detected else self.base_decay
        
        # Calculate base exponential weights
        indices = np.arange(actual_samples)
        weights = decay_rate ** (actual_samples - 1 - indices)
        
        # Adaptive weighting based on timestamps
        if timestamps and self.drift_start_time:
            weights = self._apply_adaptive_weighting(weights, timestamps[-actual_samples:])
        
        # Apply minimum weight threshold
        weights = np.maximum(weights, self.min_weight)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        logger.debug(f"⚖️ Generated {len(weights)} weights: min={weights.min():.4f}, max={weights.max():.4f}")
        return weights
    
    def _apply_adaptive_weighting(self, base_weights: np.ndarray, timestamps: List[pd.Timestamp]) -> np.ndarray:
        """Apply adaptive weighting based on drift timing"""
        
        if not self.drift_start_time:
            return base_weights
        
        # Calculate time since drift
        time_diffs = [(ts - self.drift_start_time).total_seconds() / 3600 for ts in timestamps]  # Hours
        
        # Apply additional decay for pre-drift samples
        adaptive_factors = np.ones_like(base_weights)
        for i, time_diff in enumerate(time_diffs):
            if time_diff < 0:  # Pre-drift sample
                hours_before_drift = abs(time_diff)
                additional_decay = 0.9 ** (hours_before_drift / 24)  # Daily decay
                adaptive_factors[i] = additional_decay
        
        return base_weights * adaptive_factors


class LightRetrainer(AdaptiveLearner):
    """
    Light retraining with minimal computational overhead.
    
    Performs quick model updates using:
    - Few epochs for neural networks
    - Small boosting rounds for GBMs
    - Warm starts when possible
    - Recency weighted samples
    """
    
    def __init__(self,
                 max_epochs: int = 5,
                 max_boosting_rounds: int = 50,
                 learning_rate_factor: float = 0.5,
                 enable_warm_start: bool = True,
                 min_samples: int = 100):
        """
        Initialize light retrainer
        
        Args:
            max_epochs: Maximum epochs for neural networks
            max_boosting_rounds: Maximum rounds for gradient boosting
            learning_rate_factor: Factor to reduce learning rate
            enable_warm_start: Enable warm starting when possible
            min_samples: Minimum samples required for retraining
        """
        self.max_epochs = max_epochs
        self.max_boosting_rounds = max_boosting_rounds
        self.learning_rate_factor = learning_rate_factor
        self.enable_warm_start = enable_warm_start
        self.min_samples = min_samples
        
        self.recency_weighter = RecencyWeighter()
        self.adaptation_history = []
        
        logger.info(f"🔄 Initialized light retrainer: epochs={max_epochs}, boosting={max_boosting_rounds}")
    
    def should_adapt(self, drift_info: Dict[str, Any]) -> bool:
        """Determine if light retraining is appropriate"""
        
        severity = drift_info.get('severity', 'minor')
        monitor_type = drift_info.get('monitor_type', '')
        
        # Light retrain for moderate drift or specific monitor types
        if severity in ['moderate', 'minor']:
            return True
        
        # Light retrain for certain monitor types regardless of severity
        if monitor_type in ['Feature Distribution', 'Residual Drift']:
            return True
        
        return False
    
    def adapt(self, model: Any, data: Dict[str, Any], drift_info: Dict[str, Any]) -> AdaptationResult:
        """
        Perform light retraining
        
        Args:
            model: Model to retrain
            data: Training data {'X': features, 'y': targets, 'timestamps': optional}
            drift_info: Information about detected drift
            
        Returns:
            AdaptationResult with performance metrics
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Extract data
            X = data['X']
            y = data['y']
            timestamps = data.get('timestamps', None)
            
            if len(X) < self.min_samples:
                return AdaptationResult(
                    strategy=AdaptationStrategy.LIGHT_RETRAIN,
                    success=False,
                    execution_time=0.0,
                    performance_before=0.0,
                    performance_after=0.0,
                    warnings=[f"Insufficient samples: {len(X)} < {self.min_samples}"]
                )
            
            # Calculate performance before adaptation
            perf_before = self._calculate_performance(model, X, y)
            
            # Set drift state for recency weighting
            self.recency_weighter.set_drift_state(True, start_time)
            
            # Calculate sample weights
            sample_weights = self.recency_weighter.calculate_weights(len(X), timestamps)
            
            # Perform model-specific light retraining
            adapted_model = self._light_retrain_model(model, X, y, sample_weights, drift_info)
            
            # Calculate performance after adaptation
            perf_after = self._calculate_performance(adapted_model, X, y)
            
            # Calculate execution time
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Update model in-place if successful
            if perf_after >= perf_before * 0.95:  # Accept if not significantly worse
                self._update_model_inplace(model, adapted_model)
                success = True
                warnings = []
            else:
                success = False
                warnings = [f"Performance degraded: {perf_before:.4f} -> {perf_after:.4f}"]
            
            result = AdaptationResult(
                strategy=AdaptationStrategy.LIGHT_RETRAIN,
                success=success,
                execution_time=execution_time,
                performance_before=perf_before,
                performance_after=perf_after,
                metadata={
                    'n_samples': len(X),
                    'weight_range': [sample_weights.min(), sample_weights.max()],
                    'drift_severity': drift_info.get('severity', 'unknown'),
                    'retrain_type': self._get_model_type(model)
                },
                warnings=warnings
            )
            
            self.adaptation_history.append(result)
            
            if success:
                logger.info(f"✅ Light retrain successful: {perf_before:.4f} -> {perf_after:.4f} in {execution_time:.2f}s")
            else:
                logger.warning("⚠️ Light retrain failed: performance declined")
            
            return result
            
        except Exception as e:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"❌ Light retrain error: {e}")
            
            return AdaptationResult(
                strategy=AdaptationStrategy.LIGHT_RETRAIN,
                success=False,
                execution_time=execution_time,
                performance_before=0.0,
                performance_after=0.0,
                warnings=[f"Adaptation failed: {str(e)}"]
            )
    
    def _light_retrain_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           sample_weights: np.ndarray, drift_info: Dict[str, Any]) -> Any:
        """Perform model-specific light retraining"""
        
        self._get_model_type(model)
        
        if 'sklearn' in str(type(model)):
            return self._light_retrain_sklearn(model, X, y, sample_weights)
        elif 'xgboost' in str(type(model)):
            return self._light_retrain_xgboost(model, X, y, sample_weights)
        elif 'lightgbm' in str(type(model)):
            return self._light_retrain_lightgbm(model, X, y, sample_weights)
        else:
            # Generic approach - try partial_fit or refit
            return self._light_retrain_generic(model, X, y, sample_weights)
    
    def _light_retrain_sklearn(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             sample_weights: np.ndarray) -> Any:
        """Light retrain for sklearn models"""
        
        from sklearn.base import clone
        
        # Clone model to avoid modifying original
        new_model = clone(model)
        
        # Check if model supports partial_fit
        if hasattr(new_model, 'partial_fit'):
            # Use partial_fit for incremental learning
            new_model.partial_fit(X, y, sample_weight=sample_weights)
        elif hasattr(new_model, 'warm_start'):
            # Use warm start for iterative models
            new_model.set_params(warm_start=True)
            if hasattr(new_model, 'n_estimators'):
                # Add a few more estimators
                current_estimators = getattr(new_model, 'n_estimators', 100)
                new_model.set_params(n_estimators=current_estimators + self.max_boosting_rounds)
            new_model.fit(X, y, sample_weight=sample_weights)
        else:
            # Full refit with sample weights
            new_model.fit(X, y, sample_weight=sample_weights)
        
        return new_model
    
    def _light_retrain_xgboost(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             sample_weights: np.ndarray) -> Any:
        """Light retrain for XGBoost models"""
        
        try:
            import xgboost as xgb
            
            # Get current parameters
            params = model.get_params()
            
            # Reduce learning rate for light retraining
            params['learning_rate'] = params.get('learning_rate', 0.1) * self.learning_rate_factor
            
            # Create DMatrix with weights
            dtrain = xgb.DMatrix(X, label=y, weight=sample_weights)
            
            # Continue training from existing model
            new_model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.max_boosting_rounds,
                xgb_model=model.get_booster() if hasattr(model, 'get_booster') else None
            )
            
            return new_model
            
        except ImportError:
            logger.warning("XGBoost not available, falling back to generic retraining")
            return self._light_retrain_generic(model, X, y, sample_weights)
    
    def _light_retrain_lightgbm(self, model: Any, X: np.ndarray, y: np.ndarray, 
                              sample_weights: np.ndarray) -> Any:
        """Light retrain for LightGBM models"""
        
        try:
            import lightgbm as lgb
            
            # Get current parameters
            params = model.get_params()
            
            # Reduce learning rate
            params['learning_rate'] = params.get('learning_rate', 0.1) * self.learning_rate_factor
            
            # Create dataset with weights
            train_data = lgb.Dataset(X, label=y, weight=sample_weights)
            
            # Continue training
            new_model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=self.max_boosting_rounds,
                init_model=model.booster_ if hasattr(model, 'booster_') else None
            )
            
            return new_model
            
        except ImportError:
            logger.warning("LightGBM not available, falling back to generic retraining")
            return self._light_retrain_generic(model, X, y, sample_weights)
    
    def _light_retrain_generic(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             sample_weights: np.ndarray) -> Any:
        """Generic light retraining approach"""
        
        from sklearn.base import clone
        
        # Clone and refit
        new_model = clone(model)
        
        # Try to fit with sample weights
        try:
            new_model.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            # Model doesn't support sample weights
            logger.warning("Model doesn't support sample weights, using unweighted refit")
            new_model.fit(X, y)
        
        return new_model
    
    def _calculate_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model performance score"""
        
        try:
            if hasattr(model, 'score'):
                return model.score(X, y)
            elif hasattr(model, 'predict'):
                predictions = model.predict(X)
                # Use MSE for regression, accuracy for classification
                if len(np.unique(y)) <= 10:  # Likely classification
                    return np.mean(predictions == y)
                else:  # Regression
                    return -np.mean((predictions - y) ** 2)  # Negative MSE
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_model_type(self, model: Any) -> str:
        """Get model type string"""
        return str(type(model).__name__)
    
    def _update_model_inplace(self, original_model: Any, adapted_model: Any):
        """Update original model with adapted parameters"""
        
        # For most sklearn models, we can copy the fitted parameters
        if hasattr(adapted_model, '__dict__'):
            for key, value in adapted_model.__dict__.items():
                if not key.startswith('_'):
                    setattr(original_model, key, value)


class OnlineUpdater(AdaptiveLearner):
    """
    Online/incremental learning for streaming data.
    
    Uses River framework for partial_fit models that can update
    between full retrains.
    """
    
    def __init__(self,
                 update_frequency: int = 10,
                 buffer_size: int = 100,
                 enable_river: bool = True):
        """
        Initialize online updater
        
        Args:
            update_frequency: How often to perform updates
            buffer_size: Size of update buffer
            enable_river: Enable River integration
        """
        self.update_frequency = update_frequency
        self.buffer_size = buffer_size
        self.enable_river = enable_river
        
        # Update tracking
        self.update_buffer = []
        self.update_count = 0
        self.last_update_time = None
        
        logger.info(f"🌊 Initialized online updater: freq={update_frequency}, buffer={buffer_size}")
    
    def should_adapt(self, drift_info: Dict[str, Any]) -> bool:
        """Determine if online update is appropriate"""
        
        severity = drift_info.get('severity', 'minor')
        monitor_type = drift_info.get('monitor_type', '')
        
        # Online updates for minor drift or continuous adaptation
        return severity in ['minor', 'none'] or 'streaming' in monitor_type.lower()
    
    def adapt(self, model: Any, data: Dict[str, Any], drift_info: Dict[str, Any]) -> AdaptationResult:
        """
        Perform online model update
        
        Args:
            model: Model to update
            data: New data batch
            drift_info: Drift information
            
        Returns:
            AdaptationResult
        """
        start_time = pd.Timestamp.now()
        
        try:
            X = data['X']
            y = data['y']
            
            # Add to update buffer
            for i in range(len(X)):
                self.update_buffer.append((X[i], y[i]))
            
            # Trim buffer if needed
            if len(self.update_buffer) > self.buffer_size:
                self.update_buffer = self.update_buffer[-self.buffer_size:]
            
            # Check if update is needed
            if len(self.update_buffer) < self.update_frequency:
                return AdaptationResult(
                    strategy=AdaptationStrategy.INCREMENTAL_UPDATE,
                    success=True,
                    execution_time=0.0,
                    performance_before=0.0,
                    performance_after=0.0,
                    metadata={'buffer_size': len(self.update_buffer), 'status': 'buffering'}
                )
            
            # Perform incremental update
            success = self._perform_incremental_update(model)
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.update_count += 1
            self.last_update_time = start_time
            
            # Clear buffer after update
            self.update_buffer = []
            
            result = AdaptationResult(
                strategy=AdaptationStrategy.INCREMENTAL_UPDATE,
                success=success,
                execution_time=execution_time,
                performance_before=0.0,  # Not easily measurable for online updates
                performance_after=0.0,
                metadata={
                    'update_count': self.update_count,
                    'buffer_processed': self.update_frequency,
                    'model_type': str(type(model).__name__)
                }
            )
            
            if success:
                logger.info(f"🌊 Online update successful: {self.update_count} total updates")
            else:
                logger.warning("⚠️ Online update failed")
            
            return result
            
        except Exception as e:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"❌ Online update error: {e}")
            
            return AdaptationResult(
                strategy=AdaptationStrategy.INCREMENTAL_UPDATE,
                success=False,
                execution_time=execution_time,
                performance_before=0.0,
                performance_after=0.0,
                warnings=[f"Online update failed: {str(e)}"]
            )
    
    def _perform_incremental_update(self, model: Any) -> bool:
        """Perform the actual incremental update"""
        
        if not self.update_buffer:
            return False
        
        try:
            # Extract buffered data
            X_buffer = np.array([x[0] for x in self.update_buffer])
            y_buffer = np.array([x[1] for x in self.update_buffer])
            
            # Try partial_fit if available
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_buffer, y_buffer)
                return True
            
            # Try River integration if enabled
            if self.enable_river:
                return self._river_update(model, X_buffer, y_buffer)
            
            # Fall back to warm start if available
            if hasattr(model, 'warm_start'):
                model.set_params(warm_start=True)
                model.fit(X_buffer, y_buffer)
                return True
            
            logger.warning("Model doesn't support incremental updates")
            return False
            
        except Exception as e:
            logger.error(f"Incremental update failed: {e}")
            return False
    
    def _river_update(self, model: Any, X: np.ndarray, y: np.ndarray) -> bool:
        """Update using River framework"""
        
        try:
            # This would integrate with River for online learning
            # For now, placeholder implementation
            logger.info("River integration not yet implemented")
            return False
        except Exception:
            return False


class RegimeResetter(AdaptiveLearner):
    """
    Hard reset of regime/HMM models when major drift occurs.
    
    Completely reinitializes regime detection models when
    market structure changes significantly.
    """
    
    def __init__(self,
                 reset_threshold_severity: str = "major",
                 cooldown_period: int = 3600,  # 1 hour
                 backup_previous_regime: bool = True):
        """
        Initialize regime resetter
        
        Args:
            reset_threshold_severity: Minimum severity to trigger reset
            cooldown_period: Minimum time between resets (seconds)
            backup_previous_regime: Whether to backup previous regime state
        """
        self.reset_threshold_severity = reset_threshold_severity
        self.cooldown_period = cooldown_period
        self.backup_previous_regime = backup_previous_regime
        
        # Reset tracking
        self.last_reset_time = None
        self.reset_count = 0
        self.regime_backups = []
        
        logger.info(f"🔄 Initialized regime resetter: threshold={reset_threshold_severity}")
    
    def should_adapt(self, drift_info: Dict[str, Any]) -> bool:
        """Determine if regime reset is needed"""
        
        severity = drift_info.get('severity', 'minor')
        
        # Check severity threshold
        severity_levels = ['minor', 'moderate', 'major', 'critical']
        threshold_level = severity_levels.index(self.reset_threshold_severity)
        current_level = severity_levels.index(severity) if severity in severity_levels else 0
        
        if current_level < threshold_level:
            return False
        
        # Check cooldown period
        if self.last_reset_time:
            time_since_reset = (pd.Timestamp.now() - self.last_reset_time).total_seconds()
            if time_since_reset < self.cooldown_period:
                return False
        
        return True
    
    def adapt(self, model: Any, data: Dict[str, Any], drift_info: Dict[str, Any]) -> AdaptationResult:
        """
        Perform regime reset
        
        Args:
            model: Model with regime components to reset
            data: Training data for regime reinitialization
            drift_info: Drift information
            
        Returns:
            AdaptationResult
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Backup current regime state if requested
            if self.backup_previous_regime:
                self._backup_regime_state(model)
            
            # Perform regime reset
            success = self._reset_regime_components(model, data)
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            if success:
                self.reset_count += 1
                self.last_reset_time = start_time
            
            result = AdaptationResult(
                strategy=AdaptationStrategy.REGIME_RESET,
                success=success,
                execution_time=execution_time,
                performance_before=0.0,
                performance_after=0.0,
                metadata={
                    'reset_count': self.reset_count,
                    'severity_triggered': drift_info.get('severity', 'unknown'),
                    'backup_created': self.backup_previous_regime
                }
            )
            
            if success:
                logger.info(f"🔄 Regime reset successful: reset #{self.reset_count}")
            else:
                logger.warning("⚠️ Regime reset failed")
            
            return result
            
        except Exception as e:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.error(f"❌ Regime reset error: {e}")
            
            return AdaptationResult(
                strategy=AdaptationStrategy.REGIME_RESET,
                success=False,
                execution_time=execution_time,
                performance_before=0.0,
                performance_after=0.0,
                warnings=[f"Regime reset failed: {str(e)}"]
            )
    
    def _backup_regime_state(self, model: Any):
        """Backup current regime state"""
        
        try:
            # Look for regime-related attributes
            regime_state = {}
            
            for attr_name in dir(model):
                if 'regime' in attr_name.lower() or 'hmm' in attr_name.lower():
                    attr_value = getattr(model, attr_name)
                    if not callable(attr_value):
                        regime_state[attr_name] = attr_value
            
            if regime_state:
                backup = {
                    'timestamp': pd.Timestamp.now(),
                    'regime_state': regime_state,
                    'reset_count': self.reset_count
                }
                self.regime_backups.append(backup)
                
                # Limit backup history
                if len(self.regime_backups) > 5:
                    self.regime_backups.pop(0)
                
                logger.info(f"💾 Backed up regime state: {len(regime_state)} attributes")
            
        except Exception as e:
            logger.warning(f"Failed to backup regime state: {e}")
    
    def _reset_regime_components(self, model: Any, data: Dict[str, Any]) -> bool:
        """Reset regime-related model components"""
        
        try:
            reset_count = 0
            
            # Reset regime-related attributes
            for attr_name in dir(model):
                if any(keyword in attr_name.lower() for keyword in ['regime', 'hmm', 'state', 'transition']):
                    if hasattr(model, attr_name) and not callable(getattr(model, attr_name)):
                        try:
                            # Initialize to default state
                            attr = getattr(model, attr_name)
                            if isinstance(attr, (np.ndarray, list)):
                                # Reset arrays/lists to zeros or empty
                                if hasattr(attr, 'shape'):
                                    setattr(model, attr_name, np.zeros_like(attr))
                                else:
                                    setattr(model, attr_name, [])
                            elif isinstance(attr, (int, float)):
                                # Reset numeric values
                                setattr(model, attr_name, 0)
                            elif isinstance(attr, dict):
                                # Clear dictionaries
                                setattr(model, attr_name, {})
                            
                            reset_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to reset {attr_name}: {e}")
            
            # If model has specific regime reset method, use it
            if hasattr(model, 'reset_regime'):
                model.reset_regime()
                reset_count += 1
            elif hasattr(model, 'reinitialize_states'):
                model.reinitialize_states()
                reset_count += 1
            
            logger.info(f"🔄 Reset {reset_count} regime components")
            return reset_count > 0
            
        except Exception as e:
            logger.error(f"Regime reset failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get regime reset statistics"""
        
        return {
            'total_resets': self.reset_count,
            'last_reset': self.last_reset_time,
            'cooldown_remaining': max(0, self.cooldown_period - (
                (pd.Timestamp.now() - self.last_reset_time).total_seconds() 
                if self.last_reset_time else self.cooldown_period
            )),
            'backups_available': len(self.regime_backups),
            'reset_threshold': self.reset_threshold_severity
        }