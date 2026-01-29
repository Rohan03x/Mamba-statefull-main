"""
Online Learning System for AI Price Forecasting

This module implements incremental learning capabilities with concept drift detection
to enable real-time model adaptation in changing market conditions.

Key Features:
- Concept drift detection (ADWIN, Page-Hinkley, DDM)
- Incremental model updates
- Performance monitoring and alerts
- Adaptive retraining triggers
- Online feature scaling and adaptation
- Model ensemble weight adaptation

Author: AI Assistant
Created: 2024
"""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Statistical imports for drift detection
try:
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not available - some drift detection methods disabled")

# ML imports
try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available - online learning limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning system"""

    # Drift detection settings
    enable_drift_detection: bool = True
    drift_detection_methods: List[str] = field(
        default_factory=lambda: ['adwin', 'page_hinkley'])
    drift_sensitivity: float = 0.05  # Significance level for drift detection
    # Minimum samples before drift detection
    min_samples_drift_detection: int = 100

    # Model update settings
    incremental_update_frequency: int = 10  # Update model every N new samples
    batch_retrain_threshold: int = 1000  # Full retrain when drift + N samples
    max_model_age_days: int = 90  # Force retrain after this many days

    # Partial retraining settings
    enable_partial_retrain: bool = True
    partial_retrain_window: int = 128  # Samples per partial retrain batch
    partial_retrain_min_samples: int = 32  # Minimum samples required to trigger partial retrain
    partial_retrain_backlog: int = 4  # Number of windows to retain for retraining backlog

    # Performance monitoring
    performance_window_size: int = 500  # Rolling window for performance tracking
    # Trigger retrain if performance drops by this %
    performance_degradation_threshold: float = 0.15
    min_performance_samples: int = 50  # Min samples for performance comparison

    # Feature adaptation
    enable_feature_adaptation: bool = True
    feature_scaling_method: str = 'robust'  # 'standard', 'robust', 'none'
    feature_selection_adaptation: bool = False

    # Model ensemble adaptation
    enable_ensemble_adaptation: bool = True
    ensemble_weight_decay: float = 0.99  # Decay factor for ensemble weights
    min_ensemble_weight: float = 0.01  # Minimum weight for any model

    # Storage and persistence
    save_adaptation_history: bool = True
    adaptation_history_max_size: int = 10000
    model_checkpoint_frequency: int = 100  # Save model every N updates

    # Advanced settings
    # Learning rate adjustment after drift
    concept_drift_adaptation_rate: float = 0.1
    use_confidence_weighted_updates: bool = True
    enable_meta_learning: bool = False  # Experimental: learn to learn


@dataclass
class DriftDetectionResult:
    """Result from drift detection algorithm"""

    drift_detected: bool
    drift_strength: float  # 0-1, strength of detected drift
    detection_method: str
    confidence: float
    timestamp: datetime
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlinePerformanceMetrics:
    """Performance metrics for online learning"""

    mae: float = float('inf')
    rmse: float = float('inf')
    mape: float = float('inf')
    direction_accuracy: float = 0.0
    samples_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Drift and adaptation metrics
    total_drifts_detected: int = 0
    models_retrained: int = 0
    incremental_updates: int = 0
    adaptation_efficiency: float = 0.0  # Performance recovery after drift
    partial_retrains: int = 0


class DriftDetector(ABC):
    """Abstract base class for drift detection algorithms"""

    @abstractmethod
    def add_sample(
            self,
            value: float,
            prediction: float = None) -> DriftDetectionResult:
        """Add new sample and check for drift"""

    @abstractmethod
    def reset(self):
        """Reset detector state"""


class ADWINDriftDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) drift detector

    Maintains a sliding window of variable size and detects changes
    in the distribution of values.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta  # Confidence level
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.last_drift_time = datetime.now()

    def add_sample(
            self,
            value: float,
            prediction: float = None) -> DriftDetectionResult:
        """Add sample and detect drift using ADWIN algorithm"""

        # Use prediction error if prediction provided
        if prediction is not None:
            sample = abs(value - prediction)
        else:
            sample = value

        self.window.append(sample)
        self.total += sample
        self.width += 1

        # Update variance incrementally
        if self.width > 1:
            self.variance = np.var(list(self.window))

        drift_detected = False
        drift_strength = 0.0

        # Check for drift using ADWIN cut detection
        if self.width > 2:
            drift_detected, drift_strength = self._detect_cut()

        if drift_detected:
            self.last_drift_time = datetime.now()

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_strength=drift_strength,
            detection_method='adwin',
            confidence=1.0 - self.delta,
            timestamp=datetime.now(),
            additional_info={
                'window_size': self.width,
                'mean': self.total / self.width if self.width > 0 else 0,
                'variance': self.variance
            }
        )

    def _detect_cut(self) -> Tuple[bool, float]:
        """Detect if there's a significant cut in the window"""
        if self.width < 2:
            return False, 0.0

        n = self.width

        # Compute harmonic mean for cut detection
        if self.variance <= 0:
            return False, 0.0

        # Simplified ADWIN cut detection
        epsilon_cut = np.sqrt(
            (2.0 * self.variance * np.log(2.0 / self.delta)) / n
        ) + (2.0 * np.log(2.0 / self.delta)) / (3.0 * n)

        # Check if cut is significant
        window_list = list(self.window)
        n1 = n // 2
        n2 = n - n1

        if n1 > 0 and n2 > 0:
            mean1 = np.mean(window_list[:n1])
            mean2 = np.mean(window_list[n1:])
            diff = abs(mean1 - mean2)

            if diff > epsilon_cut:
                # Remove old samples (cut detected)
                while len(self.window) > n2:
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1

                return True, min(1.0, diff / epsilon_cut)

        return False, 0.0

    def reset(self):
        """Reset ADWIN detector"""
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0


class PageHinkleyDriftDetector(DriftDetector):
    """Page-Hinkley drift detector

    Detects changes in the mean of a signal using cumulative sum.
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.9999):
        self.threshold = threshold
        self.alpha = alpha
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.min_pos = 0.0
        self.max_neg = 0.0
        self.mean = 0.0
        self.n_samples = 0

    def add_sample(
            self,
            value: float,
            prediction: float = None) -> DriftDetectionResult:
        """Add sample and detect drift using Page-Hinkley test"""

        # Use prediction error if available
        if prediction is not None:
            sample = value - prediction
        else:
            sample = value

        # Update running mean
        self.n_samples += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * sample

        # Update cumulative sums
        diff = sample - self.mean
        self.sum_pos = max(0, self.sum_pos + diff)
        self.sum_neg = min(0, self.sum_neg + diff)

        # Update extremes
        self.min_pos = min(self.min_pos, self.sum_pos)
        self.max_neg = max(self.max_neg, self.sum_neg)

        # Check for drift
        ph_pos = self.sum_pos - self.min_pos
        ph_neg = self.max_neg - self.sum_neg

        drift_detected = ph_pos > self.threshold or ph_neg > self.threshold
        drift_strength = max(ph_pos, ph_neg) / \
            self.threshold if self.threshold > 0 else 0

        if drift_detected:
            self.reset()

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_strength=min(1.0, drift_strength),
            detection_method='page_hinkley',
            confidence=0.95,  # Approximate confidence
            timestamp=datetime.now(),
            additional_info={
                'ph_pos': ph_pos,
                'ph_neg': ph_neg,
                'threshold': self.threshold,
                'samples': self.n_samples
            }
        )

    def reset(self):
        """Reset Page-Hinkley detector"""
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.min_pos = 0.0
        self.max_neg = 0.0
        # Keep mean and sample count for continuity


class DDMDriftDetector(DriftDetector):
    """Drift Detection Method (DDM)

    Monitors error rate and its standard deviation to detect concept drift.
    """

    def __init__(self, alpha_warning: float = 2.0, alpha_drift: float = 3.0):
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift
        self.n_samples = 0
        self.n_errors = 0
        self.error_rate = 0.0
        self.error_std = 0.0
        self.min_error_rate = float('in')
        self.min_std = float('inf')

    def add_sample(
            self,
            value: float,
            prediction: float = None) -> DriftDetectionResult:
        """Add sample and detect drift using DDM"""

        # Determine if this is an error (for regression, use threshold)
        if prediction is not None:
            error_threshold = 0.1  # 10% error threshold
            is_error = abs(value - prediction) / \
                max(abs(value), 1e-8) > error_threshold
        else:
            # For classification-like detection
            is_error = value < 0.5  # Assume binary classification

        self.n_samples += 1
        if is_error:
            self.n_errors += 1

        # Update error rate and standard deviation
        self.error_rate = self.n_errors / self.n_samples
        if self.n_samples > 1:
            self.error_std = np.sqrt(
                self.error_rate * (1 - self.error_rate) / self.n_samples
            )

        # Update minimums
        if self.error_rate + self.error_std < self.min_error_rate + self.min_std:
            self.min_error_rate = self.error_rate
            self.min_std = self.error_std

        # Check for drift
        drift_detected = False
        drift_strength = 0.0

        if self.n_samples > 30:  # Need sufficient samples
            # Warning level
            warning_threshold = self.min_error_rate + self.alpha_warning * self.min_std
            # Drift level
            drift_threshold = self.min_error_rate + self.alpha_drift * self.min_std

            current_level = self.error_rate + self.error_std

            if current_level > drift_threshold:
                drift_detected = True
                drift_strength = min(1.0, (current_level - warning_threshold) /
                                     (drift_threshold - warning_threshold))
                self.reset()

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_strength=drift_strength,
            detection_method='ddm',
            confidence=0.95,
            timestamp=datetime.now(),
            additional_info={
                'error_rate': self.error_rate,
                'error_std': self.error_std,
                'min_error_rate': self.min_error_rate,
                'samples': self.n_samples
            }
        )

    def reset(self):
        """Reset DDM detector"""
        self.n_samples = 0
        self.n_errors = 0
        self.error_rate = 0.0
        self.error_std = 0.0


class OnlineFeatureScaler:
    """Online feature scaling that adapts to new data"""

    def __init__(self, method: str = 'robust', alpha: float = 0.01):
        self.method = method
        self.alpha = alpha  # Learning rate for online updates
        self.n_samples = 0
        self.mean = None
        self.var = None
        self.median = None
        self.iqr = None
        self.quantiles = None
        self.min_val = None
        self.max_val = None

    def partial_fit(self, X: np.ndarray):
        """Update scaler with new data"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.n_samples == 0:
            # Initialize
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
            self.min_val = np.min(X, axis=0)
            self.max_val = np.max(X, axis=0)

            if self.method == 'robust':
                self.median = np.median(X, axis=0)
                self.iqr = np.percentile(
                    X, 75, axis=0) - np.percentile(X, 25, axis=0)
                self.iqr = np.maximum(self.iqr, 1e-8)  # Avoid division by zero

        else:
            # Online update
            new_mean = np.mean(X, axis=0)
            new_var = np.var(X, axis=0)

            # Exponential moving average
            self.mean = (1 - self.alpha) * self.mean + self.alpha * new_mean
            self.var = (1 - self.alpha) * self.var + self.alpha * new_var

            # Update min/max
            self.min_val = np.minimum(self.min_val, np.min(X, axis=0))
            self.max_val = np.maximum(self.max_val, np.max(X, axis=0))

            if self.method == 'robust':
                new_median = np.median(X, axis=0)
                new_iqr = np.percentile(X, 75, axis=0) - \
                    np.percentile(X, 25, axis=0)
                new_iqr = np.maximum(new_iqr, 1e-8)

                self.median = (1 - self.alpha) * self.median + \
                    self.alpha * new_median
                self.iqr = (1 - self.alpha) * self.iqr + self.alpha * new_iqr

        self.n_samples += len(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using current scaling parameters"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False

        if self.n_samples == 0:
            return X  # No scaling if not fitted

        if self.method == 'standard':
            std = np.sqrt(np.maximum(self.var, 1e-8))
            x_scaled = (X - self.mean) / std
        elif self.method == 'robust':
            x_scaled = (X - self.median) / self.iqr
        elif self.method == 'minmax':
            range_val = self.max_val - self.min_val
            range_val = np.maximum(range_val, 1e-8)
            x_scaled = (X - self.min_val) / range_val
        else:
            x_scaled = X  # No scaling

        return x_scaled.squeeze() if squeeze else x_scaled


class OnlineLearningSystem:
    """Main online learning system for adaptive forecasting"""

    def __init__(self, config: OnlineLearningConfig):
        self.config = config
        self.drift_detectors: Dict[str, DriftDetector] = {}
        self.feature_scaler = OnlineFeatureScaler(
            method=config.feature_scaling_method,
            alpha=0.01
        )

        # Performance tracking
        self.performance_history = deque(maxlen=config.performance_window_size)
        self.current_performance = OnlinePerformanceMetrics()
        self.baseline_performance = None

        # Model management
        self.model = None
        self.model_age = 0
        self.last_update_time = datetime.now()
        self.update_count = 0

        # Ensemble weights (if using ensemble)
        self.ensemble_weights = {}
        self.ensemble_performance = {}

        # Adaptation history
        self.adaptation_history = deque(
            maxlen=config.adaptation_history_max_size)

        # Recent samples buffer for partial retraining
        backlog_windows = max(1, config.partial_retrain_backlog)
        window_size = max(1, config.partial_retrain_window)
        self.recent_samples: deque = deque(
            maxlen=backlog_windows * window_size
        )
        
        # Event logging setup
        self.event_log_path = None  # Set by aggregator_panel during initialization
        self.symbol = None  # Set by aggregator_panel
        self.horizon = None  # Set by aggregator_panel

        # Initialize drift detectors
        self._initialize_drift_detectors()

        logger.info(
            f"Initialized online learning system with config: {config}")

    def _initialize_drift_detectors(self):
        """Initialize configured drift detection algorithms"""
        if not self.config.enable_drift_detection:
            return

        for method in self.config.drift_detection_methods:
            if method == 'adwin':
                self.drift_detectors[method] = ADWINDriftDetector(
                    delta=self.config.drift_sensitivity
                )
            elif method == 'page_hinkley':
                self.drift_detectors[method] = PageHinkleyDriftDetector(
                    threshold=50.0,
                    alpha=0.9999
                )
            elif method == 'ddm':
                self.drift_detectors[method] = DDMDriftDetector(
                    alpha_warning=2.0,
                    alpha_drift=3.0
                )
            else:
                logger.warning(f"Unknown drift detection method: {method}")

    def _store_recent_sample(self, features: np.ndarray, target: float):
        """Persist latest sample for potential partial retraining."""

        if not self.config.enable_partial_retrain:
            return

        try:
            features_array = np.asarray(features, dtype=float).reshape(-1)
        except Exception:
            features_array = np.array(features).reshape(-1)

        # Ensure we keep copies to avoid accidental mutation from upstream buffers
        self.recent_samples.append((features_array.copy(), float(target)))

    def add_sample(self,
                   features: np.ndarray,
                   target: float,
                   prediction: Optional[float] = None,
                   calibration_quality: Optional[float] = None) -> Dict[str, Any]:
        """Add new sample and trigger adaptations if needed.
        
        Parameters
        ----------
        calibration_quality : float, optional
            Calibration quality score in [0, 1]. When provided, learning decisions
            are gated: updates only happen when calibration_quality >= threshold.
            Default: None (no gating, always allow updates).
        """

        adaptation_info = {
            'drift_detected': False,
            'model_updated': False,
            'model_retrained': False,
            'performance_updated': False,
            'drift_results': [],
            'partial_retrain': False,
            'calibration_gated': False,
            'calibration_quality': calibration_quality,
        }

        # ─────────────────────────────────────────────────────────────────────
        # Calibration-Gated Learning: Gate updates when calibration is poor
        # ─────────────────────────────────────────────────────────────────────
        CALIB_GATE_THRESHOLD = 0.3  # Match Phase-2 calib_freeze_threshold
        learning_allowed = True
        if calibration_quality is not None:
            learning_allowed = float(calibration_quality) >= CALIB_GATE_THRESHOLD
            if not learning_allowed:
                adaptation_info['calibration_gated'] = True
                logger.debug(f"[CALIB-GATE] Online learning update skipped: "
                           f"calibration_quality={calibration_quality:.3f} < {CALIB_GATE_THRESHOLD}")

        self._store_recent_sample(features, target)

        # Update feature scaling (always allowed, even when learning is gated)
        if self.config.enable_feature_adaptation:
            self.feature_scaler.partial_fit(features.reshape(1, -1))

        # Check for concept drift (detection always runs, adaptation is gated)
        if self.config.enable_drift_detection and prediction is not None:
            drift_results = self._check_drift(target, prediction)
            adaptation_info['drift_results'] = drift_results

            # Check if any detector found drift
            if any(result.drift_detected for result in drift_results):
                adaptation_info['drift_detected'] = True
                logger.info(
                    f"Concept drift detected: {[r.detection_method for r in drift_results if r.drift_detected]}")
                
                # Log drift event to JSONL
                self._log_event('drift_detected', {
                    'methods': [r.detection_method for r in drift_results if r.drift_detected],
                    'drift_results': [{'method': r.detection_method, 'detected': r.drift_detected} for r in drift_results]
                })

                # Trigger adaptation ONLY if learning is allowed (calibration gate)
                if learning_allowed and self._handle_drift_detection(drift_results):
                    adaptation_info['partial_retrain'] = True

        # Update performance metrics (always allowed - observational, not learning)
        if prediction is not None:
            self._update_performance(target, prediction)
            adaptation_info['performance_updated'] = True

        # Check if model update is needed (GATED by calibration)
        update_needed = self._should_update_model()
        if learning_allowed and update_needed and self.model is not None:
            self._incremental_update(features, target)
            adaptation_info['model_updated'] = True

        # Check if full retrain is needed (GATED by calibration)
        retrain_needed = self._should_retrain_model(
            adaptation_info['drift_detected'])
        if learning_allowed and retrain_needed:
            logger.info("Triggering model retraining")
            adaptation_info['model_retrained'] = True
            
            # Log retraining event to JSONL
            self._log_event('model_retrain_triggered', {
                'drift_detected': adaptation_info['drift_detected'],
                'model_age': self.model_age,
                'calibration_quality': calibration_quality,
            })
            # Note: Actual retraining would be handled by external system

        # Update model age
        self.model_age += 1
        self.update_count += 1
        
        # Increment update counters when model is updated
        if adaptation_info.get('model_updated'):
            self._log_event('incremental_update', {})
        if adaptation_info.get('partial_retrain'):
            self._log_event('partial_retrain', {})

        # Store adaptation event
        if any(
            adaptation_info[key] for key in [
                'drift_detected',
                'model_updated',
                'model_retrained',
                'partial_retrain']):
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'adaptation_info': adaptation_info.copy(),
                'performance': self.current_performance
            })

        return adaptation_info

    def _check_drift(
            self,
            target: float,
            prediction: float) -> List[DriftDetectionResult]:
        """Check all drift detectors for concept drift"""
        results = []

        for name, detector in self.drift_detectors.items():
            try:
                result = detector.add_sample(target, prediction)
                results.append(result)
            except Exception as e:
                logger.warning(f"Drift detector {name} failed: {e}")

        return results

    def _handle_drift_detection(
            self, drift_results: List[DriftDetectionResult]) -> bool:
        """Handle detected concept drift

        Returns:
            bool: True when a partial retrain was performed.
        """

        # Reset performance baseline since concept has changed
        self.baseline_performance = None

        # Adjust learning parameters if configured
        if hasattr(self.model, 'learning_rate'):
            self.model.learning_rate *= (1 +
                                         self.config.concept_drift_adaptation_rate)

        # Reset or adjust ensemble weights
        if self.config.enable_ensemble_adaptation:
            self._adapt_ensemble_weights(drift_results)

        self.current_performance.total_drifts_detected += 1

        partial_retrain_performed = self._perform_partial_retrain(
            drift_results)
        return partial_retrain_performed

    def _perform_partial_retrain(
            self, drift_results: List[DriftDetectionResult]) -> bool:
        """Trigger partial model update using most recent samples after drift."""

        if not self.config.enable_partial_retrain:
            return False

        if self.model is None or not hasattr(self.model, 'partial_fit'):
            logger.debug("Partial retrain skipped: model unavailable or does not support partial_fit")
            return False

        if len(self.recent_samples) < self.config.partial_retrain_min_samples:
            logger.debug("Partial retrain skipped: insufficient recent samples (%d < %d)",
                         len(self.recent_samples), self.config.partial_retrain_min_samples)
            return False

        batch_size = min(self.config.partial_retrain_window, len(self.recent_samples))
        if batch_size <= 0:
            return False

        # Collect latest samples
        recent_batch = list(self.recent_samples)[-batch_size:]
        try:
            X_batch = np.stack([sample[0] for sample in recent_batch])
        except ValueError:
            # Fallback: reshape each sample consistently
            X_batch = np.vstack([np.asarray(sample[0]).reshape(1, -1) for sample in recent_batch])
        y_batch = np.array([sample[1] for sample in recent_batch])

        if self.config.enable_feature_adaptation:
            try:
                X_batch = self.feature_scaler.transform(X_batch)
            except Exception as exc:
                logger.warning(f"Feature scaling failed during partial retrain: {exc}")

        # Determine window information for logging
        current_window = next(
            (
                int(result.additional_info.get('window_size'))
                for result in drift_results
                if isinstance(result.additional_info, dict) and 'window_size' in result.additional_info
            ),
            batch_size
        )

        try:
            self.model.partial_fit(X_batch, y_batch)
            self.current_performance.incremental_updates += 1
            self.current_performance.partial_retrains += 1
            self.last_update_time = datetime.now()
            logger.info(
                f"[DRIFT] Triggering partial retrain (window={current_window}, batch={batch_size})"
            )
            return True
        except Exception as exc:
            logger.error(f"Partial retrain failed: {exc}")
            return False

    def _update_performance(self, target: float, prediction: float):
        """Update performance metrics with new prediction"""

        error = abs(target - prediction)
        squared_error = (target - prediction) ** 2

        # Update running metrics
        n = self.current_performance.samples_count

        # MAE
        previous_mae = self.current_performance.mae
        if not np.isfinite(previous_mae):
            previous_mae = 0.0
        self.current_performance.mae = (
            (previous_mae * n + error) / (n + 1)
        )

        # RMSE
        current_mse = self.current_performance.rmse ** 2 if n > 0 else 0
        new_mse = (current_mse * n + squared_error) / (n + 1)
        self.current_performance.rmse = np.sqrt(new_mse)

        # MAPE
        if abs(target) > 1e-8:
            percentage_error = abs(error / target) * 100
            previous_mape = self.current_performance.mape
            if not np.isfinite(previous_mape):
                previous_mape = 0.0
            self.current_performance.mape = (
                (previous_mape * n + percentage_error) / (n + 1)
            )

        # Direction accuracy (for financial data)
        if n > 0 and hasattr(self, '_last_target'):
            target_direction = 1 if target > self._last_target else 0
            pred_direction = 1 if prediction > self._last_prediction else 0
            direction_correct = 1 if target_direction == pred_direction else 0

            self.current_performance.direction_accuracy = (
                (self.current_performance.direction_accuracy *
                 (n - 1) + direction_correct) / n
            )

        self._last_target = target
        self._last_prediction = prediction

        self.current_performance.samples_count += 1
        self.current_performance.timestamp = datetime.now()

        # Add to performance history
        if len(self.performance_history) == 0 or \
           self.current_performance.samples_count % 10 == 0:  # Sample every 10 predictions
            self.performance_history.append({
                'mae': self.current_performance.mae,
                'rmse': self.current_performance.rmse,
                'mape': self.current_performance.mape,
                'direction_accuracy': self.current_performance.direction_accuracy,
                'timestamp': self.current_performance.timestamp,
                'samples': self.current_performance.samples_count
            })

        # Set baseline if not set
        if self.baseline_performance is None and self.current_performance.samples_count >= self.config.min_performance_samples:
            self.baseline_performance = OnlinePerformanceMetrics(
                mae=self.current_performance.mae,
                rmse=self.current_performance.rmse,
                mape=self.current_performance.mape,
                direction_accuracy=self.current_performance.direction_accuracy,
                samples_count=self.current_performance.samples_count,
                timestamp=self.current_performance.timestamp
            )
            logger.info(
                f"Set performance baseline: MAE={self.baseline_performance.mae:.6f}")

    def _should_update_model(self) -> bool:
        """Determine if incremental model update is needed"""

        if self.model is None:
            return False

        # Update every N samples
        if self.update_count % self.config.incremental_update_frequency == 0:
            return True

        # Update if performance is degrading significantly
        if self._is_performance_degraded():
            return True

        return False

    def _should_retrain_model(self, drift_detected: bool) -> bool:
        """Determine if full model retraining is needed"""

        # Retrain if drift detected and sufficient samples accumulated
        if drift_detected and self.current_performance.samples_count >= self.config.batch_retrain_threshold:
            return True

        # Retrain if model is too old
        time_since_update = datetime.now() - self.last_update_time
        if time_since_update.days >= self.config.max_model_age_days:
            return True

        # Retrain if performance severely degraded
        if self._is_performance_severely_degraded():
            return True

        return False

    def _is_performance_degraded(self) -> bool:
        """Check if current performance is significantly worse than baseline"""

        if self.baseline_performance is None:
            return False

        if self.current_performance.samples_count < self.config.min_performance_samples:
            return False

        # Check MAE degradation
        mae_degradation = (
            (self.current_performance.mae - self.baseline_performance.mae) /
            max(self.baseline_performance.mae, 1e-8)
        )

        return mae_degradation > self.config.performance_degradation_threshold

    def _is_performance_severely_degraded(self) -> bool:
        """Check if performance is severely degraded (2x threshold)"""

        if self.baseline_performance is None:
            return False

        mae_degradation = (
            (self.current_performance.mae - self.baseline_performance.mae) /
            max(self.baseline_performance.mae, 1e-8)
        )

        return mae_degradation > (
            2 * self.config.performance_degradation_threshold)

    def _incremental_update(self, features: np.ndarray, target: float):
        """Perform incremental model update"""

        if not SKLEARN_AVAILABLE:
            logger.warning(
                "sklearn not available - skipping incremental update")
            return

        try:
            # Scale features
            if self.config.enable_feature_adaptation:
                features_scaled = self.feature_scaler.transform(
                    features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            # Update model if it supports partial_fit
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(features_scaled, [target])
                self.current_performance.incremental_updates += 1
                logger.debug(
                    f"Performed incremental update #{self.current_performance.incremental_updates}")
            else:
                logger.warning("Model does not support incremental updates")

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")

    def _adapt_ensemble_weights(
            self, drift_results: List[DriftDetectionResult]):
        """Adapt ensemble weights based on drift detection

        Args:
            drift_results: List of drift detection results (currently unused but reserved for future enhancements)
        """
        # Note: drift_results parameter reserved for future adaptive weight
        # adjustment

        if not self.config.enable_ensemble_adaptation:
            return

        # Apply decay to all weights
        for model_name in self.ensemble_weights:
            self.ensemble_weights[model_name] *= self.config.ensemble_weight_decay

        # Ensure minimum weight
        for model_name in self.ensemble_weights:
            self.ensemble_weights[model_name] = max(
                self.ensemble_weights[model_name],
                self.config.min_ensemble_weight
            )

        # Renormalize weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for model_name in self.ensemble_weights:
                self.ensemble_weights[model_name] /= total_weight

    def compute_max_alpha_features(
            self,
            quantile_predictions: Dict[str, float],
            actual_return: float,
            calibration_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute MAX ALPHA online learning features
        
        These features process quantile forecasts AND calibration signals to:
        - Adaptively modify model trust
        - Weight signals dynamically
        - Detect regime shifts
        - Identify model instability
        
        Args:
            quantile_predictions: Dict with keys like 'q05', 'q10', ..., 'q95'
            actual_return: Realized return for evaluation
            calibration_metrics: Optional calibration error metrics
        
        Returns:
            Dictionary with 15+ max alpha features
        """
        features = {}
        
        # Extract quantile values
        q05 = quantile_predictions.get('q05', quantile_predictions.get('q_05', 0.0))
        q10 = quantile_predictions.get('q10', quantile_predictions.get('q_10', 0.0))
        q25 = quantile_predictions.get('q25', quantile_predictions.get('q_25', 0.0))
        q50 = quantile_predictions.get('q50', quantile_predictions.get('q_50', 0.0))
        q75 = quantile_predictions.get('q75', quantile_predictions.get('q_75', 0.0))
        q90 = quantile_predictions.get('q90', quantile_predictions.get('q_90', 0.0))
        q95 = quantile_predictions.get('q95', quantile_predictions.get('q_95', 0.0))
        
        # ===================================================================
        # A) DYNAMIC TRUST SCORE
        # ===================================================================
        calibration_error = calibration_metrics.get('mean_calibration_error', 0.1) if calibration_metrics else 0.1
        if isinstance(calibration_error, (list, np.ndarray)):
            calibration_error = float(np.mean(calibration_error))
        
        trust_score = 1.0 / (1.0 + abs(float(calibration_error)))
        features['trust_score'] = float(trust_score)
        
        # ===================================================================
        # B) QUANTILE SPREAD SHOCK DETECTOR
        # ===================================================================
        current_spread = q95 - q05
        
        # Use historical spread from performance history (rolling median)
        historical_spreads = []
        for perf in list(self.performance_history)[-20:]:  # Last 20 samples
            # Try to extract spread from stored data (if available)
            if 'quantile_spread' in perf:
                historical_spreads.append(perf['quantile_spread'])
        
        if historical_spreads:
            median_spread = float(np.median(historical_spreads))
            if median_spread > 1e-10:
                quantile_spread_shock = current_spread / median_spread
            else:
                quantile_spread_shock = 1.0
        else:
            quantile_spread_shock = 1.0
        
        features['quantile_spread_shock'] = float(quantile_spread_shock)
        features['quantile_spread_current'] = float(current_spread)
        
        # ===================================================================
        # C) REGIME CLASSIFICATION FROM QUANTILES
        # ===================================================================
        # Volatility regime
        interval_width_25_75 = q75 - q25
        
        # Calculate rolling median width (approximation from history)
        if historical_spreads:
            rolling_median_width = float(np.median(historical_spreads)) * 0.5  # Approximate IQR
        else:
            rolling_median_width = interval_width_25_75
        
        features['regime_vol_low'] = float(interval_width_25_75 < rolling_median_width * 0.8)
        features['regime_vol_high'] = float(interval_width_25_75 > rolling_median_width * 1.2)
        features['regime_vol_normal'] = float(not (features['regime_vol_low'] or features['regime_vol_high']))
        
        # Trend regime
        features['regime_trend_positive'] = float(q75 > 0)
        features['regime_trend_negative'] = float(q25 < 0)
        features['regime_trend_neutral'] = float(q25 <= 0 <= q75)
        
        # ===================================================================
        # D) META-MODEL DRIFT FEATURES
        # ===================================================================
        # Compare current quantiles to last prediction
        if hasattr(self, '_last_quantile_predictions'):
            last_q50 = self._last_quantile_predictions.get('q50', q50)
            last_q25 = self._last_quantile_predictions.get('q25', q25)
            last_q75 = self._last_quantile_predictions.get('q75', q75)
            
            quantile_drift_q50 = q50 - last_q50
            quantile_drift_q25 = q25 - last_q25
            quantile_drift_q75 = q75 - last_q75
            quantile_drift_avg = (quantile_drift_q50 + quantile_drift_q25 + quantile_drift_q75) / 3.0
            quantile_drift_std = float(np.std([quantile_drift_q50, quantile_drift_q25, quantile_drift_q75]))
            
            features['quantile_drift_q50'] = float(quantile_drift_q50)
            features['quantile_drift_avg'] = float(quantile_drift_avg)
            features['quantile_drift_std'] = float(quantile_drift_std)
        else:
            features['quantile_drift_q50'] = 0.0
            features['quantile_drift_avg'] = 0.0
            features['quantile_drift_std'] = 0.0
        
        # Store for next iteration
        self._last_quantile_predictions = {
            'q05': q05, 'q10': q10, 'q25': q25, 'q50': q50, 
            'q75': q75, 'q90': q90, 'q95': q95
        }
        
        # ===================================================================
        # E) FORECAST UNCERTAINTY COMPRESSION
        # ===================================================================
        # Track interval width over time
        if not hasattr(self, '_interval_width_history'):
            self._interval_width_history = deque(maxlen=20)
        
        self._interval_width_history.append(current_spread)
        
        if len(self._interval_width_history) >= 5:
            rolling_std_of_width = float(np.std(self._interval_width_history))
            if rolling_std_of_width > 1e-10:
                uncertainty_compression_ratio = current_spread / rolling_std_of_width
            else:
                uncertainty_compression_ratio = 1.0
        else:
            uncertainty_compression_ratio = 1.0
        
        features['uncertainty_compression_ratio'] = float(uncertainty_compression_ratio)
        features['uncertainty_compression_alert'] = float(uncertainty_compression_ratio < 0.7)
        
        # ===================================================================
        # F) ASYMMETRIC CALIBRATION ERROR
        # ===================================================================
        # Track upside vs downside calibration errors
        upside_error = float(actual_return > q75)  # Should occur ~25% of time
        downside_error = float(actual_return < q25)  # Should occur ~25% of time
        
        # Running average of asymmetric errors
        if not hasattr(self, '_upside_errors'):
            self._upside_errors = deque(maxlen=50)
            self._downside_errors = deque(maxlen=50)
        
        self._upside_errors.append(upside_error)
        self._downside_errors.append(downside_error)
        
        upside_calibration_error = float(np.mean(self._upside_errors)) - 0.25  # Deviation from expected 25%
        downside_calibration_error = float(np.mean(self._downside_errors)) - 0.25
        
        features['upside_calibration_error'] = float(upside_calibration_error)
        features['downside_calibration_error'] = float(downside_calibration_error)
        features['asymmetric_calibration_ratio'] = float(
            abs(downside_calibration_error) / (abs(upside_calibration_error) + 1e-10)
        )
        
        # ===================================================================
        # G) ONLINE RECALIBRATION STRENGTH
        # ===================================================================
        # Weight of recalibration required
        calibration_bias = calibration_metrics.get('reliability_bias', 0.0) if calibration_metrics else 0.0
        if isinstance(calibration_bias, (list, np.ndarray)):
            calibration_bias = float(np.mean(calibration_bias))
        
        # Quantile sharpness = inverse of interval width (tighter = sharper)
        quantile_sharpness = 1.0 / (current_spread + 1e-10)
        
        recalibration_strength = abs(float(calibration_bias)) * quantile_sharpness
        features['recalibration_strength'] = float(recalibration_strength)
        recalibration_needed = recalibration_strength > 0.5
        features['recalibration_needed'] = float(recalibration_needed)
        
        # Log recalibration event to JSONL
        if recalibration_needed:
            self._log_event('recalibration_needed', {
                'strength': float(recalibration_strength),
                'calibration_bias': float(calibration_bias),
                'quantile_sharpness': float(quantile_sharpness)
            })
        
        # ===================================================================
        # ADDITIONAL ALPHA FEATURES
        # ===================================================================
        # Model confidence score (inverse of uncertainty)
        features['model_confidence'] = float(1.0 / (1.0 + current_spread))
        
        # Prediction asymmetry (skewness of quantile distribution)
        quantiles = [q05, q10, q25, q50, q75, q90, q95]
        if len(quantiles) >= 3:
            features['prediction_skewness'] = float(
                (q75 + q25 - 2*q50) / (q75 - q25 + 1e-10)
            )
        
        # Store current spread for next iteration
        if len(self.performance_history) > 0:
            self.performance_history[-1]['quantile_spread'] = current_spread
        
        return features

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation activities"""

        recent_adaptations = list(
            self.adaptation_history)[-100:]  # Last 100 events

        return {
            'current_performance': {
                'mae': self.current_performance.mae,
                'rmse': self.current_performance.rmse,
                'mape': self.current_performance.mape,
                'direction_accuracy': self.current_performance.direction_accuracy,
                'samples_count': self.current_performance.samples_count,
                'total_drifts_detected': self.current_performance.total_drifts_detected,
                'models_retrained': self.current_performance.models_retrained,
                'incremental_updates': self.current_performance.incremental_updates,
                'partial_retrains': self.current_performance.partial_retrains
            },
            'baseline_performance': {
                'mae': self.baseline_performance.mae if self.baseline_performance else None,
                'rmse': self.baseline_performance.rmse if self.baseline_performance else None,
                'samples_count': self.baseline_performance.samples_count if self.baseline_performance else 0
            },
            'model_info': {
                'age': self.model_age,
                'update_count': self.update_count,
                'last_update': self.last_update_time.isoformat()
            },
            'drift_detectors': {
                name: {
                    'type': type(detector).__name__,
                    'samples_processed': getattr(detector, 'n_samples', 0)
                }
                for name, detector in self.drift_detectors.items()
            },
            'recent_adaptations': len(recent_adaptations),
            'ensemble_weights': self.ensemble_weights.copy() if self.ensemble_weights else {},
            'feature_scaling': {
                'method': self.config.feature_scaling_method,
                'samples_seen': self.feature_scaler.n_samples
            }
        }

    def save_state(self, filepath: str):
        """Save online learning system state"""

        state = {
            'config': self.config,
            'current_performance': self.current_performance,
            'baseline_performance': self.baseline_performance,
            'model_age': self.model_age,
            'update_count': self.update_count,
            'last_update_time': self.last_update_time.isoformat(),
            'ensemble_weights': self.ensemble_weights,
            'adaptation_history': list(self.adaptation_history),
            'performance_history': list(self.performance_history)
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Saved online learning state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self, filepath: str):
        """Load online learning system state"""

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore state
            self.current_performance = OnlinePerformanceMetrics(
                **state['current_performance'])
            if state['baseline_performance']:
                self.baseline_performance = OnlinePerformanceMetrics(
                    **state['baseline_performance'])
            self.model_age = state['model_age']
            self.update_count = state['update_count']
            self.last_update_time = datetime.fromisoformat(
                state['last_update_time'])
            self.ensemble_weights = state['ensemble_weights']

            # Reset recent sample buffer after loading persisted state
            self.recent_samples.clear()

            # Restore histories
            self.adaptation_history.extend(state['adaptation_history'])
            self.performance_history.extend(state['performance_history'])

            logger.info(f"Loaded online learning state from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")


# Example usage and factory functions
def create_online_learning_system(
    enable_drift_detection: bool = True,
    drift_sensitivity: float = 0.05,
    update_frequency: int = 10,
    **kwargs
) -> OnlineLearningSystem:
    """Factory function to create online learning system with sensible defaults"""

    config = OnlineLearningConfig(
        enable_drift_detection=enable_drift_detection,
        drift_sensitivity=drift_sensitivity,
        incremental_update_frequency=update_frequency,
        **kwargs
    )

    return OnlineLearningSystem(config)


if __name__ == "__main__":
    # Example usage
    print("🔄 Online Learning System for AI Price Forecasting")
    print("=" * 60)

    # Create system
    system = create_online_learning_system()

    # Simulate online learning
    rng = np.random.default_rng(42)
    for i in range(100):
        # Simulate features and target
        features = rng.standard_normal(5)
        target = np.sum(features) + rng.standard_normal() * 0.1
        prediction = np.sum(features) + \
            rng.standard_normal() * 0.2  # Noisy prediction

        # Add concept drift halfway through
        if i == 50:
            print("--- Introducing concept drift ---")

        if i >= 50:
            target += 0.5  # Shift in target distribution

        # Update system
        adaptation_info = system.add_sample(features, target, prediction)

        if adaptation_info['drift_detected']:
            print(f"Step {i}: Drift detected!")

        if i % 20 == 0:
            summary = system.get_adaptation_summary()
            print(
                f"Step {i}: MAE={summary['current_performance']['mae']:.4f}, "
                f"Drifts={summary['current_performance']['total_drifts_detected']}")

    # Final summary
    print("\n" + "=" * 60)
    summary = system.get_adaptation_summary()
    print("Final Summary:")
    print(f"  Total Samples: {summary['current_performance']['samples_count']}")
    print(f"  Total Drifts: {summary['current_performance']['total_drifts_detected']}")
    print(f"  MAE: {summary['current_performance']['mae']:.4f}")
    print(f"  Final MAE: {summary['current_performance']['mae']:.6f}")
    print(f"  Drifts Detected: {summary['current_performance']['total_drifts_detected']}")
    print(f"  Incremental Updates: {summary['current_performance']['incremental_updates']}")
    print(f"  Recent Adaptations: {summary['recent_adaptations']}")
    
    
# Add _log_event method to OnlineLearningSystem class
def _log_event_method(self, event_type: str, event_data: dict):
    """Log an event to JSONL file for post-processing during panel merge.
    
    Args:
        event_type: Type of event (drift_detected, model_retrain_triggered, recalibration_needed, etc.)
        event_data: Additional event metadata
    """
    if self.event_log_path is None:
        return  # Event logging not configured
    
    try:
        event_log_path = Path(self.event_log_path)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use current_timestamp if available (data timestamp), otherwise use datetime.now()
        timestamp = getattr(self, 'current_timestamp', None)
        if timestamp is not None:
            timestamp_str = pd.Timestamp(timestamp).isoformat()
        else:
            timestamp_str = datetime.now().isoformat()
        
        event = {
            'timestamp': timestamp_str,
            'symbol': self.symbol,
            'horizon': self.horizon,
            'event_type': event_type,
            'event_data': event_data
        }
        
        with open(event_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    except Exception as e:
        logger.warning(f"Failed to log event {event_type}: {e}")


# Attach _log_event method to OnlineLearningSystem
OnlineLearningSystem._log_event = _log_event_method
