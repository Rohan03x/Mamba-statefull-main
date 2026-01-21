"""
Streaming Framework for Incremental Learning

This module provides streaming data processing and incremental learning
capabilities for concept drift adaptation, including:

1. StreamingFramework: Core streaming data pipeline
2. RiverIntegration: Integration with River ML library
3. IncrementalUpdater: Batch incremental updates
4. StreamingConfig: Configuration management

Designed for real-time financial data processing with automatic
model updates between full retraining cycles.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming processing modes"""
    BATCH = "batch"              # Process in batches
    ONLINE = "online"            # Process one sample at a time
    MICRO_BATCH = "micro_batch"  # Small batch processing
    ADAPTIVE = "adaptive"        # Adaptive batch sizing


@dataclass
class StreamingConfig:
    """Configuration for streaming framework"""
    mode: StreamingMode = StreamingMode.MICRO_BATCH
    batch_size: int = 10
    max_buffer_size: int = 1000
    update_frequency: int = 100
    enable_river: bool = True
    enable_partial_fit: bool = True
    performance_tracking: bool = True
    drift_adaptation: bool = True
    memory_limit_mb: int = 500
    
    # Financial-specific settings
    enable_market_hours: bool = True
    market_open_hour: int = 9
    market_close_hour: int = 16
    timezone: str = "US/Eastern"
    
    # Adaptive settings
    adaptive_batch_min: int = 5
    adaptive_batch_max: int = 50
    performance_threshold: float = 0.95


class StreamingFramework:
    """
    Core streaming framework for incremental learning
    
    Processes streaming financial data and performs incremental
    model updates with drift adaptation.
    """
    
    def __init__(self, config: StreamingConfig):
        """
        Initialize streaming framework
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        
        # Streaming state
        self.buffer = deque(maxlen=config.max_buffer_size)
        self.current_batch = []
        self.sample_count = 0
        self.last_update_time = None
        self.is_running = False
        
        # Performance tracking
        self.performance_history = []
        self.throughput_history = []
        self.memory_usage = []
        
        # Adaptive batch sizing
        self.current_batch_size = config.batch_size
        self.performance_window = deque(maxlen=10)
        
        logger.info(f"🌊 Initialized streaming framework: mode={config.mode.value}, batch={config.batch_size}")
    
    def start_stream(self):
        """Start the streaming pipeline"""
        self.is_running = True
        self.last_update_time = pd.Timestamp.now()
        logger.info("🚀 Streaming pipeline started")
    
    def stop_stream(self):
        """Stop the streaming pipeline"""
        self.is_running = False
        logger.info("⏹️ Streaming pipeline stopped")
    
    def add_sample(self, features: Dict[str, float], target: float, 
                  timestamp: Optional[pd.Timestamp] = None) -> bool:
        """
        Add a new sample to the stream
        
        Args:
            features: Feature dictionary
            target: Target value
            timestamp: Optional timestamp
            
        Returns:
            True if batch is ready for processing
        """
        if not self.is_running:
            logger.warning("Stream not running, sample ignored")
            return False
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Check market hours if enabled
        if self.config.enable_market_hours and not self._is_market_hours(timestamp):
            return False
        
        # Add to buffer
        sample = {
            'features': features,
            'target': target,
            'timestamp': timestamp,
            'sample_id': self.sample_count
        }
        
        self.buffer.append(sample)
        self.current_batch.append(sample)
        self.sample_count += 1
        
        # Check if batch is ready
        return self._check_batch_ready()
    
    def get_batch(self) -> Optional[Dict[str, Any]]:
        """
        Get the current batch for processing
        
        Returns:
            Batch data dictionary or None if no batch ready
        """
        if not self.current_batch:
            return None
        
        # Extract batch data
        batch_features = []
        batch_targets = []
        batch_timestamps = []
        
        for sample in self.current_batch:
            batch_features.append(list(sample['features'].values()))
            batch_targets.append(sample['target'])
            batch_timestamps.append(sample['timestamp'])
        
        batch_data = {
            'X': np.array(batch_features),
            'y': np.array(batch_targets),
            'timestamps': batch_timestamps,
            'batch_size': len(self.current_batch),
            'feature_names': list(self.current_batch[0]['features'].keys()) if self.current_batch else []
        }
        
        # Clear current batch
        self.current_batch = []
        
        # Update throughput tracking
        self._update_throughput_tracking(batch_data['batch_size'])
        
        # Adaptive batch size adjustment
        if self.config.mode == StreamingMode.ADAPTIVE:
            self._adjust_batch_size()
        
        return batch_data
    
    def get_buffer_data(self, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Get data from buffer for model training
        
        Args:
            n_samples: Number of samples to retrieve (None for all)
            
        Returns:
            Buffer data dictionary
        """
        if not self.buffer:
            return {'X': np.array([]), 'y': np.array([]), 'timestamps': []}
        
        # Get samples from buffer
        samples = list(self.buffer)
        if n_samples:
            samples = samples[-n_samples:]
        
        # Extract data
        features = []
        targets = []
        timestamps = []
        
        for sample in samples:
            features.append(list(sample['features'].values()))
            targets.append(sample['target'])
            timestamps.append(sample['timestamp'])
        
        return {
            'X': np.array(features),
            'y': np.array(targets),
            'timestamps': timestamps,
            'n_samples': len(samples),
            'feature_names': list(samples[0]['features'].keys()) if samples else []
        }
    
    def _check_batch_ready(self) -> bool:
        """Check if current batch is ready for processing"""
        
        if self.config.mode == StreamingMode.ONLINE:
            return len(self.current_batch) >= 1
        elif self.config.mode == StreamingMode.BATCH:
            return len(self.current_batch) >= self.config.batch_size
        elif self.config.mode == StreamingMode.MICRO_BATCH:
            return len(self.current_batch) >= self.current_batch_size
        elif self.config.mode == StreamingMode.ADAPTIVE:
            return len(self.current_batch) >= self.current_batch_size
        
        return False
    
    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within market hours"""
        
        try:
            # Convert to market timezone
            market_time = timestamp.tz_convert(self.config.timezone) if timestamp.tz else timestamp
            
            # Check if weekday (Monday=0, Sunday=6)
            if market_time.weekday() >= 5:  # Weekend
                return False
            
            # Check hour range
            hour = market_time.hour
            return self.config.market_open_hour <= hour < self.config.market_close_hour
            
        except Exception:
            # If timezone conversion fails, assume market hours
            return True
    
    def _update_throughput_tracking(self, batch_size: int):
        """Update throughput performance tracking"""
        
        current_time = pd.Timestamp.now()
        if self.last_update_time:
            time_diff = (current_time - self.last_update_time).total_seconds()
            throughput = batch_size / max(time_diff, 0.001)  # samples per second
            
            self.throughput_history.append({
                'timestamp': current_time,
                'throughput': throughput,
                'batch_size': batch_size,
                'time_diff': time_diff
            })
            
            # Limit history
            if len(self.throughput_history) > 100:
                self.throughput_history.pop(0)
        
        self.last_update_time = current_time
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance"""
        
        if len(self.performance_window) < 3:
            return
        
        # Calculate recent performance trend
        recent_performance = np.mean(list(self.performance_window)[-3:])
        
        if recent_performance > self.config.performance_threshold:
            # Performance is good, can increase batch size
            self.current_batch_size = min(
                self.current_batch_size + 1,
                self.config.adaptive_batch_max
            )
        else:
            # Performance declining, reduce batch size
            self.current_batch_size = max(
                self.current_batch_size - 1,
                self.config.adaptive_batch_min
            )
        
        logger.debug(f"🔧 Adjusted batch size to {self.current_batch_size}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming framework statistics"""
        
        avg_throughput = np.mean([h['throughput'] for h in self.throughput_history]) if self.throughput_history else 0
        
        return {
            'is_running': self.is_running,
            'total_samples': self.sample_count,
            'buffer_size': len(self.buffer),
            'current_batch_size': self.current_batch_size,
            'avg_throughput': avg_throughput,
            'performance_history_length': len(self.performance_history),
            'mode': self.config.mode.value,
            'last_update': self.last_update_time
        }


class IncrementalUpdater:
    """
    Incremental model updater for streaming data
    
    Performs incremental updates using partial_fit or River integration
    """
    
    def __init__(self, 
                 streaming_framework: StreamingFramework,
                 enable_river: bool = True):
        """
        Initialize incremental updater
        
        Args:
            streaming_framework: Associated streaming framework
            enable_river: Enable River ML integration
        """
        self.streaming_framework = streaming_framework
        self.enable_river = enable_river
        
        # Update tracking
        self.models = {}
        self.update_history = []
        self.performance_metrics = {}
        
        logger.info(f"🔄 Initialized incremental updater: river={enable_river}")
    
    def register_model(self, model_name: str, model: Any, model_type: str = "sklearn"):
        """
        Register a model for incremental updates
        
        Args:
            model_name: Name identifier for the model
            model: Model object
            model_type: Type of model (sklearn, river, custom)
        """
        self.models[model_name] = {
            'model': model,
            'type': model_type,
            'last_update': None,
            'update_count': 0,
            'performance_history': []
        }
        
        logger.info(f"📝 Registered model '{model_name}' of type {model_type}")
    
    def update_models(self, batch_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update all registered models with new batch
        
        Args:
            batch_data: Batch data from streaming framework
            
        Returns:
            Dictionary of model_name -> success status
        """
        results = {}
        
        for model_name, model_info in self.models.items():
            try:
                success = self._update_single_model(model_name, model_info, batch_data)
                results[model_name] = success
                
                if success:
                    model_info['update_count'] += 1
                    model_info['last_update'] = pd.Timestamp.now()
                
            except Exception as e:
                logger.error(f"❌ Failed to update model '{model_name}': {e}")
                results[model_name] = False
        
        return results
    
    def _update_single_model(self, model_name: str, model_info: Dict[str, Any], 
                           batch_data: Dict[str, Any]) -> bool:
        """Update a single model with batch data"""
        
        model = model_info['model']
        model_type = model_info['type']
        
        X = batch_data['X']
        y = batch_data['y']
        
        if len(X) == 0:
            return False
        
        try:
            if model_type == "sklearn":
                return self._update_sklearn_model(model, X, y)
            elif model_type == "river":
                return self._update_river_model(model, X, y)
            else:
                return self._update_custom_model(model, X, y)
                
        except Exception as e:
            logger.error(f"Model update failed for {model_name}: {e}")
            return False
    
    def _update_sklearn_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> bool:
        """Update sklearn model with partial_fit"""
        
        if hasattr(model, 'partial_fit'):
            # Use partial_fit for incremental learning
            model.partial_fit(X, y)
            return True
        elif hasattr(model, 'warm_start'):
            # Use warm start for ensemble models
            model.set_params(warm_start=True)
            model.fit(X, y)
            return True
        else:
            logger.warning("Model doesn't support incremental updates")
            return False
    
    def _update_river_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> bool:
        """Update River model (placeholder for River integration)"""
        
        # Placeholder for River integration
        # In actual implementation, would use River's learn_one or learn_many
        logger.debug("River update (placeholder)")
        return True
    
    def _update_custom_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> bool:
        """Update custom model"""
        
        if hasattr(model, 'incremental_update'):
            model.incremental_update(X, y)
            return True
        else:
            return False
    
    def get_model_performance(self, model_name: str, test_data: Dict[str, Any]) -> Optional[float]:
        """
        Get current model performance
        
        Args:
            model_name: Name of the model
            test_data: Test data for evaluation
            
        Returns:
            Performance score or None if evaluation fails
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]['model']
        
        try:
            X_test = test_data['X']
            y_test = test_data['y']
            
            if hasattr(model, 'score'):
                return model.score(X_test, y_test)
            elif hasattr(model, 'predict'):
                predictions = model.predict(X_test)
                # Simple MSE for regression
                return -np.mean((predictions - y_test) ** 2)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Performance evaluation failed for {model_name}: {e}")
            return None


class RiverIntegration:
    """
    Integration with River ML library for online learning
    
    Provides wrapper for River models to work with the streaming framework
    """
    
    def __init__(self):
        """Initialize River integration"""
        self.river_available = self._check_river_availability()
        self.river_models = {}
        
        if self.river_available:
            logger.info("🌊 River ML integration available")
        else:
            logger.warning("⚠️ River ML not available, falling back to sklearn")
    
    def _check_river_availability(self) -> bool:
        """Check if River is available"""
        try:
            import river
            return True
        except ImportError:
            return False
    
    def create_river_model(self, model_type: str, **kwargs) -> Optional[Any]:
        """
        Create a River model
        
        Args:
            model_type: Type of River model to create
            **kwargs: Model parameters
            
        Returns:
            River model instance or None if River not available
        """
        if not self.river_available:
            return None
        
        try:
            import river
            from river import linear_model, ensemble, tree
            
            if model_type == "linear":
                return linear_model.LinearRegression(**kwargs)
            elif model_type == "logistic":
                return linear_model.LogisticRegression(**kwargs)
            elif model_type == "adaptive_random_forest":
                return ensemble.AdaptiveRandomForestRegressor(**kwargs)
            elif model_type == "hoeffding_tree":
                return tree.HoeffdingTreeRegressor(**kwargs)
            else:
                logger.warning(f"Unknown River model type: {model_type}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to create River model: {e}")
            return None
    
    def update_river_model(self, model: Any, features: Dict[str, float], target: float) -> bool:
        """
        Update a River model with a single sample
        
        Args:
            model: River model instance
            features: Feature dictionary
            target: Target value
            
        Returns:
            True if update successful
        """
        if not self.river_available:
            return False
        
        try:
            # River models use learn_one for single sample updates
            if hasattr(model, 'learn_one'):
                model.learn_one(features, target)
                return True
            else:
                logger.warning("River model doesn't support learn_one")
                return False
                
        except Exception as e:
            logger.error(f"River model update failed: {e}")
            return False
    
    def predict_with_river_model(self, model: Any, features: Dict[str, float]) -> Optional[float]:
        """
        Make prediction with River model
        
        Args:
            model: River model instance
            features: Feature dictionary
            
        Returns:
            Prediction or None if failed
        """
        if not self.river_available:
            return None
        
        try:
            if hasattr(model, 'predict_one'):
                return model.predict_one(features)
            else:
                logger.warning("River model doesn't support predict_one")
                return None
                
        except Exception as e:
            logger.error(f"River prediction failed: {e}")
            return None