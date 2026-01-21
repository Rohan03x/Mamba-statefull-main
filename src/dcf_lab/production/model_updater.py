"""
Production Model Updater System

This module implements adaptive model updates for the production trading system:
- Light updates for minor drift using online learning
- Scheduled full retraining for major drift
- Model validation and rollback capabilities
- Integration with drift monitoring for automated responses

Key features:
- Incremental learning with concept drift adaptation
- Full model retraining with validation
- Model version management and rollback
- Performance validation before deployment
- Integration with feature store and model registry
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, field
import sqlite3
import json
from pathlib import Path
import joblib
from abc import ABC, abstractmethod

# ML imports
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

@dataclass
class UpdateConfig:
    """Configuration for model updates"""
    
    # Light update parameters
    light_update_batch_size: int = 100
    light_update_learning_rate: float = 0.01
    light_update_window: int = 1000
    
    # Full retrain parameters
    full_retrain_lookback_days: int = 90
    full_retrain_validation_split: float = 0.2
    min_training_samples: int = 500
    
    # Update scheduling
    max_updates_per_day: int = 5
    update_cooldown_hours: int = 2
    
    # Validation thresholds
    min_performance_improvement: float = 0.02  # 2% improvement required
    max_performance_degradation: float = 0.05  # 5% degradation allowed
    
    # Model paths
    model_backup_dir: str = "models/backups"
    temp_model_dir: str = "models/temp"

@dataclass
class UpdateResult:
    """Result of a model update operation"""
    
    timestamp: str
    update_type: str  # 'light', 'full'
    success: bool
    model_version: str
    previous_version: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    rollback_available: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

class BaseModelUpdater(ABC):
    """Abstract base class for model updating strategies"""
    
    @abstractmethod
    def update_model(self, model: BaseEstimator, new_data: pd.DataFrame, 
                    new_targets: np.ndarray) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Update model with new data"""
        pass
    
    @abstractmethod
    def validate_update(self, original_model: BaseEstimator, updated_model: BaseEstimator,
                       validation_data: pd.DataFrame, validation_targets: np.ndarray) -> Dict[str, float]:
        """Validate updated model performance"""
        pass

class LightModelUpdater(BaseModelUpdater):
    """Implements light model updates using online learning"""
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_count = 0
        
    def update_model(self, model: BaseEstimator, new_data: pd.DataFrame, 
                    new_targets: np.ndarray) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Update model incrementally with new data"""
        
        try:
            # Check if model supports partial_fit (online learning)
            if hasattr(model, 'partial_fit'):
                return self._partial_fit_update(model, new_data, new_targets)
            elif hasattr(model, 'set_params'):
                return self._retrain_subset_update(model, new_data, new_targets)
            else:
                raise ValueError("Model does not support incremental updates")
                
        except Exception as e:
            logger.error(f"Light update failed: {e}")
            return model, {'error': str(e)}
    
    def _partial_fit_update(self, model: BaseEstimator, new_data: pd.DataFrame, 
                           new_targets: np.ndarray) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Update using partial_fit for online learning models"""
        
        # Process data in batches for stability
        n_samples = len(new_data)
        batch_losses = []
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            
            batch_X = new_data.iloc[start_idx:end_idx]
            batch_y = new_targets[start_idx:end_idx]
            
            # Apply partial fit
            if self.update_count == 0 and hasattr(model, 'classes_'):
                # For classifiers, need to specify classes on first update
                model.partial_fit(batch_X, batch_y, classes=model.classes_)
            else:
                model.partial_fit(batch_X, batch_y)
            
            # Calculate batch loss for monitoring
            if hasattr(model, 'score'):
                batch_score = model.score(batch_X, batch_y)
                batch_losses.append(1.0 - batch_score)  # Convert to loss
        
        self.update_count += 1
        
        # Return updated model and metrics
        metrics = {
            'batches_processed': len(batch_losses),
            'avg_batch_loss': np.mean(batch_losses) if batch_losses else 0.0,
            'samples_processed': n_samples,
            'update_count': self.update_count
        }
        
        return model, metrics
    
    def _retrain_subset_update(self, model: BaseEstimator, new_data: pd.DataFrame, 
                              new_targets: np.ndarray) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Update by retraining on a subset including new data"""
        
        # For models without partial_fit, we do a limited retrain
        # This is more expensive but still lighter than full retrain
        
        # Limit training data size
        max_training_size = 5000
        if len(new_data) > max_training_size:
            # Sample recent data
            indices = np.random.choice(len(new_data), max_training_size, replace=False)
            training_X = new_data.iloc[indices]
            training_y = new_targets[indices]
        else:
            training_X = new_data
            training_y = new_targets
        
        # Retrain model
        model.fit(training_X, training_y)
        
        # Calculate metrics
        train_score = model.score(training_X, training_y)
        
        metrics = {
            'training_samples': len(training_X),
            'training_score': train_score,
            'retrain_type': 'subset'
        }
        
        return model, metrics
    
    def validate_update(self, original_model: BaseEstimator, updated_model: BaseEstimator,
                       validation_data: pd.DataFrame, validation_targets: np.ndarray) -> Dict[str, float]:
        """Validate light update performance"""
        
        try:
            # Get predictions from both models
            original_pred = original_model.predict(validation_data)
            updated_pred = updated_model.predict(validation_data)
            
            # Calculate metrics for both models
            original_metrics = self._calculate_metrics(validation_targets, original_pred)
            updated_metrics = self._calculate_metrics(validation_targets, updated_pred)
            
            # Calculate improvement
            improvement = {}
            for metric_name in original_metrics:
                original_val = original_metrics[metric_name]
                updated_val = updated_metrics[metric_name]
                
                if metric_name in ['rmse', 'mae']:  # Lower is better
                    improvement[f'{metric_name}_improvement'] = (original_val - updated_val) / original_val
                else:  # Higher is better (r2, etc.)
                    improvement[f'{metric_name}_improvement'] = (updated_val - original_val) / abs(original_val) if original_val != 0 else 0
            
            # Combine all metrics
            validation_metrics = {
                **{f'original_{k}': v for k, v in original_metrics.items()},
                **{f'updated_{k}': v for k, v in updated_metrics.items()},
                **improvement
            }
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'validation_error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

class FullModelUpdater(BaseModelUpdater):
    """Implements full model retraining"""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        
    def update_model(self, model: BaseEstimator, new_data: pd.DataFrame, 
                    new_targets: np.ndarray) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Fully retrain model with new data"""
        
        try:
            # Prepare training data
            training_X, training_y = self._prepare_training_data(new_data, new_targets)
            
            if len(training_X) < self.config.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(training_X)} < {self.config.min_training_samples}")
            
            # Clone model to avoid modifying original
            from sklearn.base import clone
            new_model = clone(model)
            
            # Retrain
            logger.info(f"Starting full retrain with {len(training_X)} samples")
            new_model.fit(training_X, training_y)
            
            # Calculate training metrics
            train_pred = new_model.predict(training_X)
            training_metrics = {
                'training_samples': len(training_X),
                'training_rmse': np.sqrt(mean_squared_error(training_y, train_pred)),
                'training_mae': mean_absolute_error(training_y, train_pred),
                'training_r2': r2_score(training_y, train_pred)
            }
            
            logger.info(f"Full retrain completed. Training R²: {training_metrics['training_r2']:.4f}")
            
            return new_model, training_metrics
            
        except Exception as e:
            logger.error(f"Full retrain failed: {e}")
            return model, {'error': str(e)}
    
    def _prepare_training_data(self, new_data: pd.DataFrame, 
                              new_targets: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for full retrain"""
        
        # For now, just use the provided data
        # In a real system, this would combine with historical data from feature store
        
        # Remove any invalid data
        valid_mask = ~(pd.isna(new_data).any(axis=1) | pd.isna(new_targets))
        
        clean_X = new_data[valid_mask].copy()
        clean_y = new_targets[valid_mask]
        
        logger.info(f"Prepared {len(clean_X)} clean samples for training")
        
        return clean_X, clean_y
    
    def validate_update(self, original_model: BaseEstimator, updated_model: BaseEstimator,
                       validation_data: pd.DataFrame, validation_targets: np.ndarray) -> Dict[str, float]:
        """Validate full retrain performance"""
        
        try:
            # Split validation data for proper evaluation
            val_size = len(validation_data)
            split_idx = int(val_size * 0.5)
            
            # Use first half for comparison
            val_X_1 = validation_data.iloc[:split_idx]
            val_y_1 = validation_targets[:split_idx]
            
            # Use second half for final validation
            val_X_2 = validation_data.iloc[split_idx:]
            val_y_2 = validation_targets[split_idx:]
            
            # Get predictions
            original_pred_1 = original_model.predict(val_X_1)
            updated_pred_1 = updated_model.predict(val_X_1)
            
            original_pred_2 = original_model.predict(val_X_2)
            updated_pred_2 = updated_model.predict(val_X_2)
            
            # Calculate metrics
            metrics = {}
            
            # First validation set (comparison)
            metrics.update({
                'original_rmse_1': np.sqrt(mean_squared_error(val_y_1, original_pred_1)),
                'updated_rmse_1': np.sqrt(mean_squared_error(val_y_1, updated_pred_1)),
                'original_r2_1': r2_score(val_y_1, original_pred_1),
                'updated_r2_1': r2_score(val_y_1, updated_pred_1)
            })
            
            # Second validation set (final)
            metrics.update({
                'original_rmse_2': np.sqrt(mean_squared_error(val_y_2, original_pred_2)),
                'updated_rmse_2': np.sqrt(mean_squared_error(val_y_2, updated_pred_2)),
                'original_r2_2': r2_score(val_y_2, original_pred_2),
                'updated_r2_2': r2_score(val_y_2, updated_pred_2)
            })
            
            # Calculate improvements
            rmse_improvement = (metrics['original_rmse_2'] - metrics['updated_rmse_2']) / metrics['original_rmse_2']
            r2_improvement = (metrics['updated_r2_2'] - metrics['original_r2_2']) / abs(metrics['original_r2_2']) if metrics['original_r2_2'] != 0 else 0
            
            metrics.update({
                'rmse_improvement': rmse_improvement,
                'r2_improvement': r2_improvement,
                'validation_samples': len(validation_data)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Full retrain validation failed: {e}")
            return {'validation_error': str(e)}

class ModelVersionManager:
    """Manages model versions and rollback capabilities"""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.versions_db_path = self.base_path / "versions.db"
        self._init_version_db()
        
    def _init_version_db(self):
        """Initialize model version database"""
        
        conn = sqlite3.connect(self.versions_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                performance_metrics TEXT,
                validation_metrics TEXT,
                model_path TEXT NOT NULL,
                is_active BOOLEAN DEFAULT FALSE,
                parent_version TEXT,
                update_type TEXT,
                notes TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_model_version(self, model: BaseEstimator, version_id: str, 
                          update_result: UpdateResult) -> str:
        """Save a new model version"""
        
        # Create version directory
        version_dir = self.base_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'version_id': version_id,
            'timestamp': update_result.timestamp,
            'update_type': update_result.update_type,
            'performance_metrics': update_result.performance_metrics,
            'validation_metrics': update_result.validation_metrics
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Record in database
        conn = sqlite3.connect(self.versions_db_path)
        conn.execute("""
            INSERT INTO model_versions 
            (version_id, timestamp, model_type, performance_metrics, validation_metrics,
             model_path, parent_version, update_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            update_result.timestamp,
            type(model).__name__,
            json.dumps(update_result.performance_metrics),
            json.dumps(update_result.validation_metrics),
            str(model_path),
            update_result.previous_version,
            update_result.update_type
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Saved model version {version_id}")
        return str(model_path)
    
    def load_model_version(self, version_id: str) -> Optional[BaseEstimator]:
        """Load a specific model version"""
        
        try:
            version_dir = self.base_path / version_id
            model_path = version_dir / "model.pkl"
            
            if model_path.exists():
                return joblib.load(model_path)
            else:
                logger.error(f"Model file not found for version {version_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model version {version_id}: {e}")
            return None
    
    def set_active_version(self, version_id: str):
        """Set a version as the active production model"""
        
        conn = sqlite3.connect(self.versions_db_path)
        
        # Deactivate all versions
        conn.execute("UPDATE model_versions SET is_active = FALSE")
        
        # Activate specified version
        conn.execute("UPDATE model_versions SET is_active = TRUE WHERE version_id = ?", (version_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Set version {version_id} as active")
    
    def get_active_version(self) -> Optional[str]:
        """Get the currently active model version"""
        
        conn = sqlite3.connect(self.versions_db_path)
        cursor = conn.execute("SELECT version_id FROM model_versions WHERE is_active = TRUE")
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_version_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent version history"""
        
        conn = sqlite3.connect(self.versions_db_path)
        cursor = conn.execute("""
            SELECT version_id, timestamp, model_type, performance_metrics, 
                   validation_metrics, is_active, update_type
            FROM model_versions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'version_id': row[0],
                'timestamp': row[1],
                'model_type': row[2],
                'performance_metrics': json.loads(row[3]) if row[3] else {},
                'validation_metrics': json.loads(row[4]) if row[4] else {},
                'is_active': bool(row[5]),
                'update_type': row[6]
            })
        
        return history

class ProductionModelUpdater:
    """Main model updater for production system"""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        
        # Initialize updaters
        self.light_updater = LightModelUpdater(
            learning_rate=self.config.light_update_learning_rate,
            batch_size=self.config.light_update_batch_size
        )
        self.full_updater = FullModelUpdater(self.config)
        
        # Initialize version manager
        self.version_manager = ModelVersionManager()
        
        # Update tracking
        self.update_history = []
        self.last_update_time = None
        self.daily_update_count = 0
        self.last_reset_date = datetime.now().date()
        
    def light_update(self, model: BaseEstimator, new_data: pd.DataFrame, 
                    new_targets: np.ndarray, validation_data: Optional[pd.DataFrame] = None,
                    validation_targets: Optional[np.ndarray] = None) -> UpdateResult:
        """Perform light model update"""
        
        if not self._can_update():
            return UpdateResult(
                timestamp=datetime.now().isoformat(),
                update_type='light',
                success=False,
                model_version='',
                previous_version='',
                error_message="Update rate limit exceeded"
            )
        
        start_time = datetime.now()
        current_version = self.version_manager.get_active_version() or 'unknown'
        
        try:
            # Perform light update
            updated_model, update_metrics = self.light_updater.update_model(
                model, new_data, new_targets
            )
            
            # Validate if validation data provided
            validation_metrics = {}
            if validation_data is not None and validation_targets is not None:
                validation_metrics = self.light_updater.validate_update(
                    model, updated_model, validation_data, validation_targets
                )
            
            # Check if update is acceptable
            if not self._validate_update_quality(validation_metrics, 'light'):
                return UpdateResult(
                    timestamp=start_time.isoformat(),
                    update_type='light',
                    success=False,
                    model_version=current_version,
                    previous_version=current_version,
                    performance_metrics=update_metrics,
                    validation_metrics=validation_metrics,
                    error_message="Update failed validation checks"
                )
            
            # Create new version
            new_version = self._generate_version_id('light')
            
            # Create update result
            update_result = UpdateResult(
                timestamp=start_time.isoformat(),
                update_type='light',
                success=True,
                model_version=new_version,
                previous_version=current_version,
                performance_metrics=update_metrics,
                validation_metrics=validation_metrics,
                details={
                    'samples_processed': len(new_data),
                    'update_duration': (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Save updated model
            self.version_manager.save_model_version(updated_model, new_version, update_result)
            self.version_manager.set_active_version(new_version)
            
            # Update tracking
            self._record_update(update_result)
            
            logger.info(f"Light update completed: {new_version}")
            return update_result
            
        except Exception as e:
            logger.error(f"Light update failed: {e}")
            return UpdateResult(
                timestamp=start_time.isoformat(),
                update_type='light',
                success=False,
                model_version=current_version,
                previous_version=current_version,
                error_message=str(e)
            )
    
    def full_retrain(self, model: BaseEstimator, training_data: pd.DataFrame, 
                    training_targets: np.ndarray, validation_data: pd.DataFrame,
                    validation_targets: np.ndarray) -> UpdateResult:
        """Perform full model retrain"""
        
        start_time = datetime.now()
        current_version = self.version_manager.get_active_version() or 'unknown'
        
        try:
            # Perform full retrain
            updated_model, training_metrics = self.full_updater.update_model(
                model, training_data, training_targets
            )
            
            # Validate update
            validation_metrics = self.full_updater.validate_update(
                model, updated_model, validation_data, validation_targets
            )
            
            # Check if update is acceptable
            if not self._validate_update_quality(validation_metrics, 'full'):
                return UpdateResult(
                    timestamp=start_time.isoformat(),
                    update_type='full',
                    success=False,
                    model_version=current_version,
                    previous_version=current_version,
                    performance_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    error_message="Full retrain failed validation checks"
                )
            
            # Create new version
            new_version = self._generate_version_id('full')
            
            # Create update result
            update_result = UpdateResult(
                timestamp=start_time.isoformat(),
                update_type='full',
                success=True,
                model_version=new_version,
                previous_version=current_version,
                performance_metrics=training_metrics,
                validation_metrics=validation_metrics,
                details={
                    'training_samples': len(training_data),
                    'validation_samples': len(validation_data),
                    'retrain_duration': (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Save updated model
            self.version_manager.save_model_version(updated_model, new_version, update_result)
            
            # Only activate if significantly better than current
            if self._should_activate_new_model(validation_metrics):
                self.version_manager.set_active_version(new_version)
                logger.info(f"Full retrain completed and activated: {new_version}")
            else:
                logger.info(f"Full retrain completed but not activated: {new_version}")
                update_result.details['activated'] = False
            
            # Update tracking
            self._record_update(update_result)
            
            return update_result
            
        except Exception as e:
            logger.error(f"Full retrain failed: {e}")
            return UpdateResult(
                timestamp=start_time.isoformat(),
                update_type='full',
                success=False,
                model_version=current_version,
                previous_version=current_version,
                error_message=str(e)
            )
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to a specific model version"""
        
        try:
            # Check if version exists
            model = self.version_manager.load_model_version(version_id)
            if model is None:
                logger.error(f"Cannot rollback: version {version_id} not found")
                return False
            
            # Set as active
            self.version_manager.set_active_version(version_id)
            
            logger.info(f"Rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _can_update(self) -> bool:
        """Check if update is allowed based on rate limits"""
        
        current_date = datetime.now().date()
        
        # Reset daily counter if new day
        if current_date != self.last_reset_date:
            self.daily_update_count = 0
            self.last_reset_date = current_date
        
        # Check daily limit
        if self.daily_update_count >= self.config.max_updates_per_day:
            return False
        
        # Check cooldown period
        if self.last_update_time:
            time_since_last = datetime.now() - self.last_update_time
            if time_since_last < timedelta(hours=self.config.update_cooldown_hours):
                return False
        
        return True
    
    def _validate_update_quality(self, validation_metrics: Dict[str, float], 
                                update_type: str) -> bool:
        """Validate that update meets quality requirements"""
        
        if not validation_metrics or 'validation_error' in validation_metrics:
            return False
        
        # For light updates, allow small degradations
        if update_type == 'light':
            if 'rmse_improvement' in validation_metrics:
                # Allow small degradation for light updates
                return validation_metrics['rmse_improvement'] > -self.config.max_performance_degradation
        
        # For full retrains, require improvement
        elif update_type == 'full':
            if 'rmse_improvement' in validation_metrics:
                return validation_metrics['rmse_improvement'] > self.config.min_performance_improvement
        
        return True
    
    def _should_activate_new_model(self, validation_metrics: Dict[str, float]) -> bool:
        """Decide whether to activate a newly trained model"""
        
        if not validation_metrics:
            return False
        
        # Require significant improvement for activation
        if 'rmse_improvement' in validation_metrics:
            improvement = validation_metrics['rmse_improvement']
            return improvement > self.config.min_performance_improvement * 1.5  # 1.5x threshold
        
        return False
    
    def _generate_version_id(self, update_type: str) -> str:
        """Generate unique version ID"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_count = len(self.update_history) + 1
        
        return f"{update_type}_{timestamp}_{update_count:04d}"
    
    def _record_update(self, update_result: UpdateResult):
        """Record update in history"""
        
        self.update_history.append(update_result)
        self.last_update_time = datetime.now()
        self.daily_update_count += 1
        
    def get_current_model(self) -> Optional[BaseEstimator]:
        """Get the currently active model"""
        
        active_version = self.version_manager.get_active_version()
        if active_version:
            return self.version_manager.load_model_version(active_version)
        return None
    
    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of recent updates"""
        
        return {
            'active_version': self.version_manager.get_active_version(),
            'total_updates': len(self.update_history),
            'daily_updates': self.daily_update_count,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'recent_updates': [
                {
                    'timestamp': ur.timestamp,
                    'type': ur.update_type,
                    'success': ur.success,
                    'version': ur.model_version
                } for ur in self.update_history[-5:]
            ],
            'version_history': self.version_manager.get_version_history(5)
        }

class UpdateScheduler:
    """Schedules and coordinates model updates"""
    
    def __init__(self, updater: ProductionModelUpdater):
        self.updater = updater
        self.scheduled_updates = []
        self.is_running = False
        
    def schedule_light_update(self, data_source: Callable, delay_minutes: int = 0):
        """Schedule a light update"""
        
        schedule_time = datetime.now() + timedelta(minutes=delay_minutes)
        
        update_task = {
            'type': 'light',
            'scheduled_time': schedule_time,
            'data_source': data_source,
            'status': 'scheduled'
        }
        
        self.scheduled_updates.append(update_task)
        logger.info(f"Scheduled light update for {schedule_time}")
    
    def schedule_full_retrain(self, data_source: Callable, delay_hours: int = 0):
        """Schedule a full retrain"""
        
        schedule_time = datetime.now() + timedelta(hours=delay_hours)
        
        update_task = {
            'type': 'full',
            'scheduled_time': schedule_time,
            'data_source': data_source,
            'status': 'scheduled'
        }
        
        self.scheduled_updates.append(update_task)
        logger.info(f"Scheduled full retrain for {schedule_time}")
    
    def process_scheduled_updates(self) -> List[UpdateResult]:
        """Process any due scheduled updates"""
        
        if not self.is_running:
            return []
        
        current_time = datetime.now()
        results = []
        
        for task in self.scheduled_updates:
            if task['status'] == 'scheduled' and current_time >= task['scheduled_time']:
                
                task['status'] = 'running'
                
                try:
                    # Get data from source
                    data = task['data_source']()
                    
                    # Execute update
                    if task['type'] == 'light':
                        result = self._execute_light_update(data)
                    else:
                        result = self._execute_full_retrain(data)
                    
                    results.append(result)
                    task['status'] = 'completed'
                    
                except Exception as e:
                    logger.error(f"Scheduled update failed: {e}")
                    task['status'] = 'failed'
                    task['error'] = str(e)
        
        return results
    
    def _execute_light_update(self, data: Dict[str, Any]) -> UpdateResult:
        """Execute scheduled light update"""
        
        current_model = self.updater.get_current_model()
        
        return self.updater.light_update(
            current_model,
            data['features'],
            data['targets'],
            data.get('validation_features'),
            data.get('validation_targets')
        )
    
    def _execute_full_retrain(self, data: Dict[str, Any]) -> UpdateResult:
        """Execute scheduled full retrain"""
        
        current_model = self.updater.get_current_model()
        
        return self.updater.full_retrain(
            current_model,
            data['training_features'],
            data['training_targets'],
            data['validation_features'],
            data['validation_targets']
        )
    
    def start_scheduler(self):
        """Start the update scheduler"""
        self.is_running = True
        logger.info("Update scheduler started")
    
    def stop_scheduler(self):
        """Stop the update scheduler"""
        self.is_running = False
        logger.info("Update scheduler stopped")
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get current schedule status"""
        
        return {
            'is_running': self.is_running,
            'scheduled_updates': len([t for t in self.scheduled_updates if t['status'] == 'scheduled']),
            'completed_updates': len([t for t in self.scheduled_updates if t['status'] == 'completed']),
            'failed_updates': len([t for t in self.scheduled_updates if t['status'] == 'failed']),
            'next_update': min([t['scheduled_time'] for t in self.scheduled_updates if t['status'] == 'scheduled'], default=None)
        }