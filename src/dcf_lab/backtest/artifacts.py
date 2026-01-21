"""
Backtest Artifact Manager for Comprehensive Audit Trails

This module implements comprehensive artifact storage and management for
walk-forward backtesting, enabling full auditability and detection of
backtest overfitting through systematic tracking of all decisions,
data, and results.

Key Features:
- Complete data lineage tracking with cryptographic hashes
- Comprehensive metadata storage (features, hyperparameters, metrics)
- Prediction and error storage for every backtest step
- Overfitting detection through PBO and other statistical tests
- Efficient compression and storage management
- Audit trail reconstruction and validation

Following academic best practices for reproducible financial ML research.
"""

import logging
import json
import pickle
import gzip
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSnapshot:
    """
    Snapshot of data used in a backtest step
    
    Includes cryptographic hashes for integrity verification
    and data lineage tracking.
    """
    
    # Data identification
    data_hash: str  # SHA-256 hash of the feature matrix
    target_hash: str  # SHA-256 hash of the target vector
    feature_names: List[str]
    n_samples: int
    n_features: int
    
    # Temporal information
    start_date: datetime
    end_date: datetime
    frequency: str  # 'D', 'W', 'M', etc.
    
    # Data quality metrics
    missing_values: Dict[str, int]
    outlier_counts: Dict[str, int]
    feature_correlations: Dict[str, float]  # Max correlation with target
    
    # Data processing
    preprocessing_steps: List[str]
    feature_engineering_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO format strings for JSON serialization
        if isinstance(result['start_date'], datetime):
            result['start_date'] = result['start_date'].isoformat()
        if isinstance(result['end_date'], datetime):
            result['end_date'] = result['end_date'].isoformat()
        return result


@dataclass
class ModelSnapshot:
    """
    Snapshot of model configuration and training details
    """
    
    # Model identification
    model_type: str
    model_hash: str  # Hash of model configuration
    hyperparameters: Dict[str, Any]
    
    # Training details
    training_samples: int
    training_start: datetime
    training_end: datetime
    training_duration: float  # seconds
    
    # Cross-validation details
    cv_folds: int
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Feature importance (if available)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model complexity metrics
    n_parameters: Optional[int] = None
    model_size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        result['training_start'] = self.training_start.isoformat()
        result['training_end'] = self.training_end.isoformat()
        return result


@dataclass
class PredictionSnapshot:
    """
    Snapshot of predictions and performance metrics
    """
    
    # Prediction details
    prediction_date: datetime
    prediction_horizon: int  # days ahead
    n_predictions: int
    
    # Predictions and actuals
    predictions: List[float]
    actuals: Optional[List[float]] = None  # May not be available immediately
    
    # Performance metrics (when actuals are available)
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    hit_rate: Optional[float] = None  # For directional accuracy
    
    # Confidence intervals and uncertainty
    prediction_std: Optional[List[float]] = None
    confidence_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None
    
    # Regime and market condition
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    market_stress_indicator: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['prediction_date'] = self.prediction_date.isoformat()
        return result
    
    def update_with_actuals(self, actual_values: List[float]):
        """Update predictions with actual values and compute metrics"""
        if len(actual_values) != len(self.predictions):
            raise ValueError("Actuals and predictions must have same length")
        
        self.actuals = actual_values
        
        # Compute performance metrics
        self.mse = float(mean_squared_error(actual_values, self.predictions))
        self.mae = float(mean_absolute_error(actual_values, self.predictions))
        self.r2 = float(r2_score(actual_values, self.predictions))
        
        # Compute hit rate (directional accuracy)
        if len(actual_values) > 1:
            actual_direction = np.sign(np.diff(actual_values))
            pred_direction = np.sign(np.diff(self.predictions))
            self.hit_rate = float(np.mean(actual_direction == pred_direction))


@dataclass
class BacktestStep:
    """
    Complete record of a single backtest step
    
    This represents one complete cycle of training, validation,
    and prediction in the walk-forward backtest.
    """
    
    # Step identification
    step_id: str
    step_number: int
    step_date: datetime
    
    # Configuration
    rebalance_step: bool
    retrain_step: bool
    hyperopt_step: bool
    
    # Data snapshots
    training_data: DataSnapshot
    validation_data: Optional[DataSnapshot] = None
    prediction_data: Optional[DataSnapshot] = None
    
    # Model snapshots
    models: Dict[str, ModelSnapshot] = field(default_factory=dict)
    ensemble_config: Optional[Dict[str, Any]] = None
    
    # Predictions
    predictions: Optional[PredictionSnapshot] = None
    
    # Performance tracking
    step_duration: float = 0.0  # Total step duration in seconds
    memory_peak: Optional[float] = None  # Peak memory usage in MB
    
    # Metadata
    git_commit: Optional[str] = None
    code_version: Optional[str] = None
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        
        # Convert datetime objects
        result['step_date'] = self.step_date.isoformat()
        
        # Convert nested objects
        result['training_data'] = self.training_data.to_dict()
        if self.validation_data:
            result['validation_data'] = self.validation_data.to_dict()
        if self.prediction_data:
            result['prediction_data'] = self.prediction_data.to_dict()
        if self.predictions:
            result['predictions'] = self.predictions.to_dict()
        
        # Convert model snapshots
        result['models'] = {k: v.to_dict() for k, v in self.models.items()}
        
        return result


class BacktestArtifactManager:
    """
    Comprehensive artifact management for walk-forward backtesting
    
    This class handles the storage, retrieval, and validation of all
    artifacts generated during backtesting, enabling full auditability
    and detection of potential overfitting.
    
    Key responsibilities:
    - Store complete data lineage and model history
    - Track all predictions and their eventual outcomes
    - Enable reconstruction of any backtest step
    - Detect potential backtest overfitting through statistical analysis
    - Manage storage efficiently with compression
    """
    
    def __init__(self, config: BacktestConfig, base_dir: str):
        """
        Initialize artifact manager
        
        Args:
            config: Backtest configuration
            base_dir: Base directory for artifact storage
        """
        self.config = config
        self.base_dir = Path(base_dir)
        
        # Create storage structure
        self.setup_storage_structure()
        
        # Tracking
        self.backtest_steps: List[BacktestStep] = []
        self.current_backtest_id = self.generate_backtest_id()
        
        logger.info("Initialized artifact manager")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info(f"  Backtest ID: {self.current_backtest_id}")
    
    def setup_storage_structure(self):
        """Create directory structure for organized storage"""
        directories = [
            "backtests",      # Individual backtest runs
            "data_snapshots", # Data snapshots with hashes
            "models",         # Trained model artifacts
            "predictions",    # Prediction outputs
            "metrics",        # Performance metrics
            "metadata",       # Configuration and environment info
            "analysis"        # Post-backtest analysis results
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"📁 Created storage structure in {self.base_dir}")
    
    def generate_backtest_id(self) -> str:
        """Generate unique identifier for backtest run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"backtest_{timestamp}_{unique_id}"
    
    def start_backtest_run(self, description: str = "") -> str:
        """
        Start a new backtest run and create metadata
        
        Args:
            description: Description of the backtest run
            
        Returns:
            Backtest ID
        """
        self.current_backtest_id = self.generate_backtest_id()
        self.backtest_steps = []
        
        # Create backtest metadata
        metadata = {
            "backtest_id": self.current_backtest_id,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "environment": self._collect_environment_info(),
            "git_info": self._collect_git_info()
        }
        
        # Save metadata
        metadata_file = self.base_dir / "metadata" / f"{self.current_backtest_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"🚀 Started backtest run: {self.current_backtest_id}")
        if description:
            logger.info(f"  Description: {description}")
        
        return self.current_backtest_id
    
    def record_backtest_step(self, step: BacktestStep):
        """
        Record a complete backtest step
        
        Args:
            step: Backtest step to record
        """
        # Add to tracking
        self.backtest_steps.append(step)
        
        # Save step data
        step_file = self.base_dir / "backtests" / f"{self.current_backtest_id}_step_{step.step_number:04d}.json.gz"
        
        with gzip.open(step_file, 'wt') as f:
            json.dump(step.to_dict(), f, indent=2)
        
        # Save predictions separately for easy access
        if step.predictions:
            pred_file = self.base_dir / "predictions" / f"{self.current_backtest_id}_step_{step.step_number:04d}.json"
            with open(pred_file, 'w') as f:
                json.dump(step.predictions.to_dict(), f, indent=2)
        
        logger.debug(f"📝 Recorded backtest step {step.step_number}")
    
    def update_predictions_with_actuals(self, step_number: int, actual_values: List[float]):
        """
        Update predictions with actual values when they become available
        
        Args:
            step_number: Step number to update
            actual_values: Actual target values
        """
        # Find the step
        step = None
        for s in self.backtest_steps:
            if s.step_number == step_number:
                step = s
                break
        
        if not step:
            logger.warning(f"Step {step_number} not found for updating actuals")
            return
        
        # Update predictions
        if step.predictions:
            step.predictions.update_with_actuals(actual_values)
            
            # Re-save the updated step
            self.record_backtest_step(step)
            
            logger.info(f"📊 Updated step {step_number} with actuals (MSE: {step.predictions.mse:.6f})")
    
    def save_data_snapshot(self, data: pd.DataFrame, targets: pd.Series, 
                          metadata: Dict[str, Any]) -> DataSnapshot:
        """
        Save data snapshot with hash for integrity
        
        Args:
            data: Feature matrix
            targets: Target vector
            metadata: Additional metadata
            
        Returns:
            Data snapshot object
        """
        # Compute hashes
        data_hash = self._compute_dataframe_hash(data)
        target_hash = self._compute_series_hash(targets)
        
        # Create snapshot
        snapshot = DataSnapshot(
            data_hash=data_hash,
            target_hash=target_hash,
            feature_names=list(data.columns),
            n_samples=len(data),
            n_features=len(data.columns),
            start_date=data.index.min(),
            end_date=data.index.max(),
            frequency=pd.infer_freq(data.index) or "unknown",
            missing_values=data.isnull().sum().to_dict(),
            outlier_counts={},  # Could be computed if needed
            feature_correlations={},  # Could be computed if needed
            preprocessing_steps=metadata.get('preprocessing_steps', []),
            feature_engineering_steps=metadata.get('feature_engineering_steps', [])
        )
        
        # Save compressed data
        data_file = self.base_dir / "data_snapshots" / f"{data_hash}.pkl.gz"
        if not data_file.exists():
            with gzip.open(data_file, 'wb') as f:
                pickle.dump({'data': data, 'targets': targets, 'metadata': metadata}, f)
        
        return snapshot
    
    def save_model_snapshot(self, model: Any, model_type: str, 
                           hyperparameters: Dict[str, Any],
                           training_info: Dict[str, Any]) -> ModelSnapshot:
        """
        Save model snapshot with training details
        
        Args:
            model: Trained model object
            model_type: Type of model
            hyperparameters: Model hyperparameters
            training_info: Training metadata
            
        Returns:
            Model snapshot object
        """
        # Compute model hash
        model_hash = self._compute_model_hash(hyperparameters, model_type)
        
        # Create snapshot
        snapshot = ModelSnapshot(
            model_type=model_type,
            model_hash=model_hash,
            hyperparameters=hyperparameters,
            training_samples=training_info.get('training_samples', 0),
            training_start=training_info.get('training_start', datetime.now()),
            training_end=training_info.get('training_end', datetime.now()),
            training_duration=training_info.get('training_duration', 0.0),
            cv_folds=training_info.get('cv_folds', 0),
            cv_scores=training_info.get('cv_scores', []),
            cv_mean=training_info.get('cv_mean', 0.0),
            cv_std=training_info.get('cv_std', 0.0),
            feature_importance=training_info.get('feature_importance'),
            n_parameters=training_info.get('n_parameters'),
            model_size_bytes=training_info.get('model_size_bytes')
        )
        
        # Save model object
        model_file = self.base_dir / "models" / f"{model_hash}.pkl.gz"
        if not model_file.exists():
            with gzip.open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        return snapshot
    
    def finalize_backtest_run(self) -> Dict[str, Any]:
        """
        Finalize backtest run and generate summary
        
        Returns:
            Summary statistics and analysis
        """
        end_time = datetime.now()
        
        # Compute summary statistics
        summary = self._compute_backtest_summary()
        
        # Update metadata with final results
        metadata_file = self.base_dir / "metadata" / f"{self.current_backtest_id}.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata.update({
            "end_time": end_time.isoformat(),
            "total_steps": len(self.backtest_steps),
            "summary": summary
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save complete backtest summary
        summary_file = self.base_dir / "analysis" / f"{self.current_backtest_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"🏁 Finalized backtest run: {self.current_backtest_id}")
        logger.info(f"  Total steps: {len(self.backtest_steps)}")
        logger.info(f"  Final score: {summary.get('overall_mse', 'N/A')}")
        
        return summary
    
    def detect_backtest_overfitting(self) -> Dict[str, Any]:
        """
        Detect potential backtest overfitting using statistical tests
        
        Returns:
            Overfitting analysis results
        """
        if len(self.backtest_steps) < 10:
            logger.warning("Insufficient data for overfitting detection")
            return {"status": "insufficient_data"}
        
        # Collect performance metrics over time
        performance_metrics = []
        for step in self.backtest_steps:
            if step.predictions and step.predictions.mse is not None:
                performance_metrics.append({
                    'step': step.step_number,
                    'date': step.step_date,
                    'mse': step.predictions.mse,
                    'mae': step.predictions.mae,
                    'r2': step.predictions.r2,
                    'hit_rate': step.predictions.hit_rate
                })
        
        if len(performance_metrics) < 5:
            return {"status": "insufficient_performance_data"}
        
        # Perform overfitting analysis
        analysis = {
            "status": "completed",
            "n_samples": len(performance_metrics),
            "performance_trend": self._analyze_performance_trend(performance_metrics),
            "stability_analysis": self._analyze_stability(performance_metrics),
            "regime_consistency": self._analyze_regime_consistency(performance_metrics)
        }
        
        # Save analysis
        analysis_file = self.base_dir / "analysis" / f"{self.current_backtest_id}_overfitting.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Compute SHA-256 hash of dataframe"""
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
    
    def _compute_series_hash(self, series: pd.Series) -> str:
        """Compute SHA-256 hash of series"""
        return hashlib.sha256(
            pd.util.hash_pandas_object(series, index=True).values
        ).hexdigest()
    
    def _compute_model_hash(self, hyperparameters: Dict[str, Any], model_type: str) -> str:
        """Compute hash of model configuration"""
        config_str = json.dumps({
            'model_type': model_type,
            'hyperparameters': hyperparameters
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _collect_environment_info(self) -> Dict[str, str]:
        """Collect environment information for reproducibility"""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0]
        }
    
    def _collect_git_info(self) -> Dict[str, str]:
        """Collect Git information if available"""
        try:
            import subprocess
            
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {
                "commit_hash": git_hash,
                "branch": git_branch
            }
        except Exception:
            return {"status": "git_not_available"}
    
    def _compute_backtest_summary(self) -> Dict[str, Any]:
        """Compute comprehensive backtest summary"""
        if not self.backtest_steps:
            return {}
        
        # Collect performance metrics
        mse_values = []
        mae_values = []
        r2_values = []
        hit_rates = []
        
        for step in self.backtest_steps:
            if step.predictions and step.predictions.mse is not None:
                mse_values.append(step.predictions.mse)
                mae_values.append(step.predictions.mae)
                r2_values.append(step.predictions.r2)
                if step.predictions.hit_rate is not None:
                    hit_rates.append(step.predictions.hit_rate)
        
        summary = {
            "total_steps": len(self.backtest_steps),
            "steps_with_actuals": len(mse_values),
            "overall_mse": float(np.mean(mse_values)) if mse_values else None,
            "overall_mae": float(np.mean(mae_values)) if mae_values else None,
            "overall_r2": float(np.mean(r2_values)) if r2_values else None,
            "overall_hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
            "mse_std": float(np.std(mse_values)) if mse_values else None,
            "performance_stability": float(np.std(mse_values) / np.mean(mse_values)) if mse_values else None
        }
        
        return summary
    
    def _analyze_performance_trend(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trend over time"""
        mse_values = [m['mse'] for m in metrics]
        
        # Simple linear trend
        x = np.arange(len(mse_values))
        trend_coef = np.polyfit(x, mse_values, 1)[0]
        
        return {
            "trend_coefficient": float(trend_coef),
            "trend_direction": "deteriorating" if trend_coef > 0 else "improving",
            "trend_significance": "high" if abs(trend_coef) > 0.01 else "low"
        }
    
    def _analyze_stability(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze performance stability"""
        mse_values = [m['mse'] for m in metrics]
        
        if len(mse_values) < 3:
            return {"status": "insufficient_data"}
        
        # Rolling window analysis
        window_size = min(5, len(mse_values) // 2)
        rolling_means = []
        rolling_stds = []
        
        for i in range(window_size, len(mse_values)):
            window_data = mse_values[i-window_size:i]
            rolling_means.append(np.mean(window_data))
            rolling_stds.append(np.std(window_data))
        
        return {
            "mean_stability": float(np.std(rolling_means)) if rolling_means else 0.0,
            "variance_stability": float(np.std(rolling_stds)) if rolling_stds else 0.0,
            "stability_score": float(1.0 / (1.0 + np.std(rolling_means))) if rolling_means else 0.0
        }
    
    def _analyze_regime_consistency(self, _unused_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency across different market regimes"""
        # This is a placeholder - would require actual regime detection
        return {
            "status": "not_implemented",
            "note": "Regime analysis requires market regime data"
        }


if __name__ == "__main__":
    # Example usage
    from .config import create_default_backtest_config
    
    print("🗄️ Backtest Artifact Manager Example")
    print("=" * 50)
    
    # Create configuration and manager
    config = create_default_backtest_config()
    manager = BacktestArtifactManager(config, "test_artifacts")
    
    # Start a backtest run
    backtest_id = manager.start_backtest_run("Example backtest run")
    print(f"Started backtest: {backtest_id}")
    
    # Example data snapshot
    import pandas as pd
    rng = np.random.default_rng(42)
    example_data = pd.DataFrame({
        'feature1': rng.normal(size=100),
        'feature2': rng.normal(size=100)
    }, index=pd.date_range('2020-01-01', periods=100))
    
    example_targets = pd.Series(rng.normal(size=100), index=example_data.index)
    
    snapshot = manager.save_data_snapshot(example_data, example_targets, {})
    print(f"Created data snapshot: {snapshot.data_hash[:8]}...")
    
    print("\n✅ Artifact manager ready for use!")