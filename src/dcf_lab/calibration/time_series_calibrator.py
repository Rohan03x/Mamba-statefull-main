"""
Time-Series Aware Probability Calibration System

Implements proper calibration for financial forecasts using TimeSeriesSplit to
avoid data leakage while maintaining temporal dependencies. Stores calibration
history and provides weekly recalibration functionality.

Key Features:
- TimeSeriesSplit to respect temporal ordering
- Persistent calibration store (parquet format)
- Weekly recalibration schedule
- Multiple calibration methods (Platt scaling, Isotonic regression)
- Calibration quality metrics and monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple
import joblib

from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


class TimeSeriesCalibrator:
    """
    Time-series aware probability calibration system for financial forecasts.
    
    Maintains calibration without data leakage by using TimeSeriesSplit and
    stores historical calibration performance for monitoring drift.
    """
    
    def __init__(self, 
                 calibration_store_path: str = "data/calibration_store.parquet",
                 method: str = "isotonic",
                 n_splits: int = 5,
                 recalibration_days: int = 7):
        """
        Initialize the time-series calibrator.
        
        Args:
            calibration_store_path: Path to persistent calibration storage
            method: Calibration method ('isotonic' or 'sigmoid')
            n_splits: Number of TimeSeriesSplit folds for calibration
            recalibration_days: How often to recalibrate (days)
        """
        self.calibration_store_path = Path(calibration_store_path)
        self.method = method
        self.n_splits = n_splits
        self.recalibration_days = recalibration_days
        
        # Create directories if needed
        self.calibration_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize calibration store
        self.calibration_df = self._load_calibration_store()
        
        # Current calibrator (fitted model)
        self.calibrator = None
        self.last_calibration_date = None
        self.calibration_metrics = {}
        
        logger.info(f"TimeSeriesCalibrator initialized with {method} method, "
                   f"{n_splits} splits, {recalibration_days}d recalibration cycle")
    
    def _load_calibration_store(self) -> pd.DataFrame:
        """Load existing calibration store or create empty one."""
        if self.calibration_store_path.exists():
            try:
                df = pd.read_parquet(self.calibration_store_path)
                logger.info(f"Loaded calibration store with {len(df)} records")
                return df
            except Exception as e:
                logger.warning(f"Failed to load calibration store: {e}")
                return self._create_empty_store()
        else:
            logger.info("Creating new calibration store")
            return self._create_empty_store()
    
    def _create_empty_store(self) -> pd.DataFrame:
        """Create empty calibration store with proper schema."""
        return pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'predicted_prob': pd.Series(dtype='float64'),
            'actual_outcome': pd.Series(dtype='int64'),
            'model_version': pd.Series(dtype='string'),
            'ticker': pd.Series(dtype='string'),
            'forecast_horizon_days': pd.Series(dtype='int64'),
            'calibrated_prob': pd.Series(dtype='float64'),
            'calibration_method': pd.Series(dtype='string')
        })
    
    def store_prediction(self, 
                        timestamp: datetime,
                        predicted_prob: float,
                        ticker: str,
                        model_version: str = "ensemble_v1",
                        forecast_horizon_days: int = 30) -> None:
        """
        Store a prediction for future calibration assessment.
        
        Args:
            timestamp: When the prediction was made
            predicted_prob: Raw model probability (0-1)
            ticker: Stock ticker
            model_version: Model identifier
            forecast_horizon_days: Forecast time horizon
        """
        new_record = {
            'timestamp': timestamp,
            'predicted_prob': predicted_prob,
            'actual_outcome': None,  # Will be filled when outcome is known
            'model_version': model_version,
            'ticker': ticker,
            'forecast_horizon_days': forecast_horizon_days,
            'calibrated_prob': None,  # Will be filled after calibration
            'calibration_method': None
        }
        
        # Add to store
        new_df = pd.DataFrame([new_record])
        self.calibration_df = pd.concat([self.calibration_df, new_df], 
                                       ignore_index=True)
        
        # Save periodically (every 10 records to avoid too frequent I/O)
        if len(self.calibration_df) % 10 == 0:
            self._save_calibration_store()
    
    def update_outcomes(self, 
                       lookback_days: int = 60) -> int:
        """
        Update actual outcomes for predictions where the forecast horizon has passed.
        
        Args:
            lookback_days: How far back to look for outcome updates
            
        Returns:
            Number of outcomes updated
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=lookback_days)
        
        # Find predictions ready for outcome evaluation
        ready_mask = (
            (self.calibration_df['actual_outcome'].isna()) &
            (self.calibration_df['timestamp'] >= cutoff_time) &
            (self.calibration_df['timestamp'] <= 
             current_time - timedelta(days=self.calibration_df['forecast_horizon_days'].fillna(30)))
        )
        
        ready_predictions = self.calibration_df[ready_mask].copy()
        
        if len(ready_predictions) == 0:
            logger.info("No predictions ready for outcome evaluation")
            return 0
        
        updated_count = 0
        
        # Update outcomes (simplified - would integrate with real price data)
        for idx, row in ready_predictions.iterrows():
            try:
                # In practice, this would fetch real price data and compute outcome
                # For now, simulate outcome based on historical patterns
                outcome = self._simulate_outcome(row['predicted_prob'], row['ticker'])
                
                self.calibration_df.loc[idx, 'actual_outcome'] = outcome
                updated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to update outcome for {row['ticker']}: {e}")
                continue
        
        if updated_count > 0:
            self._save_calibration_store()
            logger.info(f"Updated {updated_count} prediction outcomes")
        
        return updated_count
    
    def _simulate_outcome(self, predicted_prob: float, ticker: str) -> int:
        """
        Simulate outcome for testing (replace with real price data fetch).
        
        Args:
            predicted_prob: Model predicted probability
            ticker: Stock ticker
            
        Returns:
            Binary outcome (0 or 1)
        """
        # Simulate with some correlation to predicted probability
        # Add noise to make calibration realistic
        noise = self.rng.normal(0, 0.1)
        adjusted_prob = np.clip(predicted_prob + noise, 0, 1)
        
        return int(self.rng.random() < adjusted_prob)
    
    def needs_recalibration(self) -> bool:
        """Check if recalibration is needed based on schedule and data availability."""
        if self.last_calibration_date is None:
            return True
        
        days_since_calibration = (datetime.now() - self.last_calibration_date).days
        if days_since_calibration >= self.recalibration_days:
            # Check if we have enough new data
            complete_data = self.calibration_df[
                self.calibration_df['actual_outcome'].notna()
            ]
            
            if len(complete_data) >= 50:  # Minimum data for reliable calibration
                return True
        
        return False
    
    def fit_calibration(self, 
                       min_samples: int = 100,
                       max_samples: int = 5000) -> Dict:
        """
        Fit calibration model using TimeSeriesSplit to avoid data leakage.
        
        Args:
            min_samples: Minimum samples needed for calibration
            max_samples: Maximum samples to use (for performance)
            
        Returns:
            Calibration metrics and diagnostics
        """
        # Get complete data for calibration
        complete_data = self.calibration_df[
            self.calibration_df['actual_outcome'].notna()
        ].copy()
        
        if len(complete_data) < min_samples:
            raise ValueError(f"Insufficient data for calibration: {len(complete_data)} < {min_samples}")
        
        # Sort by timestamp for time-series split
        complete_data = complete_data.sort_values('timestamp')
        
        # Limit data size for performance
        if len(complete_data) > max_samples:
            complete_data = complete_data.tail(max_samples)
        
        # Prepare data
        X = complete_data['predicted_prob'].values.reshape(-1, 1)
        y = complete_data['actual_outcome'].values
        
        # Use TimeSeriesSplit to respect temporal ordering
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Fit calibrator
        if self.method == "isotonic":
            base_calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # sigmoid/platt
            base_calibrator = LogisticRegression()
        
        # Cross-validation calibration
        calibrated_probs = np.zeros_like(y, dtype=float)
        
        for train_idx, val_idx in tscv.split(X):
            x_train, x_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Fit on training fold
            if self.method == "isotonic":
                temp_cal = IsotonicRegression(out_of_bounds='clip')
                temp_cal.fit(x_train.ravel(), y_train)
                calibrated_probs[val_idx] = temp_cal.predict(x_val.ravel())
            else:
                temp_cal = LogisticRegression()
                temp_cal.fit(x_train, y_train)
                calibrated_probs[val_idx] = temp_cal.predict_proba(x_val)[:, 1]
        
        # Fit final calibrator on all data
        self.calibrator = base_calibrator
        if self.method == "isotonic":
            self.calibrator.fit(X.ravel(), y)
        else:
            self.calibrator.fit(X, y)
        
        # Compute calibration metrics
        metrics = self._compute_calibration_metrics(
            y, complete_data['predicted_prob'].values, calibrated_probs
        )
        
        self.calibration_metrics = metrics
        self.last_calibration_date = datetime.now()
        
        # Save calibrator
        self._save_calibrator()
        
        logger.info(f"Calibration completed with {len(complete_data)} samples")
        logger.info(f"Brier score improvement: {metrics['brier_improvement']:.4f}")
        
        return metrics
    
    def _compute_calibration_metrics(self, 
                                   y_true: np.ndarray,
                                   y_prob_raw: np.ndarray,
                                   y_prob_cal: np.ndarray) -> Dict:
        """Compute calibration quality metrics."""
        metrics = {}
        
        # Brier scores
        brier_raw = brier_score_loss(y_true, y_prob_raw)
        brier_cal = brier_score_loss(y_true, y_prob_cal)
        metrics['brier_raw'] = brier_raw
        metrics['brier_calibrated'] = brier_cal
        metrics['brier_improvement'] = brier_raw - brier_cal
        
        # Log losses
        try:
            metrics['log_loss_raw'] = log_loss(y_true, y_prob_raw)
            metrics['log_loss_calibrated'] = log_loss(y_true, y_prob_cal)
        except ValueError:
            metrics['log_loss_raw'] = np.nan
            metrics['log_loss_calibrated'] = np.nan
        
        # Calibration curve analysis
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob_raw, n_bins=10)  # noqa
            metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        except Exception:
            metrics['calibration_error'] = np.nan
        
        # Basic statistics
        metrics['n_samples'] = len(y_true)
        metrics['base_rate'] = np.mean(y_true)
        metrics['pred_mean'] = np.mean(y_prob_raw)
        metrics['calibration_date'] = datetime.now()
        
        return metrics
    
    def calibrate_probability(self, raw_probability: float) -> float:
        """
        Apply calibration to a raw model probability.
        
        Args:
            raw_probability: Raw model output (0-1)
            
        Returns:
            Calibrated probability
        """
        if self.calibrator is None:
            logger.warning("No calibrator fitted, returning raw probability")
            return raw_probability
        
        try:
            prob_array = np.array([[raw_probability]])
            
            if self.method == "isotonic":
                calibrated = self.calibrator.predict(prob_array.ravel())[0]
            else:
                calibrated = self.calibrator.predict_proba(prob_array)[0, 1]
            
            return float(np.clip(calibrated, 0, 1))
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}, returning raw probability")
            return raw_probability
    
    def get_calibration_diagnostics(self) -> Dict:
        """Get current calibration diagnostics and metrics."""
        diagnostics = {
            'calibrator_fitted': self.calibrator is not None,
            'last_calibration': self.last_calibration_date,
            'total_predictions': len(self.calibration_df),
            'predictions_with_outcomes': len(self.calibration_df[
                self.calibration_df['actual_outcome'].notna()
            ]),
            'needs_recalibration': self.needs_recalibration(),
            'calibration_metrics': self.calibration_metrics
        }
        
        return diagnostics
    
    def _save_calibration_store(self) -> None:
        """Save calibration store to parquet."""
        try:
            self.calibration_df.to_parquet(self.calibration_store_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save calibration store: {e}")
    
    def _save_calibrator(self) -> None:
        """Save fitted calibrator model."""
        if self.calibrator is not None:
            calibrator_path = self.calibration_store_path.parent / "calibrator.pkl"
            try:
                joblib.dump({
                    'calibrator': self.calibrator,
                    'method': self.method,
                    'last_calibration_date': self.last_calibration_date,
                    'metrics': self.calibration_metrics
                }, calibrator_path)
            except Exception as e:
                logger.error(f"Failed to save calibrator: {e}")
    
    def load_calibrator(self) -> bool:
        """Load previously saved calibrator."""
        calibrator_path = self.calibration_store_path.parent / "calibrator.pkl"
        
        if not calibrator_path.exists():
            return False
        
        try:
            saved_data = joblib.load(calibrator_path)
            self.calibrator = saved_data['calibrator']
            self.method = saved_data['method']
            self.last_calibration_date = saved_data['last_calibration_date']
            self.calibration_metrics = saved_data.get('metrics', {})
            
            logger.info(f"Loaded calibrator from {self.last_calibration_date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            return False


class CalibrationManager:
    """
    High-level manager for the calibration system.
    
    Handles automatic recalibration scheduling and integration with
    the main forecasting pipeline.
    """
    
    def __init__(self, calibrator: TimeSeriesCalibrator):
        self.calibrator = calibrator
        
        # Load existing calibrator if available
        self.calibrator.load_calibrator()
    
    def process_prediction(self, 
                          raw_probability: float,
                          ticker: str,
                          timestamp: Optional[datetime] = None) -> Tuple[float, Dict]:
        """
        Process a prediction through the calibration system.
        
        Args:
            raw_probability: Raw model probability
            ticker: Stock ticker
            timestamp: Prediction timestamp (defaults to now)
            
        Returns:
            Tuple of (calibrated_probability, diagnostics)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store prediction for future calibration assessment
        self.calibrator.store_prediction(
            timestamp=timestamp,
            predicted_prob=raw_probability,
            ticker=ticker
        )
        
        # Apply current calibration
        calibrated_prob = self.calibrator.calibrate_probability(raw_probability)
        
        # Check if recalibration is needed (async in production)
        needs_recal = self.calibrator.needs_recalibration()
        
        diagnostics = {
            'raw_probability': raw_probability,
            'calibrated_probability': calibrated_prob,
            'calibration_adjustment': calibrated_prob - raw_probability,
            'needs_recalibration': needs_recal,
            'calibrator_age_days': (
                (datetime.now() - self.calibrator.last_calibration_date).days
                if self.calibrator.last_calibration_date else None
            )
        }
        
        return calibrated_prob, diagnostics
    
    def run_maintenance(self) -> Dict:
        """
        Run calibration maintenance tasks.
        
        Returns:
            Maintenance report
        """
        report = {
            'maintenance_date': datetime.now(),
            'outcomes_updated': 0,
            'recalibration_performed': False,
            'calibration_metrics': None
        }
        
        # Update outcomes
        report['outcomes_updated'] = self.calibrator.update_outcomes()
        
        # Recalibrate if needed
        if self.calibrator.needs_recalibration():
            try:
                metrics = self.calibrator.fit_calibration()
                report['recalibration_performed'] = True
                report['calibration_metrics'] = metrics
                logger.info("Calibration maintenance completed successfully")
            except Exception as e:
                logger.error(f"Calibration maintenance failed: {e}")
                report['error'] = str(e)
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Initialize calibration system
    calibrator = TimeSeriesCalibrator()
    manager = CalibrationManager(calibrator)
    
    # Simulate some predictions and outcomes
    print("Testing calibration system...")
    
    # Generate test data
    rng = np.random.default_rng(42)
    for i in range(200):
        timestamp = datetime.now() - timedelta(days=100-i//2)
        raw_prob = rng.beta(2, 5)  # Skewed toward lower probabilities
        
        calibrated_prob, diagnostics = manager.process_prediction(
            raw_probability=raw_prob,
            ticker="AAPL",
            timestamp=timestamp
        )
        
        if i % 50 == 0:
            print(f"Prediction {i}: {raw_prob:.3f} → {calibrated_prob:.3f}")
    
    # Update outcomes and run maintenance
    print("\nRunning maintenance...")
    report = manager.run_maintenance()
    print(f"Maintenance report: {report}")
    
    # Show diagnostics
    diagnostics = calibrator.get_calibration_diagnostics()
    print(f"\nCalibration diagnostics: {diagnostics}")