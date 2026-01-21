"""
Time-Series Cross-Validation with Leakage Protection

Implementation of financial time-series cross-validation including:
- Walk-forward validation (rolling forecasting origin)
- Purged k-fold with embargo periods
- Conformal prediction wrappers
- Leakage-safe splitting for overlapping labels

References:
- Advances in Financial Machine Learning (Lopez de Prado)
- TimeSeriesSplit (scikit-learn)
- Rolling origin evaluation (Bailey et al.)
"""

import logging
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CVConfig:
    """Configuration for time-series cross-validation"""
    # Basic parameters
    n_splits: int = 5
    test_size: Optional[int] = None  # days in test set
    gap: int = 1  # embargo days between train/test
    
    # Advanced parameters
    purge_window: int = 5  # days to purge overlapping labels
    min_train_size: int = 252  # minimum training days (1 year)
    max_train_size: Optional[int] = None  # max days (None = expanding)
    
    # Prediction horizon
    horizon: int = 1  # days ahead to predict
    target_type: str = "log_returns"  # log_returns, abs_returns, direction
    
    # Calibration
    calibration_method: str = "isotonic"  # isotonic, platt, none
    conformal_alpha: float = 0.05  # for 95% prediction intervals


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time Series Cross-Validator with purging and embargo periods.
    
    Implements the purged k-fold cross-validation from Lopez de Prado's
    "Advances in Financial Machine Learning" adapted for financial time series.
    
    Key features:
    - Chronological splits (no shuffle)
    - Purging overlapping samples from training set
    - Embargo period between train/test to prevent leakage
    - Rolling or expanding window options
    """
    
    def __init__(self, config: CVConfig):
        self.config = config
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[
                  Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Features with DatetimeIndex
            y: Target variable (optional)
            groups: Group labels (not used in time series)
            
        Yields:
            train_indices, test_indices: Tuples of train/test indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine test size
        if self.config.test_size is None:
            # ~1 month
            test_size = max(21, n_samples // (self.config.n_splits + 1))
        else:
            test_size = self.config.test_size
            
        # Calculate split points
        test_starts = []
        for i in range(self.config.n_splits):
            # Start from sufficient training data
            min_start = self.config.min_train_size + test_size * i
            test_start = min_start + i * (test_size + self.config.gap)
            if test_start + test_size >= n_samples:
                break
            test_starts.append(test_start)
            
        n_splits = len(test_starts)
        msg = f"Generated {n_splits} CV splits with test_size={test_size}"
        logger.info(msg)
        
        for fold_idx, test_start in enumerate(test_starts):
            test_end = min(test_start + test_size, n_samples)
            test_indices = indices[test_start:test_end]
            
            # Determine training window
            if self.config.max_train_size is None:
                # Expanding window
                train_start = 0
            else:
                # Rolling window
                train_start = max(0, test_start - self.config.max_train_size)
                
            # Apply embargo and purging
            train_end = test_start - self.config.gap
            purge_start = train_end - self.config.purge_window
            
            # Create training indices with purging
            train_indices_raw = indices[train_start:purge_start]
            
            # Filter out any overlapping samples
            train_indices = self._purge_overlapping_samples(
                train_indices_raw, test_indices, X, y
            )
            
            # Validation checks
            if len(train_indices) < self.config.min_train_size:
                train_size = len(train_indices)
                min_size = self.config.min_train_size
                logger.warning(f"Fold {fold_idx}: Training set too small "
                               f"({train_size} < {min_size})")
                continue
                
            if len(test_indices) == 0:
                logger.warning(f"Fold {fold_idx}: Empty test set")
                continue
                
            train_dates = X.index[train_indices]
            test_dates = X.index[test_indices]
            if not len(train_dates) or not len(test_dates):
                logger.warning(f"Fold {fold_idx}: Missing dates for fold leak check")
                continue

            train_last = pd.Timestamp(train_dates[-1])
            test_first = pd.Timestamp(test_dates[0])
            if train_last >= test_first:
                raise AssertionError(
                    "Fold time ordering violated: training window overlaps validation window"
                    f" (train_end={train_last}, valid_start={test_first}, fold={fold_idx})"
                )

            logger.debug(f"Fold {fold_idx}: train={len(train_indices)}, "
                         f"test={len(test_indices)}, "
                         f"gap={self.config.gap} days")
                        
            yield train_indices, test_indices
            
    def _purge_overlapping_samples(self, train_indices: np.ndarray,
                                   test_indices: np.ndarray,
                                   X: pd.DataFrame,
                                   y: Optional[pd.Series]) -> np.ndarray:
        """
        Remove training samples that overlap with test labels.
        
        For multi-period returns (e.g., 5-day forward returns), we need to
        ensure training samples don't use information from the test period.
        """
        if self.config.horizon <= 1:
            return train_indices  # No overlap for 1-day horizon
            
        # Get test period dates
        test_dates = X.index[test_indices]
        test_start = test_dates.min()
        
        # Remove training samples whose labels overlap with test period
        train_dates = X.index[train_indices]
        
        # For H-day forward returns, label at time t uses info from t to t+H
        # So we purge training samples where label_end >= test_start
        purge_cutoff = test_start - timedelta(days=self.config.horizon)
        
        valid_mask = train_dates <= purge_cutoff
        purged_indices = train_indices[valid_mask]
        
        purged_count = len(train_indices) - len(purged_indices)
        if purged_count > 0:
            logger.debug(f"Purged {purged_count} overlapping training samples")
            
        return purged_indices
        
    def get_n_splits(self, X: Optional[pd.DataFrame] = None,
                     y: Optional[pd.Series] = None,
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.config.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Implements rolling forecasting origin evaluation that mimics live trading:
    - Train on historical data
    - Predict next period
    - Move window forward
    - Repeat
    """
    
    def __init__(self, config: CVConfig):
        self.config = config
        self.splitter = PurgedTimeSeriesSplit(config)
        
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      fit_params: Optional[dict] = None) -> dict:
        """
        Perform walk-forward validation on a model.
        
        Args:
            model: Scikit-learn compatible model
            X: Features with DatetimeIndex
            y: Target variable
            fit_params: Additional parameters for model.fit()
            
        Returns:
            Dictionary with validation results
        """
        fit_params = fit_params or {}
        
        fold_results = []
        predictions = []
        actuals = []
        fold_indices = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(X, y)):
            try:
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Clone and fit model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train, **fit_params)
                
                # Make predictions
                if hasattr(model_clone, 'predict_proba'):
                    y_pred_proba = model_clone.predict_proba(X_test)
                    y_pred = model_clone.predict(X_test)
                else:
                    y_pred = model_clone.predict(X_test)
                    y_pred_proba = None
                    
                # Calculate fold metrics
                fold_metrics = self._calculate_metrics(
                    y_test, y_pred, y_pred_proba, fold_idx
                )
                fold_results.append(fold_metrics)
                
                # Store for aggregated metrics
                predictions.extend(y_pred)
                actuals.extend(y_test.values)
                fold_indices.extend([fold_idx] * len(y_test))
                
                logger.debug(f"Fold {fold_idx} complete: "
                           f"RMSE={fold_metrics.get('rmse', 'N/A'):.4f}")
                           
            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {e}")
                continue
                
        # Aggregate results
        results = {
            'fold_results': fold_results,
            'aggregated_metrics': self._aggregate_metrics(fold_results),
            'predictions': predictions,
            'actuals': actuals,
            'fold_indices': fold_indices,
            'config': self.config
        }
        
        return results
        
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray], 
                          fold_idx: int) -> dict:
        """Calculate comprehensive metrics for a single fold."""
        metrics = {'fold': fold_idx}
        
        # Basic regression metrics
        residuals = y_true.values - y_pred
        metrics['rmse'] = np.sqrt(np.mean(residuals**2))
        metrics['mae'] = np.mean(np.abs(residuals))
        metrics['mape'] = np.mean(np.abs(residuals / (y_true.values + 1e-8))) * 100
        
        # Directional accuracy (for returns)
        if self.config.target_type in ['log_returns', 'abs_returns']:
            y_true_sign = np.sign(y_true.values)
            y_pred_sign = np.sign(y_pred)
            metrics['directional_accuracy'] = np.mean(y_true_sign == y_pred_sign)
            
        # Information Coefficient (Spearman correlation)
        from scipy.stats import spearmanr
        ic, ic_pvalue = spearmanr(y_true.values, y_pred)
        metrics['information_coefficient'] = ic
        metrics['ic_pvalue'] = ic_pvalue
        
        # Probabilistic metrics (if available)
        if y_pred_proba is not None and self.config.target_type == 'direction':
            try:
                from sklearn.metrics import brier_score_loss, log_loss
                y_true_binary = (y_true.values > 0).astype(int)
                
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    proba_positive = y_pred_proba[:, 1]
                    metrics['brier_score'] = brier_score_loss(y_true_binary, proba_positive)
                    metrics['log_loss'] = log_loss(y_true_binary, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probabilistic metrics: {e}")
                
        return metrics
        
    def _aggregate_metrics(self, fold_results: List[dict]) -> dict:
        """Aggregate metrics across all folds."""
        if not fold_results:
            return {}
            
        # Calculate means and stds across folds
        metrics_to_aggregate = ['rmse', 'mae', 'mape', 'directional_accuracy', 
                              'information_coefficient', 'brier_score', 'log_loss']
        
        aggregated = {}
        for metric in metrics_to_aggregate:
            values = [fold.get(metric) for fold in fold_results if fold.get(metric) is not None]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_values'] = values
                
        aggregated['n_folds'] = len(fold_results)
        return aggregated


class ConformalPredictor:
    """
    Conformal prediction wrapper for guaranteed interval coverage.
    
    Implements inductive conformal prediction to provide distribution-free
    prediction intervals with finite-sample coverage guarantees.
    """
    
    def __init__(self, model, alpha: float = 0.05):
        """
        Initialize conformal predictor.
        
        Args:
            model: Base prediction model
            alpha: Miscoverage rate (0.05 for 95% intervals)
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Fit the model and calibrate prediction intervals.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Fit base model on training data
        self.model.fit(X_train, y_train)
        
        # Generate predictions on calibration set
        y_cal_pred = self.model.predict(X_cal)
        
        # Calculate nonconformity scores (absolute residuals)
        scores = np.abs(y_cal.values - y_cal_pred)
        
        # Find quantile for desired coverage
        n_cal = len(scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.quantile = np.quantile(scores, q_level)
        
        logger.info(f"Conformal quantile at {1-self.alpha:.1%} coverage: {self.quantile:.4f}")
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point predictions and prediction intervals.
        
        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds)
        """
        if self.quantile is None:
            raise ValueError("Must call fit() before predict()")
            
        point_pred = self.model.predict(X)
        lower_bound = point_pred - self.quantile
        upper_bound = point_pred + self.quantile
        
        return point_pred, lower_bound, upper_bound


class CalibrationWrapper:
    """
    Probability calibration wrapper for financial models.
    
    Implements isotonic regression or Platt scaling to calibrate
    predicted probabilities for better reliability.
    """
    
    def __init__(self, model, method: str = "isotonic"):
        """
        Initialize calibration wrapper.
        
        Args:
            model: Base model that outputs probabilities
            method: Calibration method ('isotonic' or 'platt')
        """
        self.model = model
        self.method = method
        self.calibrator = None
        
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == "platt":
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Fit model and calibrate probabilities.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features  
            y_cal: Calibration targets
        """
        # Fit base model
        self.model.fit(X_train, y_train)
        
        # Get uncalibrated probabilities on calibration set
        if hasattr(self.model, 'predict_proba'):
            uncalibrated_probs = self.model.predict_proba(X_cal)
            if uncalibrated_probs.shape[1] == 2:  # Binary classification
                uncalibrated_probs = uncalibrated_probs[:, 1]
        else:
            # For regression, use predictions as "probabilities"
            uncalibrated_probs = self.model.predict(X_cal)
            
        # Convert targets to binary if needed
        if self.method == "platt":
            y_cal_binary = (y_cal.values > 0).astype(int)
            self.calibrator.fit(uncalibrated_probs.reshape(-1, 1), y_cal_binary)
        else:
            # Isotonic regression works with continuous targets
            self.calibrator.fit(uncalibrated_probs, y_cal.values)
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate calibrated probabilities."""
        if self.calibrator is None:
            raise ValueError("Must call fit() before predict_proba()")
            
        # Get uncalibrated probabilities
        if hasattr(self.model, 'predict_proba'):
            uncalibrated_probs = self.model.predict_proba(X)
            if uncalibrated_probs.shape[1] == 2:
                uncalibrated_probs = uncalibrated_probs[:, 1]
        else:
            uncalibrated_probs = self.model.predict(X)
            
        # Apply calibration
        if self.method == "platt":
            calibrated_probs = self.calibrator.predict_proba(
                uncalibrated_probs.reshape(-1, 1)
            )[:, 1]
        else:
            calibrated_probs = self.calibrator.predict(uncalibrated_probs)
            
        return calibrated_probs


def create_target_variable(df: pd.DataFrame, config: CVConfig) -> pd.Series:
    """
    Create target variable based on configuration.
    
    Args:
        df: DataFrame with price data (must have 'close' column)
        config: CV configuration specifying target type and horizon
        
    Returns:
        Target variable series
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
        
    prices = df['close']
    
    if config.target_type == 'log_returns':
        # H-day log returns
        target = np.log(prices.shift(-config.horizon) / prices)
    elif config.target_type == 'abs_returns':
        # H-day absolute returns
        target = (prices.shift(-config.horizon) / prices) - 1
    elif config.target_type == 'direction':
        # Direction of H-day returns (binary)
        returns = prices.shift(-config.horizon) / prices - 1
        target = (returns > 0).astype(int)
    else:
        raise ValueError(f"Unknown target type: {config.target_type}")
        
    # Remove NaN values from the end (due to forward shift)
    target = target.dropna()
    
    logger.info(f"Created {config.target_type} target with {config.horizon}-day horizon: "
               f"{len(target)} samples")
    
    return target


# Example usage and testing functions
def example_usage():
    """Demonstrate usage of the time-series CV framework."""
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Synthetic price data with trend and noise
    np.random.seed(42)
    trend = np.linspace(100, 200, n_samples)
    noise = np.random.normal(0, 10, n_samples)
    prices = trend + noise
    
    df = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_samples))
    }, index=dates)
    
    # Create features (simple example)
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    df = df.dropna()
    
    # Configuration
    config = CVConfig(
        n_splits=5,
        test_size=21,  # 1 month
        gap=1,
        purge_window=5,
        horizon=1,
        target_type='log_returns'
    )
    
    # Create target
    target = create_target_variable(df, config)
    
    # Align features and target
    common_index = df.index.intersection(target.index)
    X = df.loc[common_index, ['returns', 'sma_20', 'volatility']]
    y = target.loc[common_index]
    
    # Test CV splitter
    splitter = PurgedTimeSeriesSplit(config)
    
    print(f"Testing CV splitter with {len(X)} samples")
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        train_dates = X.index[train_idx]
        test_dates = X.index[test_idx]
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_idx)} samples)")
        print(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_idx)} samples)")
        print(f"  Gap:   {(test_dates[0] - train_dates[-1]).days} days")
        print()
        
    # Test with a simple model
    from sklearn.linear_model import Ridge
    
    model = Ridge(alpha=1.0)
    validator = WalkForwardValidator(config)
    
    print("Running walk-forward validation...")
    results = validator.validate_model(model, X, y)
    
    print("\nValidation Results:")
    print(f"Number of folds: {results['aggregated_metrics']['n_folds']}")
    print(f"Mean RMSE: {results['aggregated_metrics']['rmse_mean']:.4f} "
          f"(±{results['aggregated_metrics']['rmse_std']:.4f})")
    print(f"Mean MAE: {results['aggregated_metrics']['mae_mean']:.4f} "
          f"(±{results['aggregated_metrics']['mae_std']:.4f})")
    print(f"Mean Directional Accuracy: "
          f"{results['aggregated_metrics']['directional_accuracy_mean']:.3f}")


if __name__ == "__main__":
    example_usage()