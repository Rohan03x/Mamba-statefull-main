"""
Enhanced Time-Series Cross-Validation with Rigorous Leakage Protection

This module implements state-of-the-art financial time-series cross-validation
with comprehensive leakage prevention measures based on industry best practices.

Key Enhancements:
1. Proper gap/embargo enforcement
2. Rigorous overlapping label purging  
3. Minimum training size enforcement
4. Feature lag alignment validation
5. Missing data handling per fold
6. Label alignment verification

References:
- Advances in Financial Machine Learning (Lopez de Prado)
- QuantInsti Blog on Financial ML
- Bailey et al. on Rolling Origin Evaluation
"""

import logging
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCVConfig:
    """Enhanced configuration for rigorous time-series cross-validation"""
    # Basic parameters
    n_splits: int = 5
    test_size: Optional[int] = None  # days in test set
    
    # Gap/Embargo parameters (CRITICAL for leakage prevention)
    gap: int = 2  # MANDATORY gap between train/test (minimum 1)
    embargo_periods: int = 1  # Additional embargo after test period
    
    # Purging parameters for overlapping labels
    purge_window: int = 5  # days to purge overlapping labels
    strict_purging: bool = True  # Enable strict purging for multi-day horizons
    
    # Training window constraints
    min_train_size: int = 504  # 2 years minimum (increased from 252)
    max_train_size: Optional[int] = None  # max days (None = expanding)
    skip_insufficient_folds: bool = True  # Skip folds with insufficient data
    
    # Prediction horizon and target
    horizon: int = 1  # days ahead to predict
    target_type: str = "log_returns"  # log_returns, abs_returns, direction
    
    # Feature validation
    validate_feature_lags: bool = True  # Check for future leakage in features
    standardize_per_fold: bool = True  # Fit scaler only on training data
    
    # Missing data handling
    handle_missing_per_fold: bool = True  # Handle missing data separately per fold
    missing_strategy: str = "drop"  # drop, forward_fill, train_median
    
    # Label alignment
    verify_label_alignment: bool = True  # Check for incomplete/misaligned labels
    drop_incomplete_targets: bool = True  # Drop samples with incomplete targets


class RigorousTimeSeriesSplit(BaseCrossValidator):
    """
    Rigorous Time Series Cross-Validator with comprehensive leakage protection.
    
    This implementation addresses all common sources of leakage in financial ML:
    - Mandatory gaps/embargos between train/test
    - Proper purging of overlapping labels
    - Per-fold standardization
    - Missing data handling
    - Feature lag validation
    """
    
    def __init__(self, config: EnhancedCVConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration for common mistakes"""
        if self.config.gap < 1:
            raise ValueError("Gap must be at least 1 day to prevent leakage")
            
        if self.config.horizon > 1 and self.config.purge_window < self.config.horizon:
            logger.warning(f"Purge window ({self.config.purge_window}) should be "
                          f"at least as large as horizon ({self.config.horizon})")
                          
        if self.config.min_train_size < 252:
            logger.warning(f"Minimum training size ({self.config.min_train_size}) "
                          f"is less than 1 year. Consider increasing.")
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[
                  Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices with rigorous leakage protection.
        
        Enhanced splitting logic that ensures:
        1. Mandatory gaps between train/test
        2. Proper purging for multi-horizon targets
        3. Minimum training size enforcement
        4. Feature lag validation
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for time-series splitting")
            
        # Validate features for potential leakage
        if self.config.validate_feature_lags:
            self._validate_feature_lags(X)
            
        # Validate label alignment
        if self.config.verify_label_alignment and y is not None:
            self._validate_label_alignment(X, y)
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine test size
        if self.config.test_size is None:
            # Default: ~1 month, but ensure sufficient splits
            test_size = max(21, n_samples // (self.config.n_splits + 2))
        else:
            test_size = self.config.test_size
            
        # Calculate total required space per fold
        min_total_space = (self.config.min_train_size + 
                          self.config.gap + 
                          self.config.purge_window + 
                          test_size + 
                          self.config.embargo_periods)
        
        if min_total_space > n_samples:
            raise ValueError(f"Dataset too small ({n_samples}) for configuration. "
                           f"Need at least {min_total_space} samples.")
        
        # Generate split points with proper spacing
        test_starts = []
        current_start = self.config.min_train_size + self.config.gap + self.config.purge_window
        
        while current_start + test_size + self.config.embargo_periods <= n_samples:
            test_starts.append(current_start)
            # Move forward by test_size + embargo for next fold
            current_start += test_size + self.config.embargo_periods
            
            if len(test_starts) >= self.config.n_splits:
                break
                
        logger.info(f"Generated {len(test_starts)} rigorous CV splits:")
        logger.info(f"  Test size: {test_size} days")
        logger.info(f"  Gap: {self.config.gap} days")
        logger.info(f"  Purge window: {self.config.purge_window} days")
        logger.info(f"  Embargo: {self.config.embargo_periods} days")
        logger.info(f"  Min train size: {self.config.min_train_size} days")
        
        valid_folds = 0
        for fold_idx, test_start in enumerate(test_starts):
            test_end = test_start + test_size
            test_indices = indices[test_start:test_end]
            
            # Determine training window end (with gap)
            train_end = test_start - self.config.gap
            
            # Apply purging for overlapping labels
            if self.config.strict_purging and self.config.horizon > 1:
                # For multi-day horizons, purge more aggressively
                purge_end = train_end - self.config.purge_window
            else:
                purge_end = train_end
                
            # Determine training window start
            if self.config.max_train_size is None:
                # Expanding window
                train_start = 0
            else:
                # Rolling window
                train_start = max(0, purge_end - self.config.max_train_size)
                
            # Create training indices
            train_indices_raw = indices[train_start:purge_end]
            
            # Advanced purging for overlapping samples
            train_indices = self._advanced_purge_overlapping_samples(
                train_indices_raw, test_indices, X, y
            )
            
            # Validation checks
            if len(train_indices) < self.config.min_train_size:
                if self.config.skip_insufficient_folds:
                    logger.warning(f"Fold {fold_idx}: Skipping due to insufficient "
                                 f"training data ({len(train_indices)} < "
                                 f"{self.config.min_train_size})")
                    continue
                else:
                    logger.error(f"Fold {fold_idx}: Training set too small!")
                    
            if len(test_indices) == 0:
                logger.warning(f"Fold {fold_idx}: Empty test set")
                continue
                
            # Verify no leakage
            train_dates = X.index[train_indices]
            test_dates = X.index[test_indices]
            actual_gap = (test_dates.min() - train_dates.max()).days
            
            if actual_gap < self.config.gap:
                logger.error(f"Fold {fold_idx}: Gap violation! "
                           f"Actual: {actual_gap}, Required: {self.config.gap}")
                continue
                
            logger.info(f"Fold {fold_idx}: "
                       f"Train: {train_dates.min().date()} to {train_dates.max().date()} "
                       f"({len(train_indices)} samples), "
                       f"Gap: {actual_gap} days, "
                       f"Test: {test_dates.min().date()} to {test_dates.max().date()} "
                       f"({len(test_indices)} samples)")
                       
            valid_folds += 1
            yield train_indices, test_indices
            
        if valid_folds == 0:
            raise ValueError("No valid folds generated. Check configuration.")
            
        logger.info(f"✓ Generated {valid_folds} valid folds with rigorous leakage protection")
    
    def _validate_feature_lags(self, X: pd.DataFrame):
        """Validate that features don't contain future information"""
        from ..features import FeatureCleaner
        
        # Use the comprehensive feature cleaner
        cleaner = FeatureCleaner(strict_mode=True)
        feature_names = X.columns.tolist()
        
        # Run automated feature cleaning
        clean_features, cleaning_report = cleaner.clean_features(
            feature_names, auto_remove=False
        )
        
        # Check if any banned features were found
        if cleaning_report["banned_count"] > 0:
            banned_list = [f[0] for f in cleaning_report["banned_features"]]
            raise ValueError(
                f"Detected {cleaning_report['banned_count']} look-ahead features: {banned_list}. "
                f"These features contain future information and must be removed!"
            )
        
        # Log warnings for suspicious features
        if cleaning_report["suspicious_count"] > 0:
            suspicious_list = [f[0] for f in cleaning_report["suspicious_features"]]
            logger.warning(f"Found {cleaning_report['suspicious_count']} suspicious features: {suspicious_list}")
        
        logger.info(f"✅ Feature validation passed: {len(clean_features)} clean features validated")
    
    def _validate_label_alignment(self, X: pd.DataFrame, y: pd.Series):
        """Validate proper label alignment and completeness"""
        # Check for misaligned indices
        if not X.index.equals(y.index):
            logger.warning("X and y indices are not perfectly aligned")
            
        # Check for NaN targets (especially at the end for forward-looking targets)
        if y.isna().sum() > 0:
            nan_positions = y.isna()
            if nan_positions.iloc[-self.config.horizon:].any():
                logger.info(f"Found {nan_positions.sum()} NaN targets, "
                           f"including {nan_positions.iloc[-self.config.horizon:].sum()} "
                           f"at the end (expected for {self.config.horizon}-day horizon)")
            else:
                logger.warning(f"Found {nan_positions.sum()} scattered NaN targets - "
                              f"check label creation logic")
    
    def _advanced_purge_overlapping_samples(self, train_indices: np.ndarray,
                                          test_indices: np.ndarray,
                                          X: pd.DataFrame,
                                          y: Optional[pd.Series]) -> np.ndarray:
        """
        Advanced purging for overlapping labels with detailed validation.
        
        For H-day forward returns, ensures training samples don't use
        information that overlaps with test period.
        """
        if self.config.horizon <= 1:
            return train_indices  # No overlap for 1-day horizon
            
        # Get date ranges
        test_dates = X.index[test_indices]
        test_start = test_dates.min()
        test_dates.max()
        
        train_dates = X.index[train_indices]
        
        # For H-day forward returns, a training sample at time t
        # has a label that spans from t to t+H
        # We must ensure no training label overlaps with test period
        
        # Calculate the latest date a training sample can have
        # such that its H-day label doesn't overlap with test
        latest_safe_date = test_start - timedelta(days=self.config.horizon)
        
        # Additional safety margin for weekends/holidays
        safety_margin = timedelta(days=1)
        latest_safe_date -= safety_margin
        
        # Filter training samples
        safe_mask = train_dates <= latest_safe_date
        purged_indices = train_indices[safe_mask]
        
        purged_count = len(train_indices) - len(purged_indices)
        if purged_count > 0:
            logger.debug(f"Advanced purging removed {purged_count} overlapping samples")
            logger.debug(f"Latest safe training date: {latest_safe_date.date()}")
            logger.debug(f"Test period starts: {test_start.date()}")
            
        return purged_indices
        
    def get_n_splits(self, X: Optional[pd.DataFrame] = None,
                     y: Optional[pd.Series] = None,
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.config.n_splits


class PerFoldStandardizer:
    """
    Per-fold standardization to prevent data leakage.
    
    Fits scaler only on training data within each fold,
    then applies to both train and test sets.
    """
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.scalers = {}  # Store scalers per fold
        
    def fit_transform_fold(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit scaler on training data and transform both train/test.
        
        Args:
            X_train: Training features
            X_test: Test features  
            fold_idx: Fold identifier
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        # Create scaler for this fold
        scaler = StandardScaler()
        
        # Fit only on training data
        scaler.fit(X_train[self.feature_columns])
        
        # Transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[self.feature_columns] = scaler.transform(X_train[self.feature_columns])
        X_test_scaled[self.feature_columns] = scaler.transform(X_test[self.feature_columns])
        
        # Store scaler for this fold
        self.scalers[fold_idx] = scaler
        
        logger.debug(f"Fold {fold_idx}: Fitted scaler on {len(X_train)} training samples")
        
        return X_train_scaled, X_test_scaled


class MissingDataHandler:
    """
    Per-fold missing data handling to prevent leakage.
    """
    
    def __init__(self, strategy: str = "drop"):
        """
        Initialize missing data handler.
        
        Args:
            strategy: 'drop', 'forward_fill', 'train_median'
        """
        self.strategy = strategy
        self.fill_values = {}  # Store fill values per fold
        
    def handle_fold(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                         pd.Series, pd.Series]:
        """
        Handle missing data for a single fold.
        
        Args:
            X_train, X_test: Feature DataFrames
            y_train, y_test: Target Series
            fold_idx: Fold identifier
            
        Returns:
            Tuple of (X_train_clean, X_test_clean, y_train_clean, y_test_clean)
        """
        if self.strategy == "drop":
            # Drop any rows with missing values
            train_complete = X_train.dropna().index
            test_complete = X_test.dropna().index
            
            # Also drop corresponding target values
            y_train_complete = y_train.dropna().index
            y_test_complete = y_test.dropna().index
            
            # Find intersection of complete samples
            train_valid = train_complete.intersection(y_train_complete)
            test_valid = test_complete.intersection(y_test_complete)
            
            X_train_clean = X_train.loc[train_valid]
            X_test_clean = X_test.loc[test_valid]
            y_train_clean = y_train.loc[train_valid]
            y_test_clean = y_test.loc[test_valid]
            
            dropped_train = len(X_train) - len(X_train_clean)
            dropped_test = len(X_test) - len(X_test_clean)
            
            if dropped_train > 0 or dropped_test > 0:
                logger.debug(f"Fold {fold_idx}: Dropped {dropped_train} train, "
                           f"{dropped_test} test samples due to missing data")
                           
        elif self.strategy == "train_median":
            # Fill missing values with training data medians
            fill_values = X_train.median()
            self.fill_values[fold_idx] = fill_values
            
            X_train_clean = X_train.fillna(fill_values)
            X_test_clean = X_test.fillna(fill_values)
            y_train_clean = y_train.copy()
            y_test_clean = y_test.copy()
            
        elif self.strategy == "forward_fill":
            # Forward fill within each dataset separately
            X_train_clean = X_train.fillna(method='ffill')
            X_test_clean = X_test.fillna(method='ffill')
            y_train_clean = y_train.copy()
            y_test_clean = y_test.copy()
            
        else:
            raise ValueError(f"Unknown missing data strategy: {self.strategy}")
            
        return X_train_clean, X_test_clean, y_train_clean, y_test_clean


def create_enhanced_target_variable(df: pd.DataFrame, 
                                  config: EnhancedCVConfig) -> pd.Series:
    """
    Create target variable with enhanced validation and alignment checks.
    
    Args:
        df: DataFrame with price data (must have 'close' column)
        config: Enhanced CV configuration
        
    Returns:
        Target variable series with proper validation
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
        
    prices = df['close']
    
    # Validate price data
    if prices.isna().sum() > 0:
        logger.warning(f"Found {prices.isna().sum()} missing prices - "
                      f"this will affect target creation")
        
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
        
    # Validate target creation
    expected_nans = config.horizon
    actual_nans = target.isna().sum()
    
    if config.drop_incomplete_targets:
        target = target.dropna()
        logger.info(f"Dropped {actual_nans} incomplete targets at end of series")
    else:
        if actual_nans != expected_nans:
            logger.warning(f"Unexpected number of NaN targets: "
                          f"{actual_nans} (expected ~{expected_nans})")
    
    # Additional validation
    if len(target) == 0:
        raise ValueError("No valid targets created - check input data")
        
    if config.target_type in ['log_returns', 'abs_returns']:
        # Check for extreme values that might indicate data issues
        extreme_threshold = 0.5  # 50% single-day move
        extreme_mask = np.abs(target) > extreme_threshold
        if extreme_mask.sum() > 0:
            logger.warning(f"Found {extreme_mask.sum()} extreme target values "
                          f"(>{extreme_threshold:.1%}) - check data quality")
    
    logger.info(f"Created {config.target_type} target with {config.horizon}-day horizon:")
    logger.info(f"  Samples: {len(target)}")
    logger.info(f"  Mean: {target.mean():.6f}")
    logger.info(f"  Std: {target.std():.6f}")
    logger.info(f"  Min: {target.min():.6f}")
    logger.info(f"  Max: {target.max():.6f}")
    
    return target


# Example usage demonstrating rigorous leakage protection
def demonstrate_rigorous_cv():
    """Demonstrate the enhanced CV with rigorous leakage protection"""
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_samples)))
    
    df = pd.DataFrame({
        'close': prices,
        'volume': rng.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Add proper lagged features (no look-ahead)
    df['returns_1d'] = df['close'].pct_change()
    df['returns_5d'] = df['close'].pct_change(5)  # 5-day lagged returns
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['returns_1d'].rolling(20).std()
    
    # Remove initial NaN values
    df = df.dropna()
    
    # Enhanced configuration
    config = EnhancedCVConfig(
        n_splits=3,
        test_size=30,
        gap=3,  # 3-day gap
        embargo_periods=2,  # 2-day embargo
        purge_window=5,
        horizon=5,  # 5-day prediction horizon
        target_type='log_returns',
        min_train_size=504,  # 2 years
        strict_purging=True,
        validate_feature_lags=True,
        verify_label_alignment=True
    )
    
    # Create target
    target = create_enhanced_target_variable(df, config)
    
    # Align data
    common_index = df.index.intersection(target.index)
    X = df.loc[common_index, ['returns_1d', 'returns_5d', 'sma_20', 'volatility']]
    y = target.loc[common_index]
    
    # Test rigorous CV
    splitter = RigorousTimeSeriesSplit(config)
    
    print("Testing Rigorous Time-Series Cross-Validation:")
    print("=" * 60)
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        train_dates = X.index[train_idx]
        test_dates = X.index[test_idx]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} "
              f"({len(train_idx)} samples)")
        print(f"  Test:  {test_dates[0].date()} to {test_dates[-1].date()} "
              f"({len(test_idx)} samples)")
        
        gap = (test_dates[0] - train_dates[-1]).days
        print(f"  Gap: {gap} days")
        
        # Verify no leakage
        assert gap >= config.gap, f"Gap violation in fold {fold_idx}!"
        
    print("\n✓ All leakage protection tests passed!")


if __name__ == "__main__":
    demonstrate_rigorous_cv()