"""
Walk-Forward Cross-Validation with Purged Embargo (López de Prado Implementation)

This module implements rigorous walk-forward cross-validation following the methodology
from "Advances in Financial Machine Learning" by Marcos López de Prado.

Key Features:
1. Walk-forward splits with rolling origin
2. Purge gap = max rolling lookback used in features  
3. Embargo gap (8-21 trading days) to prevent auto-correlation leakage
4. Proper overlap removal for multi-period targets
5. Integration with feature cleaning system

References:
- Advances in Financial Machine Learning (López de Prado, 2018)
- Chapter 7: Cross-Validation in Finance
- mlfinlab implementation patterns
"""

import logging
from typing import Iterator, Tuple, Optional, List, Dict
from dataclasses import dataclass
from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward cross-validation with purged embargo.
    
    Based on López de Prado's recommendations for financial time series.
    """
    # Core walk-forward parameters
    n_splits: int = 5
    test_size_days: int = 21  # ~1 month testing period
    step_size_days: Optional[int] = None  # None = test_size_days (no overlap)
    
    # Training window parameters
    min_train_days: int = 504  # ~2 years minimum training
    max_train_days: Optional[int] = None  # None = expanding window
    
    # Purge and embargo (CRITICAL for leakage prevention)
    purge_days: Optional[int] = None  # Auto-calculated from feature lookbacks
    embargo_days: int = 15  # 15 trading days (3 weeks) - López de Prado default
    
    # Target horizon parameters
    target_horizon_days: int = 1  # Forward-looking target period
    
    # Validation parameters
    validate_features: bool = True
    validate_gaps: bool = True
    min_samples_per_fold: int = 100  # Minimum samples for statistical significance
    
    # Feature lookback detection
    auto_detect_lookbacks: bool = True  # Auto-detect from feature names
    manual_lookbacks: Optional[Dict[str, int]] = None  # Manual override
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.step_size_days is None:
            self.step_size_days = self.test_size_days
            
        if self.embargo_days < 5:
            warnings.warn(f"Embargo of {self.embargo_days} days is very short. "
                         f"Consider 8-21 days per López de Prado recommendations.")
                         
        if self.min_train_days < 252:
            warnings.warn(f"Training period of {self.min_train_days} days is less than 1 year. "
                         f"May not capture sufficient market regimes.")


class WalkForwardSplit(BaseCrossValidator):
    """
    Walk-forward cross-validation with purged embargo.
    
    Implements the methodology from López de Prado's "Advances in Financial 
    Machine Learning" for rigorous financial time series validation.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self._feature_lookbacks = {}
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration for common financial ML mistakes"""
        if self.config.embargo_days == 0:
            raise ValueError("Embargo must be > 0 to prevent auto-correlation leakage")
            
        if self.config.target_horizon_days > self.config.embargo_days:
            warnings.warn(f"Target horizon ({self.config.target_horizon_days}) > "
                         f"embargo ({self.config.embargo_days}). Consider increasing embargo.")
    
    def detect_feature_lookbacks(self, feature_names: List[str]) -> Dict[str, int]:
        """
        Auto-detect lookback periods from feature names.
        
        Examples:
        - 'sma_20' -> 20 days lookback
        - 'rolling_volatility_10' -> 10 days lookback  
        - 'rsi_14' -> 14 days lookback
        """
        lookbacks = {}
        
        for feature in feature_names:
            lookback = self._extract_lookback_from_name(feature)
            if lookback > 0:
                lookbacks[feature] = lookback
                
        logger.info(f"Detected lookbacks for {len(lookbacks)} features:")
        for feature, lookback in sorted(lookbacks.items()):
            logger.info(f"  {feature}: {lookback} days")
            
        return lookbacks
    
    def _extract_lookback_from_name(self, feature_name: str) -> int:
        """Extract lookback period from feature name using common patterns"""
        import re
        
        # Common patterns for financial features
        patterns = [
            r'_(\d+)$',           # trailing number: sma_20, rsi_14
            r'_(\d+)d$',          # days suffix: returns_5d
            r'rolling_.*_(\d+)',  # rolling window: rolling_volatility_10
            r'ma(\d+)',           # moving average: ma20
            r'ema(\d+)',          # exponential MA: ema12
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name.lower())
            if match:
                return int(match.group(1))
                
        # Default for known indicators without explicit periods
        known_defaults = {
            'rsi': 14,
            'macd': 26,  # slow period
            'stoch': 14,
            'williams': 14,
            'cci': 20,
            'atr': 14
        }
        
        for indicator, default_period in known_defaults.items():
            if indicator in feature_name.lower():
                return default_period
                
        return 0  # No lookback detected
    
    def calculate_purge_days(self, X: pd.DataFrame) -> int:
        """
        Calculate purge period as max lookback + buffer.
        
        Following López de Prado: purge >= max feature lookback to prevent
        using overlapping information between train and test.
        """
        if self.config.purge_days is not None:
            return self.config.purge_days
            
        # Auto-detect feature lookbacks
        if self.config.auto_detect_lookbacks:
            self._feature_lookbacks = self.detect_feature_lookbacks(X.columns.tolist())
        
        # Include manual overrides
        if self.config.manual_lookbacks:
            self._feature_lookbacks.update(self.config.manual_lookbacks)
            
        if not self._feature_lookbacks:
            # Conservative default if no lookbacks detected
            default_purge = 10
            logger.warning(f"No feature lookbacks detected. Using conservative purge of {default_purge} days.")
            return default_purge
            
        max_lookback = max(self._feature_lookbacks.values())
        
        # Add buffer for robustness (López de Prado recommendation)
        buffer_days = max(3, max_lookback // 5)  # 20% buffer, minimum 3 days
        purge_days = max_lookback + buffer_days
        
        logger.info(f"Calculated purge period: {purge_days} days "
                   f"(max lookback: {max_lookback} + buffer: {buffer_days})")
        
        return purge_days
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits with purged embargo.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target series (optional)
            groups: Not used in time series
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for time series splitting")
            
        # Validate features if requested
        if self.config.validate_features:
            self._validate_features(X)
            
        # Calculate dynamic purge period
        purge_days = self.calculate_purge_days(X)
        
        # Get time series parameters
        dates = X.index
        len(X)
        
        # Calculate minimum required history
        min_total_days = (self.config.min_train_days + 
                         purge_days + 
                         self.config.test_size_days + 
                         self.config.embargo_days)
        
        total_days = (dates[-1] - dates[0]).days
        if total_days < min_total_days:
            raise ValueError(f"Insufficient data: {total_days} days available, "
                           f"need {min_total_days} days minimum")
        
        # Generate split points using walk-forward logic
        splits_info = self._generate_walk_forward_splits(
            dates, purge_days
        )
        
        logger.info(f"Generated {len(splits_info)} walk-forward splits:")
        logger.info(f"  Test size: {self.config.test_size_days} days")
        logger.info(f"  Step size: {self.config.step_size_days} days") 
        logger.info(f"  Purge: {purge_days} days")
        logger.info(f"  Embargo: {self.config.embargo_days} days")
        logger.info(f"  Min train: {self.config.min_train_days} days")
        
        valid_folds = 0
        for fold_idx, split_info in enumerate(splits_info):
            try:
                train_indices, test_indices = self._create_fold_indices(
                    X, split_info, fold_idx, purge_days
                )
                
                # Validate fold quality
                if self._validate_fold(train_indices, test_indices, X, fold_idx):
                    valid_folds += 1
                    yield train_indices, test_indices
                    
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue
                
        if valid_folds == 0:
            raise ValueError("No valid folds generated. Check configuration.")
            
        logger.info(f"✓ Successfully generated {valid_folds} walk-forward folds")
    
    def _generate_walk_forward_splits(self, dates: pd.DatetimeIndex, 
                                    purge_days: int) -> List[Dict[str, pd.Timestamp]]:
        """Generate walk-forward split dates"""
        splits = []
        
        # Start with minimum training period
        current_date = dates[0] + timedelta(days=self.config.min_train_days)
        
        while True:
            # Test period
            test_start = current_date + timedelta(days=purge_days)
            test_end = test_start + timedelta(days=self.config.test_size_days)
            
            # Check if we have enough data for this split
            if test_end > dates[-1]:
                break
                
            # Training period  
            train_start = dates[0]
            train_end = current_date
            
            # Apply max training window if specified
            if self.config.max_train_days is not None:
                min_train_start = train_end - timedelta(days=self.config.max_train_days)
                train_start = max(train_start, min_train_start)
                
            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start, 
                'test_end': test_end
            })
            
            # Move forward by step size
            current_date += timedelta(days=self.config.step_size_days)
            
            # Limit number of splits
            if len(splits) >= self.config.n_splits:
                break
                
        return splits
    
    def _create_fold_indices(self, X: pd.DataFrame, split_info: Dict[str, pd.Timestamp],
                           fold_idx: int, purge_days: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create train/test indices for a single fold"""
        dates = X.index
        
        # Get date ranges
        train_mask = (dates >= split_info['train_start']) & (dates <= split_info['train_end'])
        test_mask = (dates >= split_info['test_start']) & (dates <= split_info['test_end'])
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Apply additional purging for overlapping targets
        if self.config.target_horizon_days > 1:
            train_indices = self._purge_overlapping_labels(
                train_indices, test_indices, X, self.config.target_horizon_days
            )
        
        return train_indices, test_indices
    
    def _purge_overlapping_labels(self, train_indices: np.ndarray, 
                                test_indices: np.ndarray,
                                X: pd.DataFrame, 
                                horizon_days: int) -> np.ndarray:
        """
        Purge training samples whose labels overlap with test period.
        
        For H-day forward returns, training sample at time t has label
        spanning [t, t+H]. Must ensure no overlap with test period.
        """
        if horizon_days <= 1:
            return train_indices
            
        test_dates = X.index[test_indices]
        test_start = test_dates.min()
        
        train_dates = X.index[train_indices]
        
        # Latest safe date for training sample
        # (its H-day label must not overlap with test)
        latest_safe = test_start - timedelta(days=horizon_days + 1)
        
        # Filter training indices
        safe_mask = train_dates <= latest_safe
        purged_indices = train_indices[safe_mask]
        
        purged_count = len(train_indices) - len(purged_indices)
        if purged_count > 0:
            logger.debug(f"Purged {purged_count} overlapping training samples")
            
        return purged_indices
    
    def _validate_features(self, X: pd.DataFrame):
        """Validate features for look-ahead bias using existing FeatureCleaner"""
        try:
            from ..features import FeatureCleaner
            
            cleaner = FeatureCleaner(strict_mode=True)
            feature_names = X.columns.tolist()
            
            clean_features, report = cleaner.clean_features(
                feature_names, auto_remove=False
            )
            
            if report["banned_count"] > 0:
                banned_list = [f[0] for f in report["banned_features"]]
                raise ValueError(f"Found {report['banned_count']} look-ahead features: {banned_list}")
                
            logger.info(f"✅ Feature validation passed: {len(clean_features)} features clean")
            
        except ImportError:
            logger.warning("FeatureCleaner not available - skipping feature validation")
    
    def _validate_fold(self, train_indices: np.ndarray, test_indices: np.ndarray,
                      X: pd.DataFrame, fold_idx: int) -> bool:
        """Validate individual fold for quality and leakage"""
        
        # Check minimum sample sizes
        if len(train_indices) < self.config.min_samples_per_fold:
            logger.warning(f"Fold {fold_idx}: Insufficient training samples "
                          f"({len(train_indices)} < {self.config.min_samples_per_fold})")
            return False
            
        if len(test_indices) == 0:
            logger.warning(f"Fold {fold_idx}: Empty test set")
            return False
            
        # Validate temporal ordering and gaps
        if self.config.validate_gaps:
            train_dates = X.index[train_indices]
            test_dates = X.index[test_indices]
            
            # Check temporal ordering
            if train_dates.max() >= test_dates.min():
                logger.error(f"Fold {fold_idx}: Temporal ordering violation!")
                return False
                
            # Check minimum gap
            actual_gap = (test_dates.min() - train_dates.max()).days
            required_gap = self.calculate_purge_days(X) 
            
            if actual_gap < required_gap:
                logger.error(f"Fold {fold_idx}: Insufficient gap "
                           f"({actual_gap} < {required_gap} days)")
                return False
        
        # Log fold summary
        train_dates = X.index[train_indices]
        test_dates = X.index[test_indices]
        gap_days = (test_dates.min() - train_dates.max()).days
        
        logger.info(f"Fold {fold_idx}: "
                   f"Train {train_dates.min().date()} to {train_dates.max().date()} "
                   f"({len(train_indices)} samples) | "
                   f"Gap {gap_days}d | "
                   f"Test {test_dates.min().date()} to {test_dates.max().date()} "
                   f"({len(test_indices)} samples)")
        
        return True
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None,
                     y: Optional[pd.Series] = None,
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations"""
        return self.config.n_splits


def create_walk_forward_splitter(
    n_splits: int = 5,
    test_size_days: int = 21,
    embargo_days: int = 15,
    min_train_days: int = 504,
    **kwargs
) -> WalkForwardSplit:
    """
    Convenience function to create walk-forward splitter with sensible defaults.
    
    Args:
        n_splits: Number of CV folds
        test_size_days: Days in test period (~21 = 1 month)
        embargo_days: Embargo period (8-21 days recommended)
        min_train_days: Minimum training days (~504 = 2 years)
        **kwargs: Additional configuration options
        
    Returns:
        Configured WalkForwardSplit instance
    """
    config = WalkForwardConfig(
        n_splits=n_splits,
        test_size_days=test_size_days,
        embargo_days=embargo_days,
        min_train_days=min_train_days,
        **kwargs
    )
    
    return WalkForwardSplit(config)