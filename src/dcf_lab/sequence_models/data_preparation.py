"""
Data Preparation for Multi-horizon Sequence Models

This module implements comprehensive data preparation following TFT best practices:
- Static features: unchanging characteristics
- Known-future features: calendars, expiries, scheduled events
- Observed-past features: prices, volatility, news sentiment

Key components:
- Feature type classification and validation
- Sliding window generation with multiple horizons
- Proper temporal alignment and lookahead prevention
- Batch creation for efficient training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature types following TFT taxonomy"""
    
    # Static features (unchanging)
    static_features: List[str] = field(default_factory=list)
    
    # Known-future features (calendars, expiries, etc.)
    known_future_features: List[str] = field(default_factory=list)
    
    # Observed-past features (prices, vol, news)
    observed_past_features: List[str] = field(default_factory=list)
    
    # Target variables for different horizons
    target_features: List[str] = field(default_factory=list)
    
    # Feature normalization methods
    normalization_methods: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate feature configuration"""
        all_features = (self.static_features + 
                       self.known_future_features + 
                       self.observed_past_features +
                       self.target_features)
        
        # Check for overlaps
        feature_sets = [
            set(self.static_features),
            set(self.known_future_features), 
            set(self.observed_past_features),
            set(self.target_features)
        ]
        
        for i, set1 in enumerate(feature_sets):
            for j, set2 in enumerate(feature_sets):
                if i != j and set1.intersection(set2):
                    logger.warning(f"Feature overlap detected between sets {i} and {j}")
                    return False
        
        return len(all_features) > 0

@dataclass 
class WindowConfig:
    """Configuration for sliding window generation"""
    
    # Input sequence length
    input_length: int = 60
    
    # Multiple prediction horizons
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 20, 60])
    
    # Step size for sliding windows
    step_size: int = 1
    
    # Minimum samples required
    min_samples: int = 100
    
    # Gap between input and prediction (to prevent lookahead)
    prediction_gap: int = 0
    
    def validate(self) -> bool:
        """Validate window configuration"""
        if self.input_length <= 0:
            return False
        if not self.prediction_horizons or any(h <= 0 for h in self.prediction_horizons):
            return False
        if self.step_size <= 0:
            return False
        if self.min_samples <= 0:
            return False
        return True

class FeatureSplitter:
    """Split features according to TFT taxonomy"""
    
    def __init__(self, feature_config: FeatureConfig):
        self.config = feature_config
        if not self.config.validate():
            raise ValueError("Invalid feature configuration")
    
    def split_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into TFT feature categories
        
        Args:
            data: Input dataframe with all features
            
        Returns:
            Dictionary with split feature DataFrames
        """
        
        result = {}
        
        # Static features (broadcast to all time steps)
        if self.config.static_features:
            available_static = [f for f in self.config.static_features if f in data.columns]
            if available_static:
                result['static'] = data[available_static].copy()
            else:
                result['static'] = pd.DataFrame()
        else:
            result['static'] = pd.DataFrame()
        
        # Known-future features
        if self.config.known_future_features:
            available_known = [f for f in self.config.known_future_features if f in data.columns]
            if available_known:
                result['known_future'] = data[available_known].copy()
            else:
                result['known_future'] = pd.DataFrame()
        else:
            result['known_future'] = pd.DataFrame()
        
        # Observed-past features  
        if self.config.observed_past_features:
            available_observed = [f for f in self.config.observed_past_features if f in data.columns]
            if available_observed:
                result['observed_past'] = data[available_observed].copy()
            else:
                result['observed_past'] = pd.DataFrame()
        else:
            result['observed_past'] = pd.DataFrame()
        
        # Target features
        if self.config.target_features:
            available_targets = [f for f in self.config.target_features if f in data.columns]
            if available_targets:
                result['targets'] = data[available_targets].copy()
            else:
                result['targets'] = pd.DataFrame()
        else:
            result['targets'] = pd.DataFrame()
        
        # Log feature availability
        for category, df in result.items():
            logger.info(f"{category}: {len(df.columns)} features - {list(df.columns)}")
        
        return result
    
    def validate_split(self, split_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that split data is consistent"""
        
        if not split_data:
            return False
        
        # Check that we have some features
        total_features = sum(len(df.columns) for df in split_data.values())
        if total_features == 0:
            logger.error("No features found in split data")
            return False
        
        # Check temporal consistency
        lengths = [len(df) for df in split_data.values() if len(df) > 0]
        if len(set(lengths)) > 1:
            logger.error(f"Inconsistent data lengths: {lengths}")
            return False
        
        # Check for required categories
        if len(split_data.get('observed_past', pd.DataFrame()).columns) == 0:
            logger.warning("No observed-past features found")
        
        if len(split_data.get('targets', pd.DataFrame()).columns) == 0:
            logger.error("No target features found")
            return False
        
        return True

class MultiHorizonTargets:
    """Generate targets for multiple prediction horizons"""
    
    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [1, 5, 20, 60]
    
    def create_targets(self, data: pd.DataFrame, 
                      target_columns: List[str],
                      method: str = 'returns') -> pd.DataFrame:
        """
        Create multi-horizon targets
        
        Args:
            data: Input data
            target_columns: Columns to create targets from
            method: Target creation method ('returns', 'levels', 'changes')
            
        Returns:
            DataFrame with multi-horizon targets
        """
        
        targets = pd.DataFrame(index=data.index)
        
        for col in target_columns:
            if col not in data.columns:
                logger.warning(f"Target column {col} not found in data")
                continue
            
            series = data[col]
            
            for horizon in self.horizons:
                target_name = f"{col}_h{horizon}"
                
                if method == 'returns':
                    # Forward returns
                    targets[target_name] = series.pct_change(horizon).shift(-horizon)
                elif method == 'levels':
                    # Forward levels
                    targets[target_name] = series.shift(-horizon)
                elif method == 'changes':
                    # Forward changes
                    targets[target_name] = series.diff(horizon).shift(-horizon)
                else:
                    raise ValueError(f"Unknown target method: {method}")
        
        # Drop rows with NaN targets (at the end due to forward shift)
        targets = targets.dropna()
        
        logger.info(f"Created {len(targets.columns)} multi-horizon targets for {len(targets)} samples")
        return targets

class WindowGenerator:
    """Generate sliding windows for sequence models"""
    
    def __init__(self, window_config: WindowConfig):
        self.config = window_config
        if not self.config.validate():
            raise ValueError("Invalid window configuration")
    
    def create_windows(self, split_data: Dict[str, pd.DataFrame],
                      targets: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create sliding windows for all feature types
        
        Args:
            split_data: Split feature data
            targets: Multi-horizon targets
            
        Returns:
            Dictionary with windowed arrays
        """
        
        # Align data and targets temporally
        common_index = split_data['observed_past'].index.intersection(targets.index)
        if len(common_index) < self.config.min_samples:
            raise ValueError(f"Insufficient samples: {len(common_index)} < {self.config.min_samples}")
        
        # Sort by time
        common_index = common_index.sort_values()
        
        # Determine valid window positions
        max_horizon = max(self.config.prediction_horizons)
        input_length = self.config.input_length
        
        # Account for input length and maximum prediction horizon
        valid_start = input_length - 1
        valid_end = len(common_index) - max_horizon - self.config.prediction_gap
        
        if valid_end <= valid_start:
            raise ValueError("Not enough data for windows with current configuration")
        
        # Generate window start positions
        start_positions = list(range(valid_start, valid_end, self.config.step_size))
        n_windows = len(start_positions)
        
        logger.info(f"Generating {n_windows} windows with input_length={input_length}")
        
        result = {}
        
        # Process observed-past features (time-varying input)
        if 'observed_past' in split_data and len(split_data['observed_past'].columns) > 0:
            obs_data = split_data['observed_past'].loc[common_index]
            n_obs_features = len(obs_data.columns)
            
            observed_windows = np.zeros((n_windows, input_length, n_obs_features))
            
            for i, start_pos in enumerate(start_positions):
                end_pos = start_pos + 1  # end_pos is exclusive
                window_data = obs_data.iloc[start_pos - input_length + 1:end_pos]
                observed_windows[i] = window_data.values
            
            result['observed_past'] = observed_windows
        
        # Process known-future features (time-varying, known ahead)
        if 'known_future' in split_data and len(split_data['known_future'].columns) > 0:
            known_data = split_data['known_future'].loc[common_index]
            n_known_features = len(known_data.columns)
            
            # Known-future includes input + prediction horizons
            max_total_length = input_length + max_horizon
            known_windows = np.zeros((n_windows, max_total_length, n_known_features))
            
            for i, start_pos in enumerate(start_positions):
                # Include input period + prediction horizons
                end_pos = min(start_pos + max_horizon + 1, len(known_data))
                window_start = start_pos - input_length + 1
                
                if end_pos <= len(known_data):
                    window_data = known_data.iloc[window_start:end_pos]
                    # Pad if necessary
                    if len(window_data) < max_total_length:
                        padding = np.zeros((max_total_length - len(window_data), n_known_features))
                        window_values = np.vstack([window_data.values, padding])
                    else:
                        window_values = window_data.values[:max_total_length]
                    
                    known_windows[i] = window_values
            
            result['known_future'] = known_windows
        
        # Process static features (time-invariant)
        if 'static' in split_data and len(split_data['static'].columns) > 0:
            static_data = split_data['static'].loc[common_index]
            n_static_features = len(static_data.columns)
            
            static_windows = np.zeros((n_windows, n_static_features))
            
            for i, start_pos in enumerate(start_positions):
                # Use static features from the prediction time point
                static_windows[i] = static_data.iloc[start_pos].values
            
            result['static'] = static_windows
        
        # Process targets for all horizons
        targets_aligned = targets.loc[common_index]
        n_target_features = len(targets_aligned.columns)
        
        target_windows = np.zeros((n_windows, len(self.config.prediction_horizons), n_target_features))
        
        for i, start_pos in enumerate(start_positions):
            # Extract targets for each horizon
            for j, horizon in enumerate(self.config.prediction_horizons):
                target_pos = start_pos + self.config.prediction_gap + horizon
                if target_pos < len(targets_aligned):
                    target_windows[i, j] = targets_aligned.iloc[target_pos].values
                else:
                    # Use NaN for out-of-bounds targets
                    target_windows[i, j] = np.nan
        
        result['targets'] = target_windows
        
        # Log window shapes
        for key, array in result.items():
            logger.info(f"{key} windows shape: {array.shape}")
        
        return result

class SequenceDataProcessor:
    """Main data processor for sequence models"""
    
    def __init__(self, feature_config: FeatureConfig, window_config: WindowConfig):
        self.feature_config = feature_config
        self.window_config = window_config
        
        self.feature_splitter = FeatureSplitter(feature_config)
        self.target_generator = MultiHorizonTargets(window_config.prediction_horizons)
        self.window_generator = WindowGenerator(window_config)
    
    def process_data(self, data: pd.DataFrame,
                    target_method: str = 'returns') -> Dict[str, np.ndarray]:
        """
        Complete data processing pipeline
        
        Args:
            data: Input dataframe
            target_method: Target creation method
            
        Returns:
            Dictionary with processed windows
        """
        
        logger.info(f"Processing data with {len(data)} samples and {len(data.columns)} features")
        
        # Step 1: Split features by type
        split_data = self.feature_splitter.split_features(data)
        
        if not self.feature_splitter.validate_split(split_data):
            raise ValueError("Feature split validation failed")
        
        # Step 2: Generate multi-horizon targets
        targets = self.target_generator.create_targets(
            data, self.feature_config.target_features, target_method
        )
        
        if targets.empty:
            raise ValueError("No targets generated")
        
        # Step 3: Create sliding windows
        windows = self.window_generator.create_windows(split_data, targets)
        
        # Step 4: Validate windows
        self._validate_windows(windows)
        
        logger.info("Data processing completed successfully")
        return windows
    
    def _validate_windows(self, windows: Dict[str, np.ndarray]) -> bool:
        """Validate generated windows"""
        
        if not windows:
            raise ValueError("No windows generated")
        
        # Check that targets exist
        if 'targets' not in windows:
            raise ValueError("No target windows found")
        
        # Check shape consistency
        n_windows = windows['targets'].shape[0]
        
        for key, array in windows.items():
            if array.shape[0] != n_windows:
                raise ValueError(f"Inconsistent window count for {key}: {array.shape[0]} vs {n_windows}")
        
        # Check for excessive NaN values
        target_nan_ratio = np.isnan(windows['targets']).mean()
        if target_nan_ratio > 0.1:
            logger.warning(f"High NaN ratio in targets: {target_nan_ratio:.1%}")
        
        logger.info(f"Window validation passed for {n_windows} windows")
        return True
    
    def create_train_val_split(self, windows: Dict[str, np.ndarray],
                              val_ratio: float = 0.2,
                              time_based: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Create train/validation split
        
        Args:
            windows: Processed windows
            val_ratio: Validation set ratio
            time_based: Whether to use time-based split (recommended)
            
        Returns:
            Tuple of (train_windows, val_windows)
        """
        
        n_windows = windows['targets'].shape[0]
        
        if time_based:
            # Time-based split (later data for validation)
            split_idx = int(n_windows * (1 - val_ratio))
        else:
            # Random split (not recommended for time series)
            np.random.seed(42)
            indices = np.random.permutation(n_windows)
            split_idx = int(n_windows * (1 - val_ratio))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
        
        train_windows = {}
        val_windows = {}
        
        for key, array in windows.items():
            if time_based:
                train_windows[key] = array[:split_idx]
                val_windows[key] = array[split_idx:]
            else:
                train_windows[key] = array[train_indices]
                val_windows[key] = array[val_indices]
        
        logger.info(f"Split: {len(train_windows['targets'])} train, {len(val_windows['targets'])} validation")
        return train_windows, val_windows