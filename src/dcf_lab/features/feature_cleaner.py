"""
Feature Cleaning and Validation System for Financial ML

This module implements comprehensive feature cleaning to prevent data leakage:
1. Blacklists future-leaking feature names
2. Validates feature computation logic
3. Ensures only lagged features are used
4. Automates rejection of bad features

Key functions:
- clean_features(): Remove banned features automatically
- validate_features(): Comprehensive feature validation
- audit_feature_pipeline(): Full audit before training
"""

import logging
import re
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureCleaner:
    """
    Comprehensive feature cleaning and validation system.
    
    Automatically removes features with future-leaking characteristics
    and validates that all remaining features use only historical data.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize feature cleaner.
        
        Args:
            strict_mode: If True, raises errors for violations. If False, only warns.
        """
        self.strict_mode = strict_mode
        
        # Banned keywords that indicate future leakage
        self.banned_keywords = [
            "forward", "future", "next", "ahead", "leak", "BAD", 
            "lookahead", "future_", "forward_", "next_", "coming",
            "upcoming", "later", "afterwards", "subsequent"
        ]
        
        # Patterns that indicate future information
        self.banned_patterns = [
            r".*_\d+d_future.*",     # e.g., returns_5d_future
            r".*forward_\d+.*",       # e.g., forward_5_returns
            r".*next_\d+.*",         # e.g., next_5_volatility
            r".*shift_-\d+.*",       # e.g., price_shift_-5 (negative shift = future)
            r".*lead_\d+.*",         # e.g., returns_lead_3
        ]
        
        # Allowed feature prefixes (historical computations)
        self.allowed_prefixes = [
            "returns_", "sma_", "ema_", "rsi_", "volatility_", "volume_",
            "macd_", "bb_", "atr_", "adx_", "cci_", "stoch_", "williams_",
            "momentum_", "roc_", "tsi_", "mfi_", "obv_", "vwap_",
            "lag_", "rolling_", "ewm_", "expanding_", "shift_" # positive shifts only
        ]
        
        # Required look-back indicators (must have numeric suffix)
        self.lookback_required = [
            "sma", "ema", "rsi", "volatility", "rolling", "ewm"
        ]
        
        self.cleaning_stats = {
            "total_features": 0,
            "banned_removed": [],
            "suspicious_patterns": [],
            "validation_warnings": [],
            "clean_features": []
        }
    
    def clean_features(self, feature_names: List[str], 
                      auto_remove: bool = True) -> Tuple[List[str], Dict]:
        """
        Clean feature list by removing future-leaking features.
        
        Args:
            feature_names: List of feature names to clean
            auto_remove: If True, automatically remove bad features
            
        Returns:
            Tuple of (clean_features, cleaning_report)
        """
        logger.info(f"🧹 Starting feature cleaning on {len(feature_names)} features")
        
        self.cleaning_stats["total_features"] = len(feature_names)
        banned_features = []
        suspicious_features = []
        clean_features = []
        
        for feature in feature_names:
            is_banned, reason = self._is_feature_banned(feature)
            
            if is_banned:
                banned_features.append((feature, reason))
                if not auto_remove:
                    clean_features.append(feature)  # Keep if not auto-removing
            else:
                is_suspicious, warning = self._is_feature_suspicious(feature)
                if is_suspicious:
                    suspicious_features.append((feature, warning))
                    if self.strict_mode and auto_remove:
                        banned_features.append((feature, f"SUSPICIOUS: {warning}"))
                    else:
                        clean_features.append(feature)
                        if warning:
                            self.cleaning_stats["validation_warnings"].append(warning)
                else:
                    clean_features.append(feature)
        
        # Update stats
        self.cleaning_stats["banned_removed"] = banned_features
        self.cleaning_stats["suspicious_patterns"] = suspicious_features
        self.cleaning_stats["clean_features"] = clean_features
        
        # Log results
        if banned_features:
            logger.warning(f"🚫 Removed {len(banned_features)} banned features:")
            for feature, reason in banned_features:
                logger.warning(f"  - {feature}: {reason}")
        
        if suspicious_features and not auto_remove:
            logger.warning(f"⚠️ Found {len(suspicious_features)} suspicious features:")
            for feature, warning in suspicious_features:
                logger.warning(f"  - {feature}: {warning}")
        
        logger.info(f"✅ Feature cleaning complete: {len(clean_features)}/{len(feature_names)} features retained")
        
        # Generate detailed report
        cleaning_report = {
            "original_count": len(feature_names),
            "clean_count": len(clean_features),
            "banned_count": len(banned_features),
            "suspicious_count": len(suspicious_features),
            "banned_features": banned_features,
            "suspicious_features": suspicious_features,
            "clean_features": clean_features,
            "removal_rate": len(banned_features) / len(feature_names) if feature_names else 0
        }
        
        return clean_features, cleaning_report
    
    def _is_feature_banned(self, feature_name: str) -> Tuple[bool, str]:
        """Check if feature name contains banned keywords or patterns."""
        feature_lower = feature_name.lower()
        
        # Check banned keywords
        for keyword in self.banned_keywords:
            if keyword.lower() in feature_lower:
                return True, f"Contains banned keyword: '{keyword}'"
        
        # Check banned patterns
        for pattern in self.banned_patterns:
            if re.match(pattern, feature_lower):
                return True, f"Matches banned pattern: {pattern}"
        
        # Check for negative shifts (future data)
        if "shift" in feature_lower and re.search(r"shift.*-\d+", feature_lower):
            return True, "Contains negative shift (future data)"
        
        return False, ""
    
    def _is_feature_suspicious(self, feature_name: str) -> Tuple[bool, str]:
        """Check if feature might be suspicious but not definitively banned."""
        feature_lower = feature_name.lower()
        
        # Check for missing look-back windows only for certain prefixes
        missing_lookback_prefixes = ["rolling", "ewm", "expanding"] 
        for prefix in missing_lookback_prefixes:
            if feature_lower.startswith(prefix) and not re.search(r"\d+", feature_name):
                return True, f"Missing look-back window specification for {prefix}"
        
        # Allow standard financial feature naming patterns
        # Standard patterns: sma_20, rsi_14, ema_12, volatility_10d, returns_5d, etc.
        standard_patterns = [
            r"^(sma|ema|rsi)_\d+$",                    # sma_20, ema_12, rsi_14
            r"^(returns|log_returns)_\d+d?$",          # returns_1d, log_returns_5d
            r"^volatility_\d+d?$",                     # volatility_10d
            r"^momentum_\d+d?$",                       # momentum_20d
            r"^volume_(sma|ratio)_\d+d?$",             # volume_sma_20, volume_ratio_20d
            r"^(macd|macd_signal|macd_histogram)$",    # MACD components
        ]
        
        # Check if feature matches standard patterns
        for pattern in standard_patterns:
            if re.match(pattern, feature_lower):
                return False, ""  # Standard pattern, not suspicious
        
        # Check for unusual naming patterns only if not standard
        if re.search(r"_\d+$", feature_name) and "lag" not in feature_lower:
            # Only flag if doesn't match standard patterns
            if not any(re.match(pattern, feature_lower) for pattern in standard_patterns):
                return True, "Unusual numeric suffix pattern"
        
        return False, ""
    
    def validate_features(self, df: pd.DataFrame, feature_cols: List[str],
                         check_data_quality: bool = True) -> Dict:
        """
        Comprehensive feature validation including data quality checks.
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names
            check_data_quality: Whether to perform data quality validation
            
        Returns:
            Validation report dictionary
        """
        logger.info(f"🔍 Validating {len(feature_cols)} features")
        
        validation_report = {
            "feature_count": len(feature_cols),
            "validation_passed": True,
            "errors": [],
            "warnings": [],
            "data_quality": {},
            "leakage_checks": {}
        }
        
        # 1. Name-based validation
        clean_features, cleaning_report = self.clean_features(feature_cols, auto_remove=False)
        
        if cleaning_report["banned_count"] > 0:
            validation_report["validation_passed"] = False
            validation_report["errors"].append(
                f"Found {cleaning_report['banned_count']} banned features"
            )
        
        # 2. Data quality validation
        if check_data_quality and not df.empty:
            validation_report["data_quality"] = self._validate_data_quality(df, feature_cols)
        
        # 3. Leakage pattern detection
        validation_report["leakage_checks"] = self._detect_leakage_patterns(df, feature_cols)
        
        # 4. Feature correlation analysis
        if len(feature_cols) > 1 and not df.empty:
            validation_report["correlation_analysis"] = self._analyze_feature_correlations(
                df, feature_cols
            )
        
        # Log results
        if validation_report["validation_passed"]:
            logger.info("✅ Feature validation PASSED")
        else:
            logger.error("❌ Feature validation FAILED")
            for error in validation_report["errors"]:
                logger.error(f"  - {error}")
        
        for warning in validation_report["warnings"]:
            logger.warning(f"  ⚠️ {warning}")
        
        return validation_report
    
    def _validate_data_quality(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Validate data quality of features."""
        quality_report = {
            "missing_data": {},
            "infinite_values": {},
            "constant_features": [],
            "high_missing_features": [],
            "data_type_issues": {}
        }
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            # Missing data
            missing_pct = series.isna().sum() / len(series) * 100
            quality_report["missing_data"][col] = missing_pct
            
            if missing_pct > 50:
                quality_report["high_missing_features"].append(col)
            
            # Infinite values
            if pd.api.types.is_numeric_dtype(series):
                inf_count = np.isinf(series).sum()
                if inf_count > 0:
                    quality_report["infinite_values"][col] = inf_count
                
                # Constant features
                if series.nunique() <= 1:
                    quality_report["constant_features"].append(col)
            
            # Data type validation
            if not pd.api.types.is_numeric_dtype(series):
                quality_report["data_type_issues"][col] = str(series.dtype)
        
        return quality_report
    
    def _detect_leakage_patterns(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Detect potential data leakage patterns in feature data."""
        leakage_report = {
            "future_nan_patterns": [],
            "perfect_correlations": [],
            "suspicious_distributions": []
        }
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            # Check NaN patterns (future NaNs might indicate look-ahead)
            if series.isna().sum() > 0:
                # Find if NaNs are clustered at the end (good) or scattered (suspicious)
                nan_positions = series.isna()
                
                # Check if last N values are NaN (expected for forward-looking features)
                last_10_pct = int(len(series) * 0.1)
                if last_10_pct > 0:
                    recent_nans = nan_positions.iloc[-last_10_pct:].sum()
                    total_nans = nan_positions.sum()
                    
                    if recent_nans < total_nans * 0.5:  # Less than 50% of NaNs at end
                        leakage_report["future_nan_patterns"].append(
                            f"{col}: Scattered NaNs (potential look-ahead)"
                        )
        
        return leakage_report
    
    def _analyze_feature_correlations(self, df: pd.DataFrame, 
                                    feature_cols: List[str]) -> Dict:
        """Analyze feature correlations for potential leakage."""
        try:
            feature_df = df[feature_cols].select_dtypes(include=[np.number])
            if feature_df.empty:
                return {"error": "No numeric features for correlation analysis"}
            
            corr_matrix = feature_df.corr()
            
            # Find perfect or near-perfect correlations (potential duplicates)
            high_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.95 and not np.isnan(corr_val):
                        high_corrs.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val
                        })
            
            return {
                "high_correlations": high_corrs,
                "correlation_matrix_shape": corr_matrix.shape,
                "max_correlation": corr_matrix.abs().max().max() if not corr_matrix.empty else 0
            }
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def audit_feature_pipeline(self, df: pd.DataFrame, feature_cols: List[str],
                              auto_fix: bool = True) -> Tuple[List[str], Dict]:
        """
        Complete feature audit pipeline.
        
        Args:
            df: DataFrame with features
            feature_cols: Original feature list
            auto_fix: Whether to automatically fix issues
            
        Returns:
            Tuple of (final_clean_features, comprehensive_audit_report)
        """
        logger.info("🔬 Starting comprehensive feature audit pipeline")
        
        # Step 1: Name-based cleaning
        clean_features, cleaning_report = self.clean_features(feature_cols, auto_remove=auto_fix)
        
        # Step 2: Data validation
        validation_report = self.validate_features(df, clean_features, check_data_quality=True)
        
        # Step 3: Final filtering based on data quality
        if auto_fix and "data_quality" in validation_report:
            dq = validation_report["data_quality"]
            
            # Remove features with severe data quality issues
            features_to_remove = set()
            features_to_remove.update(dq.get("constant_features", []))
            features_to_remove.update(dq.get("high_missing_features", []))
            features_to_remove.update(dq.get("data_type_issues", {}).keys())
            
            final_features = [f for f in clean_features if f not in features_to_remove]
            
            if features_to_remove:
                logger.warning(f"🗑️ Removed {len(features_to_remove)} features due to data quality issues:")
                for feature in features_to_remove:
                    logger.warning(f"  - {feature}")
        else:
            final_features = clean_features
        
        # Comprehensive audit report
        audit_report = {
            "original_feature_count": len(feature_cols),
            "final_feature_count": len(final_features),
            "features_removed": len(feature_cols) - len(final_features),
            "removal_rate": (len(feature_cols) - len(final_features)) / len(feature_cols) if feature_cols else 0,
            "cleaning_report": cleaning_report,
            "validation_report": validation_report,
            "final_features": final_features,
            "audit_passed": validation_report["validation_passed"] and len(final_features) > 0
        }
        
        # Final audit summary
        if audit_report["audit_passed"]:
            logger.info(f"✅ Feature audit PASSED: {len(final_features)} clean features ready")
        else:
            logger.error("❌ Feature audit FAILED: Issues found in feature pipeline")
        
        return final_features, audit_report


def create_clean_feature_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Create a clean feature set from raw financial data.
    
    Builds only lagged features with proper look-back windows.
    No future-leaking features allowed.
    
    Args:
        df: Raw financial DataFrame with OHLCV data
        
    Returns:
        Tuple of (features_df, clean_feature_names, feature_metadata)
    """
    logger.info("🏗️ Creating clean feature set from financial data")
    
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    features_df = df.copy()
    feature_metadata = {
        "created_features": [],
        "feature_types": {},
        "lookback_windows": {}
    }
    
    # Price-based features (all lagged)
    logger.info("Adding price-based features...")
    
    # Returns (various horizons, all historical)
    for lag in [1, 2, 3, 5, 10, 20]:
        feat_name = f"returns_{lag}d"
        features_df[feat_name] = features_df['close'].pct_change(lag)
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "returns"
        feature_metadata["lookback_windows"][feat_name] = lag
    
    # Log returns
    for lag in [1, 5, 10]:
        feat_name = f"log_returns_{lag}d"
        features_df[feat_name] = np.log(features_df['close'] / features_df['close'].shift(lag))
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "log_returns"
        feature_metadata["lookback_windows"][feat_name] = lag
    
    # Moving averages (SMA)
    logger.info("Adding moving average features...")
    for window in [5, 10, 20, 50, 100, 200]:
        feat_name = f"sma_{window}"
        features_df[feat_name] = features_df['close'].rolling(window, min_periods=window).mean()
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "sma"
        feature_metadata["lookback_windows"][feat_name] = window
    
    # Exponential moving averages (EMA)
    for window in [12, 26, 50]:
        feat_name = f"ema_{window}"
        features_df[feat_name] = features_df['close'].ewm(span=window, min_periods=window).mean()
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "ema"
        feature_metadata["lookback_windows"][feat_name] = window
    
    # RSI (Relative Strength Index)
    logger.info("Adding technical indicator features...")
    for window in [14, 21, 30]:
        feat_name = f"rsi_{window}"
        features_df[feat_name] = calculate_rsi(features_df['close'], window)
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "rsi"
        feature_metadata["lookback_windows"][feat_name] = window
    
    # Volatility (rolling standard deviation)
    for window in [5, 10, 20, 30]:
        feat_name = f"volatility_{window}d"
        features_df[feat_name] = features_df['returns_1d'].rolling(window, min_periods=window).std()
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "volatility"
        feature_metadata["lookback_windows"][feat_name] = window
    
    # Momentum indicators
    for window in [10, 20, 30]:
        feat_name = f"momentum_{window}d"
        features_df[feat_name] = features_df['close'] / features_df['close'].shift(window) - 1
        feature_metadata["created_features"].append(feat_name)
        feature_metadata["feature_types"][feat_name] = "momentum"
        feature_metadata["lookback_windows"][feat_name] = window
    
    # Volume-based features (if available)
    if 'volume' in features_df.columns:
        logger.info("Adding volume-based features...")
        
        # Volume moving averages
        for window in [10, 20, 50]:
            feat_name = f"volume_sma_{window}"
            features_df[feat_name] = features_df['volume'].rolling(window, min_periods=window).mean()
            feature_metadata["created_features"].append(feat_name)
            feature_metadata["feature_types"][feat_name] = "volume_sma"
            feature_metadata["lookback_windows"][feat_name] = window
        
        # Volume ratio (current vs average)
        features_df['volume_ratio_20d'] = features_df['volume'] / features_df['volume_sma_20']
        feature_metadata["created_features"].append('volume_ratio_20d')
        feature_metadata["feature_types"]['volume_ratio_20d'] = "volume_ratio"
        feature_metadata["lookback_windows"]['volume_ratio_20d'] = 20
    
    # MACD
    ema_12 = features_df['close'].ewm(span=12).mean()
    ema_26 = features_df['close'].ewm(span=26).mean()
    features_df['macd'] = ema_12 - ema_26
    features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
    features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
    
    for feat in ['macd', 'macd_signal', 'macd_histogram']:
        feature_metadata["created_features"].append(feat)
        feature_metadata["feature_types"][feat] = "macd"
        feature_metadata["lookback_windows"][feat] = 26  # Uses 26-day EMA
    
    # Get only the created features
    clean_features = feature_metadata["created_features"]
    
    logger.info(f"✅ Created {len(clean_features)} clean features")
    logger.info(f"Feature types: {len(set(feature_metadata['feature_types'].values()))}")
    
    return features_df, clean_features, feature_metadata


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using only historical data."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Example usage and validation
def demonstrate_feature_cleaning():
    """Demonstrate the feature cleaning system."""
    
    # Create sample data with both good and bad features
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_samples = len(dates)
    
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_samples)))
    
    df = pd.DataFrame({
        'close': prices,
        'volume': rng.lognormal(10, 1, n_samples),
        'high': prices * (1 + np.abs(rng.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(rng.normal(0, 0.01, n_samples))),
    }, index=dates)
    
    # Add good features
    df['returns_1d'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['volatility_10d'] = df['returns_1d'].rolling(10).std()
    
    # Add BAD features (should be removed)
    df['forward_returns_BAD'] = df['close'].pct_change(-5)  # Look-ahead!
    df['future_volatility_BAD'] = df['returns_1d'].rolling(10).std().shift(-10)
    df['next_5d_returns'] = df['close'].pct_change(-5)
    df['leak_feature'] = df['close'].shift(-1)  # Tomorrow's price!
    
    # Test feature cleaning
    cleaner = FeatureCleaner(strict_mode=True)
    
    all_features = ['returns_1d', 'sma_20', 'rsi_14', 'volatility_10d', 
                   'forward_returns_BAD', 'future_volatility_BAD', 
                   'next_5d_returns', 'leak_feature']
    
    print("🧪 Testing Feature Cleaning System")
    print("=" * 50)
    
    # Test cleaning
    clean_features, cleaning_report = cleaner.clean_features(all_features)
    
    print(f"Original features: {len(all_features)}")
    print(f"Clean features: {len(clean_features)}")
    print(f"Features removed: {cleaning_report['banned_count']}")
    
    # Test full audit
    final_features, audit_report = cleaner.audit_feature_pipeline(df, all_features)
    
    print("\nFinal audit results:")
    print(f"✅ Audit passed: {audit_report['audit_passed']}")
    print(f"🧹 Final features: {len(final_features)}")
    
    return final_features, audit_report


if __name__ == "__main__":
    demonstrate_feature_cleaning()