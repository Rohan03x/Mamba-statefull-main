"""
Regime Classification Module for Stage A
Classifies market conditions as bull, bear, or sideways based on multiple metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Classifies market regimes using rolling metrics:
    - Bull: Strong uptrend with moderate volatility
    - Bear: Downtrend or high volatility correction
    - Sideways: Range-bound or choppy conditions
    """
    
    def __init__(
        self,
        lookback_periods: Dict[str, int] = None,
        bull_threshold: float = 0.02,
        bear_threshold: float = -0.02,
        vol_bull_max: float = 0.5,
        vol_bear_min: float = 1.0,
        drawdown_bear_threshold: float = -0.15
    ):
        """
        Args:
            lookback_periods: Dict of metric names to lookback days
            bull_threshold: Min return for bull classification
            bear_threshold: Max return for bear classification
            vol_bull_max: Max volatility zscore for bull
            vol_bear_min: Min volatility zscore for bear
            drawdown_bear_threshold: Max drawdown for bear
        """
        self.lookback_periods = lookback_periods or {
            'return_short': 21,   # 1 month
            'return_medium': 63,  # 3 months
            'return_long': 126,   # 6 months
            'volatility': 21,
            'drawdown': 63
        }
        
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.vol_bull_max = vol_bull_max
        self.vol_bear_min = vol_bear_min
        self.drawdown_bear_threshold = drawdown_bear_threshold
        
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling features for regime classification.
        
        Args:
            df: DataFrame with 'close' prices and datetime index
            
        Returns:
            DataFrame with regime features added
        """
        df = df.copy()
        
        # Rolling returns
        for name, period in self.lookback_periods.items():
            if 'return' in name:
                df[f'{name}_{period}d'] = df['close'].pct_change(period)
        
        # Rolling volatility (21-day standard deviation)
        vol_period = self.lookback_periods['volatility']
        df['volatility'] = df['close'].pct_change().rolling(vol_period).std() * np.sqrt(252)
        
        # Volatility zscore (normalized)
        vol_mean = df['volatility'].rolling(252).mean()
        vol_std = df['volatility'].rolling(252).std()
        df['volatility_zscore'] = (df['volatility'] - vol_mean) / (vol_std + 1e-8)
        
        # Rolling drawdown
        dd_period = self.lookback_periods['drawdown']
        rolling_max = df['close'].rolling(dd_period).max()
        df['drawdown'] = (df['close'] - rolling_max) / rolling_max
        
        # Trend strength (% of days up in rolling window)
        df['trend_strength'] = (df['close'].pct_change() > 0).rolling(21).mean()
        
        return df
    
    def classify_regime(self, row: pd.Series) -> str:
        """
        Classify a single row into bull/bear/sideways regime.
        
        Args:
            row: Series with regime features
            
        Returns:
            Regime label: 'bull', 'bear', or 'sideways'
        """
        # Extract features with safe defaults
        ret_medium = row.get('return_medium_63d', 0)
        ret_long = row.get('return_long_126d', 0)
        vol_z = row.get('volatility_zscore', 0)
        drawdown = row.get('drawdown', 0)
        trend_strength = row.get('trend_strength', 0.5)
        
        # Handle NaN values
        if pd.isna(ret_medium) or pd.isna(vol_z):
            return 'sideways'
        
        # Bear market conditions (prioritize)
        if drawdown < self.drawdown_bear_threshold:
            return 'bear'
        
        if ret_medium < self.bear_threshold and vol_z > self.vol_bear_min:
            return 'bear'
        
        if ret_long < self.bear_threshold * 2:  # Sustained decline
            return 'bear'
        
        # Bull market conditions
        if ret_medium > self.bull_threshold and vol_z < self.vol_bull_max and trend_strength > 0.55:
            return 'bull'
        
        if ret_long > self.bull_threshold * 2 and vol_z < self.vol_bull_max:
            return 'bull'
        
        # Default to sideways
        return 'sideways'
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime for entire DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with 'regime' column added
        """
        logger.info("🔍 Computing regime features...")
        df = self.compute_regime_features(df)
        
        logger.info("📊 Classifying regimes...")
        df['regime'] = df.apply(self.classify_regime, axis=1)
        
        # Log regime distribution
        regime_counts = df['regime'].value_counts()
        total = len(df)
        logger.info("📊 Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / total * 100
            logger.info(f"   {regime}: {count} samples ({pct:.1f}%)")
        
        return df
    
    def compute_regime_weights(self, regimes: pd.Series) -> pd.Series:
        """
        Compute sample weights to balance regime representation.
        Uses inverse frequency weighting.
        
        Args:
            regimes: Series of regime labels
            
        Returns:
            Series of sample weights (same index as input)
        """
        regime_counts = regimes.value_counts()
        total = len(regimes)
        
        # Inverse frequency weights
        regime_weights = {}
        for regime, count in regime_counts.items():
            # Weight inversely proportional to frequency
            regime_weights[regime] = total / (len(regime_counts) * count)
        
        # Map weights to samples
        sample_weights = regimes.map(regime_weights)
        
        # Normalize to sum to total samples
        sample_weights = sample_weights * (total / sample_weights.sum())
        
        logger.info("⚖️  Regime Weights (balanced):")
        for regime in regime_counts.index:
            avg_weight = sample_weights[regimes == regime].mean()
            logger.info(f"   {regime}: {avg_weight:.3f}")
        
        return sample_weights


def detect_regime_for_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    classifier: RegimeClassifier = None
) -> Tuple[str, Dict]:
    """
    Detect dominant regime for a specific time period.
    
    Args:
        df: DataFrame with price data and datetime index
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        classifier: Optional RegimeClassifier instance
        
    Returns:
        Tuple of (dominant_regime, regime_stats)
    """
    if classifier is None:
        classifier = RegimeClassifier()
    
    # Filter to period
    mask = (df.index >= start_date) & (df.index <= end_date)
    period_df = df[mask].copy()
    
    if len(period_df) == 0:
        logger.warning(f"No data found for period {start_date} to {end_date}")
        return 'sideways', {}
    
    # Classify
    period_df = classifier.classify_dataframe(period_df)
    
    # Get dominant regime (mode)
    regime_counts = period_df['regime'].value_counts()
    dominant_regime = regime_counts.idxmax()
    
    # Compute statistics
    total = len(period_df)
    regime_stats = {
        'dominant_regime': dominant_regime,
        'bull_pct': regime_counts.get('bull', 0) / total * 100,
        'bear_pct': regime_counts.get('bear', 0) / total * 100,
        'sideways_pct': regime_counts.get('sideways', 0) / total * 100,
        'total_samples': total,
        'start_date': start_date,
        'end_date': end_date
    }
    
    logger.info(f"📊 Period {start_date} to {end_date}: {dominant_regime} "
                f"({regime_stats[f'{dominant_regime}_pct']:.1f}%)")
    
    return dominant_regime, regime_stats


def save_regime_labels(df: pd.DataFrame, output_path: str):
    """
    Save regime labels to CSV for later use.
    
    Args:
        df: DataFrame with 'regime' column
        output_path: Path to save CSV
    """
    # Select only relevant columns
    regime_cols = ['regime']
    
    # Add regime features if present
    feature_cols = [col for col in df.columns if any(x in col for x in 
                   ['return_', 'volatility', 'drawdown', 'trend_strength'])]
    
    save_cols = regime_cols + feature_cols
    save_cols = [col for col in save_cols if col in df.columns]
    
    df[save_cols].to_csv(output_path)
    logger.info(f"💾 Saved regime labels to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate bull/bear cycles
    returns = np.random.randn(len(dates)) * 0.01
    returns[:1000] += 0.001  # Bull phase
    returns[1000:1500] -= 0.002  # Bear phase
    returns[1500:2500] += 0.0005  # Sideways
    returns[2500:] += 0.001  # Bull again
    
    prices = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Classify
    classifier = RegimeClassifier()
    df = classifier.classify_dataframe(df)
    
    # Compute weights
    weights = classifier.compute_regime_weights(df['regime'])
    
    print("\n✅ Regime classification complete!")
    print(f"Total samples: {len(df)}")
    print(f"Unique regimes: {df['regime'].nunique()}")
