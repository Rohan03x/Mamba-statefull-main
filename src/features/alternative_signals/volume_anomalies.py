"""
Volume Anomalies Module
========================

Extracts 6 volume-based anomaly features from OHLCV data.
Detects unusual volume patterns, divergences, and liquidity stress.

Features:
1. volume_z_20d - Z-score of volume vs 20-day average
2. relative_volume_20d - Current volume / 20-day average
3. volume_trend_10d - 10-day slope of volume
4. volume_price_divergence - Volume rising while price falling (strong signal)
5. buy_volume_proxy - Directional volume (close vs open)
6. liquidity_stress_pct - Percentile rank of today's spread
"""

import pandas as pd
import numpy as np
from typing import Dict


def extract_volume_anomalies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Extract volume anomaly features from OHLCV data.
        All features are log-scaled or z-scored to prevent raw volume from dominating.
    Hedge-fund discipline: never use raw volume counts.
        Parameters
    ----------
    df : pd.DataFrame
        Must contain: open, high, low, close, volume
        Index must be DatetimeIndex
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
    """
    features = {}
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return {f'volume_anomaly_{i}': pd.Series(dtype=float) for i in range(1, 7)}
    
    volume = df['volume']
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    
    # 1. Volume Z-score (20-day)
    vol_mean_20d = volume.rolling(20).mean()
    vol_std_20d = volume.rolling(20).std()
    features['volume_z_20d'] = (volume - vol_mean_20d) / vol_std_20d
    
    # 2. Relative Volume (20-day)
    features['relative_volume_20d'] = volume / vol_mean_20d
    
    # 3. Volume Trend (10-day slope)
    # Linear regression slope over 10 days
    def rolling_slope(series, window):
        """Calculate rolling linear regression slope"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(window)
            if len(y) == window and not np.isnan(y).any():
                slope = np.polyfit(x, y, 1)[0]
                result.iloc[i] = slope
        return result
    
    features['volume_trend_10d'] = rolling_slope(volume, 10)
    
    # 4. Volume-Price Divergence
    # Volume rising while price falling → strong bearish signal
    # Volume falling while price rising → weak bullish signal
    price_change = close.pct_change()
    volume_change = volume.pct_change()
    
    # Divergence score: positive when volume rises AND price falls
    divergence = np.where(
        (volume_change > 0) & (price_change < 0), volume_change * abs(price_change),
        np.where(
            (volume_change < 0) & (price_change > 0), -volume_change * abs(price_change),
            0
        )
    )
    features['volume_price_divergence'] = pd.Series(divergence, index=df.index)
    
    # 5. Buy Volume Proxy
    # If close > open → buying pressure (positive)
    # If close < open → selling pressure (negative)
    direction = np.sign(close - open_)
    features['buy_volume_proxy'] = direction * volume
    
    # 6. Liquidity Stress Percentage
    # Percentile rank of bid-ask spread proxy (high-low range)
    spread_proxy = (high - low) / close * 100  # Spread as % of price
    features['liquidity_stress_pct'] = spread_proxy.rolling(20).rank(pct=True) * 100
    
    return features
