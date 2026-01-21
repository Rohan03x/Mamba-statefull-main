"""
Price Anomalies Module
======================

Extracts 7 price-based anomaly features from OHLCV data.
All features are alpha-positive signals used by quantitative traders.

Features:
1. gap_up_pct - Positive gap from previous close
2. gap_down_pct - Negative gap from previous close
3. overnight_return - Return from prev close to open
4. intraday_range_pct - (High - Low) / Close
5. high_low_volatility_ratio - Parkinson volatility estimator
6. trend_acceleration - Slope change (3d vs 10d)
7. mean_reversion_signal - Z-score from SMA20
"""

import pandas as pd
import numpy as np
from typing import Dict


def extract_price_anomalies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Extract price anomaly features from OHLCV data.
    
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
    required = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        return {f'price_anomaly_{i}': pd.Series(dtype=float) for i in range(1, 8)}
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    
    # 1. Gap Up Percentage
    prev_close = close.shift(1)
    gap = open_ - prev_close
    features['gap_up_pct'] = (gap / prev_close * 100).clip(lower=0)  # Only positive gaps
    
    # 2. Gap Down Percentage
    features['gap_down_pct'] = (gap / prev_close * 100).clip(upper=0).abs()  # Only negative gaps
    
    # 3. Overnight Return
    features['overnight_return'] = (open_ - prev_close) / prev_close * 100
    
    # 4. Intraday Range Percentage
    features['intraday_range_pct'] = (high - low) / close * 100
    
    # 5. High-Low Volatility Ratio (Parkinson estimator)
    # Annualized volatility using high-low range
    hl_ratio = np.log(high / low) ** 2
    features['high_low_volatility_ratio'] = (hl_ratio.rolling(20).mean() * np.sqrt(252 / (4 * np.log(2)))) * 100
    
    # 6. Trend Acceleration (normalized)
    # Slope of last 3 days vs slope of last 10 days, z-scored over 60-day window
    returns = close.pct_change()
    slope_3d = returns.rolling(3).mean()
    slope_10d = returns.rolling(10).mean()
    slope_diff = (slope_3d - slope_10d) * 100
    # Z-score normalization over 60-day window
    slope_mean = slope_diff.rolling(60).mean()
    slope_std = slope_diff.rolling(60).std()
    features['trend_acceleration'] = (slope_diff - slope_mean) / slope_std.replace(0, 1)
    
    # 7. Mean Reversion Signal (winsorized)
    # Z-score from 20-day SMA, winsorized at ±3 sigma
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    z_score = (close - sma20) / std20.replace(0, 1)
    features['mean_reversion_signal'] = z_score.clip(-3, 3)  # Winsorize tails
    
    return features
