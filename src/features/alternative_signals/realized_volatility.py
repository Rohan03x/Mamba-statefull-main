"""
Realized Volatility Module
===========================

Extracts 6 realized volatility features from OHLC data.
Different volatility estimators capture different aspects of price dynamics.

Features:
1. rv_5d - 5-day realized volatility (annualized)
2. rv_10d - 10-day realized volatility
3. rv_20d - 20-day realized volatility
4. rv_ratio_5_20 - Short-term vs long-term vol ratio
5. close_to_close_volatility - Traditional close-to-close vol
6. open_to_close_volatility - Intraday volatility estimator
"""

import pandas as pd
import numpy as np
from typing import Dict


def extract_realized_volatility(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Extract realized volatility features from OHLC data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain: open, high, low, close
        Index must be DatetimeIndex
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
        
    Notes
    -----
    All volatilities are annualized (multiplied by sqrt(252)).
    Expressed as percentages (multiplied by 100).
    """
    features = {}
    
    if df.empty or 'close' not in df.columns:
        return {f'realized_vol_{i}': pd.Series(dtype=float) for i in range(1, 7)}
    
    close = df['close']
    open_ = df.get('open', close)  # Fallback to close if open not available
    
    # Calculate returns
    returns = close.pct_change()
    
    # 1. Realized Volatility - 5 Day
    # Standard deviation of returns over 5 days, annualized
    features['rv_5d'] = returns.rolling(5).std() * np.sqrt(252) * 100
    
    # 2. Realized Volatility - 10 Day
    features['rv_10d'] = returns.rolling(10).std() * np.sqrt(252) * 100
    
    # 3. Realized Volatility - 20 Day
    features['rv_20d'] = returns.rolling(20).std() * np.sqrt(252) * 100
    
    # 4. RV Ratio (5-day / 20-day)
    # Ratio > 1 → recent volatility higher than longer-term
    # Ratio < 1 → recent volatility lower (calming down)
    rv_5d = returns.rolling(5).std()
    rv_20d = returns.rolling(20).std()
    features['rv_ratio_5_20'] = rv_5d / rv_20d
    
    # 5. Close-to-Close Volatility
    # Traditional volatility using only closing prices
    # Already calculated above as rv_20d, but let's make it explicit
    features['close_to_close_volatility'] = returns.rolling(20).std() * np.sqrt(252) * 100
    
    # 6. Open-to-Close Volatility (Intraday)
    # Captures intraday volatility separately from overnight gaps
    intraday_returns = (close - open_) / open_
    features['open_to_close_volatility'] = intraday_returns.rolling(20).std() * np.sqrt(252) * 100
    
    return features
