"""
Macro Interaction Features Module
==================================

Extracts 4 critical interaction features that combine microstructure with macro regime.
These features help the model understand when microstructure matters more.

Features:
1. gap_vs_vix_interaction - Gap size × VIX level (volatility context)
2. overnight_return_z - Z-scored overnight returns (normalized regime detector)
3. intraday_range_z - Z-scored intraday range (normalized volatility)
4. beta_vix_interaction - Market beta × VIX change (systematic risk × volatility)

Why these matter:
- Gaps behave differently in high-VIX vs low-VIX environments
- Z-scores remove symbol-specific scale issues
- Beta × VIX detects when market shocks dominate idiosyncratic signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def extract_macro_interactions(
    df: pd.DataFrame,
    vix_series: Optional[pd.Series] = None,
    spy_series: Optional[pd.Series] = None
) -> Dict[str, pd.Series]:
    """
    Extract macro interaction features.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    vix_series : pd.Series, optional
        VIX index series (aligned to df.index)
    spy_series : pd.Series, optional
        SPY price series for beta calculation
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
    """
    features = {}
    
    if df.empty:
        return {
            'gap_vs_vix_interaction': pd.Series(dtype=float),
            'overnight_return_z': pd.Series(dtype=float),
            'intraday_range_z': pd.Series(dtype=float),
            'beta_vix_interaction': pd.Series(dtype=float)
        }
    
    required = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        logger.warning("Missing OHLC columns for macro interactions")
        return features
    
    # 1. Gap vs VIX Interaction
    # Larger gaps in high-VIX environments are more meaningful
    prev_close = df['close'].shift(1)
    gap_pct = ((df['open'] - prev_close) / prev_close * 100).abs()
    
    if vix_series is not None and len(vix_series) > 0:
        # Align VIX to df index
        vix_aligned = vix_series.reindex(df.index, method='ffill')
        # Normalize VIX to 0-1 scale (typical range 10-80)
        vix_norm = (vix_aligned - 10) / 70
        vix_norm = vix_norm.clip(0, 1)
        
        # Interaction: gap × VIX
        # High values = significant gap in volatile environment
        features['gap_vs_vix_interaction'] = gap_pct * vix_norm
    else:
        # Fallback: just the gap (no macro context)
        features['gap_vs_vix_interaction'] = gap_pct
    
    # 2. Overnight Return Z-score
    # Normalized overnight returns (removes symbol-specific scale)
    overnight_ret = (df['open'] - prev_close) / prev_close * 100
    ret_mean = overnight_ret.rolling(60).mean()
    ret_std = overnight_ret.rolling(60).std()
    features['overnight_return_z'] = (overnight_ret - ret_mean) / ret_std.replace(0, 1)
    
    # 3. Intraday Range Z-score
    # Normalized intraday volatility (high values = unusual volatility)
    intraday_range = (df['high'] - df['low']) / df['close'] * 100
    range_mean = intraday_range.rolling(60).mean()
    range_std = intraday_range.rolling(60).std()
    features['intraday_range_z'] = (intraday_range - range_mean) / range_std.replace(0, 1)
    
    # 4. Beta × VIX Change Interaction
    # Systematic risk × volatility regime change
    if spy_series is not None and vix_series is not None:
        # Calculate 20-day rolling beta
        spy_aligned = spy_series.reindex(df.index, method='ffill')
        stock_returns = df['close'].pct_change()
        market_returns = spy_aligned.pct_change()
        
        # Rolling covariance / variance
        cov = stock_returns.rolling(20).cov(market_returns)
        var = market_returns.rolling(20).var()
        beta = cov / var.replace(0, 1)
        
        # VIX 5-day change
        vix_aligned = vix_series.reindex(df.index, method='ffill')
        vix_change = vix_aligned.pct_change(5)
        
        # Interaction: beta × VIX change
        # High values = high-beta stock in rising VIX (danger zone)
        features['beta_vix_interaction'] = beta * vix_change
    else:
        # Fallback: zeros
        features['beta_vix_interaction'] = pd.Series(0.0, index=df.index)
    
    # Safety: clip extreme values
    for key in features:
        features[key] = features[key].replace([np.inf, -np.inf], np.nan)
        features[key] = features[key].fillna(0.0)
        # Winsorize at ±5 sigma
        features[key] = features[key].clip(-5, 5)
    
    return features


__all__ = ['extract_macro_interactions']
