"""
Earnings & Seasonality Module
==============================

Extracts 6 event-driven features (hedge-fund grade).
Focuses on high-alpha signals: earnings timing and month-end rebalancing.

Features:
1. turn_of_month_flag - Binary flag for month-end effects (institutional rebalancing)
2. rv_z_20 - Continuous volatility regime (z-scored, replaces categorical labels)
3. days_to_next_earnings - Days until next earnings
4. days_since_last_earnings - Days since last earnings
5. earnings_runup_10d - 10-day return before earnings (anticipation)
6. post_earnings_drift_5d - 5-day return after earnings (continuation)

REMOVED (hedge-fund discipline):
- week_of_year, day_of_week (prevent calendar overfitting - Mamba learns periodicity)
- volatility_regime_label (categorical → continuous rv_z_20 for better Mamba learning)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


def extract_earnings_seasonality(
    df: pd.DataFrame,
    symbol: str,
    earnings_dates: Optional[pd.DataFrame] = None
) -> Dict[str, pd.Series]:
    """
    Extract earnings and seasonality features.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    symbol : str
        Stock ticker symbol
    earnings_dates : pd.DataFrame, optional
        DataFrame with 'date' column containing earnings announcement dates
        If None, will attempt to fetch from EODHD
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
    """
    features = {}
    
    if df.empty:
        return {f'earnings_seasonality_{i}': pd.Series(dtype=float) for i in range(1, 9)}
    
    # Calendar-based features (HEDGE-FUND DISCIPLINE)
    dates = pd.to_datetime(df.index)
    
    # REMOVED: week_of_year, day_of_week (encourage calendar overfitting)
    # Mamba learns periodicity from returns/volume directly
    
    # 1. Turn of Month Flag
    # Last 3 or first 3 trading days of month
    month = pd.Series(dates.month, index=df.index)
    month_changed = month != month.shift(1)
    month_will_change = month != month.shift(-1)
    
    # First 3 days of month
    first_3 = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if month_changed.iloc[i]:
            # Start of new month
            first_3.iloc[i:min(i+3, len(df))] = True
    
    # Last 3 days of month
    last_3 = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if i < len(df) - 1 and month_will_change.iloc[i]:
            # End of month approaching
            last_3.iloc[max(0, i-2):i+1] = True
    
    features['turn_of_month_flag'] = (first_3 | last_3).astype(int)
    
    # 2. Volatility Regime (continuous z-score, not categorical)
    # rv_z_20 replaces volatility_regime_label for Mamba compatibility
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100
        
        # Z-score of volatility over 60-day window (continuous regime indicator)
        vol_mean = vol_20d.rolling(60).mean()
        vol_std = vol_20d.rolling(60).std()
        features['rv_z_20'] = (vol_20d - vol_mean) / vol_std.replace(0, 1)
    else:
        features['rv_z_20'] = pd.Series(0.0, index=df.index)
    
    # Earnings-based features (require earnings dates)
    if earnings_dates is not None and not earnings_dates.empty and 'date' in earnings_dates.columns:
        earnings_dates_list = pd.to_datetime(earnings_dates['date']).sort_values().tolist()
        
        # Initialize arrays
        days_to_next = pd.Series(np.nan, index=df.index)
        days_since_last = pd.Series(np.nan, index=df.index)
        runup_10d = pd.Series(np.nan, index=df.index)
        drift_5d = pd.Series(np.nan, index=df.index)
        
        for i, date in enumerate(df.index):
            current_date = pd.to_datetime(date)
            
            # Find next and previous earnings dates
            future_earnings = [d for d in earnings_dates_list if d > current_date]
            past_earnings = [d for d in earnings_dates_list if d <= current_date]
            
            # 1. Days to Next Earnings
            if future_earnings:
                days_to_next.iloc[i] = (future_earnings[0] - current_date).days
            
            # 2. Days Since Last Earnings
            if past_earnings:
                days_since_last.iloc[i] = (current_date - past_earnings[-1]).days
            
            # 3. Earnings Runup (10 days before)
            if future_earnings and 'close' in df.columns:
                next_earnings = future_earnings[0]
                days_until = (next_earnings - current_date).days
                if 0 <= days_until <= 10:
                    # We're in the runup period
                    start_idx = max(0, i - 10)
                    if start_idx < i:
                        runup_return = (df['close'].iloc[i] / df['close'].iloc[start_idx] - 1) * 100
                        runup_10d.iloc[i] = runup_return
            
            # 4. Post-Earnings Drift (5 days after)
            if past_earnings and 'close' in df.columns:
                last_earnings = past_earnings[-1]
                days_since = (current_date - last_earnings).days
                if 0 <= days_since <= 5:
                    # We're in the drift period
                    # Find earnings date index
                    earnings_idx = df.index.get_indexer([last_earnings], method='nearest')[0]
                    if earnings_idx < i and earnings_idx >= 0:
                        drift_return = (df['close'].iloc[i] / df['close'].iloc[earnings_idx] - 1) * 100
                        drift_5d.iloc[i] = drift_return
        
        features['days_to_next_earnings'] = days_to_next
        features['days_since_last_earnings'] = days_since_last
        features['earnings_runup_10d'] = runup_10d
        features['post_earnings_drift_5d'] = drift_5d
    else:
        # No earnings data available - return NaN series
        features['days_to_next_earnings'] = pd.Series(np.nan, index=df.index)
        features['days_since_last_earnings'] = pd.Series(np.nan, index=df.index)
        features['earnings_runup_10d'] = pd.Series(np.nan, index=df.index)
        features['post_earnings_drift_5d'] = pd.Series(np.nan, index=df.index)
    
    return features
