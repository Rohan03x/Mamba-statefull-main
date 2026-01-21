"""
Regime-aware feature generation for market adaptation.

Adds explicit market regime signals that vary across time periods to enable
model adaptation across different market conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _sanitize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all datetime columns/index to tz-naive to avoid tz-aware/naive comparison bugs."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            pass
    
    # Sanitize datetime columns
    for col in ("date", "timestamp", "time", "Date", "Timestamp", "Time"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
            except Exception:
                pass
    return df


def add_vix_regime_features(data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add VIX percentile regime features.
    
    Args:
        data: Main dataframe with market data
        vix_data: VIX time series (if None, will try to compute from data)
    
    Returns:
        DataFrame with added VIX regime features:
        - vix_percentile: Rolling percentile of VIX (0-100)
        - vix_regime: Categorical (low_vol, normal, elevated, crisis)
        - vix_regime_change: Binary signal for regime transitions
    """
    df = _sanitize_datetime_index(data.copy())
    
    if vix_data is not None and 'close' in vix_data.columns:
        # Merge VIX data
        vix_data = _sanitize_datetime_index(vix_data.copy())
        vix_series = vix_data['close'].reindex(df.index, method='ffill')
    elif 'VIX' in df.columns:
        vix_series = df['VIX']
    else:
        logger.warning("No VIX data available, skipping VIX regime features")
        return df
    
    # Calculate rolling percentile (252 trading days = 1 year window)
    df['vix_percentile'] = vix_series.rolling(window=252, min_periods=63).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    # Create regime categories
    df['vix_regime'] = pd.cut(
        df['vix_percentile'],
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=['low_vol', 'normal', 'elevated', 'crisis']
    )
    
    # One-hot encode regimes
    regime_dummies = pd.get_dummies(df['vix_regime'], prefix='vix')
    df = pd.concat([df, regime_dummies], axis=1)
    
    # Detect regime changes
    df['vix_regime_change'] = (df['vix_regime'] != df['vix_regime'].shift(1)).astype(int)
    
    logger.info("✅ Added VIX regime features (percentile, 4 regime dummies, regime_change)")
    return df


def add_economic_cycle_features(data: pd.DataFrame, economic_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add economic cycle regime features.
    
    Uses GDP growth, unemployment, and sentiment indicators to classify cycles.
    
    Returns:
        DataFrame with added features:
        - econ_cycle: expansion, slowdown, recession, recovery
        - cycle_phase_*: One-hot encoded cycles
    """
    df = data.copy()
    
    if economic_data is None:
        # Simplified proxy: use SPY momentum + volatility
        if 'SPY' in df.columns:
            spy_returns = df['SPY'].pct_change()
            df['SPY'].rolling(50).mean()
            spy_ma_200 = df['SPY'].rolling(200).mean()
            
            # Expansion: SPY above MA200, positive momentum
            # Slowdown: SPY above MA200, negative momentum  
            # Recession: SPY below MA200, negative momentum
            # Recovery: SPY below MA200, positive momentum
            
            momentum = (spy_returns.rolling(20).mean() > 0).astype(int)
            trend = (df['SPY'] > spy_ma_200).astype(int)
            
            conditions = [
                (trend == 1) & (momentum == 1),  # expansion
                (trend == 1) & (momentum == 0),  # slowdown
                (trend == 0) & (momentum == 0),  # recession
                (trend == 0) & (momentum == 1),  # recovery
            ]
            choices = ['expansion', 'slowdown', 'recession', 'recovery']
            df['econ_cycle'] = np.select(conditions, choices, default='normal')
            
            # One-hot encode
            cycle_dummies = pd.get_dummies(df['econ_cycle'], prefix='cycle')
            df = pd.concat([df, cycle_dummies], axis=1)
            
            logger.info("✅ Added economic cycle features (4 cycle phases)")
        else:
            logger.warning("No SPY data available for economic cycle proxy")
    
    return df


def add_market_regime_labels(data: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """
    Add bull/bear/sideways regime labels based on price trends.
    
    Args:
        data: DataFrame with 'close' or 'Close' price
        lookback: Days to look back for trend calculation
    
    Returns:
        DataFrame with:
        - market_regime: bull, bear, sideways
        - regime_*: One-hot encoded regimes
        - regime_strength: Magnitude of trend (0-1)
    """
    df = _sanitize_datetime_index(data.copy())
    
    # Get price column
    price_col = 'close' if 'close' in df.columns else 'Close'
    if price_col not in df.columns:
        logger.warning("No price column found, skipping market regime features")
        return df
    
    prices = df[price_col]
    
    # Calculate trend: compare current price to MA and trend slope
    ma = prices.rolling(lookback).mean()
    ma_slope = ma.diff(20) / ma  # 20-day slope as percentage
    
    # Bull: price above MA and positive slope
    # Bear: price below MA and negative slope
    # Sideways: mixed signals
    
    above_ma = (prices > ma).astype(int)
    positive_slope = (ma_slope > 0.001).astype(int)  # 0.1% threshold
    negative_slope = (ma_slope < -0.001).astype(int)
    
    conditions = [
        (above_ma == 1) & (positive_slope == 1),  # bull
        (above_ma == 0) & (negative_slope == 1),  # bear
    ]
    choices = ['bull', 'bear']
    df['market_regime'] = np.select(conditions, choices, default='sideways')
    
    # One-hot encode
    regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
    df = pd.concat([df, regime_dummies], axis=1)
    
    # Regime strength (absolute value of normalized slope)
    df['regime_strength'] = np.abs(ma_slope).clip(0, 0.05) / 0.05  # Normalize to 0-1
    
    logger.info("✅ Added market regime labels (bull/bear/sideways, strength)")
    return df


def add_time_cyclical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time encodings to capture seasonal patterns.
    
    Returns:
        DataFrame with:
        - month_sin, month_cos: Cyclical month encoding
        - quarter_sin, quarter_cos: Cyclical quarter encoding
        - day_of_week: Numeric day (0=Monday, 4=Friday)
        - days_since_year_start: Days elapsed in current year
    """
    df = data.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, skipping time features")
        return df
    
    # Month cyclical encoding (12 months)
    month = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Quarter cyclical encoding (4 quarters)
    quarter = df.index.quarter
    df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    # Day of week (0=Monday through 4=Friday for trading days)
    df['day_of_week'] = df.index.dayofweek
    
    # Days since start of year (normalized to 0-1)
    df['days_since_year_start'] = df.index.dayofyear / 365.25
    
    logger.info("✅ Added cyclical time features (month, quarter, day_of_week, year_progress)")
    return df


def add_event_distance_features(data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add features measuring days since significant market events.
    
    Returns:
        DataFrame with:
        - days_since_vol_spike: Days since VIX spike (>90th percentile)
        - days_since_drawdown: Days since -5% daily drop
        - recent_volatility_flag: Binary flag for recent volatility (last 30 days)
    """
    df = data.copy()
    
    # Get returns
    if 'close' in df.columns:
        returns = df['close'].pct_change()
    elif 'Close' in df.columns:
        returns = df['Close'].pct_change()
    else:
        logger.warning("No price data for event distance features")
        return df
    
    # Days since major drawdown (-5% or worse)
    drawdown_events = (returns < -0.05).astype(int)
    df['days_since_drawdown'] = _days_since_event(drawdown_events)
    
    # Days since VIX spike
    if vix_data is not None and 'close' in vix_data.columns:
        vix_series = vix_data['close'].reindex(df.index, method='ffill')
        vix_threshold = vix_series.rolling(252).quantile(0.9)
        vix_spike = (vix_series > vix_threshold).astype(int)
        df['days_since_vol_spike'] = _days_since_event(vix_spike)
    elif 'VIX' in df.columns:
        vix_threshold = df['VIX'].rolling(252).quantile(0.9)
        vix_spike = (df['VIX'] > vix_threshold).astype(int)
        df['days_since_vol_spike'] = _days_since_event(vix_spike)
    
    # Recent volatility flag (any event in last 30 days)
    if 'days_since_drawdown' in df.columns:
        df['recent_volatility_flag'] = (df['days_since_drawdown'] <= 30).astype(int)
    
    logger.info("✅ Added event distance features (days_since_drawdown, days_since_vol_spike, recent_volatility)")
    return df


def _days_since_event(event_series: pd.Series) -> pd.Series:
    """
    Calculate days since last event (where event_series == 1).
    
    Returns:
        Series with integer days since last event, capped at 999.
    """
    days_since = pd.Series(index=event_series.index, dtype=int)
    counter = 999  # Start with large number
    
    for idx in event_series.index:
        if event_series.loc[idx] == 1:
            counter = 0
        else:
            counter = min(counter + 1, 999)  # Cap at 999
        days_since.loc[idx] = counter
    
    return days_since


def add_all_regime_features(
    data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
    economic_data: Optional[pd.DataFrame] = None,
    enable_vix: bool = True,
    enable_economic: bool = True,
    enable_market: bool = True,
    enable_time: bool = True,
    enable_events: bool = True
) -> pd.DataFrame:
    """
    Add all regime-aware features to the dataframe.
    
    This is the main entry point for adding regime features.
    
    Args:
        data: Main market data dataframe
        vix_data: Optional VIX time series
        economic_data: Optional economic indicators
        enable_*: Flags to enable/disable feature groups
    
    Returns:
        DataFrame with all enabled regime features added
    """
    df = data.copy()
    
    if enable_vix:
        df = add_vix_regime_features(df, vix_data)
    
    if enable_economic:
        df = add_economic_cycle_features(df, economic_data)
    
    if enable_market:
        df = add_market_regime_labels(df)
    
    if enable_time:
        df = add_time_cyclical_features(df)
    
    if enable_events:
        df = add_event_distance_features(df, vix_data)
    
    n_new_features = len(df.columns) - len(data.columns)
    logger.info(f"🎯 Added {n_new_features} regime-aware features total")
    
    return df
