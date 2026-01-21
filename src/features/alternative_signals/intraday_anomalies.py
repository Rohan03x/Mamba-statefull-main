"""
Intraday Anomalies Module
=========================

Extracts 4 best intraday features for daily models (hedge-fund grade).
Focuses on open/close effects with highest alpha.

Features (4):
1. opening_reversal - Open to close price movement (institutional positioning)
2. closing_ramp - Last hour strength relative to range (institutional accumulation)
3. opening_volume_surge - First 30min volume vs daily average (information flow)
4. intraday_volatility_ratio - High/Low range vs overnight gap (range efficiency)

REMOVED (noisy for daily models, hedge-fund discipline):
- midday_drift (redundant with returns, low alpha)
- time_weighted_returns (experimental, inconsistent)
- session_momentum (redundant with returns)
- liquidity_hourglass (noisy without true intraday data)

Data Source: EODHD Intraday API (5-minute bars)
Fallback: OHLC-based proxies when intraday unavailable

Author: DCF Lab Team, Refactored: Jan 2026
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_intraday_anomalies(
    df: pd.DataFrame,
    symbol: str,
    intraday_df: Optional[pd.DataFrame] = None,
    eodhd_provider=None
) -> Dict[str, pd.Series]:
    """
    Extract time-of-day anomaly features from intraday data.
    
    Args:
        df: Daily OHLCV DataFrame with date index
        symbol: Stock symbol (e.g., 'AAPL')
        intraday_df: Optional pre-fetched intraday data (5min bars)
        eodhd_provider: EODHD provider instance to fetch intraday data
        
    Returns:
        Dictionary of feature name -> pd.Series (date-indexed)
    """
    features = {}
    
    # Try to get intraday data if not provided
    if intraday_df is None and eodhd_provider is not None:
        try:
            # EODHD intraday endpoint: GET /intraday/{symbol}?interval=5m&from=YYYY-MM-DD&to=YYYY-MM-DD
            intraday_df = _fetch_eodhd_intraday(
                symbol=symbol,
                start=df.index.min(),
                end=df.index.max(),
                provider=eodhd_provider
            )
        except Exception as e:
            logger.debug(f"Could not fetch intraday data for {symbol}: {e}")
    
    # If we have intraday data, compute precise time-of-day features
    if intraday_df is not None and not intraday_df.empty:
        features.update(_compute_intraday_features(intraday_df, df))
    else:
        # Fallback: Use OHLC-based proxies
        logger.debug(f"Using OHLC proxies for intraday features ({symbol})")
        features.update(_compute_ohlc_proxies(df))
    
    return features


def _fetch_eodhd_intraday(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    provider,
    interval: str = '5m'
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday data from EODHD API.
    
    EODHD Intraday API:
    provider.get_intraday_prices(symbol, interval='5m', start_date='YYYY-MM-DD', end_date='YYYY-MM-DD')
    
    Intervals: 1m, 5m, 1h
    Returns: DataFrame with datetime index and OHLCV columns
    """
    try:
        # Check if provider has intraday_prices method
        if not hasattr(provider, 'get_intraday_prices'):
            logger.debug("EODHD provider does not support get_intraday_prices()")
            return None

        # EODHD intraday endpoints typically reject very large date ranges.
        # For multi-year daily panels, we intentionally fall back to OHLC proxies.
        max_days = int((__import__("os").getenv("EODHD_INTRADAY_MAX_DAYS") or "60").strip() or "60")
        span_days = int((end.normalize() - start.normalize()).days)
        if span_days > max_days:
            logger.debug(
                "Skipping EODHD intraday fetch for %s: requested %d days (> %d)",
                symbol,
                span_days,
                max_days,
            )
            return None
        
        # Convert timestamps to date strings for EODHD API
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        intraday_data = provider.get_intraday_prices(
            symbol=symbol,
            interval=interval,
            start_date=start_str,
            end_date=end_str
        )
        
        if intraday_data is None or intraday_data.empty:
            return None
        
        # Normalize column names to lowercase
        intraday_data.columns = intraday_data.columns.str.lower()
        
        # Ensure datetime index if not already
        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            if 'datetime' in intraday_data.columns:
                intraday_data['datetime'] = pd.to_datetime(intraday_data['datetime'])
                intraday_data = intraday_data.set_index('datetime')
            elif 'timestamp' in intraday_data.columns:
                intraday_data.index = pd.to_datetime(intraday_data['timestamp'], unit='s')
        
        return intraday_data
        
    except Exception as e:
        logger.debug(f"Failed to fetch EODHD intraday data: {e}")
        return None


def _compute_intraday_features(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame
) -> Dict[str, pd.Series]:
    """
    Compute precise intraday features from 5-minute bars.
    
    Time zones:
    - Opening session: 9:30am - 10:00am ET (first 30 min)
    - Midday session: 10:00am - 2:00pm ET (middle 4 hours)
    - Closing session: 2:00pm - 4:00pm ET (last 2 hours)
    """
    features = {}
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in intraday_df.columns for col in required):
        logger.debug("Missing required columns in intraday data")
        return _compute_ohlc_proxies(daily_df)
    
    # Group by date
    intraday_df['date'] = intraday_df.index.date
    daily_features = []
    
    for date, day_data in intraday_df.groupby('date'):
        if day_data.empty:
            continue
        
        # Extract time of day (assuming ET timezone, adjust if needed)
        day_data = day_data.copy()
        day_data['time'] = day_data.index.time
        
        # Define sessions (using hour for simplicity)
        day_data['hour'] = day_data.index.hour
        day_data['minute'] = day_data.index.minute
        
        # Opening session: 9:30-10:00 (first 30 minutes)
        opening_mask = (
            ((day_data['hour'] == 9) & (day_data['minute'] >= 30)) |
            (day_data['hour'] == 10) & (day_data['minute'] == 0)
        )
        opening_data = day_data[opening_mask]
        
        # Midday session: 10:00-14:00
        midday_mask = (day_data['hour'] >= 10) & (day_data['hour'] < 14)
        midday_data = day_data[midday_mask]
        
        # Closing session: 14:00-16:00 (last 2 hours)
        closing_mask = (day_data['hour'] >= 14) & (day_data['hour'] < 16)
        closing_data = day_data[closing_mask]
        
        # Feature 1: Opening reversal (open-to-10am mean reversion)
        opening_reversal = 0.0
        if not opening_data.empty and len(opening_data) >= 2:
            open_price = opening_data.iloc[0]['open']
            price_10am = opening_data.iloc[-1]['close']
            opening_reversal = (price_10am - open_price) / open_price if open_price > 0 else 0.0
        
        # Feature 2: Midday drift (10am-2pm directional bias)
        midday_drift = 0.0
        if not midday_data.empty and len(midday_data) >= 2:
            midday_start = midday_data.iloc[0]['open']
            midday_end = midday_data.iloc[-1]['close']
            midday_drift = (midday_end - midday_start) / midday_start if midday_start > 0 else 0.0
        
        # Feature 3: Closing ramp (2pm-close acceleration)
        closing_ramp = 0.0
        if not closing_data.empty and len(closing_data) >= 2:
            price_2pm = closing_data.iloc[0]['open']
            close_price = closing_data.iloc[-1]['close']
            closing_ramp = (close_price - price_2pm) / price_2pm if price_2pm > 0 else 0.0
        
        # Feature 4: Opening volume surge (first 30min vs daily avg)
        opening_volume_surge = 0.0
        if not opening_data.empty and 'volume' in opening_data.columns:
            opening_vol = opening_data['volume'].sum()
            daily_vol = day_data['volume'].sum()
            avg_vol_per_bar = daily_vol / len(day_data) if len(day_data) > 0 else 1
            expected_opening_vol = avg_vol_per_bar * len(opening_data)
            opening_volume_surge = (opening_vol / expected_opening_vol - 1.0) if expected_opening_vol > 0 else 0.0
        
        # Feature 5: Intraday volatility ratio (intraday range / overnight gap)
        intraday_volatility_ratio = 1.0
        if not day_data.empty and len(day_data) >= 2:
            intraday_high = day_data['high'].max()
            intraday_low = day_data['low'].min()
            intraday_range = (intraday_high - intraday_low) / intraday_low if intraday_low > 0 else 0.0
            
            # Get previous close for overnight gap
            prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
            if prev_date.date() in intraday_df['date'].values:
                prev_data = intraday_df[intraday_df['date'] == prev_date.date()]
                if not prev_data.empty:
                    prev_close = prev_data.iloc[-1]['close']
                    current_open = day_data.iloc[0]['open']
                    overnight_gap = abs(current_open - prev_close) / prev_close if prev_close > 0 else 0.0
                    intraday_volatility_ratio = intraday_range / overnight_gap if overnight_gap > 0 else 1.0
        
        # Feature 6: Time-weighted returns (VWAP-based)
        time_weighted_returns = 0.0
        if not day_data.empty and 'volume' in day_data.columns:
            vwap = (day_data['close'] * day_data['volume']).sum() / day_data['volume'].sum()
            simple_return = (day_data.iloc[-1]['close'] - day_data.iloc[0]['open']) / day_data.iloc[0]['open']
            vwap_return = (vwap - day_data.iloc[0]['open']) / day_data.iloc[0]['open']
            time_weighted_returns = vwap_return - simple_return  # Difference shows timing alpha
        
        # Feature 7: Session momentum (price persistence across sessions)
        session_momentum = 0.0
        if not opening_data.empty and not midday_data.empty and not closing_data.empty:
            opening_ret = opening_reversal
            midday_ret = midday_drift
            closing_ret = closing_ramp
            
            # Momentum = sign consistency across sessions
            signs = np.sign([opening_ret, midday_ret, closing_ret])
            session_momentum = np.mean(signs) if len(signs) > 0 else 0.0
        
        # Feature 8: Liquidity hourglass (volume U-shape)
        liquidity_hourglass = 0.0
        if not opening_data.empty and not midday_data.empty and not closing_data.empty:
            opening_vol_avg = opening_data['volume'].mean() if not opening_data.empty else 0
            midday_vol_avg = midday_data['volume'].mean() if not midday_data.empty else 0
            closing_vol_avg = closing_data['volume'].mean() if not closing_data.empty else 0
            
            # U-shape: high at open/close, low in middle
            if midday_vol_avg > 0:
                edge_vol = (opening_vol_avg + closing_vol_avg) / 2
                liquidity_hourglass = (edge_vol / midday_vol_avg - 1.0)
        
        daily_features.append({
            'date': pd.to_datetime(date),
            'opening_reversal': opening_reversal,
            'closing_ramp': closing_ramp,
            'opening_volume_surge': opening_volume_surge,
            'intraday_volatility_ratio': intraday_volatility_ratio,
            # REMOVED: midday_drift, time_weighted_returns, session_momentum, liquidity_hourglass
        })
    
    # Convert to DataFrame and extract Series
    if daily_features:
        features_df = pd.DataFrame(daily_features).set_index('date')
        for col in features_df.columns:
            features[col] = features_df[col]
    
    return features


def _compute_ohlc_proxies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute OHLC-based proxies when intraday data is unavailable.
    
    Proxies:
    - opening_reversal ≈ (high - open) / (high - low) - suggests opening direction
    - midday_drift ≈ (close - open) / open - full day return as proxy
    - closing_ramp ≈ (close - low) / (high - low) - closing strength
    - opening_volume_surge ≈ volume zscore (can't decompose intraday)
    - intraday_volatility_ratio ≈ (high - low) / |open - prev_close|
    - time_weighted_returns ≈ 0 (can't compute without intraday)
    - session_momentum ≈ sign((close - open) / open)
    - liquidity_hourglass ≈ 0 (can't compute without intraday)
    """
    features = {}
    
    required = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required):
        logger.warning("Missing OHLC columns for intraday proxies")
        return features
    
    # HEDGE-FUND DISCIPLINE: Keep only 4 best intraday features
    # REMOVED: midday_drift, time_weighted_returns, session_momentum, liquidity_hourglass
    
    # 1. Opening reversal proxy: Where did price go after open?
    range_total = df['high'] - df['low']
    range_from_open_to_high = df['high'] - df['open']
    features['opening_reversal'] = (
        range_from_open_to_high / range_total
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # 2. Closing ramp proxy: Close relative to low (closing strength)
    features['closing_ramp'] = (
        (df['close'] - df['low']) / range_total
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # 3. Opening volume surge proxy: Volume Z-score
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        vol_std = df['volume'].rolling(window=20, min_periods=1).std()
        features['opening_volume_surge'] = (
            (df['volume'] - vol_mean) / vol_std
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        features['opening_volume_surge'] = pd.Series(0.0, index=df.index)
    
    # 4. Intraday volatility ratio: Intraday range vs overnight gap
    prev_close = df['close'].shift(1)
    overnight_gap = ((df['open'] - prev_close) / prev_close).abs()
    intraday_range = (df['high'] - df['low']) / df['low']
    features['intraday_volatility_ratio'] = (
        intraday_range / overnight_gap
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    return features


__all__ = ['extract_intraday_anomalies']
