"""
Event-Time Bar Generators

Implements dollar bars and volatility bars from 1-minute OHLCV data.
Aggregates intraday event-time bars to daily features.

Reference: Marcos López de Prado - "Advances in Financial Machine Learning"
- Dollar bars: sample when cumulative dollar volume exceeds threshold
- Volatility bars: sample when cumulative |return| exceeds threshold

Key insight: Event-time bars normalize information arrival rate,
making each bar approximately equally informative for learning.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EventBarConfig:
    """Configuration for event-time bar construction."""
    
    # Threshold multipliers (applied to trailing median)
    k_dollar: float = 1.0  # Dollar bar threshold multiplier
    k_vol: float = 1.0     # Volatility bar threshold multiplier
    
    # Trailing lookback for adaptive thresholds
    threshold_lookback_days: int = 20
    
    # Minimum bars per day (fallback if threshold too high)
    min_bars_per_day: int = 5
    
    # Maximum bars per day (cap if threshold too low)
    max_bars_per_day: int = 200


# ---------------------------------------------------------------------------
# Dollar Bars
# ---------------------------------------------------------------------------

def compute_dollar_bars(
    intraday_df: pd.DataFrame,
    threshold: float,
    min_bars: int = 1,
) -> pd.DataFrame:
    """
    Construct dollar bars from 1-minute OHLCV data.
    
    A new bar is emitted when cumulative dollar volume exceeds threshold.
    
    Args:
        intraday_df: DataFrame with columns [open, high, low, close, volume, dollar_volume]
                     and DatetimeIndex
        threshold: Dollar volume threshold for bar emission
        min_bars: Minimum bars to return (forces at least this many)
        
    Returns:
        DataFrame with columns:
        - bar_start: start timestamp of bar
        - bar_end: end timestamp of bar
        - open, high, low, close: OHLC prices
        - volume: total volume in bar
        - dollar_volume: total dollar volume in bar
        - bar_duration_seconds: time span of bar
        - n_ticks: number of 1-min bars aggregated
    """
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame()
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in intraday_df.columns]
    if missing:
        logger.warning(f"Missing columns for dollar bars: {missing}")
        return pd.DataFrame()
    
    # Compute dollar volume if not present
    if 'dollar_volume' not in intraday_df.columns:
        intraday_df = intraday_df.copy()
        intraday_df['dollar_volume'] = intraday_df['close'] * intraday_df['volume']
    
    bars = []
    cumulative_dv = 0.0
    bar_start_idx = 0
    
    timestamps = intraday_df.index.to_numpy()
    opens = intraday_df['open'].to_numpy()
    highs = intraday_df['high'].to_numpy()
    lows = intraday_df['low'].to_numpy()
    closes = intraday_df['close'].to_numpy()
    volumes = intraday_df['volume'].to_numpy()
    dollar_volumes = intraday_df['dollar_volume'].to_numpy()
    
    for i in range(len(intraday_df)):
        cumulative_dv += dollar_volumes[i]
        
        if cumulative_dv >= threshold:
            # Emit bar
            bar = {
                'bar_start': timestamps[bar_start_idx],
                'bar_end': timestamps[i],
                'open': opens[bar_start_idx],
                'high': highs[bar_start_idx:i+1].max(),
                'low': lows[bar_start_idx:i+1].min(),
                'close': closes[i],
                'volume': volumes[bar_start_idx:i+1].sum(),
                'dollar_volume': cumulative_dv,
                'n_ticks': i - bar_start_idx + 1,
            }
            bars.append(bar)
            
            # Reset for next bar
            cumulative_dv = 0.0
            bar_start_idx = i + 1
    
    # Handle remaining data (partial bar at end)
    if bar_start_idx < len(intraday_df) and cumulative_dv > 0:
        i = len(intraday_df) - 1
        bar = {
            'bar_start': timestamps[bar_start_idx],
            'bar_end': timestamps[i],
            'open': opens[bar_start_idx],
            'high': highs[bar_start_idx:].max(),
            'low': lows[bar_start_idx:].min(),
            'close': closes[i],
            'volume': volumes[bar_start_idx:].sum(),
            'dollar_volume': cumulative_dv,
            'n_ticks': i - bar_start_idx + 1,
        }
        bars.append(bar)
    
    if not bars:
        return pd.DataFrame()
    
    result = pd.DataFrame(bars)
    
    # Compute bar duration
    result['bar_duration_seconds'] = (
        (pd.to_datetime(result['bar_end']) - pd.to_datetime(result['bar_start']))
        .dt.total_seconds()
    )
    
    return result


# ---------------------------------------------------------------------------
# Volatility Bars
# ---------------------------------------------------------------------------

def compute_volatility_bars(
    intraday_df: pd.DataFrame,
    threshold: float,
    min_bars: int = 1,
) -> pd.DataFrame:
    """
    Construct volatility bars from 1-minute OHLCV data.
    
    A new bar is emitted when cumulative |log return| exceeds threshold.
    
    Args:
        intraday_df: DataFrame with columns [open, high, low, close, volume]
                     and DatetimeIndex
        threshold: Cumulative |return| threshold for bar emission
        min_bars: Minimum bars to return
        
    Returns:
        Same schema as compute_dollar_bars, plus:
        - cumulative_abs_return: total |return| in bar
    """
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame()
    
    # Compute log returns
    df = intraday_df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_return'] = df['log_return'].abs()
    df = df.iloc[1:]  # Drop first row (NaN return)
    
    if df.empty:
        return pd.DataFrame()
    
    bars = []
    cumulative_vol = 0.0
    bar_start_idx = 0
    
    timestamps = df.index.to_numpy()
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    volumes = df['volume'].to_numpy()
    abs_returns = df['abs_return'].to_numpy()
    
    # Compute dollar volume if available
    if 'dollar_volume' in df.columns:
        dollar_volumes = df['dollar_volume'].to_numpy()
    else:
        dollar_volumes = closes * volumes
    
    for i in range(len(df)):
        cumulative_vol += abs_returns[i]
        
        if cumulative_vol >= threshold:
            # Emit bar
            bar = {
                'bar_start': timestamps[bar_start_idx],
                'bar_end': timestamps[i],
                'open': opens[bar_start_idx],
                'high': highs[bar_start_idx:i+1].max(),
                'low': lows[bar_start_idx:i+1].min(),
                'close': closes[i],
                'volume': volumes[bar_start_idx:i+1].sum(),
                'dollar_volume': dollar_volumes[bar_start_idx:i+1].sum(),
                'cumulative_abs_return': cumulative_vol,
                'n_ticks': i - bar_start_idx + 1,
            }
            bars.append(bar)
            
            # Reset for next bar
            cumulative_vol = 0.0
            bar_start_idx = i + 1
    
    # Handle remaining data
    if bar_start_idx < len(df) and cumulative_vol > 0:
        i = len(df) - 1
        bar = {
            'bar_start': timestamps[bar_start_idx],
            'bar_end': timestamps[i],
            'open': opens[bar_start_idx],
            'high': highs[bar_start_idx:].max(),
            'low': lows[bar_start_idx:].min(),
            'close': closes[i],
            'volume': volumes[bar_start_idx:].sum(),
            'dollar_volume': dollar_volumes[bar_start_idx:].sum(),
            'cumulative_abs_return': cumulative_vol,
            'n_ticks': i - bar_start_idx + 1,
        }
        bars.append(bar)
    
    if not bars:
        return pd.DataFrame()
    
    result = pd.DataFrame(bars)
    
    # Compute bar duration
    result['bar_duration_seconds'] = (
        (pd.to_datetime(result['bar_end']) - pd.to_datetime(result['bar_start']))
        .dt.total_seconds()
    )
    
    return result


# ---------------------------------------------------------------------------
# Adaptive threshold computation
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(
    symbol: str,
    as_of_date: str,
    config: Optional[EventBarConfig] = None,
) -> Dict[str, float]:
    """
    Compute adaptive thresholds for dollar and volatility bars.
    
    Uses trailing 20-day median of daily statistics.
    
    Args:
        symbol: Stock symbol
        as_of_date: Reference date for trailing lookback
        config: EventBarConfig with multipliers
        
    Returns:
        Dict with 'dollar_threshold' and 'vol_threshold'
    """
    from src.data_sources.intraday_cache import compute_trailing_daily_stats
    
    if config is None:
        config = EventBarConfig()
    
    stats = compute_trailing_daily_stats(
        symbol=symbol,
        as_of_date=as_of_date,
        lookback_days=config.threshold_lookback_days,
    )
    
    if stats is None:
        logger.warning(f"Could not compute trailing stats for {symbol}")
        return {'dollar_threshold': None, 'vol_threshold': None}
    
    # Dollar bar threshold: fraction of daily dollar volume
    # Targeting ~50-100 bars per day (adjust k_dollar accordingly)
    # Default k_dollar=1.0 means threshold = median_daily_dv / expected_bars
    expected_bars_per_day = 50
    dollar_threshold = (
        stats['median_daily_dollar_volume'] / expected_bars_per_day
    ) * config.k_dollar
    
    # Volatility bar threshold: fraction of daily realized vol
    # Targeting ~30-80 bars per day
    expected_vol_bars_per_day = 40
    vol_threshold = (
        stats['median_daily_realized_vol'] / expected_vol_bars_per_day
    ) * config.k_vol
    
    return {
        'dollar_threshold': dollar_threshold,
        'vol_threshold': vol_threshold,
        'stats': stats,
    }


# ---------------------------------------------------------------------------
# Daily aggregation of event-time bars
# ---------------------------------------------------------------------------

def aggregate_event_bars_to_daily(
    dollar_bars: pd.DataFrame,
    vol_bars: pd.DataFrame,
    date: str,
) -> Dict[str, float]:
    """
    Aggregate intraday event-time bars into daily features.
    
    Args:
        dollar_bars: DataFrame from compute_dollar_bars for the day
        vol_bars: DataFrame from compute_volatility_bars for the day
        date: Date string (YYYY-MM-DD) for the day
        
    Returns:
        Dict of daily features with 'event_time_' prefix
    """
    features = {}
    
    # Dollar bar features
    if dollar_bars is not None and not dollar_bars.empty:
        n_dollar = len(dollar_bars)
        features['event_time_n_dollar_bars'] = n_dollar
        
        # Duration statistics
        durations = dollar_bars['bar_duration_seconds'].values
        features['event_time_dollar_avg_duration_sec'] = durations.mean()
        features['event_time_dollar_duration_std'] = durations.std() if len(durations) > 1 else 0.0
        features['event_time_dollar_duration_cv'] = (
            durations.std() / (durations.mean() + 1e-8) if len(durations) > 1 else 0.0
        )
        
        # Bar size statistics
        sizes = dollar_bars['dollar_volume'].values
        features['event_time_dollar_max_bar_size'] = sizes.max()
        features['event_time_dollar_bar_size_skew'] = (
            pd.Series(sizes).skew() if len(sizes) > 2 else 0.0
        )
        
        # Arrival rate (bars per hour of trading, assuming 6.5 hour day)
        features['event_time_dollar_arrival_rate'] = n_dollar / 6.5
        
        # N-ticks per bar (how many 1-min bars aggregated)
        n_ticks = dollar_bars['n_ticks'].values
        features['event_time_dollar_avg_ticks_per_bar'] = n_ticks.mean()
        
    else:
        # No dollar bars - fill with zeros/NaN
        features['event_time_n_dollar_bars'] = 0
        features['event_time_dollar_avg_duration_sec'] = np.nan
        features['event_time_dollar_duration_std'] = np.nan
        features['event_time_dollar_duration_cv'] = np.nan
        features['event_time_dollar_max_bar_size'] = np.nan
        features['event_time_dollar_bar_size_skew'] = np.nan
        features['event_time_dollar_arrival_rate'] = 0.0
        features['event_time_dollar_avg_ticks_per_bar'] = np.nan
    
    # Volatility bar features
    if vol_bars is not None and not vol_bars.empty:
        n_vol = len(vol_bars)
        features['event_time_n_vol_bars'] = n_vol
        
        # Duration statistics
        durations = vol_bars['bar_duration_seconds'].values
        features['event_time_vol_avg_duration_sec'] = durations.mean()
        features['event_time_vol_duration_std'] = durations.std() if len(durations) > 1 else 0.0
        
        # Arrival rate
        features['event_time_vol_arrival_rate'] = n_vol / 6.5
        
        # Cumulative return per bar
        if 'cumulative_abs_return' in vol_bars.columns:
            returns = vol_bars['cumulative_abs_return'].values
            features['event_time_vol_avg_abs_return'] = returns.mean()
        
    else:
        features['event_time_n_vol_bars'] = 0
        features['event_time_vol_avg_duration_sec'] = np.nan
        features['event_time_vol_duration_std'] = np.nan
        features['event_time_vol_arrival_rate'] = 0.0
        features['event_time_vol_avg_abs_return'] = np.nan
    
    # Cross-bar type features
    if (dollar_bars is not None and not dollar_bars.empty and 
        vol_bars is not None and not vol_bars.empty):
        
        # Ratio of dollar bars to vol bars (regime indicator)
        features['event_time_dollar_vs_vol_ratio'] = (
            len(dollar_bars) / (len(vol_bars) + 1e-8)
        )
        
        # Duration correlation (do they cluster together?)
        # Higher ratio = more volume-driven market
        # Lower ratio = more volatility-driven market
        
    else:
        features['event_time_dollar_vs_vol_ratio'] = np.nan
    
    # Add has_data flag
    features['event_time_has_data'] = float(
        features['event_time_n_dollar_bars'] > 0 or 
        features['event_time_n_vol_bars'] > 0
    )
    
    return features


# ---------------------------------------------------------------------------
# Full pipeline: intraday data → event bars → daily features
# ---------------------------------------------------------------------------

def generate_event_time_features_for_day(
    symbol: str,
    date: str,
    config: Optional[EventBarConfig] = None,
) -> Dict[str, float]:
    """
    Generate event-time bar features for a single day.
    
    Args:
        symbol: Stock symbol
        date: Date string (YYYY-MM-DD)
        config: EventBarConfig
        
    Returns:
        Dict of daily features with 'event_time_' prefix
    """
    from src.data_sources.intraday_cache import load_intraday_range
    
    if config is None:
        config = EventBarConfig()
    
    # Load intraday data for the day
    intraday = load_intraday_range(
        symbol=symbol,
        start_date=date,
        end_date=date,
        fetch_missing=True,
    )
    
    if intraday is None or intraday.empty:
        logger.debug(f"No intraday data for {symbol} on {date}")
        return {'event_time_has_data': 0.0}
    
    # Compute adaptive thresholds
    thresholds = compute_adaptive_thresholds(symbol, date, config)
    
    if thresholds['dollar_threshold'] is None:
        # Fallback: use simple percentile-based threshold
        daily_dv = intraday['dollar_volume'].sum()
        thresholds['dollar_threshold'] = daily_dv / 50  # Target ~50 bars
        thresholds['vol_threshold'] = 0.001  # 0.1% cumulative return
    
    # Generate dollar bars
    dollar_bars = compute_dollar_bars(
        intraday,
        threshold=thresholds['dollar_threshold'],
    )
    
    # Generate volatility bars
    vol_bars = compute_volatility_bars(
        intraday,
        threshold=thresholds['vol_threshold'],
    )
    
    # Aggregate to daily features
    features = aggregate_event_bars_to_daily(dollar_bars, vol_bars, date)
    
    # Add threshold info for debugging
    features['event_time_dollar_threshold'] = thresholds['dollar_threshold']
    features['event_time_vol_threshold'] = thresholds['vol_threshold']
    
    return features


def generate_event_time_features_range(
    symbol: str,
    start_date: str,
    end_date: str,
    config: Optional[EventBarConfig] = None,
) -> pd.DataFrame:
    """
    Generate event-time bar features for a date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: EventBarConfig
        
    Returns:
        DataFrame with DatetimeIndex and event_time_* columns
    """
    from src.data_sources.intraday_cache import load_intraday_range
    
    if config is None:
        config = EventBarConfig()
    
    # Load all intraday data for the range
    intraday = load_intraday_range(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetch_missing=True,
    )
    
    if intraday is None or intraday.empty:
        logger.warning(f"No intraday data for {symbol} in range {start_date} to {end_date}")
        return pd.DataFrame()
    
    # Get unique trading days
    if 'day_id' in intraday.columns:
        days = intraday['day_id'].unique()
        days = [str(d)[:4] + '-' + str(d)[4:6] + '-' + str(d)[6:8] for d in sorted(days)]
    else:
        days = intraday.index.normalize().unique().strftime('%Y-%m-%d').tolist()
    
    logger.info(f"Generating event-time features for {symbol}: {len(days)} days")
    
    all_features = []
    
    for date in days:
        # Filter to this day's data
        day_data = intraday[intraday.index.date == pd.Timestamp(date).date()]
        
        if day_data.empty:
            continue
        
        # Compute thresholds (use full range stats for stability)
        # For efficiency, compute once and reuse
        daily_dv = day_data['dollar_volume'].sum()
        dollar_threshold = max(daily_dv / 50, 1000)  # At least $1000 per bar
        
        # Compute returns for vol threshold
        returns = np.log(day_data['close'] / day_data['close'].shift(1)).dropna()
        daily_vol = returns.abs().sum()
        vol_threshold = max(daily_vol / 40, 0.0005)  # At least 0.05%
        
        # Generate bars
        dollar_bars = compute_dollar_bars(day_data, dollar_threshold)
        vol_bars = compute_volatility_bars(day_data, vol_threshold)
        
        # Aggregate to features
        features = aggregate_event_bars_to_daily(dollar_bars, vol_bars, date)
        features['date'] = date
        features['event_time_dollar_threshold'] = dollar_threshold
        features['event_time_vol_threshold'] = vol_threshold
        
        all_features.append(features)
    
    if not all_features:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_features)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    logger.info(f"✅ Generated {len(df)} days of event-time features for {symbol}")
    return df
