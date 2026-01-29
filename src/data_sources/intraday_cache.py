"""
Intraday Data Cache Layer for Event-Time Bars

Smart caching for 1-minute EODHD data with monthly chunks.
Supports incremental updates and efficient range queries.

Cache Structure:
    cache/data_sources/eodhd_intraday/{SYMBOL}/
    ├── 2024-01.parquet  (monthly chunks)
    ├── 2024-02.parquet
    └── ...

Each monthly file contains:
    - timestamp: minute-level datetime index
    - open, high, low, close, volume: OHLCV data
    - day_id: YYYYMMDD integer for daily aggregation
    - dollar_volume: close * volume (precomputed)
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _get_intraday_cache_dir() -> Path:
    """Return the intraday cache directory."""
    from src.cache_paths import EODHD_CACHE_ROOT
    cache_dir = EODHD_CACHE_ROOT / "intraday"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_symbol_cache_dir(symbol: str) -> Path:
    """Return the cache directory for a specific symbol."""
    sym_dir = _get_intraday_cache_dir() / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    return sym_dir


def _get_monthly_cache_path(symbol: str, year: int, month: int) -> Path:
    """Return the path to a monthly cache file."""
    return _get_symbol_cache_dir(symbol) / f"{year:04d}-{month:02d}.parquet"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_monthly_cache(symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
    """Load a monthly cache file if it exists."""
    path = _get_monthly_cache_path(symbol, year, month)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if 'timestamp' in df.columns and df.index.name != 'timestamp':
            df = df.set_index('timestamp')
        return df
    except Exception as e:
        logger.warning(f"Failed to load intraday cache {path}: {e}")
        return None


def _save_monthly_cache(symbol: str, year: int, month: int, df: pd.DataFrame) -> bool:
    """Save a monthly cache file."""
    path = _get_monthly_cache_path(symbol, year, month)
    try:
        # Ensure index is named
        if df.index.name != 'timestamp':
            df.index.name = 'timestamp'
        df.to_parquet(path, index=True)
        logger.debug(f"Saved intraday cache: {path} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.warning(f"Failed to save intraday cache {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Month range utilities
# ---------------------------------------------------------------------------

def _get_months_in_range(start_date: str, end_date: str) -> List[Tuple[int, int]]:
    """Return list of (year, month) tuples covering the date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return months


def _get_days_in_month(year: int, month: int) -> List[str]:
    """Return list of trading day strings (YYYY-MM-DD) in a month."""
    import calendar
    _, last_day = calendar.monthrange(year, month)
    days = []
    for day in range(1, last_day + 1):
        dt = datetime(year, month, day)
        # Skip weekends
        if dt.weekday() < 5:  # Mon-Fri
            days.append(dt.strftime("%Y-%m-%d"))
    return days


# ---------------------------------------------------------------------------
# EODHD API fetching
# ---------------------------------------------------------------------------

def _fetch_intraday_from_api(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """Fetch intraday data from EODHD API."""
    try:
        from src.data_sources.eodhd_provider import get_eodhd_provider
        
        provider = get_eodhd_provider()
        df = provider.get_intraday_prices(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
        
        if df is None or df.empty:
            return None
        
        # Standardize columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })
        
        # Keep only OHLCV
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        
        # Add derived columns
        df['dollar_volume'] = df['close'] * df['volume']
        df['day_id'] = df.index.strftime('%Y%m%d').astype(int)
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to fetch intraday data for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Smart cache loading with gap filling
# ---------------------------------------------------------------------------

def load_intraday_range(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1m",
    fetch_missing: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load intraday data for a date range, using cache and fetching missing data.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Intraday interval ('1m', '5m', '1h')
        fetch_missing: If True, fetch missing data from API
        
    Returns:
        DataFrame with 1-minute OHLCV data, or None if no data available
    """
    symbol = symbol.upper()
    months = _get_months_in_range(start_date, end_date)
    
    all_dfs = []
    months_to_update = []
    
    for year, month in months:
        cached = _load_monthly_cache(symbol, year, month)
        if cached is not None and not cached.empty:
            all_dfs.append(cached)
        else:
            months_to_update.append((year, month))
    
    # Fetch missing months
    if fetch_missing and months_to_update:
        logger.info(f"Fetching {len(months_to_update)} missing months for {symbol}")
        
        for year, month in months_to_update:
            # Compute month boundaries as date strings
            month_start_str = f"{year:04d}-{month:02d}-01"
            if month == 12:
                month_end_str = f"{year + 1:04d}-01-01"
            else:
                month_end_str = f"{year:04d}-{month + 1:02d}-01"
            
            # Convert to timestamps for comparison
            month_start_ts = pd.Timestamp(month_start_str)
            month_end_ts = pd.Timestamp(month_end_str)
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            # Adjust to actual range
            actual_start = max(month_start_ts, start_ts).strftime('%Y-%m-%d')
            actual_end = min(month_end_ts, end_ts).strftime('%Y-%m-%d')
            
            fetched = _fetch_intraday_from_api(symbol, actual_start, actual_end, interval)
            
            if fetched is not None and not fetched.empty:
                # Merge with any existing partial data
                existing = _load_monthly_cache(symbol, year, month)
                if existing is not None and not existing.empty:
                    # Combine and deduplicate
                    combined = pd.concat([existing, fetched])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                    fetched = combined
                
                _save_monthly_cache(symbol, year, month, fetched)
                all_dfs.append(fetched)
    
    if not all_dfs:
        return None
    
    # Combine all months
    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    
    # Filter to exact date range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    
    if df.empty:
        return None
    
    logger.info(f"✅ Loaded {len(df)} intraday bars for {symbol} ({start_date} to {end_date})")
    return df


# ---------------------------------------------------------------------------
# Cache inspection utilities
# ---------------------------------------------------------------------------

def get_cached_months(symbol: str) -> List[Tuple[int, int]]:
    """Return list of (year, month) tuples that are cached for a symbol."""
    sym_dir = _get_symbol_cache_dir(symbol)
    if not sym_dir.exists():
        return []
    
    months = []
    for path in sym_dir.glob("*.parquet"):
        try:
            name = path.stem  # e.g., "2024-01"
            year, month = name.split("-")
            months.append((int(year), int(month)))
        except:
            continue
    
    return sorted(months)


def get_cache_coverage(symbol: str) -> Optional[Tuple[str, str, int]]:
    """
    Return cache coverage for a symbol.
    
    Returns:
        Tuple of (earliest_date, latest_date, total_bars) or None if no cache
    """
    months = get_cached_months(symbol)
    if not months:
        return None
    
    all_dfs = []
    for year, month in months:
        df = _load_monthly_cache(symbol, year, month)
        if df is not None and not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    combined = pd.concat(all_dfs)
    return (
        combined.index.min().strftime("%Y-%m-%d"),
        combined.index.max().strftime("%Y-%m-%d"),
        len(combined),
    )


def clear_symbol_cache(symbol: str) -> int:
    """Clear all cached intraday data for a symbol. Returns number of files deleted."""
    sym_dir = _get_symbol_cache_dir(symbol)
    if not sym_dir.exists():
        return 0
    
    count = 0
    for path in sym_dir.glob("*.parquet"):
        path.unlink()
        count += 1
    
    return count


# ---------------------------------------------------------------------------
# Trailing statistics for threshold computation
# ---------------------------------------------------------------------------

def compute_trailing_daily_stats(
    symbol: str,
    as_of_date: str,
    lookback_days: int = 20,
) -> Optional[dict]:
    """
    Compute trailing daily statistics for event-bar threshold computation.
    
    Returns:
        Dict with:
        - median_daily_dollar_volume
        - median_daily_realized_vol
        - median_daily_bar_count
    """
    end_date = pd.Timestamp(as_of_date)
    start_date = end_date - pd.Timedelta(days=lookback_days * 2)  # Extra buffer for weekends
    
    df = load_intraday_range(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        fetch_missing=True,
    )
    
    if df is None or df.empty:
        return None
    
    # Group by day
    daily = df.groupby('day_id').agg({
        'dollar_volume': 'sum',
        'close': ['first', 'last', 'max', 'min'],
        'volume': 'sum',
    })
    
    # Flatten columns
    daily.columns = ['_'.join([str(c) for c in col]).strip('_') for col in daily.columns]
    
    # Compute realized vol (high-low range proxy)
    daily['realized_vol'] = (daily['close_max'] - daily['close_min']) / daily['close_first']
    
    # Take last N trading days
    daily = daily.tail(lookback_days)
    
    if len(daily) < 5:
        logger.warning(f"Insufficient data for trailing stats: {len(daily)} days")
        return None
    
    return {
        'median_daily_dollar_volume': daily['dollar_volume_sum'].median(),
        'median_daily_realized_vol': daily['realized_vol'].median(),
        'median_daily_bar_count': len(df) / len(daily),  # Average bars per day
        'lookback_days': len(daily),
    }
