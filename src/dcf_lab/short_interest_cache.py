"""
Short Interest Historical Cache

Stores short interest snapshots locally to build historical time series.
Over time, this cache will accumulate enough data points for zscore calculations.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class ShortInterestCache:
    """
    Local cache for short interest data
    
    Accumulates historical snapshots to enable:
    - 1-year zscore calculations (needs 12+ data points)
    - 3-month accurate change (vs estimated)
    - Trend analysis
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory for cache files (default: from cache_paths)
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Use centralized cache paths (with legacy fallback)
            try:
                from src.cache_paths import resolve_short_interest_cache_dir
                self.cache_dir = resolve_short_interest_cache_dir()
            except ImportError:
                self.cache_dir = Path("data/short_interest_cache")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_file(self, ticker: str) -> Path:
        """Get cache file path for a ticker"""
        return self.cache_dir / f"{ticker.upper()}_short_history.json"
    
    def save_snapshot(self, ticker: str, data: Dict) -> None:
        """
        Save a short interest snapshot
        
        Args:
            ticker: Stock ticker
            data: Short interest data dict with keys:
                - date: Settlement date (YYYY-MM-DD or timestamp)
                - shares_short: Total shares short
                - short_pct_float: Short % of float
                - days_to_cover: Days to cover
                - source: Data source (EODHD, yfinance, etc.)
        """
        try:
            cache_file = self._get_cache_file(ticker)
            
            # Load existing cache
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Normalize date
            date = data.get('date')
            if isinstance(date, (int, float)):
                # Unix timestamp
                date = datetime.fromtimestamp(date).strftime('%Y-%m-%d')
            elif isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            
            # Create snapshot
            snapshot = {
                'date': date,
                'shares_short': data.get('shares_short'),
                'short_pct_float': data.get('short_pct_float'),
                'days_to_cover': data.get('days_to_cover'),
                'source': data.get('source', 'UNKNOWN'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if already exists (avoid duplicates)
            existing_dates = {s['date'] for s in history}
            if date not in existing_dates:
                history.append(snapshot)
                
                # Sort by date (newest first)
                history.sort(key=lambda x: x['date'], reverse=True)
                
                # Save back to file
                with open(cache_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                logger.debug(f"✅ Cached short interest for {ticker} ({date}) - total: {len(history)} snapshots")
            else:
                logger.debug(f"⏭️  Snapshot for {ticker} ({date}) already cached")
                
        except Exception as e:
            logger.warning(f"Failed to cache short interest for {ticker}: {e}")
    
    def get_history(self, ticker: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """
        Get historical short interest from cache
        
        Args:
            ticker: Stock ticker
            lookback_days: How many days back to retrieve
            
        Returns:
            DataFrame with columns: date, shares_short, short_pct_float, days_to_cover
            Sorted by date (newest first)
        """
        try:
            cache_file = self._get_cache_file(ticker)
            
            if not cache_file.exists():
                logger.debug(f"No cache file for {ticker}")
                return None
            
            with open(cache_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by lookback
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            df = df[df['date'] >= cutoff]
            
            # Sort by date (newest first)
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.debug(f"✅ Retrieved {len(df)} cached short interest snapshots for {ticker}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cache for {ticker}: {e}")
            return None
    
    def get_count(self, ticker: str) -> int:
        """Get number of cached snapshots for a ticker"""
        try:
            cache_file = self._get_cache_file(ticker)
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    history = json.load(f)
                return len(history)
            return 0
        except:
            return 0


def get_short_interest_cache() -> ShortInterestCache:
    """Get singleton cache instance"""
    return ShortInterestCache()
