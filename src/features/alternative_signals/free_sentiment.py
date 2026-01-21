"""
Free Sentiment Module
======================

Extracts 4 sentiment features from free public data sources.
Changes and z-scores are more predictive than absolute counts (hedge-fund discipline).

Features:
1. google_trends_score - Google search interest (0-100)
2. news_volume_count - Daily news article count
3. news_volume_z - Z-scored news volume (20-day, regime detector)
4. news_volume_change - 5-day change in news volume (spike detector)

Note: Requires API integration. Returns placeholder zeros for now.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
import time
import random
import os

logger = logging.getLogger(__name__)


def extract_free_sentiment(
    symbol: str,
    date_range: pd.DatetimeIndex,
    eodhd_provider: Optional[object] = None
) -> Dict[str, pd.Series]:
    """
    Extract sentiment features from free APIs.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    date_range : pd.DatetimeIndex
        Date range to extract features for
    eodhd_provider : object, optional
        EODHD provider instance for news API
        
    Returns
    -------
    dict
        Dictionary of feature_name -> pd.Series
        
    Notes
    -----
    - Google Trends: Requires pytrends library (pip install pytrends)
    - News: Uses EODHD news endpoint (included with EODHD subscription)
    
    Features may return NaN if API is unavailable or rate limited.
    Implements caching to minimize API calls.
    """
    features = {}
    
    if len(date_range) == 0:
        return {
            'google_trends_score': pd.Series(dtype=float),
            'news_volume_count': pd.Series(dtype=float),
            'news_volume_z': pd.Series(dtype=float),
            'news_volume_change': pd.Series(dtype=float)
        }
    
    # 1. Google Trends Score (100% free, no credentials)
    features['google_trends_score'] = _get_google_trends(symbol, date_range)
    
    # 2. News Volume Count (EODHD API)
    if eodhd_provider is not None:
        news_count = _get_news_volume(symbol, date_range, eodhd_provider)
    else:
        news_count = pd.Series(0.0, index=date_range)
    
    features['news_volume_count'] = news_count
    
    # 3. News Volume Z-score (changes are regime indicators)
    # Z-score over 20-day window
    news_mean = news_count.rolling(20).mean()
    news_std = news_count.rolling(20).std()
    features['news_volume_z'] = (news_count - news_mean) / news_std.replace(0, 1)
    
    # 4. News Volume Change (5-day)
    # Spikes indicate regime shifts or events
    features['news_volume_change'] = news_count.diff(5)
    
    return features


def _get_google_trends(symbol: str, date_range: pd.DatetimeIndex) -> pd.Series:
    """
    Get Google Trends score using pytrends.
    
    Returns weekly search interest (0-100) for the ticker/company name.
    100% free API - no credentials required.
    """

    def _cache_path() -> Path:
        # Repo root = .../src/features/alternative_signals/free_sentiment.py -> parents[3]
        root = Path(__file__).resolve().parents[3]
        cache_dir = root / 'data_cache' / 'google_trends'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{symbol.lower()}.parquet"

    def _read_cache() -> Optional[pd.Series]:
        try:
            path = _cache_path()
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            if df is None or df.empty:
                return None
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date']).set_index('date').sort_index()
            if 'score' not in df.columns:
                return None
            s = df['score']
            s.index = pd.to_datetime(s.index).normalize()
            # Convert to daily and ffill to requested range.
            calendar = pd.date_range(start=date_range.min(), end=date_range.max(), freq='D')
            s = s.reindex(calendar).ffill()
            return s.reindex(date_range, method='ffill')
        except Exception:
            return None

    def _write_cache(series: pd.Series) -> None:
        try:
            path = _cache_path()
            series = pd.Series(series, copy=True)
            series.index = pd.to_datetime(series.index).normalize()
            df_new = series.to_frame('score')
            df_new.insert(0, 'date', df_new.index)

            if path.exists():
                try:
                    df_old = pd.read_parquet(path)
                    if df_old is not None and not df_old.empty:
                        if 'date' in df_old.columns:
                            df_old['date'] = pd.to_datetime(df_old['date'], errors='coerce')
                            df_old = df_old.dropna(subset=['date']).set_index('date').sort_index()
                        if 'score' in df_old.columns:
                            s_old = df_old['score']
                            s_old.index = pd.to_datetime(s_old.index).normalize()
                            s_merged = s_old.combine_first(series)
                            df_new = s_merged.to_frame('score')
                            df_new.insert(0, 'date', df_new.index)
                except Exception:
                    pass

            df_new.to_parquet(path, index=False)
        except Exception:
            return

    try:
        # pytrends calls urllib3 Retry(method_whitelist=...), which was removed in urllib3>=2.
        # Provide a local compatibility shim so Google Trends can work without pinning urllib3.
        try:
            import inspect
            from urllib3.util.retry import Retry

            if (
                'method_whitelist' not in inspect.signature(Retry.__init__).parameters
                and not getattr(Retry.__init__, '_accepts_method_whitelist', False)
            ):
                _orig_retry_init = Retry.__init__

                def _patched_retry_init(self, *args, method_whitelist=None, allowed_methods=None, **kwargs):
                    if allowed_methods is None and method_whitelist is not None:
                        allowed_methods = method_whitelist
                    return _orig_retry_init(self, *args, allowed_methods=allowed_methods, **kwargs)

                _patched_retry_init._accepts_method_whitelist = True  # type: ignore[attr-defined]
                Retry.__init__ = _patched_retry_init  # type: ignore[assignment]
        except Exception:
            pass

        from pytrends.request import TrendReq

        def _parse_bool_env(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            raw = raw.strip().lower()
            if raw in {'1', 'true', 't', 'yes', 'y', 'on'}:
                return True
            if raw in {'0', 'false', 'f', 'no', 'n', 'off'}:
                return False
            return default

        def _parse_float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except Exception:
                return default

        def _parse_int_env(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except Exception:
                return default

        def _parse_proxies_env(name: str) -> Optional[list]:
            raw = os.getenv(name)
            if raw is None:
                return None
            parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
            proxies = [p for p in parts if p.lower().startswith('https://')]
            return proxies or None

        def _make_pytrends() -> TrendReq:
            """Create a TrendReq client with safe defaults and optional proxy config.

            Env vars (all optional):
            - GOOGLE_TRENDS_PROXIES: comma/semicolon-separated HTTPS proxy URLs (must include port)
            - GOOGLE_TRENDS_TIMEOUT_CONNECT / GOOGLE_TRENDS_TIMEOUT_READ: seconds
            - GOOGLE_TRENDS_RETRIES: int
            - GOOGLE_TRENDS_BACKOFF_FACTOR: float
            - GOOGLE_TRENDS_VERIFY_SSL: true/false (defaults to true)
            """
            proxies = _parse_proxies_env('GOOGLE_TRENDS_PROXIES')
            timeout_connect = _parse_float_env('GOOGLE_TRENDS_TIMEOUT_CONNECT', 10.0)
            timeout_read = _parse_float_env('GOOGLE_TRENDS_TIMEOUT_READ', 25.0)
            retries = _parse_int_env('GOOGLE_TRENDS_RETRIES', 2)
            backoff_factor = _parse_float_env('GOOGLE_TRENDS_BACKOFF_FACTOR', 0.5)
            verify_ssl = _parse_bool_env('GOOGLE_TRENDS_VERIFY_SSL', True)

            kwargs = {
                'hl': 'en-US',
                'tz': 360,
                'timeout': (timeout_connect, timeout_read),
                'retries': retries,
                'backoff_factor': backoff_factor,
            }

            if proxies is not None:
                kwargs['proxies'] = proxies

            # Only pass requests_args when we need to override defaults.
            if not verify_ssl:
                kwargs['requests_args'] = {'verify': False}

            # Some pytrends versions reject some kwargs; progressively fall back.
            try:
                return TrendReq(**kwargs)
            except TypeError:
                for key in ['requests_args', 'backoff_factor', 'retries', 'proxies', 'timeout']:
                    kwargs.pop(key, None)
                    try:
                        return TrendReq(**kwargs)
                    except TypeError:
                        continue
                return TrendReq(hl='en-US', tz=360)
        
        # Map ticker to company name for better results
        ticker_to_company = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix',
            # Add more as needed
        }
        
        search_term = ticker_to_company.get(symbol, symbol)
        
        # Try cache first (reduces 429s and makes runs deterministic once warm)
        cached = _read_cache()
        if cached is not None and not cached.empty:
            non_null = int(cached.notna().sum())
            if non_null >= int(max(5, 0.5 * len(date_range))):
                logger.info(f"✅ Google Trends (cache): {symbol} -> {non_null}/{len(date_range)} days")
                return cached

        pytrends = _make_pytrends()
        
        # Build payload for the symbol
        start_date = date_range.min().strftime('%Y-%m-%d')
        end_date = date_range.max().strftime('%Y-%m-%d')
        timeframe = f'{start_date} {end_date}'
        
        # Gentle jitter to avoid synchronized bursts in parallel runs.
        time.sleep(random.uniform(0.25, 1.0))

        pytrends.build_payload([search_term], timeframe=timeframe)
        trends_df = pytrends.interest_over_time()
        
        if not trends_df.empty and search_term in trends_df.columns:
            # Resample to daily and forward-fill
            trends_series = trends_df[search_term].resample('D').ffill()
            result = trends_series.reindex(date_range, method='ffill')
            # Persist cache for future runs.
            _write_cache(trends_series)
            logger.info(f"✅ Google Trends: {symbol} -> {len(result.dropna())}/{len(result)} days")
            return result
        else:
            logger.warning(f"Google Trends: No data for {symbol}/{search_term}")
            cached = _read_cache()
            return cached if cached is not None else pd.Series(np.nan, index=date_range)
            
    except ImportError:
        logger.warning("pytrends not installed. Install with: pip install pytrends")
        cached = _read_cache()
        return cached if cached is not None else pd.Series(np.nan, index=date_range)
    except Exception as e:
        logger.warning(f"Google Trends error for {symbol}: {e}")
        cached = _read_cache()
        return cached if cached is not None else pd.Series(np.nan, index=date_range)


def _get_news_volume(
    symbol: str,
    date_range: pd.DatetimeIndex,
    eodhd_provider: object
) -> pd.Series:
    """
    Get news article count from EODHD news API.
    
    News volume is a strong predictor of volatility.
    High news volume often precedes large price moves.
    """
    try:
        # Initialize with zeros (no news = 0 articles, not NaN)
        # Normalize to date (midnight) to make membership/indexing robust across tz-aware payloads.
        date_index = pd.DatetimeIndex(pd.to_datetime(date_range).normalize())
        news_counts = pd.Series(0, index=date_index)
        
        # EODHD news endpoint
        start_date = date_range.min().strftime('%Y-%m-%d')
        end_date = date_range.max().strftime('%Y-%m-%d')
        
        # Fetch news for the date range
        news_data = eodhd_provider.get_news(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=1000  # Get up to 1000 articles
        )
        
        if news_data:
            # Count articles per day
            for article in news_data:
                if 'date' in article:
                    ts = pd.to_datetime(article.get('date'), errors='coerce', utc=True)
                    if pd.isna(ts):
                        continue
                    article_date = ts.tz_convert(None).normalize()
                    if article_date in news_counts.index:
                        news_counts.loc[article_date] += 1
            
            non_zero = news_counts[news_counts > 0]
            logger.info(f"✅ News Volume: {symbol} -> {len(non_zero)} days with news, total {news_counts.sum()} articles")
        else:
            logger.warning(f"News Volume: No data for {symbol}")
        
        return news_counts
        
    except Exception as e:
        logger.warning(f"News volume error for {symbol}: {e}")
        return pd.Series(0, index=date_range)  # Return zeros, not NaN

