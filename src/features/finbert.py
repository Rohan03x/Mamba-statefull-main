import json
import os
import glob
import logging
from typing import Optional
from pathlib import Path

import pandas as pd

# Load environment variables for subprocess workers
try:
    from dotenv import load_dotenv
    _repo_root = Path(__file__).resolve().parents[2]  # Go up from src/features/ to repo root
    _env_path = _repo_root / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    pass  # dotenv not available or .env missing

logger = logging.getLogger(__name__)

try:
    from src.dcf_lab.finbert_sentiment import analyze_finbert_sentiment  # type: ignore
    logger.info("✅ FinBERT transformer loaded successfully")
except Exception as e:
    analyze_finbert_sentiment = None  # type: ignore
    import sys
    sys.stderr.write(f"❌ FinBERT transformer import failed: {e}\n")
    sys.stderr.flush()

try:
    from src.dcf_lab.gdelt_auto_fetcher import GDELTAutoFetcher  # type: ignore
    _GDELT_AUTO_AVAIL = True
except Exception:
    GDELTAutoFetcher = None  # type: ignore
    _GDELT_AUTO_AVAIL = False

try:
    # Optional: defeatbeta-api as a robust news source.
    # IMPORTANT: keep this lazy, defeatbeta-api prints banners and may download NLTK assets at import time.
    Ticker = None  # type: ignore
    _DEFEATBETA_AVAIL = None
except Exception:  # pragma: no cover
    Ticker = None  # type: ignore
    _DEFEATBETA_AVAIL = None


def _lazy_import_defeatbeta() -> bool:
    global Ticker, _DEFEATBETA_AVAIL
    if _DEFEATBETA_AVAIL is not None:
        return bool(_DEFEATBETA_AVAIL)
    try:
        from defeatbeta_api.data.ticker import Ticker as _Ticker  # type: ignore

        Ticker = _Ticker  # type: ignore
        _DEFEATBETA_AVAIL = True
    except Exception:  # pragma: no cover
        Ticker = None  # type: ignore
        _DEFEATBETA_AVAIL = False
    return bool(_DEFEATBETA_AVAIL)

gdelt = None  # type: ignore
_GDELT_AVAIL = None


def _lazy_import_gdelt() -> bool:
    global gdelt, _GDELT_AVAIL
    if _GDELT_AVAIL is not None:
        return bool(_GDELT_AVAIL)
    try:
        import gdelt as _gdelt  # type: ignore

        gdelt = _gdelt  # type: ignore
        _GDELT_AVAIL = True
    except Exception:  # pragma: no cover
        gdelt = None  # type: ignore
        _GDELT_AVAIL = False
    return bool(_GDELT_AVAIL)

try:
    from google.cloud import bigquery
    # NOW OPTIMIZED: Using partitioned tables to minimize data scanned
    # Previous issue: CONTAINS_SUBSTR on full table = 200 TiB scan
    # New approach: _PARTITIONTIME filter + REGEXP_CONTAINS = ~GB scan per query
    _BIGQUERY_AVAIL = True  # Re-enabled with optimized queries
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore
    _BIGQUERY_AVAIL = False

# StockNewsAPI removed - using BigQuery GDELT as primary source
_STOCKNEWS_AVAIL = False
STOCKNEWS_API_KEY = ''

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore

DATA_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_cache')
NY_TZ = 'America/New_York'
os.makedirs(DATA_CACHE, exist_ok=True)


def _load_news_from_cache(symbol: str) -> Optional[pd.DataFrame]:
    # PRIORITY 1: Check for unified GDELT cache (parquet format)
    # Try multiple possible locations: cache/gdelt, data/cache/gdelt, data_cache/gdelt
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    possible_paths = [
        os.path.join(repo_root, 'cache', 'gdelt', f"{symbol.upper()}_gdelt_unified.parquet"),
        os.path.join('cache', 'gdelt', f"{symbol.upper()}_gdelt_unified.parquet"),
        os.path.join('data', 'cache', 'gdelt', f"{symbol.upper()}_gdelt_unified.parquet"),
        os.path.join(DATA_CACHE, 'gdelt', f"{symbol.upper()}_gdelt_unified.parquet"),
    ]
    
    for unified_cache in possible_paths:
        if os.path.exists(unified_cache):
            try:
                print(f"📦 Loading from unified GDELT cache: {unified_cache}")
                df = pd.read_parquet(unified_cache)
                df.index = pd.to_datetime(df.index)
                print(f"   Loaded {len(df):,} days ({df.index.min().date()} to {df.index.max().date()})")
                return df
            except Exception as e:
                print(f"⚠️  Unified cache load failed: {e}")
    
    # PRIORITY 2: Check for GDELT 1.0 cache (parquet format)
    possible_paths = [
        os.path.join(repo_root, 'cache', 'gdelt', f"{symbol.upper()}_gdelt1_news.parquet"),
        os.path.join('cache', 'gdelt', f"{symbol.upper()}_gdelt1_news.parquet"),
        os.path.join('data', 'cache', 'gdelt', f"{symbol.upper()}_gdelt1_news.parquet"),
        os.path.join(DATA_CACHE, 'gdelt', f"{symbol.upper()}_gdelt1_news.parquet"),
    ]
    
    for gdelt1_cache in possible_paths:
        if os.path.exists(gdelt1_cache):
            try:
                print(f"📦 Loading from GDELT 1.0 cache: {gdelt1_cache}")
                df = pd.read_parquet(gdelt1_cache)
                df.index = pd.to_datetime(df.index)
                print(f"   Loaded {len(df):,} days ({df.index.min().date()} to {df.index.max().date()})")
                return df
            except Exception as e:
                print(f"⚠️  GDELT 1.0 cache load failed: {e}")
    
    # PRIORITY 3: Legacy company-news JSON cache
    pattern = os.path.join(DATA_CACHE, f"company-news_news_{symbol.upper()}_*.json")
    candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
    if not candidates:
        return None
    try:
        with open(candidates[0], 'r', encoding='utf-8') as f:
            items = json.load(f)
        rows = []
        for it in items:
            # Support common timestamp keys from various caches/providers
            raw = (
                it.get('date')
                or it.get('publishedAt')
                or it.get('datetime')
                or it.get('providerPublishTime')
                or it.get('time')
                or it.get('timestamp')
            )
            if isinstance(raw, (int, float)):
                dt = pd.to_datetime(raw, unit='s', utc=True, errors='coerce')
            else:
                dt = pd.to_datetime(raw, utc=True, errors='coerce')
            if pd.isna(dt):
                continue
            rows.append({'date': dt.tz_convert(NY_TZ).date(), 'title': it.get('title') or it.get('headline') or ''})
        if not rows:
            return None
        df = pd.DataFrame(rows).dropna()
        df = df.groupby('date')['title'].apply(lambda s: '. '.join([str(x) for x in s if x])).to_frame('headline')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _download_news_yf(symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
    """Best-effort lightweight news fetch via yfinance Ticker.news (limited)."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        rows = []
        for it in news:
            ts = it.get('providerPublishTime') or it.get('published_at') or it.get('time')
            if ts is None:
                continue
            dt = pd.to_datetime(ts, unit='s', utc=True, errors='coerce') if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True, errors='coerce')
            if pd.isna(dt):
                continue
            rows.append({'date': dt.tz_convert(NY_TZ).date(), 'title': it.get('title') or ''})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        cutoff = pd.Timestamp.utcnow().tz_convert(NY_TZ).normalize() - pd.Timedelta(days=days)
        df = df[pd.to_datetime(df['date']) >= cutoff]
        df = df.groupby('date')['title'].apply(lambda s: '. '.join([str(x) for x in s if x])).to_frame('headline')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def _download_news_defeatbeta(symbol: str, days: int = 14) -> Optional[pd.DataFrame]:
    """Download news from defeatbeta-api for historical coverage."""
    if not _lazy_import_defeatbeta():
        return None
    try:
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = Ticker(symbol)
        news_df = ticker.news(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if news_df is None or news_df.empty:
            return None
        
        # Convert to expected format
        result_rows = []
        for idx, row in news_df.iterrows():
            dt = pd.to_datetime(idx) if not isinstance(idx, pd.Timestamp) else idx
            
            headline = row.get('headline', '') or row.get('title', '')
            summary = row.get('summary', '') or row.get('description', '')
            combined = f"{headline}. {summary}".strip() if summary else headline.strip()
            
            if combined:
                result_rows.append({
                    'date': dt.date() if hasattr(dt, 'date') else dt,
                    'headline': combined
                })
        
        if not result_rows:
            return None
        
        df = pd.DataFrame(result_rows)
        df.index = pd.to_datetime(df['date'])
        return df
        
    except Exception as e:
        return None
    """Attempt to coerce various defeatbeta news object shapes into a DataFrame."""
    df = None
    for attr in ('to_pandas', 'to_dataframe', 'df', 'data'):
        obj = getattr(news_obj, attr, None)
        if callable(obj):
            try:
                df = obj()
                break
            except Exception:
                continue
        elif isinstance(obj, pd.DataFrame):
            df = obj
            break
    if df is None and isinstance(news_obj, pd.DataFrame):
        df = news_obj
    return df


def _normalize_headlines_df(df: pd.DataFrame, days: int) -> Optional[pd.DataFrame]:
    """Normalize arbitrary news DataFrame to daily headline aggregation within the last N days."""
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    title_col = pick('title', 'headline')
    time_col = pick(
        'publishedat',
        'publish_time',
        'report_date',
        'datetime',
        'providerpublishtime',
        'date',
        'time',
        'timestamp',
    )
    if not title_col or not time_col:
        return None
    sub = df[[title_col, time_col]].copy()
    sub.rename(columns={title_col: 'title', time_col: 'when'}, inplace=True)
    # Parse to tz-aware timestamps
    tvals = []
    for val in sub['when'].tolist():
        if isinstance(val, (int, float)):
            tvals.append(pd.to_datetime(val, unit='s', utc=True, errors='coerce'))
        else:
            tvals.append(pd.to_datetime(val, utc=True, errors='coerce'))
    sub['ts'] = tvals
    sub = sub.dropna(subset=['ts'])
    # Compute cutoff date in NY timezone and filter by date
    cutoff_date = (pd.Timestamp.utcnow().tz_convert(NY_TZ).normalize() - pd.Timedelta(days=days)).date()
    sub['date'] = sub['ts'].dt.tz_convert(NY_TZ).dt.date
    sub = sub[sub['date'] >= cutoff_date]
    if sub.empty:
        return None
    daily = sub.groupby('date')['title'].apply(lambda s: '. '.join([str(x) for x in s if x])).to_frame('headline')
    daily.index = pd.to_datetime(daily.index)
    return daily


def _download_news_gdelt_bigquery(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch historical news via GDELT BigQuery (fast, no multiprocessing issues)."""
    if not _BIGQUERY_AVAIL:
        return None
    
    # Check cache first
    cache_dir = Path("data/cache/gdelt")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.upper()}_gdelt_news.parquet"
    
    # Parse dates
    if start and end:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
    else:
        end_dt = pd.Timestamp.utcnow()
        start_dt = end_dt - pd.Timedelta(days=365)
    
    # Try loading from cache
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            cached_df.index = pd.to_datetime(cached_df.index)
            
            # Filter by requested date range
            mask = (cached_df.index >= start_dt.date()) & (cached_df.index <= end_dt.date())
            filtered_df = cached_df[mask]
            
            if not filtered_df.empty:
                print(f"🎯 CACHE HIT: Loaded {len(filtered_df)} days of GDELT news from cache")
                return filtered_df
            else:
                print(f"📦 Cache exists but no news in range [{start_dt.date()}, {end_dt.date()})")
        except Exception as e:
            print(f"⚠️  Cache read failed: {e}, re-fetching")
    
    try:
        print(f"📰 Fetching news for {symbol} from GDELT BigQuery...")
        
        # Use default credentials (set via GOOGLE_APPLICATION_CREDENTIALS env var)
        client = bigquery.Client()
        
        # OPTIMIZED BigQuery query using DATE partitioning to minimize data scanned
        # Map symbol to company name for better matching
        company_name = symbol.replace('.US', '').upper()
        company_keywords = [company_name]
        
        # Add common company names
        if 'AAPL' in symbol.upper():
            company_keywords = ['APPLE', 'AAPL']
        elif 'MSFT' in symbol.upper():
            company_keywords = ['MICROSOFT', 'MSFT']
        elif 'GOOGL' in symbol.upper() or 'GOOG' in symbol.upper():
            company_keywords = ['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOG']
        elif 'TSLA' in symbol.upper():
            company_keywords = ['TESLA', 'TSLA', 'MUSK']
        elif 'NVDA' in symbol.upper():
            company_keywords = ['NVIDIA', 'NVDA']
        
        # Build search pattern
        search_pattern = '|'.join(company_keywords)
        
        # OPTIMIZED: Use DATE partitioning (indexed) to scan only needed dates
        # This reduces scanned data from 200 TiB to ~GB range
        # LIMIT increased to 50000 to capture all available data for the symbol
        query = f"""
        SELECT 
            DATE(_PARTITIONTIME) as date,
            V2Themes as themes
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP('{start_dt.strftime('%Y-%m-%d')}')
          AND _PARTITIONTIME < TIMESTAMP('{end_dt.strftime('%Y-%m-%d')}')
          AND REGEXP_CONTAINS(V2Themes, r'(?i)({search_pattern})')
        LIMIT 50000
        """
        
        print(f"   Querying GDELT for {start_dt.date()} to {end_dt.date()}...")
        print(f"   (Using partitioned table to minimize data scanned)")
        
        # Execute query and get bytes processed estimate
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        # Show cost estimate
        bytes_processed = query_job.total_bytes_processed or 0
        gb_processed = bytes_processed / (1024**3)
        estimated_cost = gb_processed * 0.005  # $5 per TB = $0.005 per GB
        print(f"   📊 Scanned: {gb_processed:.2f} GB (est. cost: ${estimated_cost:.4f})")
        
        if df is None or df.empty:
            print(f"   No GDELT BigQuery results found")
            return None
        
        print(f"   Retrieved {len(df)} events from GDELT BigQuery")
        
        # Use themes as headline
        df['headline'] = df['themes'].fillna('')
        df = df[['date', 'headline']].copy()
        df = df.dropna(subset=['date'])
        
        # Group by date
        daily = df.groupby('date')['headline'].apply(
            lambda s: '. '.join([str(x)[:200] for x in s if x])
        ).to_frame('headline')
        
        print(f"✅ Loaded {len(daily)} days of news from GDELT BigQuery")
        
        # Save to cache
        try:
            # Merge with existing cache if present
            if cache_file.exists():
                existing_df = pd.read_parquet(cache_file)
                existing_df.index = pd.to_datetime(existing_df.index)
                
                # Combine and deduplicate
                combined_df = pd.concat([existing_df, daily])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                combined_df.to_parquet(cache_file)
                print(f"💾 Updated GDELT cache: {cache_file} ({len(combined_df)} days total)")
            else:
                daily.to_parquet(cache_file)
                print(f"💾 Created GDELT cache: {cache_file} ({len(daily)} days)")
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
        
        return daily
        
    except Exception as e:
        print(f"❌ GDELT BigQuery fetch failed: {e}")
        return None


def _download_news_gdelt(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch historical news via GDELT (1979-2013 + 2015-present); return daily headlines DataFrame."""
    if not _GDELT_AVAIL:
        return None
    
    all_results = []
    
    # Parse date range
    if start:
        start_dt = pd.Timestamp(start)
    else:
        start_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
    
    if end:
        end_dt = pd.Timestamp(end)
    else:
        end_dt = pd.Timestamp.utcnow()
    
    # GDELT version boundaries
    gdelt1_end = pd.Timestamp('2013-12-31')
    gdelt2_start = pd.Timestamp('2015-02-18')
    
    # Try GDELT 1.0 for 1979-2013 range
    if start_dt <= gdelt1_end:
        try:
            print(f"📰 Fetching news for {symbol} from GDELT 1.0 (1979-2013)...")
            gd1 = gdelt.gdelt(version=1)
            
            # Clip to GDELT 1.0 range
            v1_start = start_dt
            v1_end = min(end_dt, gdelt1_end)
            
            # GDELT 1.0 uses YYYYMMDD format
            start_str = v1_start.strftime('%Y%m%d')
            end_str = v1_end.strftime('%Y%m%d')
            
            results = gd1.Search(
                date=[start_str, end_str],
                coverage=True,
                translation=False
            )
            
            if results is not None and not results.empty:
                # GDELT 1.0 has different schema - extract what we can
                if 'SQLDATE' in results.columns:
                    results['date'] = pd.to_datetime(results['SQLDATE'], format='%Y%m%d', errors='coerce')
                elif 'DATE' in results.columns:
                    results['date'] = pd.to_datetime(results['DATE'], format='%Y%m%d', errors='coerce')
                else:
                    results['date'] = pd.NaT
                
                results = results.dropna(subset=['date'])
                
                # Use available text columns as headline
                headline_col = None
                for col in ['Actor1Name', 'Actor2Name', 'EventCode', 'QuadClass']:
                    if col in results.columns:
                        headline_col = col
                        break
                
                if headline_col:
                    results['headline'] = results[headline_col].astype(str)
                    
                    # Filter for symbol mentions (GDELT has no native symbol filtering)
                    # Extract company name from symbol (e.g., AAPL -> Apple)
                    company_keywords = [symbol.replace('.US', '').upper()]
                    if symbol.startswith('AAPL'):
                        company_keywords.extend(['APPLE', 'AAPL'])
                    elif symbol.startswith('MSFT'):
                        company_keywords.extend(['MICROSOFT', 'MSFT'])
                    elif symbol.startswith('GOOGL') or symbol.startswith('GOOG'):
                        company_keywords.extend(['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOG'])
                    elif symbol.startswith('TSLA'):
                        company_keywords.extend(['TESLA', 'TSLA'])
                    elif symbol.startswith('NVDA'):
                        company_keywords.extend(['NVIDIA', 'NVDA'])
                    
                    # Filter rows mentioning the company
                    mask = results['headline'].str.contains('|'.join(company_keywords), case=False, na=False)
                    filtered_results = results[mask]
                    
                    if not filtered_results.empty:
                        all_results.append(filtered_results[['date', 'headline']])
                        print(f"   Loaded {len(filtered_results)} {symbol}-related events from GDELT 1.0 (filtered from {len(results)} total)")
                    else:
                        print(f"   No {symbol}-related events found in GDELT 1.0 (from {len(results)} total events)")
        
        except Exception as e:
            print(f"   GDELT 1.0 fetch failed: {e}")
    
    # Try GDELT 2.0 for 2015-present range
    if end_dt >= gdelt2_start:
        try:
            print(f"📰 Fetching news for {symbol} from GDELT 2.0 (2015-present)...")
            gd2 = gdelt.gdelt(version=2)
            
            # Clip to GDELT 2.0 range
            v2_start = max(start_dt, gdelt2_start)
            v2_end = end_dt
            
            # GDELT 2.0 uses YYYY-MM-DD HH:MM:SS format
            start_str = v2_start.strftime('%Y-%m-%d %H:%M:%S')
            end_str = v2_end.strftime('%Y-%m-%d %H:%M:%S')
            
            results = gd2.Search(
                date=[start_str, end_str],
                table='gkg',  # Global Knowledge Graph - richer article data
                coverage=True,
                translation=False
            )
            
            if results is not None and not results.empty:
                # Extract date
                if 'DATE' in results.columns:
                    results['date'] = pd.to_datetime(results['DATE'], format='%Y%m%d%H%M%S', errors='coerce')
                else:
                    results['date'] = pd.NaT
                
                results = results.dropna(subset=['date'])
                
                # Use V2Themes or V2Locations as headline
                if 'V2Themes' in results.columns:
                    results['headline'] = results['V2Themes'].astype(str)
                elif 'V2Locations' in results.columns:
                    results['headline'] = results['V2Locations'].astype(str)
                else:
                    results['headline'] = 'News event'
                
                # Filter for symbol mentions (GDELT has no native symbol filtering)
                company_keywords = [symbol.replace('.US', '').upper()]
                if symbol.startswith('AAPL'):
                    company_keywords.extend(['APPLE', 'AAPL'])
                elif symbol.startswith('MSFT'):
                    company_keywords.extend(['MICROSOFT', 'MSFT'])
                elif symbol.startswith('GOOGL') or symbol.startswith('GOOG'):
                    company_keywords.extend(['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOG'])
                elif symbol.startswith('TSLA'):
                    company_keywords.extend(['TESLA', 'TSLA'])
                elif symbol.startswith('NVDA'):
                    company_keywords.extend(['NVIDIA', 'NVDA'])
                
                # Filter rows mentioning the company
                mask = results['headline'].str.contains('|'.join(company_keywords), case=False, na=False)
                filtered_results = results[mask]
                
                if not filtered_results.empty:
                    all_results.append(filtered_results[['date', 'headline']])
                    print(f"   Loaded {len(filtered_results)} {symbol}-related articles from GDELT 2.0 (filtered from {len(results)} total)")
                else:
                    print(f"   No {symbol}-related articles found in GDELT 2.0 (from {len(results)} total events)")
        
        except Exception as e:
            print(f"   GDELT 2.0 fetch failed: {e}")
    
    # Combine results from both versions
    if not all_results:
        print(f"   No GDELT data found for date range")
        return None
    
    try:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.dropna(subset=['date', 'headline'])
        
        # Group by date
        combined['date_only'] = combined['date'].dt.date
        daily = combined.groupby('date_only')['headline'].apply(
            lambda s: '. '.join([str(x)[:200] for x in s if x])
        ).to_frame('headline')
        daily.index = pd.to_datetime(daily.index)
        
        print(f"✅ Loaded {len(daily)} days of news from GDELT (combined 1.0 + 2.0)")
        return daily
    
    except Exception as e:
        print(f"❌ GDELT combining failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _download_news_defeatbeta(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Fetch news via defeatbeta-api for a date range; return daily headlines DataFrame."""
    if not _lazy_import_defeatbeta():
        print(f"⚠️  defeatbeta-api not available")
        return None
    try:
        print(f"📰 Fetching news for {symbol} from {start} to {end} using defeatbeta-api...")
        tk = Ticker(symbol)
        # If no date range specified, get last 30 days
        if not start or not end:
            end_dt = pd.Timestamp.utcnow().tz_convert(NY_TZ).normalize()
            start_dt = end_dt - pd.Timedelta(days=30)
            start = start_dt.strftime('%Y-%m-%d')
            end = end_dt.strftime('%Y-%m-%d')
        
        # Fetch all available news (defeatbeta has limited date range, March 2025+)
        news_obj = tk.news()
        df = news_obj.get_news_list()
        
        if df is None or df.empty:
            print(f"   No news data from defeatbeta-api")
            return None
            
        print(f"   Retrieved {len(df)} news items from defeatbeta-api (recent data only)")
        
        # Parse defeatbeta response
        if 'title' not in df.columns or 'report_date' not in df.columns:
            print(f"   Missing required columns in defeatbeta response")
            return None
        
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.dropna(subset=['report_date', 'title'])
        
        if df.empty:
            print(f"   No valid news data after filtering")
            return None
        
        # Group by date
        df['date'] = df['report_date'].dt.date
        daily = df.groupby('date')['title'].apply(lambda s: '. '.join([str(x) for x in s if x])).to_frame('headline')
        daily.index = pd.to_datetime(daily.index)
        print(f"✅ Loaded {len(daily)} days of news from defeatbeta-api")
        return daily
        
    except Exception as e:
        print(f"❌ defeatbeta-api fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _download_news_tiingo(symbol: str, days: int = 14) -> Optional[pd.DataFrame]:
    """Fetch historical news from Tiingo API; return daily headlines DataFrame."""
    try:
        # Try to import Tiingo provider
        from src.dcf_lab.providers.tiingo_provider import TiingoProvider
        
        provider = TiingoProvider()
        news_list = provider.get_news(symbol, days=days)
        
        if not news_list:
            return None
        
        # Convert to DataFrame format similar to yfinance
        records = []
        for article in news_list:
            records.append({
                'published': article.get('published', ''),
                'title': article.get('title', ''),
                'summary': article.get('summary', '')
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df['ts'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
        df = df.dropna(subset=['ts'])
        
        # Combine title and summary for richer sentiment analysis
        df['headline'] = df.apply(
            lambda row: f"{row['title']}. {row['summary']}" if row['summary'] else row['title'],
            axis=1
        )
        
        # Group by date and combine headlines
        df['date'] = df['ts'].dt.tz_convert(NY_TZ).dt.date
        cutoff_date = (pd.Timestamp.utcnow().tz_convert(NY_TZ).normalize() - pd.Timedelta(days=days)).date()
        df = df[df['date'] >= cutoff_date]
        
        if df.empty:
            return None
        
        daily = df.groupby('date')['headline'].apply(lambda s: '. '.join([str(x) for x in s if x])).to_frame('headline')
        daily.index = pd.to_datetime(daily.index)
        return daily
        
    except Exception:
        return None


def _apply_gap_policy(
    features: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    gap_thresh: int,
    freq: str = "B",
) -> pd.DataFrame:
    """Apply t-1 shift, optional reindex to business days, bounded carry-forward by gap_thresh, then zero-fill."""
    out = features.copy()
    # Enforce t-1 cutoff to avoid same-day leakage
    out = out.shift(1)
    # If a range is provided, reindex to full business days span
    idx_start = None
    idx_end = None
    if start:
        idx_start = pd.to_datetime(start)
    elif len(out.index):
        idx_start = out.index.min()
    if end:
        idx_end = pd.to_datetime(end)
    elif len(out.index):
        idx_end = out.index.max()
    if idx_start is not None and idx_end is not None:
        full_idx = pd.date_range(idx_start, idx_end, freq=freq)
        out = out.reindex(full_idx)
    # Apply carry-forward limited by gap_thresh
    try:
        lim = int(max(0, gap_thresh))
    except Exception:
        lim = 0
    if lim > 0:
        out = out.ffill(limit=lim)
    # Any remaining gaps become neutral/zero
    out = out.fillna(0.0)
    # Final filter for safety when no reindexing happened
    if start:
        out = out[out.index >= pd.to_datetime(start)]
    if end:
        out = out[out.index <= pd.to_datetime(end)]
    return out


def fetch(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ffill_days: int = 1,
    gap_thresh: int = 0,
) -> pd.DataFrame:
    """
    Build daily FinBERT sentiment features from cached or freshly fetched headlines.
    - PRIMARY: Uses GDELTAutoFetcher for intelligent source selection
      * GDELT 2.0 BigQuery for 2015+ (fast, partitioned, low-cost)
      * GDELT 1.0 parallel download for 2013 and prior (comprehensive historical coverage)
    - Falls back to defeatbeta-api, legacy GDELT, yfinance news (NOT price proxy)
    - Uses neutral fill (0.0) for days without news articles
    - Let ML model learn temporal patterns during training
    Returns empty DataFrame if FinBERT transformer is unavailable.
    """
    # Fail fast if FinBERT not available - no price proxy fallback
    if analyze_finbert_sentiment is None:
        # Write error to file for debugging subprocess issues
        try:
            error_file = Path("/tmp/finbert_error.txt")
            with open(error_file, 'a') as f:
                f.write(f"{pd.Timestamp.now()}: FinBERT transformer not available\n")
        except:
            pass
        logger.error("❌ FinBERT transformer not available - cannot generate sentiment features")
        logger.error("💡 Install transformers: pip install transformers safetensors")
        return pd.DataFrame()

    def _choose_news_source(sym: str, start: Optional[str] = None, end: Optional[str] = None):
        # AAPL: Use GDELT cache only (legacy behavior)
        if sym.upper() == 'AAPL':
            print("🔒 AAPL: Using GDELT cache only")
            df = _load_news_from_cache(sym)
            if df is not None and not df.empty:
                return df, 'gdelt_cache'
            print(f"❌ No GDELT cache found for {sym}")
            return None, None
        
        # All other symbols: Use EODHD from 2020 onwards
        print(f"📰 {sym}: Using EODHD news (2020+ coverage)")
        
        # Import EODHD provider
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
        except ImportError:
            print("❌ EODHD provider not available")
            return None, None
        
        eodhd = get_eodhd_provider()
        if not eodhd.api_key:
            # Write error to file for debugging
            try:
                with open("/tmp/finbert_error.txt", 'a') as f:
                    f.write(f"{pd.Timestamp.now()}: EODHD API key not configured for {sym}\n")
            except:
                pass
            print("❌ EODHD API key not configured")
            return None, None
        
        # Determine date range (clip to 2020+ for EODHD)
        start_dt = pd.Timestamp(start) if start else pd.Timestamp('2020-01-01')
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.now()
        
        # EODHD only has data from 2020 onwards
        eodhd_start = max(start_dt, pd.Timestamp('2020-01-01'))
        
        if eodhd_start > end_dt:
            print(f"⚠️  Requested period ({start_dt.date()} to {end_dt.date()}) is before 2020 - no EODHD data available")
            return None, None
        
        print(f"   Fetching from {eodhd_start.date()} to {end_dt.date()}")
        
        # Fetch news articles (no limit - get all available historical data)
        articles = eodhd.get_news(
            sym,
            start_date=eodhd_start.strftime('%Y-%m-%d'),
            end_date=end_dt.strftime('%Y-%m-%d')
        )
        
        if not articles:
            print(f"❌ No EODHD articles found for {sym}")
            return None, None
        
        # Convert to DataFrame
        rows = []
        for article in articles:
            article_date = article.get('date', '')
            title = article.get('title', '')
            
            if not article_date or not title:
                continue
            
            # Parse date
            dt = pd.to_datetime(article_date, utc=True, errors='coerce')
            if pd.isna(dt):
                continue
            
            rows.append({
                'date': dt.tz_convert(NY_TZ).date(),
                'title': title
            })
        
        if not rows:
            print(f"❌ No valid articles after parsing for {sym}")
            return None, None
        
        # Aggregate by date
        df = pd.DataFrame(rows)
        df = df.groupby('date')['title'].apply(lambda s: '. '.join(str(x) for x in s if x)).to_frame('headline')
        df.index = pd.to_datetime(df.index)
        
        print(f"✅ EODHD: {len(df)} days with news ({df.index.min().date()} to {df.index.max().date()})")
        return df, 'eodhd'

    news_df, source = _choose_news_source(symbol, start=start, end=end)
    
    # If no news data available, return neutral-filled DataFrame for the requested range
    # This is acceptable - not all symbols have news coverage, and the model can learn from has_data=0
    if news_df is None or news_df.empty:
        print(f"⚠️  No news data available for {symbol} - using neutral fill for entire date range")
        print("💡 This is acceptable - FINBERT_HAS_DATA=0 tells the model no news was available")
        
        # Return neutral-filled features for the requested range
        if start and end:
            full_calendar = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq='D')
            out = pd.DataFrame({
                'FINBERT_SCORE': 0.0,
                'FINBERT_NEUTRAL': 0.0,
                'FINBERT_CONFIDENCE': 0.0,
                'FINBERT_HAS_DATA': 0.0,  # Critical: marks all days as no real data
            }, index=full_calendar)
            out.attrs['provenance'] = {'source': 'finbert', 'news': 'none', 'neutral_fill': True}
            out.attrs['telemetry'] = {
                'rows': int(len(out)),
                'cols': 4,
                'news_days': 0,
                'neutral_days': int(len(out)),
                'proxy': False,
                'no_news_coverage': True
            }
            print(f"📊 FinBERT: Returning {len(out)} neutral-fill days (0% news coverage)")
            return out
        else:
            # No date range specified, return empty
            return pd.DataFrame()
    
    # Validate date coverage (informational only, no fallback)
    if start and end:
        news_start = pd.Timestamp(news_df.index.min())
        news_end = pd.Timestamp(news_df.index.max())
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end)
        
        if news_end < requested_start or news_start > requested_end:
            print(f"⚠️  News data ({news_start.date()} to {news_end.date()}) doesn't overlap with requested range ({requested_start.date()} to {requested_end.date()})")
            print("💡 Continuing with available news data (neutral fill will handle gaps)")
        
        if len(news_df) < 50:
            print(f"⚠️  Limited news coverage: {len(news_df)} days with articles")
            print("💡 Days without news will use neutral fill (0.0 score, 0.0 confidence)")

    # Score headlines with FinBERT
    texts = news_df['headline'].astype(str).tolist()
    scores = analyze_finbert_sentiment(texts)
    df = pd.DataFrame(scores, index=news_df.index).sort_index()
    
    # Ensure index is DatetimeIndex for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Aggregate to business-day averages (only for days with news)
    daily = df.resample('D').mean().dropna(how='all')
    
    # Feature transforms
    daily['FINBERT_SCORE'] = daily.get('positive', 0.0) - daily.get('negative', 0.0)
    daily['FINBERT_NEUTRAL'] = daily.get('neutral', 0.0)
    daily['FINBERT_CONFIDENCE'] = daily.get('confidence', 0.0)  # Model confidence score
    
    # Create full date range and apply neutral fill for days without news
    if start and end:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        full_calendar = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Reindex to full calendar with neutral fill (0.0, 0.0, 0.0)
        out = daily[['FINBERT_SCORE', 'FINBERT_NEUTRAL', 'FINBERT_CONFIDENCE']].reindex(full_calendar, fill_value=0.0)
        
        # Add data availability mask: 1.0 for real news days, 0.0 for neutral fill
        # This lets LSTM learn different patterns for days with/without news
        out['FINBERT_HAS_DATA'] = 0.0
        # Mark days that had real news data
        mask_dates = daily.index.intersection(full_calendar)
        out.loc[mask_dates, 'FINBERT_HAS_DATA'] = 1.0
        
        # Mark days with actual news (confidence > 0)
        news_days = len(daily)
        neutral_days = len(out) - news_days
        print(f"📊 FinBERT coverage: {news_days} news days ({100*news_days/len(out):.1f}%), {neutral_days} neutral fill days ({100*neutral_days/len(out):.1f}%)")
    else:
        out = daily[['FINBERT_SCORE', 'FINBERT_NEUTRAL', 'FINBERT_CONFIDENCE']].copy()
        out['FINBERT_HAS_DATA'] = 1.0
    
    # Apply gap policy (informational, doesn't change neutral fill)
    out = _apply_gap_policy(out, start=start, end=end, gap_thresh=gap_thresh, freq='D')
    
    out.attrs['provenance'] = {'source': 'finbert', 'news': source or 'unknown', 'neutral_fill': True}
    out.attrs['telemetry'] = {
        'rows': int(out.shape[0]),
        'cols': int(out.shape[1]),
        'news_days': int(news_days) if start and end else int(len(daily)),
        'neutral_days': int(neutral_days) if start and end else 0,
        'ffill_days': 0,  # No forward-fill with neutral fill strategy
        'gap_thresh': int(max(0, gap_thresh) if isinstance(gap_thresh, (int, float)) else 0),
        't_minus_1_cutoff': True,
        'align': 'backward',
        'proxy': False  # Real FinBERT sentiment, not price proxy
    }
    return out


__all__ = ['fetch']
