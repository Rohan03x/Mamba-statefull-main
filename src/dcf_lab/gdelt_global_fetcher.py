#!/usr/bin/env python3
"""
GDELT Global Event Fetcher for doc_embedding_novelty_hf

Fetches global GDELT events (NOT symbol-specific) for detecting:
- Macro regime changes
- Geopolitical shifts
- Regulatory changes
- Global risk events

Data Sources:
- GDELT 1.0 Events: 1979-2013 (parallel download)
- GDELT 2.0 Events + GKG: 2015+ (BigQuery)

Pulls:
- GKG: themes, entities, tone, locations, summary text
- Events: event codes, actors, Goldstein scale, descriptions

Author: DCF Lab Team
Created: 2025-11-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import logging
import multiprocessing
import os
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prefer_thread_pool() -> bool:
    """Return True when process pools are unsafe (daemon process) or disabled by env."""
    env = os.getenv("GDELT_DISABLE_PROCESS_POOL", "").strip().lower() in {"1", "true", "yes", "on"}
    if env:
        return True

    # If Hugging Face / torch stack is already imported, avoid forking a process pool.
    # Forking after tokenizers/torch have initialized native thread pools can cause
    # deadlocks and massive copy-on-write memory amplification.
    if any(mod in sys.modules for mod in ("tokenizers", "transformers", "torch", "sentence_transformers")):
        return True

    try:
        return bool(multiprocessing.current_process().daemon)
    except Exception:
        return False

# GDELT version boundaries
GDELT1_START = pd.Timestamp('1979-02-18')
GDELT1_END = pd.Timestamp('2013-12-31')
GDELT2_START = pd.Timestamp('2015-02-18')  # GDELT 2.0 starts Feb 18, 2015
# GAP: 2014-01-01 to 2015-02-17 (~13.5 months) - NO DATA AVAILABLE

# Try imports
try:
    import gdelt
    _GDELT1_AVAIL = True
except ImportError:
    gdelt = None
    _GDELT1_AVAIL = False
    logger.warning("GDELT 1.0 library not available - install with: pip install gdelt")

try:
    from google.cloud import bigquery
    _GDELT2_AVAIL = True
except ImportError:
    bigquery = None
    _GDELT2_AVAIL = False
    logger.warning("BigQuery not available - install with: pip install google-cloud-bigquery")

# Global BigQuery client (singleton to avoid re-initialization issues)
_BQ_CLIENT = None


# Thematic clusters for event classification
EVENT_CLUSTERS = {
    'macro': [
        # Keywords
        'ECON', 'BANK', 'FIN', 'TRADE', 'MARKET', 'STOCK', 'CURRENCY', 'DEBT', 'FISCAL', 'MONETARY',
        'GDP', 'INFLATION', 'UNEMPLOYMENT', 'INTEREST', 'FED', 'ECB', 'BOJ', 'PBOC',
        # CAMEO event codes for economic events
        '031', '032', '033', '034',  # Economic cooperation
        '071', '072', '073',         # Economic aid
        '161', '162', '163', '164', '165', '166',  # Economic protest
    ],
    'geopolitical': [
        # Keywords
        'MIL', 'PROTEST', 'REBEL', 'TERROR', 'WAR', 'DIPLOMACY', 'SANCTION', 'TREATY',
        'MILITARY', 'DIPLOMATIC', 'ALLIANCE', 'NATO', 'CHINA', 'RUSSIA', 'UKRAINE',
        # CAMEO codes for military/diplomatic
        '19',   # Military force
        '20',   # Unconventional violence
        '04',   # Diplomatic consultation
        '05',   # Diplomatic cooperation
        '13',   # Diplomatic protest
        '173',  # Military protest
    ],
    'regulatory': [
        # Keywords
        'LAW', 'REGULATE', 'POLICY', 'LEGAL', 'COURT', 'LEGISLATION', 'COMPLIANCE', 'TAX',
        'ANTITRUST', 'SEC', 'FTC', 'INVESTIGATION', 'FINE', 'LAWSUIT', 'SETTLEMENT',
        # CAMEO codes
        '1231', '1232', '1233',  # Legal actions
        '175',  # Judicial protest
    ],
    'energy': [
        'OIL', 'GAS', 'ENERGY', 'PETROLEUM', 'OPEC', 'COAL', 'RENEWABLE', 'NUCLEAR',
        'CRUDE', 'BRENT', 'WTI', 'NATURAL GAS', 'PIPELINE', 'SOLAR', 'WIND'
    ],
    'conflict': [
        # Keywords
        'ATTACK', 'ASSAULT', 'FIGHT', 'KILL', 'INJURE', 'BOMB', 'STRIKE', 'CLASH',
        'VIOLENCE', 'TERRORIST', 'RIOT', 'SHOOTING', 'COUP',
        # CAMEO codes for violence/protest
        '14',   # Protest
        '18',   # Assault
        '145',  # Violent protest
        '180', '181', '182', '183', '184', '185', '186',  # Types of assault
    ],
    'tech': [
        'TECH', 'CYBER', 'DIGITAL', 'AI', 'DATA', 'INTERNET', 'SOFTWARE', 'INNOVATION',
        'HACK', 'BREACH', 'PRIVACY', 'SURVEILLANCE', 'BLOCKCHAIN', 'BITCOIN'
    ]
}


def _download_gdelt1_month(args: Tuple) -> Optional[pd.DataFrame]:
    """Download GDELT 1.0 data for a single month (worker function)"""
    year, month, cache_path, sample_size, download_enabled = args
    
    if not _GDELT1_AVAIL and download_enabled:
        return None
    
    try:
        # Check cache first
        if cache_path.exists():
            # OPTIMIZED: Use pyarrow for fast row-group sampling on large files
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(cache_path)
            total_rows = parquet_file.metadata.num_rows
            
            # If sample needed and file is large, use row-group sampling
            if sample_size and total_rows > sample_size * 2:
                # Read every Nth row group for speed
                num_row_groups = parquet_file.num_row_groups
                step = max(1, num_row_groups // 5)  # Read ~20% of row groups
                row_groups = list(range(0, num_row_groups, step))
                
                df = parquet_file.read_row_groups(row_groups[:10]).to_pandas()
                
                # Then sample from subset
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
            else:
                # Small file or no sampling needed - read all
                df = pd.read_parquet(cache_path)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
            
            return df
        
        # Download if enabled
        if not download_enabled:
            logger.debug(f"Cache miss for {year}-{month:02d}, skipping (cache-only mode)")
            return None
        
        # Download from GDELT servers
        gd = gdelt.gdelt(version=1)
        date_str = f"{year} {month}"
        
        logger.info(f"  📥 Downloading {year}-{month:02d}: sampling to {sample_size} events...")
        results = gd.Search(date_str, table='events', coverage=True)
        
        if results is None or results.empty:
            logger.warning(f"    ⚠️  No events found for {year}-{month:02d}")
            return None
        
        # Select key columns for global events
        cols_to_keep = [
            'SQLDATE', 'EventCode', 'EventBaseCode', 'EventRootCode',
            'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles',
            'AvgTone', 'Actor1Name', 'Actor2Name', 'Actor1CountryCode', 'Actor2CountryCode',
            'ActionGeo_FullName', 'ActionGeo_CountryCode', 'SOURCEURL'
        ]
        
        available_cols = [col for col in cols_to_keep if col in results.columns]
        results = results[available_cols].copy()
        
        # Parse date
        if 'SQLDATE' in results.columns:
            results['date'] = pd.to_datetime(results['SQLDATE'], format='%Y%m%d', errors='coerce')
            results = results.dropna(subset=['date'])
        
        # Sample to limit if needed
        if sample_size and len(results) > sample_size:
            results = results.sample(n=sample_size, random_state=42)
        
        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(cache_path, index=False, engine='pyarrow', compression='snappy')
        
        logger.info(f"    ✅ Saved {len(results):,} events to {cache_path.name}")
        return results
        
    except Exception as e:
        logger.error(f"    ❌ Error downloading {year}-{month:02d}: {e}")
        return None


class GDELTGlobalFetcher:
    """
    Fetcher for global GDELT events (NOT symbol-specific)
    Used for detecting macro regime changes and geopolitical shifts
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_workers: Optional[int] = None):
        """
        Initialize GDELT global event fetcher
        
        Args:
            cache_dir: Directory for caching GDELT data (default: from cache_paths)
            max_workers: Max parallel workers for GDELT 1.0 (defaults to 80% of CPU count)
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Use centralized cache paths (with legacy fallback)
            try:
                from src.cache_paths import resolve_gdelt_global_cache_dir
                self.cache_dir = resolve_gdelt_global_cache_dir()
            except ImportError:
                self.cache_dir = Path("data/cache/gdelt_global")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, int(cpu_count * 0.8))
        else:
            self.max_workers = max_workers
        
        logger.info(f"🌍 GDELT Global Fetcher initialized with {self.max_workers} workers")
    
    def _get_bq_client(self):
        """Get or create BigQuery client (lazy singleton)"""
        global _BQ_CLIENT
        if _BQ_CLIENT is None and _GDELT2_AVAIL:
            try:
                _BQ_CLIENT = bigquery.Client()
                logger.info("✅ BigQuery client initialized")
            except Exception as e:
                logger.warning(f"BigQuery client initialization failed: {e}")
        return _BQ_CLIENT
    
    def fetch_gdelt1_events(self, start_date: str, end_date: str, max_events_per_month: int = 2000, use_download: bool = False) -> pd.DataFrame:
        """
        Fetch GDELT 1.0 events in parallel with per-month limit
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_events_per_month: Maximum events per month (default: 2000)
            use_download: If True, download from GDELT servers. If False, cache-only.
            
        Returns:
            DataFrame with global events
        """
        if use_download and not _GDELT1_AVAIL:
            logger.error("GDELT 1.0 library not available - install with: pip install gdelt")
            return pd.DataFrame()
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Don't auto-clip - respect user's requested dates within GDELT 1.0 range
        if start < GDELT1_START:
            logger.warning(f"Start date {start.date()} before GDELT 1.0 start {GDELT1_START.date()}, using {GDELT1_START.date()}")
            start = GDELT1_START
        if end > GDELT1_END:
            logger.warning(f"End date {end.date()} after GDELT 1.0 end {GDELT1_END.date()}, using {GDELT1_END.date()}")
            end = GDELT1_END
        
        if start > end:
            return pd.DataFrame()
        
        # Generate month list
        months = []
        current = start
        while current <= end:
            cache_path = self.cache_dir / "gdelt1" / f"{current.year}" / f"{current.year}_{current.month:02d}.parquet"
            months.append((current.year, current.month, cache_path, max_events_per_month, use_download))
            current = current + pd.DateOffset(months=1)
        
        total_months = len(months)
        
        logger.info(f"📰 Fetching GDELT 1.0 events: {start.date()} to {end.date()}")
        logger.info(f"🎯 Target: {max_events_per_month:,} events per month across {total_months} months")
        if use_download:
            logger.info(f"⚡ Downloading in parallel with {self.max_workers} workers...")
        else:
            logger.info(f"📦 Cache-only mode: Loading with {self.max_workers} workers...")
        
        # Parallel loading/downloading with in-worker sampling (FAST!)
        all_data = []
        total_events = 0
        loaded_count = 0

        executor_cls = ThreadPoolExecutor if _prefer_thread_pool() else ProcessPoolExecutor
        if executor_cls is ThreadPoolExecutor:
            logger.warning(
                "GDELTGlobalFetcher: using ThreadPoolExecutor for monthly fetch (daemon/process-pool disabled)"
            )

        with executor_cls(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_download_gdelt1_month, args): args for args in months}

            for future in as_completed(futures):
                result = future.result()
                if result is not None and not result.empty:
                    # Sampling already done in worker - just collect
                    all_data.append(result)
                    total_events += len(result)
                    loaded_count += 1

                    if loaded_count % 10 == 0 or loaded_count == total_months:
                        logger.info(
                            f"⚡ Progress: {loaded_count}/{total_months} months, {total_events:,} events"
                        )
        
        if not all_data:
            logger.warning("No GDELT 1.0 data found")
            return pd.DataFrame()
        
        # Merge and filter date range
        df = pd.concat(all_data, ignore_index=True)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ GDELT 1.0: {len(df):,} events from {len(all_data)} months")
        return df
    
    def fetch_gdelt2_events(self, start_date: str, end_date: str, use_bigquery: bool = False) -> pd.DataFrame:
        """
        Fetch GDELT 2.0 events from BigQuery or cache
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_bigquery: If True, download from BigQuery and cache. If False, load from cache only.
            
        Returns:
            DataFrame with global events
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Clip to GDELT 2.0 range
        start = max(start, GDELT2_START)
        
        if start > end:
            return pd.DataFrame()
        
        logger.info(f"📰 Fetching GDELT 2.0 events: {start.date()} to {end.date()}")
        
        if use_bigquery:
            # Download from BigQuery
            if _GDELT2_AVAIL:
                bq_client = self._get_bq_client()
                if bq_client is not None:
                    return self._fetch_gdelt2_bigquery_monthly(start, end, bq_client)
            logger.error("BigQuery not available!")
            return pd.DataFrame()
        else:
            # CACHE-ONLY MODE: Load from cache only, no downloads
            logger.info("📦 Cache-only mode: Loading from cached files...")
            return self._fetch_gdelt2_from_cache(start, end)
    
    def _fetch_gdelt2_from_cache(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Load GDELT 2.0 from cache only (no downloads)"""
        cache_dir = self.cache_dir / "gdelt2"
        
        # Generate date range
        date_range = pd.date_range(start, end, freq='D')
        
        all_data = []
        corrupted_files = 0
        
        for date in date_range:
            year = date.year
            cache_path = cache_dir / str(year) / f"{date.strftime('%Y%m%d')}.parquet"
            
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    
                    # VALIDATION: Check if data is corrupted
                    if 'EventCode' in df.columns:
                        # EventCode should be numeric (CAMEO codes like "042", "190")
                        # If first value is a word like "CORPORATION", data is corrupted
                        sample_code = df['EventCode'].dropna().iloc[0] if len(df) > 0 else None
                        if sample_code and isinstance(sample_code, str):
                            if not sample_code.replace('.','').isdigit():
                                logger.warning(f"⚠️  Corrupted cache detected in {cache_path.name}: EventCode='{sample_code}' (should be numeric)")
                                corrupted_files += 1
                                continue
                    
                    # VALIDATION: Check date consistency
                    if 'SQLDATE' in df.columns:
                        # SQLDATE should match the filename date
                        expected_sqldate = int(date.strftime('%Y%m%d'))
                        actual_sqldates = df['SQLDATE'].dropna().astype(str).str[:8].unique()
                        if len(actual_sqldates) > 0:
                            actual_sqldate = int(actual_sqldates[0]) if actual_sqldates[0].isdigit() else 0
                            if actual_sqldate != expected_sqldate and abs(actual_sqldate - expected_sqldate) > 100:
                                logger.warning(f"⚠️  Date mismatch in {cache_path.name}: expected {expected_sqldate}, got {actual_sqldate}")
                                corrupted_files += 1
                                continue
                    
                    # CRITICAL: Filter to only requested date (cache may have bad dates)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df[df['date'].dt.date == date.date()]
                    
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.debug(f"Failed to load cache {cache_path}: {e}")
                    corrupted_files += 1
        
        if corrupted_files > 0:
            logger.error(f"❌ Found {corrupted_files} corrupted cache files! GDELT 2.0 cache needs to be rebuilt.")
            logger.error(f"   Run: rm -rf {cache_dir}/* && python tools/download_gdelt_global_cache.py")
        
        if not all_data:
            logger.warning(f"No cached GDELT 2.0 data found for {start.date()} to {end.date()}")
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Loaded {len(df):,} events from {len(all_data)} cached files")
        return df
    
    def _fetch_gdelt2_bigquery_monthly(self, start: pd.Timestamp, end: pd.Timestamp, bq_client) -> pd.DataFrame:
        """
        Fetch GDELT 2.0 from BigQuery with monthly sampling (2000 events per month)
        
        Args:
            start: Start timestamp
            end: End timestamp
            bq_client: BigQuery client
            
        Returns:
            DataFrame with sampled events
        """
        from google.cloud import bigquery
        
        all_dfs = []
        current_month = start
        
        while current_month <= end:
            # Get month boundaries
            month_start = current_month.replace(day=1)
            if current_month.month == 12:
                month_end = current_month.replace(year=current_month.year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                month_end = current_month.replace(month=current_month.month + 1, day=1) - pd.Timedelta(days=1)
            
            month_end = min(month_end, end)
            
            logger.info(f"  📥 Downloading {month_start.strftime('%Y-%m')}: max 2000 events...")
            
            # Query with LIMIT 2000 per month
            query = f"""
            SELECT 
                SQLDATE,
                EventCode,
                EventBaseCode,
                EventRootCode,
                QuadClass,
                GoldsteinScale,
                NumMentions,
                AvgTone,
                Actor1Name,
                Actor1CountryCode,
                Actor2Name,
                Actor2CountryCode,
                ActionGeo_FullName as location,
                ActionGeo_CountryCode,
                SOURCEURL as url
            FROM 
                `gdelt-bq.gdeltv2.events`
            WHERE 
                SQLDATE >= {month_start.strftime('%Y%m%d')}
                AND SQLDATE <= {month_end.strftime('%Y%m%d')}
            ORDER BY RAND()
            LIMIT 2000
            """
            
            try:
                df = bq_client.query(query).to_dataframe()
                
                if len(df) > 0:
                    # Convert SQLDATE to datetime
                    df['date'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d')
                    df = df.drop(columns=['SQLDATE'])
                    
                    # Save to cache (monthly file)
                    cache_dir = self.cache_dir / "gdelt2" / str(month_start.year)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_dir / f"{month_start.strftime('%Y%m')}.parquet"
                    
                    df.to_parquet(cache_file, index=False, engine='pyarrow', compression='snappy')
                    logger.info(f"    ✅ Saved {len(df):,} events to {cache_file.name}")
                    
                    all_dfs.append(df)
                else:
                    logger.warning(f"    ⚠️  No events found for {month_start.strftime('%Y-%m')}")
                    
            except Exception as e:
                logger.error(f"    ❌ Error querying {month_start.strftime('%Y-%m')}: {e}")
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"✅ Downloaded {len(combined):,} total events across {len(all_dfs)} months")
            return combined
        else:
            logger.warning("No events downloaded from BigQuery")
            return pd.DataFrame()

    def _fetch_gdelt2_bigquery(self, start: pd.Timestamp, end: pd.Timestamp, bq_client) -> pd.DataFrame:
        
        # Query GDELT 2.0 events with DATE FILTERING and 2M LIMIT
        # SQLDATE format: YYYYMMDD (8 digits, stored as INT64)
        # Use SQLDATE integer comparison for efficient filtering
        start_int = int(start.strftime('%Y%m%d'))
        end_int = int(end.strftime('%Y%m%d'))
        
        query = f"""
        SELECT 
            DATE(PARSE_TIMESTAMP('%Y%m%d', CAST(SQLDATE AS STRING))) as date,
            EventCode,
            EventBaseCode,
            EventRootCode,
            GoldsteinScale,
            NumMentions,
            NumSources,
            NumArticles,
            AvgTone,
            Actor1Name,
            Actor2Name,
            Actor1CountryCode,
            Actor2CountryCode,
            ActionGeo_FullName as location,
            ActionGeo_CountryCode as country_code,
            SOURCEURL as url
        FROM 
            `gdelt-bq.gdeltv2.events`
        WHERE 
            SQLDATE >= {start_int}
            AND SQLDATE <= {end_int}
        LIMIT 2000000
        """
        
        logger.info(f"🔍 Query will scan date range {start_int} to {end_int} (LIMIT 2,000,000 articles)")
        
        try:
            job = bq_client.query(query)
            logger.info(f"⏳ Query submitted, waiting for results...")
            df = job.result(timeout=300).to_dataframe()  # 5 minute timeout
            logger.info(f"✅ GDELT 2.0: {len(df):,} events retrieved (unique dates: {df['date'].nunique() if len(df) > 0 else 0})")
            return df
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            return pd.DataFrame()
    
    def _download_gdelt2_csv_for_date(self, args: Tuple) -> Optional[pd.DataFrame]:
        """Download GDELT 2.0 CSV for a single date (worker function)"""
        date, cache_path = args
        
        # Check cache first
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except:
                pass
        
        import io
        import zipfile
        from urllib.request import urlopen
        from urllib.parse import urljoin
        
        # GDELT 2.0 updates every 15 minutes - download 4 files per day (every 6 hours)
        timestamps = [
            date.replace(hour=0, minute=0),
            date.replace(hour=6, minute=0),
            date.replace(hour=12, minute=0),
            date.replace(hour=18, minute=0),
        ]
        
        all_events = []
        base_url = "http://data.gdeltproject.org/gdeltv2/"
        
        for ts in timestamps:
            filename = ts.strftime("%Y%m%d%H%M%S") + ".export.CSV.zip"
            url = urljoin(base_url, filename)
            
            try:
                with urlopen(url, timeout=30) as response:
                    zip_data = response.read()
                
                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as csv_file:
                        # GDELT 2.0 Event format (58 columns, tab-delimited)
                        df = pd.read_csv(
                            csv_file,
                            sep='\t',
                            header=None,
                            usecols=[1, 26, 27, 28, 30, 31, 32, 33, 34, 6, 16, 7, 17, 53, 57],
                            names=['SQLDATE', 'EventCode', 'EventBaseCode', 'EventRootCode',
                                   'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles',
                                   'AvgTone', 'Actor1Name', 'Actor2Name', 'Actor1CountryCode',
                                   'Actor2CountryCode', 'location', 'url'],
                            dtype=str,
                            on_bad_lines='skip'
                        )
                        
                        if not df.empty:
                            # Convert numeric columns
                            df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
                            df['NumArticles'] = pd.to_numeric(df['NumArticles'], errors='coerce')
                            # NO FILTERING - collect all events
                            all_events.append(df)
            except:
                continue
        
        if not all_events:
            return None
        
        result = pd.concat(all_events, ignore_index=True)
        result['date'] = pd.to_datetime(result['SQLDATE'], format='%Y%m%d', errors='coerce')
        result = result.dropna(subset=['date'])
        
        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        
        return result
    
    def _fetch_gdelt2_csv(self, start: pd.Timestamp, end: pd.Timestamp, max_articles: int = 2_000_000) -> pd.DataFrame:
        """Fetch GDELT 2.0 via direct CSV downloads (parallel across workers with article limit)"""
        logger.info(f"📦 Downloading GDELT 2.0 CSV files: {start.date()} to {end.date()}")
        logger.info(f"🎯 Target: {max_articles:,} total articles")
        
        # Generate date list
        date_list = []
        current = start
        while current <= end:
            cache_path = self.cache_dir / "gdelt2" / str(current.year) / f"{current.strftime('%Y%m%d')}.parquet"
            date_list.append((current, cache_path))
            current += timedelta(days=1)
        
        logger.info(f"📅 Total days: {len(date_list)} | Workers: {self.max_workers}")
        
        all_results = []
        completed = 0
        total_articles = 0
        
        executor_cls = ThreadPoolExecutor if _prefer_thread_pool() else ProcessPoolExecutor
        if executor_cls is ThreadPoolExecutor:
            logger.warning("GDELTGlobalFetcher: using ThreadPoolExecutor for GDELT2 CSV fetch (daemon/process-pool disabled)")
        with executor_cls(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_gdelt2_csv_for_date, args): args[0] for args in date_list}
            
            for future in as_completed(futures):
                date = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        # Check if we still need more articles
                        if total_articles < max_articles:
                            remaining = max_articles - total_articles
                            result = result.head(remaining)
                            all_results.append(result)
                            total_articles += len(result)
                            
                            if completed % 50 == 0:
                                logger.info(f"[{completed}/{len(date_list)}] Downloaded {len(result)} events | Total: {total_articles:,}/{max_articles:,}")
                            
                            if total_articles >= max_articles:
                                logger.info(f"🎯 Reached target of {max_articles:,} articles")
                except Exception as exc:
                    if completed % 100 == 0:
                        logger.debug(f"[{completed}/{len(date_list)}] {date.date()}: {exc}")
        
        if not all_results:
            logger.warning("No GDELT 2.0 CSV data found")
            return pd.DataFrame()
        
        df = pd.concat(all_results, ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ GDELT 2.0 CSV: {len(df):,} events from {len(all_results)} days")
        return df
    
    def _fetch_from_merged_cache(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load events from merged cache (unified GDELT 1.0 + 2.0)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with events from merged cache
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Generate month list
        months = []
        current = start
        while current <= end:
            cache_path = self.cache_dir / f"{current.year}" / f"{current.year}_{current.month:02d}.parquet"
            if cache_path.exists():
                months.append(cache_path)
            current = current + pd.DateOffset(months=1)
        
        if not months:
            logger.warning(f"No merged cache files found for {start.date()} to {end.date()}")
            return pd.DataFrame()
        
        logger.info(f"📦 Loading {len(months)} months from merged cache...")
        logger.info(f"⚡ Using {self.max_workers} workers for parallel loading...")
        
        # Parallel loading
        all_data = []
        executor_cls = ThreadPoolExecutor if _prefer_thread_pool() else ProcessPoolExecutor
        if executor_cls is ThreadPoolExecutor:
            logger.warning("GDELTGlobalFetcher: using ThreadPoolExecutor for merged cache load (daemon/process-pool disabled)")
        with executor_cls(max_workers=self.max_workers) as executor:
            futures = {executor.submit(pd.read_parquet, path): path for path in months}
            
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    path = futures[future]
                    logger.error(f"Error loading {path}: {e}")
        
        if not all_data:
            logger.warning("No data loaded from merged cache")
            return pd.DataFrame()
        
        # Combine and filter date range
        df = pd.concat(all_data, ignore_index=True)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Loaded {len(df):,} events from merged cache ({df['date'].min()} to {df['date'].max()})")
        return df
    
    def fetch(self, start_date: str, end_date: str, use_bigquery: bool = False, use_gdelt1_download: bool = False) -> pd.DataFrame:
        """
        Fetch global GDELT events from cache (downloads disabled by default)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_bigquery: If True, download GDELT 2.0 from BigQuery (2000/month). DEFAULT: False (cache-only).
            use_gdelt1_download: If True, download GDELT 1.0 from servers (2000/month). DEFAULT: False (cache-only).
            
        Returns:
            DataFrame with global events
            
        Note:
            If cache_dir points to 'merged' directory, will load from unified cache.
            Otherwise loads from separate gdelt1/gdelt2 caches.
            2014-01-01 to 2015-02-17 has NO DATA (gap between GDELT 1.0 and 2.0)
        """
        # Check if using merged cache
        if self.cache_dir.name == 'merged' or 'merged' in str(self.cache_dir):
            return self._fetch_from_merged_cache(start_date, end_date)
        
        # Original logic for separate caches
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        all_data = []
        
        # Check if request falls entirely in the gap
        gap_start = GDELT1_END + pd.Timedelta(days=1)
        gap_end = GDELT2_START - pd.Timedelta(days=1)
        
        if start >= gap_start and end <= gap_end:
            logger.warning(f"⚠️  Requested dates ({start.date()} to {end.date()}) fall entirely in GDELT gap (2014-01-01 to 2015-02-17)")
            logger.warning(f"    No data available for this period. Will return empty DataFrame.")
            return pd.DataFrame()
        
        # Warn if request overlaps with gap
        if start <= gap_end and end >= gap_start:
            logger.warning(f"⚠️  Date range ({start.date()} to {end.date()}) overlaps with GDELT gap (2014-01-01 to 2015-02-17)")
            logger.warning(f"    No events will be available for dates in the gap period.")
        
        # Fetch GDELT 1.0 data (1979-2013)
        if start <= GDELT1_END:
            gdelt1_end = min(end, GDELT1_END)
            df1 = self.fetch_gdelt1_events(start.strftime('%Y-%m-%d'), gdelt1_end.strftime('%Y-%m-%d'), 
                                          max_events_per_month=2000, use_download=use_gdelt1_download)
            if not df1.empty:
                df1['source'] = 'gdelt1'
                # Normalize column names (GDELT 1.0 uses ActionGeo_FullName, we need 'location')
                if 'ActionGeo_FullName' in df1.columns and 'location' not in df1.columns:
                    df1['location'] = df1['ActionGeo_FullName']
                all_data.append(df1)
                min_date = df1['date'].min()
                max_date = df1['date'].max()
                min_str = min_date.date() if hasattr(min_date, 'date') else str(min_date)
                max_str = max_date.date() if hasattr(max_date, 'date') else str(max_date)
                logger.info(f"📊 GDELT 1.0: {len(df1)} events ({min_str} to {max_str})")
        
        # Fetch GDELT 2.0 data (2015+)
        if end >= GDELT2_START:
            gdelt2_start = max(start, GDELT2_START)
            df2 = self.fetch_gdelt2_events(gdelt2_start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), use_bigquery=use_bigquery)
            if not df2.empty:
                df2['source'] = 'gdelt2'
                all_data.append(df2)
                min_date = df2['date'].min()
                max_date = df2['date'].max()
                min_str = min_date.date() if hasattr(min_date, 'date') else str(min_date)
                max_str = max_date.date() if hasattr(max_date, 'date') else str(max_date)
                logger.info(f"📊 GDELT 2.0: {len(df2)} events ({min_str} to {max_str})")
        
        if not all_data:
            logger.warning("No GDELT data found")
            return pd.DataFrame()
        
        # Merge
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Total: {len(df)} global events ({start.date()} to {end.date()})")
        return df
    
    def build_event_text(self, row: pd.Series) -> str:
        """
        Build unified event text from GDELT row
        
        Concatenates: actors + event codes + location + tone
        Maximum length ~512 tokens for HF models
        """
        parts = []
        
        # Actors
        if pd.notna(row.get('Actor1Name')) and str(row['Actor1Name']).strip():
            actor1 = str(row['Actor1Name']).strip()
            if actor1 and not actor1.replace('.','').replace('-','').isdigit():  # Skip if just numbers
                parts.append(f"Actor1: {actor1}")
        
        if pd.notna(row.get('Actor2Name')) and str(row['Actor2Name']).strip():
            actor2 = str(row['Actor2Name']).strip()
            if actor2 and not actor2.replace('.','').replace('-','').isdigit():  # Skip if just numbers
                parts.append(f"Actor2: {actor2}")
        
        # Event codes (these contain useful keywords)
        if pd.notna(row.get('EventCode')):
            event_code = str(row['EventCode']).strip()
            if event_code and len(event_code) > 1:
                parts.append(f"Event: {event_code}")
        
        if pd.notna(row.get('EventBaseCode')):
            base_code = str(row['EventBaseCode']).strip()
            if base_code and len(base_code) > 1:
                parts.append(f"Category: {base_code}")
                
        if pd.notna(row.get('EventRootCode')):
            root_code = str(row['EventRootCode']).strip()
            if root_code and len(root_code) > 1:
                parts.append(f"Root: {root_code}")
        
        # Country codes
        if pd.notna(row.get('Actor1CountryCode')):
            cc1 = str(row['Actor1CountryCode']).strip()
            if cc1 and len(cc1) <= 3 and cc1.isalpha():  # Valid country code
                parts.append(f"Country1: {cc1}")
                
        if pd.notna(row.get('Actor2CountryCode')):
            cc2 = str(row['Actor2CountryCode']).strip()
            if cc2 and len(cc2) <= 3 and cc2.isalpha():  # Valid country code
                parts.append(f"Country2: {cc2}")
        
        # Location
        if pd.notna(row.get('location')):
            loc = str(row['location']).strip()
            if loc and len(loc) > 1 and not loc.replace('.','').replace('-','').isdigit():
                parts.append(f"Location: {loc}")
        
        # Tone/sentiment
        if pd.notna(row.get('AvgTone')):
            try:
                tone = float(row['AvgTone'])
                if tone < -3:
                    parts.append("negative-tone")
                elif tone > 3:
                    parts.append("positive-tone")
            except:
                pass
        
        # Goldstein scale (conflict cooperation scale)
        if pd.notna(row.get('GoldsteinScale')):
            try:
                goldstein = float(row['GoldsteinScale'])
                if goldstein < -5:
                    parts.append("high-conflict")
                elif goldstein > 5:
                    parts.append("high-cooperation")
            except:
                pass
        
        # GKG themes (GDELT 2.0 only - may not exist in cache)
        if pd.notna(row.get('themes')):
            themes = str(row['themes']).split(';')[:5]  # First 5 themes
            parts.extend([f"Theme:{t}" for t in themes if len(t) > 2])
        
        # GKG entities (GDELT 2.0 only - may not exist in cache)
        if pd.notna(row.get('persons')):
            persons = str(row['persons']).split(';')[:3]  # First 3 persons
            parts.extend([f"Person:{p}" for p in persons if len(p) > 2])
            
        if pd.notna(row.get('organizations')):
            orgs = str(row['organizations']).split(';')[:3]  # First 3 orgs
            parts.extend([f"Org:{o}" for o in orgs if len(o) > 2])
        
        # Join and truncate to 512 tokens (~2048 chars)
        text = ' '.join(parts) if parts else "global-event"
        return text[:2048]
    
    def classify_event_cluster(self, row: pd.Series) -> Dict[str, bool]:
        """
        Classify event into thematic clusters
        
        Returns:
            Dict mapping cluster names to boolean membership
        """
        # Build event text
        text = self.build_event_text(row).upper()
        
        # Check each cluster
        clusters = {}
        for cluster_name, keywords in EVENT_CLUSTERS.items():
            clusters[cluster_name] = any(keyword in text for keyword in keywords)
        
        return clusters


__all__ = ['GDELTGlobalFetcher', 'EVENT_CLUSTERS']
