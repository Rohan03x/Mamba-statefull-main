#!/usr/bin/env python3
"""
Automatic GDELT Data Fetcher - Intelligent Data Source Selection

Automatically fetches GDELT data using the optimal source:
- GDELT 2.0 (BigQuery): 2015-04-01 onwards (fast, partitioned queries)
- GDELT 1.0 (Python library): 1979-02-18 to 2013-12-31 (parallel download)

Features:
- Auto-detects date range and splits between GDELT 1.0 and 2.0
- Parallel processing for GDELT 1.0 (divides by available threads)
- Efficient BigQuery queries for GDELT 2.0
- Automatic caching and merging
- Scales across symbols and walk-forward parameters
- Thread-safe processing for multiple symbols

Author: DCF Lab Team
Created: 2025-11-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import logging
import os
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GDELT version boundaries
GDELT1_START = pd.Timestamp('1979-02-18')
GDELT1_END = pd.Timestamp('2013-12-31')
GDELT2_START = pd.Timestamp('2015-04-01')

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


class GDELTAutoFetcher:
    """
    Intelligent GDELT data fetcher that automatically selects optimal data source
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_workers: Optional[int] = None):
        """
        Initialize GDELT auto-fetcher
        
        Args:
            cache_dir: Directory for caching GDELT data (default: from cache_paths)
            max_workers: Max parallel workers (defaults to CPU count)
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Use centralized cache paths (with legacy fallback)
            try:
                from src.cache_paths import resolve_gdelt_cache_dir
                self.cache_dir = resolve_gdelt_cache_dir()
            except ImportError:
                self.cache_dir = Path("data/cache/gdelt")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect optimal worker count
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            # Use 80% of available threads for GDELT 1.0
            self.max_workers = max(1, int(cpu_count * 0.8))
        else:
            self.max_workers = max_workers
        
        logger.info(f"🚀 GDELT AutoFetcher initialized with {self.max_workers} workers")
    
    def get_company_keywords(self, symbol: str) -> List[str]:
        """Get search keywords for a company symbol"""
        symbol_upper = symbol.replace('.US', '').upper()
        
        # Symbol-specific keywords
        keyword_map = {
            'AAPL': ['APPLE', 'AAPL', 'STEVE JOBS', 'TIM COOK', 'IPHONE', 'IPAD', 'MACBOOK'],
            'MSFT': ['MICROSOFT', 'MSFT', 'SATYA NADELLA', 'BILL GATES', 'WINDOWS', 'AZURE'],
            'GOOGL': ['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOG', 'SUNDAR PICHAI', 'ANDROID'],
            'GOOG': ['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOG', 'SUNDAR PICHAI', 'ANDROID'],
            'TSLA': ['TESLA', 'TSLA', 'ELON MUSK', 'MODEL 3', 'MODEL S', 'CYBERTRUCK'],
            'NVDA': ['NVIDIA', 'NVDA', 'JENSEN HUANG', 'GEFORCE', 'CUDA'],
        }
        
        return keyword_map.get(symbol_upper, [symbol_upper])
    
    def _download_month_gdelt1(self, args: Tuple) -> Optional[pd.DataFrame]:
        """Download GDELT 1.0 data for a single month (worker function)"""
        year, month, symbol, keywords = args
        
        month_label = f"{year}-{month:02d}"
        
        try:
            # Create GDELT instance
            gd = gdelt.gdelt(version=1)
            
            # Calculate date range
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year}-{month:02d}-31"
            else:
                next_month = pd.Timestamp(f"{year}-{month:02d}-01") + pd.DateOffset(months=1)
                end_date = (next_month - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.debug(f"[{symbol}] [{month_label}] Downloading...")
            
            # Download full month
            results = gd.Search([start_date, end_date], coverage=True)
            
            if results is None or results.empty:
                return None
            
            # Filter for company mentions
            text_cols = [col for col in ['SOURCEURL', 'Actor1Name', 'Actor2Name'] 
                        if col in results.columns]
            
            if not text_cols:
                return None
            
            # Build filter mask
            mask = pd.Series([False] * len(results))
            for col in text_cols:
                for keyword in keywords:
                    mask |= results[col].astype(str).str.contains(keyword, case=False, na=False)
            
            filtered = results[mask].copy()
            
            if filtered.empty:
                return None
            
            # Extract date and headline
            filtered['date'] = pd.to_datetime(filtered['SQLDATE'], format='%Y%m%d', errors='coerce')
            
            # Combine actor names and URL as headline
            headline_parts = []
            if 'Actor1Name' in filtered.columns:
                headline_parts.append(filtered['Actor1Name'].fillna(''))
            if 'Actor2Name' in filtered.columns:
                headline_parts.append(filtered['Actor2Name'].fillna(''))
            if 'SOURCEURL' in filtered.columns:
                headline_parts.append(filtered['SOURCEURL'].fillna(''))
            
            filtered['headline'] = ' '.join([filtered[col].fillna('') for col in 
                                            [c for c in ['Actor1Name', 'Actor2Name', 'SOURCEURL'] 
                                             if c in filtered.columns]]).str.strip()
            filtered['headline'] = filtered['headline'].replace('', f'{symbol} event')
            
            # Keep only date and headline
            result_df = filtered[['date', 'headline']].copy()
            result_df = result_df.dropna(subset=['date'])
            
            logger.info(f"[{symbol}] [{month_label}] ✅ {len(result_df)} events ({100*len(result_df)/len(results):.3f}%)")
            return result_df
            
        except Exception as e:
            logger.error(f"[{symbol}] [{month_label}] ❌ Error: {e}")
            return None
    
    def fetch_gdelt1_range(self, symbol: str, start_date: pd.Timestamp, 
                           end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Fetch GDELT 1.0 data in parallel across date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with date index and headline column
        """
        if not _GDELT1_AVAIL:
            logger.error("GDELT 1.0 library not available")
            return None
        
        logger.info(f"📰 [{symbol}] Fetching GDELT 1.0 data: {start_date.date()} to {end_date.date()}")
        
        keywords = self.get_company_keywords(symbol)
        
        # Generate list of months to download
        months_list = []
        current = start_date
        while current <= end_date:
            months_list.append((current.year, current.month, symbol, keywords))
            current += pd.DateOffset(months=1)
        
        logger.info(f"📊 [{symbol}] Processing {len(months_list)} months with {self.max_workers} workers")
        
        # Download in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_month = {
                executor.submit(self._download_month_gdelt1, args): f"{args[0]}-{args[1]:02d}"
                for args in months_list
            }
            
            for future in as_completed(future_to_month):
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    month_label = future_to_month[future]
                    logger.error(f"[{symbol}] [{month_label}] Exception: {e}")
        
        if not all_results:
            logger.warning(f"[{symbol}] No GDELT 1.0 data found")
            return None
        
        # Combine and aggregate
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Group by date
        daily = combined_df.groupby('date')['headline'].apply(
            lambda s: '. '.join([str(x)[:200] for x in s if x])
        ).to_frame('headline')
        
        logger.info(f"✅ [{symbol}] GDELT 1.0: {len(daily)} days with news")
        return daily
    
    def fetch_gdelt2_range(self, symbol: str, start_date: pd.Timestamp,
                           end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Fetch GDELT 2.0 data via BigQuery
        
        Args:
            symbol: Stock symbol
            start_date: Start date (must be >= 2015-04-01)
            end_date: End date
            
        Returns:
            DataFrame with date index and headline column
        """
        if not _GDELT2_AVAIL:
            logger.error("BigQuery not available")
            return None
        
        logger.info(f"📰 [{symbol}] Fetching GDELT 2.0 data: {start_date.date()} to {end_date.date()}")
        
        try:
            client = bigquery.Client()
            
            keywords = self.get_company_keywords(symbol)
            search_pattern = '|'.join(keywords)
            
            # Optimized BigQuery query with partitioning
            query = f"""
            SELECT 
                DATE(_PARTITIONTIME) as date,
                V2Themes as themes
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`
            WHERE _PARTITIONTIME >= TIMESTAMP('{start_date.strftime('%Y-%m-%d')}')
              AND _PARTITIONTIME < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
              AND REGEXP_CONTAINS(V2Themes, r'(?i)({search_pattern})')
            LIMIT 50000
            """
            
            logger.info(f"   [{symbol}] Querying BigQuery (partitioned)...")
            
            query_job = client.query(query)
            df = query_job.to_dataframe()
            
            # Show cost estimate
            bytes_processed = query_job.total_bytes_processed or 0
            gb_processed = bytes_processed / (1024**3)
            estimated_cost = gb_processed * 0.005
            logger.info(f"   [{symbol}] Scanned: {gb_processed:.2f} GB (${estimated_cost:.4f})")
            
            if df is None or df.empty:
                logger.warning(f"[{symbol}] No GDELT 2.0 data found")
                return None
            
            # Process themes as headlines
            df['headline'] = df['themes'].fillna('')
            df = df[['date', 'headline']].copy()
            df = df.dropna(subset=['date'])
            
            # Group by date
            daily = df.groupby('date')['headline'].apply(
                lambda s: '. '.join([str(x)[:200] for x in s if x])
            ).to_frame('headline')
            
            logger.info(f"✅ [{symbol}] GDELT 2.0: {len(daily)} days with news")
            return daily
            
        except Exception as e:
            logger.error(f"❌ [{symbol}] GDELT 2.0 fetch failed: {e}")
            return None
    
    def fetch(self, symbol: str, start_date: str, end_date: str,
              force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Automatically fetch GDELT data using optimal sources
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Force re-download even if cached
            
        Returns:
            DataFrame with date index and headline column, or None
        """
        # Parse dates
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        cache_file = self.cache_dir / f"{symbol.upper()}.US_gdelt_news.parquet"
        
        # Check cache first
        if cache_file.exists() and not force_refresh:
            try:
                cached_df = pd.read_parquet(cache_file)
                cached_df.index = pd.to_datetime(cached_df.index)
                
                # Filter by requested range (compare Timestamp to Timestamp)
                mask = (cached_df.index >= start_dt) & (cached_df.index <= end_dt)
                filtered = cached_df[mask]
                
                if not filtered.empty:
                    logger.info(f"🎯 [{symbol}] CACHE HIT: {len(filtered)} days loaded")
                    return filtered
                else:
                    logger.info(f"📦 [{symbol}] Cache exists but empty for range")
            except Exception as e:
                logger.warning(f"[{symbol}] Cache read failed: {e}, re-fetching")
        
        logger.info(f"🚀 [{symbol}] Auto-fetching GDELT data: {start_dt.date()} to {end_dt.date()}")
        
        # Determine which GDELT versions to use
        gdelt1_df = None
        gdelt2_df = None
        
        # GDELT 1.0 range (1979-02-18 to 2013-12-31)
        if start_dt <= GDELT1_END:
            gdelt1_start = max(start_dt, GDELT1_START)
            gdelt1_end = min(end_dt, GDELT1_END)
            
            logger.info(f"📅 [{symbol}] GDELT 1.0 range: {gdelt1_start.date()} to {gdelt1_end.date()}")
            gdelt1_df = self.fetch_gdelt1_range(symbol, gdelt1_start, gdelt1_end)
        
        # GDELT 2.0 range (2015-04-01 onwards)
        if end_dt >= GDELT2_START:
            gdelt2_start = max(start_dt, GDELT2_START)
            gdelt2_end = end_dt
            
            logger.info(f"📅 [{symbol}] GDELT 2.0 range: {gdelt2_start.date()} to {gdelt2_end.date()}")
            gdelt2_df = self.fetch_gdelt2_range(symbol, gdelt2_start, gdelt2_end)
        
        # Merge results
        results = []
        if gdelt1_df is not None:
            results.append(gdelt1_df)
        if gdelt2_df is not None:
            results.append(gdelt2_df)
        
        if not results:
            logger.error(f"❌ [{symbol}] No data retrieved from any source")
            return None
        
        # Combine and deduplicate
        combined_df = pd.concat(results)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        logger.info(f"✅ [{symbol}] Combined: {len(combined_df)} days total")
        
        # Save to cache
        try:
            combined_df.to_parquet(cache_file)
            logger.info(f"💾 [{symbol}] Saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"[{symbol}] Cache save failed: {e}")
        
        return combined_df


def fetch_gdelt_for_walkforward(symbol: str, wf_start: str, wf_end: str,
                                 max_workers: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch GDELT data for walk-forward parameters
    
    Args:
        symbol: Stock symbol
        wf_start: Walk-forward start date (e.g., '2010-01-01')
        wf_end: Walk-forward end date (e.g., '2025-07-23')
        max_workers: Max parallel workers (auto-detected if None)
        
    Returns:
        DataFrame with GDELT news data
    """
    fetcher = GDELTAutoFetcher(max_workers=max_workers)
    return fetcher.fetch(symbol, wf_start, wf_end)


if __name__ == '__main__':
    # Demo: Fetch GDELT data for AAPL covering walk-forward period
    print("=" * 80)
    print("GDELT AUTO-FETCHER DEMO")
    print("=" * 80)
    print()
    
    # Example: Walk-forward from 2010 to 2025
    symbol = 'AAPL'
    wf_start = '2010-01-01'
    wf_end = '2025-07-23'
    
    print(f"Symbol: {symbol}")
    print(f"Walk-Forward Range: {wf_start} to {wf_end}")
    print()
    
    df = fetch_gdelt_for_walkforward(symbol, wf_start, wf_end)
    
    if df is not None:
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Total days with news: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print()
        print("Sample:")
        print(df.head(10))
        print()
        print("✅ SUCCESS!")
    else:
        print()
        print("❌ No data retrieved")
    
    print()
    print("=" * 80)
