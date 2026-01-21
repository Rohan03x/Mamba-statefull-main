"""
FINRA Short Interest Signal Generator

This module implements short interest tracking and squeeze probability analysis
using FINRA short interest data to generate contrarian trading signals.

Key Features:
- Track short interest changes from FINRA data
- Calculate short squeeze probability metrics
- Generate contrarian signals based on short positioning
- Identify crowded shorts and potential reversal points
- Integration with market microstructure data

Data Sources:
- FINRA Short Interest reporting
- Real-time short borrowing rates
- Options flow for gamma squeeze detection
- Market maker inventory signals
"""

import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import cache for historical storage
try:
    from src.dcf_lab.short_interest_cache import get_short_interest_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Short interest cache not available")


@dataclass
@dataclass
class ShortInterestMetrics:
    """Short interest metrics for a security"""
    ticker: str
    report_date: str
    total_short_interest: int
    days_to_cover: float
    short_ratio: float  # Short interest / float
    change_from_prior: float  # % change from previous report
    borrowing_rate: Optional[float] = None
    utilization_rate: Optional[float] = None
    squeeze_probability: float = 0.0
    signal_strength: float = 0.0
    shares_short_prior_month: Optional[int] = None  # For EODHD 1-month change


@dataclass
class ShortSqueezeSignal:
    """Short squeeze signal with timing and strength"""
    ticker: str
    signal_date: str
    signal_type: str  # squeeze_risk, squeeze_setup, squeeze_active
    probability: float
    strength: float  # 0-1 signal strength
    drivers: List[str]  # What's driving the signal
    timeframe: str  # short_term, medium_term, long_term
    risk_level: str  # low, medium, high
    
    
class ShortInterestAnalyzer:
    """
    FINRA Short Interest Analysis System
    
    Tracks short interest changes and generates squeeze probability signals
    for contrarian trading strategies and risk management.
    """
    
    def __init__(self, finra_username: Optional[str] = None,
                 finra_password: Optional[str] = None):
        self.finra_username = finra_username
        self.finra_password = finra_password
        
        # FINRA data endpoints
        self.finra_base_url = "https://www.finra.org/finra-data"
        self.short_interest_url = (f"{self.finra_base_url}/"
                                   "browse-catalog/short-sale-volume-data")

        # Alternative data sources
        self.iex_base_url = "https://cloud.iexapis.com/stable"
        self.polygon_base_url = "https://api.polygon.io"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DCF Lab Research Tool/1.0 (research@dcflab.com)',
            'Accept': 'application/json'
        })

        # Short squeeze thresholds
        self.squeeze_thresholds = {
            'high_short_ratio': 0.20,  # 20%+ short interest
            'extreme_short_ratio': 0.30,  # 30%+ extreme
            'high_days_to_cover': 5.0,  # 5+ days to cover
            'extreme_days_to_cover': 10.0,  # 10+ days extreme
            'high_borrow_rate': 0.05,  # 5%+ borrow rate
            'extreme_borrow_rate': 0.15,  # 15%+ extreme borrow rate
            'high_utilization': 0.80,  # 80%+ utilization
            'extreme_utilization': 0.95  # 95%+ extreme
        }

    def _try_real_short_data_sources(self, ticker: str, 
                                   lookback_days: int) -> Optional[List[Dict[str, Any]]]:
        """
        Attempt to fetch real short interest data from available APIs
        
        This method tries multiple real data sources in order:
        1. SEC EDGAR API for official filings
        2. Financial data APIs (Alpha Vantage, IEX, etc.)
        3. Web scraping of FINRA public data
        
        Returns None if no real data is available.
        """
        # Try SEC EDGAR API for official short interest reports
        try:
            edgar_data = self._get_edgar_short_data(ticker, lookback_days)
            if edgar_data:
                logger.info(f"Retrieved real data from SEC EDGAR for {ticker}")
                return edgar_data
        except Exception as e:
            logger.debug(f"SEC EDGAR failed: {e}")
        
        # Try financial data APIs
        try:
            api_data = self._get_api_short_data(ticker, lookback_days)
            if api_data:
                logger.info(f"Retrieved real data from financial APIs for {ticker}")
                return api_data
        except Exception as e:
            logger.debug(f"Financial APIs failed: {e}")
        
        # Try web scraping FINRA public data
        try:
            scraped_data = self._get_finra_web_data(ticker, lookback_days)
            if scraped_data:
                logger.info(f"Retrieved real data from FINRA web for {ticker}")
                return scraped_data
        except Exception as e:
            logger.debug(f"FINRA web scraping failed: {e}")
        
        logger.info(f"All real data sources failed for {ticker}, using simulation")
        return None
    
    def _get_edgar_short_data(self, ticker: str, 
                            lookback_days: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch from SEC EDGAR API - real implementation placeholder"""
        # Real implementation would use SEC's official API
        # For now, return None to trigger fallback
        return None
    
    def _get_api_short_data(self, ticker: str, 
                          lookback_days: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch from financial data APIs - real implementation placeholder"""
        # Real implementation would use APIs like:
        # - Alpha Vantage fundamental data
        # - IEX Cloud short interest
        # - Polygon.io institutional data
        # For now, return None to trigger fallback
        return None
    
    def _get_finra_web_data(self, ticker: str, 
                          lookback_days: int) -> Optional[List[Dict[str, Any]]]:
        """Fetch from FINRA website - real implementation placeholder"""
        # Real implementation would scrape FINRA's public short interest reports
        # For now, return None to trigger fallback
        return None

    def get_short_interest_data(
            self, ticker: str,
            lookback_days: int = 90) -> List[ShortInterestMetrics]:
        """
        Get short interest data for a ticker
        
        Data sources (priority order):
        1. NASDAQ API (primary - 12+ months historical, auto-cached)
        2. EODHD fundamentals API (fallback - current + prior month)
        3. FINRA provider (last resort)
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: How many days of history to retrieve
            
        Returns:
            List of ShortInterestMetrics ordered by date (newest first)
        """
        try:
            logger.info(f"🔍 Fetching short interest data for {ticker}")
            
            # PRIORITY 1: Try NASDAQ API (has 12+ months of historical data)
            short_data = self._get_nasdaq_short_data(ticker)
            
            if short_data:
                logger.info(f"✅ Using NASDAQ short interest data for {ticker} ({len(short_data)} reports)")
            else:
                # PRIORITY 2: Try EODHD fundamentals API (has current + prior month)
                short_data = self._get_eodhd_short_data(ticker)
                
                if short_data:
                    logger.info(f"✅ Using EODHD short interest data for {ticker}")
                else:
                    # PRIORITY 3: Try Yahoo Finance (yfinance)
                    short_data = self._get_yahoo_short_data(ticker)
                    
                    if short_data:
                        logger.info(f"✅ Using Yahoo Finance short interest data for {ticker}")
                    else:
                        # PRIORITY 4: Try Finviz scraper (free, no API key)
                        short_data = self._get_finviz_short_data(ticker)
                        
                        if short_data:
                            logger.info(f"✅ Using Finviz short interest data for {ticker}")
                        else:
                            # PRIORITY 5: Try FINRA provider (last resort)
                            try:
                                from .data.finra_short_interest import FINRAShortInterestProvider
                                finra_provider = FINRAShortInterestProvider()
                                real_data = finra_provider.get_short_interest_data(ticker)
                                
                                if real_data and real_data.get('data_source') == 'FINRA_REAL':
                                    logger.info(f"✅ Using real FINRA data for {ticker}")
                                    converted = self._convert_finra_data(real_data, ticker)
                                    short_data = [converted] if converted else None
                                else:
                                    logger.warning(f"No valid FINRA data for {ticker}, skipping fallback")
                                    short_data = None
                                    
                            except ImportError:
                                logger.warning("FINRA provider not available")
                                short_data = None

            if not short_data:
                logger.warning(f"No short interest data found for {ticker}")
                return []

            # Calculate derived metrics
            metrics_list = []
            for i, data in enumerate(short_data):
                metrics = self._calculate_short_metrics(data, short_data, i)
                if metrics:
                    metrics_list.append(metrics)

            count = len(metrics_list)
            logger.info(f"📊 Retrieved {count} short interest reports")
            return metrics_list
            
        except Exception as e:
            logger.error(f"❌ Error fetching short interest data: {e}")
            return []
    
    def _get_nasdaq_short_data(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch historical short interest from NASDAQ API with auto-caching
        
        NASDAQ provides official FINRA short interest reports with:
        - 12-24 months of historical data
        - Bi-monthly settlement dates (15th and end of month)
        - Settlement date, short interest, avg daily volume, days to cover
        
        AUTOMATICALLY CACHES all historical snapshots for zscore calculation
        
        Returns:
            List of dicts with short interest data (newest first), or None if unavailable
        """
        try:
            from src.data_sources.nasdaq_short_interest_provider import get_nasdaq_short_interest_provider
            
            logger.debug(f"🔑 Attempting NASDAQ API for short interest: {ticker}")
            
            nasdaq = get_nasdaq_short_interest_provider()
            history_df = nasdaq.get_short_interest_history(ticker)
            
            if history_df is None or len(history_df) == 0:
                logger.debug(f"No short interest data from NASDAQ for {ticker}")
                return None
            
            # Get ticker info for float calculation
            shares_float = None
            shares_outstanding = None
            try:
                from src.data_sources.eodhd_provider import get_eodhd_provider
                eodhd = get_eodhd_provider()
                fundamentals = eodhd.get_fundamentals(ticker)
                shares_stats = fundamentals.get('SharesStats', {})
                shares_float = shares_stats.get('SharesFloat')
                shares_outstanding = shares_stats.get('SharesOutstanding')
            except Exception:
                # Estimate if not available
                shares_outstanding = 1000000000  # Default 1B
                shares_float = int(shares_outstanding * 0.8)  # Default 80%
            
            # Convert DataFrame to list of dicts
            short_data_list = []
            
            for _, row in history_df.iterrows():
                settlement_date = row['settlement_date']
                short_interest = row['short_interest']
                avg_volume = row['avg_daily_volume']
                days_to_cover = row['days_to_cover']
                
                # Calculate short % of float
                short_pct_float = short_interest / shares_float if shares_float else 0.05
                
                short_data = {
                    'ticker': ticker,
                    'report_date': settlement_date.strftime('%Y-%m-%d'),
                    'total_short_interest': short_interest,
                    'shares_outstanding': shares_outstanding,
                    'float_shares': shares_float,
                    'avg_daily_volume': avg_volume,
                    'short_ratio': short_pct_float,
                    'short_pct_float': short_pct_float,
                    'short_pct_outstanding': short_interest / shares_outstanding if shares_outstanding else 0.01,
                    'shares_short_prior_month': None,  # Not applicable for NASDAQ
                    'source': 'NASDAQ',
                    'data_quality': 'HIGH'
                }
                
                short_data_list.append(short_data)
                
                # ✅ AUTO-CACHE each historical snapshot
                if CACHE_AVAILABLE:
                    try:
                        cache = get_short_interest_cache()
                        cache.save_snapshot(ticker, {
                            'date': settlement_date,
                            'shares_short': short_interest,
                            'short_pct_float': short_pct_float,
                            'days_to_cover': days_to_cover,
                            'source': 'NASDAQ'
                        })
                    except Exception as e:
                        logger.debug(f"Cache save failed (non-critical): {e}")
            
            # Report caching status
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    total_cached = cache.get_count(ticker)
                    logger.info(f"💾 Auto-cached {len(short_data_list)} NASDAQ snapshots ({total_cached} total in cache for {ticker})")
                except Exception:
                    pass
            
            logger.info(f"✅ NASDAQ: Fetched {len(short_data_list)} short interest reports for {ticker} (quality=HIGH)")
            return short_data_list
            
        except ImportError:
            logger.debug(f"NASDAQ provider not available for {ticker}")
            return None
        except Exception as e:
            logger.debug(f"NASDAQ short interest fetch failed for {ticker}: {e}")
            return None
    
    def _get_eodhd_short_data(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch short interest data from EODHD fundamentals API
        
        EODHD provides these short interest fields in Technicals section:
        - SharesShort: Total shares sold short (current)
        - SharesShortPriorMonth: Prior month short interest
        - ShortRatio: Days to cover ratio
        - ShortPercent: Short % of outstanding shares
        - ShortPercentFloat (in SharesStats): Short % of float
        
        Returns:
            List with single dict containing latest short interest data, or None if unavailable
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 Attempting EODHD provider for short interest: {ticker}")
            
            eodhd = get_eodhd_provider()
            fundamentals = eodhd.get_fundamentals(ticker)
            
            if not fundamentals:
                logger.debug(f"No fundamentals data from EODHD for {ticker}")
                return None
            
            # Extract short interest from Technicals section (primary) and SharesStats (supplemental)
            shares_stats = fundamentals.get('SharesStats', {})
            technicals = fundamentals.get('Technicals', {})
            
            # Get short interest fields (Technicals has most complete data)
            shares_short = technicals.get('SharesShort') or shares_stats.get('SharesShort')
            shares_short_prior = technicals.get('SharesShortPriorMonth') or shares_stats.get('SharesShortPriorMonth')
            short_ratio = technicals.get('ShortRatio') or shares_stats.get('ShortRatio')
            short_pct_out = technicals.get('ShortPercent') or shares_stats.get('ShortPercentOutstanding')
            short_pct_float = shares_stats.get('ShortPercentFloat') or technicals.get('ShortPercentFloat')
            
            # Need at least shares_short to proceed
            if not shares_short or shares_short == 0:
                logger.debug(f"No valid SharesShort in EODHD data for {ticker}")
                return None
            
            # Get shares outstanding and float for calculations
            shares_outstanding = shares_stats.get('SharesOutstanding', 1000000000)  # Default 1B
            shares_float = shares_stats.get('SharesFloat', int(shares_outstanding * 0.8))  # Default 80%
            
            # Estimate average daily volume from price history if needed
            avg_daily_volume = 25000000  # Default fallback
            try:
                price_history = eodhd.get_eod_prices(ticker, lookback_days=30)
                if price_history is not None and 'volume' in price_history.columns:
                    avg_daily_volume = int(price_history['volume'].mean())
            except Exception:
                pass
            
            # Calculate missing fields from available data
            if shares_short is None and short_pct_float is not None and shares_float:
                shares_short = int(shares_float * short_pct_float)
            
            if short_ratio is None and shares_short and avg_daily_volume > 0:
                short_ratio = shares_short / avg_daily_volume
            
            # Build short interest data dict
            short_data = {
                'ticker': ticker,
                'report_date': fundamentals.get('General', {}).get('UpdatedAt', 
                                                                   datetime.now().strftime('%Y-%m-%d')),
                'total_short_interest': shares_short or int(shares_float * (short_pct_float or 0.05)),
                'shares_outstanding': shares_outstanding,
                'float_shares': shares_float,
                'avg_daily_volume': avg_daily_volume,
                'short_ratio': short_ratio,
                'short_pct_float': short_pct_float,
                'short_pct_outstanding': short_pct_out,
                'shares_short_prior_month': shares_short_prior,
                'source': 'EODHD',
                'data_quality': 'HIGH' if shares_short else 'ESTIMATED'
            }
            
            # ✅ SAVE TO LOCAL CACHE for historical accumulation
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    cache.save_snapshot(ticker, {
                        'date': short_data['report_date'],
                        'shares_short': short_data['total_short_interest'],
                        'short_pct_float': short_data['short_pct_float'],
                        'days_to_cover': short_data['short_ratio'],
                        'source': 'EODHD'
                    })
                    count = cache.get_count(ticker)
                    logger.debug(f"💾 Cached short interest snapshot ({count} total for {ticker})")
                except Exception as e:
                    logger.debug(f"Cache save failed (non-critical): {e}")
            
            logger.info(f"✅ EODHD: Fetched short interest for {ticker} (quality={short_data['data_quality']})")
            return [short_data]
            
        except Exception as e:
            logger.debug(f"EODHD short interest fetch failed for {ticker}: {e}")
            return None
    
    def _get_yahoo_short_data(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch short interest data from Yahoo Finance via yfinance
        
        Yahoo Finance provides:
        - sharesShort: Total shares sold short (current)
        - sharesShortPriorMonth: Prior month short interest
        - shortRatio: Days to cover ratio
        - shortPercentOfFloat: Short % of float
        - dateShortInterest: Date of short interest data (timestamp)
        
        Returns:
            List with single dict containing latest short interest data, or None if unavailable
        """
        try:
            import yfinance as yf
            
            logger.debug(f"🔑 Attempting Yahoo Finance for short interest: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract short interest fields
            shares_short = info.get('sharesShort')
            shares_short_prior = info.get('sharesShortPriorMonth')
            short_ratio = info.get('shortRatio')
            short_pct_float = info.get('shortPercentOfFloat')
            date_short_interest = info.get('dateShortInterest')
            
            # Need at least current shares short
            if not shares_short or shares_short == 0:
                logger.debug(f"No sharesShort in Yahoo Finance data for {ticker}")
                return None
            
            # Get additional info
            shares_outstanding = info.get('sharesOutstanding', 1000000000)
            float_shares = info.get('floatShares') or info.get('impliedSharesOutstanding')
            if not float_shares:
                float_shares = int(shares_outstanding * 0.8)
            
            avg_daily_volume = info.get('averageVolume') or info.get('averageVolume10days', 25000000)
            
            # Convert timestamp to date
            if date_short_interest:
                from datetime import datetime as dt
                report_date = dt.fromtimestamp(date_short_interest).strftime('%Y-%m-%d')
            else:
                report_date = datetime.now().strftime('%Y-%m-%d')
            
            # Calculate missing fields
            if short_pct_float is None and float_shares > 0:
                short_pct_float = shares_short / float_shares
            
            if short_ratio is None and avg_daily_volume > 0:
                short_ratio = shares_short / avg_daily_volume
            
            short_pct_out = shares_short / shares_outstanding if shares_outstanding > 0 else short_pct_float
            
            # Build short interest data dict
            short_data = {
                'ticker': ticker,
                'report_date': report_date,
                'total_short_interest': shares_short,
                'shares_outstanding': shares_outstanding,
                'float_shares': float_shares,
                'avg_daily_volume': avg_daily_volume,
                'short_ratio': short_ratio,
                'short_pct_float': short_pct_float,
                'short_pct_outstanding': short_pct_out,
                'shares_short_prior_month': shares_short_prior,
                'source': 'YAHOO',
                'data_quality': 'HIGH'
            }
            
            # ✅ SAVE TO LOCAL CACHE for historical accumulation
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    cache.save_snapshot(ticker, {
                        'date': report_date,
                        'shares_short': shares_short,
                        'short_pct_float': short_pct_float,
                        'days_to_cover': short_ratio,
                        'source': 'YAHOO'
                    })
                    count = cache.get_count(ticker)
                    logger.debug(f"💾 Cached Yahoo short interest snapshot ({count} total for {ticker})")
                except Exception as e:
                    logger.debug(f"Cache save failed (non-critical): {e}")
            
            logger.info(f"✅ YAHOO: Fetched short interest for {ticker} (quality=HIGH)")
            return [short_data]
            
        except ImportError:
            logger.debug("yfinance not installed")
            return None
        except Exception as e:
            logger.debug(f"Yahoo Finance short interest fetch failed for {ticker}: {e}")
            return None
    
    def _get_finviz_short_data(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch short interest data from Finviz.com web scraping (free, no API key)
        
        Finviz provides:
        - Short Float: % of float sold short
        - Short Ratio: Days to cover
        - Short Interest: Total shares short
        
        Returns:
            List with single dict containing latest short interest data, or None if unavailable
        """
        try:
            from src.data_sources.finviz_short_interest import get_finviz_short_interest_scraper
            
            logger.debug(f"🔑 Attempting Finviz scraper for short interest: {ticker}")
            
            scraper = get_finviz_short_interest_scraper()
            finviz_data = scraper.get_short_interest(ticker)
            
            if not finviz_data:
                logger.debug(f"No short interest from Finviz for {ticker}")
                return None
            
            # Get additional data from EODHD for shares outstanding/float
            shares_outstanding = None
            float_shares = None
            avg_daily_volume = None
            
            try:
                from src.data_sources.eodhd_provider import get_eodhd_provider
                eodhd = get_eodhd_provider()
                fundamentals = eodhd.get_fundamentals(ticker)
                
                if fundamentals:
                    shares_stats = fundamentals.get('SharesStats', {})
                    shares_outstanding = shares_stats.get('SharesOutstanding')
                    float_shares = shares_stats.get('SharesFloat')
                    
                    # Get avg volume from price history
                    price_history = eodhd.get_eod_prices(ticker, lookback_days=30)
                    if price_history is not None and 'volume' in price_history.columns:
                        avg_daily_volume = int(price_history['volume'].mean())
            except Exception:
                pass
            
            # Use defaults if EODHD unavailable
            if not shares_outstanding:
                shares_outstanding = 1000000000  # Default 1B
            if not float_shares:
                float_shares = int(shares_outstanding * 0.8)  # Default 80%
            if not avg_daily_volume:
                avg_daily_volume = 25000000  # Default 25M
            
            # Extract Finviz data
            short_pct_float = finviz_data.get('short_float_pct', 0.0)
            short_ratio_dtc = finviz_data.get('short_ratio', 0.0)
            shares_short = finviz_data.get('short_interest_shares')
            
            # Calculate shares short from % if not provided
            if not shares_short and short_pct_float > 0:
                shares_short = int(float_shares * short_pct_float)
            
            # Calculate days to cover from shares if not provided
            if short_ratio_dtc == 0.0 and shares_short and avg_daily_volume > 0:
                short_ratio_dtc = shares_short / avg_daily_volume
            
            # Build short interest data dict
            short_data = {
                'ticker': ticker,
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'total_short_interest': shares_short or int(float_shares * short_pct_float),
                'shares_outstanding': shares_outstanding,
                'float_shares': float_shares,
                'avg_daily_volume': avg_daily_volume,
                'short_ratio': short_ratio_dtc,
                'short_pct_float': short_pct_float,
                'short_pct_outstanding': (shares_short / shares_outstanding) if shares_short and shares_outstanding else short_pct_float * 0.8,
                'shares_short_prior_month': None,  # Finviz doesn't provide historical
                'source': 'FINVIZ',
                'data_quality': 'MEDIUM'  # Web scraping, current snapshot only
            }
            
            # ✅ SAVE TO LOCAL CACHE for historical accumulation
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    cache.save_snapshot(ticker, {
                        'date': short_data['report_date'],
                        'shares_short': short_data['total_short_interest'],
                        'short_pct_float': short_pct_float,
                        'days_to_cover': short_ratio_dtc,
                        'source': 'FINVIZ'
                    })
                    count = cache.get_count(ticker)
                    logger.debug(f"💾 Cached Finviz short interest snapshot ({count} total for {ticker})")
                except Exception as e:
                    logger.debug(f"Cache save failed (non-critical): {e}")
            
            logger.info(f"✅ FINVIZ: Fetched short interest for {ticker} (quality=MEDIUM)")
            return [short_data]
            
        except ImportError as e:
            logger.debug(f"Finviz scraper not available: {e}")
            return None
        except Exception as e:
            logger.debug(f"Finviz short interest fetch failed for {ticker}: {e}")
            return None
    
    def _get_finra_short_data(self, ticker: str,
                              lookback_days: int) -> List[Dict[str, Any]]:
        """Get short interest data from FINRA and real sources"""
        try:
            # ===== IMPLEMENT REAL SHORT-INTEREST SIGNALS =====
            
            # First try real data sources
            real_data = self._try_real_short_data_sources(ticker, lookback_days)
            if real_data:
                logger.info("✅ Retrieved real short interest data from live sources")
                return real_data
            
            # MOCK DATA DISABLED
            logger.error(f"Mock short interest data disabled for {ticker} - implement real FINRA API")
            return []
            
            if False:  # Disabled mock path
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                mock_data = []
            current_date = start_date
            
            # Get market context for more realistic simulation
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 500000000000)  # Default 500B
                shares_outstanding = info.get('sharesOutstanding', 1000000000)  
                float_shares = info.get('floatShares', int(shares_outstanding * 0.8))
                avg_volume = info.get('averageVolume', 25000000)
                
                # Market-informed short interest estimation
                cap_tier = 'large' if market_cap > 50e9 else ('mid' if market_cap > 2e9 else 'small')
                base_short_ratios = {'large': 0.03, 'mid': 0.05, 'small': 0.08}  # Historical averages
                base_short_ratio = base_short_ratios.get(cap_tier, 0.05)
                
            except Exception as e:
                logger.warning(f"Market data unavailable, using defaults: {e}")
                shares_outstanding = 1000000000
                float_shares = 800000000  
                avg_volume = 25000000
                base_short_ratio = 0.05
            
            while current_date <= end_date:
                # FINRA reports twice monthly (15th and end of month)
                if current_date.day in [15, 28, 29, 30, 31]:
                    # Market-informed variation instead of pure random
                    rng = np.random.default_rng(int(current_date.timestamp()) % 2**32)
                    market_stress = 0.1 * np.sin(current_date.timetuple().tm_yday / 365 * 2 * np.pi)
                    variation = rng.normal(market_stress, 0.08)  # Market-informed volatility
                    
                    short_ratio = max(0.01, base_short_ratio * (1 + variation))
                    short_interest = int(float_shares * short_ratio)
                    
                    mock_data.append({
                        'ticker': ticker,
                        'report_date': current_date.strftime('%Y-%m-%d'),
                        'total_short_interest': short_interest,
                        'shares_outstanding': shares_outstanding,
                        'float_shares': float_shares,
                        'avg_daily_volume': avg_volume,
                        'source': 'enhanced_simulation'
                    })
                
                current_date += timedelta(days=1)
            
            # Sort by date (newest first)
            mock_data.sort(key=lambda x: x['report_date'], reverse=True)
            
            logger.info(f"📋 Enhanced short data: {len(mock_data)} reports for {ticker}")
            return mock_data
            
        except Exception as e:
            logger.warning(f"Error fetching FINRA data: {e}")
            return []
    
    def _get_alternative_short_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Get short data from alternative sources when FINRA unavailable"""
        # MOCK DATA DISABLED
        logger.error("Mock alternative short data disabled")
        return []
        
        try:
            if False:  # Disabled
                end_date = datetime.now()
                mock_data = [{
                'ticker': ticker,
                'report_date': end_date.strftime('%Y-%m-%d'),
                'total_short_interest': 45000000,
                'shares_outstanding': 1000000000,
                'float_shares': 800000000,
                'avg_daily_volume': 25000000,
                'borrow_rate': 0.02,  # 2% borrow rate
                'utilization_rate': 0.65  # 65% utilization
            }]

            logger.info("📊 Using alternative short interest data")
            return mock_data

        except Exception as e:
            logger.warning(f"Error fetching alternative short data: {e}")
            return []
    
    def _convert_finra_data(self, finra_data: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
        """
        Convert FINRA provider data to internal format
        
        Returns None if data is invalid/missing to avoid constant fallback values
        """
        try:
            # Reject invalid/empty FINRA data
            if not finra_data or finra_data.get('data_source') not in ['FINRA_REAL', 'FINRA_ESTIMATED']:
                logger.warning(f"Rejecting invalid FINRA data for {ticker} (source={finra_data.get('data_source')})")
                return None
            
            return {
                'ticker': ticker,
                'report_date': finra_data.get('last_updated', datetime.now().strftime('%Y-%m-%d')),
                'total_short_interest': finra_data.get('short_volume', 50000),
                'float_shares': finra_data.get('total_volume', 1000000) * 10,  # Estimate float
                'avg_daily_volume': finra_data.get('avg_daily_volume', 1000000),
                'short_ratio': finra_data.get('short_ratio', 0.05),
                'days_to_cover': finra_data.get('days_to_cover', 2.0),
                'data_source': finra_data.get('data_source', 'UNKNOWN'),
                'quality_score': finra_data.get('quality_score', 0.5)
            }
        except Exception as e:
            logger.warning(f"Error converting FINRA data for {ticker}: {e}")
            # DO NOT return constant fallback - return None to skip family
            return None
    
    def _calculate_short_metrics(
            self, data: Dict[str, Any],
            historical_data: List[Dict[str, Any]],
            index: int) -> Optional[ShortInterestMetrics]:
        """Calculate short interest metrics from raw data"""
        try:
            # Basic calculations
            short_interest = data['total_short_interest']
            float_shares = data['float_shares']
            avg_daily_volume = data['avg_daily_volume']

            if float_shares > 0:
                short_ratio = short_interest / float_shares
            else:
                short_ratio = 0

            if avg_daily_volume > 0:
                days_to_cover = short_interest / avg_daily_volume
            else:
                days_to_cover = 0

            # Calculate change from prior report
            change_from_prior = 0.0
            if index + 1 < len(historical_data):
                prior_short = historical_data[index + 1][
                    'total_short_interest']
                if prior_short > 0:
                    change_from_prior = ((short_interest - prior_short) /
                                         prior_short)
            
            # Calculate squeeze probability
            squeeze_prob = self._calculate_squeeze_probability(data)
            
            metrics = ShortInterestMetrics(
                ticker=data['ticker'],
                report_date=data['report_date'],
                total_short_interest=short_interest,
                days_to_cover=days_to_cover,
                short_ratio=short_ratio,
                change_from_prior=change_from_prior,
                borrowing_rate=data.get('borrow_rate'),
                utilization_rate=data.get('utilization_rate'),
                squeeze_probability=squeeze_prob,
                shares_short_prior_month=data.get('shares_short_prior_month')  # From EODHD
            )
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating short metrics: {e}")
            return None
    
    def _get_total_options_oi(self, ticker: str) -> Optional[int]:
        """
        Get total options open interest (calls + puts) from EODHD
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Total options open interest, or None if unavailable
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"Fetching options OI for {ticker}")
            
            eodhd = get_eodhd_provider()
            if not eodhd or not eodhd.api_key:
                logger.debug("EODHD API key not configured")
                return None
            
            # Get options chain
            options_df = eodhd.get_options(ticker)
            
            if options_df is None or options_df.empty:
                logger.debug(f"No options data for {ticker}")
                return None
            
            # Sum open interest across all strikes and expirations
            if 'openInterest' in options_df.columns:
                total_oi = int(options_df['openInterest'].sum())
                logger.debug(f"✅ Total options OI for {ticker}: {total_oi:,}")
                return total_oi
            else:
                logger.debug("No openInterest column in options data")
                return None
                
        except Exception as e:
            logger.debug(f"Error fetching options OI for {ticker}: {e}")
            return None
    
    def _calculate_squeeze_probability(self, data: Dict[str, Any]) -> float:
        """Calculate short squeeze probability based on multiple factors"""
        try:
            probability = 0.0
            
            # Factor 1: Short ratio (40% weight)
            short_ratio = data['total_short_interest'] / data['float_shares']
            if short_ratio >= self.squeeze_thresholds['extreme_short_ratio']:
                probability += 0.4
            elif short_ratio >= self.squeeze_thresholds['high_short_ratio']:
                probability += 0.25
            
            # Factor 2: Days to cover (30% weight)
            days_to_cover = (data['total_short_interest'] /
                             data['avg_daily_volume'])
            extreme_dtc = self.squeeze_thresholds['extreme_days_to_cover']
            high_dtc = self.squeeze_thresholds['high_days_to_cover']
            if days_to_cover >= extreme_dtc:
                probability += 0.3
            elif days_to_cover >= high_dtc:
                probability += 0.2
            
            # Factor 3: Borrow rate (20% weight)
            borrow_rate = data.get('borrow_rate', 0)
            if borrow_rate >= self.squeeze_thresholds['extreme_borrow_rate']:
                probability += 0.2
            elif borrow_rate >= self.squeeze_thresholds['high_borrow_rate']:
                probability += 0.1
            
            # Factor 4: Utilization (10% weight)
            utilization = data.get('utilization_rate', 0)
            if utilization >= self.squeeze_thresholds['extreme_utilization']:
                probability += 0.1
            elif utilization >= self.squeeze_thresholds['high_utilization']:
                probability += 0.05
            
            return min(probability, 1.0)  # Cap at 100%
            
        except Exception as e:
            logger.warning(f"Error calculating squeeze probability: {e}")
            return 0.0
    
    def generate_short_signals(
            self, ticker: str,
            lookback_days: int = 90) -> List[ShortSqueezeSignal]:
        """
        Generate short squeeze and contrarian signals
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Historical data window
            
        Returns:
            List of ShortSqueezeSignal objects
        """
        try:
            logger.info(f"🎯 Generating short interest signals for {ticker}")
            
            # Get short interest data
            short_data = self.get_short_interest_data(ticker, lookback_days)
            
            if not short_data:
                logger.warning("No short data available for signal generation")
                return []

            signals = []

            # Analyze latest data for current signals
            latest = short_data[0]

            # Signal 1: High short interest setup
            high_short_thresh = self.squeeze_thresholds['high_short_ratio']
            if latest.short_ratio >= high_short_thresh:
                signal = self._create_high_short_signal(latest, short_data)
                if signal:
                    signals.append(signal)

            # Signal 2: Short interest momentum
            if len(short_data) >= 2:
                momentum_signal = self._create_momentum_signal(latest, short_data)
                if momentum_signal:
                    signals.append(momentum_signal)

            # Signal 3: Extreme positioning signals
            extreme_signal = self._create_extreme_positioning_signal(latest)
            if extreme_signal:
                signals.append(extreme_signal)

            # Signal 4: Contrarian opportunity
            contrarian_signal = self._create_contrarian_signal(latest, short_data)
            if contrarian_signal:
                signals.append(contrarian_signal)
            
            logger.info(f"📈 Generated {len(signals)} short interest signals")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating short signals: {e}")
            return []
    
    def _create_high_short_signal(self, latest: ShortInterestMetrics, 
                                 historical: List[ShortInterestMetrics]) -> Optional[ShortSqueezeSignal]:
        """Create signal for high short interest levels"""
        try:
            drivers = []
            strength = 0.0
            
            if latest.short_ratio >= self.squeeze_thresholds['extreme_short_ratio']:
                drivers.append(f"Extreme short ratio: {latest.short_ratio:.1%}")
                strength += 0.4
            elif latest.short_ratio >= self.squeeze_thresholds['high_short_ratio']:
                drivers.append(f"High short ratio: {latest.short_ratio:.1%}")
                strength += 0.25
            
            if latest.days_to_cover >= self.squeeze_thresholds['high_days_to_cover']:
                drivers.append(f"High days to cover: {latest.days_to_cover:.1f}")
                strength += 0.3
            
            if latest.borrowing_rate and latest.borrowing_rate >= self.squeeze_thresholds['high_borrow_rate']:
                drivers.append(f"High borrow rate: {latest.borrowing_rate:.1%}")
                strength += 0.2
            
            if strength < 0.3:  # Minimum threshold for signal
                return None
            
            risk_level = "high" if strength >= 0.7 else "medium" if strength >= 0.5 else "low"
            
            return ShortSqueezeSignal(
                ticker=latest.ticker,
                signal_date=latest.report_date,
                signal_type="squeeze_setup",
                probability=latest.squeeze_probability,
                strength=min(strength, 1.0),
                drivers=drivers,
                timeframe="medium_term",
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.warning(f"Error creating high short signal: {e}")
            return None
    
    def _create_momentum_signal(self, latest: ShortInterestMetrics, 
                               historical: List[ShortInterestMetrics]) -> Optional[ShortSqueezeSignal]:
        """Create signal based on short interest momentum"""
        try:
            if latest.change_from_prior == 0:
                return None
            
            drivers = []
            strength = 0.0
            signal_type = "squeeze_risk"
            
            # Increasing short interest (potential setup)
            if latest.change_from_prior > 0.1:  # 10%+ increase
                drivers.append(f"Large short increase: {latest.change_from_prior:.1%}")
                strength += 0.3
                signal_type = "squeeze_setup"
            elif latest.change_from_prior > 0.05:  # 5%+ increase
                drivers.append(f"Short interest increasing: {latest.change_from_prior:.1%}")
                strength += 0.2
            
            # Decreasing short interest (potential covering)
            elif latest.change_from_prior < -0.1:  # 10%+ decrease
                drivers.append(f"Large short covering: {latest.change_from_prior:.1%}")
                strength += 0.4
                signal_type = "squeeze_active"
            elif latest.change_from_prior < -0.05:  # 5%+ decrease
                drivers.append(f"Short covering detected: {latest.change_from_prior:.1%}")
                strength += 0.3
                signal_type = "squeeze_active"
            
            if strength < 0.2:
                return None
            
            return ShortSqueezeSignal(
                ticker=latest.ticker,
                signal_date=latest.report_date,
                signal_type=signal_type,
                probability=latest.squeeze_probability,
                strength=min(strength, 1.0),
                drivers=drivers,
                timeframe="short_term",
                risk_level="medium"
            )
            
        except Exception as e:
            logger.warning(f"Error creating momentum signal: {e}")
            return None
    
    def _create_extreme_positioning_signal(self, 
                                         latest: ShortInterestMetrics) -> Optional[ShortSqueezeSignal]:
        """Create signal for extreme short positioning"""
        try:
            drivers = []
            strength = 0.0
            
            # Check for extreme metrics
            if (latest.short_ratio >= self.squeeze_thresholds['extreme_short_ratio'] and
                latest.days_to_cover >= self.squeeze_thresholds['extreme_days_to_cover']):
                
                drivers.append("Extreme short positioning across metrics")
                strength += 0.6
                
                if latest.borrowing_rate and latest.borrowing_rate >= self.squeeze_thresholds['extreme_borrow_rate']:
                    drivers.append("Critical borrow rate levels")
                    strength += 0.3
                
                if latest.utilization_rate and latest.utilization_rate >= self.squeeze_thresholds['extreme_utilization']:
                    drivers.append("Extreme share utilization")
                    strength += 0.1
            
            if strength < 0.5:  # High threshold for extreme signals
                return None
            
            return ShortSqueezeSignal(
                ticker=latest.ticker,
                signal_date=latest.report_date,
                signal_type="squeeze_risk",
                probability=latest.squeeze_probability,
                strength=min(strength, 1.0),
                drivers=drivers,
                timeframe="medium_term",
                risk_level="high"
            )
            
        except Exception as e:
            logger.warning(f"Error creating extreme positioning signal: {e}")
            return None
    
    def _create_contrarian_signal(self, latest: ShortInterestMetrics, 
                                 historical: List[ShortInterestMetrics]) -> Optional[ShortSqueezeSignal]:
        """Create contrarian signals based on crowded shorts"""
        try:
            # Look for potential contrarian opportunities
            # when shorts are crowded but fundamentals may be improving
            
            drivers = []
            strength = 0.0
            
            # High short interest + recent covering could signal bottom
            if (latest.short_ratio >= self.squeeze_thresholds['high_short_ratio'] and
                latest.change_from_prior < -0.05):  # 5%+ recent decrease
                
                drivers.append("Potential short exhaustion signal")
                strength += 0.4
            
            # Very high days to cover suggests illiquid shorts
            if latest.days_to_cover >= self.squeeze_thresholds['extreme_days_to_cover']:
                drivers.append("Illiquid short positioning")
                strength += 0.3
            
            if strength < 0.3:
                return None
            
            return ShortSqueezeSignal(
                ticker=latest.ticker,
                signal_date=latest.report_date,
                signal_type="contrarian_opportunity",
                probability=latest.squeeze_probability * 0.7,  # Reduce prob for contrarian
                strength=min(strength, 1.0),
                drivers=drivers,
                timeframe="long_term",
                risk_level="medium"
            )
            
        except Exception as e:
            logger.warning(f"Error creating contrarian signal: {e}")
            return None
    
    def _metrics_to_features(self, metrics: 'ShortInterestMetrics', 
                            historical_data: List['ShortInterestMetrics'],
                            current_index: int) -> Dict[str, float]:
        """
        Convert a single ShortInterestMetrics object to feature dictionary
        
        Args:
            metrics: Current metrics snapshot
            historical_data: Full historical list for calculating changes/zscores
            current_index: Index of current metrics in historical_data list
            
        Returns:
            Dictionary of features for this date (without 'short_interest_' prefix)
        """
        try:
            features = {}
            
            # Get options open interest for short_to_oi_ratio
            total_options_oi = self._get_total_options_oi(metrics.ticker)
            
            # Core metrics (no prefix)
            features['percent'] = metrics.short_ratio * 100
            features['ratio'] = metrics.total_short_interest / (metrics.total_short_interest / metrics.days_to_cover) if metrics.days_to_cover > 0 else 0.0
            features['days_to_cover'] = metrics.days_to_cover
            features['float_short_pct'] = metrics.short_ratio
            
            if total_options_oi and total_options_oi > 0:
                features['short_to_oi_ratio'] = metrics.total_short_interest / total_options_oi
            else:
                features['short_to_oi_ratio'] = metrics.squeeze_probability
            
            # Behavioral (use pre-calculated change_from_prior)
            features['change_1m'] = metrics.change_from_prior
            
            # Calculate 3m change if enough history
            change_3m = 0.0
            if current_index + 2 < len(historical_data):
                prior_3m = historical_data[current_index + 2]
                if prior_3m.total_short_interest > 0:
                    change_3m = (metrics.total_short_interest - prior_3m.total_short_interest) / prior_3m.total_short_interest
            features['change_3m'] = change_3m
            
            # Z-scores (calculate from cache if available)
            # IMPORTANT: z-scores must be computed "as-of" the report date to avoid lookahead.
            zscore_1y = 0.0
            try:
                asof = pd.to_datetime(metrics.report_date, errors="coerce")
                if isinstance(asof, pd.Timestamp) and pd.notna(asof):
                    cutoff = asof - pd.Timedelta(days=365)

                    # historical_data is ordered newest->oldest; indices >= current_index are <= asof.
                    past = []
                    for m in historical_data[current_index:]:
                        try:
                            dt = pd.to_datetime(getattr(m, "report_date", None), errors="coerce")
                            if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
                                continue
                            if dt <= asof and dt >= cutoff:
                                past.append(float(getattr(m, "short_ratio", 0.0)))
                        except Exception:
                            continue

                    if len(past) >= 12:
                        short_mean = float(np.mean(past))
                        short_std = float(np.std(past))
                        if short_std > 1e-6:
                            zscore_1y = (metrics.short_ratio - short_mean) / short_std
            except Exception:
                pass
            
            features['zscore_1y'] = zscore_1y
            features['pct_zscore_3y'] = 0.0  # Will be calculated with more data
            features['momentum'] = features['change_1m'] - (change_3m / 3.0)
            
            # Squeeze risk
            squeeze_flag = 1.0 if (metrics.days_to_cover > 5.0 and metrics.short_ratio > 0.10) else 0.0
            features['squeeze_risk_flag'] = squeeze_flag
            
            dtc_component = min(metrics.days_to_cover / 10.0 * 50.0, 50.0)
            float_component = min(metrics.short_ratio / 0.20 * 50.0, 50.0)
            features['squeeze_risk_score'] = dtc_component + float_component
            features['squeeze_probability'] = metrics.squeeze_probability
            
            # Utilization & context
            if metrics.utilization_rate is not None:
                features['shares_on_loan_pct'] = metrics.utilization_rate * 100
            else:
                estimated_on_loan = metrics.total_short_interest * 1.2
                float_shares = metrics.total_short_interest / metrics.short_ratio if metrics.short_ratio > 0 else 1e9
                features['shares_on_loan_pct'] = (estimated_on_loan / float_shares) * 100
            
            if metrics.borrowing_rate is not None:
                features['borrow_rate'] = metrics.borrowing_rate * 100
            else:
                if metrics.short_ratio > 0.30:
                    features['borrow_rate'] = 15.0 + (metrics.short_ratio - 0.30) * 100
                elif metrics.short_ratio > 0.20:
                    features['borrow_rate'] = 5.0 + (metrics.short_ratio - 0.20) * 100
                elif metrics.short_ratio > 0.10:
                    features['borrow_rate'] = 1.0 + (metrics.short_ratio - 0.10) * 40
                else:
                    features['borrow_rate'] = 0.1 + metrics.short_ratio * 9
            
            features['borrow_rate_zscore_3y'] = 0.0  # Will be calculated with more data
            
            # Short vs institutional
            short_vs_inst = 0.0
            try:
                from src.data_sources.eodhd_provider import get_eodhd_provider
                eodhd = get_eodhd_provider()
                fundamentals = eodhd.get_fundamentals(metrics.ticker)
                
                if fundamentals:
                    shares_stats = fundamentals.get('SharesStats', {})
                    inst_pct = shares_stats.get('PercentInstitutions')
                    
                    if inst_pct is not None and inst_pct > 0:
                        short_vs_inst = (metrics.short_ratio * 100) / inst_pct
                    else:
                        short_vs_inst = metrics.short_ratio * 100
            except Exception:
                short_vs_inst = metrics.short_ratio * 10
            
            features['short_vs_institutional'] = short_vs_inst
            
            return features
            
        except Exception as e:
            logger.error(f"Error converting metrics to features: {e}")
            return {}
    
    def get_short_interest_features(self, ticker: str, 
                                   lookback_days: int = 90) -> Dict[str, float]:
        """
        Generate 15+ HIGH-ALPHA short interest features (NO MODELS, RAW METRICS ONLY)
        
        A. Core Metrics (5)
        B. Behavioral Signals (4) 
        C. Squeeze Risk (3)
        D. Utilization & Context (5+): shares_on_loan_pct, short_interest_pct_zscore_3y,
                                       borrow_rate_zscore_3y, short_vs_institutional,
                                       borrow_rate (if available)
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Historical data window (90 days for 1m/3m changes, 1095 for 3y zscore)
            
        Returns:
            Dictionary of 15+ short interest features
        """
        try:
            # Fetch extended history for 3-year z-score (1095 days)
            extended_lookback = max(lookback_days, 1095)
            short_data = self.get_short_interest_data(ticker, extended_lookback)
            
            if not short_data:
                logger.warning(f"No short interest data for {ticker}")
                return {}
            
            latest = short_data[0]
            features = {}
            
            # Get options open interest for short_to_oi_ratio
            total_options_oi = self._get_total_options_oi(ticker)
            
            # ===== A. SHORT INTEREST CORE (5 features) =====
            
            # 1. short_interest_percent: Short interest as % of float
            features['short_interest_percent'] = latest.short_ratio * 100  # Convert to %
            
            # 2. short_interest_ratio: Short interest / Average daily volume
            if latest.days_to_cover > 0:
                features['short_interest_ratio'] = latest.total_short_interest / (
                    latest.total_short_interest / latest.days_to_cover  # Reconstruct avg daily volume
                )
            else:
                features['short_interest_ratio'] = 0.0
            
            # 3. days_to_cover: Days to cover at average daily volume
            features['days_to_cover'] = latest.days_to_cover
            
            # 4. float_short_pct: Short interest / float shares (same as #1 but scaled 0-1)
            features['float_short_pct'] = latest.short_ratio
            
            # 5. short_to_oi_ratio: Short interest / Total options open interest
            if total_options_oi and total_options_oi > 0:
                features['short_to_oi_ratio'] = latest.total_short_interest / total_options_oi
                logger.debug(f"✅ short_to_oi_ratio calculated: {features['short_to_oi_ratio']:.4f} (short={latest.total_short_interest:,}, oi={total_options_oi:,})")
            else:
                # Fallback to squeeze probability as proxy if OI not available
                features['short_to_oi_ratio'] = latest.squeeze_probability
                logger.debug(f"⚠️  short_to_oi_ratio using squeeze_probability proxy (no OI data)")
            
            # ===== B. SHORT INTEREST BEHAVIOR (4 features) =====
            
            # 6. short_interest_change_1m: 1-month change in short interest
            change_1m = 0.0
            
            # PRIORITY 1: Use EODHD's SharesShortPriorMonth if available (snapshot mode)
            if hasattr(latest, 'shares_short_prior_month') and latest.shares_short_prior_month:
                if latest.shares_short_prior_month > 0:
                    change_1m = (latest.total_short_interest - latest.shares_short_prior_month) / latest.shares_short_prior_month
                    logger.debug(f"✅ Using EODHD SharesShortPriorMonth for 1m change: {change_1m:.4f}")
            
            # PRIORITY 2: Calculate from historical time series if available
            elif len(short_data) >= 2:
                # Find report ~30 days ago (FINRA reports bi-monthly)
                for hist in short_data[1:]:
                    try:
                        days_diff = (pd.to_datetime(latest.report_date) - 
                                   pd.to_datetime(hist.report_date)).days
                        if 25 <= days_diff <= 35:  # ~1 month window
                            if hist.total_short_interest > 0:
                                change_1m = (latest.total_short_interest - hist.total_short_interest) / hist.total_short_interest
                            break
                    except Exception:
                        continue
            
            features['short_interest_change_1m'] = change_1m
            
            # 7. short_interest_change_3m: 3-month change in short interest
            # NOTE: EODHD only provides 1-month history (current + prior month)
            # Cannot calculate 3-month change without FINRA historical data
            # For now, estimate from 1m change * 3 (linear approximation)
            change_3m = 0.0
            if len(short_data) >= 3:
                for hist in short_data[1:]:
                    try:
                        days_diff = (pd.to_datetime(latest.report_date) - 
                                   pd.to_datetime(hist.report_date)).days
                        if 80 <= days_diff <= 100:  # ~3 month window
                            if hist.total_short_interest > 0:
                                change_3m = (latest.total_short_interest - hist.total_short_interest) / hist.total_short_interest
                            break
                    except Exception:
                        continue
            else:
                # Estimate from 1-month change if no historical data
                # Assume linear change: 3m ≈ 1m * 3 (rough approximation)
                if change_1m != 0:
                    change_3m = change_1m * 3.0
            
            features['short_interest_change_3m'] = change_3m
            
            # 8. short_interest_zscore_1y: 1-year Z-score of short interest %
            # PRIORITY 1: Use local cache if available (builds up over time)
            zscore_1y = 0.0
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    history_df = cache.get_history(ticker, lookback_days=365)
                    
                    if history_df is not None and len(history_df) >= 12:
                        # Have 12+ snapshots - calculate real zscore
                        short_pcts = history_df['short_pct_float'].dropna().values
                        if len(short_pcts) >= 12:
                            short_mean = np.mean(short_pcts)
                            short_std = np.std(short_pcts)
                            if short_std > 1e-6:
                                zscore_1y = (latest.short_ratio - short_mean) / short_std
                                logger.info(f"✅ Calculated 1y zscore from {len(short_pcts)} cached snapshots: {zscore_1y:.3f}")
                            else:
                                logger.debug("⚠️  Low variation in historical short interest (using 0)")
                    else:
                        count = len(history_df) if history_df is not None else 0
                        logger.debug(f"⏳ Building history: {count}/12 snapshots needed for 1y zscore")
                except Exception as e:
                    logger.debug(f"Cache lookup failed (non-critical): {e}")
            
            # PRIORITY 2: Fallback to available data from API (2 points)
            if zscore_1y == 0.0:
                short_pcts = [d.short_ratio for d in short_data if d.short_ratio is not None]
                if len(short_pcts) >= 2:
                    short_mean = np.mean(short_pcts)
                    short_std = np.std(short_pcts)
                    if short_std > 1e-6:
                        zscore_1y = (latest.short_ratio - short_mean) / short_std
                        logger.debug(f"⚠️  Using {len(short_pcts)}-point zscore (insufficient for 1y)")
            
            features['short_interest_zscore_1y'] = zscore_1y
            
            # 8b. short_interest_pct_zscore_3y: 3-year Z-score for longer-term context
            zscore_3y = 0.0
            if CACHE_AVAILABLE:
                try:
                    cache = get_short_interest_cache()
                    history_df = cache.get_history(ticker, lookback_days=1095)  # 3 years
                    
                    if history_df is not None and len(history_df) >= 36:
                        # Have 36+ snapshots (3 years monthly) - calculate real zscore
                        short_pcts = history_df['short_pct_float'].dropna().values
                        if len(short_pcts) >= 36:
                            short_mean_3y = np.mean(short_pcts)
                            short_std_3y = np.std(short_pcts)
                            if short_std_3y > 1e-6:
                                zscore_3y = (latest.short_ratio - short_mean_3y) / short_std_3y
                                logger.info(f"✅ Calculated 3y zscore from {len(short_pcts)} cached snapshots: {zscore_3y:.3f}")
                    else:
                        count = len(history_df) if history_df is not None else 0
                        logger.debug(f"⏳ Building history: {count}/36 snapshots needed for 3y zscore")
                except Exception as e:
                    logger.debug(f"Cache lookup failed (non-critical): {e}")
            
            # Fallback: use available data (same as 1y if insufficient)
            if zscore_3y == 0.0 and len(short_data) >= 36:
                short_pcts_3y = [d.short_ratio for d in short_data[:36] if d.short_ratio is not None]
                if len(short_pcts_3y) >= 36:
                    short_mean_3y = np.mean(short_pcts_3y)
                    short_std_3y = np.std(short_pcts_3y)
                    if short_std_3y > 1e-6:
                        zscore_3y = (latest.short_ratio - short_mean_3y) / short_std_3y
            
            features['short_interest_pct_zscore_3y'] = zscore_3y
            
            # 9. short_interest_momentum: Acceleration of short interest changes
            # (change_1m - change_3m/3) measures if shorts accelerating or decelerating
            features['short_interest_momentum'] = change_1m - (change_3m / 3.0)
            
            # ===== C. SQUEEZE RISK SIGNALS (3 features) =====
            
            # 10. squeeze_risk_flag: Binary flag (days_to_cover > 5 AND float_short_pct > 10%)
            squeeze_flag = 1.0 if (latest.days_to_cover > 5.0 and latest.short_ratio > 0.10) else 0.0
            features['squeeze_risk_flag'] = squeeze_flag
            
            # 11. squeeze_risk_score: Continuous score (0-100)
            # Formula: Weighted combination of days_to_cover and float_short_pct
            # Days to cover component (0-50): min(days_to_cover / 10 * 50, 50)
            # Float short component (0-50): min(float_short_pct * 100 / 20 * 50, 50)
            dtc_component = min(latest.days_to_cover / 10.0 * 50.0, 50.0)
            float_component = min(latest.short_ratio / 0.20 * 50.0, 50.0)
            features['squeeze_risk_score'] = dtc_component + float_component
            
            # 12. LEGACY: Keep old squeeze_probability for backward compatibility
            features['squeeze_probability'] = latest.squeeze_probability
            
            # ===== D. UTILIZATION & HISTORICAL CONTEXT (5+ features) =====
            
            # 13. shares_on_loan_pct: Shares on loan / float (utilization proxy)
            # EODHD doesn't provide on-loan data directly
            # Proxy: Use utilization_rate if available, else estimate from short interest
            if latest.utilization_rate is not None:
                features['shares_on_loan_pct'] = latest.utilization_rate * 100  # Convert to %
                logger.debug(f"✅ Using real utilization_rate: {features['shares_on_loan_pct']:.2f}%")
            else:
                # Estimate: Shares on loan ≈ short interest * 1.2 (shorts are ~80% of loans)
                # Then divide by float to get percentage
                estimated_on_loan = latest.total_short_interest * 1.2
                float_shares = latest.total_short_interest / latest.short_ratio if latest.short_ratio > 0 else 1e9
                features['shares_on_loan_pct'] = (estimated_on_loan / float_shares) * 100
                logger.debug(f"⚠️  Using estimated on-loan from short interest: {features['shares_on_loan_pct']:.2f}%")
            
            # 14. borrow_rate: Stock borrow rate (if available from data)
            if latest.borrowing_rate is not None:
                features['borrow_rate'] = latest.borrowing_rate * 100  # Convert to %
                logger.debug(f"✅ Using real borrow_rate: {features['borrow_rate']:.2f}%")
            else:
                # Estimate from short interest level (higher short % → higher borrow rate)
                # Typical range: 0.1% (low short) to 50%+ (extreme short)
                if latest.short_ratio > 0.30:
                    features['borrow_rate'] = 15.0 + (latest.short_ratio - 0.30) * 100  # Extreme
                elif latest.short_ratio > 0.20:
                    features['borrow_rate'] = 5.0 + (latest.short_ratio - 0.20) * 100  # High
                elif latest.short_ratio > 0.10:
                    features['borrow_rate'] = 1.0 + (latest.short_ratio - 0.10) * 40  # Elevated
                else:
                    features['borrow_rate'] = 0.1 + latest.short_ratio * 9  # Normal
                logger.debug(f"⚠️  Using estimated borrow_rate: {features['borrow_rate']:.2f}%")
            
            # 15. borrow_rate_zscore_3y: 3-year Z-score of borrow rate (historical context)
            borrow_zscore_3y = 0.0
            
            # Try to calculate from cache if we have historical borrow rates
            # NOTE: Most data sources don't provide historical borrow rates
            # For now, use short interest zscore as proxy (correlated)
            # Future: Enhance cache to store borrow rates separately
            if zscore_3y != 0.0:
                # Borrow rate correlates with short interest
                # Use short interest zscore as proxy for borrow rate zscore
                borrow_zscore_3y = zscore_3y * 0.8  # Slightly dampen (imperfect correlation)
                logger.debug(f"⚠️  Using short_interest_zscore as proxy for borrow_rate_zscore_3y: {borrow_zscore_3y:.3f}")
            
            features['borrow_rate_zscore_3y'] = borrow_zscore_3y
            
            # 16. short_vs_institutional: Short interest vs institutional ownership
            # Compare crowded shorts to institutional positioning
            short_vs_inst = 0.0
            try:
                from src.data_sources.eodhd_provider import get_eodhd_provider
                eodhd = get_eodhd_provider()
                fundamentals = eodhd.get_fundamentals(ticker)
                
                if fundamentals:
                    # Get institutional ownership % from SharesStats
                    shares_stats = fundamentals.get('SharesStats', {})
                    inst_pct = shares_stats.get('PercentInstitutions')
                    
                    if inst_pct is not None:
                        # Ratio: short_pct / institutional_pct
                        # >1 = shorts > institutions (crowded short)
                        # <1 = institutions > shorts (fundamental support)
                        if inst_pct > 0:
                            short_vs_inst = (latest.short_ratio * 100) / inst_pct
                            logger.debug(f"✅ Calculated short_vs_institutional: {short_vs_inst:.3f} (short={latest.short_ratio*100:.1f}%, inst={inst_pct:.1f}%)")
                        else:
                            # Low institutional ownership
                            short_vs_inst = latest.short_ratio * 100  # Just use short %
                    else:
                        logger.debug("⚠️  No institutional ownership data available")
                        # Fallback: use short ratio as proxy
                        short_vs_inst = latest.short_ratio * 10  # Scale to reasonable range
                        
            except Exception as e:
                logger.debug(f"Could not fetch institutional data: {e}")
                # Fallback: use short ratio
                short_vs_inst = latest.short_ratio * 10
            
            features['short_vs_institutional'] = short_vs_inst
            
            logger.info(f"📊 Generated {len(features)} short interest features for {ticker}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating short interest features: {e}")
            return {}