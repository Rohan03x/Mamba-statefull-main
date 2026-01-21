"""
Backup Price Sources & Volatility Provider for DCF Lab

This module provides backup price data sources and volatility indicators:
- Stooq CSV data (indexes, equities, ETFs) as Yahoo Finance backup
- Cboe VIX daily closes for volatility/fear gauge
- FRED VIX data (VIXCLS) for historical analysis
- Additional volatility metrics and calculations

Key Features:
- Stooq free historical data access
- VIX volatility tracking
- Backup when Yahoo Finance throttles
- Historical volatility calculations
- Market stress indicators
- Comprehensive volatility analysis

Data Sources:
- Stooq: https://stooq.com/db/h/ (CSV downloads)
- Cboe VIX: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
- FRED VIX: https://fred.stlouisfed.org/series/VIXCLS

Author: DCF Lab Team
Created: 2025-09-18
"""

import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BackupPriceConfig:
    """Configuration for backup price sources"""
    stooq_base_url: str = "https://stooq.com/q/d/l/"
    cboe_vix_url: str = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    # Will use existing FRED provider if available
    fred_api_key: Optional[str] = None
    user_agent: str = "DCF Lab Backup Price Provider (contact@dcflab.com)"
    timeout: int = 30
    retries: int = 3
    cache_dir: str = "./cache/backup_prices"
    enable_cache: bool = True
    max_cache_age_hours: int = 6  # Refresh more frequently for price data


class StooqProvider:
    """
    Stooq CSV Data Provider

    Provides access to Stooq's free historical price data as backup
    when primary sources (Yahoo Finance) are unavailable or throttled.
    """

    def __init__(self, config: BackupPriceConfig):
        self.config = config
        self._session = None

        # Create cache directory
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Get configured requests session"""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set headers
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'text/csv,application/csv'
            })

        return self._session

    def _get_cache_path(self, symbol: str, period: str = "d") -> Path:
        """Get cache file path for symbol"""
        return Path(self.config.cache_dir) / f"stooq_{symbol}_{period}.csv"

    def _load_from_cache(self, symbol: str,
                         period: str = "d") -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self.config.enable_cache:
            return None

        cache_path = self._get_cache_path(symbol, period)

        if not cache_path.exists():
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=self.config.max_cache_age_hours):
            logger.debug(f"Cache expired for {symbol}")
            return None

        try:
            df = pd.read_csv(
                cache_path,
                parse_dates=['Date'],
                index_col='Date')
            logger.debug(f"Loaded {symbol} from cache ({len(df)} records)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(
            self,
            symbol: str,
            data: pd.DataFrame,
            period: str = "d"):
        """Save data to cache"""
        if not self.config.enable_cache:
            return

        cache_path = self._get_cache_path(symbol, period)

        try:
            data.to_csv(cache_path)
            logger.debug(f"Saved {symbol} to cache ({len(data)} records)")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    def get_historical_data(
            self,
            symbol: str,
            period: str = "d",
            start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical price data from Stooq

        Args:
            symbol: Stock symbol (e.g., 'AAPL.US', 'SPY.US')
            period: Data period ('d' for daily, 'w' for weekly, 'm' for monthly)
            start_date: Start date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cached_data = self._load_from_cache(symbol, period)
        if cached_data is not None:
            return cached_data

        # Format symbol for Stooq (add .US if not present)
        if '.' not in symbol:
            stooq_symbol = f"{symbol.upper()}.US"
        else:
            stooq_symbol = symbol.upper()

        # Build URL parameters
        params = {
            's': stooq_symbol,
            'i': period  # d, w, m
        }

        # Add date range if specified
        if start_date:
            params['d1'] = start_date.replace('-', '')

        try:
            logger.info(f"Fetching {symbol} data from Stooq...")
            session = self._get_session()

            response = session.get(
                self.config.stooq_base_url,
                params=params,
                timeout=self.config.timeout)
            response.raise_for_status()

            # Parse CSV data
            csv_data = response.text
            df = pd.read_csv(io.StringIO(csv_data))

            # Standardize column names
            if not df.empty:
                df.columns = df.columns.str.title()

                # Ensure we have the expected columns
                expected_cols = [
                    'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = set(expected_cols) - set(df.columns)

                if missing_cols:
                    logger.warning(
                        f"Missing columns for {symbol}: {missing_cols}")

                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Sort by date
                df.sort_index(inplace=True)

                # Cache the data
                self._save_to_cache(symbol, df, period)

                logger.info(
                    f"Retrieved {
                        len(df)} records for {symbol} from Stooq")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get Stooq data for {symbol}: {e}")
            return pd.DataFrame()


class VIXProvider:
    """
    VIX Volatility Data Provider

    Provides access to VIX (Volatility Index) data from multiple sources:
    - Cboe VIX historical data
    - FRED VIX series (VIXCLS)
    - Calculated volatility metrics
    """

    def __init__(self, config: BackupPriceConfig):
        self.config = config
        self._session = None

        # Create cache directory
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Get configured requests session"""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set headers
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'text/csv'
            })

        return self._session

    def get_cboe_vix_data(self) -> pd.DataFrame:
        """
        Get VIX historical data from Cboe

        Returns:
            DataFrame with VIX open, high, low, close data
        """
        cache_path = Path(self.config.cache_dir) / "cboe_vix_history.csv"

        # Check cache
        if (self.config.enable_cache and cache_path.exists()):
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=self.config.max_cache_age_hours):
                try:
                    df = pd.read_csv(
                        cache_path,
                        parse_dates=['Date'],
                        index_col='Date')
                    logger.info(
                        f"Loaded VIX data from cache ({
                            len(df)} records)")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load VIX cache: {e}")

        try:
            logger.info("Fetching VIX data from Cboe...")
            session = self._get_session()

            response = session.get(
                self.config.cboe_vix_url,
                timeout=self.config.timeout)
            response.raise_for_status()

            # Parse CSV data
            df = pd.read_csv(io.StringIO(response.text))

            # Clean and standardize data
            if not df.empty:
                # Rename columns to standard format
                df.columns = df.columns.str.strip().str.title()
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)

                # Sort by date
                df.sort_index(inplace=True)

                # Cache the data
                if self.config.enable_cache:
                    df.to_csv(cache_path)

                logger.info(f"Retrieved {len(df)} VIX records from Cboe")
                return df
            else:
                logger.warning("No VIX data returned from Cboe")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get VIX data from Cboe: {e}")
            return pd.DataFrame()

    def calculate_volatility_metrics(
            self,
            price_data: pd.DataFrame,
            window: int = 20) -> pd.DataFrame:
        """
        Calculate various volatility metrics from price data

        Args:
            price_data: DataFrame with OHLC price data
            window: Rolling window for calculations

        Returns:
            DataFrame with volatility metrics
        """
        if price_data.empty or 'Close' not in price_data.columns:
            return pd.DataFrame()

        df = price_data.copy()

        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Historical volatility (annualized)
        df['Volatility_20d'] = df['Returns'].rolling(
            window=window).std() * np.sqrt(252)

        # Parkinson volatility (using OHLC)
        if all(col in df.columns for col in ['High', 'Low', 'Open']):
            df['Parkinson_Vol'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                (np.log(df['High'] / df['Low']) ** 2)
            ).rolling(window=window).mean() * np.sqrt(252)

            # Garman-Klass volatility
            df['GK_Vol'] = np.sqrt(
                0.5 * (np.log(df['High'] / df['Low']) ** 2) -
                (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
            ).rolling(window=window).mean() * np.sqrt(252)

        # Rolling min/max volatility
        df['Vol_Min_20d'] = df['Volatility_20d'].rolling(window=window).min()
        df['Vol_Max_20d'] = df['Volatility_20d'].rolling(window=window).max()

        # Volatility percentiles
        df['Vol_Percentile'] = df['Volatility_20d'].rolling(
            window=252).rank(
            pct=True)

        return df


class BackupPriceProvider:
    """
    Comprehensive Backup Price Sources & Volatility Provider

    Combines Stooq CSV data and VIX volatility metrics as backup
    for primary price data sources.
    """

    def __init__(self, config: Optional[BackupPriceConfig] = None):
        """Initialize backup price provider"""
        self.config = config or BackupPriceConfig()
        self.stooq = StooqProvider(self.config)
        self.vix = VIXProvider(self.config)

    def get_backup_price_data(
            self, symbol: str, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive backup price data for a symbol

        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval

        Returns:
            Dict with price data, volatility metrics, and metadata
        """
        result = {
            'symbol': symbol,
            'source': 'stooq_backup',
            'retrieved_at': datetime.now().isoformat(),
            'price_data': pd.DataFrame(),
            'volatility_metrics': pd.DataFrame(),
            'success': False,
            'error': None
        }

        try:
            # Get price data from Stooq
            price_data = self.stooq.get_historical_data(
                symbol, start_date=start_date)

            if not price_data.empty:
                result['price_data'] = price_data

                # Calculate volatility metrics
                vol_metrics = self.vix.calculate_volatility_metrics(price_data)
                result['volatility_metrics'] = vol_metrics

                result['success'] = True
                result['record_count'] = len(price_data)
                result['date_range'] = {
                    'start': price_data.index.min().strftime('%Y-%m-%d'),
                    'end': price_data.index.max().strftime('%Y-%m-%d')
                }

                logger.info(f"Successfully retrieved backup data for {symbol}")
            else:
                result['error'] = "No data available from Stooq"
                logger.warning(f"No backup data available for {symbol}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to get backup data for {symbol}: {e}")

        return result

    def get_vix_data(self) -> Dict[str, Any]:
        """
        Get comprehensive VIX volatility data

        Returns:
            Dict with VIX data and volatility analysis
        """
        result = {
            'source': 'cboe_vix',
            'retrieved_at': datetime.now().isoformat(),
            'vix_data': pd.DataFrame(),
            'volatility_analysis': {},
            'success': False,
            'error': None
        }

        try:
            # Get VIX data
            vix_data = self.vix.get_cboe_vix_data()

            if not vix_data.empty:
                result['vix_data'] = vix_data

                # Perform volatility analysis
                if 'Close' in vix_data.columns:
                    close_prices = vix_data['Close'].dropna()

                    result['volatility_analysis'] = {
                        'current_vix': float(close_prices.iloc[-1])
                        if len(close_prices) > 0 else None,
                        'avg_30d': float(close_prices.tail(30).mean())
                        if len(close_prices) >= 30 else None,
                        'avg_1y': float(close_prices.tail(252).mean())
                        if len(close_prices) >= 252 else None,
                        'min_1y': float(close_prices.tail(252).min())
                        if len(close_prices) >= 252 else None,
                        'max_1y': float(close_prices.tail(252).max())
                        if len(close_prices) >= 252 else None,
                        'percentile_1y':
                        float(
                            close_prices.tail(252).rank(pct=True).iloc[-1])
                        if len(close_prices) >= 252 else None}

                result['success'] = True
                result['record_count'] = len(vix_data)
                result['date_range'] = {
                    'start': vix_data.index.min().strftime('%Y-%m-%d'),
                    'end': vix_data.index.max().strftime('%Y-%m-%d')
                }

                logger.info(
                    f"Successfully retrieved VIX data ({
                        len(vix_data)} records)")
            else:
                result['error'] = "No VIX data available"
                logger.warning("No VIX data available")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to get VIX data: {e}")

        return result

    def get_market_stress_indicators(self) -> Dict[str, Any]:
        """
        Get comprehensive market stress and volatility indicators

        Returns:
            Dict with multiple stress indicators and analysis
        """
        result = {
            'retrieved_at': datetime.now().isoformat(),
            'indicators': {},
            'analysis': {},
            'success': False,
            'error': None
        }

        try:
            # Get VIX data for stress analysis
            vix_result = self.get_vix_data()

            if vix_result['success']:
                vix_data = vix_result['vix_data']
                vix_analysis = vix_result['volatility_analysis']

                # Calculate stress indicators
                if 'Close' in vix_data.columns:
                    vix_data['Close'].dropna()

                    # VIX-based stress levels
                    current_vix = vix_analysis.get('current_vix')
                    if current_vix:
                        if current_vix < 15:
                            stress_level = "Low"
                            stress_description = "Complacency/Low volatility environment"
                        elif current_vix < 20:
                            stress_level = "Normal"
                            stress_description = "Normal market conditions"
                        elif current_vix < 30:
                            stress_level = "Elevated"
                            stress_description = "Heightened uncertainty"
                        elif current_vix < 40:
                            stress_level = "High"
                            stress_description = "Significant market stress"
                        else:
                            stress_level = "Extreme"
                            stress_description = "Crisis/panic conditions"

                        result['indicators'] = {
                            'vix_stress_level': stress_level,
                            'vix_description': stress_description,
                            'current_vix': current_vix,
                            'vix_percentile_1y': vix_analysis.get('percentile_1y'),
                            'vix_vs_avg': current_vix - vix_analysis.get('avg_1y', current_vix)
                        }

                        # Additional analysis
                        result['analysis'] = {
                            'volatility_regime': 'High'
                            if current_vix > 25 else 'Low',
                            'fear_greed_indicator': 'Fear'
                            if current_vix > 25 else 'Greed',
                            'market_outlook': stress_description}

                result['success'] = True
                logger.info("Successfully calculated market stress indicators")
            else:
                result['error'] = vix_result.get(
                    'error', 'Failed to get VIX data')

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to calculate stress indicators: {e}")

        return result


def get_backup_provider(
        config: Optional[BackupPriceConfig] = None) -> BackupPriceProvider:
    """Factory function to create backup price provider"""
    return BackupPriceProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_backup_provider()

    # Test backup price data
    print("=== Backup Price Data Test ===")
    result = provider.get_backup_price_data("AAPL", start_date="2024-01-01")

    if result['success']:
        print(
            f"Successfully retrieved {
                result['record_count']} records for {
                result['symbol']}")
        print(
            f"Date range: {
                result['date_range']['start']} to {
                result['date_range']['end']}")

        # Show latest prices
        price_data = result['price_data']
        if not price_data.empty:
            print("\nLatest 5 trading days:")
            print(
                price_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

        # Show volatility metrics
        vol_data = result['volatility_metrics']
        if not vol_data.empty and 'Volatility_20d' in vol_data.columns:
            latest_vol = vol_data['Volatility_20d'].dropna().iloc[-1]
            print(f"\nCurrent 20-day volatility: {latest_vol:.2%}")
    else:
        print(f"Failed to get backup data: {result['error']}")

    # Test VIX data
    print("\n=== VIX Volatility Test ===")
    vix_result = provider.get_vix_data()

    if vix_result['success']:
        vix_analysis = vix_result['volatility_analysis']
        print(f"Current VIX: {vix_analysis.get('current_vix', 'N/A')}")
        print(f"30-day average: {vix_analysis.get('avg_30d', 'N/A'):.2f}")
        print(
            f"1-year percentile: {vix_analysis.get('percentile_1y', 'N/A'):.1%}")
    else:
        print(f"Failed to get VIX data: {vix_result['error']}")

    # Test market stress indicators
    print("\n=== Market Stress Indicators ===")
    stress_result = provider.get_market_stress_indicators()

    if stress_result['success']:
        indicators = stress_result['indicators']
        analysis = stress_result['analysis']

        print(f"Stress Level: {indicators.get('vix_stress_level', 'N/A')}")
        print(f"Description: {indicators.get('vix_description', 'N/A')}")
        print(f"Volatility Regime: {analysis.get('volatility_regime', 'N/A')}")
        print(f"Fear/Greed: {analysis.get('fear_greed_indicator', 'N/A')}")
    else:
        print(f"Failed to get stress indicators: {stress_result['error']}")
