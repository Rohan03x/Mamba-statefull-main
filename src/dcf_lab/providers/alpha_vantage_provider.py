"""
Alpha Vantage Data Provider
==========================

Comprehensive integration with Alpha Vantage API for:
- Stock price data (intraday, daily, weekly, monthly)
- Fundamental data (income statements, balance sheets, cash flow)
- Economic indicators
- Forex and cryptocurrency data
- Technical indicators

API Documentation: https://www.alphavantage.co/documentation/
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
try:
    # Prefer local utils for rate limiting/backoff if available
    from src.dcf_lab.utils.rate import backoff_try  # type: ignore
except Exception:  # fallback no-op
    backoff_try = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AlphaVantageQuote:
    """Alpha Vantage quote data structure"""
    symbol: str
    price: float
    change: float
    change_percent: str
    volume: int
    timestamp: str


class AlphaVantageProvider:
    """
    Alpha Vantage API provider for comprehensive financial data

    Key Features:
    - Real-time and historical stock data
    - Fundamental data (financials, earnings, overview)
    - Economic indicators
    - Technical indicators
    - Forex and crypto data
    - High-frequency intraday data
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage provider

        Args:
            api_key: Alpha Vantage API key (get from https://www.alphavantage.co/support/#api-key)
                    If None, will try to get from ALPHA_VANTAGE_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.warning(
                "No Alpha Vantage API key provided. Using demo key with limited functionality.")
            self.api_key = "demo"  # Demo key for testing

        self.base_url = "https://www.alphavantage.co/query"
        self._cache = {}
        self._last_request_time = 0
        # Alpha Vantage allows 5 calls per minute for free tier
        self._min_request_interval = 12

    def _rate_limit(self):
        """Rate limiting for Alpha Vantage API"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """Make API request with error handling"""
        params['apikey'] = self.api_key

        self._rate_limit()

        def _req():
            return requests.get(self.base_url, params=params, timeout=30)

        try:
            if backoff_try is not None:
                response = backoff_try(_req)
            else:
                response = _req()
            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if 'Error Message' in data:
                raise ValueError(
                    f"Alpha Vantage API error: {
                        data['Error Message']}")
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")

            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage API request failed: {e}")
            raise

    def get_daily_data(self,
                       symbol: str,
                       outputsize: str = "compact") -> pd.DataFrame:
        """
        Get daily OHLCV data

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize
        }

        try:
            data = self._make_request(params)

            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"No daily data found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = [
                'open',
                'high',
                'low',
                'close',
                'adjusted_close',
                'volume',
                'dividend',
                'split']

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Failed to get daily data for {symbol}: {e}")
            return pd.DataFrame()

    def get_intraday_data(self,
                          symbol: str,
                          interval: str = "5min",
                          outputsize: str = "compact") -> pd.DataFrame:
        """
        Get intraday OHLCV data

        Args:
            symbol: Stock symbol
            interval: 1min, 5min, 15min, 30min, 60min
            outputsize: 'compact' (latest 100 points) or 'full'

        Returns:
            DataFrame with intraday OHLCV data
        """
        def _aliases(sym: str) -> List[str]:
            al: List[str] = [sym]
            # GOOGL/GOOG cross-try
            if sym.upper() == 'GOOGL':
                al.append('GOOG')
            elif sym.upper() == 'GOOG':
                al.append('GOOGL')
            # dot/dash variant (e.g., BRK.B -> BRK-B)
            if '.' in sym:
                al.append(sym.replace('.', '-'))
            if '-' in sym:
                al.append(sym.replace('-', '.'))
            # dedupe preserving order
            seen = set()
            out: List[str] = []
            for s in al:
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            return out

        def _fetch_once(sym: str) -> pd.DataFrame:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': sym,
                'interval': interval,
                'outputsize': outputsize
            }
            data = self._make_request(params)
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                return pd.DataFrame()
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        try:
            for sym_try in _aliases(symbol):
                df = _fetch_once(sym_try)
                if not df.empty:
                    if sym_try != symbol:
                        logger.info(f"Intraday data fetched using alias '{sym_try}' for requested symbol '{symbol}'")
                    return df
            logger.warning(f"No intraday data found for {symbol} (tried aliases: {', '.join(_aliases(symbol))})")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get intraday data for {symbol}: {e}")
            return pd.DataFrame()

    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get company overview and fundamental data

        Args:
            symbol: Stock symbol

        Returns:
            Dict with company overview data
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }

        try:
            data = self._make_request(params)

            if not data or 'Symbol' not in data:
                logger.error(f"No overview data found for {symbol}")
                return {}

            # Parse key metrics
            overview = {
                'symbol': data.get('Symbol'),
                'name': data.get('Name'),
                'sector': data.get('Sector'),
                'industry': data.get('Industry'),
                'market_cap': self._parse_number(
                    data.get('MarketCapitalization')),
                'pe_ratio': self._parse_number(data.get('PERatio')),
                'peg_ratio': self._parse_number(data.get('PEGRatio')),
                'price_to_book': self._parse_number(
                    data.get('PriceToBookRatio')),
                'div_yield': self._parse_number(data.get('DividendYield')),
                'eps': self._parse_number(data.get('EPS')),
                'revenue_ttm': self._parse_number(data.get('RevenueTTM')),
                'gross_profit_ttm': self._parse_number(
                    data.get('GrossProfitTTM')),
                'ebitda': self._parse_number(data.get('EBITDA')),
                '52_week_high': self._parse_number(data.get('52WeekHigh')),
                '52_week_low': self._parse_number(data.get('52WeekLow')),
                'beta': self._parse_number(data.get('Beta')),
                'shares_outstanding': self._parse_number(
                    data.get('SharesOutstanding')),
                'description': data.get('Description', '')}

            return overview

        except Exception as e:
            logger.error(f"Failed to get overview for {symbol}: {e}")
            return {}

    def get_earnings(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get earnings data (annual and quarterly)

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (annual_earnings, quarterly_earnings) DataFrames
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol
        }

        try:
            data = self._make_request(params)

            annual_earnings = pd.DataFrame()
            quarterly_earnings = pd.DataFrame()

            if 'annualEarnings' in data:
                annual_df = pd.DataFrame(data['annualEarnings'])
                if not annual_df.empty:
                    annual_df['fiscalDateEnding'] = pd.to_datetime(
                        annual_df['fiscalDateEnding'])
                    annual_df['reportedEPS'] = pd.to_numeric(
                        annual_df['reportedEPS'], errors='coerce')
                    annual_earnings = annual_df.set_index('fiscalDateEnding')

            if 'quarterlyEarnings' in data:
                quarterly_df = pd.DataFrame(data['quarterlyEarnings'])
                if not quarterly_df.empty:
                    quarterly_df['fiscalDateEnding'] = pd.to_datetime(
                        quarterly_df['fiscalDateEnding'])
                    quarterly_df['reportedEPS'] = pd.to_numeric(
                        quarterly_df['reportedEPS'], errors='coerce')
                    quarterly_df['estimatedEPS'] = pd.to_numeric(
                        quarterly_df['estimatedEPS'], errors='coerce')
                    quarterly_df['surprise'] = pd.to_numeric(
                        quarterly_df['surprise'], errors='coerce')
                    quarterly_df['surprisePercentage'] = pd.to_numeric(
                        quarterly_df['surprisePercentage'], errors='coerce')
                    quarterly_earnings = quarterly_df.set_index(
                        'fiscalDateEnding')

            return annual_earnings, quarterly_earnings

        except Exception as e:
            logger.error(f"Failed to get earnings for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_income_statement(
            self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get income statement data (annual and quarterly)

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (annual_income, quarterly_income) DataFrames
        """
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol
        }

        try:
            data = self._make_request(params)

            annual_income = self._process_annual_reports(
                data.get('annualReports', []))
            quarterly_income = self._process_quarterly_reports(
                data.get('quarterlyReports', []))

            return annual_income, quarterly_income

        except Exception as e:
            logger.error(f"Failed to get income statement for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _process_annual_reports(self, reports: List[Dict]) -> pd.DataFrame:
        """Process annual reports data"""
        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.set_index('fiscalDateEnding')

        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _process_quarterly_reports(self, reports: List[Dict]) -> pd.DataFrame:
        """Process quarterly reports data"""
        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.set_index('fiscalDateEnding')

        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_economic_indicator(
            self,
            function: str,
            interval: str = "monthly") -> pd.DataFrame:
        """
        Get economic indicator data

        Args:
            function: Economic function (e.g., 'REAL_GDP', 'CPI', 'UNEMPLOYMENT')
            interval: 'monthly', 'quarterly', 'annual'

        Returns:
            DataFrame with economic indicator data
        """
        params = {
            'function': function,
            'interval': interval
        }

        try:
            data = self._make_request(params)

            # Alpha Vantage economic data has varying structure
            data_key = 'data'
            if data_key not in data:
                logger.error(f"No economic data found for function {function}")
                return pd.DataFrame()

            df = pd.DataFrame(data[data_key])
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.sort_index()

            return df

        except Exception as e:
            logger.error(f"Failed to get economic indicator {function}: {e}")
            return pd.DataFrame()

    def get_forex_data(self,
                       from_symbol: str = "USD",
                       to_symbol: str = "EUR",
                       outputsize: str = "compact") -> pd.DataFrame:
        """
        Get foreign exchange rate data

        Args:
            from_symbol: Base currency
            to_symbol: Quote currency
            outputsize: 'compact' or 'full'

        Returns:
            DataFrame with forex OHLC data
        """
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'outputsize': outputsize
        }

        try:
            data = self._make_request(params)

            time_series_key = 'Time Series (FX Daily)'
            if time_series_key not in data:
                logger.error(
                    f"No forex data found for {from_symbol}/{to_symbol}")
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(
                f"Failed to get forex data for {from_symbol}/{to_symbol}: {e}")
            return pd.DataFrame()

    def _parse_number(self, value: str) -> Optional[float]:
        """Parse string number, handling None and 'None' values"""
        if value is None or value == 'None' or value == '-':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def get_comprehensive_data(self, symbol: str) -> Dict:
        """
        Get comprehensive data for a symbol including price, fundamentals, and earnings

        Args:
            symbol: Stock symbol

        Returns:
            Dict with all available data
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_sources': []
        }

        try:
            # Get daily price data
            daily_data = self.get_daily_data(symbol)
            if not daily_data.empty:
                result['daily_data'] = daily_data
                result['data_sources'].append('daily_prices')

                # Add latest price info
                latest = daily_data.iloc[-1]
                result['latest_price'] = {
                    'date': daily_data.index[-1].strftime('%Y-%m-%d'),
                    'close': latest['close'],
                    'volume': latest['volume'],
                    'change': latest['close'] - daily_data.iloc[-2]['close']
                    if len(daily_data) > 1 else 0}

            # Get company overview
            overview = self.get_company_overview(symbol)
            if overview:
                result['overview'] = overview
                result['data_sources'].append('company_overview')

            # Get earnings
            annual_earnings, quarterly_earnings = self.get_earnings(symbol)
            if not annual_earnings.empty or not quarterly_earnings.empty:
                result['earnings'] = {
                    'annual': annual_earnings,
                    'quarterly': quarterly_earnings
                }
                result['data_sources'].append('earnings')

            return result

        except Exception as e:
            logger.error(f"Failed to get comprehensive data for {symbol}: {e}")
            result['error'] = str(e)
            return result
