"""
FRED (Federal Reserve Economic Data) API Integration
==================================================

Comprehensive integration with St. Louis Fed's FRED API for:
- Macro indicators (CPI, unemployment, GDP, etc.)
- Interest rates and yield curves
- Economic policy indicators
- Market benchmarks and volatility indices
- Historical vintages for backtesting

API Documentation: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class FREDSeries:
    """FRED data series metadata"""
    id: str
    title: str
    frequency: str
    units: str
    last_updated: str
    notes: str


class FREDProvider:
    """
    FRED API provider for comprehensive economic data

    Key Features:
    - Macro indicators (inflation, employment, growth)
    - Interest rates and yield curves
    - Market volatility indices
    - Real-time and vintage data
    - Automated data validation
    """

    # Essential macro indicators for financial modeling
    CORE_SERIES = {
        # Inflation & Prices
        'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
        'CPILFESL': 'Consumer Price Index: All Items Less Food and Energy',
        'PCEPI': 'Personal Consumption Expenditures: Chain-type Price Index',

        # Employment
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'All Employees, Total Nonfarm',
        'CIVPART': 'Civilian Labor Force Participation Rate',

        # Growth & Output
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real Gross Domestic Product',
        'INDPRO': 'Industrial Production Index',

        # Interest Rates
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Constant Maturity Rate',
        'DGS3MO': '3-Month Treasury Constant Maturity Rate',
        'DGS2': '2-Year Treasury Constant Maturity Rate',
        'DGS5': '5-Year Treasury Constant Maturity Rate',
        'DGS30': '30-Year Treasury Constant Maturity Rate',

        # Money Supply
        'M1SL': 'M1 Money Stock',
        'M2SL': 'M2 Money Stock',

        # Market Indicators
        'VIXCLS': 'CBOE Volatility Index: VIX',
        'DEXUSEU': 'U.S. / Euro Foreign Exchange Rate',
        'DEXJPUS': 'Japan / U.S. Foreign Exchange Rate',

        # Housing
        'HOUST': 'Housing Starts: Total: New Privately Owned Housing Units Started',
        'CSUSHPISA': 'S&P/Case-Shiller U.S. National Home Price Index',

        # Credit & Banking
        'TOTRESNS': 'Total Reserve Balances Maintained',
        'WALCL': 'All Federal Reserve Banks: Total Assets',
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED provider

        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
                    If None, will try to get from FRED_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            logger.warning(
                "No FRED API key provided. Some features may be limited.")

        self.base_url = "https://api.stlouisfed.org/fred"
        self._cache = {}
        self._last_request_time = 0
        self._min_request_interval = 2.0  # Increased rate limiting for 429 errors
        self._request_count = 0
        self._daily_limit = 120000  # FRED daily limit

    def _rate_limit(self):
        """Enhanced rate limiting with exponential backoff for 429 errors"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with error handling and exponential backoff for 429 errors"""
        if self.api_key:
            params['api_key'] = self.api_key
        params['file_type'] = 'json'

        url = f"{self.base_url}/{endpoint}"
        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            self._rate_limit()
            
            try:
                response = requests.get(url, params=params, timeout=30)
                
                # Handle 429 Too Many Requests with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"FRED API rate limit hit (429), retrying in {delay:.1f}s... (attempt {attempt+1}/{max_retries+1})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("FRED API rate limit exceeded, max retries reached")
                        raise requests.exceptions.HTTPError(f"429 Too Many Requests after {max_retries} retries")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries and ("timeout" in str(e).lower() or "connection" in str(e).lower()):
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"FRED API request failed ({e}), retrying in {delay:.1f}s... (attempt {attempt+1}/{max_retries+1})")
                    time.sleep(delay)
                    continue
                logger.error(f"FRED API request failed: {e}")
                raise

    def get_series_info(self, series_id: str) -> FREDSeries:
        """Get metadata for a FRED series"""
        try:
            data = self._make_request('series', {'series_id': series_id})
            series_data = data['seriess'][0]

            return FREDSeries(
                id=series_data['id'],
                title=series_data['title'],
                frequency=series_data['frequency'],
                units=series_data['units'],
                last_updated=series_data['last_updated'],
                notes=series_data.get('notes', '')
            )
        except Exception as e:
            logger.error(f"Failed to get series info for {series_id}: {e}")
            raise

    def get_series_data(self,
                        series_id: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        frequency: Optional[str] = None,
                        transform: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for a FRED series

        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            frequency: Data frequency (d, w, m, q, a)
            transform: Data transformation (lin, chg, ch1, pch, pc1, pca, cch, cca, log)

        Returns:
            DataFrame with date index and value column
        """
        params = {'series_id': series_id}

        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        if frequency:
            params['frequency'] = frequency
        if transform:
            params['transformation'] = transform

        try:
            data = self._make_request('series/observations', params)
            observations = data['observations']

            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Convert values to numeric, handling '.' as NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna()

            # Keep only value column
            df = df[['value']]
            df.columns = [series_id]

            return df

        except Exception as e:
            logger.error(f"Failed to get data for series {series_id}: {e}")
            raise

    def get_multiple_series(self,
                            series_ids: List[str],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for multiple FRED series and align them

        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            frequency: Data frequency for all series

        Returns:
            DataFrame with date index and columns for each series
        """
        series_data = []

        for series_id in series_ids:
            try:
                df = self.get_series_data(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                series_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to get data for {series_id}: {e}")
                continue

        if not series_data:
            return pd.DataFrame()

        # Combine all series
        combined = pd.concat(series_data, axis=1)
        combined = combined.sort_index()

        return combined

    def get_macro_dashboard(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict:
        """
        Get comprehensive macro economic dashboard

        Returns:
            Dict with categorized macro indicators
        """
        if not start_date:
            start_date = (
                datetime.now() -
                timedelta(
                    days=365 *
                    2)).strftime('%Y-%m-%d')

        # Organize series by category
        categories = {
            'inflation': ['CPIAUCSL', 'CPILFESL', 'PCEPI'],
            'employment': ['UNRATE', 'PAYEMS', 'CIVPART'],
            'growth': ['GDP', 'GDPC1', 'INDPRO'],
            'rates': ['FEDFUNDS', 'DGS10', 'DGS3MO', 'DGS2', 'DGS5', 'DGS30'],
            'money': ['M1SL', 'M2SL'],
            'markets': ['VIXCLS'],
            'housing': ['HOUST', 'CSUSHPISA'],
            'credit': ['TOTRESNS', 'WALCL']
        }

        dashboard = {}

        for category, series_list in categories.items():
            try:
                df = self.get_multiple_series(
                    series_ids=series_list,
                    start_date=start_date,
                    end_date=end_date,
                    frequency='m'  # Monthly frequency for dashboard
                )

                if not df.empty:
                    # Calculate summary statistics
                    latest_values = df.iloc[-1].to_dict()
                    monthly_changes = df.pct_change().iloc[-1].to_dict()
                    annual_changes = df.pct_change(
                        12).iloc[-1].to_dict() if len(df) >= 12 else {}

                    dashboard[category] = {
                        'data': df,
                        'latest_values': latest_values,
                        'monthly_changes': monthly_changes,
                        'annual_changes': annual_changes,
                        'series_count': len(df.columns),
                        'data_points': len(df),
                        'date_range': {
                            'start': df.index.min().strftime('%Y-%m-%d'),
                            'end': df.index.max().strftime('%Y-%m-%d')
                        }
                    }

            except Exception as e:
                logger.warning(f"Failed to get {category} data: {e}")
                dashboard[category] = {'error': str(e)}

        return dashboard

    def get_yield_curve(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get Treasury yield curve data

        Args:
            date: Specific date (YYYY-MM-DD) or None for latest

        Returns:
            DataFrame with maturities and yields
        """
        # Treasury yield series by maturity
        yield_series = {
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }

        # Convert maturity to years for plotting
        maturity_years = {
            '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3,
            '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
        }

        end_date = date if date else datetime.now().strftime('%Y-%m-%d')
        start_date = (
            datetime.strptime(
                end_date,
                '%Y-%m-%d') -
            timedelta(
                days=30)).strftime('%Y-%m-%d')

        try:
            df = self.get_multiple_series(
                series_ids=list(yield_series.values()),
                start_date=start_date,
                end_date=end_date,
                frequency='d'
            )

            if df.empty:
                return pd.DataFrame()

            # Get latest yields
            latest_yields = df.iloc[-1]

            # Create yield curve DataFrame
            curve_data = []
            for maturity, series_id in yield_series.items():
                if series_id in latest_yields and not pd.isna(
                        latest_yields[series_id]):
                    curve_data.append({
                        'maturity': maturity,
                        'years': maturity_years[maturity],
                        # Convert to decimal
                        'yield': latest_yields[series_id] / 100,
                        'series_id': series_id
                    })

            curve_df = pd.DataFrame(curve_data)
            curve_df = curve_df.sort_values('years')

            # Add curve metrics
            if len(curve_df) >= 2:
                curve_df['spread_vs_3m'] = curve_df['yield'] - \
                    curve_df.iloc[0]['yield']

                # Calculate curve steepness (10Y - 2Y spread)
                ten_year_yield = curve_df[curve_df['maturity'] ==
                                          '10Y']['yield'].iloc[0] if '10Y' in curve_df['maturity'].values else None
                two_year_yield = curve_df[curve_df['maturity'] ==
                                          '2Y']['yield'].iloc[0] if '2Y' in curve_df['maturity'].values else None

                steepness = (
                    ten_year_yield -
                    two_year_yield) if (
                    ten_year_yield and two_year_yield) else None

                return curve_df, {
                    'date': end_date,
                    'steepness_2s10s': steepness,
                    'short_rate': curve_df.iloc[0]['yield'],
                    'long_rate': curve_df.iloc[-1]['yield'],
                    'inversion_points': len(curve_df[curve_df['spread_vs_3m'] < 0])
                }

            return curve_df, {'date': end_date}

        except Exception as e:
            logger.error(f"Failed to get yield curve: {e}")
            raise

    def calculate_economic_surprises(self,
                                     series_id: str) -> pd.DataFrame:
        """
        Calculate economic data surprises vs expectations

        Args:
            series_id: FRED series identifier

        Returns:
            DataFrame with actual values, forecasts, and surprises
        """
        try:
            # Get 2 years of data for modeling
            start_date = (
                datetime.now() -
                timedelta(
                    days=730)).strftime('%Y-%m-%d')
            df = self.get_series_data(
                series_id=series_id, start_date=start_date)

            if df.empty or len(df) < 12:
                return pd.DataFrame()

            # Simple forecast using moving average (can be enhanced with ML)
            df['forecast'] = df[series_id].rolling(window=6).mean().shift(1)
            df['surprise'] = df[series_id] - df['forecast']
            df['surprise_zscore'] = (
                df['surprise'] - df['surprise'].mean()) / df['surprise'].std()

            # Add interpretation
            df['surprise_direction'] = np.where(
                df['surprise'] > 0, 'positive', 'negative')
            df['surprise_magnitude'] = np.where(
                df['surprise_zscore'].abs() > 2, 'high',
                np.where(df['surprise_zscore'].abs() > 1, 'medium', 'low')
            )

            return df.dropna()

        except Exception as e:
            logger.error(f"Failed to calculate surprises for {series_id}: {e}")
            raise

    def get_recession_indicators(self) -> Dict:
        """
        Get recession probability indicators

        Returns:
            Dict with various recession indicators
        """
        results = {}

        try:
            # Yield curve inversion
            df_10y = self.get_series_data('DGS10', start_date='2020-01-01')
            df_2y = self.get_series_data('DGS2', start_date='2020-01-01')

            if not df_10y.empty and not df_2y.empty:
                spread = df_10y['DGS10'] - df_2y['DGS2']
                current_spread = spread.iloc[-1]
                inversion_days = len(spread[spread < 0])

                results['yield_curve'] = {
                    'current_spread': current_spread,
                    'inverted': current_spread < 0,
                    'inversion_days_last_year': inversion_days,
                    'recession_signal': current_spread < -0.5
                }

            # Sahm Rule (unemployment rate indicator)
            df_unrate = self.get_series_data(
                'UNRATE', start_date='2020-01-01', frequency='m')
            if not df_unrate.empty and len(df_unrate) >= 12:
                current_rate = df_unrate['UNRATE'].iloc[-1]
                min_12m = df_unrate['UNRATE'].rolling(12).min().iloc[-1]
                sahm_indicator = current_rate - min_12m

                results['sahm_rule'] = {
                    'current_rate': current_rate,
                    'sahm_indicator': sahm_indicator,
                    'recession_signal': sahm_indicator >= 0.5
                }

            # Industrial production decline
            df_indpro = self.get_series_data(
                'INDPRO', start_date='2020-01-01', frequency='m')
            if not df_indpro.empty and len(df_indpro) >= 6:
                recent_change = (
                    df_indpro['INDPRO'].iloc[-1] / df_indpro['INDPRO'].iloc
                    [-6] - 1) * 100

                results['industrial_production'] = {
                    'six_month_change': recent_change,
                    'recession_signal': recent_change < -5
                }

            return results

        except Exception as e:
            logger.error(f"Failed to get recession indicators: {e}")
            return {'error': str(e)}
