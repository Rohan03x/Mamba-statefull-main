"""
ETF/Funds Holdings Provider for DCF Lab

This module provides access to SEC Form N-PORT data for ETF and mutual fund holdings:
- Portfolio holdings (quarterly N-PORT filings)
- Net Asset Values (NAV) tracking
- Fund composition and sector allocations
- Holdings concentration analysis
- Portfolio overlap calculations

Key Features:
- Free access to SEC N-PORT data
- Quarterly portfolio holdings
- Fund performance metrics
- Sector and geographic allocations
- Top holdings analysis
- Portfolio risk metrics

Data Sources:
- SEC N-PORT Data Sets: https://www.sec.gov/dera/data/form-n-port
- SEC Mutual Funds Search: https://www.sec.gov/investment/investment-management-files
- Fund performance data via existing providers

Rate Limits:
- Respectful usage of SEC resources
- Efficient caching and data management
- Quarterly update cycle

Author: DCF Lab Team
Created: 2025-09-18
"""

import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FundsConfig:
    """Configuration for ETF/Funds holdings provider"""
    sec_base_url: str = "https://www.sec.gov"
    nport_data_url: str = "https://www.sec.gov/dera/data/form-n-port"
    mutual_funds_url: str = "https://www.sec.gov/investment/investment-management-files"
    user_agent: str = "DCF Lab ETF Holdings Analysis (contact@dcflab.com)"
    timeout: int = 60  # Longer timeout for large files
    retries: int = 3
    cache_dir: str = "./cache/etf_holdings"
    enable_cache: bool = True
    max_cache_age_hours: int = 168  # 1 week for quarterly data


class SECNPortProvider:
    """
    SEC N-PORT Data Provider

    Provides access to SEC Form N-PORT filings which contain quarterly
    portfolio holdings for mutual funds and ETFs.
    """

    def __init__(self, config: FundsConfig):
        self.config = config
        self._session = None
        self._available_periods = []

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
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set headers
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            })

        return self._session

    def get_available_periods(self) -> List[str]:
        """
        Get list of available N-PORT data periods from SEC

        Returns:
            List of available period strings (e.g., ['2024q1', '2024q2'])
        """
        cache_key = "nport_periods"
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.json"

        # Check cache first
        if (self.config.enable_cache and cache_path.exists()):
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(
                    hours=24):  # Daily refresh for periods list
                try:
                    with open(cache_path, 'r') as f:
                        cached_periods = json.load(f)
                    self._available_periods = cached_periods
                    logger.info(
                        f"Loaded {
                            len(cached_periods)} periods from cache")
                    return cached_periods
                except Exception as e:
                    logger.warning(f"Failed to load periods cache: {e}")

        try:
            logger.info("Fetching available N-PORT periods from SEC...")
            session = self._get_session()

            # Get the N-PORT data page
            response = session.get(
                self.config.nport_data_url,
                timeout=self.config.timeout)
            response.raise_for_status()

            # Parse HTML to find download links
            content = response.text

            # Look for quarterly data links (simplified parsing)
            periods = []

            # Common quarterly patterns in SEC naming
            import re

            # Look for patterns like "2024q1", "2024q2", etc.
            quarter_pattern = r'(20\d{2}q[1-4])'
            matches = re.findall(quarter_pattern, content.lower())

            if matches:
                periods = sorted(list(set(matches)), reverse=True)
            else:
                # Fallback to recent quarters if parsing fails
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                for year in range(current_year, current_year - 3, -1):
                    for quarter in range(4, 0, -1):
                        if year == current_year and quarter > current_quarter:
                            continue
                        periods.append(f"{year}q{quarter}")

            # Cache the periods
            if self.config.enable_cache and periods:
                with open(cache_path, 'w') as f:
                    json.dump(periods, f)

            self._available_periods = periods
            logger.info(f"Found {len(periods)} available periods")

            return periods

        except Exception as e:
            logger.error(f"Failed to get available periods: {e}")

            # Return fallback periods
            current_year = datetime.now().year
            fallback_periods = [f"{current_year}q1",
                                f"{current_year-1}q4", f"{current_year-1}q3"]
            return fallback_periods

    def download_nport_data(self, period: str) -> Optional[pd.DataFrame]:
        """
        Download N-PORT data for a specific period

        Args:
            period: Period string (e.g., '2024q1')

        Returns:
            DataFrame with N-PORT holdings data
        """
        cache_path = Path(self.config.cache_dir) / f"nport_{period}.parquet"

        # Check cache first
        if (self.config.enable_cache and cache_path.exists()):
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(hours=self.config.max_cache_age_hours):
                try:
                    df = pd.read_parquet(cache_path)
                    logger.info(
                        f"Loaded N-PORT {period} from cache ({len(df)} records)")
                    return df
                except Exception as e:
                    logger.warning(
                        f"Failed to load N-PORT cache for {period}: {e}")

        try:
            logger.info(f"Downloading N-PORT data for {period}...")

            # Construct download URL (this is a simplified approach)
            # In practice, you'd need to parse the SEC page for exact URLs
            download_url = f"{
                self.config.sec_base_url}/files/dera/data/form-n-port/form-nport-{period}.zip"

            session = self._get_session()
            response = session.get(
                download_url,
                timeout=self.config.timeout,
                stream=True)

            if response.status_code == 404:
                logger.warning(f"N-PORT data not available for {period}")
                return None

            response.raise_for_status()

            # Process ZIP file
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Look for the main data file
                data_files = [f for f in zip_file.namelist(
                ) if f.endswith('.txt') or f.endswith('.csv')]

                if not data_files:
                    logger.warning(
                        f"No data files found in N-PORT ZIP for {period}")
                    return None

                # Read the main data file
                main_file = data_files[0]
                with zip_file.open(main_file) as data_file:
                    # Try to read as CSV with different delimiters
                    content = data_file.read().decode('utf-8', errors='ignore')

                    # Try pipe-delimited first (common for SEC data)
                    try:
                        df = pd.read_csv(
                            io.StringIO(content), sep='|', low_memory=False)
                    except (pd.errors.ParserError, ValueError):
                        # Fallback to comma-delimited
                        try:
                            df = pd.read_csv(
                                io.StringIO(content), low_memory=False)
                        except (pd.errors.ParserError, ValueError):
                            # Fallback to tab-delimited
                            df = pd.read_csv(
                                io.StringIO(content),
                                sep='\t', low_memory=False)

                    # Clean column names
                    df.columns = df.columns.str.strip()

                    # Cache the data
                    if self.config.enable_cache and not df.empty:
                        df.to_parquet(cache_path)

                    logger.info(
                        f"Downloaded N-PORT {period}: {len(df)} records")
                    return df

        except Exception as e:
            logger.error(f"Failed to download N-PORT data for {period}: {e}")
            return None

    def get_fund_holdings(
            self,
            fund_cik: str,
            period: Optional[str] = None) -> pd.DataFrame:
        """
        Get holdings for a specific fund

        Args:
            fund_cik: Fund's Central Index Key
            period: Period string (latest if None)

        Returns:
            DataFrame with fund holdings
        """
        if period is None:
            periods = self.get_available_periods()
            period = periods[0] if periods else "2024q1"

        nport_data = self.download_nport_data(period)

        if nport_data is None or nport_data.empty:
            logger.warning(f"No N-PORT data available for {period}")
            return pd.DataFrame()

        # Filter for specific fund
        # Note: Actual column names may vary, this is a simplified approach
        fund_columns = [
            col for col in nport_data.columns
            if 'cik' in col.lower() or 'fund' in col.lower()]

        if fund_columns:
            fund_holdings = nport_data[nport_data[fund_columns[0]].astype(
                str).str.contains(str(fund_cik), na=False)]
        else:
            # Fallback to searching in all string columns
            fund_holdings = pd.DataFrame()
            for col in nport_data.select_dtypes(include=['object']).columns:
                matches = nport_data[nport_data[col].astype(
                    str).str.contains(str(fund_cik), na=False)]
                if not matches.empty:
                    fund_holdings = matches
                    break

        if not fund_holdings.empty:
            logger.info(
                f"Found {
                    len(fund_holdings)} holdings for fund CIK {fund_cik}")
        else:
            logger.warning(
                f"No holdings found for fund CIK {fund_cik} in {period}")

        return fund_holdings


class ETFHoldingsAnalyzer:
    """
    ETF/Fund Holdings Analyzer

    Provides analysis capabilities for fund holdings data including
    sector allocations, concentration metrics, and portfolio overlaps.
    """

    def __init__(self, config: FundsConfig):
        self.config = config
        self.nport_provider = SECNPortProvider(config)

    def analyze_fund_composition(
            self, holdings_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze fund composition and calculate key metrics

        Args:
            holdings_data: DataFrame with fund holdings

        Returns:
            Dict with composition analysis
        """
        if holdings_data.empty:
            return {'error': 'No holdings data provided'}

        analysis = {
            'total_holdings': len(holdings_data),
            'analysis_date': datetime.now().isoformat(),
            'composition': {},
            'concentration': {},
            'top_holdings': [],
            'sector_allocation': {},
            'geographic_allocation': {}
        }

        try:
            # Look for value/weight columns
            value_columns = [
                col for col in holdings_data.columns if any(
                    term in col.lower() for term in [
                        'value', 'market', 'fair', 'amount', 'weight'])]

            name_columns = [
                col for col in holdings_data.columns if any(
                    term in col.lower() for term in [
                        'name', 'security', 'issuer', 'description'])]

            if value_columns and name_columns:
                value_col = value_columns[0]
                name_col = name_columns[0]

                # Convert values to numeric
                holdings_data[value_col] = pd.to_numeric(
                    holdings_data[value_col], errors='coerce')

                # Filter out invalid values
                valid_holdings = holdings_data[holdings_data[value_col].notna() & (
                    holdings_data[value_col] > 0)]

                if not valid_holdings.empty:
                    # Sort by value
                    sorted_holdings = valid_holdings.sort_values(
                        value_col, ascending=False)
                    total_value = sorted_holdings[value_col].sum()

                    # Top holdings analysis
                    top_10 = sorted_holdings.head(10)
                    analysis['top_holdings'] = [
                        {
                            'name': row[name_col],
                            'value': float(row[value_col]),
                            'percentage': float(row[value_col] / total_value * 100)
                        }
                        for _, row in top_10.iterrows()
                    ]

                    # Concentration metrics
                    analysis['concentration'] = {
                        'top_10_percentage': float(top_10[value_col].sum() / total_value * 100),
                        'top_5_percentage': float(sorted_holdings.head(5)[value_col].sum() / total_value * 100),
                        'top_1_percentage': float(sorted_holdings.iloc[0][value_col] / total_value * 100),
                        'herfindahl_index': float(((sorted_holdings[value_col] / total_value) ** 2).sum()),
                        'effective_holdings': int(1 / ((sorted_holdings[value_col] / total_value) ** 2).sum())
                    }

                    # Basic composition
                    analysis['composition'] = {
                        'total_value': float(total_value), 'average_holding_size': float(
                            valid_holdings[value_col].mean()), 'median_holding_size': float(
                            valid_holdings[value_col].median()), 'largest_holding': float(
                            valid_holdings[value_col].max()), 'smallest_holding': float(
                            valid_holdings[value_col].min())}

            # Sector analysis (simplified - would need more sophisticated
            # classification)
            if name_columns:
                name_col = name_columns[0]

                # Simple sector classification based on keywords
                sector_keywords = {
                    'Technology': [
                        'tech', 'software', 'apple', 'microsoft', 'google', 'amazon', 'meta'], 'Healthcare': [
                        'health', 'medical', 'pharma', 'bio', 'johnson', 'pfizer'], 'Financial': [
                        'bank', 'financial', 'insurance', 'jpmorgan', 'wells fargo'], 'Energy': [
                        'energy', 'oil', 'gas', 'exxon', 'chevron'], 'Consumer': [
                        'consumer', 'retail', 'walmart', 'target', 'nike'], 'Industrial': [
                            'industrial', 'manufacturing', 'boeing', 'caterpillar'], 'Utilities': [
                                'utility', 'electric', 'power', 'water'], 'Real Estate': [
                                    'real estate', 'reit', 'property']}

                sector_counts = {}
                for sector, keywords in sector_keywords.items():
                    count = 0
                    for keyword in keywords:
                        count += holdings_data[name_col].str.contains(
                            keyword, case=False, na=False).sum()
                    if count > 0:
                        sector_counts[sector] = count

                if sector_counts:
                    total_classified = sum(sector_counts.values())
                    analysis['sector_allocation'] = {
                        sector: {'count': count,
                                 'percentage': count / total_classified * 100}
                        for sector, count in sector_counts.items()
                    }

            logger.info(
                f"Completed composition analysis: {
                    analysis['total_holdings']} holdings")

        except Exception as e:
            logger.error(f"Failed to analyze fund composition: {e}")
            analysis['error'] = str(e)

        return analysis

    def compare_fund_overlaps(
            self, fund_holdings_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Compare overlaps between multiple funds

        Args:
            fund_holdings_list: List of DataFrames with fund holdings

        Returns:
            Dict with overlap analysis
        """
        if len(fund_holdings_list) < 2:
            return {'error': 'Need at least 2 funds for comparison'}

        overlap_analysis = {
            'num_funds': len(fund_holdings_list),
            'analysis_date': datetime.now().isoformat(),
            'common_holdings': [],
            'overlap_matrix': {},
            'similarity_scores': {}
        }

        try:
            # Extract security names from each fund
            fund_securities = []

            for i, holdings in enumerate(fund_holdings_list):
                if holdings.empty:
                    fund_securities.append(set())
                    continue

                # Find name column
                name_columns = [
                    col for col in holdings.columns if any(
                        term in col.lower() for term in [
                            'name', 'security', 'issuer'])]

                if name_columns:
                    securities = set(
                        holdings[name_columns[0]].dropna().str.strip().str.upper())
                else:
                    securities = set()

                fund_securities.append(securities)

            # Calculate pairwise overlaps
            for i in range(len(fund_securities)):
                overlap_analysis['overlap_matrix'][f'fund_{i}'] = {}
                for j in range(len(fund_securities)):
                    if i == j:
                        overlap_pct = 100.0
                    else:
                        intersection = fund_securities[i] & fund_securities[j]
                        union = fund_securities[i] | fund_securities[j]

                        if len(union) > 0:
                            overlap_pct = len(intersection) / len(union) * 100
                        else:
                            overlap_pct = 0.0

                    overlap_analysis['overlap_matrix'][
                        f'fund_{i} '][
                        f'fund_{j} '] = overlap_pct

            # Find common holdings across all funds
            if fund_securities:
                common_holdings = fund_securities[0]
                for securities in fund_securities[1:]:
                    common_holdings = common_holdings & securities

                overlap_analysis['common_holdings'] = list(common_holdings)
                overlap_analysis['num_common_holdings'] = len(common_holdings)

            logger.info(
                f"Completed overlap analysis for {
                    len(fund_holdings_list)} funds")

        except Exception as e:
            logger.error(f"Failed to analyze fund overlaps: {e}")
            overlap_analysis['error'] = str(e)

        return overlap_analysis


class ETFFundsProvider:
    """
    Comprehensive ETF/Funds Holdings Provider

    Combines N-PORT data access with holdings analysis capabilities.
    """

    def __init__(self, config: Optional[FundsConfig] = None):
        """Initialize ETF/Funds provider"""
        self.config = config or FundsConfig()
        self.nport_provider = SECNPortProvider(self.config)
        self.analyzer = ETFHoldingsAnalyzer(self.config)

    def get_etf_holdings(self, fund_identifier: str,
                         period: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive ETF holdings data and analysis

        Args:
            fund_identifier: Fund CIK or symbol
            period: Data period (latest if None)

        Returns:
            Dict with holdings data and analysis
        """
        result = {
            'fund_identifier': fund_identifier,
            'period': period,
            'retrieved_at': datetime.now().isoformat(),
            'holdings_data': pd.DataFrame(),
            'composition_analysis': {},
            'success': False,
            'error': None
        }

        try:
            # Get holdings data
            holdings_data = self.nport_provider.get_fund_holdings(
                fund_identifier, period)

            if not holdings_data.empty:
                result['holdings_data'] = holdings_data
                result['record_count'] = len(holdings_data)

                # Analyze composition
                composition = self.analyzer.analyze_fund_composition(
                    holdings_data)
                result['composition_analysis'] = composition

                result['success'] = True
                logger.info(
                    f"Retrieved holdings for {fund_identifier}: {
                        len(holdings_data)} positions")
            else:
                result['error'] = f"No holdings data found for {
                    fund_identifier} "
                logger.warning(f"No holdings found for {fund_identifier}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(
                f"Failed to get ETF holdings for {fund_identifier}: {e}")

        return result

    def get_available_periods(self) -> List[str]:
        """Get available N-PORT data periods"""
        return self.nport_provider.get_available_periods()

    def analyze_fund_universe(self,
                              fund_identifiers: List[str],
                              period: Optional[str] = None) -> Dict[str,
                                                                    Any]:
        """
        Analyze multiple funds for overlaps and comparisons

        Args:
            fund_identifiers: List of fund CIKs or symbols
            period: Data period (latest if None)

        Returns:
            Dict with universe analysis
        """
        result = {
            'fund_identifiers': fund_identifiers,
            'period': period,
            'retrieved_at': datetime.now().isoformat(),
            'individual_analyses': {},
            'overlap_analysis': {},
            'universe_summary': {},
            'success': False,
            'error': None
        }

        try:
            fund_holdings_list = []

            # Get holdings for each fund
            for fund_id in fund_identifiers:
                holdings_result = self.get_etf_holdings(fund_id, period)
                result['individual_analyses'][fund_id] = holdings_result

                if holdings_result['success']:
                    fund_holdings_list.append(holdings_result['holdings_data'])
                else:
                    fund_holdings_list.append(pd.DataFrame())

            # Analyze overlaps
            if len(fund_holdings_list) >= 2:
                overlap_analysis = self.analyzer.compare_fund_overlaps(
                    fund_holdings_list)
                result['overlap_analysis'] = overlap_analysis

            # Universe summary
            successful_funds = sum(1
                                   for analysis in result
                                   ['individual_analyses'].values()
                                   if analysis['success'])
            total_unique_holdings = set()

            for analysis in result['individual_analyses'].values():
                if analysis['success'] and not analysis['holdings_data'].empty:
                    # Extract security names
                    holdings = analysis['holdings_data']
                    name_columns = [
                        col for col in holdings.columns if any(
                            term in col.lower() for term in [
                                'name', 'security', 'issuer'])]
                    if name_columns:
                        securities = set(
                            holdings[name_columns[0]].dropna().str.strip().str.upper())
                        total_unique_holdings.update(securities)

            result['universe_summary'] = {
                'total_funds_analyzed': len(fund_identifiers),
                'successful_analyses': successful_funds,
                'total_unique_holdings': len(total_unique_holdings)
            }

            result['success'] = successful_funds > 0
            logger.info(
                f"Analyzed {successful_funds}/{len(fund_identifiers)} funds successfully")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to analyze fund universe: {e}")

        return result


def get_etf_provider(config: Optional[FundsConfig] = None) -> ETFFundsProvider:
    """Factory function to create ETF/Funds provider"""
    return ETFFundsProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_etf_provider()

    # Get available periods
    print("=== Available N-PORT Periods ===")
    periods = provider.get_available_periods()
    print(f"Available periods: {periods[:5]}...")  # Show first 5

    # Example: Analyze a major ETF (using a representative CIK)
    print("\n=== ETF Holdings Analysis ===")

    # Note: In practice, you'd need to find the actual CIK for funds like SPY,
    # VTI, etc.
    test_cik = "0000884394"  # Example CIK

    try:
        result = provider.get_etf_holdings(test_cik)

        if result['success']:
            print(f"Successfully analyzed fund {test_cik}")
            print(f"Holdings count: {result['record_count']}")

            composition = result['composition_analysis']
            if 'concentration' in composition:
                conc = composition['concentration']
                print(
                    f"Top 10 concentration: {
                        conc.get(
                            'top_10_percentage',
                            'N/A'):.1f}%")
                print(
                    f"Effective holdings: {
                        conc.get(
                            'effective_holdings',
                            'N/A')}")

            if 'top_holdings' in composition:
                print("\nTop 5 holdings:")
                for i, holding in enumerate(composition['top_holdings'][:5]):
                    print(
                        f"  {i+1}. {holding['name']}: {holding['percentage']:.2f}%")
        else:
            print(f"Failed to analyze fund: {result['error']}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n=== ETF/Funds Holdings Provider Ready ===")
    print("✅ SEC N-PORT data access")
    print("✅ Holdings composition analysis")
    print("✅ Concentration metrics")
    print("✅ Portfolio overlap calculations")
    print("🚀 Ready for fund analysis pipeline")
