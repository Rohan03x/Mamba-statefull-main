"""
SEC XBRL Data Provider for DCF Lab

This module provides access to SEC XBRL API endpoints for fundamental financial data:
- Company Facts API (all concepts by CIK)
- Company Concept API (one tag over time)
- EDGAR Submissions (filings discovery)
- CIK-Ticker mapping

Key Features:
- Free access to official SEC data
- Structured XBRL facts and concepts
- Historical filings discovery
- Automatic CIK resolution
- Rate limiting compliance
- Comprehensive error handling

Data Sources:
- SEC Company Facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
- SEC Company Concept: https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json
- SEC EDGAR Submissions: https://data.sec.gov/submissions/CIK{cik}.json
- SEC Company Tickers: https://www.sec.gov/files/company_tickers.json

Rate Limits:
- 10 requests per second per IP
- User-Agent header required
- Respectful usage patterns

Author: DCF Lab Team
Created: 2025-09-18
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SECConfig:
    """Configuration for SEC XBRL provider"""
    base_url: str = "https://data.sec.gov/api/xbrl"
    submissions_url: str = "https://data.sec.gov/submissions"
    tickers_url: str = "https://www.sec.gov/files/company_tickers.json"
    user_agent: str = "DCF Lab Research Tool (contact@dcflab.com)"
    rate_limit: float = 0.1  # 10 requests per second
    timeout: int = 30
    retries: int = 3
    cache_dir: str = "./cache/sec_xbrl"
    enable_cache: bool = True


class SECXBRLProvider:
    """
    SEC XBRL Data Provider

    Provides access to SEC XBRL APIs for fundamental financial data including:
    - Company facts (all XBRL concepts for a company)
    - Company concepts (historical data for specific XBRL tags)
    - EDGAR submissions and filings
    - CIK-ticker mapping and resolution
    """

    def __init__(self, config: Optional[SECConfig] = None):
        """Initialize SEC XBRL provider with configuration"""
        self.config = config or SECConfig()
        self._session = None
        self._cik_ticker_map = {}
        self._ticker_cik_map = {}
        self._last_request_time = 0

        # Create cache directory
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Get configured requests session with retry strategy"""
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

            # Set user agent (required by SEC)
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            })

        return self._session

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.config.rate_limit:
            sleep_time = self.config.rate_limit - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make rate-limited request to SEC API"""
        self._rate_limit()

        session = self._get_session()

        try:
            logger.debug(f"Making request to: {url}")
            response = session.get(
                url, params=params, timeout=self.config.timeout)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"SEC API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode SEC response: {e}")
            raise

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key"""
        return Path(self.config.cache_dir) / f"{cache_key}.json"

    def _load_from_cache(
            self,
            cache_key: str,
            max_age_hours: int = 24) -> Optional[Dict]:
        """Load data from cache if valid"""
        if not self.config.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=max_age_hours):
            logger.debug(f"Cache expired for {cache_key}")
            return None

        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        if not self.config.enable_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")

    def load_cik_ticker_mapping(self) -> Dict[str, Dict]:
        """
        Load CIK-ticker mapping from SEC

        Returns:
            Dict with CIK->ticker and ticker->CIK mappings
        """
        cache_key = "cik_ticker_mapping"
        cached_data = self._load_from_cache(
            cache_key, max_age_hours=168)  # 1 week cache

        if cached_data:
            self._cik_ticker_map = cached_data.get('cik_to_ticker', {})
            self._ticker_cik_map = cached_data.get('ticker_to_cik', {})
            logger.info(
                f"Loaded {len(self._cik_ticker_map)} CIK-ticker mappings from cache")
            return cached_data

        try:
            logger.info("Loading CIK-ticker mapping from SEC...")
            data = self._make_request(self.config.tickers_url)

            # Process the mapping
            cik_to_ticker = {}
            ticker_to_cik = {}

            for entry in data.values():
                cik = str(entry['cik_str']).zfill(10)  # Pad with zeros
                ticker = entry['ticker'].upper()
                title = entry['title']

                cik_to_ticker[cik] = {
                    'ticker': ticker,
                    'title': title
                }
                ticker_to_cik[ticker] = {
                    'cik': cik,
                    'title': title
                }

            # Cache the mapping
            mapping_data = {
                'cik_to_ticker': cik_to_ticker,
                'ticker_to_cik': ticker_to_cik,
                'updated': datetime.now().isoformat()
            }

            self._save_to_cache(cache_key, mapping_data)

            self._cik_ticker_map = cik_to_ticker
            self._ticker_cik_map = ticker_to_cik

            logger.info(f"Loaded {len(cik_to_ticker)} CIK-ticker mappings")
            return mapping_data

        except Exception as e:
            logger.error(f"Failed to load CIK-ticker mapping: {e}")
            raise

    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK from ticker symbol

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK string or None if not found
        """
        if not self._ticker_cik_map:
            self.load_cik_ticker_mapping()

        ticker = ticker.upper()
        company_info = self._ticker_cik_map.get(ticker)

        if company_info:
            return company_info['cik']

        return None

    def get_ticker_from_cik(self, cik: str) -> Optional[str]:
        """
        Get ticker from CIK

        Args:
            cik: Central Index Key

        Returns:
            Ticker symbol or None if not found
        """
        if not self._cik_ticker_map:
            self.load_cik_ticker_mapping()

        cik = str(cik).zfill(10)  # Pad with zeros
        company_info = self._cik_ticker_map.get(cik)

        if company_info:
            return company_info['ticker']

        return None

    def get_company_facts(self, identifier: str) -> Dict:
        """
        Get all XBRL facts for a company

        Args:
            identifier: Either ticker symbol or CIK

        Returns:
            Dict containing all company facts from SEC XBRL API
        """
        # Resolve CIK if ticker provided
        if identifier.isdigit() and len(identifier) <= 10:
            cik = str(identifier).zfill(10)
        else:
            cik = self.get_cik_from_ticker(identifier)
            if not cik:
                raise ValueError(
                    f"Could not resolve CIK for identifier: {identifier}")

        cache_key = f"company_facts_{cik}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)

        if cached_data:
            logger.info(f"Loaded company facts for CIK {cik} from cache")
            return cached_data

        try:
            url = f"{self.config.base_url}/companyfacts/CIK{cik}.json"
            logger.info(f"Fetching company facts for CIK {cik}...")

            data = self._make_request(url)

            # Add metadata
            data['_metadata'] = {
                'retrieved_at': datetime.now().isoformat(),
                'cik': cik,
                'identifier': identifier
            }

            self._save_to_cache(cache_key, data)
            logger.info(
                f"Retrieved company facts for {
                    data.get(
                        'entityName',
                        'Unknown')}")

            return data

        except Exception as e:
            logger.error(f"Failed to get company facts for CIK {cik}: {e}")
            raise

    def get_company_concept(
            self,
            identifier: str,
            taxonomy: str,
            tag: str) -> Dict:
        """
        Get historical data for specific XBRL concept

        Args:
            identifier: Either ticker symbol or CIK
            taxonomy: XBRL taxonomy (e.g., 'us-gaap', 'dei', 'srt')
            tag: XBRL tag (e.g., 'Revenues', 'Assets', 'CommonStockSharesOutstanding')

        Returns:
            Dict containing historical concept data
        """
        # Resolve CIK if ticker provided
        if identifier.isdigit() and len(identifier) <= 10:
            cik = str(identifier).zfill(10)
        else:
            cik = self.get_cik_from_ticker(identifier)
            if not cik:
                raise ValueError(
                    f"Could not resolve CIK for identifier: {identifier}")

        cache_key = f"company_concept_{cik}_{taxonomy}_{tag}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)

        if cached_data:
            logger.info(f"Loaded concept {tag} for CIK {cik} from cache")
            return cached_data

        try:
            url = f"{
                self.config.base_url}/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
            logger.info(f"Fetching concept {tag} for CIK {cik}...")

            data = self._make_request(url)

            # Add metadata
            data['_metadata'] = {
                'retrieved_at': datetime.now().isoformat(),
                'cik': cik,
                'identifier': identifier,
                'taxonomy': taxonomy,
                'tag': tag
            }

            self._save_to_cache(cache_key, data)
            logger.info(
                f"Retrieved concept {tag} for {
                    data.get(
                        'entityName',
                        'Unknown')}")

            return data

        except Exception as e:
            logger.error(f"Failed to get concept {tag} for CIK {cik}: {e}")
            raise

    def get_company_submissions(self, identifier: str) -> Dict:
        """
        Get EDGAR submissions for a company

        Args:
            identifier: Either ticker symbol or CIK

        Returns:
            Dict containing submission history and filings
        """
        # Resolve CIK if ticker provided
        if identifier.isdigit() and len(identifier) <= 10:
            cik = str(identifier).zfill(10)
        else:
            cik = self.get_cik_from_ticker(identifier)
            if not cik:
                raise ValueError(
                    f"Could not resolve CIK for identifier: {identifier}")

        cache_key = f"company_submissions_{cik}"
        cached_data = self._load_from_cache(cache_key, max_age_hours=24)

        if cached_data:
            logger.info(f"Loaded submissions for CIK {cik} from cache")
            return cached_data

        try:
            url = f"{self.config.submissions_url}/CIK{cik}.json"
            logger.info(f"Fetching submissions for CIK {cik}...")

            data = self._make_request(url)

            # Add metadata
            data['_metadata'] = {
                'retrieved_at': datetime.now().isoformat(),
                'cik': cik,
                'identifier': identifier
            }

            self._save_to_cache(cache_key, data)
            logger.info(
                f"Retrieved {
                    len(
                        data.get(
                            'filings',
                            {}).get(
                            'recent',
                            {}).get(
                            'accessionNumber',
                            []))} recent filings")

            return data

        except Exception as e:
            logger.error(f"Failed to get submissions for CIK {cik}: {e}")
            raise

    def get_financial_statements(
            self, identifier: str) -> Dict[str, pd.DataFrame]:
        """
        Extract key financial statement data from company facts

        Args:
            identifier: Either ticker symbol or CIK

        Returns:
            Dict with 'income_statement', 'balance_sheet', 'cash_flow' DataFrames
        """
        facts = self.get_company_facts(identifier)

        # Common GAAP tags for financial statements
        income_statement_tags = [
            'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
            'CostOfRevenue', 'CostOfGoodsAndServicesSold',
            'GrossProfit', 'OperatingIncomeLoss', 'NetIncomeLoss',
            'EarningsPerShareBasic', 'EarningsPerShareDiluted'
        ]

        balance_sheet_tags = [
            'Assets', 'AssetsCurrent', 'AssetsNoncurrent',
            'Liabilities', 'LiabilitiesCurrent', 'LiabilitiesNoncurrent',
            'StockholdersEquity', 'RetainedEarningsAccumulatedDeficit',
            'CommonStockSharesOutstanding'
        ]

        cash_flow_tags = [
            'NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInFinancingActivities',
            'CashAndCashEquivalentsAtCarryingValue'
        ]

        def extract_statement_data(
                tags: List[str],
                statement_name: str) -> pd.DataFrame:
            """Extract data for specific financial statement"""
            statement_data = []

            us_gaap = facts.get('facts', {}).get('us-gaap', {})

            for tag in tags:
                if tag in us_gaap:
                    concept_data = us_gaap[tag]

                    # Get units (usually 'USD' for financial data)
                    for unit, unit_data in concept_data.get(
                            'units', {}).items():
                        for item in unit_data:
                            # Skip items without end date (instantaneous items)
                            if 'end' not in item:
                                continue

                            statement_data.append(
                                {'tag': tag, 'label': concept_data.get(
                                    'label', tag),
                                 'description': concept_data.get(
                                     'description', ''),
                                 'value': item.get('val'),
                                 'unit': unit, 'end_date': item.get('end'),
                                 'start_date': item.get('start'),
                                 'filed_date': item.get('filed'),
                                 'form': item.get('form'),
                                 'fiscal_year': item.get('fy'),
                                 'fiscal_period': item.get('fp')})

            if statement_data:
                df = pd.DataFrame(statement_data)
                df['end_date'] = pd.to_datetime(df['end_date'])
                df = df.sort_values(['tag', 'end_date'])
                return df
            else:
                return pd.DataFrame()

        return {
            'income_statement': extract_statement_data(
                income_statement_tags,
                'Income Statement'),
            'balance_sheet': extract_statement_data(
                balance_sheet_tags,
                'Balance Sheet'),
            'cash_flow': extract_statement_data(
                cash_flow_tags,
                'Cash Flow'),
            'metadata': {
                'entity_name': facts.get('entityName'),
                'cik': facts.get('cik'),
                'retrieved_at': facts.get(
                    '_metadata',
                        {}).get('retrieved_at')}}

    def search_concepts(self, identifier: str, search_term: str) -> List[Dict]:
        """
        Search for XBRL concepts containing search term

        Args:
            identifier: Either ticker symbol or CIK
            search_term: Term to search for in concept labels/descriptions

        Returns:
            List of matching concepts with metadata
        """
        facts = self.get_company_facts(identifier)

        matches = []
        us_gaap = facts.get('facts', {}).get('us-gaap', {})

        search_lower = search_term.lower()

        for tag, concept_data in us_gaap.items():
            label = concept_data.get('label', '').lower()
            description = concept_data.get('description', '').lower()

            if (search_lower in label or
                search_lower in description or
                    search_lower in tag.lower()):

                # Count available data points
                total_values = 0
                for unit_data in concept_data.get('units', {}).values():
                    total_values += len(unit_data)

                matches.append({
                    'tag': tag,
                    'label': concept_data.get('label', tag),
                    'description': concept_data.get('description', ''),
                    'total_values': total_values,
                    'units': list(concept_data.get('units', {}).keys())
                })

        return sorted(matches, key=lambda x: x['total_values'], reverse=True)

    def get_peer_analysis_data(
            self,
            tickers: List[str],
            concepts: List[str]) -> pd.DataFrame:
        """
        Get comparative data for peer analysis

        Args:
            tickers: List of ticker symbols
            concepts: List of XBRL concept tags

        Returns:
            DataFrame with comparative financial data
        """
        peer_data = []

        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}...")

                for concept in concepts:
                    try:
                        data = self.get_company_concept(
                            ticker, 'us-gaap', concept)

                        entity_name = data.get('entityName', ticker)
                        units = data.get('units', {})

                        # Process USD values (most common for financial
                        # metrics)
                        if 'USD' in units:
                            for item in units['USD']:
                                if 'end' in item and 'val' in item:
                                    peer_data.append({
                                        'ticker': ticker,
                                        'entity_name': entity_name,
                                        'concept': concept,
                                        'value': item['val'],
                                        'end_date': item['end'],
                                        'fiscal_year': item.get('fy'),
                                        'fiscal_period': item.get('fp'),
                                        'form': item.get('form')
                                    })

                    except Exception as e:
                        logger.warning(
                            f"Failed to get {concept} for {ticker}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                continue

        if peer_data:
            df = pd.DataFrame(peer_data)
            df['end_date'] = pd.to_datetime(df['end_date'])
            return df.sort_values(['ticker', 'concept', 'end_date'])
        else:
            return pd.DataFrame()


def get_sec_provider(config: Optional[SECConfig] = None) -> SECXBRLProvider:
    """Factory function to create SEC XBRL provider"""
    return SECXBRLProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_sec_provider()

    # Load CIK-ticker mapping
    mapping = provider.load_cik_ticker_mapping()
    print(f"Loaded {len(mapping['cik_to_ticker'])} companies")

    # Example: Get Apple's financial data
    try:
        print("\n=== Apple Inc. Financial Data ===")

        # Get CIK
        aapl_cik = provider.get_cik_from_ticker("AAPL")
        print(f"AAPL CIK: {aapl_cik}")

        # Get company facts
        facts = provider.get_company_facts("AAPL")
        print(f"Entity: {facts.get('entityName')}")
        print(f"Available taxonomies: {list(facts.get('facts', {}).keys())}")

        # Get financial statements
        statements = provider.get_financial_statements("AAPL")

        for statement_name, df in statements.items():
            if statement_name != 'metadata' and not df.empty:
                print(f"\n{statement_name.title()} - Latest 5 entries:")
                print(df[['tag', 'label', 'value', 'end_date', 'form']].tail())

        # Search for revenue concepts
        revenue_concepts = provider.search_concepts("AAPL", "revenue")
        print(f"\nFound {len(revenue_concepts)} revenue-related concepts")
        for concept in revenue_concepts[:3]:
            print(f"- {concept['tag']}: {concept['label']}")

    except Exception as e:
        print(f"Error: {e}")
