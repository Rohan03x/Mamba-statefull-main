"""
SEC XBRL API Provider
====================

Integration with the SEC's EDGAR XBRL API for official company fundamentals,
financial statements, and regulatory filings.

SEC EDGAR API provides:
- Company facts (financial data)
- Filing information
- Insider trading data
- Ownership information
- Structured XBRL financial statements

API Documentation: https://www.sec.gov/edgar/sec-api-documentation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """SEC filing metadata"""
    accession_number: str
    filing_date: str
    report_date: str
    form: str
    company_name: str
    cik: str


class SECXBRLProvider:
    """
    SEC EDGAR XBRL API provider for official financial data

    Key Features:
    - Company facts from official SEC filings
    - Historical financial statements (10-K, 10-Q)
    - Real-time filing information
    - Insider trading data
    - No API key required (rate limited)
    """

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize SEC XBRL provider

        Args:
            user_agent: Required User-Agent header for SEC API compliance
                       Should include company/email per SEC guidelines
        """
        self.user_agent = user_agent or "DCF-Suite/1.0 (research@example.com)"
        self.base_url = "https://data.sec.gov"
        self._cache = {}
        self._last_request_time = 0
        self._min_request_interval = 0.1  # SEC rate limit: 10 requests per second

        # Common headers required by SEC
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }

    def _rate_limit(self):
        """Rate limiting for SEC API (10 requests per second)"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str) -> Dict:
        """Make API request with proper headers and error handling"""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC API request failed: {e}")
            raise

    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) from stock ticker

        Args:
            ticker: Stock symbol (e.g., 'AAPL')

        Returns:
            CIK string with leading zeros, or None if not found
        """
        # Use hardcoded mapping for common stocks for reliability
        common_ciks = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'META': '0001326801',
            'NVDA': '0001045810',
            'IBM': '0000051143',
            'JPM': '0000019617',
            'JNJ': '0000200406'
        }

        ticker_upper = ticker.upper()
        if ticker_upper in common_ciks:
            return common_ciks[ticker_upper]

        try:
            # Try to get from SEC API as backup
            data = self._make_request("files/company_tickers_exchange.json")
            cik = self._search_ticker_in_data(data, ticker_upper)
            return cik

        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None

    def _search_ticker_in_data(self, data: Dict, ticker: str) -> Optional[str]:
        """Search for ticker in SEC data structure"""
        # Search through the data structure
        if 'data' in data:
            for entry in data['data']:
                if len(entry) >= 2 and entry[1] == ticker:
                    return str(entry[0]).zfill(10)

        # Fallback search methods
        for key, entry in data.get('fields', {}).items():
            if isinstance(
                entry,
                dict) and entry.get(
                'ticker',
                    '').upper() == ticker:
                return str(entry['cik']).zfill(10)

        return None

    def get_company_facts(self, cik: str) -> Dict:
        """
        Get all company facts for a given CIK

        Args:
            cik: Central Index Key (10-digit string with leading zeros)

        Returns:
            Dict with comprehensive company facts
        """
        try:
            endpoint = f"api/xbrl/companyfacts/CIK{cik}.json"
            data = self._make_request(endpoint)

            return {
                'cik': data.get('cik'),
                'name': data.get('entityName'),
                'facts': data.get('facts', {}),
                'retrieved_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get company facts for CIK {cik}: {e}")
            return {}

    def get_financial_statements(self, cik: str,
                                 concept: str = "Assets",
                                 taxonomy: str = "us-gaap") -> pd.DataFrame:
        """
        Get financial statement data for a specific concept

        Args:
            cik: Central Index Key
            concept: XBRL concept (e.g., 'Assets', 'Revenues', 'NetIncomeLoss')
            taxonomy: Taxonomy (us-gaap, dei, invest)

        Returns:
            DataFrame with historical financial data
        """
        try:
            facts = self.get_company_facts(cik)

            if not facts or 'facts' not in facts:
                return pd.DataFrame()

            # Navigate to the specific concept
            if taxonomy not in facts['facts']:
                logger.warning(f"Taxonomy {taxonomy} not found for CIK {cik}")
                return pd.DataFrame()

            if concept not in facts['facts'][taxonomy]:
                logger.warning(
                    f"Concept {concept} not found in {taxonomy} for CIK {cik}")
                return pd.DataFrame()

            concept_data = facts['facts'][taxonomy][concept]

            # Extract units (usually USD)
            if 'units' not in concept_data:
                return pd.DataFrame()

            # Process all unit types (USD, shares, etc.)
            all_data = []
            for unit, values in concept_data['units'].items():
                for entry in values:
                    all_data.append({
                        'end_date': entry.get('end'),
                        'value': entry.get('val'),
                        'unit': unit,
                        'form': entry.get('form'),
                        'frame': entry.get('frame'),
                        'accession_number': entry.get('accn'),
                        'filed_date': entry.get('filed')
                    })

            if not all_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['end_date'] = pd.to_datetime(df['end_date'])
            df['filed_date'] = pd.to_datetime(df['filed_date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            # Sort by end date
            df = df.sort_values('end_date')

            return df

        except Exception as e:
            logger.error(
                f"Failed to get financial statements for CIK {cik}, concept {concept}: {e}")
            return pd.DataFrame()

    def get_key_financial_metrics(self, cik: str) -> Dict:
        """
        Get key financial metrics from SEC filings

        Args:
            cik: Central Index Key

        Returns:
            Dict with key financial metrics
        """
        key_concepts = {
            'assets': 'Assets',
            'liabilities': 'Liabilities',
            'stockholders_equity': 'StockholdersEquity',
            'revenues': 'Revenues',
            'net_income': 'NetIncomeLoss',
            'cash_and_equivalents': 'CashAndCashEquivalentsAtCarryingValue',
            'total_debt': 'DebtCurrent',
            'shares_outstanding': 'CommonStockSharesOutstanding'
        }

        metrics = {}

        for metric_name, concept in key_concepts.items():
            try:
                df = self.get_financial_statements(cik, concept)
                if not df.empty:
                    # Get latest annual data (10-K forms)
                    annual_data = df[df['form'] == '10-K'].copy()
                    if not annual_data.empty:
                        latest = annual_data.iloc[-1]
                        metrics[metric_name] = {
                            'value': latest['value'],
                            'date': latest['end_date'].strftime('%Y-%m-%d'),
                            'unit': latest['unit'],
                            'form': latest['form']
                        }

                    # Also get latest quarterly data (10-Q forms)
                    quarterly_data = df[df['form'] == '10-Q'].copy()
                    if not quarterly_data.empty:
                        latest_q = quarterly_data.iloc[-1]
                        metrics[f"{metric_name}_quarterly"] = {
                            'value': latest_q['value'],
                            'date': latest_q['end_date'].strftime('%Y-%m-%d'),
                            'unit': latest_q['unit'],
                            'form': latest_q['form']
                        }

            except Exception as e:
                logger.warning(
                    f"Failed to get {metric_name} for CIK {cik}: {e}")
                continue

        return metrics

    def get_recent_filings(self, cik: str,
                           form_types: List[str] = None,
                           limit: int = 10) -> List[SECFiling]:
        """
        Get recent filings for a company

        Args:
            cik: Central Index Key
            form_types: List of form types to filter (e.g., ['10-K', '10-Q'])
            limit: Maximum number of filings to return

        Returns:
            List of SECFiling objects
        """
        try:
            endpoint = f"submissions/CIK{cik}.json"
            data = self._make_request(endpoint)

            if 'filings' not in data or 'recent' not in data['filings']:
                return []

            recent = data['filings']['recent']
            filings = []

            # Combine all filing data
            for i in range(len(recent.get('accessionNumber', []))):
                form = recent['form'][i]

                # Filter by form type if specified
                if form_types and form not in form_types:
                    continue

                filing = SECFiling(
                    accession_number=recent['accessionNumber'][i],
                    filing_date=recent['filingDate'][i],
                    report_date=recent['reportDate'][i],
                    form=form,
                    company_name=data.get('name', ''),
                    cik=cik
                )

                filings.append(filing)

                if len(filings) >= limit:
                    break

            return filings

        except Exception as e:
            logger.error(f"Failed to get recent filings for CIK {cik}: {e}")
            return []

    def get_comprehensive_financials(self, ticker: str) -> Dict:
        """
        Get comprehensive financial data for a ticker

        Args:
            ticker: Stock symbol

        Returns:
            Dict with comprehensive financial information
        """
        try:
            # Get CIK from ticker
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                return {'error': f'CIK not found for ticker {ticker}'}

            result = {
                'ticker': ticker,
                'cik': cik,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'SEC EDGAR XBRL'
            }

            # Get company facts
            facts = self.get_company_facts(cik)
            if facts:
                result['company_name'] = facts.get('name')
                result['company_facts'] = facts

            # Get key financial metrics
            metrics = self.get_key_financial_metrics(cik)
            if metrics:
                result['key_metrics'] = metrics

            # Get recent filings
            filings = self.get_recent_filings(
                cik, ['10-K', '10-Q', '8-K'], limit=5)
            result['recent_filings'] = [
                {
                    'form': f.form,
                    'filing_date': f.filing_date,
                    'report_date': f.report_date,
                    'accession_number': f.accession_number
                }
                for f in filings
            ]

            return result

        except Exception as e:
            logger.error(
                f"Failed to get comprehensive financials for {ticker}: {e}")
            return {'error': str(e)}

    def calculate_financial_ratios(self, metrics: Dict) -> Dict:
        """
        Calculate financial ratios from SEC metrics

        Args:
            metrics: Key financial metrics dict

        Returns:
            Dict with calculated ratios
        """
        ratios = {}

        try:
            ratios.update(self._calculate_leverage_ratios(metrics))
            ratios.update(self._calculate_profitability_ratios(metrics))
            ratios.update(self._calculate_efficiency_ratios(metrics))
        except Exception as e:
            logger.warning(f"Error calculating ratios: {e}")

        return ratios

    def _calculate_leverage_ratios(self, metrics: Dict) -> Dict:
        """Calculate leverage ratios"""
        ratios = {}

        if 'total_debt' in metrics and 'stockholders_equity' in metrics:
            debt = metrics['total_debt']['value']
            equity = metrics['stockholders_equity']['value']
            if equity and equity != 0:
                ratios['debt_to_equity'] = debt / equity

        return ratios

    def _calculate_profitability_ratios(self, metrics: Dict) -> Dict:
        """Calculate profitability ratios"""
        ratios = {}

        # Return on assets (ROA)
        if 'net_income' in metrics and 'assets' in metrics:
            net_income = metrics['net_income']['value']
            assets = metrics['assets']['value']
            if assets and assets != 0:
                ratios['return_on_assets'] = net_income / assets

        # Return on equity (ROE)
        if 'net_income' in metrics and 'stockholders_equity' in metrics:
            net_income = metrics['net_income']['value']
            equity = metrics['stockholders_equity']['value']
            if equity and equity != 0:
                ratios['return_on_equity'] = net_income / equity

        return ratios

    def _calculate_efficiency_ratios(self, metrics: Dict) -> Dict:
        """Calculate efficiency ratios"""
        ratios = {}

        # Asset turnover
        if 'revenues' in metrics and 'assets' in metrics:
            revenues = metrics['revenues']['value']
            assets = metrics['assets']['value']
            if assets and assets != 0:
                ratios['asset_turnover'] = revenues / assets

        return ratios
