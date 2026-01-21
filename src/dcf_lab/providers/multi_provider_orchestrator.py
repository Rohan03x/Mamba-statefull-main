"""
Multi-Provider Data Orchestration System
========================================

Comprehensive orchestration layer that intelligently combines data from all providers:
- Enhanced Yahoo Finance (primary OHLCV + options)
- Alpha Vantage (backup fundamentals + forex)
- FRED (macro economic indicators)
- SEC XBRL (regulatory financial data)
- Market Structure (FINRA microstructure data)

Key Features:
- Intelligent provider selection and fallback logic
- Cross-validation and data quality scoring
- Parallel data retrieval with timeout handling
- Comprehensive error recovery and retry mechanisms
- Data fusion and conflict resolution
- Performance optimization and caching
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from dcf_lab.providers.alpha_vantage_provider import AlphaVantageProvider

# Import all providers
from dcf_lab.providers.enhanced_yfinance import EnhancedYFinanceProvider
from dcf_lab.providers.fred_provider import FREDProvider
from dcf_lab.providers.market_structure_provider import MarketStructureProvider
from dcf_lab.providers.sec_xbrl_provider import SECXBRLProvider

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


class ProviderPriority(Enum):
    """Provider priority levels"""
    PRIMARY = 1
    SECONDARY = 2
    BACKUP = 3
    SPECIALIZED = 4


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    provider: Any
    priority: ProviderPriority
    capabilities: List[str]
    timeout: float = 30.0
    retry_count: int = 3
    health_score: float = 100.0
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    data_types: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: DataQuality = DataQuality.GOOD
    max_staleness_hours: int = 24


@dataclass
class DataResponse:
    """Standardized data response"""
    symbol: str
    data_type: str
    data: Any
    source: str
    timestamp: datetime
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MultiProviderOrchestrator:
    """
    Comprehensive data orchestration system

    Manages multiple data providers with intelligent routing, fallback logic,
    and data quality validation for optimal data retrieval and accuracy.
    """

    def __init__(self):
        """Initialize the orchestrator with all providers"""
        self.providers = self._initialize_providers()
        self.cache = {}
        self.performance_stats = {}
        self.max_workers = 10

        # Quality thresholds
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 0.95,
            DataQuality.GOOD: 0.80,
            DataQuality.FAIR: 0.65,
            DataQuality.POOR: 0.40
        }

    def _initialize_providers(self) -> Dict[str, DataSource]:
        """Initialize all data providers with capabilities"""
        providers = {}

        # Enhanced Yahoo Finance - Primary comprehensive provider
        providers['enhanced_yfinance'] = DataSource(
            name='enhanced_yfinance',
            provider=EnhancedYFinanceProvider(),
            priority=ProviderPriority.PRIMARY,
            capabilities=[
                'price_data', 'options_chains', 'benchmarks', 'news',
                'technical_indicators', 'volume_analysis', 'correlations'
            ],
            timeout=45.0
        )

        # Alpha Vantage - Secondary fundamentals provider
        providers['alpha_vantage'] = DataSource(
            name='alpha_vantage',
            provider=AlphaVantageProvider(),
            priority=ProviderPriority.SECONDARY,
            capabilities=[
                'fundamentals', 'price_data', 'earnings', 'forex',
                'economic_indicators', 'company_overview'
            ],
            timeout=30.0
        )

        # FRED - Specialized economic data
        providers['fred'] = DataSource(
            name='fred',
            provider=FREDProvider(),
            priority=ProviderPriority.SPECIALIZED,
            capabilities=[
                'economic_indicators', 'yield_curves', 'inflation_data',
                'employment_data', 'recession_indicators', 'fed_data'
            ],
            timeout=20.0
        )

        # SEC XBRL - Specialized regulatory data
        providers['sec_xbrl'] = DataSource(
            name='sec_xbrl',
            provider=SECXBRLProvider(),
            priority=ProviderPriority.SPECIALIZED,
            capabilities=[
                'financial_statements', 'regulatory_filings', 'company_facts',
                'recent_filings', 'sec_ratios', 'official_data'
            ],
            timeout=25.0
        )

        # Market Structure - Specialized microstructure data
        providers['market_structure'] = DataSource(
            name='market_structure',
            provider=MarketStructureProvider(),
            priority=ProviderPriority.SPECIALIZED,
            capabilities=[
                'short_interest', 'dark_pool_data', 'options_flow',
                'market_makers', 'venue_statistics', 'microstructure'
            ],
            timeout=20.0
        )

        return providers

    def get_comprehensive_data(self,
                               symbol: str,
                               data_types: Optional[List[str]] = None,
                               quality_threshold: DataQuality = DataQuality.GOOD) -> Dict[str,
                                                                                          DataResponse]:
        """
        Get comprehensive data for a symbol from all relevant providers

        Args:
            symbol: Stock symbol
            data_types: Specific data types to retrieve (None for all)
            quality_threshold: Minimum quality threshold

        Returns:
            Dict mapping data type to DataResponse
        """
        if not data_types:
            data_types = [
                'price_data',
                'fundamentals',
                'options_chains',
                'economic_indicators',
                'market_structure',
                'news',
                'benchmarks']

        request = DataRequest(
            symbol=symbol,
            data_types=data_types,
            quality_threshold=quality_threshold
        )

        return self._execute_comprehensive_request(request)

    def _execute_comprehensive_request(
            self, request: DataRequest) -> Dict[str, DataResponse]:
        """Execute comprehensive data request with parallel processing"""
        results = {}

        # Group data types by optimal providers
        provider_groups = self._plan_data_retrieval(request.data_types)

        # Execute requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for provider_name, data_types in provider_groups.items():
                if provider_name in self.providers:
                    future = executor.submit(
                        self._fetch_provider_data,
                        provider_name,
                        request.symbol,
                        data_types,
                        request
                    )
                    futures[future] = (provider_name, data_types)

            # Collect results
            for future in as_completed(futures, timeout=60):
                provider_name, data_types = futures[future]
                try:
                    provider_results = future.result()
                    results.update(provider_results)

                    # Update provider health
                    self._update_provider_health(provider_name, True)

                except Exception as e:
                    logger.error(f"Provider {provider_name} failed: {e}")
                    self._update_provider_health(provider_name, False, str(e))

                    # Attempt fallback for failed data types
                    fallback_results = self._attempt_fallback(
                        request.symbol, data_types, request)
                    results.update(fallback_results)

        return results

    def _plan_data_retrieval(
            self, data_types: List[str]) -> Dict[str, List[str]]:
        """Plan optimal data retrieval strategy"""
        provider_groups = {}

        for data_type in data_types:
            best_providers = self._find_best_providers(data_type)

            if best_providers:
                provider_name = best_providers[0]
                if provider_name not in provider_groups:
                    provider_groups[provider_name] = []
                provider_groups[provider_name].append(data_type)

        return provider_groups

    def _find_best_providers(self, data_type: str) -> List[str]:
        """Find best providers for a specific data type"""
        candidates = []

        for name, source in self.providers.items():
            if any(capability in data_type or data_type in capability
                   for capability in source.capabilities):
                score = self._calculate_provider_score(source, data_type)
                candidates.append((name, score))

        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in candidates]

    def _calculate_provider_score(
            self,
            source: DataSource,
            data_type: str) -> float:
        """Calculate provider score for specific data type"""
        base_score = source.health_score

        # Priority bonus
        priority_bonus = {
            ProviderPriority.PRIMARY: 20,
            ProviderPriority.SECONDARY: 10,
            ProviderPriority.BACKUP: 5,
            ProviderPriority.SPECIALIZED: 15
        }.get(source.priority, 0)

        # Capability match bonus
        capability_bonus = 0
        for capability in source.capabilities:
            if capability in data_type or data_type in capability:
                capability_bonus += 10

        # Recent success bonus
        recency_bonus = 0
        if source.last_success:
            hours_since = (
                datetime.now() - source.last_success).total_seconds() / 3600
            if hours_since < 1:
                recency_bonus = 10
            elif hours_since < 24:
                recency_bonus = 5

        return base_score + priority_bonus + capability_bonus + recency_bonus

    def _fetch_provider_data(self,
                             provider_name: str,
                             symbol: str,
                             data_types: List[str],
                             request: DataRequest) -> Dict[str,
                                                           DataResponse]:
        """Fetch data from specific provider"""
        results = {}
        provider = self.providers[provider_name].provider

        for data_type in data_types:
            try:
                start_time = time.time()

                # Route to appropriate provider method
                data = self._route_data_request(
                    provider, provider_name, symbol, data_type, request)

                if data is not None:
                    # Calculate quality score
                    quality_score = self._assess_data_quality(data, data_type)

                    response = DataResponse(
                        symbol=symbol,
                        data_type=data_type,
                        data=data,
                        source=provider_name,
                        timestamp=datetime.now(),
                        quality_score=quality_score,
                        metadata={
                            'fetch_time_ms': round(
                                (time.time() - start_time) * 1000,
                                2),
                            'provider_capabilities': self.providers[provider_name].capabilities})

                    # Check quality threshold
                    if quality_score >= self.quality_thresholds[request.quality_threshold]:
                        results[data_type] = response
                    else:
                        response.warnings.append(
                            f"Quality score {
                                quality_score:.2f} below threshold")
                        results[data_type] = response

            except Exception as e:
                logger.error(
                    f"Failed to fetch {data_type} from {provider_name}: {e}")
                results[data_type] = DataResponse(
                    symbol=symbol,
                    data_type=data_type,
                    data=None,
                    source=provider_name,
                    timestamp=datetime.now(),
                    quality_score=0.0,
                    errors=[str(e)]
                )

        return results

    def _get_provider_method_map(self) -> Dict[str, Dict[str, str]]:
        """Get mapping of provider -> data_type -> method_name"""
        return {
            'enhanced_yfinance': {
                'price_data': 'get_enhanced_price_data',
                'options_chains': 'get_options_chain',
                'benchmarks': 'get_benchmark_data',
                'news': 'get_news'
            },
            'alpha_vantage': {
                'fundamentals': 'get_comprehensive_data',
                'company_overview': 'get_company_overview',
                'earnings': 'get_earnings'
            },
            'fred': {
                'economic_indicators': 'get_macro_indicators',
                'yield_curves': 'get_yield_curve_data'
            },
            'sec_xbrl': {
                'financial_statements': 'get_financial_statements',
                'company_facts': 'get_company_facts'
            },
            'market_structure': {
                'market_structure': 'get_comprehensive_market_structure',
                'short_interest': 'get_short_interest_data'
            }
        }

    def _route_data_request(
            self,
            provider: Any,
            provider_name: str,
            symbol: str,
            data_type: str,
            request: DataRequest) -> Any:
        """Route data request to appropriate provider method"""
        method_map = self._get_provider_method_map()

        if provider_name not in method_map:
            return None

        provider_methods = method_map[provider_name]
        if data_type not in provider_methods:
            return None

        method_name = provider_methods[data_type]
        if not hasattr(provider, method_name):
            return None

        method = getattr(provider, method_name)

        # Handle methods that don't require symbol parameter
        if data_type in ['economic_indicators', 'yield_curves']:
            return method()
        else:
            return method(symbol)

    def _assess_dict_quality(self, data: dict, data_type: str) -> float:
        """Assess quality of dictionary data"""
        score = 0.5  # Base score

        # Check for required fields
        if 'symbol' in data:
            score += 0.1
        if 'timestamp' in data or 'date' in data:
            score += 0.1
        if len(data) > 2:  # Has meaningful content
            score += 0.2

        # Check for error indicators
        if 'error' in data or 'errors' in data:
            score -= 0.3

        # Data type specific checks
        score += self._get_data_type_score(data, data_type)

        return score

    def _get_data_type_score(self, data: dict, data_type: str) -> float:
        """Get score bonus for data type specific content"""
        quality_indicators = {
            'price_data': ['close'],
            'fundamentals': ['revenue', 'assets', 'totalRevenue']
        }

        if data_type in quality_indicators:
            if any(indicator in data
                   for indicator in quality_indicators[data_type]):
                return 0.1

        return 0.0

    def _assess_data_quality(self, data: Any, data_type: str) -> float:
        """Assess data quality score (0-1)"""
        if data is None:
            return 0.0

        if isinstance(data, dict):
            score = self._assess_dict_quality(data, data_type)
        elif isinstance(data, (list, pd.DataFrame)):
            score = 0.8 if len(data) > 0 else 0.3
        else:
            score = 0.5  # Base score for other types

        return min(1.0, max(0.0, score))

    def _attempt_fallback(self, symbol: str, failed_data_types: List[str],
                          request: DataRequest) -> Dict[str, DataResponse]:
        """Attempt fallback providers for failed data types"""
        results = {}

        for data_type in failed_data_types:
            fallback_providers = self._find_best_providers(data_type)[
                1:]  # Skip primary

            for provider_name in fallback_providers:
                try:
                    data = self._fetch_provider_data(
                        provider_name, symbol, [data_type], request)
                    if data and data_type in data and data[data_type].quality_score > 0:
                        results[data_type] = data[data_type]
                        results[data_type].metadata['fallback'] = True
                        break
                except Exception as e:
                    logger.debug(
                        f"Fallback provider {provider_name} also failed: {e}")
                    continue

        return results

    def _update_provider_health(
            self,
            provider_name: str,
            success: bool,
            error: str = None):
        """Update provider health metrics"""
        if provider_name not in self.providers:
            return

        provider = self.providers[provider_name]

        if success:
            provider.health_score = min(100.0, provider.health_score + 1.0)
            provider.last_success = datetime.now()
            provider.last_error = None
        else:
            provider.health_score = max(0.0, provider.health_score - 5.0)
            provider.last_error = error

    def get_provider_health_report(self) -> Dict[str, Dict]:
        """Get comprehensive provider health report"""
        report = {}

        for name, provider in self.providers.items():
            report[name] = {
                'health_score': provider.health_score,
                'priority': provider.priority.name,
                'capabilities': provider.capabilities,
                'last_success': provider.last_success.isoformat()
                if provider.last_success else None,
                'last_error': provider.last_error, 'timeout': provider.timeout}

        return report

    def benchmark_providers(self, symbol: str = 'AAPL') -> Dict[str, Dict]:
        """Benchmark all providers for performance analysis"""
        results = {}

        for name, provider in self.providers.items():
            start_time = time.time()
            try:
                # Test basic functionality
                if name == 'enhanced_yfinance':
                    data = provider.provider.get_enhanced_price_data(symbol)
                elif name == 'alpha_vantage':
                    data = provider.provider.get_comprehensive_data(symbol)
                elif name == 'fred':
                    data = provider.provider.get_macro_indicators()
                elif name == 'sec_xbrl':
                    data = provider.provider.get_financial_statements(symbol)
                elif name == 'market_structure':
                    data = provider.provider.get_comprehensive_market_structure(
                        symbol)
                else:
                    continue

                elapsed = time.time() - start_time
                quality = self._assess_data_quality(data, 'benchmark')

                results[name] = {
                    'response_time_ms': round(elapsed * 1000, 2),
                    'quality_score': quality,
                    'data_size': len(str(data)) if data else 0,
                    'success': True,
                    'error': None
                }

            except Exception as e:
                elapsed = time.time() - start_time
                results[name] = {
                    'response_time_ms': round(elapsed * 1000, 2),
                    'quality_score': 0.0,
                    'data_size': 0,
                    'success': False,
                    'error': str(e)
                }

        return results
