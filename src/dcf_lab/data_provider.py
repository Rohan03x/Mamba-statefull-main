"""Provider protocol and factory for swappable data backends.

This module defines a DataProvider Protocol and a factory `get_provider`
that instantiates the configured backend. Supported options include:

- ciq_api:     S&P Capital IQ API (primary; requires credentials)
- ciq_excel:   Excel plug-in export reader (fallback)
- ciq_snowflake: Snowflake/Xpressfeed adapter (scale)
- tiingo:      Tiingo professional financial data (recommended)
- yfinance:    Yahoo Finance via yfinance (deprecated - replaced with Tiingo)

Provider is selected via the `PROVIDER` environment variable or an
explicit argument to `get_provider`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Protocol, Tuple


class DataProvider(Protocol):
    def get_profile(self, ticker: str) -> Dict:
        ...

    def get_market(self, ticker: str) -> Dict:
        ...

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        """Return standardized financial statements for last `years` and LTM.

        Expected keys:
          - income: list[dict]
          - balance: list[dict]
          - cashflow: list[dict]
        Each item must include a `date` field (ISO date) and standardized
        columns per mapping defined in `mapping.py`.
        """
        ...

    def get_estimates(self, ticker: str) -> Dict:
        ...

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        ...


def _get_provider_mapping() -> Dict[str, Tuple[str, str]]:
    """Get mapping of provider names to (module_path, class_name)"""
    return {
        "ciq_api": (".providers.ciq_api", "CIQAPIProvider"),
        "enhanced_yfinance": (".providers.enhanced_yfinance", "EnhancedYFinanceProvider"),
        "yfinance": (".providers.dev_yf", "YFinanceProvider"),
        "tiingo": (".providers.tiingo_provider", "TiingoProvider"),
        "ciq_excel": (".providers.ciq_excel", "CIQExcelProvider"),
        "ciq_snowflake": (".providers.ciq_snowflake", "CIQSnowflakeProvider"),
        "auto": (".providers.auto", "AutoProvider"),
        "ciq_gds": (".providers.ciq_gds", "CIQGDSProvider"),
        "alpha": (".providers.dev_alpha", "AlphaProvider"),

        "fred": (".providers.fred_provider", "FREDProvider"),
        "sec_xbrl": (".providers.sec_xbrl_provider", "SECXBRLProvider"),
        "finnhub": (".providers.finnhub_api", "FinnhubProvider"),
        "ciq_capiq": (".providers.ciq_capiq", "CIQCapIQProvider"),
        "fmp": (".providers.fmp_api", "FMPProvider"),
        "market_structure": (".providers.market_structure_provider", "MarketStructureProvider"),
    }


def get_provider(name: str | None = None) -> DataProvider:
    provider_name = (
        name or os.environ.get(
            "PROVIDER",
            "yfinance")).strip().lower()
    provider_mapping = _get_provider_mapping()

    if provider_name not in provider_mapping:
        raise ValueError(f"Unknown provider: {provider_name}")

    module_path, class_name = provider_mapping[provider_name]

    # Dynamic import and instantiation
    try:
        # Try relative import first
        module = __import__(module_path, fromlist=[class_name], level=1)
    except (ImportError, KeyError):
        # Fall back to absolute import
        full_module_path = f"dcf_lab{module_path}" if module_path.startswith('.') else module_path
        module = __import__(full_module_path, fromlist=[class_name])
    provider_class = getattr(module, class_name)
    return provider_class()
