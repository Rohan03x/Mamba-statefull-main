from __future__ import annotations

from typing import Dict, List


class AutoProvider:
    """Fallback provider that tries multiple backends in order.

    Order is controlled by the `AUTO_ORDER` environment variable (comma-separated)
    or defaults to: eodhd, tiingo, finnhub, yfinance, alpha.
    """

    def __init__(self) -> None:
        import os

        # Enhanced: EODHD as primary, TiingoProvider as secondary
        order = (os.environ.get("AUTO_ORDER")
                 or "eodhd,tiingo,finnhub,yfinance,alpha").split(",")
        order = [o.strip() for o in order if o.strip()]
        self.providers = []
        for name in order:
            name = name.lower()
            try:
                self.providers.append(self._make(name))
            except Exception:
                # Skip unknown/misconfigured providers
                continue

    def _make(self, name: str):
        if name == "eodhd":
            from src.data_sources.eodhd_provider import EODHDProvider
            return EODHDProvider()
        if name == "tiingo":
            from .tiingo_provider import TiingoProvider

            return TiingoProvider()
        if name == "finnhub":
            from .finnhub_api import FinnhubProvider

            return FinnhubProvider()
        if name == "yfinance":
            from .dev_yf import YFinanceProvider

            return YFinanceProvider()
        if name == "alpha":
            from .dev_alpha import AlphaProvider

            return AlphaProvider()
        if name == "ciq_gds":
            from .ciq_gds import CIQGDSProvider

            return CIQGDSProvider()
        if name == "ciq_api":
            from .ciq_api import CIQAPIProvider

            return CIQAPIProvider()
        if name == "ciq_excel":
            from .ciq_excel import CIQExcelProvider

            return CIQExcelProvider()
        raise ValueError(name)

    # ----------------------------- helpers -----------------------------
    @staticmethod
    def _fin_ok(fin: Dict) -> bool:
        try:
            return bool(
                fin and len(
                    fin.get(
                        "income",
                        [])) and len(
                    fin.get(
                        "balance",
                        [])) and len(
                    fin.get(
                        "cashflow",
                        [])))
        except Exception:
            return False

    @staticmethod
    def _tiingo_fin_ok(fin: Dict) -> bool:
        """Enhanced: Validate TiingoProvider financial data structure"""
        try:
            # TiingoProvider uses these actual keys: income_statements, balance_sheets, cash_flows
            return bool(
                fin and (
                    (fin.get("income_statements") and len(fin["income_statements"])) or
                    (fin.get("balance_sheets") and len(fin["balance_sheets"])) or
                    (fin.get("cash_flows") and len(fin["cash_flows"])) or
                    # Also support alternative naming formats
                    (fin.get("incomeStatement") and len(fin["incomeStatement"])) or
                    (fin.get("balanceSheet") and len(fin["balanceSheet"])) or
                    (fin.get("cashFlowStatement") and len(fin["cashFlowStatement"]))
                ))
        except Exception:
            return False

    # ------------------------------ API --------------------------------
    def get_profile(self, ticker: str) -> Dict:
        for p in self.providers:
            try:
                prof = p.get_profile(ticker) or {}
                if prof:
                    return prof
            except Exception:
                continue
        return {"ticker": ticker}

    def get_market(self, ticker: str) -> Dict:
        for p in self.providers:
            try:
                m = p.get_market(ticker) or {}
                # Enhanced: Accept TiingoProvider's rich fundamental data
                enhanced_fields = ["price", "marketcap", "shares", "beta", 
                                 "market_cap", "pe_ratio", "pb_ratio", "enterprise_value"]
                if any(m.get(k) for k in enhanced_fields):
                    return m
            except Exception:
                continue
        return {"ticker": ticker}

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        for p in self.providers:
            try:
                fin = p.get_financials(ticker, years) or {}
                # Enhanced: Accept TiingoProvider's comprehensive financial data
                if self._fin_ok(fin) or self._tiingo_fin_ok(fin):
                    return fin
            except Exception:
                continue
        return {"income": [], "balance": [], "cashflow": []}

    def get_estimates(self, ticker: str) -> Dict:
        for p in self.providers:
            try:
                est = p.get_estimates(ticker) or {}
                if est:
                    return est
            except Exception:
                continue
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """Enhanced: Aggregate news from multiple providers, prioritizing quality sources"""
        all_news = []
        seen_urls = set()
        
        for p in self.providers:
            try:
                n = p.get_news(ticker, days) or []
                if n:
                    # Deduplicate by URL and add provider metadata
                    for article in n:
                        url = article.get('url', '')
                        if url and url not in seen_urls:
                            article['provider_source'] = getattr(p, '__class__', type(p)).__name__
                            all_news.append(article)
                            seen_urls.add(url)
                        elif not url:  # Articles without URLs (still valuable)
                            article['provider_source'] = getattr(p, '__class__', type(p)).__name__
                            all_news.append(article)
            except Exception:
                continue
        
        # Sort by publication date (newest first) if available
        try:
            from datetime import datetime
            def parse_date(article):
                pub_date = article.get('published', '')
                if pub_date:
                    try:
                        if 'T' in pub_date:
                            return datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        else:
                            return datetime.strptime(pub_date[:10], '%Y-%m-%d')
                    except:
                        pass
                return datetime.min
            
            all_news.sort(key=parse_date, reverse=True)
        except:
            pass
        
        return all_news
