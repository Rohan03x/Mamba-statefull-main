from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..data_sources import (
    alpha_balance,
    alpha_cashflow,
    alpha_income,
    alpha_overview,
    overview_fields,
    parse_financials,
)


class AlphaProvider:
    """Dev provider using Alpha Vantage helpers (requires API key)."""

    def get_profile(self, ticker: str) -> Dict:
        ov = alpha_overview(ticker) or {}
        f = overview_fields(ov)
        return {
            "ticker": ticker,
            "name": ov.get("Name") or "",
            "country": ov.get("Country"),
            "sector": ov.get("Sector"),
            "currency": f.get("currency") or "USD",
        }

    def get_market(self, ticker: str) -> Dict:
        ov = alpha_overview(ticker) or {}
        f = overview_fields(ov)
        return {
            "ticker": ticker,
            "price": None,  # Alpha overview does not provide RT price here
            "shares": f.get("shares"),
            "marketcap": f.get("marketcap"),
            "beta": f.get("beta"),
        }

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        inc = alpha_income(ticker) or {}
        bal = alpha_balance(ticker) or {}
        cfs = alpha_cashflow(ticker) or {}
        inc_a, bal_a, cfs_a = parse_financials(inc, bal, cfs)
        # Limit to `years`

        def last_n(df: pd.DataFrame) -> List[Dict]:
            if "date" in df.columns:
                df = df.sort_values("date").tail(years)
            return df.to_dict(orient="records")

        return {
            "income": last_n(inc_a),
            "balance": last_n(bal_a),
            "cashflow": last_n(cfs_a),
        }

    def get_estimates(self, ticker: str) -> Dict:
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        return []
