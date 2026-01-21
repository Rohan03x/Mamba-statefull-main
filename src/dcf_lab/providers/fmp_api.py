from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from ..data_sources import FMP_BASE, _cached_get, fmp_balance, fmp_cashflow, fmp_income
from ..settings import CACHE_TTL_FINANCIALS_SEC, CACHE_TTL_MARKET_SEC


class FMPProvider:
    """Financial Modeling Prep provider (REST).

    Env: FMP_KEY
    """

    def __init__(self) -> None:
        self.key = os.environ.get("FMP_KEY", "")

    # ---------------------------- helpers ----------------------------
    @staticmethod
    def _to_df(rows: List[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows or [])
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        return df

    @staticmethod
    def _rename_income(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mapping = {
            "revenue": "totalRevenue",
            "totalRevenue": "totalRevenue",
            "grossProfit": "grossProfit",
            "operatingIncome": "operatingIncome",
            "ebitda": "ebitda",
            "ebit": "operatingIncome",
            "netIncome": "netIncome",
            "incomeTaxExpense": "incomeTaxExpense",
            "interestExpense": "interestExpense",
        }
        df = df.rename(
            columns={
                k: v for k,
                v in mapping.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _rename_balance(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mapping = {
            "cashAndCashEquivalents": "cashAndCashEquivalentsAtCarryingValue",
            "cashAndShortTermInvestments": "cashAndCashEquivalentsAtCarryingValue",
            "shortTermDebt": "shortTermDebt",
            "longTermDebt": "longTermDebt",
            "totalStockholdersEquity": "totalShareholderEquity",
            "totalShareholdersEquity": "totalShareholderEquity",
            "totalShareholderEquity": "totalShareholderEquity",
            "totalCurrentAssets": "totalCurrentAssets",
            "totalCurrentLiabilities": "totalCurrentLiabilities",
            "propertyPlantEquipmentNet": "propertyPlantEquipmentNet",
            "totalAssets": "totalAssets",
            "totalLiabilities": "totalLiabilities",
        }
        df = df.rename(
            columns={
                k: v for k,
                v in mapping.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _rename_cashflow(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mapping = {
            "capitalExpenditure": "capitalExpenditures",
            "capitalExpenditures": "capitalExpenditures",
            "depreciationAndAmortization": "depreciationAndAmortization",
        }
        df = df.rename(
            columns={
                k: v for k,
                v in mapping.items() if k in df.columns})
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ------------------------------ API ------------------------------
    def get_profile(self, ticker: str) -> Dict:
        data = _cached_get(f"{FMP_BASE}/profile/{ticker}",
                           {"apikey": self.key}, CACHE_TTL_FINANCIALS_SEC)
        row = (data or [{}])[0] if isinstance(data, list) else {}
        return {
            "ticker": ticker,
            "name": row.get("companyName") or "",
            "country": row.get("country"),
            "sector": row.get("sector"),
            "currency": row.get("currency") or "USD",
        }

    def get_market(self, ticker: str) -> Dict:
        q = _cached_get(f"{FMP_BASE}/quote/{ticker}",
                        {"apikey": self.key}, CACHE_TTL_MARKET_SEC)
        qrow = (q or [{}])[0] if isinstance(q, list) else {}
        prof = _cached_get(f"{FMP_BASE}/profile/{ticker}",
                           {"apikey": self.key}, CACHE_TTL_FINANCIALS_SEC)
        prow = (prof or [{}])[0] if isinstance(prof, list) else {}
        return {
            "ticker": ticker,
            "price": qrow.get("price"),
            "marketcap": qrow.get("marketCap") or prow.get("mktCap"),
            "shares": prow.get("sharesOutstanding"),
            "beta": prow.get("beta"),
        }

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        inc_df = self._rename_income(fmp_income(ticker, key=self.key))
        bal_df = self._rename_balance(fmp_balance(ticker, key=self.key))
        cf_df = self._rename_cashflow(fmp_cashflow(ticker, key=self.key))
        # Keep last N years
        for df in (inc_df, bal_df, cf_df):
            if not df.empty and "date" in df.columns:
                df.sort_values("date", inplace=True)
        inc_df = inc_df.tail(years)
        bal_df = bal_df.tail(years)
        cf_df = cf_df.tail(years)
        return {
            "income": inc_df.to_dict(orient="records"),
            "balance": bal_df.to_dict(orient="records"),
            "cashflow": cf_df.to_dict(orient="records"),
        }

    def get_estimates(self, ticker: str) -> Dict:
        # FMP has estimates endpoints; defer for now
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        # Use stock_news with a reasonable limit
        data = _cached_get(f"{FMP_BASE}/stock_news",
                           {"tickers": ticker,
                            "limit": 50,
                            "apikey": self.key},
                           CACHE_TTL_MARKET_SEC)
        out = []
        for n in data or []:
            out.append(
                {
                    "date": n.get("publishedDate"),
                    "headline": n.get("title"),
                    "source": n.get("site"),
                    "url": n.get("url"),
                    "tickers": [ticker],
                }
            )
        return out
