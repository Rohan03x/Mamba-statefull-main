from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..settings import (
    CACHE_DIR,
    CACHE_TTL_FINANCIALS_SEC,
    CACHE_TTL_MARKET_SEC,
    CACHE_TTL_NEWS_SEC,
)


def _cache_path(prefix: str, key: str) -> str:
    safe = prefix.replace("/", "_") + "_" + key
    return os.path.join(CACHE_DIR, safe + ".json")


def _load_cache(path: str, ttl: int) -> Optional[dict]:
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


class FinnhubProvider:
    """Provider using Finnhub REST API.

    Requires FINNHUB_TOKEN in environment.
    Endpoints used:
      - /stock/profile2
      - /quote
      - /stock/financials (income_statement, balance_sheet, cash_flow)
      - /company-news
      - /stock/metric (beta, etc.)
    """

    def __init__(self) -> None:
        self.base = os.environ.get(
            "FINNHUB_BASE_URL",
            "https://finnhub.io/api/v1").rstrip("/")
        self.token = os.environ.get("FINNHUB_TOKEN")
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get(self, path: str, params: Dict, ttl: int, cache_key: str) -> Dict:
        params = dict(params or {})
        if self.token:
            params.setdefault("token", self.token)
        cache_fp = _cache_path(path.strip("/"), cache_key)
        cached = _load_cache(cache_fp, ttl)
        if cached is not None:
            return cached
        url = f"{self.base}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=40)
        try:
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            data = {
                "_error": str(e),
                "_status": getattr(
                    r,
                    "status_code",
                    None)}
        _save_cache(cache_fp, data)
        return data

    # ---------------------------- API methods ---------------------------
    def get_profile(self, ticker: str) -> Dict:
        d = self._get("stock/profile2",
                      {"symbol": ticker},
                      CACHE_TTL_FINANCIALS_SEC,
                      f"profile_{ticker}")
        metrics = self._get("stock/metric",
                            {"symbol": ticker,
                             "metric": "all"},
                            CACHE_TTL_FINANCIALS_SEC,
                            f"metric_{ticker}")
        beta = None
        try:
            beta = metrics.get("metric", {}).get("beta")
        except Exception:
            pass
        return {
            "ticker": ticker,
            "name": d.get("name") or "",
            "country": d.get("country"),
            "sector": d.get("finnhubIndustry"),
            "currency": d.get("currency") or "USD",
            "beta": beta,
        }

    def get_market(self, ticker: str) -> Dict:
        prof = self._get("stock/profile2",
                         {"symbol": ticker},
                         CACHE_TTL_FINANCIALS_SEC,
                         f"profile_{ticker}")
        q = self._get("quote",
                      {"symbol": ticker},
                      CACHE_TTL_MARKET_SEC,
                      f"quote_{ticker}")
        price = q.get("c") or q.get("pc")
        shares = prof.get("shareOutstanding")
        mc = prof.get("marketCapitalization")
        if not mc and price and shares:
            try:
                mc = float(price) * float(shares)
            except Exception:
                mc = None
        beta = self._get("stock/metric",
                         {"symbol": ticker,
                          "metric": "all"},
                         CACHE_TTL_FINANCIALS_SEC,
                         f"metric_{ticker}").get("metric",
                                                 {}).get("beta")
        return {
            "ticker": ticker,
            "price": price,
            "shares": shares,
            "marketcap": mc,
            "beta": beta}

    def _fin_table(
            self,
            ticker: str,
            statement: str,
            years: int) -> pd.DataFrame:
        # statement: income_statement | balance_sheet | cash_flow
        data = self._get(
            "stock/financials",
            {"symbol": ticker, "statement": statement, "freq": "annual"},
            CACHE_TTL_FINANCIALS_SEC,
            f"fin_{statement}_{ticker}",
        )
        rows = data.get("data") or []
        # If endpoint not permitted on your plan or empty, fallback to
        # financials-reported
        if (isinstance(data, dict) and data.get("_status") == 403) or not rows:
            rep = self._get(
                "stock/financials-reported",
                {"symbol": ticker, "freq": "annual"},
                CACHE_TTL_FINANCIALS_SEC,
                f"fin_reported_{ticker}",
            )
            rdata = rep.get("data") or []
            if rdata:
                def pick_report(report: dict, section: str) -> dict:
                    items = (report or {}).get(section) or []
                    out = {}
                    for it in items:
                        lbl = str(it.get("label") or "").strip().lower()
                        val = it.get("value")
                        out[lbl] = val
                    return out
                mapped = []
                for r in rdata:
                    end = r.get("endDate") or r.get("period") or r.get("year")
                    if isinstance(end, int):
                        end = f"{end}-12-31"
                    sec_key = {
                        "income_statement": "ic",
                        "balance_sheet": "bs",
                        "cash_flow": "c"}[statement]
                    sec = pick_report(r.get("report", {}), sec_key)
                    if not sec:
                        continue
                    row = {"date": end}
                    # Map labels to canonical columns (best-effort)

                    def set_if(keys: list[str], out_key: str):
                        for k in keys:
                            if k in sec and sec[k] is not None:
                                row[out_key] = sec[k]
                                return
                    if statement == "income_statement":
                        set_if(["revenue", "total revenue"], "totalRevenue")
                        set_if(["gross profit"], "grossProfit")
                        set_if(["operating income", "ebit"], "operatingIncome")
                        set_if(["ebitda"], "ebitda")
                        set_if(["net income"], "netIncome")
                        set_if(["income tax expense"], "incomeTaxExpense")
                        set_if(["interest expense"], "interestExpense")
                    elif statement == "balance_sheet":
                        set_if(
                            ["cash and cash equivalents"],
                            "cashAndCashEquivalentsAtCarryingValue")
                        set_if(["short term debt"], "shortTermDebt")
                        set_if(["long term debt"], "longTermDebt")
                        set_if(["total shareholders' equity",
                                "total shareholders’ equity",
                                "total shareholders equity"],
                               "totalShareholderEquity")
                        set_if(["total current assets"], "totalCurrentAssets")
                        set_if(
                            ["total current liabilities"],
                            "totalCurrentLiabilities")
                        set_if(["property, plant & equipment net",
                                "property, plant and equipment net"],
                               "propertyPlantEquipmentNet")
                    else:  # cash_flow
                        set_if(["capital expenditures"], "capitalExpenditures")
                        set_if(["depreciation & amortization",
                                "depreciation and amortization"],
                               "depreciationAndAmortization")
                    mapped.append(row)
                df = pd.DataFrame(mapped)
                if not df.empty and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(
                        subset=["date"]).sort_values("date").tail(years)
                return df
        # Normalize keys and construct date
        out = []
        for r in rows:
            d = {}
            # prefer endDate; else calendarYear
            end = r.get("endDate") or r.get("period") or r.get(
                "year") or r.get("calendarYear")
            if isinstance(end, int):
                end = f"{end}-12-31"
            d["date"] = end
            # Merge all numeric fields as-is
            for k, v in r.items():
                if k in (
                    "symbol",
                    "reportType",
                    "period",
                    "year",
                        "calendarYear"):
                    continue
                d[k] = v
            out.append(d)
        df = pd.DataFrame(out)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").tail(years)
        return df

    def get_financials(self, ticker: str,
                       years: int = 5) -> Dict[str, List[Dict]]:
        inc = self._fin_table(ticker, "income_statement", years)
        bal = self._fin_table(ticker, "balance_sheet", years)
        cfs = self._fin_table(ticker, "cash_flow", years)
        # Map common keys to our canonical names where needed

        def rename(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
            cols = {k: v for k, v in mapping.items() if k in df.columns}
            df = df.rename(columns=cols)
            # Drop duplicated columns after rename (e.g., revenue and
            # totalRevenue -> totalRevenue)
            if not df.empty:
                df = df.loc[:, ~df.columns.duplicated(keep="last")]
                # Convert numerics
                for c in df.columns:
                    if c != "date":
                        df[c] = pd.to_numeric(df[c], errors="coerce")
            return df

        inc = rename(
            inc,
            {
                "revenue": "totalRevenue",
                "totalRevenue": "totalRevenue",
                "grossProfit": "grossProfit",
                "operatingIncome": "operatingIncome",
                "ebitda": "ebitda",
                "netIncome": "netIncome",
                "incomeTaxExpense": "incomeTaxExpense",
                "interestExpense": "interestExpense",
            },
        )
        bal = rename(
            bal,
            {"cashAndCashEquivalents": "cashAndCashEquivalentsAtCarryingValue",
             "cashAndCashEquivalentsAtCarryingValue":
             "cashAndCashEquivalentsAtCarryingValue",
             "shortTermDebt": "shortTermDebt", "longTermDebt": "longTermDebt",
             "totalShareholdersEquity": "totalShareholderEquity",
             "totalShareholderEquity": "totalShareholderEquity",
             "totalCurrentAssets": "totalCurrentAssets",
             "totalCurrentLiabilities": "totalCurrentLiabilities",
             "propertyPlantEquipmentNet": "propertyPlantEquipmentNet",
             "netPPE": "propertyPlantEquipmentNet", },)
        cfs = rename(
            cfs,
            {
                "capitalExpenditure": "capitalExpenditures",
                "capitalExpenditures": "capitalExpenditures",
                "depreciationAndAmortization": "depreciationAndAmortization",
            },
        )
        return {
            "income": inc.to_dict(orient="records"),
            "balance": bal.to_dict(orient="records"),
            "cashflow": cfs.to_dict(orient="records"),
        }

    def get_estimates(self, ticker: str) -> Dict:
        # Finnhub provides various estimate endpoints; keep minimal for now
        return {}

    def get_news(self, ticker: str, days: int = 7) -> List[Dict]:
        to_dt = datetime.now(timezone.utc).date()
        from_dt = to_dt - timedelta(days=days)
        data = self._get(
            "company-news",
            {"symbol": ticker,
    "from": from_dt.isoformat(),
     "to": to_dt.isoformat()},
            CACHE_TTL_NEWS_SEC,
            f"news_{ticker}_{days}",
        )
        items = []
        for n in data or []:
            items.append(
                {
                    "date": n.get("datetime"),
                    "headline": n.get("headline"),
                    "source": n.get("source"),
                    "url": n.get("url"),
                    "tickers": [ticker],
                }
            )
        return items
