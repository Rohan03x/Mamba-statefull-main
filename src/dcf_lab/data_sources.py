import hashlib
import json
import os
import pickle
import time

import pandas as pd
import requests
import yfinance as yf

from .settings import CACHE_TTL_SECONDS
from .utils import try_float, sanitize_dt_index

CACHE_DIR = os.environ.get("DCF_LAB_CACHE_DIR", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ALPHA_URL = "https://www.alphavantage.co/query"


def _cache_key(url, params): return hashlib.md5(
    (url + json.dumps(params, sort_keys=True)).encode()).hexdigest()


def _cached_get(url, params, ttl=6*3600):
    fp = os.path.join(CACHE_DIR, _cache_key(url, params)+".json")
    if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < ttl:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    r = requests.get(url, params=params, timeout=40)
    r.raise_for_status()
    data = r.json()
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _cached_pickle(fp: str, fetcher, ttl: int = CACHE_TTL_SECONDS):
    if os.path.exists(fp) and time.time() - os.path.getmtime(fp) < ttl:
        with open(fp, "rb") as f:
            return pickle.load(f)
    data = fetcher()
    with open(fp, "wb") as f:
        pickle.dump(data, f)
    return data


def _alpha(function, symbol, key=None):
    key = key or os.environ.get("ALPHA_VANTAGE_KEY", "")
    if not key:
        raise RuntimeError("Missing Alpha Vantage key")
    data = _cached_get(
        ALPHA_URL, {
            "function": function, "symbol": symbol, "apikey": key})
    if isinstance(data, dict) and "Note" in data:
        raise RuntimeError("Alpha Vantage rate limit: "+data.get("Note", ""))
    if isinstance(data, dict) and "Error Message" in data:
        raise RuntimeError("Alpha Vantage error: " +
                           data.get("Error Message", ""))
    return data


def alpha_overview(
    symbol, key=None): return _cached_get(
        ALPHA_URL, {
            "function": "OVERVIEW", "symbol": symbol, "apikey": key or os.environ.get(
                "ALPHA_VANTAGE_KEY", "")})


def alpha_income(
    symbol,
    key=None): return _alpha(
        "INCOME_STATEMENT",
        symbol,
    key)


def alpha_balance(
    symbol,
    key=None): return _alpha(
        "BALANCE_SHEET",
        symbol,
    key)


def alpha_cashflow(symbol, key=None): return _alpha("CASH_FLOW", symbol, key)


def parse_financials(income_obj, balance_obj, cash_obj):
    def _to_df(obj, key):
        arr = obj.get(key, [])
        if not isinstance(arr, list) or not arr:
            return pd.DataFrame()
        df = pd.DataFrame(arr)
        if "date" not in df.columns and "fiscalDateEnding" in df.columns:
            df["date"] = pd.to_datetime(
                df["fiscalDateEnding"], errors="coerce")
        for c in df.columns:
            if c not in (
                "date",
                "fiscalDateEnding",
                "reportedCurrency",
                    "symbol"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "date" in df.columns:
            df = df.sort_values("date")
        return df
    return _to_df(
        income_obj, "annualReports"), _to_df(
        balance_obj, "annualReports"), _to_df(
            cash_obj, "annualReports")


def overview_fields(overview):
    return {
        "marketcap": try_float(overview.get("MarketCapitalization")),
        "beta": try_float(overview.get("Beta")),
        "shares": try_float(overview.get("SharesOutstanding")),
        "currency": overview.get("Currency") or "USD",
        "name": overview.get("Name") or ""
    }


def yf_statements(symbol: str):
    """Retrieve financial statements and price history via yfinance."""
    fp = os.path.join(CACHE_DIR, f"yf_{symbol}_statements.pkl")

    def _fetch():
        t = yf.Ticker(symbol)
        income = t.financials.T.reset_index().rename(columns={"index": "date"})
        balance = t.balance_sheet.T.reset_index().rename(
            columns={"index": "date"})
        cash = t.cashflow.T.reset_index().rename(columns={"index": "date"})
        price = t.history(period="max")
        return income, balance, cash, price

    income, balance, cash, price = _cached_pickle(fp, _fetch)
    for df in (income, balance, cash):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
    
    # Global datetime sanitation: make all timestamps tz-naive UTC
    price = sanitize_dt_index(price)
    for df in (income, balance, cash):
        if "date" in df.columns:
            df = sanitize_dt_index(df, col="date")
    
    return income, balance, cash, price


def yf_overview(symbol: str):
    """Return basic company overview fields from yfinance."""
    t = yf.Ticker(symbol)
    info = getattr(t, "info", {}) or {}
    return {
        "marketcap": try_float(info.get("marketCap")),
        "beta": try_float(info.get("beta")),
        "shares": try_float(info.get("sharesOutstanding")),
        "currency": info.get("currency") or "USD",
        "name": info.get("shortName") or "",
    }


FMP_BASE = "https://financialmodelingprep.com/api/v3"


def fmp_income(symbol: str, key: str | None = None) -> pd.DataFrame:
    key = key or os.environ.get("FMP_KEY", "")
    data = _cached_get(f"{FMP_BASE}/income-statement/{symbol}",
                       {"apikey": key, "limit": 120})
    return pd.DataFrame(data)


def fmp_balance(symbol: str, key: str | None = None) -> pd.DataFrame:
    key = key or os.environ.get("FMP_KEY", "")
    data = _cached_get(
        f"{FMP_BASE} /balance-sheet-statement/{symbol} ",
        {"apikey": key, "limit": 120})
    return pd.DataFrame(data)


def fmp_cashflow(symbol: str, key: str | None = None) -> pd.DataFrame:
    key = key or os.environ.get("FMP_KEY", "")
    data = _cached_get(
        f"{FMP_BASE} /cash-flow-statement/{symbol} ",
        {"apikey": key, "limit": 120})
    return pd.DataFrame(data)


def fmp_enterprise_value(symbol: str, key: str | None = None) -> pd.DataFrame:
    key = key or os.environ.get("FMP_KEY", "")
    data = _cached_get(
        f"{FMP_BASE}/enterprise-values/{symbol}", {"apikey": key, "limit": 40})
    return pd.DataFrame(data)
