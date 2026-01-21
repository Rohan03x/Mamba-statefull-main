from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd


def _best_match(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    if df.empty or "Mnemonic" not in df.columns:
        return None
    name_col = None
    for c in ["Mnemonic Name", "MnemonicName", "Name", "Description"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return None
    s = df[[name_col, "Mnemonic"]].dropna().copy()
    s[name_col] = s[name_col].astype(str).str.lower()
    for p in patterns:
        m = s[s[name_col].str.contains(p.lower())]
        if not m.empty:
            return str(m.iloc[0]["Mnemonic"]).strip()
    return None


def load_mnemonics_from_excel(path: str) -> Dict[str, Dict[str, str]]:
    """Read an Excel of data items and heuristically map canonical fields to mnemonics.

    Returns a dict with keys: inc, bs, cf, market mapping canonical_field -> mnemonic.
    Unknown fields are omitted.
    """
    if not path or not os.path.exists(path):
        return {}
    xl = pd.ExcelFile(path)
    # Merge all sheets vertically to search across
    frames = []
    for s in xl.sheet_names:
        try:
            frames.append(pd.read_excel(xl, s))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return {}
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    # Income statement mnemonics
    inc_map = {
        "totalRevenue": _best_match(df, ["total revenue", "revenue"]),
        "ebitda": _best_match(df, ["ebitda", "ebita"]),
        "operatingIncome": _best_match(df, ["operating income", "ebit"]),
        "netIncome": _best_match(df, ["net income"]),
        "grossProfit": _best_match(df, ["gross profit"]),
        "interestExpense": _best_match(df, ["interest expense"]),
        "incomeTaxExpense": _best_match(df, ["income tax expense", "tax provision"]),
    }
    # Balance sheet
    bs_map = {
        "cashAndCashEquivalentsAtCarryingValue": _best_match(df, ["cash and cash equivalents", "cash & equivalents"]),
        "shortTermDebt": _best_match(df, ["short term debt", "st debt"]),
        "longTermDebt": _best_match(df, ["long term debt", "lt debt"]),
        "totalDebt": _best_match(df, ["total debt"]),
        "totalAssets": _best_match(df, ["total assets"]),
        "totalLiabilities": _best_match(df, ["total liabilities"]),
        "totalShareholderEquity": _best_match(df, ["total shareholders", "total equity"]),
        "totalCurrentAssets": _best_match(df, ["total current assets"]),
        "totalCurrentLiabilities": _best_match(df, ["total current liabilities"]),
        "propertyPlantEquipmentNet": _best_match(df, ["property, plant", "ppe", "plant and equipment net"]),
    }
    # Cash flow
    cf_map = {
        "capitalExpenditures": _best_match(df, ["capital expenditures", "capex"]),
        "depreciationAndAmortization": _best_match(df, ["depreciation", "amortization"]),
    }
    # Market
    market_map = {
        "price": _best_match(df, ["close price", "last price"]),
        "shares": _best_match(df, ["shares diluted", "diluted shares"]),
        "marketcap": _best_match(df, ["market cap"]),
        "beta": _best_match(df, ["beta"]),
    }
    # Strip Nones
    inc_map = {k: v for k, v in inc_map.items() if v}
    bs_map = {k: v for k, v in bs_map.items() if v}
    cf_map = {k: v for k, v in cf_map.items() if v}
    market_map = {k: v for k, v in market_map.items() if v}
    return {"inc": inc_map, "bs": bs_map, "c": cf_map, "market": market_map}
