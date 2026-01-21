from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

# Example mapping snippet and common aliases
CIQ_TO_CANON = {
    "Revenue": "totalRevenue",
    "Total Revenue": "totalRevenue",
    "TotalRevenue": "totalRevenue",
    "EBITDA": "ebitda",
    "EBIT": "operatingIncome",
    "Operating Income": "operatingIncome",
    "Net Income": "netIncome",
    "Income Tax Expense": "incomeTaxExpense",
    "Interest Expense": "interestExpense",
    "Gross Profit": "grossProfit",
    "D&A": "depreciationAndAmortization",
    "Depreciation & Amortization": "depreciationAndAmortization",
    "Capital Expenditures": "capitalExpenditures",
    "Cash & Equivalents": "cashAndCashEquivalentsAtCarryingValue",
    "Short Term Debt": "shortTermDebt",
    "Long Term Debt": "longTermDebt",
    "Total Debt": "totalDebt",
    "Total Equity": "totalShareholderEquity",
    "Total Current Assets": "totalCurrentAssets",
    "Total Current Liabilities": "totalCurrentLiabilities",
    "PP&E": "propertyPlantEquipmentNet",
}


def _to_df(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if not df.empty:
        if "date" in df.columns:
            df.loc[:, "date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop duplicated columns keeping the last occurrence
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


def normalize_financials(fin: Dict[str,
                                   List[Dict]]) -> Tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         pd.DataFrame]:
    """Return income, balance, cashflow DataFrames with common aliases.

    The model code searches for specific column names, so we align likely
    synonyms to those names when present in provider payloads.
    """
    inc = _to_df(fin.get("income", []))
    bal = _to_df(fin.get("balance", []))
    cfs = _to_df(fin.get("cashflow", []))

    # Apply column mapping if provider used descriptive headers
    def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        alias = {
            # Common yfinance / FMP / CIQ variants
            "Total Revenue": "totalRevenue",
            "Revenue": "totalRevenue",
            "Operating Income": "operatingIncome",
            "Gross Profit": "grossProfit",
            "Depreciation And Amortization": "depreciationAndAmortization",
            "Depreciation Amortization Depletion": "depreciationAndAmortization",
            "Depreciation": "depreciationAndAmortization",
            "Amortization": "depreciationAndAmortization",
            "Tax Provision": "incomeTaxExpense",
            "Income Tax Expense": "incomeTaxExpense",
            "Interest Expense": "interestExpense",
            "Interest Expense Non Operating": "interestExpense",
            "Net Income": "netIncome",
            "EBITDA": "ebitda",
            "EBIT": "operatingIncome",
            # Cash flow
            "Capital Expenditure": "capitalExpenditures",
            "Capital Expenditures": "capitalExpenditures",
        }
        new_cols = {}
        for c in df.columns:
            if c in ("date", "symbol", "reportedCurrency"):
                continue
            mapped = CIQ_TO_CANON.get(c) or alias.get(c)
            if mapped:
                new_cols[c] = mapped
        if new_cols:
            df = df.rename(columns=new_cols)
        return df

    inc = rename_cols(inc)
    bal = rename_cols(bal)
    cfs = rename_cols(cfs)

    # Deduplicate again after renaming and coerce numerics
    def _dedupe_and_numeric(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        for c in df.columns:
            if c != "date":
                df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")
        return df

    inc = _dedupe_and_numeric(inc)
    bal = _dedupe_and_numeric(bal)
    cfs = _dedupe_and_numeric(cfs)

    # Derived fields where possible
    if not inc.empty:
        if "operatingIncome" not in inc.columns and set(
                ["ebit", "EBIT"]).intersection(inc.columns):
            src = "ebit" if "ebit" in inc.columns else "EBIT"
            inc["operatingIncome"] = inc[src]
        if "grossProfit" not in inc.columns and "gross_profit" in inc.columns:
            inc["grossProfit"] = inc["gross_profit"]

    if not bal.empty and "totalDebt" not in bal.columns:
        # Construct total debt from short + long when present
        if set(["shortTermDebt", "longTermDebt"]).issubset(bal.columns):
            bal["totalDebt"] = bal["shortTermDebt"].fillna(
                0) + bal["longTermDebt"].fillna(0)

    return inc, bal, cfs
