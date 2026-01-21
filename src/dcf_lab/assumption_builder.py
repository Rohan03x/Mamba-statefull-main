"""Derive forecast assumptions from historical financial statements."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import clamp


def _hist_n(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the last ``n`` non-null rows of ``df`` sorted by date.

    More robust: if a ``date`` column is missing, try to construct it from index or
    known alternatives; never raise on missing ``date``.
    """
    if df is None or df.empty:
        return df
    data = df.copy()
    if "date" not in data.columns:
        # Try common alternatives
        if "fiscalDateEnding" in data.columns:
            data["date"] = pd.to_datetime(
                data["fiscalDateEnding"], errors="coerce")
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(
                columns={data.index.name or "index": "date"})
        else:
            # Fallback: coerce index to datetime
            try:
                data["date"] = pd.to_datetime(data.index, errors="coerce")
            except Exception:
                data["date"] = pd.NaT
    else:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]) if "date" in data.columns else data
    if "date" in data.columns:
        data = data.sort_values("date").tail(n)
    else:
        data = data.tail(n)
    return data.copy()


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        # Sum across duplicate columns if any
        return x.sum(axis=1, numeric_only=True)
    if isinstance(x, pd.Series):
        return x
    try:
        return pd.to_numeric(pd.Series(x), errors="coerce")
    except Exception:
        return pd.Series([], dtype=float)


def _avg_ratio(numer, denom, default: float) -> float:
    """Return the ratio of two numeric sequences, guarding against division by zero."""
    try:
        nser = _to_series(numer)
        dser = _to_series(denom)
        numerator_sum = float(nser.sum(skipna=True)) if len(nser) else 0.0
        denominator_sum = float(dser.sum(skipna=True)) if len(dser) else 0.0
        return default if denominator_sum == 0 else numerator_sum / denominator_sum
    except Exception:
        return default


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column found in ``df`` from ``candidates``."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _calculate_revenue_cagr(inc5: pd.DataFrame, rev_col: str) -> float:
    """Calculate revenue compound annual growth rate."""
    rev_vals = inc5.get(rev_col)
    if isinstance(rev_vals, pd.DataFrame):
        rev_vals = rev_vals.iloc[:, 0]
    rev_series = pd.to_numeric(rev_vals, errors="coerce")
    rev = pd.DataFrame({"date": inc5.get("date"), "rev": rev_series}).dropna(
        subset=["date", "rev"]).set_index("date").sort_index()

    if len(rev) >= 2:
        years = (rev.index[-1].year - rev.index[0].year) or 1
        first = float(rev["rev"].iloc[0]) or 1e-6
        last = float(rev["rev"].iloc[-1])
        cagr = (last / max(1e-6, first)) ** (1 / years) - 1
        return clamp(cagr, -0.5, 0.5)
    else:
        return 0.05


def _calculate_margins(inc5: pd.DataFrame, rev_col: str) -> tuple:
    """Calculate gross margin, EBIT margin, and OPEX margin."""
    gp_col = _find_col(inc5, ["grossProfit", "GrossProfit"])
    op_col = _find_col(inc5, ["operatingIncome", "OperatingIncome", "ebit"])

    gross_margin = _avg_ratio(inc5.get(gp_col, 0), inc5.get(rev_col, 0), 0.40)
    ebit_margin = _avg_ratio(inc5.get(op_col, 0), inc5.get(rev_col, 0), 0.15)
    opex_margin = max(0.0, gross_margin - ebit_margin)

    return gross_margin, ebit_margin, opex_margin, op_col


def _calculate_da_ratio(
        inc5: pd.DataFrame,
        cfs5: pd.DataFrame,
        rev_col: str) -> float:
    """Calculate depreciation and amortization to revenue ratio."""
    da_col = _find_col(
        inc5, [
            "depreciationAndAmortization", "depreciation", "amortization"])
    da_series = inc5.get(da_col, pd.Series(
        [np.nan])) if da_col else pd.Series([np.nan])

    if isinstance(da_series, pd.Series) and da_series.isna().all():
        da_series = cfs5.get("depreciationAndAmortization", 0)

    return _avg_ratio(da_series, inc5.get(rev_col, 0), 0.04)


def _calculate_nwc_ratio(
        bal5: pd.DataFrame,
        inc5: pd.DataFrame,
        rev_col: str) -> float:
    """Calculate net working capital to revenue ratio."""
    if {"totalCurrentAssets", "totalCurrentLiabilities"}.issubset(
            bal5.columns):
        nwc = bal5["totalCurrentAssets"].astype(
            float) - bal5["totalCurrentLiabilities"].astype(float)
        return _avg_ratio(nwc, inc5.get(rev_col, 0), 0.03)
    else:
        return 0.03


def _calculate_tax_rate(
        inc5: pd.DataFrame,
        cfs5: pd.DataFrame,
        op_col: str) -> float:
    """Calculate effective tax rate."""
    def _as_series(x):
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, 0]
        return pd.Series(x)

    tax_series = inc5.get("incomeTaxExpense", pd.Series([np.nan]))
    if isinstance(tax_series, pd.Series) and tax_series.isna().all():
        tax_series = cfs5.get("incomeTaxExpense", 0)

    tax_exp = _as_series(tax_series)
    interest = _as_series(inc5.get("interestExpense", 0))
    ebit = _as_series(inc5.get(op_col, 0))
    pretax = (pd.Series(ebit) - pd.Series(interest)).replace(0, np.nan)
    eff_tax = _avg_ratio(tax_exp, pretax, 0.21)

    return float(max(0.0, min(0.35, eff_tax)))


def _calculate_cost_of_debt(bal5: pd.DataFrame, inc5: pd.DataFrame) -> float:
    """Calculate cost of debt."""
    debt_cols = [
        c for c in [
            "longTermDebt",
            "shortTermDebt"] if c in bal5.columns]

    if debt_cols:
        debt_series = bal5[debt_cols].fillna(0).sum(axis=1).astype(float)
        avg_debt = float(debt_series.replace(0, np.nan).mean())
        interest = inc5.get("interestExpense", 0)
        avg_interest = float(pd.Series(interest).replace(0, np.nan).mean())
        cod = (avg_interest / avg_debt) if avg_debt else np.nan
    else:
        cod = np.nan

    return cod if not np.isnan(cod) and cod > 0 else 0.05


def infer_drivers(inc5: pd.DataFrame, bal5: pd.DataFrame,
                  cfs5: pd.DataFrame) -> Dict[str, float]:
    """Compute key ratios and growth assumptions from historical data."""
    rev_col = _find_col(
        inc5, [
            "totalRevenue", "revenue", "TotalRevenue", "Revenue"])
    if not rev_col:
        raise RuntimeError("Revenue column not found")

    # Calculate individual components
    rev_cagr = _calculate_revenue_cagr(inc5, rev_col)
    gross_margin, ebit_margin, opex_margin, op_col = _calculate_margins(
        inc5, rev_col)
    da_to_rev = _calculate_da_ratio(inc5, cfs5, rev_col)
    nwc_to_rev = _calculate_nwc_ratio(bal5, inc5, rev_col)
    eff_tax = _calculate_tax_rate(inc5, cfs5, op_col)
    cost_of_debt = _calculate_cost_of_debt(bal5, inc5)

    return {
        "rev_col": rev_col,
        "rev_cagr": float(rev_cagr),
        "gross_margin": float(gross_margin),
        "ebit_margin": float(ebit_margin),
        "opex_margin": float(opex_margin),
        "da_to_rev": float(da_to_rev),
        "capex_to_rev": float(
            _avg_ratio(
                abs(
                    cfs5.get(
                        "capitalExpenditures",
                        0)),
                inc5.get(
                    rev_col,
                    0),
                0.05)),
        "nwc_to_rev": float(nwc_to_rev),
        "eff_tax": float(eff_tax),
        "cost_of_debt": float(cost_of_debt),
    }
