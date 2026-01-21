"""
Core features implementation - direct access to features.py functions.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


# Directly implement the functions from features.py to avoid import issues
def compute_drivers(inc: pd.DataFrame, bal: pd.DataFrame,
                    cfs: pd.DataFrame) -> Dict[str, float]:
    """
    Infer drivers from historical financial data.

    This is a simplified version of the function from features.py to avoid import issues.
    """
    from dcf_lab.assumption_builder import _hist_n, infer_drivers
    return infer_drivers(_hist_n(inc), _hist_n(bal), _hist_n(cfs))


def _extract_enterprise_value_inputs(
        market: Dict, bal: pd.DataFrame) -> tuple[float, float, float, float | None]:
    """Extract market cap, cash, and debt to calculate enterprise value."""
    mc = float(market.get("marketcap") or 0)
    cash = float(bal.get("cashAndCashEquivalentsAtCarryingValue",
                 pd.Series([0])).iloc[-1] if not bal.empty else 0)

    debt_cols = [
        c for c in [
            "longTermDebt",
            "shortTermDebt",
            "totalDebt"] if c in bal.columns]
    debt = float(bal[debt_cols].fillna(0).sum(axis=1).iloc[-1]
                 ) if debt_cols and not bal.empty else 0.0

    ev = mc + debt - cash if mc else None

    return mc, cash, debt, ev


def _extract_financial_metrics(
        inc: pd.DataFrame, bal: pd.DataFrame) -> tuple[float, float, float, float]:
    """Extract key financial metrics from income statement and balance sheet."""
    last_revenue = float(inc.get("totalRevenue", pd.Series([0])).iloc[-1])
    last_ebitda = float(inc.get("ebitda", pd.Series([0])).iloc[-1])
    last_net_income = float(inc.get("netIncome", pd.Series([0])).iloc[-1])
    last_book_value = float(
        bal.get("totalStockholderEquity", pd.Series([0])).iloc[-1])

    return last_revenue, last_ebitda, last_net_income, last_book_value


def _calculate_multiples(
    mc: float,
    ev: float | None,
    last_revenue: float,
    last_ebitda: float,
    last_net_income: float,
    last_book_value: float
) -> Dict[str, float | None]:
    """Calculate various valuation multiples based on financial metrics."""
    result = {}

    # Price-based multiples
    if mc and last_revenue:
        result["p_sales"] = mc / last_revenue

    if mc and last_net_income and last_net_income > 0:
        result["pe"] = mc / last_net_income

    if mc and last_book_value and last_book_value > 0:
        result["pb"] = mc / last_book_value

    # Enterprise value-based multiples
    if ev and last_revenue:
        result["ev_sales"] = ev / last_revenue

    if ev and last_ebitda and last_ebitda > 0:
        result["ev_ebitda"] = ev / last_ebitda

    return result


def live_multiples(market: Dict, inc: pd.DataFrame,
                   bal: pd.DataFrame) -> Dict[str, float | None]:
    """
    Compute EV-based and price-based multiples from last available LTM/year.

    This is copied from features.py to avoid import issues.
    """
    # Extract enterprise value components
    mc, _, _, ev = _extract_enterprise_value_inputs(market, bal)

    result = {}
    if not inc.empty and not bal.empty:
        # Extract financial metrics
        last_revenue, last_ebitda, last_net_income, last_book_value = _extract_financial_metrics(
            inc, bal)

        # Calculate and return multiples
        result = _calculate_multiples(
            mc, ev, last_revenue, last_ebitda, last_net_income, last_book_value
        )

    return result
