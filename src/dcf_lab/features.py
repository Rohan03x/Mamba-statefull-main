from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .assumption_builder import _hist_n, infer_drivers


def compute_drivers(inc: pd.DataFrame, bal: pd.DataFrame,
                    cfs: pd.DataFrame) -> Dict[str, float]:
    return infer_drivers(_hist_n(inc), _hist_n(bal), _hist_n(cfs))


def _extract_market_value_components(
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


def _extract_income_statement_metrics(
        inc: pd.DataFrame) -> tuple[float | None, float, float | None, float | None]:
    """Extract key metrics from the income statement."""
    ebit = float(inc.get("operatingIncome", pd.Series(
        [np.nan])).iloc[-1]) if not inc.empty else None
    da = float(inc.get("depreciationAndAmortization",
                       pd.Series([0])).iloc[-1]) if not inc.empty else 0.0
    ebitda = (ebit + da) if ebit is not None else None
    net_income = float(inc.get("netIncome", pd.Series(
        [np.nan])).iloc[-1]) if not inc.empty else None

    return ebit, da, ebitda, net_income


def _calculate_pe_ratio(
        market: Dict,
        net_income: float | None) -> float | None:
    """Calculate P/E ratio based on price, shares, and net income."""
    price = float(market.get("price") or 0)
    shares = float(market.get("shares") or 0)

    pe = None
    if price and shares and net_income and net_income != 0:
        eps = net_income / shares
        pe = price / eps if eps != 0 else None

    return pe


def live_multiples(market: Dict, inc: pd.DataFrame,
                   bal: pd.DataFrame) -> Dict[str, float | None]:
    """Compute EV-based and price-based multiples from last available LTM/year."""
    # Extract market value components
    _, _, _, ev = _extract_market_value_components(market, bal)

    # Extract income statement metrics
    ebit, _, ebitda, net_income = _extract_income_statement_metrics(inc)

    # Calculate multiples
    out: Dict[str, float | None] = {"ev": ev}
    out["ev_ebitda"] = (ev / ebitda) if ev and ebitda and ebitda != 0 else None
    out["ev_ebit"] = (ev / ebit) if ev and ebit and ebit != 0 else None

    # Calculate PE ratio
    pe = _calculate_pe_ratio(market, net_income)
    out["pe"] = pe

    return out
