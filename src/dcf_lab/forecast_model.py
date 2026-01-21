"""Build forward financial statement projections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .assumption_builder import _hist_n
from .settings import DEFAULT_MARKET_RISK_PREMIUM, DEFAULT_RISK_FREE_RATE


@dataclass
class ForecastInputs:
    """Inputs controlling the forecast and valuation model."""
    years: int = 7
    terminal_method: str = "gordon"
    terminal_growth: float = 0.025
    exit_multiple: float = 10.0
    rf_rate: float = DEFAULT_RISK_FREE_RATE
    market_risk_premium: float = DEFAULT_MARKET_RISK_PREMIUM
    beta: Optional[float] = None
    cost_of_debt: Optional[float] = None
    tax_rate: Optional[float] = None
    shares_outstanding: Optional[float] = None
    cash: Optional[float] = None
    total_debt: Optional[float] = None
    minority_interest: float = 0.0
    preferred_equity: float = 0.0


def build_forecast(
    inc_a: pd.DataFrame,
    bal_a: pd.DataFrame,
    cfs_a: pd.DataFrame,
    drivers: Dict[str, float],
    inputs: ForecastInputs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Return projected income statement, balance sheet and cash flow dataframes."""
    def _to_float(x, default: float = 0.0) -> float:
        try:
            if isinstance(x, pd.Series):
                if not x.empty:
                    return float(x.iloc[0])
                return default
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    inc5 = _hist_n(inc_a.copy())
    bal5 = _hist_n(bal_a.copy())
    last_inc = inc5.sort_values("date").iloc[-1] if not inc5.empty else None
    last_bal = bal5.sort_values("date").iloc[-1] if not bal5.empty else None
    last_rev = _to_float(
        last_inc.get(
            drivers["rev_col"]) if last_inc is not None else 0.0)
    cash0 = _to_float(
        last_bal.get("cashAndCashEquivalentsAtCarryingValue")
        if last_bal is not None else 0.0)
    ltd0 = _to_float(last_bal.get("longTermDebt")
                     if last_bal is not None else 0.0)
    std0 = _to_float(last_bal.get("shortTermDebt")
                     if last_bal is not None else 0.0)
    debt0 = ltd0 + std0
    tca0 = _to_float(last_bal.get("totalCurrentAssets")
                     if last_bal is not None else 0.0)
    tcl0 = _to_float(last_bal.get("totalCurrentLiabilities")
                     if last_bal is not None else 0.0)
    nwc0 = tca0 - tcl0
    ppe0 = _to_float(last_bal.get("propertyPlantEquipmentNet")
                     if last_bal is not None else 0.0)
    equity0 = _to_float(last_bal.get("totalShareholderEquity")
                        if last_bal is not None else 0.0)
    cash0 = inputs.cash if inputs.cash is not None else cash0
    debt0 = inputs.total_debt if inputs.total_debt is not None else debt0

    def growth_path(start: float, terminal: float, years: int) -> list[float]:
        if years == 1:
            return [terminal]
        return [start + (terminal - start) * i / (years - 1)
                for i in range(years)]

    # Allow per-year growth path if provided by ML
    if isinstance(
        drivers, dict) and isinstance(
        drivers.get("rev_growth_path"), (list, tuple)) and len(
            drivers.get("rev_growth_path")) == inputs.years:
        g_list = list(drivers.get("rev_growth_path"))
    else:
        g_list = growth_path(
            max(drivers["rev_cagr"], -0.2), inputs.terminal_growth, inputs.years)
    cod = inputs.cost_of_debt if inputs.cost_of_debt is not None else drivers[
        "cost_of_debt"]
    tax = inputs.tax_rate if inputs.tax_rate is not None else drivers["eff_tax"]

    rows_is: list[dict] = []
    rows_bs: list[dict] = []
    rows_cf: list[dict] = []
    rev_prev = last_rev
    cash = cash0
    debt = debt0
    nwc_prev = nwc0
    ppe = ppe0
    equity = equity0
    for y, g in enumerate(g_list, start=1):
        rev = rev_prev * (1 + g)
        # Per-year margin/ratio paths (fallback to constants)
        gm_path = drivers.get("gross_margin_path") if isinstance(
            drivers, dict) else None
        em_path = drivers.get("ebit_margin_path") if isinstance(
            drivers, dict) else None
        dar_path = drivers.get("da_to_rev_path") if isinstance(
            drivers, dict) else None
        cx_path = drivers.get("capex_to_rev_path") if isinstance(
            drivers, dict) else None
        nw_path = drivers.get("nwc_to_rev_path") if isinstance(
            drivers, dict) else None

        gross_margin = (gm_path[y-1] if gm_path and len(gm_path)
                        # type: ignore[index]
                        >= y else drivers["gross_margin"])
        ebit_margin = (em_path[y-1] if em_path and len(em_path) >=
                       y else drivers["ebit_margin"])    # type: ignore[index]
        da_to_rev = (dar_path[y-1] if dar_path and len(dar_path)
                     >= y else drivers["da_to_rev"])     # type: ignore[index]
        capex_to_rev = (cx_path[y-1] if cx_path and len(cx_path)
                        # type: ignore[index]
                        >= y else drivers["capex_to_rev"])
        nwc_to_rev = (nw_path[y-1] if nw_path and len(nw_path) >=
                      y else drivers["nwc_to_rev"])      # type: ignore[index]

        gross = rev * gross_margin
        opex = rev * max(0.0, gross_margin - ebit_margin)
        ebit = rev * ebit_margin
        da = rev * da_to_rev
        interest = debt * cod
        ebt = ebit - interest
        taxes = max(0.0, ebt) * tax
        net = ebt - taxes
        nwc_target = rev * nwc_to_rev
        delta_nwc = nwc_target - nwc_prev
        capex = rev * capex_to_rev
        nopat = ebit * (1 - tax)
        cfo = nopat + da - delta_nwc
        cfi = -capex
        cff = 0.0
        delta_cash = cfo + cfi + cff
        cash = cash + delta_cash
        ppe = max(0.0, ppe + capex - da)
        equity = equity + net
        rows_is.append({
            "year": y,
            "revenue": rev,
            "gross_profit": gross,
            "opex": opex,
            "ebit": ebit,
            "interest_expense": interest,
            "taxes": taxes,
            "net_income": net,
            "d_and_a": da,
        })
        rows_bs.append({
            "year": y,
            "cash": cash,
            "nwc": nwc_target,
            "ppene": ppe,
            "total_debt": debt,
            "equity": equity,
            "assets_model": cash + nwc_target + ppe,
            "liab_plus_equity_model": debt + equity,
        })
        rows_cf.append({
            "year": y,
            "cfo": cfo,
            "cfi": cfi,
            "cf": cff,
            "delta_cash": delta_cash,
            "fcf": nopat + da - capex - delta_nwc,
            "delta_nwc": delta_nwc,
            "capex": capex,
        })
        rev_prev = rev
        nwc_prev = nwc_target

    is_df = pd.DataFrame(rows_is)
    bs_df = pd.DataFrame(rows_bs)
    cf_df = pd.DataFrame(rows_cf)
    starting = {
        "revenue": _to_float(last_rev),
        "cash": _to_float(cash0),
        "debt": _to_float(debt0),
        "nwc": _to_float(nwc0),
        "ppene": _to_float(ppe0),
        "equity": _to_float(equity0),
    }
    return is_df, bs_df, cf_df, starting
