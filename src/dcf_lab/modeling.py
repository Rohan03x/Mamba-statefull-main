from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import clamp, gordon_terminal_value, wacc


@dataclass
class Inputs:
    years: int = 7
    terminal_method: str = "gordon"
    terminal_growth: float = 0.025
    exit_multiple: float = 10.0
    rf_rate: float = 0.04
    market_risk_premium: float = 0.05
    beta: Optional[float] = None
    cost_of_debt: Optional[float] = None
    tax_rate: Optional[float] = None
    shares_outstanding: Optional[float] = None
    cash: Optional[float] = None
    total_debt: Optional[float] = None
    minority_interest: float = 0.0
    preferred_equity: float = 0.0


def _hist5(df):
    if df is None or df.empty:
        return df
    return df.sort_values("date").dropna(subset=["date"]).tail(5).copy()


def _avg_ratio(num, den, default):
    try:
        n = float(pd.Series(num).sum())
        d = float(pd.Series(den).sum())
        return default if d == 0 else n/d
    except Exception:
        return default


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def infer_drivers(inc5, bal5, cfs5):
    rev_col = _find_col(
        inc5, [
            "totalRevenue", "revenue", "TotalRevenue", "Revenue"])
    if not rev_col:
        raise RuntimeError("Revenue column not found")
    rev = inc5[["date", rev_col]].dropna().set_index("date").sort_index()
    if len(rev) >= 2:
        years = (rev.index[-1].year - rev.index[0].year) or 1
        cagr = (rev.iloc[-1, 0]/max(1e-6, rev.iloc[0, 0]))**(1/years)-1
        rev_cagr = clamp(cagr, -0.5, 0.5)
    else:
        rev_cagr = 0.05
    gp_col = _find_col(inc5, ["grossProfit", "GrossProfit"])
    op_col = _find_col(inc5, ["operatingIncome", "OperatingIncome", "ebit"])
    da_col = _find_col(
        inc5, [
            "depreciationAndAmortization", "depreciation", "amortization"])
    gross_margin = _avg_ratio(inc5.get(gp_col, 0), inc5.get(rev_col, 0), 0.40)
    ebit_margin = _avg_ratio(inc5.get(op_col, 0), inc5.get(rev_col, 0), 0.15)
    opex_margin = max(0.0, gross_margin - ebit_margin)
    da_to_rev = _avg_ratio(inc5.get(da_col, 0), inc5.get(rev_col, 0), 0.04)
    if {"totalCurrentAssets", "totalCurrentLiabilities"}.issubset(
            bal5.columns):
        nwc = bal5["totalCurrentAssets"].astype(
            float)-bal5["totalCurrentLiabilities"].astype(float)
        nwc_to_rev = _avg_ratio(nwc, inc5.get(rev_col, 0), 0.03)
    else:
        nwc_to_rev = 0.03
    tax_exp = inc5.get("incomeTaxExpense", 0)
    interest = inc5.get("interestExpense", 0)
    ebit = inc5.get(op_col, 0)
    pretax = (pd.Series(ebit)-pd.Series(interest)).replace(0, np.nan)
    eff_tax = _avg_ratio(tax_exp, pretax, 0.21)
    eff_tax = float(
        max(0.0, min(0.35, eff_tax)))
    debt_cols = [
        c for c in [
            "longTermDebt",
            "shortTermDebt"] if c in bal5.columns]
    if debt_cols:
        debt_series = bal5[debt_cols].fillna(0).sum(axis=1).astype(float)
        avg_debt = float(
            debt_series.replace(
                0, np.nan).mean())
        avg_interest = float(
            pd.Series(interest).replace(
                0, np.nan).mean())
        cod = (avg_interest/avg_debt) if avg_debt else np.nan
    else:
        cod = np.nan
    if np.isnan(cod) or cod <= 0:
        cod = 0.05
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
                        'capitalExpenditures',
                        0)),
                inc5.get(
                    rev_col,
                    0),
                0.05)),
        "nwc_to_rev": float(nwc_to_rev),
        "eff_tax": float(eff_tax),
        "cost_of_debt": float(cod)}


def _extract_historical_data(inc_a, bal_a, cfs_a):
    """Extract and process historical financial data."""
    inc5 = _hist5(inc_a.copy())
    bal5 = _hist5(bal_a.copy())
    cfs5 = _hist5(cfs_a.copy())
    drivers = infer_drivers(inc5, bal5, cfs5)

    last_inc = inc5.sort_values("date").iloc[-1] if not inc5.empty else None
    last_bal = bal5.sort_values("date").iloc[-1] if not bal5.empty else None

    last_rev = float(
        last_inc.get(
            drivers["rev_col"],
            0.0)) if last_inc is not None else 0.0
    cash0 = float(
        last_bal.get(
            "cashAndCashEquivalentsAtCarryingValue",
            0.0)) if last_bal is not None else 0.0

    debt_long = float(last_bal.get("longTermDebt", 0.0)
                      ) if last_bal is not None else 0.0
    debt_short = float(last_bal.get("shortTermDebt", 0.0)
                       ) if last_bal is not None else 0.0
    debt0 = debt_long + debt_short

    current_assets = float(
        last_bal.get(
            "totalCurrentAssets",
            0.0)) if last_bal is not None else 0.0
    current_liabilities = float(
        last_bal.get(
            "totalCurrentLiabilities",
            0.0)) if last_bal is not None else 0.0
    nwc0 = current_assets - current_liabilities

    ppe0 = float(
        last_bal.get(
            "propertyPlantEquipmentNet",
            0.0)) if last_bal is not None else 0.0
    equity0 = float(
        last_bal.get(
            "totalShareholderEquity",
            0.0)) if last_bal is not None else 0.0

    return drivers, last_rev, cash0, debt0, nwc0, ppe0, equity0


def _calculate_growth_path(start_growth, terminal_growth, years):
    """Calculate the growth path from start to terminal growth."""
    if years == 1:
        return [terminal_growth]
    return [start_growth + (terminal_growth - start_growth)
            * i / (years - 1) for i in range(years)]


def _calculate_financial_metrics(inputs, drivers, marketcap, equity0, debt0):
    """Calculate key financial metrics needed for forecasting."""
    beta = inputs.beta if inputs.beta is not None else 1.0
    coe = inputs.rf_rate + beta * inputs.market_risk_premium
    cod = inputs.cost_of_debt if inputs.cost_of_debt is not None else drivers[
        "cost_of_debt"]
    tax = inputs.tax_rate if inputs.tax_rate is not None else drivers["eff_tax"]
    equity_value = marketcap or max(equity0, 0.0)
    wacc_rate = wacc(coe, cod, equity_value, debt0, tax) or 0.09

    return coe, cod, tax, wacc_rate


def _forecast_financials(
        g_list,
        drivers,
        last_rev,
        cash0,
        debt0,
        nwc0,
        ppe0,
        equity0,
        cod,
        tax):
    """Generate forecasted financial statements based on growth projections."""
    rows_is = []
    rows_bs = []
    rows_cf = []

    rev_prev = last_rev
    cash = cash0
    debt = debt0
    nwc_prev = nwc0
    ppe = ppe0
    equity = equity0

    for y, g in enumerate(g_list, start=1):
        # Income statement calculations
        rev = rev_prev * (1 + g)
        gross = rev * drivers["gross_margin"]
        opex = rev * drivers["opex_margin"]
        ebit = rev * drivers["ebit_margin"]
        da = rev * drivers["da_to_rev"]
        interest = debt * cod
        ebt = ebit - interest
        taxes = max(0.0, ebt) * tax
        net = ebt - taxes

        # Cash flow and balance sheet calculations
        nwc_target = rev * drivers["nwc_to_rev"]
        delta_nwc = nwc_target - nwc_prev
        capex = rev * drivers["capex_to_rev"]
        nopat = ebit * (1 - tax)
        cfo = nopat + da - delta_nwc
        cfi = -capex
        cff = 0.0
        delta_cash = cfo + cfi + cff
        cash = cash + delta_cash
        ppe = max(0.0, ppe + capex - da)
        equity = equity + net

        # Append rows for each financial statement
        rows_is.append({
            "year": y, "revenue": rev, "gross_profit": gross, "opex": opex,
            "ebit": ebit, "interest_expense": interest, "taxes": taxes,
            "net_income": net, "d_and_a": da
        })

        rows_bs.append({
            "year": y, "cash": cash, "nwc": nwc_target, "ppene": ppe,
            "total_debt": debt, "equity": equity,
            "assets_model": cash + nwc_target + ppe,
            "liab_plus_equity_model": debt + equity
        })

        rows_cf.append({
            "year": y, "cfo": cfo, "cfi": cfi, "cf": cff,
            "delta_cash": delta_cash, "fcf": nopat + da - capex - delta_nwc,
            "delta_nwc": delta_nwc, "capex": capex
        })

        rev_prev = rev
        nwc_prev = nwc_target

    return pd.DataFrame(rows_is), pd.DataFrame(rows_bs), pd.DataFrame(rows_cf)


def _calculate_valuation(is_df, cf_df, inputs, wacc_rate, debt0, cash0):
    """Calculate enterprise value, equity value, and per-share value."""
    pv_fcff = sum(cf / ((1 + wacc_rate) ** (i + 1))
                  for i, cf in enumerate(cf_df["fcf"].tolist()))
    last_fcff = cf_df["fcf"].iloc[-1]
    last_ebitda = (is_df["ebit"] + is_df["d_and_a"]).iloc[-1]

    if inputs.terminal_method == "gordon":
        tv = gordon_terminal_value(
            last_fcff * (1 + inputs.terminal_growth),
            wacc_rate, inputs.terminal_growth)
        if tv is None:
            tv = last_ebitda * 8.0
    else:
        tv = last_ebitda * inputs.exit_multiple

    pv_tv = tv / ((1 + wacc_rate) ** inputs.years)
    enterprise_value = pv_fcff + pv_tv
    net_debt = (inputs.total_debt or debt0) - (inputs.cash or cash0)
    equity_value = enterprise_value - net_debt - \
        (inputs.minority_interest or 0.0) - (inputs.preferred_equity or 0.0)

    per_share = None
    if inputs.shares_outstanding and inputs.shares_outstanding > 0:
        per_share = equity_value / inputs.shares_outstanding

    return pv_fcff, pv_tv, enterprise_value, equity_value, per_share


def three_statement_forecast(inc_a, bal_a, cfs_a, marketcap, inputs: Inputs):
    """Generate a three-statement financial forecast and valuation."""
    # Extract historical data and drivers
    drivers, last_rev, cash0, debt0, nwc0, ppe0, equity0 = _extract_historical_data(
        inc_a, bal_a, cfs_a)

    # Apply user inputs if provided
    cash0 = inputs.cash if inputs.cash else cash0
    debt0 = inputs.total_debt if inputs.total_debt else debt0

    # Calculate growth path and financial metrics
    g_list = _calculate_growth_path(
        max(drivers["rev_cagr"], -0.2),
        inputs.terminal_growth,
        inputs.years
    )

    coe, cod, tax, wacc_rate = _calculate_financial_metrics(
        inputs, drivers, marketcap, equity0, debt0
    )

    # Generate forecasted financials
    is_df, bs_df, cf_df = _forecast_financials(
        g_list, drivers, last_rev, cash0, debt0, nwc0, ppe0, equity0, cod, tax
    )

    # Calculate valuation metrics
    pv_fcff, pv_tv, enterprise_value, equity_value_model, per_share = _calculate_valuation(
        is_df, cf_df, inputs, wacc_rate, debt0, cash0)

    # Prepare summary and assumptions for return
    summary = {
        "wacc": wacc_rate,
        "cost_of_equity": coe,
        "cost_of_debt": cod,
        "pv_fcf": pv_fcff,
        "pv_terminal": pv_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value_model,
        "per_share": per_share
    }

    assumptions = {
        "drivers": drivers,
        "starting": {
            "revenue": float(last_rev),
            "cash": float(cash0),
            "debt": float(debt0),
            "nwc": float(nwc0),
            "ppene": float(ppe0),
            "equity": float(equity0)
        }
    }

    return is_df, bs_df, cf_df, summary, assumptions
