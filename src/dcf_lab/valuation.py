"""Valuation utilities for computing enterprise and equity value."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .forecast_model import ForecastInputs
from .utils import gordon_terminal_value, wacc


def discounted_cash_flow(
    cf_df: pd.DataFrame,
    is_df: pd.DataFrame,
    drivers: Dict[str, float],
    inputs: ForecastInputs,
    marketcap: Optional[float],
    starting: Dict[str, float],
) -> Dict[str, float]:
    """Compute a DCF valuation from forecast cash flows."""
    last_fcff = cf_df["fcf"].iloc[-1]
    last_ebitda = (is_df["ebit"] + is_df["d_and_a"]).iloc[-1]
    cod = inputs.cost_of_debt if inputs.cost_of_debt is not None else drivers[
        "cost_of_debt"]
    tax = inputs.tax_rate if inputs.tax_rate is not None else drivers["eff_tax"]
    beta = inputs.beta if inputs.beta is not None else 1.0
    coe = inputs.rf_rate + beta * inputs.market_risk_premium
    equity_value = marketcap or max(starting.get("equity", 0.0), 0.0)
    wacc_rate = wacc(
        coe,
        cod,
        equity_value,
        starting.get(
            "debt",
            0.0),
        tax) or 0.09
    pv_fcff = sum(cf / ((1 + wacc_rate) ** (i + 1))
                  for i, cf in enumerate(cf_df["fcf"].tolist()))
    if inputs.terminal_method == "gordon":
        tv = gordon_terminal_value(
            last_fcff * (
                1 + inputs.terminal_growth),
            wacc_rate,
            inputs.terminal_growth) or last_ebitda * 8.0
    else:
        tv = last_ebitda * inputs.exit_multiple
    pv_tv = tv / ((1 + wacc_rate) ** inputs.years)
    enterprise_value = pv_fcff + pv_tv
    net_debt = (
        inputs.total_debt if inputs.total_debt is not None 
        else starting.get("debt", 0.0)
    ) - (
        inputs.cash if inputs.cash is not None 
        else starting.get("cash", 0.0)
    )
    equity_value_model = enterprise_value - net_debt - \
        inputs.minority_interest - inputs.preferred_equity
    per_share = (
        equity_value_model / inputs.shares_outstanding
        if (inputs.shares_outstanding and inputs.shares_outstanding > 0)
        else None
    )
    return {
        "wacc": wacc_rate,
        "cost_of_equity": coe,
        "cost_of_debt": cod,
        "pv_fcf": pv_fcff,
        "pv_terminal": pv_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value_model,
        "per_share": per_share,
    }


def _create_adjusted_inputs(
    inputs: ForecastInputs,
    base_g: float,
    delta_g: float,
) -> ForecastInputs:
    """Create adjusted inputs with modified terminal growth rate."""
    return ForecastInputs(
        years=inputs.years,
        terminal_method=inputs.terminal_method,
        terminal_growth=base_g + delta_g,
        exit_multiple=inputs.exit_multiple,
        rf_rate=inputs.rf_rate,
        market_risk_premium=inputs.market_risk_premium,
        beta=inputs.beta,
        cost_of_debt=inputs.cost_of_debt,
        tax_rate=inputs.tax_rate,
        shares_outstanding=inputs.shares_outstanding,
        cash=inputs.cash,
        total_debt=inputs.total_debt,
        minority_interest=inputs.minority_interest,
        preferred_equity=inputs.preferred_equity,
    )


def _calculate_adjusted_beta(
    new_inputs: ForecastInputs,
    drivers: Dict[str, float],
    target_wacc: float,
    marketcap: Optional[float],
    starting: Dict[str, float],
) -> float:
    """Calculate adjusted beta to achieve target WACC."""
    cod = (
        new_inputs.cost_of_debt if new_inputs.cost_of_debt is not None 
        else drivers["cost_of_debt"]
    )
    tax = new_inputs.tax_rate if new_inputs.tax_rate is not None else drivers[
        "eff_tax"]
    E = marketcap or max(starting.get("equity", 0.0), 0.0)
    D = starting.get("debt", 0.0)
    V = E + D if (E + D) > 0 else 1.0
    cod_at = float(cod or 0) * (1 - float(tax or 0))
    mrp = new_inputs.market_risk_premium
    rf = new_inputs.rf_rate

    if not mrp:
        return new_inputs.beta or 1.0

    return ((target_wacc - (D / V) * cod_at) - (E / V) * rf) / ((E / V) * mrp)


def _create_grid_row(
    cf_df: pd.DataFrame,
    is_df: pd.DataFrame,
    drivers: Dict[str, float],
    new_inputs: ForecastInputs,
    target_wacc: float,
    marketcap: Optional[float],
    starting: Dict[str, float],
) -> Dict[str, Any]:
    """Create a single row for the sensitivity grid."""
    new_inputs.beta = _calculate_adjusted_beta(
        new_inputs, drivers, target_wacc, marketcap, starting)
    d = discounted_cash_flow(
        cf_df,
        is_df,
        drivers,
        new_inputs,
        marketcap,
        starting)

    return {
        "wacc": round(target_wacc, 4),
        "g": round(new_inputs.terminal_growth, 4),
        "equity_value": d.get("equity_value"),
        "per_share": d.get("per_share"),
    }


def sensitivity_grid(
    cf_df: pd.DataFrame,
    is_df: pd.DataFrame,
    drivers: Dict[str, float],
    inputs: ForecastInputs,
    marketcap: Optional[float],
    starting: Dict[str, float],
    wacc_step: float = 0.01,
    g_step: float = 0.005,
) -> pd.DataFrame:
    """Compute a 2D sensitivity grid for WACC ±1% and g ±0.5%."""
    base = discounted_cash_flow(
        cf_df,
        is_df,
        drivers,
        inputs,
        marketcap,
        starting)
    base_wacc = float(base["wacc"])
    base_g = float(inputs.terminal_growth)
    rows = []

    wacc_variations = (-wacc_step, 0.0, wacc_step)
    growth_variations = (-g_step, 0.0, g_step)

    for dw in wacc_variations:
        for dg in growth_variations:
            new_inputs = _create_adjusted_inputs(inputs, base_g, dg)
            target_wacc = base_wacc + dw
            row = _create_grid_row(
                cf_df,
                is_df,
                drivers,
                new_inputs,
                target_wacc,
                marketcap,
                starting)
            rows.append(row)

    return pd.DataFrame(rows)


def monte_carlo(
    inc_df: pd.DataFrame,
    bal_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    drivers: Dict[str, float],
    inputs: ForecastInputs,
    marketcap: Optional[float],
    num_simulations: int = 1000,
) -> Tuple[list[float], list[float]]:
    """Run Monte Carlo over key drivers and WACC; return equity and per-share arrays."""
    import numpy as np

    from .forecast_model import build_forecast

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)

    eq_values: list[float] = []
    per_share_vals: list[float] = []

    for _ in range(num_simulations):
        d = drivers.copy()
        # Perturb growth/margins within reasonable bounds
        d["rev_cagr"] = float(d.get("rev_cagr", 0.05)) * \
            (1 + rng.normal(0, 0.3))
        d["ebit_margin"] = float(
            d.get("ebit_margin", 0.15)) + rng.normal(0, 0.02)
        d["gross_margin"] = float(
            d.get("gross_margin", 0.4)) + rng.normal(0, 0.02)
        d["da_to_rev"] = max(
            0.0,
            float(
                d.get(
                    "da_to_rev",
                    0.04)) +
            rng.normal(
                0,
                0.01))
        d["capex_to_rev"] = max(
            0.0,
            float(
                d.get(
                    "capex_to_rev",
                    0.05)) +
            rng.normal(
                0,
                0.01))
        d["nwc_to_rev"] = max(
            0.0,
            float(
                d.get(
                    "nwc_to_rev",
                    0.03)) +
            rng.normal(
                0,
                0.01))
        # Create a shallow copy of inputs and perturb beta (WACC) and terminal
        # growth slightly
        inp = ForecastInputs(
            years=inputs.years,
            terminal_method=inputs.terminal_method,
            terminal_growth=float(inputs.terminal_growth) +
            rng.normal(0, 0.0025),
            exit_multiple=inputs.exit_multiple,
            rf_rate=inputs.rf_rate,
            market_risk_premium=inputs.market_risk_premium,
            beta=(inputs.beta if inputs.beta is not None else 1.0) +
            rng.normal(0, 0.15),
            cost_of_debt=inputs.cost_of_debt,
            tax_rate=inputs.tax_rate,
            shares_outstanding=inputs.shares_outstanding,
            cash=inputs.cash,
            total_debt=inputs.total_debt,
            minority_interest=inputs.minority_interest,
            preferred_equity=inputs.preferred_equity,
        )
        is_f, _, cf_f, starting = build_forecast(
            inc_df, bal_df, cfs_df, d, inp)
        res = discounted_cash_flow(cf_f, is_f, d, inp, marketcap, starting)
        eq_values.append(float(res.get("equity_value") or 0))
        per_share_vals.append(float(res.get("per_share") or 0))

    return eq_values, per_share_vals
