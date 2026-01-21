"""Command-line interface tying data retrieval, forecasting and valuation."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .assumption_builder import _hist_n, infer_drivers
from .data_sources import (
    alpha_balance,
    alpha_cashflow,
    alpha_income,
    alpha_overview,
    overview_fields,
    parse_financials,
    yf_overview,
    yf_statements,
)
from .exporters import to_excel
from .forecast_model import ForecastInputs, build_forecast
from .settings import DEFAULT_MARKET_RISK_PREMIUM, DEFAULT_RISK_FREE_RATE
from .valuation import discounted_cash_flow


def _fetch_alpha(ticker: str):
    overview = alpha_overview(ticker) or {}
    inc = alpha_income(ticker) or {}
    bal = alpha_balance(ticker) or {}
    cfs = alpha_cashflow(ticker) or {}
    inc_a, bal_a, cfs_a = parse_financials(inc, bal, cfs)
    ov = overview_fields(overview)
    return inc_a, bal_a, cfs_a, ov


def _fetch_yf(ticker: str):
    inc, bal, cfs, _price = yf_statements(ticker)
    ov = yf_overview(ticker)
    return inc, bal, cfs, ov


def run(
    ticker: str,
    source: str = "alpha",
    years: int = 7,
    terminal_method: str = "gordon",
    terminal_growth: float = 0.025,
    exit_multiple: float = 10.0,
    rf: float = DEFAULT_RISK_FREE_RATE,
    mrp: float = DEFAULT_MARKET_RISK_PREMIUM,
    beta: Optional[float] = None,
    tax: Optional[float] = None,
    cod: Optional[float] = None,
    cash: Optional[float] = None,
    debt: Optional[float] = None,
    shares: Optional[float] = None,
    excel_out: Optional[str] = None,
) -> dict:
    """Execute the valuation workflow and return a summary."""
    try:
        if source == "alpha":
            inc_a, bal_a, cfs_a, ov = _fetch_alpha(ticker)
        elif source == "yfinance":
            inc_a, bal_a, cfs_a, ov = _fetch_yf(ticker)
        else:
            raise ValueError("Unknown data source")
        if inc_a.empty or bal_a.empty or cfs_a.empty:
            raise RuntimeError("Missing financials for ticker")
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        sys.exit(1)

    drivers = infer_drivers(_hist_n(inc_a), _hist_n(bal_a), _hist_n(cfs_a))
    inputs = ForecastInputs(
        years=years,
        terminal_method=terminal_method,
        terminal_growth=float(terminal_growth),
        exit_multiple=float(exit_multiple),
        rf_rate=float(rf),
        market_risk_premium=float(mrp),
        beta=beta if beta is not None else ov.get("beta"),
        tax_rate=tax,
        cost_of_debt=cod,
        shares_outstanding=shares if shares is not None else ov.get("shares"),
        cash=cash,
        total_debt=debt,
    )
    is_df, bs_df, cf_df, starting = build_forecast(
        inc_a, bal_a, cfs_a, drivers, inputs)
    summary = discounted_cash_flow(
        cf_df,
        is_df,
        drivers,
        inputs,
        ov.get("marketcap"),
        starting)
    if excel_out:
        to_excel(excel_out, ticker, is_df, bs_df, cf_df, summary,
                 {"drivers": drivers, "starting": starting})
    return {"ticker": ticker, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--source", choices=["alpha", "yfinance"], default="alpha")
    p.add_argument("--years", type=int, default=7)
    p.add_argument(
        "--terminal",
        choices=[
            "gordon",
            "exit_multiple"],
        default="gordon")
    p.add_argument("--g", type=float, default=0.025)
    p.add_argument("--multiple", type=float, default=10.0)
    p.add_argument("--r", type=float, default=DEFAULT_RISK_FREE_RATE)
    p.add_argument("--mrp", type=float, default=DEFAULT_MARKET_RISK_PREMIUM)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--tax", type=float, default=None)
    p.add_argument("--cod", type=float, default=None)
    p.add_argument("--cash", type=float, default=None)
    p.add_argument("--debt", type=float, default=None)
    p.add_argument("--shares", type=float, default=None)
    p.add_argument("--excel", default=None)
    a = p.parse_args()
    result = run(
        a.ticker,
        a.source,
        a.years,
        a.terminal,
        a.g,
        a.multiple,
        a.rf,
        a.mrp,
        a.beta,
        a.tax,
        a.cod,
        a.cash,
        a.debt,
        a.shares,
        a.excel,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
