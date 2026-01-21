from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .assumption_builder import _hist_n
from .utils import clamp


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _trend_series(
        y: pd.Series,
        horizon: int,
        lo: float | None = None,
        hi: float | None = None) -> list[float]:
    s = _to_num(y).dropna()
    if len(s) < 2:
        return [float(s.iloc[-1]) if len(s) else 0.0] * horizon
    x = np.arange(len(s), dtype=float)
    yy = s.values.astype(float)
    try:
        b, a = np.polyfit(x, yy, 1)
        out = [float(a + b * (len(s) + i)) for i in range(1, horizon + 1)]
    except Exception:
        out = [float(s.iloc[-1])] * horizon
    if lo is not None or hi is not None:
        out = [
            float(
                clamp(
                    v,
                    lo if lo is not None else v,
                    hi if hi is not None else v)) for v in out]
    return out


def _trend_ratio(
        num: pd.Series,
        den: pd.Series,
        horizon: int,
        lo=0.0,
        hi=0.9) -> list[float]:
    s = (_to_num(num) / _to_num(den)
         ).replace([np.inf, -np.inf], np.nan).dropna()
    return _trend_series(s, horizon, lo, hi)


def build_driver_paths(inc_df: pd.DataFrame,
                       bal_df: pd.DataFrame,
                       cf_df: pd.DataFrame,
                       years: int,
                       rev_col: str) -> Dict[str,
                                             list[float]]:
    inc5 = _hist_n(inc_df)
    bal5 = _hist_n(bal_df)
    cf5 = _hist_n(cf_df)

    rev = _to_num(inc5.get(rev_col, pd.Series(dtype=float)))
    # Growth path from log trend -> annual growth
    s = rev.replace(0, np.nan).dropna()
    if len(s) >= 2:
        x = np.arange(len(s), dtype=float)
        y = np.log(s.values.astype(float))
        try:
            b, _ = np.polyfit(x, y, 1)
            g = float(np.exp(b) - 1.0)
        except Exception:
            g = 0.05
    else:
        g = 0.05
    rev_growth_path = [float(clamp(g, -0.2, 0.5))] * years

    gp = _to_num(inc5.get("grossProfit", pd.Series(dtype=float)))
    ebit = _to_num(inc5.get("operatingIncome", pd.Series(dtype=float)))
    da = _to_num(
        inc5.get(
            "depreciationAndAmortization",
            pd.Series(
                dtype=float)))
    capex = _to_num(
        cf5.get(
            "capitalExpenditures",
            pd.Series(
                dtype=float))).abs()

    da_path = _trend_ratio(da, rev, years, 0.0, 0.2)
    capex_path = _trend_ratio(capex, rev, years, 0.0, 0.2)
    gm_path = _trend_ratio(gp, rev, years, 0.0, 0.9)
    em_path = _trend_ratio(ebit, rev, years, -0.1, 0.6)

    if {"totalCurrentAssets", "totalCurrentLiabilities"}.issubset(
            bal5.columns):
        nwc = _to_num(bal5["totalCurrentAssets"]) - \
            _to_num(bal5["totalCurrentLiabilities"])
        nwc_path = _trend_ratio(nwc, rev, years, -0.1, 0.3)
    else:
        nwc_path = [0.03] * years

    return {
        "rev_growth_path": rev_growth_path,
        "gross_margin_path": gm_path,
        "ebit_margin_path": em_path,
        "da_to_rev_path": da_path,
        "capex_to_rev_path": capex_path,
        "nwc_to_rev_path": nwc_path,
    }
