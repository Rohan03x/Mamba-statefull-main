from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .assumption_builder import _hist_n
from .utils import clamp


def _safe_series(x: pd.Series | pd.DataFrame | None) -> pd.Series:
    if x is None:
        return pd.Series([], dtype=float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series([], dtype=float)
        x = x.iloc[:, 0]
    return pd.to_numeric(pd.Series(x), errors="coerce")


def _trend_last(series: pd.Series) -> float | None:
    s = pd.to_numeric(series.dropna(), errors="coerce")
    if len(s) < 2:
        return float(s.iloc[-1]) if len(s) else None
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    try:
        b, a = np.polyfit(x, y, 1)  # y = a + b*x
        return float(a + b * (len(s)))
    except Exception:
        return float(s.iloc[-1])


def _trend_growth(series: pd.Series) -> float | None:
    s = pd.to_numeric(series.dropna(), errors="coerce")
    if len(s) < 2:
        return None
    # log-linear growth: log(y) = a + b*t => growth ~ exp(b) - 1
    x = np.arange(len(s), dtype=float)
    y = np.log(np.clip(s.values.astype(float), 1e-6, None))
    try:
        b, a = np.polyfit(x, y, 1)
        g = np.exp(b) - 1.0
        return float(clamp(g, -0.5, 0.5))
    except Exception:
        return None


def estimate_drivers_ml(
    inc_df: pd.DataFrame,
    bal_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    base: Dict[str, float],
) -> Dict[str, float]:
    """Return drivers estimated by simple trend models (ML-lite).

    - Revenue growth from log-linear regression on revenue history
    - Margins/ratios from linear trend on historical ratios (last-5)
    Falls back to base values when insufficient data.
    """
    inc5 = _hist_n(inc_df)
    bal5 = _hist_n(bal_df)
    cfs5 = _hist_n(cfs_df)

    # Revenue growth
    rev_col = base.get("rev_col", "totalRevenue")
    rev = _safe_series(inc5.get(rev_col))
    g = _trend_growth(rev)

    # Margins and ratios
    gp = _safe_series(inc5.get("grossProfit"))
    ebit = _safe_series(inc5.get("operatingIncome"))
    da = _safe_series(inc5.get("depreciationAndAmortization"))
    capex = _safe_series(cfs5.get("capitalExpenditures"))

    # revenue aligned
    if len(rev) and len(gp):
        gross_margin_hist = (
            gp / rev).replace([np.inf, -np.inf], np.nan).dropna()
        gm_last = _trend_last(gross_margin_hist) if len(
            gross_margin_hist) else None
    else:
        gm_last = None

    if len(rev) and len(ebit):
        ebit_margin_hist = (
            ebit / rev).replace([np.inf, -np.inf], np.nan).dropna()
        em_last = _trend_last(ebit_margin_hist) if len(
            ebit_margin_hist) else None
    else:
        em_last = None

    da_to_rev_hist = (
        da / rev).replace(
        [np.inf, -np.inf],
        np.nan).dropna() if len(rev) and len(da) else pd.Series(
        [])
    capex_to_rev_hist = (
        abs(capex) / rev).replace(
        [np.inf, -np.inf],
        np.nan).dropna() if len(rev) and len(capex) else pd.Series(
        [])

    da_last = _trend_last(da_to_rev_hist) if len(da_to_rev_hist) else None
    capex_last = _trend_last(capex_to_rev_hist) if len(
        capex_to_rev_hist) else None

    # NWC ratio from balance sheet if available
    if set(["totalCurrentAssets", "totalCurrentLiabilities"]).issubset(
            bal5.columns) and rev is not None and len(rev):
        nwc = pd.to_numeric(
            bal5["totalCurrentAssets"],
            errors="coerce") - pd.to_numeric(
            bal5["totalCurrentLiabilities"],
            errors="coerce")
        rev_al = pd.to_numeric(inc5[rev_col], errors="coerce")
        nwc_ratio_hist = (nwc /
                          rev_al).replace([np.inf, -
                                           np.inf], np.nan).dropna()
        nwc_last = _trend_last(nwc_ratio_hist) if len(nwc_ratio_hist) else None
    else:
        nwc_last = None

    out = dict(base)
    if g is not None:
        out["rev_cagr"] = float(g)
    if gm_last is not None:
        out["gross_margin"] = float(clamp(gm_last, 0.0, 0.9))
    if em_last is not None:
        out["ebit_margin"] = float(clamp(em_last, -0.1, 0.6))
    if da_last is not None:
        out["da_to_rev"] = float(clamp(da_last, 0.0, 0.2))
    if capex_last is not None:
        out["capex_to_rev"] = float(clamp(capex_last, 0.0, 0.2))
    if nwc_last is not None:
        out["nwc_to_rev"] = float(clamp(nwc_last, -0.1, 0.3))

    return out
