from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore

from .assumption_builder import _hist_n
from .utils import clamp


def _to_num(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=float)
    return pd.to_numeric(pd.Series(s), errors="coerce")


def arima_forecast(series: pd.Series,
                   horizon: int = 7) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
    """Return point forecast and (low, high) arrays for the next `horizon` steps.

    Falls back to last value if ARIMA is unavailable or fitting fails.
    """
    s = _to_num(series).dropna()
    if len(s) < 3 or ARIMA is None:
        last = float(s.iloc[-1]) if len(s) else 0.0
        y = np.full(horizon, last, dtype=float)
        lo = y * 0.95
        hi = y * 1.05
        return y, lo, hi
    try:
        model = ARIMA(s.values.astype(float), order=(1, 1, 1))
        res = model.fit()
        fc = res.get_forecast(steps=horizon)
        y = fc.predicted_mean
        ci = fc.conf_int(alpha=0.1)
        lo = ci[:, 0]
        hi = ci[:, 1]
        return y, lo, hi
    except Exception:
        last = float(s.iloc[-1]) if len(s) else 0.0
        y = np.full(horizon, last, dtype=float)
        lo = y * 0.95
        hi = y * 1.05
        return y, lo, hi


def build_paths_arima(inc_df: pd.DataFrame,
                      bal_df: pd.DataFrame,
                      years: int,
                      rev_col: str) -> Dict[str,
                                            list[float]]:
    """Build per‑year paths using ARIMA on revenue and margin series.

    Returns paths and basic CI for revenue (lo/hi arrays as percentages vs last).
    """
    inc5 = _hist_n(inc_df)
    if inc5.empty or rev_col not in inc5.columns:
        return {}
    rev_hist = _to_num(inc5[rev_col]).dropna()
    if rev_hist.empty:
        return {}
    # Forecast absolute revenue, convert to growth path vs last historical
    # value
    last_rev = float(rev_hist.iloc[-1]) or 1e-6
    y, lo, hi = arima_forecast(rev_hist, years)
    # Derive per-year growth from level forecasts (relative to last)
    # Ensure non-negative levels
    y = np.maximum(y, 1e-6)
    levels = np.concatenate([[last_rev], y])
    g = []
    for i in range(1, len(levels)):
        g.append(
            float(
                clamp(
                    levels[i] / max(1e-6, levels[i - 1]) - 1.0, -0.5, 0.5)))

    # Margins from simple moving average (could also ARIMA if desired)
    def ma(col: str, lo_b: float, hi_b: float, default: float) -> list[float]:
        s = _to_num(inc5.get(col)).dropna()
        if len(s) == 0:
            val = default
        else:
            val = float(clamp(s.tail(3).mean(), lo_b, hi_b))
        return [val] * years

    gm_path = ma("grossProfit", 0.0, 0.9, 0.4)
    # Convert GP to margin by dividing later; keep as margin directly if
    # already margin-based
    if "grossProfit" in inc5.columns:
        # estimate historical margin
        s_gp = _to_num(inc5.get("grossProfit")).dropna()
        s_m = (s_gp / rev_hist).replace([np.inf, -np.inf], np.nan).dropna()
        gm = float(clamp(s_m.tail(3).mean() if len(s_m) else 0.4, 0.0, 0.9))
        gm_path = [gm] * years
    em_path = ma("operatingIncome", -0.1, 0.6, 0.15)
    # Convert operating income to margin
    s_ebit = _to_num(inc5.get("operatingIncome")).dropna()
    s_em = (s_ebit / rev_hist).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s_em):
        em_val = float(clamp(s_em.tail(3).mean(), -0.1, 0.6))
        em_path = [em_val] * years

    return {
        "rev_growth_path": g,
        "gross_margin_path": gm_path,
        "ebit_margin_path": em_path,
        # da/capex/nwc left to other ML/trend modules for now
    }
