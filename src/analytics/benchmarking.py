from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkFramework:
    """Defines what "the market" means for evaluation.

    Policy defaults:
      - Primary benchmark: SPY (or equivalent total-return index)
      - Secondary benchmarks: optional (equal-weight universe, sector-neutral)
      - Risk-free: used for excess-return Sharpe/Sortino, and optionally excess Jensen alpha.
    """

    primary: str = "SPY"
    secondary: Tuple[str, ...] = ()
    trading_days: int = 252


def _to_series(x: Union[pd.Series, pd.DataFrame], *, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x
    else:
        if x.shape[1] != 1:
            raise ValueError(f"Expected single column for {name}, got shape={x.shape}")
        s = x.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.name = name
    return s


def _daily_rf_from_annual(rf_annual: float, *, trading_days: int) -> float:
    # Convert simple annual rate to an equivalent daily geometric rate.
    if rf_annual <= -1:
        return 0.0
    return float((1.0 + float(rf_annual)) ** (1.0 / float(trading_days)) - 1.0)


def _rolling_beta(rp: pd.Series, rb: pd.Series, window: int) -> pd.Series:
    cov = rp.rolling(window=window, min_periods=max(5, window // 5)).cov(rb)
    var = rb.rolling(window=window, min_periods=max(5, window // 5)).var()
    return (cov / var.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_ir(alpha: pd.Series, window: int, *, trading_days: int) -> pd.Series:
    mu = alpha.rolling(window=window, min_periods=max(5, window // 5)).mean()
    sd = alpha.rolling(window=window, min_periods=max(5, window // 5)).std()
    return (mu / (sd + 1e-12) * np.sqrt(trading_days)).replace([np.inf, -np.inf], np.nan)


def _rolling_sharpe(excess: pd.Series, window: int, *, trading_days: int) -> pd.Series:
    mu = excess.rolling(window=window, min_periods=max(5, window // 5)).mean()
    sd = excess.rolling(window=window, min_periods=max(5, window // 5)).std()
    return (mu / (sd + 1e-12) * np.sqrt(trading_days)).replace([np.inf, -np.inf], np.nan)


def _rolling_sortino(excess: pd.Series, window: int, *, trading_days: int) -> pd.Series:
    def _sortino(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size < 5:
            return np.nan
        downside = x[x < 0.0]
        if downside.size < 2:
            return np.nan
        dd = float(np.sqrt(np.mean(downside * downside)))
        if dd <= 0:
            return np.nan
        return float(np.mean(x) / (dd + 1e-12) * np.sqrt(trading_days))

    return excess.rolling(window=window, min_periods=max(5, window // 5)).apply(_sortino, raw=True)


def _rolling_tracking_error(active: pd.Series, window: int, *, trading_days: int) -> pd.Series:
    sd = active.rolling(window=window, min_periods=max(5, window // 5)).std()
    return (sd * np.sqrt(trading_days)).replace([np.inf, -np.inf], np.nan)


def _rolling_cvar(returns: pd.Series, window: int, *, alpha: float) -> pd.Series:
    a = float(alpha)

    def _cvar(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            return np.nan
        q = float(np.quantile(x, a))
        tail = x[x <= q]
        if tail.size == 0:
            return np.nan
        return float(np.mean(tail))

    return returns.rolling(window=window, min_periods=max(10, window // 5)).apply(_cvar, raw=True)


def _rolling_capture(
    rp: pd.Series,
    rb: pd.Series,
    window: int,
    *,
    direction: str,
) -> pd.Series:
    direction = str(direction).lower().strip()
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")

    # Rolling apply on DataFrames can be shape-fragile across pandas versions;
    # implement capture with an explicit window loop for correctness.
    pr = rp.to_numpy(dtype=float, copy=False)
    br = rb.to_numpy(dtype=float, copy=False)
    out = np.full(pr.shape[0], np.nan, dtype=float)
    min_periods = max(10, window // 5)

    for i in range(pr.shape[0]):
        start = i - window + 1
        if start < 0:
            continue

        pr_w = pr[start : i + 1]
        br_w = br[start : i + 1]

        mask = br_w > 0.0 if direction == "up" else br_w < 0.0
        if int(mask.sum()) < 3:
            continue

        pr_sel = pr_w[mask]
        br_sel = br_w[mask]
        if pr_sel.shape[0] < min_periods or br_sel.shape[0] < min_periods:
            continue

        pr_c = float(np.prod(1.0 + pr_sel) - 1.0)
        br_c = float(np.prod(1.0 + br_sel) - 1.0)
        if abs(br_c) < 1e-12:
            continue
        out[i] = pr_c / br_c

    return pd.Series(out, index=rp.index)


def compute_benchmark_timeseries(
    *,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    windows: Sequence[int] = (20, 63, 126),
    risk_free: Union[float, pd.Series] = 0.0,
    trading_days: int = 252,
    cvar_alpha: float = 0.05,
    target_vol_annual: Optional[float] = None,
    use_excess_alpha: bool = False,
) -> pd.DataFrame:
    """Compute daily benchmarked metrics and rolling aggregates.

    Required core outputs (per user spec):
      - Rolling beta (20/63/126)
      - Jensen alpha (daily + rolling)
      - Information ratio (rolling)
      - Active return + tracking error (rolling)
      - Up/down capture (rolling)
      - Sharpe/Sortino, MaxDD, CVaR, vol-target error

    Notes:
      - Alpha is computed as: alpha_t = R_p,t - beta_t * R_b,t
      - If use_excess_alpha=True and risk_free provided, alpha becomes excess Jensen alpha:
            (R_p - R_f) - beta * (R_b - R_f)
    """

    rp = _to_series(portfolio_returns, name="rp").copy()
    rb = _to_series(benchmark_returns, name="rb").copy()

    # Align / clean.
    df = pd.concat([rp, rb], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    rp = df["rp"].astype(float)
    rb = df["rb"].astype(float)

    if isinstance(risk_free, pd.Series):
        rf = pd.to_numeric(risk_free, errors="coerce").reindex(df.index).fillna(0.0).astype(float)
    else:
        rf = pd.Series(
            _daily_rf_from_annual(float(risk_free), trading_days=trading_days),
            index=df.index,
            name="rf",
            dtype=float,
        )

    out = pd.DataFrame(index=df.index)
    out["rp"] = rp
    out["rb"] = rb
    out["rf"] = rf
    out["active"] = rp - rb
    out["excess_p"] = rp - rf
    out["excess_b"] = rb - rf

    for w in windows:
        w = int(w)
        beta = _rolling_beta(rp, rb, w)
        out[f"beta_{w}"] = beta

        if use_excess_alpha:
            alpha = (rp - rf) - beta * (rb - rf)
        else:
            alpha = rp - beta * rb
        out[f"alpha_{w}"] = alpha
        out[f"alpha_roll_{w}"] = alpha.rolling(window=w, min_periods=max(5, w // 5)).mean()
        out[f"alpha_cum_{w}"] = (1.0 + alpha.fillna(0.0)).cumprod() - 1.0

        out[f"ir_{w}"] = _rolling_ir(alpha, w, trading_days=trading_days)
        out[f"te_{w}"] = _rolling_tracking_error(out["active"], w, trading_days=trading_days)
        out[f"sharpe_{w}"] = _rolling_sharpe(out["excess_p"], w, trading_days=trading_days)
        out[f"sortino_{w}"] = _rolling_sortino(out["excess_p"], w, trading_days=trading_days)

        out[f"up_capture_{w}"] = _rolling_capture(rp, rb, w, direction="up")
        out[f"down_capture_{w}"] = _rolling_capture(rp, rb, w, direction="down")

        out[f"cvar_{w}"] = _rolling_cvar(out["excess_p"], w, alpha=cvar_alpha)

        realized_vol = out["rp"].rolling(window=w, min_periods=max(5, w // 5)).std() * np.sqrt(trading_days)
        out[f"realized_vol_{w}"] = realized_vol
        if target_vol_annual is not None:
            out[f"vol_target_error_{w}"] = realized_vol - float(target_vol_annual)

    # Non-rolling, committee-level stats on the full series.
    eq = (1.0 + rp.fillna(0.0)).cumprod()
    peak = eq.cummax()
    out["drawdown"] = (eq / peak.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    return out


def summarize_benchmark(
    *,
    ts: pd.DataFrame,
    windows: Sequence[int] = (20, 63, 126),
) -> Dict[str, float]:
    """Create a compact summary dict for reporting."""
    out: Dict[str, float] = {}

    rp = ts.get("rp")
    if isinstance(rp, pd.Series) and not rp.empty:
        eq = (1.0 + rp.fillna(0.0)).cumprod()
        peak = eq.cummax()
        dd = (eq / peak.replace(0.0, np.nan) - 1.0).min()
        out["max_drawdown"] = float(dd) if dd == dd else 0.0

    for w in windows:
        w = int(w)
        for k in ("beta", "ir", "te", "sharpe", "sortino", "up_capture", "down_capture"):
            col = f"{k}_{w}"
            if col in ts.columns:
                s = pd.to_numeric(ts[col], errors="coerce").dropna()
                if not s.empty:
                    out[f"{col}_last"] = float(s.iloc[-1])
                    out[f"{col}_mean"] = float(s.mean())

        a = f"alpha_{w}"
        if a in ts.columns:
            s = pd.to_numeric(ts[a], errors="coerce").dropna()
            if not s.empty:
                out[f"alpha_{w}_mean_daily"] = float(s.mean())

    return out
