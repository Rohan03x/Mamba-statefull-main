"""candle_mechanics

Compact daily OHLCV-derived feature family.

Design goals:
- Leak-safe (uses only current and past data; rolling baselines are shifted)
- Scale-free numeric encodings (ATR/Range normalization; ratios/log-returns)
- Compact (~20–60 columns)
- No named chart patterns; features are continuous/proxy flags

Expected input OHLCV schema (lowercase): open, high, low, close, volume
Index should be trading sessions (daily bars).
"""

from __future__ import annotations

from typing import Optional, Union

from datetime import datetime
import numpy as np
import pandas as pd


DateLike = Union[str, datetime, pd.Timestamp]


def _as_timestamp(value: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    hl = (high - low).abs()
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def compute_features(
    ohlcv: pd.DataFrame,
    *,
    add_seasonality: bool = True,
) -> pd.DataFrame:
    """Compute candle mechanics features from a daily OHLCV frame."""

    if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        return pd.DataFrame()

    required = {"open", "high", "low", "close"}
    missing = required.difference(set(map(str.lower, ohlcv.columns)))
    if missing:
        raise ValueError(f"candle_mechanics requires columns {sorted(required)}; missing {sorted(missing)}")

    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()

    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else None

    eps = 1e-12

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_open = open_.shift(1)

    rng = (high - low).abs()
    body = close - open_

    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    tr = _true_range(high, low, prev_close)
    atr_14 = tr.rolling(14, min_periods=1).mean()
    atr_60 = tr.rolling(60, min_periods=1).mean()

    # Baselines used as denominators should be shifted to avoid using today's value
    atr_14_prev = atr_14.shift(1).fillna(atr_14)
    atr_60_prev = atr_60.shift(1).fillna(atr_60)

    # Range-based anatomy (scale-free)
    inv_rng = 1.0 / (rng + eps)
    inv_atr = 1.0 / (atr_14_prev + eps)

    close_pos = ((close - low) * inv_rng).clip(0.0, 1.0) * 2.0 - 1.0

    # Returns / momentum
    safe_close = close.clip(lower=eps)
    logret_1d = np.log(safe_close).diff().fillna(0.0)
    ret_1d = safe_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_5d = safe_close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Volatility measures
    rv_5 = logret_1d.rolling(5, min_periods=1).std(ddof=0).fillna(0.0)
    rv_20 = logret_1d.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)

    # Breakout distances vs prior window extremes (shifted baselines)
    prior_high_20 = high.rolling(20, min_periods=1).max().shift(1)
    prior_low_20 = low.rolling(20, min_periods=1).min().shift(1)
    prior_close_20 = close.rolling(20, min_periods=1).mean().shift(1)

    dist_to_high_20_atr = (close - prior_high_20) * inv_atr
    dist_to_low_20_atr = (close - prior_low_20) * inv_atr
    dist_to_mean_20_atr = (close - prior_close_20) * inv_atr

    # Gap and intraday extension
    gap_atr = (open_ - prev_close) * inv_atr
    close_vs_prev_close_atr = (close - prev_close) * inv_atr
    high_vs_prev_close_atr = (high - prev_close) * inv_atr
    low_vs_prev_close_atr = (low - prev_close) * inv_atr

    # Shape/imbalance encodings
    body_pct_range = (body * inv_rng).clip(-5.0, 5.0)
    upper_wick_pct_range = (upper_wick * inv_rng).clip(0.0, 5.0)
    lower_wick_pct_range = (lower_wick * inv_rng).clip(0.0, 5.0)
    wick_imbalance = ((upper_wick - lower_wick) * inv_rng).clip(-5.0, 5.0)

    body_atr = (body * inv_atr).clip(-10.0, 10.0)
    range_atr = (rng * inv_atr).clip(0.0, 20.0)

    # Simple multi-day state proxies
    trend_3 = logret_1d.rolling(3, min_periods=1).sum().fillna(0.0)
    trend_5 = logret_1d.rolling(5, min_periods=1).sum().fillna(0.0)

    sign = np.sign(logret_1d)
    sign_sum_3 = sign.rolling(3, min_periods=1).sum().fillna(0.0)
    sign_sum_5 = sign.rolling(5, min_periods=1).sum().fillna(0.0)

    # Range z-score vs prior window (shifted baseline to avoid using today's range)
    rng_mean_20 = rng.rolling(20, min_periods=1).mean().shift(1)
    rng_std_20 = rng.rolling(20, min_periods=1).std(ddof=0).shift(1).replace(0.0, np.nan)
    rng_z_20 = ((rng - rng_mean_20) / (rng_std_20 + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)

    body_mean_20 = body.rolling(20, min_periods=1).mean().shift(1)
    body_std_20 = body.rolling(20, min_periods=1).std(ddof=0).shift(1).replace(0.0, np.nan)
    body_z_20 = ((body - body_mean_20) / (body_std_20 + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)

    # Bar relation flags (numeric)
    inside_bar = ((high <= prev_high) & (low >= prev_low)).astype(float)
    outside_bar = ((high >= prev_high) & (low <= prev_low)).astype(float)

    # Direction change proxy
    prev_body = (prev_close - prev_open)
    dir_change = (np.sign(body.fillna(0.0)) != np.sign(prev_body.fillna(0.0))).astype(float)

    # ATR regime
    atr_ratio = (atr_14_prev / (atr_60_prev + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)

    # Volume proxies (all baselines shifted)
    if volume is not None:
        safe_vol = volume.clip(lower=0.0)
        vol_mean_20 = safe_vol.rolling(20, min_periods=1).mean().shift(1)
        vol_ratio_20 = (safe_vol / (vol_mean_20 + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 20.0)
        vol_log_chg = np.log(safe_vol + 1.0).diff().fillna(0.0).clip(-10.0, 10.0)
    else:
        vol_ratio_20 = pd.Series(1.0, index=df.index)
        vol_log_chg = pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "has_data": (~(open_.isna() | high.isna() | low.isna() | close.isna())).astype(float),
            "close_pos": close_pos.fillna(0.0),
            "body_pct_range": body_pct_range.fillna(0.0),
            "upper_wick_pct_range": upper_wick_pct_range.fillna(0.0),
            "lower_wick_pct_range": lower_wick_pct_range.fillna(0.0),
            "wick_imbalance": wick_imbalance.fillna(0.0),
            "body_atr14": body_atr.fillna(0.0),
            "range_atr14": range_atr.fillna(0.0),
            "gap_atr14": gap_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0),
            "close_vs_prev_close_atr14": close_vs_prev_close_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0),
            "high_vs_prev_close_atr14": high_vs_prev_close_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0),
            "low_vs_prev_close_atr14": low_vs_prev_close_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0),
            "logret_1d": logret_1d,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "rv_5": rv_5,
            "rv_20": rv_20,
            "trend_3": trend_3,
            "trend_5": trend_5,
            "sign_sum_3": sign_sum_3,
            "sign_sum_5": sign_sum_5,
            "dist_to_high_20_atr14": dist_to_high_20_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20.0, 20.0),
            "dist_to_low_20_atr14": dist_to_low_20_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20.0, 20.0),
            "dist_to_mean_20_atr14": dist_to_mean_20_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20.0, 20.0),
            "rng_z_20": rng_z_20,
            "body_z_20": body_z_20,
            "inside_bar": inside_bar.fillna(0.0),
            "outside_bar": outside_bar.fillna(0.0),
            "dir_change": dir_change.fillna(0.0),
            "atr_ratio_14_60": atr_ratio,
            "vol_ratio_20": vol_ratio_20,
            "vol_log_chg": vol_log_chg,
        },
        index=df.index,
    )

    if add_seasonality:
        # Keep this tiny: 2 columns for DOW + 2 for month.
        dow = pd.Index(out.index).dayofweek.astype(float)
        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        moy = pd.Index(out.index).month.astype(float)
        out["moy_sin"] = np.sin(2.0 * np.pi * (moy - 1.0) / 12.0)
        out["moy_cos"] = np.cos(2.0 * np.pi * (moy - 1.0) / 12.0)

    # Ensure numeric stability
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out


def fetch(
    symbol: str,
    start: DateLike,
    end: DateLike,
    *,
    price_df: Optional[pd.DataFrame] = None,
    add_seasonality: bool = True,
) -> pd.DataFrame:
    """Compute candle mechanics for symbol within [start, end].

    This family intentionally does not fetch OHLCV itself; callers should pass
    a price frame produced by the pipeline's strict source policy.
    """

    if price_df is None:
        raise ValueError("candle_mechanics.fetch requires price_df (OHLCV) to be provided")

    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)

    features = compute_features(price_df, add_seasonality=add_seasonality)
    if features.empty:
        return features

    # Filter after computation so rolling windows can use earlier history.
    features = features.loc[(features.index >= start_ts) & (features.index <= end_ts)]
    features.attrs = {"provenance": {"symbol": symbol, "family": "candle_mechanics"}}
    return features
