"""Market regime detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import sanitize_dt_index


def _naive_utc_ts(x):
    """Convert any datetime-like object to tz-naive UTC Timestamp.
    
    Safe to call on None, strings, Timestamps (tz-aware or tz-naive).
    Returns None if input is None, otherwise a tz-naive Timestamp.
    """
    if x is None:
        return None
    t = pd.to_datetime(x, utc=True)
    if hasattr(t, 'tz_localize'):
        return t.tz_localize(None)
    return t

try:  # Optional dependency
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GaussianHMM = None  # type: ignore


@dataclass
class RegimeDetectionConfig:
    kmeans_clusters: int = 4
    random_state: int = 42
    min_history: int = 60


def detect_regime(
    df_prices: pd.DataFrame,
    df_macro: Optional[pd.DataFrame] = None,
    config: Optional[RegimeDetectionConfig] = None,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Detect market regimes using price and macro features.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Price history with at least a ``Close`` column indexed by date.
    df_macro : pd.DataFrame, optional
        Macro indicators (e.g., VIX, yield curve data) indexed by date.
    config : RegimeDetectionConfig, optional
        Configuration overrides for detection.
    start_ts : pd.Timestamp, optional
        Start timestamp for filtering (will be sanitized to tz-naive UTC).
    end_ts : pd.Timestamp, optional
        End timestamp for filtering (will be sanitized to tz-naive UTC).

    Returns
    -------
    pd.Series
        Regime labels {0..k-1} aligned to the available history.
    """

    cfg = config or RegimeDetectionConfig()

    if df_prices is None or df_prices.empty:
        return pd.Series(dtype=int)

    # Sanitize prices to tz-naive UTC BEFORE any filtering
    prices = _sanitize_price_frame(df_prices)
    
    # Sanitize start/end timestamps to tz-naive UTC
    start_ts = _naive_utc_ts(start_ts)
    end_ts = _naive_utc_ts(end_ts)
    
    # Now safe to filter with tz-naive ↔ tz-naive comparison
    if start_ts is not None:
        prices = prices.loc[prices.index >= start_ts]
    if end_ts is not None:
        prices = prices.loc[prices.index <= end_ts]
    
    if prices.empty:
        return pd.Series(dtype=int)
    
    macro = _sanitize_macro_frame(df_macro, prices.index)

    feature_frame = build_regime_features(prices, macro)
    if feature_frame.empty or len(feature_frame) < cfg.min_history:
        return pd.Series(dtype=int)

    labels = kmeans_k4_labels(feature_frame, cfg.kmeans_clusters, cfg.random_state)
    return pd.Series(labels, index=feature_frame.index, name="regime")


def build_regime_features(
    prices: pd.DataFrame,
    macro: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create regime features from price and macro inputs."""

    features = pd.DataFrame(index=prices.index)
    close = prices.get("Close")
    if close is None:
        close = prices.iloc[:, 0]

    log_returns = np.log(close).diff()
    features["ret_5"] = close.pct_change(5)
    features["ret_21"] = close.pct_change(21)
    features["ret_63"] = close.pct_change(63)
    features["vol_20"] = log_returns.rolling(20).std() * np.sqrt(252)
    features["vol_63"] = log_returns.rolling(63).std() * np.sqrt(252)

    if macro is not None and not macro.empty:
        macro_aligned = macro.reindex(features.index).ffill().bfill()
        if "VIX" in macro_aligned.columns:
            features["vix_level"] = macro_aligned["VIX"]
            features["vix_slope"] = macro_aligned["VIX"].diff(5)
        if {"DGS10", "DGS2"}.issubset(macro_aligned.columns):
            features["yield_curve"] = macro_aligned["DGS10"] - macro_aligned["DGS2"]
        if "PMI" in macro_aligned.columns:
            features["pmi"] = macro_aligned["PMI"]
        if "CREDIT_SPREAD" in macro_aligned.columns:
            features["credit_spread"] = macro_aligned["CREDIT_SPREAD"]

    return features.dropna()


def kmeans_k4_labels(
    feature_frame: pd.DataFrame,
    clusters: int = 4,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster observations into regimes using KMeans with HMM fallback."""

    if feature_frame.empty:
        return np.array([], dtype=int)

    scaler = StandardScaler()
    data = scaler.fit_transform(feature_frame.values)

    try:
        km = KMeans(n_clusters=clusters, random_state=random_state, n_init=10)
        return km.fit_predict(data)
    except Exception:  # pragma: no cover - fallback path
        if GaussianHMM is not None and len(data) > clusters:
            try:
                hmm = GaussianHMM(
                    n_components=clusters,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=random_state,
                )
                hmm.fit(data)
                return hmm.predict(data)
            except Exception:
                pass
        return np.zeros(len(feature_frame), dtype=int)


def _sanitize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize price data to tz-naive UTC."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = sanitize_dt_index(out)  # Make tz-naive UTC
    out = out.sort_index().ffill()
    return out


def _sanitize_macro_frame(
    df: Optional[pd.DataFrame],
    index: pd.Index,
) -> Optional[pd.DataFrame]:
    """Normalize macro data to tz-naive UTC and align to index."""
    if df is None or df.empty:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = sanitize_dt_index(out)  # Make tz-naive UTC
    out = out.sort_index().reindex(index.union(out.index)).interpolate().ffill().bfill()
    rename_map = {
        "Close": "VIX",
        "Adj Close": "VIX",
        "VIXCLS": "VIX",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out
