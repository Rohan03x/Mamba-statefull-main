"""Macroeconomic TST HF generator wrapper.

This generator now delegates to ``src.dcf_lab.modules.macro_tst_hf`` so the
heavy API fetching + feature construction happens in one place with proper
caching.  prep_families can therefore reuse the warmed cache on subsequent
Stage-A passes instead of recomputing per split.

Contract:
    build(symbol, horizon, start, end, out_path, raw_source_cfg, compute_cfg)
    - Writes a parquet aligned to [start, end) with ``date``/``score``/``conf``
      plus every raw feature emitted by the module (Stage A consumes the full
      feature set via *_features.parquet caches).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:  # pragma: no cover - import side effects handled upstream
    from src.dcf_lab.modules.macro_tst_hf import MacroTSTHF
except ImportError as exc:  # pragma: no cover - fail fast for missing deps
    raise ImportError(
        "macro_tst_hf HF generator requires src.dcf_lab.modules.macro_tst_hf"
    ) from exc

LOGGER = logging.getLogger(__name__)

_SUPPORTED_MACRO_KWARGS = {
    "model_id",
    "revision",
    "context_length",
    "zscore_window",
    "min_periods",
    "min_windows",
    "finetune_epochs",
    "lr",
    "lags",
    "d_model_scale",
    "encoder_layers",
    "dropout",
    "min_coverage",
    "confidence_scale",
    "use_gpu",
    "fallback_beta",
}


def _coerce_timestamp(value: str, *, label: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid {label} date for macro_tst_hf: {value}")
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _build_macro_module(compute_cfg: Optional[Dict[str, Any]]) -> MacroTSTHF:
    cfg = compute_cfg or {}
    kwargs: Dict[str, Any] = {}
    for key in _SUPPORTED_MACRO_KWARGS:
        if key in cfg:
            kwargs[key] = cfg[key]
    # Backwards compatibility: allow registry "model" key.
    if "model_id" not in kwargs and "model" in cfg:
        kwargs["model_id"] = cfg["model"]
    return MacroTSTHF(**kwargs)


def _prepare_signal_frame(
    df: Optional[pd.DataFrame],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    if df is None or df.empty:
        return _create_empty_signal(start_dt, end_dt)
    working = df.copy()
    if isinstance(working.index, pd.DatetimeIndex):
        index = working.index
    elif "date" in working.columns:
        index = pd.to_datetime(working["date"], errors="coerce")
        working = working.drop(columns=["date"])
    else:
        # Be tolerant of module outputs that use a non-DatetimeIndex (e.g. object
        # Index of date strings/Timestamps). If coercion fails entirely, treat it
        # as a contract violation.
        coerced = pd.to_datetime(working.index, errors="coerce")
        if getattr(coerced, "isna", None) is not None and bool(coerced.isna().all()):
            raise ValueError("macro_tst_hf module output missing datetime index")
        index = coerced

    index = pd.to_datetime(index, errors="coerce")
    if getattr(index, "isna", None) is not None and bool(index.isna().all()):
        raise ValueError("macro_tst_hf module output missing datetime index")
    index = index.tz_localize(None)

    # Drop any rows whose index could not be parsed.
    if getattr(index, "isna", None) is not None and bool(index.isna().any()):
        keep_mask = ~index.isna()
        working = working.loc[keep_mask].copy()
        index = index[keep_mask]
    mask = (index >= start_dt) & (index < end_dt)
    working = working.loc[mask].copy()
    if working.empty:
        return _create_empty_signal(start_dt, end_dt)
    working.index = index[mask]
    working = working.sort_index()
    # Reset index and ensure the date column exists (handles both named and unnamed indices)
    working = working.reset_index()
    # Rename whatever the index column is to 'date'
    if 'index' in working.columns:
        working = working.rename(columns={'index': 'date'})
    elif 'date' not in working.columns:
        # If index had a different name, rename it
        index_col = working.columns[0]  # reset_index() puts index as first column
        working = working.rename(columns={index_col: 'date'})
    working["date"] = working["date"].dt.normalize()
    if "score" not in working.columns:
        working["score"] = 0.0
    if "conf" not in working.columns:
        working["conf"] = 0.0
    # Ensure date column is the first for readability
    ordered_cols = ["date"] + [c for c in working.columns if c != "date"]
    return working[ordered_cols]


def _create_empty_signal(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D", inclusive="left")
    return pd.DataFrame({
        "date": date_range,
        "score": 0.0,
        "conf": 0.0,
    })


def build(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: Dict[str, Any],  # retained for interface compatibility
    compute_cfg: Dict[str, Any],
) -> None:
    """Build macro_tst_hf features using the shared MacroTSTHF module."""
    del raw_source_cfg  # Not used now that module handles fetching internally
    start_dt = _coerce_timestamp(start, label="start")
    end_dt = _coerce_timestamp(end, label="end")
    if end_dt <= start_dt:
        raise ValueError(
            f"macro_tst_hf requires end > start (got {start_dt} → {end_dt})"
        )
    inclusive_end = end_dt - pd.Timedelta(days=1)
    LOGGER.info(
        "Building macro_tst_hf via MacroTSTHF: %s h%s [%s → %s)",
        symbol,
        horizon,
        start_dt.date(),
        end_dt.date(),
    )
    module = _build_macro_module(compute_cfg)
    if inclusive_end < start_dt:
        signal_df = _create_empty_signal(start_dt, end_dt)
    else:
        module_signal = module.emit_signal(
            symbol=symbol,
            horizon=horizon,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=inclusive_end.strftime("%Y-%m-%d"),
        )
        signal_df = _prepare_signal_frame(module_signal.df, start_dt, end_dt)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(out_file, index=False)
    LOGGER.info(
        "✅ Wrote macro_tst_hf signal: %d rows × %d columns → %s",
        len(signal_df),
        len(signal_df.columns),
        out_file,
    )
