"""Shared helpers for HF block generators."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.features.aggregator_panel import build_panel

LOGGER = logging.getLogger(__name__)


def _coerce_timestamp(value: str, *, label: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid {label} date '{value}' for HF block generator")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _resolve_cache_dir(
    out_path: str,
    raw_source_cfg: Optional[Dict[str, Any]],
    compute_cfg: Optional[Dict[str, Any]],
) -> Path:
    cfg_dir = (raw_source_cfg or {}).get("cache_dir") or (compute_cfg or {}).get("cache_dir")
    if cfg_dir:
        cache_root = Path(str(cfg_dir)).expanduser()
    else:
        cache_root = Path(out_path).parent
    return cache_root


def _resolve_view_mode(
    raw_source_cfg: Optional[Dict[str, Any]],
    compute_cfg: Optional[Dict[str, Any]],
) -> str:
    view = (compute_cfg or {}).get("view") or (raw_source_cfg or {}).get("view")
    view_mode = str(view or "both").lower()
    return view_mode


def _extract_block_frame(panel: Optional[pd.DataFrame], block_name: str) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["score", "conf"])
    score_col = None
    conf_col = None
    for candidate in (f"{block_name}_score", "score", "hf_score"):
        if candidate in panel.columns:
            score_col = candidate
            break
    for candidate in (f"{block_name}_conf", "conf", "hf_conf"):
        if candidate in panel.columns:
            conf_col = candidate
            break
    if score_col is None or conf_col is None:
        raise KeyError(
            f"HF block '{block_name}' missing score/conf columns in build_panel output"
        )
    working = panel[[score_col, conf_col]].copy()
    index = pd.to_datetime(working.index)
    index = index.tz_localize(None)
    working.index = index
    working.columns = ["score", "conf"]
    working = working.replace([pd.NA, float("inf"), float("-inf")], 0.0).sort_index()
    return working


def _empty_payload(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if end_ts < start_ts:
        return pd.DataFrame(columns=["date", "score", "conf"])
    date_index = pd.date_range(start=start_ts, end=end_ts, freq="B")
    if date_index.empty:
        date_index = pd.DatetimeIndex([start_ts])
    frame = pd.DataFrame({
        "date": date_index,
        "score": 0.0,
        "conf": 0.0,
    })
    frame["date"] = frame["date"].dt.normalize()
    return frame


def _format_payload(
    block_frame: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    if block_frame.empty:
        return pd.DataFrame(columns=["date", "score", "conf"])
    mask = (block_frame.index >= start_ts) & (block_frame.index <= end_ts)
    sliced = block_frame.loc[mask]
    if sliced.empty:
        return pd.DataFrame(columns=["date", "score", "conf"])
    sliced = sliced.replace([pd.NA], 0.0).fillna(0.0)
    payload = sliced.reset_index().rename(columns={"index": "date"})
    payload["date"] = pd.to_datetime(payload["date"]).dt.normalize()
    payload = payload.sort_values("date").drop_duplicates(subset="date", keep="last")
    payload["score"] = payload["score"].astype(float)
    payload["conf"] = payload["conf"].clip(lower=0.0).astype(float)
    return payload


def prepare_block_payload(
    block_name: str,
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    out_path: str,
    raw_source_cfg: Optional[Dict[str, Any]] = None,
    compute_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Path]:
    start_ts = _coerce_timestamp(start, label="start")
    end_ts = _coerce_timestamp(end, label="end")
    if end_ts < start_ts:
        raise ValueError(
            f"HF block '{block_name}' requires end >= start (got {start_ts.date()} > {end_ts.date()})"
        )
    cache_dir = _resolve_cache_dir(out_path, raw_source_cfg, compute_cfg)
    view_mode = _resolve_view_mode(raw_source_cfg, compute_cfg)
    try:
        panel = build_panel(
            symbol=symbol,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            families=[block_name],
            cache_dir=cache_dir,
            horizon=horizon,
            stage="B",
            view=view_mode,
        )
    except Exception as exc:
        LOGGER.warning(
            "build_panel failed for %s (symbol=%s, horizon=%s): %s",
            block_name,
            symbol,
            horizon,
            exc,
        )
        panel = None
    try:
        block_frame = _extract_block_frame(panel, block_name)
        payload = _format_payload(block_frame, start_ts, end_ts)
    except Exception as exc:
        LOGGER.warning(
            "%s: failed to extract block frame – falling back to zeros (%s)",
            block_name,
            exc,
        )
        payload = pd.DataFrame()
    if payload.empty:
        payload = _empty_payload(start_ts, end_ts)
    return payload, cache_dir


def should_update_block_cache(
    out_file: Path,
    raw_source_cfg: Optional[Dict[str, Any]] = None,
    compute_cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    for cfg in (raw_source_cfg, compute_cfg):
        if cfg and cfg.get("materialize_block_cache") is False:
            return False
    stem = out_file.stem.lower()
    return "_w" not in stem


def update_block_cache(
    payload: pd.DataFrame,
    cache_dir: Path,
    symbol: str,
    horizon: int,
    block_name: str,
) -> None:
    if payload.empty:
        return
    dest = (
        Path(cache_dir)
        / "block_hf"
        / symbol.upper()
        / str(horizon)
        / f"{block_name}.parquet"
    )
    block_df = payload.rename(
        columns={
            "score": f"{block_name}_score",
            "conf": f"{block_name}_conf",
        }
    ).copy()
    block_df = block_df[["date", f"{block_name}_score", f"{block_name}_conf"]]
    block_df["date"] = pd.to_datetime(block_df["date"]).dt.normalize()
    if dest.exists():
        try:
            existing = pd.read_parquet(dest)
            block_df = pd.concat([existing, block_df], ignore_index=True)
        except Exception as exc:
            LOGGER.warning("%s: failed to merge block cache %s – rewriting (%s)", block_name, dest, exc)
    block_df = block_df.sort_values("date").drop_duplicates(subset="date", keep="last")
    dest.parent.mkdir(parents=True, exist_ok=True)
    block_df.to_parquet(dest, index=False)
    LOGGER.info(
        "%s: updated block cache %s with %d rows",
        block_name,
        dest,
        len(block_df),
    )


__all__ = [
    "prepare_block_payload",
    "should_update_block_cache",
    "update_block_cache",
]
