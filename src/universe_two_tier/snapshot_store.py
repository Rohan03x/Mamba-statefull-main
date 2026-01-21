from __future__ import annotations

# pyright: reportUnknownMemberType=false

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import pandas as pd

from .types import UniverseConstructionConfig


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def snapshot_path(*, cfg: UniverseConstructionConfig, asof: pd.Timestamp) -> Path:
    ts = pd.Timestamp(asof).normalize().strftime("%Y%m%d")
    return Path(cfg.snapshot_dir) / f"universe_snapshot_{ts}_h{int(cfg.ranking.horizon)}.parquet"


def snapshot_meta_path(*, cfg: UniverseConstructionConfig, asof: pd.Timestamp) -> Path:
    ts = pd.Timestamp(asof).normalize().strftime("%Y%m%d")
    return Path(cfg.snapshot_dir) / f"universe_snapshot_{ts}_h{int(cfg.ranking.horizon)}.meta.json"


def write_snapshot(
    *,
    cfg: UniverseConstructionConfig,
    asof: pd.Timestamp,
    frame: pd.DataFrame,
    meta: Dict[str, Any],
    overwrite: bool = True,
) -> Path:
    out = snapshot_path(cfg=cfg, asof=asof)
    if out.exists() and not overwrite:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    meta_out = snapshot_meta_path(cfg=cfg, asof=asof)
    meta_payload: Dict[str, Any] = {
        "asof": pd.Timestamp(asof).normalize().strftime("%Y-%m-%d"),
        "horizon": int(cfg.ranking.horizon),
        "eligibility": asdict(cfg.eligibility),
        "ranking": asdict(cfg.ranking),
        "meta": meta or {},
    }
    _atomic_write_text(meta_out, json.dumps(meta_payload, indent=2, sort_keys=True, default=str))
    return out


def read_snapshot(path: str | Path) -> pd.DataFrame:
    _read_parquet = cast(Callable[..., pd.DataFrame], pd.read_parquet)
    return _read_parquet(Path(path))


def try_read_snapshot(*, cfg: UniverseConstructionConfig, asof: pd.Timestamp) -> Optional[pd.DataFrame]:
    p = snapshot_path(cfg=cfg, asof=asof)
    if not p.exists():
        return None
    try:
        _read_parquet = cast(Callable[..., pd.DataFrame], pd.read_parquet)
        return _read_parquet(p)
    except Exception:
        return None
