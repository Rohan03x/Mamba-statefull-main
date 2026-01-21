from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, cast

import pandas as pd


def _extract_weights(payload: Mapping[str, Any]) -> Dict[str, float]:
    weights_obj: Any = payload.get("family_normalized_scores") or payload.get("family_weights") or {}
    out: Dict[str, float] = {}
    if isinstance(weights_obj, dict):
        weights_dict = cast(dict[Any, Any], weights_obj)
        for k, v in weights_dict.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
    return out


def _extract_schedule(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    sched: Any = payload.get("schedule")
    if isinstance(sched, list):
        out: list[dict[str, Any]] = []
        for item in cast(list[Any], sched):
            if isinstance(item, dict):
                out.append(cast(dict[str, Any], item))
        return out
    return []


def load_stage_a_weights_asof(path: str | Path, *, asof: str | pd.Timestamp) -> Dict[str, float]:
    """Load Stage-A weights as-of a timestamp.

    Supports both:
      - static weights payload (family_normalized_scores / family_weights)
      - scheduled weights payload (schedule entries with 'asof' and 'weights')

    If both exist, explicit static weights take precedence.
    """

    p = Path(path)
    if not p.exists():
        return {}

    payload: Any = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    payload = cast(dict[str, Any], payload)

    static = _extract_weights(payload)
    if static:
        return static

    schedule = _extract_schedule(payload)
    if not schedule:
        return {}

    asof_ts = pd.Timestamp(asof).normalize()
    best: Optional[Dict[str, float]] = None
    best_asof = None

    for entry in schedule:
        entry_asof: Any = entry.get("asof")
        weights_obj: Any = entry.get("weights")
        if not entry_asof or not isinstance(weights_obj, dict):
            continue
        ts = pd.to_datetime(str(entry_asof), errors="coerce")
        if pd.isna(ts):
            continue
        ts = pd.Timestamp(ts).normalize()
        if ts <= asof_ts and (best_asof is None or ts > best_asof):
            w: Dict[str, float] = {}
            weights_dict = cast(dict[Any, Any], weights_obj)
            for k, v in weights_dict.items():
                if isinstance(v, (int, float)):
                    w[str(k)] = float(v)
            best = w
            best_asof = ts

    return best or {}
