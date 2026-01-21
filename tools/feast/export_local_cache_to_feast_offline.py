#!/usr/bin/env python3
"""Export existing prep_families parquet caches into Feast-friendly offline parquet.

Why this exists
---------------
Your current cache layout is many parquet files per symbol/horizon/family/split.
Feast's built-in FileSource wants a *single* file (or stable path) per FeatureView.

This exporter consolidates:
  data/local_cache/{symbol}_h{h}/{symbol}_h{h}_{family}_{split}_features.parquet
into:
  data/feast_offline/h{h}/{family}.parquet

The output contains columns:
  - symbol (str)         : entity join key
  - split (str)          : entity join key (train/valid)
  - event_timestamp (ts) : Feast timestamp
  - <feature columns>

Usage
-----
  python tools/feast/export_local_cache_to_feast_offline.py --horizon 63 --families quantile_forecast calibration

Notes
-----
- This does not regenerate features. It only exports what exists.
- You can pair it with tools/feast/validate_feast_offline.py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


CACHE_FILE_RE = re.compile(
    r"^(?P<symbol>[a-z0-9]+)_h(?P<horizon>\d+)_(?P<family>[a-z0-9_]+)_(?P<split>train|valid)_features\.parquet$"
)


def _discover_families(cache_root: Path, horizon: int) -> List[str]:
    """Discover all family names present in the cache root for a given horizon."""
    horizon = int(horizon)
    families = set()
    for sym_dir in sorted(cache_root.glob(f"*_h{horizon}")):
        if not sym_dir.is_dir():
            continue
        for path in sym_dir.glob(f"*_h{horizon}_*_*_features.parquet"):
            m = CACHE_FILE_RE.match(path.name)
            if not m:
                continue
            families.add(m.group("family"))
    return sorted(families)


def _iter_cache_feature_files(cache_root: Path, horizon: int, families: List[str]) -> Iterable[Path]:
    horizon = int(horizon)
    allowed = set(families)
    for sym_dir in sorted(cache_root.glob(f"*_h{horizon}")):
        if not sym_dir.is_dir():
            continue
        for path in sorted(sym_dir.glob(f"*_h{horizon}_*_train_features.parquet")):
            m = CACHE_FILE_RE.match(path.name)
            if not m:
                continue
            if m.group("family") in allowed:
                yield path
        for path in sorted(sym_dir.glob(f"*_h{horizon}_*_valid_features.parquet")):
            m = CACHE_FILE_RE.match(path.name)
            if not m:
                continue
            if m.group("family") in allowed:
                yield path


def _read_one(path: Path) -> pd.DataFrame:
    m = CACHE_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected cache filename: {path.name}")

    df = pd.read_parquet(path)

    # Accept either explicit date col or index-based date.
    # Normalize to tz-naive timestamps so we can safely concatenate/sort across
    # mixed sources (some upstream caches can be tz-aware).
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        ts = pd.to_datetime(df.index, errors="coerce", utc=True)
        # When index conversion yields a DatetimeIndex, convert to a Series-like
        # object for insertion.
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)

    out = df.copy()
    out.insert(0, "event_timestamp", ts)
    if "date" in out.columns:
        out = out.drop(columns=["date"])

    out.insert(0, "split", m.group("split"))
    out.insert(0, "symbol", m.group("symbol").upper())

    # Ensure unique columns
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="last")]

    return out


def export(cache_root: Path, out_root: Path, horizon: int, families: List[str]) -> None:
    out_root = out_root / f"h{int(horizon)}"
    out_root.mkdir(parents=True, exist_ok=True)

    if len(families) == 1 and str(families[0]).lower() == "all":
        families = _discover_families(cache_root=cache_root, horizon=horizon)
        print(f"[export] discovered families={len(families)}")

    by_family = {f: [] for f in families}

    files = list(_iter_cache_feature_files(cache_root, horizon=horizon, families=families))
    if not files:
        raise SystemExit(
            f"No cache feature files found under {cache_root} for h{horizon} and families={families}."
        )

    for path in files:
        m = CACHE_FILE_RE.match(path.name)
        if not m:
            continue
        fam = m.group("family")
        by_family[fam].append(_read_one(path))

    for fam, frames in by_family.items():
        if not frames:
            print(f"[export] skip {fam}: no input frames")
            continue
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined = combined.sort_values(["symbol", "split", "event_timestamp"]).reset_index(drop=True)

        out_path = out_root / f"{fam}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"[export] wrote {out_path} rows={len(combined)} cols={combined.shape[1]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=Path("cache/features"))
    p.add_argument("--out-root", type=Path, default=Path("cache/feast_offline"))
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument(
        "--families",
        nargs="+",
        required=True,
        help="One or more family names, or 'all' to export every family found",
    )
    args = p.parse_args()

    export(cache_root=args.cache_root, out_root=args.out_root, horizon=args.horizon, families=args.families)


if __name__ == "__main__":
    main()
