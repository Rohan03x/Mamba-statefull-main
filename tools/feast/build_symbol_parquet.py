#!/usr/bin/env python3
"""Build a single merged parquet per symbol using Feast retrieval.

What this does
--------------
- Uses Feast as the *join/selection layer* to fetch features from multiple families.
- Writes ONE parquet file containing a feature matrix for a given symbol/date range.

Important
---------
Feast does not magically "generate" model features from raw sources by itself.
Your generation still happens in `prep_families` / `build_panel`. Feast then helps:
- standardize retrieval
- enforce point-in-time joins
- produce a single merged dataset reliably

Usage
-----
python tools/feast/build_symbol_parquet.py \
  --symbol AAPL \
  --horizon 63 \
    --splits train valid \
  --start 2024-01-01 \
  --end 2024-12-31 \
    --out cache/features/AAPL_h63_merged.parquet

Notes
-----
- This PoC uses the Feast FeatureService `phase2_base_v1` (quantile_forecast, calibration, online_learning).
- Extend by adding more feature views and feature services in feature_repo/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from feast import FeatureStore


def _infer_timestamps_from_offline(
    horizon: int,
    family: str,
    symbol: str,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    path = Path("data") / "feast_offline" / f"h{int(horizon)}" / f"{family}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing offline parquet: {path}")

    df = pd.read_parquet(path, columns=["symbol", "split", "event_timestamp"])
    df = df[(df["symbol"] == symbol) & (df["split"] == split)]
    if df.empty:
        raise ValueError(f"No rows for {symbol} split={split} in {path}")

    ts = pd.to_datetime(df["event_timestamp"], utc=True).dt.tz_convert(None)
    ts = ts[(ts >= start) & (ts <= end)]
    ts = pd.DatetimeIndex(ts.dropna().drop_duplicates().sort_values())
    if len(ts) == 0:
        raise ValueError(f"No timestamps in range {start.date()}..{end.date()} for {symbol} {split}")
    return ts


def build_symbol_parquet(
    *,
    repo: str,
    symbol: str,
    horizon: int,
    splits: Sequence[str],
    start: str,
    end: str,
    service: str,
    timestamp_family: str,
) -> pd.DataFrame:
    symbol = str(symbol).upper()
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    store = FeatureStore(repo_path=repo)
    service_obj = store.get_feature_service(service)

    frames: List[pd.DataFrame] = []
    for split in splits:
        split_norm = str(split).strip().lower()
        if split_norm not in {"train", "valid"}:
            raise ValueError(f"Invalid split '{split}'. Expected train|valid")

        ts = _infer_timestamps_from_offline(
            horizon=horizon,
            family=timestamp_family,
            symbol=symbol,
            split=split_norm,
            start=start_ts,
            end=end_ts,
        )

        entity_df = pd.DataFrame(
            {
                "symbol": [symbol] * len(ts),
                "split": [split_norm] * len(ts),
                "event_timestamp": ts,
            }
        )

        df = store.get_historical_features(entity_df=entity_df, features=service_obj).to_df()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=True)
    if "event_timestamp" in out.columns:
        # If train/valid ranges overlap, we can end up with duplicate timestamps.
        # Downstream (Stage-B unified panel) expects a unique datetime index.
        if "split" in out.columns:
            split_rank = out["split"].map({"train": 0, "valid": 1}).fillna(0).astype(int)
            out = out.assign(__split_rank=split_rank)
            out = out.sort_values(["event_timestamp", "__split_rank"])
            out = out.drop_duplicates(subset=["event_timestamp"], keep="last")
            out = out.drop(columns=["__split_rank"]) 
        else:
            out = out.sort_values(["event_timestamp"]).drop_duplicates(subset=["event_timestamp"], keep="last")
        out = out.reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="feature_repo/feature_repo", help="Path to Feast repo")
    p.add_argument("--symbol", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="One or more splits to include (train and/or valid).",
    )
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--service",
        default="phase2_base_v1",
        help="Feast FeatureService name to fetch",
    )
    p.add_argument(
        "--timestamp-family",
        default="quantile_forecast",
        help="Which family offline parquet to use as the timestamp backbone",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    splits = [str(s).strip().lower() for s in (args.splits or []) if str(s).strip()]
    df = build_symbol_parquet(
        repo=args.repo,
        symbol=args.symbol,
        horizon=args.horizon,
        splits=splits,
        start=args.start,
        end=args.end,
        service=args.service,
        timestamp_family=args.timestamp_family,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[build] wrote {args.out} rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()
