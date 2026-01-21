#!/usr/bin/env python3
"""Audit unified TrackC cache panels.

Reads cache/features/*_trackc.parquet and reports:
- rows/cols
- start/end dates
- total NaNs and percent NaN
- count of all-NaN columns
- top worst columns by NaN%

Outputs:
- artifacts/trackc_audit/trackc_audit_summary.csv
- artifacts/trackc_audit/trackc_audit_worst_columns.csv

This is intentionally read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR_DEFAULT = REPO_ROOT / "cache" / "features"
OUT_DIR_DEFAULT = REPO_ROOT / "artifacts" / "trackc_audit"


@dataclass(frozen=True)
class PanelStats:
    symbol: str
    horizon: int
    rows: int
    cols: int
    start: str
    end: str
    total_cells: int
    total_nans: int
    nan_pct: float
    any_nan_cols: int
    all_nan_cols: int


def _iter_trackc_files(cache_dir: Path) -> Iterable[Path]:
    yield from sorted(cache_dir.glob("*_trackc.parquet"))


def _parse_symbol_horizon(path: Path) -> tuple[str, int]:
    # Expected: {SYMBOL}_h{H}_trackc.parquet
    name = path.name
    if "_h" not in name:
        return name.replace("_trackc.parquet", ""), -1
    sym, rest = name.split("_h", 1)
    h_str = rest.split("_", 1)[0]
    try:
        horizon = int(h_str)
    except Exception:
        horizon = -1
    return sym, horizon


def _panel_stats(df: pd.DataFrame, symbol: str, horizon: int) -> tuple[PanelStats, pd.DataFrame]:
    frame = df
    if "date" in frame.columns:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date").sort_index()
    elif not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        except Exception:
            pass

    # Stats should be about features, not the date column.
    rows = int(frame.shape[0])
    cols = int(frame.shape[1])
    start = str(frame.index.min().date()) if rows and isinstance(frame.index, pd.DatetimeIndex) else ""
    end = str(frame.index.max().date()) if rows and isinstance(frame.index, pd.DatetimeIndex) else ""

    total_cells = rows * cols
    nan_by_col = frame.isna().sum(axis=0)
    total_nans = int(nan_by_col.sum())
    nan_pct = float(total_nans / total_cells) if total_cells else 0.0
    any_nan_cols = int((nan_by_col > 0).sum())
    all_nan_cols = int((nan_by_col >= rows).sum()) if rows else int((nan_by_col > 0).sum())

    worst = (
        pd.DataFrame(
            {
                "symbol": symbol,
                "horizon": horizon,
                "column": nan_by_col.index.astype(str),
                "nan_count": nan_by_col.values.astype(int),
                "nan_pct": (nan_by_col.values / max(1, rows)).astype(float),
            }
        )
        .sort_values(["nan_pct", "nan_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    stats = PanelStats(
        symbol=symbol,
        horizon=horizon,
        rows=rows,
        cols=cols,
        start=start,
        end=end,
        total_cells=total_cells,
        total_nans=total_nans,
        nan_pct=nan_pct,
        any_nan_cols=any_nan_cols,
        all_nan_cols=all_nan_cols,
    )
    return stats, worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--top", type=int, default=25, help="Top N worst columns per symbol")
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    worst_rows: list[pd.DataFrame] = []

    files = list(_iter_trackc_files(cache_dir))
    if not files:
        print(f"No TrackC cache files found in {cache_dir}")
        return 2

    for path in files:
        symbol, horizon = _parse_symbol_horizon(path)
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            summaries.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "path": str(path),
                    "error": str(exc),
                }
            )
            continue

        stats, worst = _panel_stats(df, symbol, horizon)
        summaries.append(
            {
                **stats.__dict__,
                "path": str(path),
                "error": "",
            }
        )
        worst_rows.append(worst.head(int(args.top)))

    summary_df = pd.DataFrame(summaries).sort_values(["horizon", "symbol"], na_position="last")
    worst_df = pd.concat(worst_rows, axis=0, ignore_index=True) if worst_rows else pd.DataFrame()

    summary_path = out_dir / "trackc_audit_summary.csv"
    worst_path = out_dir / "trackc_audit_worst_columns.csv"

    summary_df.to_csv(summary_path, index=False)
    worst_df.to_csv(worst_path, index=False)

    # Print a compact console summary
    cols = [
        "symbol",
        "horizon",
        "rows",
        "cols",
        "start",
        "end",
        "nan_pct",
        "all_nan_cols",
        "any_nan_cols",
        "error",
    ]
    printable = summary_df[cols].copy()
    if "nan_pct" in printable.columns:
        printable["nan_pct"] = (printable["nan_pct"].fillna(0.0) * 100).round(3)
    print(printable.to_string(index=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: {worst_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
