#!/usr/bin/env python3
"""Audit consolidated TrackC column schemas across symbols.

Reads `cache/features/*_h63_trackc.parquet` and writes:
- a JSON artifact with full column lists per symbol
- a CSV summary with counts and diffs vs a baseline symbol (default: AAPL)

Usage:
  python tools/audit_trackc_columns.py \
    --cache-features-dir cache/features \
    --horizon 63 \
    --baseline AAPL

Notes:
- Prints only a compact summary; full column lists go into artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class SymbolAudit:
    symbol: str
    parquet_path: str
    n_rows: int
    n_cols: int
    columns: List[str]


def _discover_trackc(cache_features_dir: Path, horizon: int) -> List[Tuple[str, Path]]:
    pattern = f"*_h{horizon}_trackc.parquet"
    out: List[Tuple[str, Path]] = []
    for p in sorted(cache_features_dir.glob(pattern)):
        name = p.name
        # <SYM>_h63_trackc.parquet
        sym = name.split("_h", 1)[0].upper()
        out.append((sym, p))
    return out


def _read_columns(p: Path) -> SymbolAudit:
    df = pd.read_parquet(p)
    cols = list(map(str, df.columns.tolist()))
    return SymbolAudit(
        symbol=p.name.split("_h", 1)[0].upper(),
        parquet_path=str(p),
        n_rows=int(len(df)),
        n_cols=int(len(cols)),
        columns=cols,
    )


def _diff(a: Sequence[str], b: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (missing_from_a, extra_in_a) comparing a vs b."""
    sa: Set[str] = set(a)
    sb: Set[str] = set(b)
    missing = sorted(sb - sa)
    extra = sorted(sa - sb)
    return missing, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-features-dir", type=str, default="cache/features")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--baseline", type=str, default="AAPL")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = ap.parse_args()

    cache_features_dir = Path(args.cache_features_dir)
    artifacts_dir = Path(args.artifacts_dir)

    discovered = _discover_trackc(cache_features_dir, int(args.horizon))
    if not discovered:
        raise SystemExit(f"No TrackC parquets found in {cache_features_dir} for horizon={args.horizon}")

    audits: Dict[str, SymbolAudit] = {}
    for sym, p in discovered:
        audits[sym] = _read_columns(p)

    baseline = args.baseline.strip().upper()
    if baseline not in audits:
        raise SystemExit(f"Baseline {baseline} not found. Found symbols: {sorted(audits.keys())}")

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_dir / "schema_audits"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"trackc_columns_h{args.horizon}_{ts}.json"
    csv_path = out_dir / f"trackc_columns_summary_h{args.horizon}_{ts}.csv"

    # Write JSON with full column lists.
    payload = {
        "generated_utc": ts,
        "horizon": int(args.horizon),
        "baseline": baseline,
        "symbols": {
            sym: {
                "parquet_path": audit.parquet_path,
                "n_rows": audit.n_rows,
                "n_cols": audit.n_cols,
                "columns": audit.columns,
            }
            for sym, audit in sorted(audits.items())
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Write CSV summary.
    base_cols = audits[baseline].columns
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol",
            "n_rows",
            "n_cols",
            "missing_vs_baseline",
            "extra_vs_baseline",
        ])
        for sym, audit in sorted(audits.items()):
            missing, extra = _diff(audit.columns, base_cols)
            w.writerow([
                sym,
                audit.n_rows,
                audit.n_cols,
                len(missing),
                len(extra),
            ])

    # Print compact summary.
    print(f"TrackC schema audit: {len(audits)} symbols, horizon={args.horizon}, baseline={baseline}")
    for sym, audit in sorted(audits.items()):
        missing, extra = _diff(audit.columns, base_cols)
        print(
            f"{sym:5s}  rows={audit.n_rows:5d}  cols={audit.n_cols:5d}"
            f"  missing_vs_{baseline}={len(missing):4d}  extra_vs_{baseline}={len(extra):4d}"
        )

    print(f"\nWrote full column lists: {json_path}")
    print(f"Wrote summary CSV:       {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
