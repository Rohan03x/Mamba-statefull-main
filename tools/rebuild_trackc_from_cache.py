#!/usr/bin/env python3
"""Rebuild per-symbol merged TrackC panel from existing caches.

This is a fast path intended to refresh the consolidated parquet (cache/features/*_trackc.parquet)
without re-running walk-forward per-window generation.

Key use case:
- Compute and merge derived META family `hf_agg` (which has no standalone cache files)
  into the single TrackC parquet, using existing HF block caches.

This script is intentionally cache-first: it will not attempt to backfill missing
per-window family caches.

Example:
  /home/rohan/dcf_stage_b_clean/.venv/bin/python tools/rebuild_trackc_from_cache.py \
    --symbols MSFT NVDA AMZN META GOOGL TSLA AMD JPM UNH XOM \
    --horizon 63 \
    --start 2000-07-01 --end 2025-04-18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.family_spec import default_base_families, default_hf_blocks, default_hf_modules
from src.features.aggregator_panel import build_panel

FEATURE_PANEL_DIR = REPO_ROOT / "cache" / "features"

DEFAULT_EXCLUDE_FAMILIES: Set[str] = {"fx", "commodities", "crypto"}


def _parse_symbols(raw: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in raw:
        if not item:
            continue
        out.extend([p.strip().upper() for p in item.split(",") if p.strip()])
    seen = set()
    deduped: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


def _default_families(include_meta: bool) -> List[str]:
    fams: List[str] = []
    fams.extend([f for f in default_base_families() if f not in DEFAULT_EXCLUDE_FAMILIES])
    fams.extend(default_hf_modules())
    fams.extend(default_hf_blocks())
    if include_meta:
        fams.append("hf_agg")
    return fams


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "date" in df.columns:
        return df
    if df.index.name and str(df.index.name).lower() == "date":
        out = df.reset_index()
        out.rename(columns={out.columns[0]: "date"}, inplace=True)
        return out
    # best-effort: reset index into 'date'
    out = df.reset_index().rename(columns={"index": "date"})
    return out


def rebuild_one(
    *,
    symbol: str,
    horizon: int,
    cache_dir: Path,
    start: str,
    end: str,
    families: List[str],
    track_label: str,
) -> Optional[Path]:
    requested = list(families)
    wants_hf_agg = "hf_agg" in {f.lower() for f in requested}
    non_meta = [f for f in requested if f.lower() != "hf_agg"]

    panel = build_panel(
        symbol=symbol,
        start=start,
        end=end,
        families=non_meta,
        cache_dir=cache_dir,
        horizon=horizon,
        stage="B",
        view="both",
        window_idx=None,
    )

    if wants_hf_agg:
        hf_panel = build_panel(
            symbol=symbol,
            start=start,
            end=end,
            families=["hf_agg"],
            cache_dir=cache_dir,
            horizon=horizon,
            stage="B",
            view="both",
            window_idx=None,
        )
        if hf_panel is not None and not hf_panel.empty:
            # `build_panel(families=['hf_agg'])` returns a small "meta" panel that
            # includes shared HF columns (e.g., `has_data`, `hf_embed_*`, `is_etf`).
            # Those columns also exist in the main TrackC panel when HF modules are
            # requested, so joining naively can raise "columns overlap".
            #
            # For TrackC consolidation we only want the actual hf_agg outputs.
            hf_cols = [c for c in hf_panel.columns if str(c).lower().startswith("hf_agg_")]
            hf_panel = hf_panel[hf_cols] if hf_cols else hf_panel.iloc[:, 0:0]
            if panel is None or panel.empty:
                panel = hf_panel
            else:
                panel = panel.join(hf_panel, how="outer")

    if panel is None or panel.empty:
        return None

    included = panel.attrs.get("families") if hasattr(panel, "attrs") else []
    missing = panel.attrs.get("missing_families") if hasattr(panel, "attrs") else []
    if not isinstance(included, list):
        included = []
    if not isinstance(missing, list):
        missing = []

    if wants_hf_agg:
        has_hf_cols = any(str(c).lower().startswith("hf_agg_") for c in panel.columns)
        if has_hf_cols and "hf_agg" not in {str(f).lower() for f in included}:
            included.append("hf_agg")
        if has_hf_cols:
            missing = [f for f in missing if str(f).lower() != "hf_agg"]

    FEATURE_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_PANEL_DIR / f"{symbol.upper()}_h{horizon}_{track_label.lower()}.parquet"
    normalized = _normalize_date_column(panel)
    normalized.to_parquet(out_path, index=False)

    meta = {
        "symbol": symbol.upper(),
        "horizon": horizon,
        "track": track_label,
        "families_requested": requested,
        "families_included": included,
        "families_missing": missing,
        "coverage_start": start,
        "coverage_end": end,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild merged TrackC parquet from caches")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=63)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Symbol cache root; default=data/local_cache/<sym>_h<horizon>",
    )
    p.add_argument(
        "--families",
        default="all",
        help="'all' (base+hf) or comma-separated list",
    )
    p.add_argument(
        "--include-meta",
        action="store_true",
        help="Include meta families like hf_agg",
    )
    p.add_argument("--track-label", default="TrackC")
    args = p.parse_args(argv)
    args.symbols = _parse_symbols(args.symbols)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.families.strip().lower() == "all":
        families = _default_families(include_meta=bool(args.include_meta))
    else:
        families = [f.strip() for f in args.families.split(",") if f.strip()]

    failures = 0
    for sym in args.symbols:
        cache_dir = args.cache_dir
        if cache_dir is None:
            cache_dir = REPO_ROOT / "data" / "local_cache" / f"{sym.lower()}_h{args.horizon}"
        out = rebuild_one(
            symbol=sym,
            horizon=args.horizon,
            cache_dir=cache_dir,
            start=args.start,
            end=args.end,
            families=families,
            track_label=args.track_label,
        )
        if out is None:
            print(f"{sym}: FAILED (panel empty)")
            failures += 1
        else:
            print(f"{sym}: OK -> {out.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
