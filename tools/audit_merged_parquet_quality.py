#!/usr/bin/env python3
"""Audit merged parquet(s) for NaNs / zeros / low-variance columns per feature family.

This tool is intended to match how Stage-B/Phase-2 actually attribute columns
to families (including special mappings for legacy/non-prefixed fields).

Examples:
    # Single file
    python tools/audit_merged_parquet_quality.py --merged cache/features/AAPL_h63_merged.parquet

    # Multi-symbol (recommended)
    python tools/audit_merged_parquet_quality.py --symbols AAPL,MSFT,NVDA --horizon 63

Outputs:
    Writes CSVs under artifacts/debug/:
        - per-column metrics (NaN/zero/std/nunique + family)
        - per-family summary counts
        - meta (row/col counts)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow running as a script from the repo root without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage_b.pipeline import StageBConfig, StageBPipeline


@dataclass
class FamilyStats:
    family: str
    ncols: int
    nrows: int
    nan_frac: float
    zero_frac: float
    low_var_cols: int
    const_cols: int
    all_nan_cols: int
    has_data_col: Optional[str]
    has_data_rate: float


def _parse_symbols(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").replace(" ", "").split(","):
        part = part.strip().upper()
        if part:
            out.append(part)
    return out


def _mk_pipeline(symbol: str, horizon: int) -> StageBPipeline:
    # Minimal config: we only need it for `_infer_column_families`.
    cfg = StageBConfig(
        symbol=str(symbol).upper(),
        horizons=[int(horizon)],
        horizon_clusters={"h": [int(horizon)]},
        run_lstm_for_clusters=[],
        min_samples_for_lstm=0,
        max_lstm_folds=0,
        lstm_trials_per_cluster=0,
        top_k_families_for_seq=0,
        performance_bar_for_lstm=0.0,
        start=None,
        end=None,
    )
    return StageBPipeline(config=cfg)


def _uniq_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _vectorized_metrics(
    df: pd.DataFrame,
    *,
    low_var_eps: float,
    compute_nunique: bool,
) -> pd.DataFrame:
    """Compute per-column metrics efficiently.

    Returns a dataframe indexed by column name with:
      nan_frac, zero_frac, std, (optional) nunique, and low-variance flags.
    """

    # Keep only numeric/bool columns.
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()
    if numeric_df.empty:
        return pd.DataFrame(
            columns=["nan_frac", "zero_frac", "std", "nunique", "is_const_lowvar"],
            index=pd.Index([], name="col"),
        )

    # Normalize types and handle inf.
    for c in numeric_df.columns:
        if pd.api.types.is_bool_dtype(numeric_df[c].dtype):
            numeric_df[c] = numeric_df[c].astype(np.int8)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    nan_frac = numeric_df.isna().mean(axis=0)
    filled = numeric_df.fillna(0.0)
    zero_frac = (filled == 0.0).mean(axis=0)
    std = filled.std(axis=0, ddof=0)

    if compute_nunique:
        nunique = filled.nunique(axis=0, dropna=False)
        is_const_lowvar = (nunique <= 1) | (std <= float(low_var_eps))
    else:
        nunique = pd.Series(index=filled.columns, data=np.nan)
        is_const_lowvar = std <= float(low_var_eps)

    out = pd.DataFrame(
        {
            "nan_frac": nan_frac.astype(float),
            "zero_frac": zero_frac.astype(float),
            "std": std.astype(float),
            "nunique": nunique.astype(float) if compute_nunique else nunique,
            "is_const_lowvar": is_const_lowvar.astype(bool),
        }
    )
    out.index.name = "col"
    return out


def _compute_family_stats_from_columns(
    *,
    merged: pd.DataFrame,
    family: str,
    cols: Sequence[str],
    low_var_eps: float,
) -> Optional[FamilyStats]:
    if not cols:
        return None

    sub = merged[list(cols)]
    num_cols = [c for c in cols if (c in sub.columns) and (pd.api.types.is_numeric_dtype(sub[c]) or pd.api.types.is_bool_dtype(sub[c]))]
    if not num_cols:
        return None
    sub = sub[num_cols].astype(float)
    nrows = int(sub.shape[0])
    ncols = int(sub.shape[1])

    nan_frac = float(sub.isna().sum().sum() / max(1, (nrows * ncols)))

    # Match earlier diagnostics: treat NaN as 0 for zero/variance checks.
    sub0 = sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zero_frac = float((sub0 == 0.0).sum().sum() / max(1, (nrows * ncols)))

    low_var_cols = 0
    const_cols = 0
    all_nan_cols = int((sub.isna().all(axis=0)).sum())
    for col in sub.columns:
        s = sub[col]
        s0 = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = float(s0.std(ddof=0))
        nunique = int(s0.nunique(dropna=False))
        if (nunique <= 1) or (std <= float(low_var_eps)):
            low_var_cols += 1
            if nunique <= 1:
                const_cols += 1

    has_data_col = next((c for c in sub.columns if str(c).endswith("_has_data")), None)
    has_data_rate = float("nan")
    if has_data_col is not None:
        s = sub[has_data_col]
        if s.notna().any():
            has_data_rate = float((s.fillna(0.0) > 0).mean())

    return FamilyStats(
        family=str(family),
        ncols=ncols,
        nrows=nrows,
        nan_frac=nan_frac,
        zero_frac=zero_frac,
        low_var_cols=low_var_cols,
        const_cols=const_cols,
        all_nan_cols=all_nan_cols,
        has_data_col=str(has_data_col) if has_data_col is not None else None,
        has_data_rate=has_data_rate,
    )


def _audit_merged(
    *,
    merged_path: Path,
    pipeline: StageBPipeline,
    low_var_eps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    merged = pd.read_parquet(merged_path)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.set_index("date")
    merged = merged.sort_index()

    include_role_hints = os.getenv("AUDIT_MERGED_INCLUDE_ROLE_HINTS", "0") == "1"
    all_cols = [str(c) for c in merged.columns]
    audit_cols = all_cols
    excluded_role_hint_cols = 0
    if not include_role_hints:
        audit_cols = [c for c in all_cols if "__is_" not in c]
        excluded_role_hint_cols = len(all_cols) - len(audit_cols)

    # Only audit the selected subset to avoid drowning signal-quality checks
    # in constant role-hint channels.
    merged_audit = merged[audit_cols]

    mapping = pipeline._infer_column_families(audit_cols)

    compute_nunique = os.getenv("AUDIT_MERGED_QUALITY_NUNIQUE", "0") == "1"
    metrics = _vectorized_metrics(merged_audit, low_var_eps=float(low_var_eps), compute_nunique=compute_nunique)

    # Build detail table.
    detail = metrics.reset_index().rename(columns={"col": "col"})
    detail["family"] = detail["col"].map(lambda c: mapping.get(str(c), "__unmapped__"))
    detail["dtype"] = detail["col"].map(lambda c: str(merged_audit[str(c)].dtype) if str(c) in merged_audit.columns else "")
    detail["is_nan_gt_0"] = detail["nan_frac"] > 0.0
    detail["is_nan_ge_10"] = detail["nan_frac"] >= 0.10
    detail["is_zero_ge_99"] = detail["zero_frac"] >= 0.99
    detail["is_zero_ge_95"] = detail["zero_frac"] >= 0.95

    # Family summary based on detail flags.
    by_family = (
        detail.groupby(["family"], as_index=False)
        .agg(
            ncols=("col", "count"),
            n_nan_gt_0=("is_nan_gt_0", "sum"),
            n_nan_ge_10=("is_nan_ge_10", "sum"),
            nan_max=("nan_frac", "max"),
            n_zero_ge_99=("is_zero_ge_99", "sum"),
            n_zero_ge_95=("is_zero_ge_95", "sum"),
            zero_max=("zero_frac", "max"),
            n_const_lowvar=("is_const_lowvar", "sum"),
            std_min=("std", "min"),
        )
        .sort_values(
            ["n_nan_ge_10", "n_nan_gt_0", "n_zero_ge_99", "n_const_lowvar", "family"],
            ascending=[False, False, False, False, True],
        )
    )

    meta: Dict[str, Any] = {
        "path": str(merged_path),
        "n_rows": int(merged.shape[0]),
        "n_cols_total": int(merged.shape[1]),
        "n_cols_audited": int(merged_audit.shape[1]),
        "excluded_role_hint_cols": int(excluded_role_hint_cols),
        "n_numeric_cols": int(metrics.shape[0]),
        "n_non_numeric_cols": int(merged_audit.shape[1] - int(metrics.shape[0])),
    }
    return detail, by_family, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", help="Path to merged parquet (single-file mode)")
    p.add_argument("--symbols", help="Comma-separated symbols for multi-file mode, e.g. AAPL,MSFT,NVDA")
    p.add_argument("--horizon", type=int, default=63, help="Horizon for multi-file mode")
    p.add_argument("--parquet-dir", default="cache/features", help="Directory containing *_merged.parquet")
    p.add_argument("--outdir", default="artifacts/debug", help="Output directory")
    p.add_argument("--lowvar-eps", type=float, default=1e-12)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    merged_paths: List[Path] = []
    symbols: List[str] = []
    if args.merged:
        merged_paths = [Path(str(args.merged))]
        # Best-effort symbol inference from filename.
        symbols = [Path(str(args.merged)).name.split("_", 1)[0].upper()]
    else:
        symbols = _parse_symbols(args.symbols or "")
        if not symbols:
            raise SystemExit("Provide --merged or --symbols")
        parquet_dir = Path(str(args.parquet_dir))
        merged_paths = [parquet_dir / f"{sym}_h{int(args.horizon)}_merged.parquet" for sym in symbols]

    pipe = _mk_pipeline(symbol=symbols[0], horizon=int(args.horizon))

    details: List[pd.DataFrame] = []
    fams: List[pd.DataFrame] = []
    metas: List[Dict[str, Any]] = []
    for sym, mp in zip(symbols, merged_paths):
        detail, by_family, meta = _audit_merged(merged_path=mp, pipeline=pipe, low_var_eps=float(args.lowvar_eps))
        detail.insert(0, "symbol", sym)
        by_family.insert(0, "symbol", sym)
        meta["symbol"] = sym
        details.append(detail)
        fams.append(by_family)
        metas.append(meta)

    detail_all = pd.concat(details, axis=0, ignore_index=True)
    fam_all = pd.concat(fams, axis=0, ignore_index=True)
    meta_df = pd.DataFrame(metas)

    detail_path = outdir / f"h{int(args.horizon)}_merged_quality_by_column_{stamp}.csv"
    fam_path = outdir / f"h{int(args.horizon)}_merged_quality_by_family_{stamp}.csv"
    meta_path = outdir / f"h{int(args.horizon)}_merged_quality_meta_{stamp}.csv"
    detail_all.to_csv(detail_path, index=False)
    fam_all.to_csv(fam_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    print("Wrote:")
    print(" -", detail_path)
    print(" -", fam_path)
    print(" -", meta_path)

    # Console summary per symbol.
    for sym in symbols:
        sub = fam_all[fam_all["symbol"] == sym].copy()
        sub = sub[(sub["n_nan_gt_0"] > 0) | (sub["n_zero_ge_95"] > 0) | (sub["n_const_lowvar"] > 0)]
        sub = sub.sort_values(["n_nan_ge_10", "n_nan_gt_0", "n_zero_ge_99", "n_const_lowvar"], ascending=[False, False, False, False])
        print("\n==", sym, "==")
        if sub.empty:
            print("No families with NaNs/zero-heavy/low-variance flags.")
        else:
            print(sub.head(args.top).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
