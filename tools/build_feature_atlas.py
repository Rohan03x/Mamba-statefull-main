#!/usr/bin/env python3
"""Feature Atlas: Machine-readable map of feature families and their quality metrics.

This tool creates a comprehensive audit of all features in the merged parquets:
- Which families exist
- Which columns are actually non-stub (variance > threshold, null% < threshold)
- Which roles they play (predictive/risk/regime/hygiene)
- Which columns are safe for risk scaling, gating, veto

Outputs:
- artifacts/feature_atlas.parquet: Per-column quality metrics
- artifacts/feature_atlas_summary.md: Human-readable summary
- artifacts/feature_atlas_families.json: Family-level aggregates

Usage:
    python tools/build_feature_atlas.py --symbols AAPL,NVDA --horizon 63
    python tools/build_feature_atlas.py --universe core --horizon 63 --regime-labels
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=False)

import numpy as np
import pandas as pd

from src.cache_paths import (
    MERGED_CACHE_ROOT,
    ARTIFACTS_ROOT,
    merged_parquet_path,
    merged_meta_path,
)

LOGGER = logging.getLogger("feature_atlas")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Thresholds for stub/dead column detection
VARIANCE_THRESHOLD = 1e-10  # Below this → dead column
ZERO_PCT_THRESHOLD = 0.95  # Above this → mostly zeros (potential stub)
NULL_PCT_THRESHOLD = 0.50  # Above this → mostly null (unreliable)
STALENESS_THRESHOLD_DAYS = 10  # days_since_update > this → stale

# Regime buckets for cross-correlation analysis
REGIME_BUCKETS = ["bear", "neutral", "bull", "crisis"]

# Output paths
ATLAS_DIR = ARTIFACTS_ROOT / "feature_atlas"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ColumnQuality:
    """Quality metrics for a single column."""
    column: str
    family: str = ""
    role: str = "predictive"  # predictive, risk, regime, hygiene, label
    
    # Basic statistics
    n_rows: int = 0
    n_valid: int = 0
    n_null: int = 0
    n_zero: int = 0
    n_inf: int = 0
    
    # Derived percentages
    null_pct: float = 0.0
    zero_pct: float = 0.0
    inf_pct: float = 0.0
    
    # Distributional metrics
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p01: float = 0.0
    p05: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Staleness (for *_days_since_update columns)
    is_staleness_column: bool = False
    staleness_mean: float = 0.0
    staleness_max: float = 0.0
    staleness_pct_above_threshold: float = 0.0
    
    # Cross-correlation with labels (by regime if available)
    label_corr_overall: float = 0.0
    label_corr_bear: float = 0.0
    label_corr_neutral: float = 0.0
    label_corr_bull: float = 0.0
    label_corr_crisis: float = 0.0
    label_spearman_overall: float = 0.0
    
    # Usage permissions
    risk_scale_ok: bool = False
    gating_ok: bool = False
    veto_ok: bool = False
    
    # Quality flags
    is_stub: bool = False
    is_dead: bool = False
    is_stale: bool = False
    drop_recommended: bool = False
    mask_to_zero: bool = False
    quality_score: float = 1.0  # Composite quality [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class FamilyQuality:
    """Aggregate quality metrics for a feature family."""
    family: str
    n_columns: int = 0
    n_dead: int = 0
    n_stub: int = 0
    n_stale: int = 0
    n_usable: int = 0
    usable_pct: float = 0.0
    avg_quality_score: float = 0.0
    avg_label_corr: float = 0.0
    columns: List[str] = field(default_factory=list)
    dead_columns: List[str] = field(default_factory=list)
    stub_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_provenance(symbol: str, horizon: int) -> Dict[str, Any]:
    """Load provenance/metadata from merged.meta.json."""
    meta_path = merged_meta_path(symbol, horizon)
    if not meta_path.exists():
        LOGGER.warning(f"No meta.json found for {symbol} h{horizon}")
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load meta.json for {symbol}: {e}")
        return {}


def compute_column_quality(
    col_name: str,
    col_data: pd.Series,
    labels: Optional[pd.Series] = None,
    regime_series: Optional[pd.Series] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> ColumnQuality:
    """Compute comprehensive quality metrics for a single column."""
    cq = ColumnQuality(column=col_name)
    
    # Extract provenance info
    if provenance:
        role_map = provenance.get("column_role_map", {})
        family_map = provenance.get("column_family_map", {})
        risk_scale_ok_map = provenance.get("column_risk_scale_ok_map", {})
        gating_ok_map = provenance.get("column_gating_ok_map", {})
        veto_ok_map = provenance.get("column_veto_ok_map", {})
        
        cq.family = str(family_map.get(col_name, "unknown"))
        cq.role = str(role_map.get(col_name, "predictive"))
        cq.risk_scale_ok = bool(risk_scale_ok_map.get(col_name, False))
        cq.gating_ok = bool(gating_ok_map.get(col_name, False))
        cq.veto_ok = bool(veto_ok_map.get(col_name, False))
    
    # Basic counts
    cq.n_rows = len(col_data)
    cq.n_null = int(col_data.isna().sum())
    cq.n_valid = cq.n_rows - cq.n_null
    
    if cq.n_valid == 0:
        cq.null_pct = 1.0
        cq.is_dead = True
        cq.drop_recommended = True
        cq.quality_score = 0.0
        return cq
    
    # Convert to numeric
    numeric = pd.to_numeric(col_data, errors="coerce")
    valid_data = numeric.dropna()
    
    if len(valid_data) == 0:
        cq.null_pct = 1.0
        cq.is_dead = True
        cq.drop_recommended = True
        cq.quality_score = 0.0
        return cq
    
    # Compute percentages
    cq.null_pct = cq.n_null / cq.n_rows if cq.n_rows > 0 else 0.0
    cq.n_zero = int((valid_data == 0).sum())
    cq.zero_pct = cq.n_zero / len(valid_data) if len(valid_data) > 0 else 0.0
    cq.n_inf = int((~np.isfinite(valid_data)).sum())
    cq.inf_pct = cq.n_inf / len(valid_data) if len(valid_data) > 0 else 0.0
    
    # Distributional metrics
    arr = valid_data.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    
    if len(arr) > 0:
        cq.mean = float(np.mean(arr))
        cq.std = float(np.std(arr))
        cq.variance = float(np.var(arr))
        cq.min_val = float(np.min(arr))
        cq.max_val = float(np.max(arr))
        
        # Percentiles
        cq.p01 = float(np.percentile(arr, 1))
        cq.p05 = float(np.percentile(arr, 5))
        cq.p25 = float(np.percentile(arr, 25))
        cq.p50 = float(np.percentile(arr, 50))
        cq.p75 = float(np.percentile(arr, 75))
        cq.p95 = float(np.percentile(arr, 95))
        cq.p99 = float(np.percentile(arr, 99))
        
        # Higher moments
        if cq.std > 1e-12:
            centered = arr - cq.mean
            cq.skewness = float(np.mean(centered ** 3) / (cq.std ** 3))
            cq.kurtosis = float(np.mean(centered ** 4) / (cq.std ** 4) - 3.0)
    
    # Check if staleness column
    if "_days_since_update" in col_name.lower():
        cq.is_staleness_column = True
        cq.staleness_mean = cq.mean
        cq.staleness_max = cq.max_val
        cq.staleness_pct_above_threshold = float(np.mean(arr > STALENESS_THRESHOLD_DAYS))
        cq.is_stale = cq.staleness_mean > STALENESS_THRESHOLD_DAYS
    
    # Label correlations
    if labels is not None and len(labels) > 0:
        try:
            # Align by index
            aligned = pd.DataFrame({"feat": numeric, "label": labels}).dropna()
            if len(aligned) > 10:
                cq.label_corr_overall = float(aligned["feat"].corr(aligned["label"]))
                cq.label_spearman_overall = float(
                    aligned["feat"].rank().corr(aligned["label"].rank())
                )
                
                # By regime if available
                if regime_series is not None:
                    aligned["regime"] = regime_series.reindex(aligned.index)
                    for regime in REGIME_BUCKETS:
                        mask = aligned["regime"].str.lower() == regime.lower()
                        if mask.sum() > 10:
                            subset = aligned.loc[mask]
                            corr = float(subset["feat"].corr(subset["label"]))
                            setattr(cq, f"label_corr_{regime}", corr if np.isfinite(corr) else 0.0)
        except Exception as e:
            LOGGER.debug(f"Failed to compute label correlation for {col_name}: {e}")
    
    # Quality flags
    cq.is_dead = cq.variance < VARIANCE_THRESHOLD
    cq.is_stub = cq.zero_pct > ZERO_PCT_THRESHOLD and not cq.is_dead
    cq.drop_recommended = cq.is_dead or cq.null_pct > NULL_PCT_THRESHOLD
    cq.mask_to_zero = cq.is_stub and not cq.drop_recommended
    
    # Composite quality score
    # Factors: variance, null%, zero%, staleness, label correlation
    quality_components = []
    
    # Variance score (log-scaled)
    if cq.variance > 0:
        var_score = min(1.0, np.log1p(cq.variance / VARIANCE_THRESHOLD) / 10.0)
    else:
        var_score = 0.0
    quality_components.append(var_score * 0.3)
    
    # Null penalty
    null_score = 1.0 - cq.null_pct
    quality_components.append(null_score * 0.2)
    
    # Zero penalty
    zero_score = 1.0 - max(0.0, (cq.zero_pct - 0.5) * 2.0)
    quality_components.append(zero_score * 0.2)
    
    # Label correlation bonus
    corr_score = 0.5 + min(0.5, abs(cq.label_corr_overall) * 2.0)
    quality_components.append(corr_score * 0.3)
    
    cq.quality_score = float(np.clip(sum(quality_components), 0.0, 1.0))
    
    return cq


def build_feature_atlas(
    symbols: List[str],
    horizon: int,
    label_col: str = "fwd_ret_h63",
    regime_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, FamilyQuality], Dict[str, Any]]:
    """Build the feature atlas from merged parquets.
    
    Returns:
        - Column-level quality DataFrame
        - Family-level quality dict
        - Summary statistics dict
    """
    all_column_qualities: List[ColumnQuality] = []
    family_columns: Dict[str, List[ColumnQuality]] = {}
    
    symbols_processed = 0
    total_columns = 0
    
    for symbol in symbols:
        sym_upper = symbol.upper()
        parquet_path = merged_parquet_path(sym_upper, horizon)
        
        if not parquet_path.exists():
            LOGGER.warning(f"No merged parquet for {sym_upper} h{horizon}")
            continue
        
        LOGGER.info(f"Processing {sym_upper} h{horizon}...")
        
        try:
            df = pd.read_parquet(parquet_path)
            provenance = load_provenance(sym_upper, horizon)
        except Exception as e:
            LOGGER.error(f"Failed to load {sym_upper}: {e}")
            continue
        
        # Identify labels and regime columns
        labels = None
        if label_col in df.columns:
            labels = pd.to_numeric(df[label_col], errors="coerce")
        else:
            # Try to find a forward return column
            fwd_cols = [c for c in df.columns if c.startswith("fwd_ret_")]
            if fwd_cols:
                labels = pd.to_numeric(df[fwd_cols[0]], errors="coerce")
        
        regime_series = None
        if regime_col and regime_col in df.columns:
            regime_series = df[regime_col]
        
        # Process each column
        for col in df.columns:
            # Skip metadata columns
            if col in {"date", "symbol", "session", "row_id", "timestamp"}:
                continue
            
            cq = compute_column_quality(
                col_name=col,
                col_data=df[col],
                labels=labels,
                regime_series=regime_series,
                provenance=provenance,
            )
            
            all_column_qualities.append(cq)
            
            # Track by family
            if cq.family not in family_columns:
                family_columns[cq.family] = []
            family_columns[cq.family].append(cq)
            
            total_columns += 1
        
        symbols_processed += 1
    
    # Build column-level DataFrame
    column_df = pd.DataFrame([cq.to_dict() for cq in all_column_qualities])
    
    # Build family-level aggregates
    family_qualities: Dict[str, FamilyQuality] = {}
    for family, columns in family_columns.items():
        fq = FamilyQuality(family=family)
        fq.n_columns = len(columns)
        fq.n_dead = sum(1 for c in columns if c.is_dead)
        fq.n_stub = sum(1 for c in columns if c.is_stub)
        fq.n_stale = sum(1 for c in columns if c.is_stale)
        fq.n_usable = fq.n_columns - fq.n_dead - fq.n_stub
        fq.usable_pct = fq.n_usable / fq.n_columns if fq.n_columns > 0 else 0.0
        fq.avg_quality_score = np.mean([c.quality_score for c in columns])
        fq.avg_label_corr = np.mean([abs(c.label_corr_overall) for c in columns])
        fq.columns = [c.column for c in columns]
        fq.dead_columns = [c.column for c in columns if c.is_dead]
        fq.stub_columns = [c.column for c in columns if c.is_stub]
        family_qualities[family] = fq
    
    # Summary statistics
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "symbols_processed": symbols_processed,
        "horizon": horizon,
        "total_columns": total_columns,
        "unique_columns": len(set(c.column for c in all_column_qualities)),
        "n_families": len(family_qualities),
        "n_dead_columns": sum(fq.n_dead for fq in family_qualities.values()),
        "n_stub_columns": sum(fq.n_stub for fq in family_qualities.values()),
        "n_stale_columns": sum(fq.n_stale for fq in family_qualities.values()),
        "n_drop_recommended": sum(1 for c in all_column_qualities if c.drop_recommended),
        "n_mask_recommended": sum(1 for c in all_column_qualities if c.mask_to_zero),
        "avg_quality_score": float(np.mean([c.quality_score for c in all_column_qualities])),
        "thresholds": {
            "variance_threshold": VARIANCE_THRESHOLD,
            "zero_pct_threshold": ZERO_PCT_THRESHOLD,
            "null_pct_threshold": NULL_PCT_THRESHOLD,
            "staleness_threshold_days": STALENESS_THRESHOLD_DAYS,
        },
    }
    
    return column_df, family_qualities, summary


def generate_markdown_report(
    column_df: pd.DataFrame,
    family_qualities: Dict[str, FamilyQuality],
    summary: Dict[str, Any],
) -> str:
    """Generate a human-readable markdown summary."""
    lines = [
        "# Feature Atlas Summary",
        "",
        f"**Generated:** {summary['generated_at']}",
        f"**Horizon:** {summary['horizon']} days",
        f"**Symbols Processed:** {summary['symbols_processed']}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Columns | {summary['total_columns']:,} |",
        f"| Unique Columns | {summary['unique_columns']:,} |",
        f"| Families | {summary['n_families']} |",
        f"| Dead Columns (variance < {summary['thresholds']['variance_threshold']}) | {summary['n_dead_columns']} |",
        f"| Stub Columns (zero% > {summary['thresholds']['zero_pct_threshold']:.0%}) | {summary['n_stub_columns']} |",
        f"| Stale Columns | {summary['n_stale_columns']} |",
        f"| **Drop Recommended** | {summary['n_drop_recommended']} |",
        f"| **Mask Recommended** | {summary['n_mask_recommended']} |",
        f"| Average Quality Score | {summary['avg_quality_score']:.3f} |",
        "",
        "## Family Summary",
        "",
        "| Family | Columns | Dead | Stub | Usable | Quality | Avg |Corr| |",
        "|--------|---------|------|------|--------|---------|-----|",
    ]
    
    # Sort families by usable percentage descending
    sorted_families = sorted(
        family_qualities.values(),
        key=lambda f: (f.usable_pct, f.avg_quality_score),
        reverse=True,
    )
    
    for fq in sorted_families:
        lines.append(
            f"| {fq.family} | {fq.n_columns} | {fq.n_dead} | {fq.n_stub} | "
            f"{fq.usable_pct:.0%} | {fq.avg_quality_score:.3f} | {fq.avg_label_corr:.3f} |"
        )
    
    # Dead columns list
    dead_cols = column_df[column_df["is_dead"] == True]["column"].unique().tolist()
    if dead_cols:
        lines.extend([
            "",
            "## Dead Columns (Recommend Drop)",
            "",
            "These columns have near-zero variance and should be dropped:",
            "",
        ])
        for col in sorted(dead_cols)[:50]:  # Limit to first 50
            lines.append(f"- `{col}`")
        if len(dead_cols) > 50:
            lines.append(f"- ... and {len(dead_cols) - 50} more")
    
    # Stub columns list
    stub_cols = column_df[column_df["is_stub"] == True]["column"].unique().tolist()
    if stub_cols:
        lines.extend([
            "",
            "## Stub Columns (Recommend Mask to Zero)",
            "",
            "These columns are mostly zeros and should be masked:",
            "",
        ])
        for col in sorted(stub_cols)[:50]:
            lines.append(f"- `{col}`")
        if len(stub_cols) > 50:
            lines.append(f"- ... and {len(stub_cols) - 50} more")
    
    # Top predictive columns by label correlation
    if "label_corr_overall" in column_df.columns:
        top_corr = column_df.nlargest(20, "label_corr_overall")[
            ["column", "family", "label_corr_overall", "quality_score"]
        ]
        if len(top_corr) > 0:
            lines.extend([
                "",
                "## Top 20 Predictive Columns (by Label Correlation)",
                "",
                "| Column | Family | Correlation | Quality |",
                "|--------|--------|-------------|---------|",
            ])
            for _, row in top_corr.iterrows():
                lines.append(
                    f"| {row['column']} | {row['family']} | "
                    f"{row['label_corr_overall']:.4f} | {row['quality_score']:.3f} |"
                )
    
    lines.extend([
        "",
        "---",
        "",
        "*Generated by `tools/build_feature_atlas.py`*",
    ])
    
    return "\n".join(lines)


def save_atlas(
    column_df: pd.DataFrame,
    family_qualities: Dict[str, FamilyQuality],
    summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Save atlas outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    
    # Parquet (main machine-readable output)
    parquet_path = output_dir / "feature_atlas.parquet"
    column_df.to_parquet(parquet_path, index=False)
    outputs["parquet"] = parquet_path
    LOGGER.info(f"Saved: {parquet_path}")
    
    # Families JSON
    families_path = output_dir / "feature_atlas_families.json"
    with open(families_path, "w") as f:
        json.dump(
            {k: v.to_dict() for k, v in family_qualities.items()},
            f,
            indent=2,
            default=str,
        )
    outputs["families"] = families_path
    LOGGER.info(f"Saved: {families_path}")
    
    # Summary JSON
    summary_json_path = output_dir / "feature_atlas_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    outputs["summary_json"] = summary_json_path
    LOGGER.info(f"Saved: {summary_json_path}")
    
    # Markdown report
    md_report = generate_markdown_report(column_df, family_qualities, summary)
    md_path = output_dir / "feature_atlas_summary.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    outputs["markdown"] = md_path
    LOGGER.info(f"Saved: {md_path}")
    
    # Drop/mask list (for Track-C integration)
    drop_mask_path = output_dir / "feature_atlas_drop_mask.json"
    drop_list = column_df[column_df["drop_recommended"] == True]["column"].unique().tolist()
    mask_list = column_df[column_df["mask_to_zero"] == True]["column"].unique().tolist()
    with open(drop_mask_path, "w") as f:
        json.dump(
            {
                "drop_columns": sorted(set(drop_list)),
                "mask_columns": sorted(set(mask_list)),
                "generated_at": summary["generated_at"],
                "thresholds": summary["thresholds"],
            },
            f,
            indent=2,
        )
    outputs["drop_mask"] = drop_mask_path
    LOGGER.info(f"Saved: {drop_mask_path}")
    
    return outputs


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Feature Atlas: comprehensive feature quality audit"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL",
        help="Comma-separated list of symbols (e.g., AAPL,NVDA,MSFT)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Universe preset: 'core', 'global13', 'all'. Overrides --symbols.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=63,
        help="Horizon in trading days (default: 63)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="fwd_ret_h63",
        help="Label column for correlation analysis",
    )
    parser.add_argument(
        "--regime-col",
        type=str,
        default=None,
        help="Regime column for bucketed correlation analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ATLAS_DIR),
        help=f"Output directory (default: {ATLAS_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Determine symbols
    if args.universe:
        try:
            from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE
            symbols = list(DEFAULT_CANDIDATE_UNIVERSE)
            LOGGER.info(f"Using universe preset: {args.universe} ({len(symbols)} symbols)")
        except ImportError:
            LOGGER.warning("Could not load universe preset, using default symbols")
            symbols = args.symbols.split(",")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    LOGGER.info(f"Building Feature Atlas for {len(symbols)} symbols, horizon={args.horizon}")
    
    # Build atlas
    column_df, family_qualities, summary = build_feature_atlas(
        symbols=symbols,
        horizon=args.horizon,
        label_col=args.label_col,
        regime_col=args.regime_col,
    )
    
    # Save outputs
    output_dir = Path(args.output_dir)
    outputs = save_atlas(column_df, family_qualities, summary, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE ATLAS COMPLETE")
    print("=" * 60)
    print(f"Symbols processed: {summary['symbols_processed']}")
    print(f"Total columns: {summary['total_columns']:,}")
    print(f"Families: {summary['n_families']}")
    print(f"Dead columns: {summary['n_dead_columns']}")
    print(f"Stub columns: {summary['n_stub_columns']}")
    print(f"Drop recommended: {summary['n_drop_recommended']}")
    print(f"Mask recommended: {summary['n_mask_recommended']}")
    print(f"Avg quality score: {summary['avg_quality_score']:.3f}")
    print("=" * 60)
    print("\nOutputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
