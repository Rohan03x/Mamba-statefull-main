#!/usr/bin/env python3
"""Audit Phase-2 family coverage, 3-pillar compression, and Stage-A weights.

This script is intentionally "diagnostic": it mirrors the Phase-2 Track-C build
path (Track-A 3-pillar encoding + Track-B blocks + Track-C concat) and reports:

- Per-symbol: which raw (unified panel) columns map to which families
- Per-family: expected raw column counts vs per-symbol present/missing
- 3-pillar: method/k_final selected per Stage-A family (PCA/AE/passthrough)
- Mamba input: which encoded columns exist per family and whether Stage-A weights
  are applied post-standardization (non-1 weights on those columns)

Outputs a JSON report (default: artifacts/debug/phase2_family_audit_h{H}_label_{label}.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage_b_stateful import phase2_stateful as p2
from src.stage_b.optuna_optimizer import (
    HF_BLOCK_FAMILIES,
    STAGE_A_FAMILIES,
    TRACK_B_SUMMARY_BLOCKS,
    OptunaConfig,
    StageBOptunaOptimizer,
    _is_raw_passthrough_family,
)


def _parse_symbols(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    parts = [p.strip().upper() for p in arg.split(",") if p.strip()]
    return parts or None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _group_cols_by_family(columns: Iterable[str], mapping: Mapping[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for c in columns:
        fam = mapping.get(c)
        if not fam:
            continue
        grouped[str(fam)].append(str(c))
    return dict(grouped)


def _col_to_family_for_trackc(
    *,
    track_a: pd.DataFrame,
    block_summaries: Mapping[str, pd.DataFrame],
    optimizer: StageBOptunaOptimizer,
) -> Dict[str, str]:
    """Rebuild the same col->family mapping used for post-std weights."""
    col_to_family: Dict[str, str] = {}

    summary_weight_alias = {
        "quantile": "quantile_forecast",
        "online": "online_learning",
        "arima": "arima_forecast",
    }

    for c in list(track_a.columns):
        s = str(c)
        if "_enc_" in s:
            col_to_family[s] = s.split("_enc_", 1)[0]

    for fam in HF_BLOCK_FAMILIES:
        cols = getattr(optimizer, "family_columns", {}).get(str(fam), [])
        for c in cols or []:
            col_to_family[str(c)] = str(fam)

    for summary_name in TRACK_B_SUMMARY_BLOCKS:
        df = block_summaries.get(summary_name)
        if df is None or df.empty:
            continue
        fam_key = summary_weight_alias.get(str(summary_name), str(summary_name))
        for c in df.columns:
            col_to_family[str(c)] = fam_key

    return col_to_family


def _count_trackc_cols_by_family(
    *,
    track_c_cols: Sequence[str],
    col_to_family: Mapping[str, str],
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for c in track_c_cols:
        fam = col_to_family.get(str(c))
        if not fam:
            continue
        counts[str(fam)] += 1
    return dict(counts)


def _build_family_params(
    *,
    optimizer: StageBOptunaOptimizer,
    column_families: Mapping[str, str],
    opt_cfg: OptunaConfig,
) -> Tuple[Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]], Dict[str, int]]:
    family_sizes: Dict[str, int] = {}
    for _, fam in column_families.items():
        if fam:
            family_sizes[str(fam)] = int(family_sizes.get(str(fam), 0) + 1)

    family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
    for fam in STAGE_A_FAMILIES:
        fam = str(fam)
        family_size = int(family_sizes.get(fam, 0) or 0)
        include = bool(family_size > 0)
        if not include:
            family_params[fam] = (False, 0, "none", 0.0, {})
            continue

        if _is_raw_passthrough_family(fam):
            family_params[fam] = (True, int(family_size), "passthrough", 1.0, {})
            continue

        dim_cfg = getattr(optimizer, "family_optimal_dims", {}).get(fam)
        if dim_cfg is None:
            k_final = int(min(int(getattr(opt_cfg, "three_pillar_dim_min", 4)), int(family_size)))
            method = "pca"
        else:
            k_final = int(min(int(getattr(dim_cfg, "k_final", 0)), int(family_size)))
            method = str(getattr(dim_cfg, "method", "pca"))
        k_final = int(max(k_final, 1))
        family_params[fam] = (True, int(k_final), str(method), 1.0, {})

    return family_params, family_sizes


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols. Default: GLOBAL_OPTUNA_SYMBOLS")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--label-id", type=str, default="base")
    ap.add_argument(
        "--family-weights-path",
        type=str,
        default=str(Path("artifacts/debug/stage_a_global13_real_20260101_235926/family_weights_best.json")),
    )
    ap.add_argument("--train-start", type=str, default=p2.DEFAULT_PHASE2_TRAIN_START)
    ap.add_argument("--train-end", type=str, default=p2.DEFAULT_PHASE2_TRAIN_END)
    ap.add_argument("--oos-start", type=str, default=p2.DEFAULT_PHASE2_OOS_START)
    ap.add_argument("--oos-end", type=str, default=p2.DEFAULT_PHASE2_OOS_END)
    ap.add_argument("--three-pillar-dim-max", type=int, default=512)
    ap.add_argument("--three-pillar-pca-variance", type=float, default=0.95)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--max-unmapped-examples", type=int, default=20)

    args = ap.parse_args(list(argv) if argv is not None else None)

    symbols = _parse_symbols(args.symbols)
    if symbols is None:
        symbols = list(getattr(p2, "GLOBAL_OPTUNA_SYMBOLS"))

    horizon = int(args.horizon)
    label_id = str(args.label_id).strip() or "base"

    cfg: Dict[str, Any] = {
        "phase2_label_id": label_id,
        "phase2_family_weights_path": str(args.family_weights_path),
        "three_pillar_dim_max": int(args.three_pillar_dim_max),
        "three_pillar_pca_variance": float(args.three_pillar_pca_variance),
    }

    # Resolve Stage-A weights (static) exactly like Phase-2.
    stage_a_payload = p2._resolve_phase2_stage_a_payload(cfg, symbol=str(symbols[0]).upper(), horizon=horizon)
    stage_a_weights = (
        p2._phase2_static_stage_a_weights(payload=stage_a_payload, asof=p2._to_datetime(args.train_end))
        or p2._resolve_phase2_family_weights(cfg, symbol=str(symbols[0]).upper(), horizon=horizon)
        or {}
    )

    # Build per-symbol panels and per-symbol family mappings.
    pipelines: Dict[str, Any] = {}
    panels: Dict[str, pd.DataFrame] = {}
    per_symbol_col_families: Dict[str, Dict[str, str]] = {}
    per_symbol_unmapped: Dict[str, List[str]] = {}

    for sym in symbols:
        pipe = p2._build_stage_b_pipeline(symbol=sym, horizon=horizon, start=str(args.train_start), end=str(args.oos_end))
        panel = pipe._build_panel(horizon)
        panel = p2._apply_feature_smoothing(panel, smoothing_type="none", smoothing_window=3)

        col_fams = pipe._infer_column_families(panel.columns)
        unmapped = [str(c) for c in panel.columns if str(c) not in col_fams]

        pipelines[sym] = pipe
        panels[sym] = panel
        per_symbol_col_families[sym] = dict(col_fams)
        per_symbol_unmapped[sym] = unmapped

    # Union schema check: columns present in some symbols but not the anchor.
    anchor = str(symbols[0]).upper()
    anchor_cols = set(str(c) for c in panels[anchor].columns)
    extra_cols_by_symbol: Dict[str, List[str]] = {}
    for sym in symbols[1:]:
        extra = [str(c) for c in panels[sym].columns if str(c) not in anchor_cols]
        if extra:
            extra_cols_by_symbol[sym] = extra

    # Use Phase-2 behavior: infer mapping from anchor columns only.
    anchor_column_families = pipelines[anchor]._infer_column_families(panels[anchor].columns)

    # Pooled train panel for 3-pillar analysis.
    pooled_parts: List[pd.DataFrame] = []
    train_pos_by: Dict[str, np.ndarray] = {}
    for sym in symbols:
        idx = pd.DatetimeIndex(panels[sym].index)
        train_pos = p2._select_index_positions(idx, p2._to_datetime(args.train_start), p2._to_datetime(args.train_end))
        train_pos_by[sym] = train_pos
        pooled_parts.append(panels[sym].iloc[train_pos])
    pooled_panel = pd.concat(pooled_parts, axis=0)
    pooled_idx = np.arange(len(pooled_panel), dtype=int)

    # Build optimizer and run 3-pillar.
    opt_cfg = OptunaConfig(
        max_total_dims=0,
        horizon=horizon,
        sequence_model_type="mamba",
        use_three_pillar_dims=True,
        three_pillar_dim_max=int(args.three_pillar_dim_max),
        three_pillar_pca_variance=float(args.three_pillar_pca_variance),
    )
    try:
        setattr(opt_cfg, "weight_normalization", "none")
    except Exception:
        pass

    optimizer = StageBOptunaOptimizer(config=opt_cfg)
    optimizer._analyze_families(pooled_panel, anchor_column_families, stage_a_weights=stage_a_weights or None)

    family_params, family_sizes = _build_family_params(optimizer=optimizer, column_families=anchor_column_families, opt_cfg=opt_cfg)

    # Fit Track-A encoder on pooled train data once (as in Phase-2).
    optimizer._build_track_a(pooled_panel, anchor_column_families, family_params, pooled_idx)

    # Prepare per-family dims/method summary.
    dims_by_family: Dict[str, Any] = {}
    for fam in STAGE_A_FAMILIES:
        fam = str(fam)
        dim_cfg = getattr(optimizer, "family_optimal_dims", {}).get(fam)
        if dim_cfg is None:
            dims_by_family[fam] = {
                "method": "none",
                "k_final": 0,
                "family_size_anchor": int(family_sizes.get(fam, 0) or 0),
            }
        else:
            dims_by_family[fam] = {
                **asdict(dim_cfg),
                "family_size_anchor": int(family_sizes.get(fam, 0) or 0),
            }

    # Per-symbol: build tracks + compute weights + collect coverage.
    per_symbol: Dict[str, Any] = {}
    trackc_cols_by_symbol: Dict[str, List[str]] = {}

    for sym in symbols:
        pipe = pipelines[sym]
        panel = panels[sym]
        train_pos = train_pos_by[sym]

        pipe._current_train_idx = train_pos
        block_summaries = pipe._compute_block_summaries(panel)

        track_a, _ = optimizer._build_track_a(panel, anchor_column_families, family_params, train_pos)
        track_b = optimizer._build_track_b(panel, anchor_column_families, block_summaries, track_b_family_weights=None)
        track_c = optimizer._build_track_c(track_a, track_b, 1.0, 1.0)
        if track_c is None or track_c.empty:
            raise RuntimeError(f"Track C empty for {sym}")

        trackc_cols = [str(c) for c in track_c.select_dtypes(include=["number", "bool"]).astype(float).columns]
        trackc_cols_by_symbol[sym] = trackc_cols

        # Stage-A weights for this symbol (symbol-specific override supported).
        sym_payload = p2._resolve_phase2_stage_a_payload(cfg, symbol=sym, horizon=horizon)
        sym_weights = (
            p2._phase2_static_stage_a_weights(payload=sym_payload, asof=p2._to_datetime(args.train_end))
            or p2._resolve_phase2_family_weights(cfg, symbol=sym, horizon=horizon)
            or {}
        )

        # Narrow mapping: mirrors the mapping actually used by post-std weight application.
        col_to_family_weight_map = _col_to_family_for_trackc(
            track_a=track_a,
            block_summaries=block_summaries,
            optimizer=optimizer,
        )

        # Broad mapping: infer families directly from Track-C column names (helps detect
        # families that exist in Track-C but are NOT currently weight-mapped).
        col_to_family_inferred = pipe._infer_column_families(trackc_cols)

        trackc_counts_weight_map = _count_trackc_cols_by_family(track_c_cols=trackc_cols, col_to_family=col_to_family_weight_map)
        trackc_counts_inferred = _count_trackc_cols_by_family(track_c_cols=trackc_cols, col_to_family=col_to_family_inferred)

        w_vec, w_cols = p2._build_post_std_weights_vector(
            track_c=track_c,
            track_a=track_a,
            panel=panel,
            block_summaries=block_summaries,
            optimizer=optimizer,
            family_weights=sym_weights,
        )
        w_vec = np.asarray(w_vec, dtype=float)
        w_cols = [str(c) for c in w_cols]
        col_to_w = {c: float(w_vec[i]) for i, c in enumerate(w_cols)}

        # Family-level coverage.
        fam_rows: Dict[str, Any] = {}
        expected_cols_by_family = getattr(optimizer, "family_columns", {}) or {}

        # Collect for Stage-A + passthrough + track-b blocks.
        families_of_interest = sorted(
            set(STAGE_A_FAMILIES)
            | set(HF_BLOCK_FAMILIES)
            | set(TRACK_B_SUMMARY_BLOCKS)
            | set(str(k) for k in (sym_weights or {}).keys())
        )

        for fam in families_of_interest:
            fam = str(fam)
            expected_cols = [str(c) for c in (expected_cols_by_family.get(fam) or [])]
            present_cols = [str(c) for c in expected_cols if str(c) in panel.columns]

            enc_cols = [str(c) for c in track_a.columns if str(c).startswith(f"{fam}_enc_")]

            # Track-C family attribution under the *actual weight mapping*.
            tc_cols_weight_map = int(trackc_counts_weight_map.get(fam, 0) or 0)

            # Track-C family attribution under a *name-based inference* mapping.
            tc_cols_inferred = int(trackc_counts_inferred.get(fam, 0) or 0)

            # How many weight-mapped Track-C columns have non-1 weights.
            if tc_cols_weight_map:
                weighted_cols = [c for c in trackc_cols if col_to_family_weight_map.get(c) == fam and abs(col_to_w.get(c, 1.0) - 1.0) > 1e-12]
                weighted_cols_n = int(len(weighted_cols))
            else:
                weighted_cols_n = 0

            fam_rows[fam] = {
                "weight": _safe_float(sym_weights.get(fam)),
                "expected_raw_cols": int(len(expected_cols)),
                "present_raw_cols": int(len(present_cols)),
                "missing_raw_cols": int(max(0, len(expected_cols) - len(present_cols))),
                "track_a_enc_cols": int(len(enc_cols)),
                "track_c_cols_inferred": int(tc_cols_inferred),
                "track_c_cols_weight_map": int(tc_cols_weight_map),
                "track_c_weighted_cols": int(weighted_cols_n),
                "dim_method": dims_by_family.get(fam, {}).get("method"),
                "k_final": dims_by_family.get(fam, {}).get("k_final"),
                "raw_passthrough": bool(_is_raw_passthrough_family(fam)),
            }

        # Unmapped raw columns in the unified panel.
        unmapped = per_symbol_unmapped.get(sym, [])
        per_symbol[sym] = {
            "panel_rows": int(len(panel)),
            "panel_cols": int(len(panel.columns)),
            "unmapped_raw_cols_n": int(len(unmapped)),
            "unmapped_raw_cols_examples": unmapped[: int(args.max_unmapped_examples)],
            "track_a_shape": [int(track_a.shape[0]), int(track_a.shape[1])],
            "track_b_shape": [int(track_b.shape[0]), int(track_b.shape[1])],
            "track_c_shape": [int(track_c.shape[0]), int(track_c.shape[1])],
            "track_c_unmapped_cols_n": int(sum(1 for c in trackc_cols if c not in col_to_family_inferred)),
            "families": fam_rows,
        }

    # Verify Track-C schema consistency.
    ref_cols = trackc_cols_by_symbol[str(symbols[0]).upper()]
    schema_equal = {sym: (trackc_cols_by_symbol[sym] == ref_cols) for sym in symbols}

    stage_a_fams = sorted(str(k) for k in stage_a_weights.keys())

    report: Dict[str, Any] = {
        "horizon": horizon,
        "label_id": label_id,
        "symbols": symbols,
        "anchor_symbol": anchor,
        "stage_a_weights_path": str(args.family_weights_path),
        "stage_a_weights_n": int(len(stage_a_weights)),
        "stage_a_weight_families": stage_a_fams,
        "extra_columns_outside_anchor_by_symbol": {k: v[:100] for k, v in extra_cols_by_symbol.items()},
        "trackc_schema_equal_to_anchor": schema_equal,
        "three_pillar_dim_max": int(args.three_pillar_dim_max),
        "three_pillar_pca_variance": float(args.three_pillar_pca_variance),
        "dims_by_stage_a_family": dims_by_family,
        "per_symbol": per_symbol,
    }

    out_path = Path(args.out) if args.out else Path("artifacts/debug") / f"phase2_family_audit_h{horizon}_label_{label_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    # Console summary (high signal).
    print(f"Wrote report: {out_path}")
    print(f"Stage-A weights: n={len(stage_a_weights)} families")
    not_equal = [s for s, ok in schema_equal.items() if not ok]
    print(f"Track-C schema identical across symbols: {len(not_equal)==0}")
    if not_equal:
        print(f"  MISMATCH symbols: {not_equal[:10]}")
    if extra_cols_by_symbol:
        print(f"Columns outside anchor schema exist for {len(extra_cols_by_symbol)} symbols (see report)")

    # Report Stage-A family coverage quickly.
    missing_by_symbol: Dict[str, List[str]] = {}
    for sym in symbols:
        fam_rows = per_symbol[sym]["families"]
        missing = [f for f in stage_a_fams if int(fam_rows.get(f, {}).get("expected_raw_cols", 0)) <= 0]
        if missing:
            missing_by_symbol[sym] = missing
    print(f"Stage-A families missing (expected_raw_cols==0): {len(missing_by_symbol)} symbols")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
