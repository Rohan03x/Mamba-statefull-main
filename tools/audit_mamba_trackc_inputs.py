#!/usr/bin/env python3
"""Audit that Track-C parquets contain everything needed for Stage-B Mamba.

What this checks
- Track-C parquet exists per symbol for the given horizon.
- Column schema consistency vs a baseline symbol (important for pooled training).
- Per-family coverage using the same family naming conventions used across the repo.
- Canonical metric columns (score/conf/score_raw) where applicable.

By default this is metadata-only (fast). Use --scan-nans to additionally read
full data and report all-NaN columns (slow but more "real" readiness).

Examples
  python tools/audit_mamba_trackc_inputs.py --horizon 63
  python tools/audit_mamba_trackc_inputs.py --symbols AAPL,MSFT,NVDA --horizon 63
  python tools/audit_mamba_trackc_inputs.py --horizon 63 --scan-nans
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FEATURE_DIR_DEFAULT = REPO_ROOT / "cache" / "features"


def _parse_symbols(symbols_csv: Optional[str], *, fallback: Sequence[str]) -> List[str]:
    if symbols_csv:
        return [s.strip().upper() for s in str(symbols_csv).split(",") if s.strip()]
    return [str(s).upper() for s in fallback]


def _trackc_path(symbol: str, *, horizon: int, trackc_dir: Path) -> Path:
    return Path(trackc_dir) / f"{symbol.upper()}_h{int(horizon)}_trackc.parquet"


def _read_schema_cols(path: Path) -> List[str]:
    pf = pq.ParquetFile(path)
    return [str(c) for c in pf.schema.names]


def _is_etf(path: Path, columns: Sequence[str]) -> bool:
    if "is_etf" not in set(columns):
        return False
    try:
        table = pq.read_table(path, columns=["is_etf"])
        if table.num_rows == 0:
            return False
        col = table.column(0)
        # Robust: treat any truthy value as ETF.
        for v in col.to_pylist():
            if v is None:
                continue
            if bool(v):
                return True
        return False
    except Exception:
        # If we can't read it for some reason, fall back to non-ETF behavior.
        return False


def _diff_cols(cols: Sequence[str], baseline: Sequence[str]) -> tuple[list[str], list[str]]:
    s = set(cols)
    b = set(baseline)
    missing = sorted(b - s)
    extra = sorted(s - b)
    return missing, extra


def _doc_embed_matches(col: str) -> bool:
    lower = col.lower()
    if lower.startswith("doc_embedding_novelty_hf_"):
        return True
    # Known bare columns
    bare = {
        "n_events",
        "n_articles",
        "theme_weight",
        "baseline_mean",
        "novelty_spike_flag",
        "novelty_persistence_5d",
        "top_theme_numeric",
        "macro_novelty",
        "geopolitical_novelty",
        "regulatory_novelty",
        "energy_novelty",
        "conflict_novelty",
        "tech_novelty",
    }
    for base in bare:
        if lower == base or lower.startswith(base + "_lag"):
            return True
    return False


def _family_matches(family: str, col: str) -> bool:
    lower = col.lower()
    fam = family.lower()
    if lower == "date":
        return False

    if lower.startswith(fam + "_"):
        return True

    # Known exceptions
    if fam == "doc_embedding_novelty_hf":
        return _doc_embed_matches(col)

    if fam == "macro_tst_hf":
        if lower.startswith("macro_") or lower.startswith("macro_tst_"):
            return True

    return False


def _family_columns(family: str, columns: Sequence[str]) -> List[str]:
    return [c for c in columns if _family_matches(family, c)]


def _canonical_metric_columns(family: str) -> tuple[str, str, str]:
    # Prefer centralized mapping.
    from src.features.family_spec import canonical_metric_columns

    return canonical_metric_columns(family)


def _default_families(include_meta: bool) -> List[str]:
    from src.features.family_spec import default_base_families, default_hf_blocks, default_hf_modules

    exclude = {"fx", "commodities", "crypto"}
    fams: List[str] = []
    fams.extend([f for f in default_base_families() if f not in exclude])
    fams.extend(default_hf_modules())
    fams.extend(default_hf_blocks())
    if include_meta:
        fams.append("hf_agg")
    # de-dupe preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for f in fams:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _load_global_optuna_symbols() -> tuple[str, ...]:
    # Static parse, avoids importing pipeline.
    import ast

    pipeline_path = REPO_ROOT / "src" / "stage_b" / "pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"), filename=str(pipeline_path))
    for node in tree.body:
        target_id = None
        value_node = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_id = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_id = node.target.id
            value_node = node.value
        if target_id == "GLOBAL_OPTUNA_SYMBOLS" and value_node is not None:
            value = ast.literal_eval(value_node)
            if isinstance(value, tuple) and all(isinstance(s, str) for s in value):
                return tuple(str(s).upper() for s in value)
    raise RuntimeError("Failed to parse GLOBAL_OPTUNA_SYMBOLS from src/stage_b/pipeline.py")


@dataclass(frozen=True)
class FamilyStatus:
    family: str
    n_cols: int
    has_score: bool
    has_conf: bool
    has_score_raw: bool


@dataclass(frozen=True)
class SymbolStatus:
    symbol: str
    path: str
    ok_exists: bool
    ok_schema_vs_baseline: bool
    missing_vs_baseline: tuple[str, ...]
    extra_vs_baseline: tuple[str, ...]
    missing_families: tuple[str, ...]
    missing_metrics: Dict[str, tuple[str, ...]]
    all_nan_cols: int


def audit(
    *,
    symbols: Sequence[str],
    horizon: int,
    trackc_dir: Path,
    baseline: str,
    families: Sequence[str],
    scan_nans: bool,
    allow_etf_missing_fundamentals: bool,
) -> tuple[list[SymbolStatus], Dict[str, object]]:
    baseline_u = baseline.strip().upper()
    baseline_path = _trackc_path(baseline_u, horizon=horizon, trackc_dir=trackc_dir)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline parquet missing: {baseline_path}")

    baseline_cols = _read_schema_cols(baseline_path)
    baseline_set = set(baseline_cols)

    summary: Dict[str, object] = {
        "horizon": int(horizon),
        "trackc_dir": str(trackc_dir),
        "baseline": baseline_u,
        "baseline_cols": len(baseline_cols),
        "families": list(families),
        "scan_nans": bool(scan_nans),
    }

    out: List[SymbolStatus] = []

    for sym in symbols:
        sym_u = str(sym).upper().strip()
        path = _trackc_path(sym_u, horizon=horizon, trackc_dir=trackc_dir)
        if not path.exists():
            out.append(
                SymbolStatus(
                    symbol=sym_u,
                    path=str(path),
                    ok_exists=False,
                    ok_schema_vs_baseline=False,
                    missing_vs_baseline=tuple(sorted(baseline_set)),
                    extra_vs_baseline=tuple(),
                    missing_families=tuple(families),
                    missing_metrics={},
                    all_nan_cols=0,
                )
            )
            continue

        cols = _read_schema_cols(path)
        missing_vs_base, extra_vs_base = _diff_cols(cols, baseline_cols)
        ok_schema = len(missing_vs_base) == 0

        effective_families = list(families)
        if allow_etf_missing_fundamentals and _is_etf(path, cols):
            effective_families = [
                f
                for f in effective_families
                if f not in {"fin_g2", "fin_g3", "fin_g4", "fin_g5", "fin_g6", "fin_g7"}
            ]

        colset = set(cols)
        missing_families: List[str] = []
        missing_metrics: Dict[str, tuple[str, ...]] = {}

        for fam in effective_families:
            fam_cols = _family_columns(fam, cols)
            if len(fam_cols) == 0:
                missing_families.append(fam)
                continue
            score, conf, score_raw = _canonical_metric_columns(fam)
            mm: List[str] = []
            if score not in colset:
                mm.append("score")
            if conf not in colset:
                mm.append("conf")
            if score_raw not in colset:
                mm.append("score_raw")
            if mm:
                missing_metrics[fam] = tuple(mm)

        all_nan_cols = 0
        if scan_nans:
            import pandas as pd

            df = pd.read_parquet(path)
            frame = df
            if "date" in frame.columns:
                frame = frame.drop(columns=["date"], errors="ignore")
            frame = frame.select_dtypes(include=["number", "bool"]).astype(float)
            if len(frame) > 0 and len(frame.columns) > 0:
                nan_by_col = frame.isna().sum(axis=0)
                all_nan_cols = int((nan_by_col >= len(frame)).sum())

        out.append(
            SymbolStatus(
                symbol=sym_u,
                path=str(path),
                ok_exists=True,
                ok_schema_vs_baseline=ok_schema,
                missing_vs_baseline=tuple(missing_vs_base),
                extra_vs_baseline=tuple(extra_vs_base),
                missing_families=tuple(missing_families),
                missing_metrics=missing_metrics,
                all_nan_cols=int(all_nan_cols),
            )
        )

    return out, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit Track-C parquets for Mamba readiness")
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated list; default=GLOBAL_OPTUNA_SYMBOLS")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--trackc-dir", type=Path, default=FEATURE_DIR_DEFAULT)
    ap.add_argument("--baseline", type=str, default="AAPL")
    ap.add_argument("--include-meta", action="store_true", help="Include hf_agg/meta families")
    ap.add_argument("--scan-nans", action="store_true", help="Read full parquets and report all-NaN columns")
    ap.add_argument("--json", type=Path, default=None, help="Optional path to write JSON output")
    ap.add_argument("--fail-fast", action="store_true", help="Exit non-zero if any issues found")
    ap.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail if any symbol is missing baseline columns (default: warn-only)",
    )
    ap.add_argument(
        "--allow-etf-missing-fundamentals",
        action="store_true",
        default=True,
        help="Allow ETFs (is_etf==1) to omit fin_g2..fin_g7 families (default: on)",
    )
    args = ap.parse_args(argv)

    symbols = _parse_symbols(args.symbols, fallback=_load_global_optuna_symbols())
    families = _default_families(include_meta=bool(args.include_meta))

    statuses, summary = audit(
        symbols=symbols,
        horizon=int(args.horizon),
        trackc_dir=Path(args.trackc_dir),
        baseline=str(args.baseline),
        families=families,
        scan_nans=bool(args.scan_nans),
        allow_etf_missing_fundamentals=bool(args.allow_etf_missing_fundamentals),
    )

    bad: List[str] = []
    for st in statuses:
        if not st.ok_exists:
            bad.append(st.symbol)
            continue
        if bool(args.strict_schema) and not st.ok_schema_vs_baseline:
            bad.append(st.symbol)
            continue
        if st.missing_families:
            bad.append(st.symbol)
            continue
        if st.missing_metrics:
            bad.append(st.symbol)
            continue
        if bool(args.scan_nans) and st.all_nan_cols > 0:
            bad.append(st.symbol)
            continue

    # Compact console output
    print(
        f"mamba_trackc_audit: horizon={args.horizon} symbols={len(statuses)} baseline={str(args.baseline).upper()} "
        f"families={len(families)} scan_nans={bool(args.scan_nans)}"
    )
    for st in statuses:
        if st.symbol not in bad:
            continue
        reasons: List[str] = []
        if not st.ok_exists:
            reasons.append("missing_parquet")
        if bool(args.strict_schema) and st.ok_exists and not st.ok_schema_vs_baseline:
            reasons.append(f"missing_cols_vs_baseline={len(st.missing_vs_baseline)}")
        if st.missing_families:
            reasons.append(f"missing_families={len(st.missing_families)}")
        if st.missing_metrics:
            reasons.append(f"missing_metrics_families={len(st.missing_metrics)}")
        if bool(args.scan_nans) and st.all_nan_cols > 0:
            reasons.append(f"all_nan_cols={st.all_nan_cols}")
        print(f"- {st.symbol}: {'; '.join(reasons)}")

    payload = {
        "summary": summary,
        "bad_symbols": bad,
        "symbols": [
            {
                **st.__dict__,
                "missing_metrics": {k: list(v) for k, v in st.missing_metrics.items()},
                "missing_vs_baseline": list(st.missing_vs_baseline),
                "extra_vs_baseline": list(st.extra_vs_baseline),
                "missing_families": list(st.missing_families),
            }
            for st in statuses
        ],
    }

    if args.json is not None:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote_json: {args.json}")

    if args.fail_fast and bad:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
