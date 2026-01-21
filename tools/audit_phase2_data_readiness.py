from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# Allow running as: `python tools/audit_phase2_data_readiness.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.stage_b_stateful.group_map import build_symbol_group_map_from_eodhd_cache  # noqa: E402


def _load_global_optuna_symbols() -> tuple[str, ...]:
    """Load GLOBAL_OPTUNA_SYMBOLS without importing the full pipeline module.

    Importing `src.stage_b.pipeline` has non-trivial side effects (optional
    dependency checks, banners/logging). For readiness audits we only need the
    static constant.
    """

    pipeline_path = REPO_ROOT / "src" / "stage_b" / "pipeline.py"
    try:
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
    except Exception:
        pass

    # Fallback: import (may be noisy). Keep as last resort so the tool still works.
    from src.stage_b.pipeline import GLOBAL_OPTUNA_SYMBOLS  # type: ignore

    return tuple(str(s).upper() for s in GLOBAL_OPTUNA_SYMBOLS)


@dataclass(frozen=True)
class SymbolAudit:
    symbol: str
    ok_trackc: bool
    ok_adv_inputs: bool
    issues: tuple[str, ...]


def _iter_symbols(symbols_csv: str | None) -> list[str]:
    if symbols_csv:
        return [s.strip().upper() for s in str(symbols_csv).split(",") if s.strip()]
    return list(_load_global_optuna_symbols())


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        return {str(n) for n in pf.schema.names}
    except Exception:
        # Fallback: pandas (may read more than metadata depending on engine)
        import pandas as pd

        df = pd.read_parquet(path, engine="pyarrow")
        return {str(c) for c in df.columns}


def _trackc_path(symbol: str, *, horizon: int, trackc_dir: Path) -> Path:
    return Path(trackc_dir) / f"{symbol}_h{int(horizon)}_trackc.parquet"


def _check_adv_inputs(cols: set[str]) -> bool:
    # Matches Phase2 `_compute_adv_usd` supported columns.
    if "dollar_volume" in cols:
        return True
    if "volume" in cols:
        return True
    if "microstructure_micro_turnover" in cols:
        return True
    if "micro_turnover" in cols:
        return True
    return False


def audit(
    *,
    symbols: Sequence[str],
    horizon: int,
    trackc_dir: Path,
    require_liquidity: bool,
) -> list[SymbolAudit]:
    out: list[SymbolAudit] = []

    for sym in symbols:
        issues: list[str] = []
        path = _trackc_path(sym, horizon=horizon, trackc_dir=trackc_dir)
        if not path.exists():
            issues.append(f"missing_trackc_parquet:{path}")
            out.append(SymbolAudit(symbol=sym, ok_trackc=False, ok_adv_inputs=False, issues=tuple(issues)))
            continue

        cols = _parquet_columns(path)
        ok_adv = _check_adv_inputs(cols)
        if require_liquidity and not ok_adv:
            issues.append("missing_adv_inputs: need one of {dollar_volume, volume, microstructure_micro_turnover, micro_turnover}")

        out.append(SymbolAudit(symbol=sym, ok_trackc=True, ok_adv_inputs=ok_adv, issues=tuple(issues)))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Phase2 data readiness (Track-C + group-map prerequisites).")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols. Default: GLOBAL13 (GLOBAL_OPTUNA_SYMBOLS).")
    ap.add_argument("--horizon", default=63, type=int)
    ap.add_argument("--trackc-dir", default="cache/features", type=Path)

    ap.add_argument(
        "--preset",
        default="research",
        choices=["none", "research", "production"],
        help="Preset posture used to decide which prerequisites are required.",
    )
    ap.add_argument(
        "--capital-usd",
        default=None,
        type=float,
        help="If provided (>0) and preset=production, require ADV inputs for liquidity constraints.",
    )

    ap.add_argument("--group-map-path", default=None, type=Path)
    ap.add_argument("--eodhd-cache-dir", default="data/cache/eodhd", type=Path)
    ap.add_argument(
        "--write-default-group-map",
        action="store_true",
        help="If group map is missing, attempt to build it from local EODHD cache into artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv",
    )

    ap.add_argument("--json", action="store_true", help="Emit JSON report.")
    args = ap.parse_args()

    symbols = _iter_symbols(args.symbols)
    preset = str(args.preset).lower().strip()

    require_liquidity = False
    if preset == "production" and args.capital_usd is not None and float(args.capital_usd) > 0:
        require_liquidity = True

    # Group map: optional for research, recommended for production.
    group_map_path: Path | None = args.group_map_path
    default_map = Path("artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv")
    group_map_ok = False
    group_map_missing: tuple[str, ...] = ()

    if group_map_path is None:
        group_map_path = default_map if default_map.exists() else None

    if group_map_path is not None and Path(group_map_path).exists():
        group_map_ok = True
    elif args.write_default_group_map:
        res = build_symbol_group_map_from_eodhd_cache(
            cache_dir=Path(args.eodhd_cache_dir),
            symbols=symbols,
            out_path=default_map,
            group_key_preference=("GicSector", "Sector"),
        )
        group_map_missing = tuple(res.missing_symbols)
        group_map_ok = bool(res.group_by_symbol) and len(res.missing_symbols) == 0
        group_map_path = default_map if default_map.exists() else None

    audits = audit(
        symbols=symbols,
        horizon=int(args.horizon),
        trackc_dir=Path(args.trackc_dir),
        require_liquidity=require_liquidity,
    )

    ok_trackc = all(a.ok_trackc for a in audits)
    ok_liq = all((a.ok_adv_inputs or not require_liquidity) and a.ok_trackc for a in audits)

    # Production posture: group map should exist and cover all symbols, but don't hard-fail by default.
    prod_recommends_group_map = preset == "production"

    report = {
        "preset": preset,
        "horizon": int(args.horizon),
        "symbols": symbols,
        "trackc_dir": str(Path(args.trackc_dir)),
        "group_map_path": str(group_map_path) if group_map_path is not None else None,
        "group_map_ok": bool(group_map_ok),
        "group_map_missing_symbols": list(group_map_missing),
        "require_liquidity": bool(require_liquidity),
        "ok_trackc": bool(ok_trackc),
        "ok_liquidity_prereqs": bool(ok_liq),
        "symbols_report": [
            {
                "symbol": a.symbol,
                "ok_trackc": a.ok_trackc,
                "ok_adv_inputs": a.ok_adv_inputs,
                "issues": list(a.issues),
            }
            for a in audits
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"preset={preset} horizon={args.horizon} n_symbols={len(symbols)}")
        print(f"trackc: {'OK' if ok_trackc else 'FAIL'} (dir={args.trackc_dir})")
        if require_liquidity:
            print(f"liquidity_prereqs: {'OK' if ok_liq else 'FAIL'} (capital_usd={args.capital_usd})")
        else:
            print("liquidity_prereqs: SKIP (not required)")

        if prod_recommends_group_map:
            print(f"group_map: {'OK' if group_map_ok else 'MISSING/INCOMPLETE'} (path={group_map_path})")
            if group_map_missing:
                print(f"  missing_symbols: {', '.join(group_map_missing)}")
        else:
            print(f"group_map: {'OK' if group_map_ok else 'optional/not found'} (path={group_map_path})")

        bad = [a for a in audits if a.issues]
        if bad:
            print("\nIssues:")
            for a in bad:
                print(f"- {a.symbol}: {', '.join(a.issues)}")

    # Exit non-zero only on hard prerequisites.
    exit_code = 0
    if not ok_trackc:
        exit_code = 2
    elif require_liquidity and not ok_liq:
        exit_code = 3

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
