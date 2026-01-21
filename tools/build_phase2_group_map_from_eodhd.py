from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running as: `python tools/build_phase2_group_map_from_eodhd.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.stage_b.pipeline import GLOBAL_OPTUNA_SYMBOLS
from src.stage_b_stateful.group_map import build_symbol_group_map_from_eodhd_cache

# Import centralized cache paths
try:
    from src.cache_paths import resolve_eodhd_cache_dir
    DEFAULT_EODHD_CACHE = resolve_eodhd_cache_dir()
except ImportError:
    DEFAULT_EODHD_CACHE = Path("data/cache/eodhd")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Phase2 symbol->group map from local EODHD cache.")
    ap.add_argument(
        "--cache-dir",
        default=DEFAULT_EODHD_CACHE,
        type=Path,
        help=f"Directory containing EODHD cached JSON profiles (default: {DEFAULT_EODHD_CACHE}).",
    )
    ap.add_argument(
        "--out",
        default=Path("artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv"),
        type=Path,
        help="Output CSV path (default: artifacts/meta_optimizer/phase2_symbol_group_map_gic_sector.csv).",
    )
    ap.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: GLOBAL_OPTUNA_SYMBOLS / GLOBAL13).",
    )
    ap.add_argument(
        "--group-keys",
        default="GicSector,Sector",
        help="Comma-separated group key preference list within EODHD General (default: GicSector,Sector).",
    )

    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    else:
        symbols = list(GLOBAL_OPTUNA_SYMBOLS)

    group_keys = [k.strip() for k in str(args.group_keys).split(",") if k.strip()]

    res = build_symbol_group_map_from_eodhd_cache(
        cache_dir=Path(args.cache_dir),
        symbols=symbols,
        out_path=Path(args.out),
        group_key_preference=tuple(group_keys),
    )

    print(
        {
            "out": str(res.out_path) if res.out_path is not None else None,
            "used_key": res.used_key,
            "found": len(res.group_by_symbol),
            "missing": list(res.missing_symbols),
        }
    )


if __name__ == "__main__":
    main()
