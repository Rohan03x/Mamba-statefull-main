#!/usr/bin/env python3
"""Convenience wrapper: materialize Dagster prep-families assets for symbols and horizons.

Supports both single-symbol and parallel multi-symbol execution:
- Single symbol: --symbol AAPL --horizon 63
- Multiple symbols (parallel): --symbols AAPL NVDA TSLA --horizon 63
- From file (parallel): --symbols-file universe.csv --horizon 63

Parallel execution:
- Launches separate Dagster runs for each symbol
- Dagster's QueuedRunCoordinator executes up to 8 symbols simultaneously
- Each symbol's assets execute sequentially (respecting dependencies)

Why this exists:
- Most family assets are partitioned by symbol only.
- A small set (quantile_forecast, plus horizon-bound HF blocks) are partitioned by (horizon|symbol).
- calibration and online_learning are Phase 2 Mamba-specific and are NOT included as Dagster families.
Dagster cannot materialize assets with mismatched PartitionsDefinitions in the same run, so we launch
separate runs in the right order.

Important:
- The default (two-run) path materializes family caches only; it does NOT materialize
    the stage-b `prep_families_manifest` asset (which writes the merged parquet under cache/features).
- By default, this materializes family caches AND the merged parquet (cache/merged/).
- Use `--no-merged` to skip merged parquet creation.
- Use `--single-run` to run `prep_families_manifest` only (assumes caches already exist).

Ordering note:
- quantile_forecast is horizon-bound and must complete before forecast_hf.
    This wrapper materializes quantile_forecast first.

Examples:
  # Single symbol (original behavior)
  .venv/bin/python tools/run_dagster_prep_families_all.py --symbol AAPL --horizon 63 --background
  
  # Multiple symbols in parallel (8 at a time)
  .venv/bin/python tools/run_dagster_prep_families_all.py --symbols AAPL NVDA TSLA --horizon 63
  
  # From file (processes up to 8 symbols simultaneously)
  .venv/bin/python tools/run_dagster_prep_families_all.py --symbols-file universe.csv --horizon 63
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    args: list[str]
    log_path: Path | None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "bin" / "python"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run(spec: RunSpec, env: dict[str, str], *, wait: bool) -> int:
    if spec.log_path is None:
        print(f"[run] {spec.name}: {' '.join(spec.args)}")
        return subprocess.call(spec.args, env=env)

    _ensure_parent(spec.log_path)
    print(f"[bg]  {spec.name}: {' '.join(spec.args)}")
    print(f"      log={spec.log_path}")
    with spec.log_path.open("wb") as f:
        proc = subprocess.Popen(spec.args, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"      pid={proc.pid}")
    if not wait:
        return 0
    return int(proc.wait())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Single symbol to process (e.g., AAPL)")
    parser.add_argument("--symbols", nargs="+", help="Multiple symbols to process in parallel (e.g., AAPL NVDA TSLA)")
    parser.add_argument("--symbols-file", type=Path, help="CSV file with symbols (one per line or comma-separated)")
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument(
        "--dagster-home",
        default=None,
        help="Defaults to <repo>/.dagster_home (recommended).",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Path to python executable (defaults to <repo>/.venv/bin/python).",
    )
    parser.add_argument(
        "--module",
        default="dagster_prep_families",
        help="Dagster code location module (default: dagster_prep_families).",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch runs in background and write logs under artifacts/logs/.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "In --background mode, do not wait for each run to finish. "
            "Warning: this disables strict sequencing and can cause cache-miss failures."
        ),
    )
    parser.add_argument(
        "--no-merged",
        action="store_true",
        help=(
            "Skip merged parquet creation. Only materialize individual family caches. "
            "(By default, merged parquet is generated after family caches)"
        ),
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=(
            "Run prep_families_manifest (stage-b) for this (symbol,horizon) partition. "
            "This produces cache/merged/<SYMBOL>_h<H>_merged.parquet in one Dagster run."
        ),
    )
    parser.add_argument(
        "--enable-fallback-sources",
        action="store_true",
        help=(
            "Enable fallback data sources (yfinance, Tiingo, etc.) when primary sources fail. "
            "By default, prep families enforces strict source policy (EODHD-only)."
        ),
    )
    parser.add_argument(
        "--enable-live-fallback",
        action="store_true",
        help=(
            "Allow live API calls on cache miss. By default, prep families operates in "
            "cache-only mode to avoid expensive external fetches."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between launching runs in seconds (default: 0.5, useful for staggering API calls)",
    )

    args = parser.parse_args(argv)
    
    # Validate symbol arguments
    symbols_list = []
    if args.symbol:
        symbols_list = [args.symbol.upper().strip()]
    elif args.symbols:
        symbols_list = [s.upper().strip() for s in args.symbols]
    elif args.symbols_file:
        with open(args.symbols_file) as f:
            content = f.read()
            # Support both CSV and newline-separated
            symbols_list = [s.strip().upper() for s in content.replace(',', '\n').split('\n') if s.strip()]
    else:
        parser.error("Must provide either --symbol, --symbols, or --symbols-file")
    
    if not symbols_list:
        parser.error("No symbols to process")
    
    # Parallel mode if multiple symbols
    is_parallel = len(symbols_list) > 1

    repo_root = _repo_root()
    dagster_home = Path(args.dagster_home) if args.dagster_home else (repo_root / ".dagster_home")
    python_exe = Path(args.python) if args.python else _default_python(repo_root)

    if not python_exe.exists():
        raise SystemExit(f"Python not found at: {python_exe}")

    # Use python -m dagster to handle paths with spaces
    dagster_cmd = [str(python_exe), "-m", "dagster"]

    base_env = os.environ.copy()
    base_env["DAGSTER_HOME"] = str(dagster_home)

    # Reduce burst rate to avoid transient provider blocks (e.g., 403) during highly parallel runs.
    # Can be overridden by the caller.
    base_env.setdefault("EODHD_MIN_REQUEST_INTERVAL", "0.25")

    horizon = int(args.horizon)
    
    # For parallel mode, print overview
    if is_parallel:
        print(f"📊 Processing {len(symbols_list)} symbols in parallel:")
        print(f"   Symbols: {', '.join(symbols_list[:10])}{'...' if len(symbols_list) > 10 else ''}")
        print(f"   Horizon: {horizon}")
        print(f"   Parallelism: Symbol-level (up to 8 runs simultaneously)")
        print(f"   Asset execution: Sequential within each symbol")
        print()

    # Fallback source policy: by default, enforce strict EODHD-only and no live fallback.
    # CLI flags can relax these constraints.
    if args.enable_fallback_sources:
        base_env["DAGSTER_PREP_ENFORCE_EODHD_ONLY"] = "0"
        base_env["DAGSTER_PREP_ENFORCE_NO_PROXY_SOURCES"] = "0"
    else:
        base_env.setdefault("DAGSTER_PREP_ENFORCE_EODHD_ONLY", "1")
        base_env.setdefault("DAGSTER_PREP_ENFORCE_NO_PROXY_SOURCES", "1")

    # CHANGED: Default to enabling live fallback (allow generation when caches don't exist)
    # This ensures fresh runs can generate all families instead of requiring pre-existing caches
    base_env["DAGSTER_PREP_ENFORCE_NO_LIVE_FALLBACK"] = "0"
    base_env["STAGE_B_NO_LIVE_FALLBACK"] = "0"

    # We default to strict sequencing (wait between runs) to ensure the sequential chain
    # completes before HF blocks and to avoid cross-run cache races.
    wait = not bool(args.detach)

    ts = _timestamp()
    logs_dir = repo_root / "artifacts" / "logs"
    
    # Parallel mode: process all symbols
    if is_parallel:
        return _run_parallel(
            symbols=symbols_list,
            horizon=horizon,
            dagster_cmd=dagster_cmd,
            base_env=base_env,
            args=args,
            logs_dir=logs_dir,
            ts=ts,
            wait=wait,
        )
    
    # Single symbol mode (original behavior)
    symbol = symbols_list[0]
    
    # Create symbol-specific environment
    env = base_env.copy()
    env["DAGSTER_SYMBOLS"] = symbol
    env["DAGSTER_HORIZONS"] = str(horizon)

    symbol_log = logs_dir / f"dagster_materialize_all_symbol_{symbol}_{ts}.log" if args.background else None
    horizon_log = (
        logs_dir / f"dagster_materialize_all_horizon_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )

    merged_log = (
        logs_dir / f"dagster_materialize_with_merged_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )

    single_log = (
        logs_dir / f"dagster_materialize_single_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )
    
    horizon_pk = f"{horizon}|{symbol}"

    if args.single_run:
        single_spec = RunSpec(
            name="prep_families_manifest(single-run)",
            args=dagster_cmd + [
                "asset",
                "materialize",
                "-m",
                args.module,
                "--select",
                "prep_families_manifest",
                "--partition",
                horizon_pk,
            ],
            log_path=single_log,
        )
        return _run(single_spec, env=env, wait=wait)

    # By default, generate merged parquet after family caches
    with_merged = not args.no_merged
    
    if args.no_merged:
        print(
            "[note] --no-merged: materializing family caches only (no merged parquet)."
        )

    # 1) Symbol-only families
    symbol_spec = RunSpec(
        name="families(symbol-only)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "group:families",
            "--partition",
            symbol,
        ],
        log_path=symbol_log,
    )

    # 2) Horizon-bound families (quantile_forecast only - calibration/online_learning are Phase 2 specific)
    horizon_chain_spec = RunSpec(
        name="families(horizon-chain)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "family_quantile_forecast",
            "--partition",
            horizon_pk,
        ],
        log_path=horizon_log,
    )

    horizon_rest_log = (
        logs_dir / f"dagster_materialize_all_horizon_rest_{symbol}_h{horizon}_{ts}.log" if args.background else None
    )
    horizon_rest_spec = RunSpec(
        name="families(horizon-rest)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "family_forecast_hf",  # hf_agg removed - HF blocks consumed directly
            "--partition",
            horizon_pk,
        ],
        log_path=horizon_rest_log,
    )

    # 3) Unified merged parquet (stage-b)
    merged_spec = RunSpec(
        name="prep_families_manifest(merged)",
        args=dagster_cmd + [
            "asset",
            "materialize",
            "-m",
            args.module,
            "--select",
            "prep_families_manifest",
            "--partition",
            horizon_pk,
        ],
        log_path=merged_log,
    )

    rc = _run(symbol_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    rc = _run(horizon_chain_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    rc = _run(horizon_rest_spec, env=env, wait=wait)
    if rc != 0:
        return rc

    if with_merged:
        return _run(merged_spec, env=env, wait=wait)

    return 0


def _run_parallel(
    symbols: list[str],
    horizon: int,
    dagster_cmd: list[str],
    base_env: dict[str, str],
    args: argparse.Namespace,
    logs_dir: Path,
    ts: str,
    wait: bool,
) -> int:
    """Run multiple symbols in parallel using separate Dagster runs."""
    
    with_merged = not args.no_merged
    launched_runs = []
    
    for symbol in symbols:
        # Create symbol-specific environment
        env = base_env.copy()
        env["DAGSTER_SYMBOLS"] = symbol
        env["DAGSTER_HORIZONS"] = str(horizon)
        
        horizon_pk = f"{horizon}|{symbol}"
        
        print(f"🚀 Launching runs for {symbol}...")
        
        # 1) Symbol-only families
        symbol_log = logs_dir / f"dagster_materialize_symbol_{symbol}_{ts}.log" if args.background else None
        symbol_spec = RunSpec(
            name=f"families(symbol-only:{symbol})",
            args=dagster_cmd + [
                "job", "launch",
                "-m", args.module,
                "-j", "prep_families_families_symbol_job",
                "--config-json", '{"ops": {}, "resources": {}}',
                "-p", symbol,
            ],
            log_path=symbol_log,
        )
        
        try:
            if symbol_log:
                _ensure_parent(symbol_log)
                with symbol_log.open("wb") as f:
                    proc = subprocess.Popen(symbol_spec.args, env=env, stdout=f, stderr=subprocess.STDOUT)
                launched_runs.append((symbol, "symbol", proc))
                print(f"   ✓ Symbol-only run launched (pid={proc.pid}, log={symbol_log})")
            else:
                proc = subprocess.Popen(symbol_spec.args, env=env)
                launched_runs.append((symbol, "symbol", proc))
                print(f"   ✓ Symbol-only run launched (pid={proc.pid})")
        except Exception as e:
            print(f"   ✗ Failed to launch symbol-only run: {e}")
            continue
        
        time.sleep(args.delay)
        
        # 2) Horizon-bound families
        horizon_log = logs_dir / f"dagster_materialize_horizon_{symbol}_h{horizon}_{ts}.log" if args.background else None
        horizon_spec = RunSpec(
            name=f"families(horizon:{symbol})",
            args=dagster_cmd + [
                "job", "launch",
                "-m", args.module,
                "-j", "prep_families_families_horizon_job",
                "--config-json", '{"ops": {}, "resources": {}}',
                "-p", horizon_pk,
            ],
            log_path=horizon_log,
        )
        
        try:
            if horizon_log:
                _ensure_parent(horizon_log)
                with horizon_log.open("wb") as f:
                    proc = subprocess.Popen(horizon_spec.args, env=env, stdout=f, stderr=subprocess.STDOUT)
                launched_runs.append((symbol, "horizon", proc))
                print(f"   ✓ Horizon-bound run launched (pid={proc.pid}, log={horizon_log})")
            else:
                proc = subprocess.Popen(horizon_spec.args, env=env)
                launched_runs.append((symbol, "horizon", proc))
                print(f"   ✓ Horizon-bound run launched (pid={proc.pid})")
        except Exception as e:
            print(f"   ✗ Failed to launch horizon-bound run: {e}")
            continue
        
        time.sleep(args.delay)
        
        # 3) Merged parquet (if enabled)
        if with_merged and not args.single_run:
            merged_log = logs_dir / f"dagster_materialize_merged_{symbol}_h{horizon}_{ts}.log" if args.background else None
            merged_spec = RunSpec(
                name=f"prep_families_manifest(merged:{symbol})",
                args=dagster_cmd + [
                    "asset", "materialize",
                    "-m", args.module,
                    "--select", "prep_families_manifest",
                    "--partition", horizon_pk,
                ],
                log_path=merged_log,
            )
            
            try:
                if merged_log:
                    _ensure_parent(merged_log)
                    with merged_log.open("wb") as f:
                        proc = subprocess.Popen(merged_spec.args, env=env, stdout=f, stderr=subprocess.STDOUT)
                    launched_runs.append((symbol, "merged", proc))
                    print(f"   ✓ Merged parquet run launched (pid={proc.pid}, log={merged_log})")
                else:
                    proc = subprocess.Popen(merged_spec.args, env=env)
                    launched_runs.append((symbol, "merged", proc))
                    print(f"   ✓ Merged parquet run launched (pid={proc.pid})")
            except Exception as e:
                print(f"   ✗ Failed to launch merged parquet run: {e}")
        
        time.sleep(args.delay)
    
    print(f"\n✅ Launched {len(launched_runs)} Dagster runs for {len(symbols)} symbols")
    print(f"\n📊 Dagster will execute up to 8 symbols simultaneously (configured in .dagster_home/dagster.yaml)")
    print(f"   Each symbol's assets execute sequentially (respecting dependencies)")
    print(f"\n📊 Monitor progress:")
    print(f"   Dagster UI: dagster dev -f dagster_prep_families/definitions.py")
    print(f"   Visit: http://localhost:3000/runs")
    print(f"   Or run: dagster run list -m {args.module}")
    
    # If wait mode and not detached, wait for all processes
    if wait and not args.detach:
        print(f"\n⏳ Waiting for all runs to complete...")
        failed = []
        for symbol, run_type, proc in launched_runs:
            rc = proc.wait()
            if rc != 0:
                failed.append((symbol, run_type, rc))
                print(f"   ✗ {symbol} ({run_type}) failed with code {rc}")
            else:
                print(f"   ✓ {symbol} ({run_type}) completed")
        
        if failed:
            print(f"\n❌ {len(failed)} runs failed:")
            for symbol, run_type, rc in failed:
                print(f"   - {symbol} ({run_type}): exit code {rc}")
            return 1
        
        print(f"\n✅ All {len(launched_runs)} runs completed successfully!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
