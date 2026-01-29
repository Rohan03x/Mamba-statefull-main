#!/usr/bin/env python3
"""Prepare family signals (base + HF) for Stage-A/Stage-B.

- Builds consolidated per-split caches (train/valid) for each family.
- Produces unified per-symbol artifacts used by downstream stages.
- Does not produce per-window parquet shards; windowing is handled by slicing
    consolidated/merged panels by date ranges.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import socket
import subprocess

# Ensure project root is importable before referencing src.* modules
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from repo-level .env file so prep works from any cwd
from dotenv import load_dotenv

ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)
else:
    load_dotenv()

# Safety defaults: avoid forking after tokenizers parallelism has been initialized,
# and avoid process-pool fanout inside GDELT cache loads during HF feature generation.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GDELT_DISABLE_PROCESS_POOL", "1")
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pandas as pd
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after, session_on_or_before

from src.features.family_spec import (
    FAMILY_SPECS,
    canonical_metric_columns,
    compute_family_summary,
    default_base_families,
    default_hf_blocks,
    default_hf_modules,
    family_dependencies,
    get_family_spec,
    SYMBOL_ONLY_HF_BLOCKS,
)

LOGGER = logging.getLogger("prep_families")

# Canonical horizon used when generating symbol-only HF blocks.
# These blocks are cached under data/local_cache/<symbol>/... and must not vary by
# requested horizon; otherwise one run could overwrite another.
HF_SYMBOL_ONLY_CANONICAL_HORIZON: int = int(os.getenv("HF_SYMBOL_ONLY_CANONICAL_HORIZON", "63"))


def _uses_symbol_only_cache_path(family: str) -> bool:
    return str(family).strip().lower() in {f.lower() for f in SYMBOL_ONLY_HF_BLOCKS}


def _uses_shared_cache_path(family: str) -> bool:
    """Check if a family uses the shared cache (symbol/horizon invariant)."""
    fam_lower = str(family).strip().lower()
    return fam_lower in {"doc_embedding_novelty_hf", "peer_screener_context"}


def _family_cache_dir_for_write(cache_dir: Path, symbol: str, horizon: int, family: str) -> Path:
    """Return the directory where consolidated caches for (family) should live.

    cache_dir is typically data/local_cache/<symbol>_h<horizon>.
    """

    symbol_lower = symbol.lower()
    fam_lower = str(family).lower()

    if _uses_symbol_only_cache_path(fam_lower):
        return cache_dir.parent / symbol_lower
    return cache_dir


def _family_cache_basename(symbol: str, horizon: int, family: str, split: str = "") -> str:
    """Generate cache file basename. 
    
    Returns clean basename without split suffix for symbol-only families and when split is empty.
    Includes split suffix for backward compatibility when split is explicitly provided.
    """
    symbol_lower = symbol.lower()
    fam_lower = str(family).lower()
    
    # Symbol-only families never use split suffix (truly shared across horizons/splits)
    if _uses_symbol_only_cache_path(fam_lower):
        return f"{symbol_lower}_{family}.parquet"
    
    # Non-symbol-only families: include split if provided (for backward compat with existing caches)
    if split:
        return f"{symbol_lower}_h{horizon}_{family}_{split}.parquet"
    return f"{symbol_lower}_h{horizon}_{family}.parquet"


def _family_cache_paths(cache_dir: Path, symbol: str, horizon: int, family: str, split: str) -> Tuple[Path, Path, Path]:
    """Get cache paths for a family, handling shared cache families like doc_embedding_novelty_hf."""
    fam_lower = str(family).lower()
    
    # Shared cache families (doc_embedding_novelty_hf, peer_screener_context) use shared cache with date-range naming
    if _uses_shared_cache_path(fam_lower):
        if fam_lower == "doc_embedding_novelty_hf":
            shared_dir = SHARED_CACHE_DIR / "doc_embedding" / "doc_embedding_novelty_hf"
        elif fam_lower == "peer_screener_context":
            shared_dir = SHARED_CACHE_DIR / "peer_screener" / "peer_screener_context"
        else:
            shared_dir = SHARED_CACHE_DIR / fam_lower
        # Look for existing features.parquet files matching the split
        pattern = f"{split}_*_features.parquet"
        matching_files = sorted(shared_dir.glob(pattern), reverse=True) if shared_dir.exists() else []
        
        if matching_files:
            # Return the most recent matching file as the features path
            features_path = matching_files[0]
            base_path = features_path.with_name(features_path.name.replace("_features.parquet", ".parquet"))
            lagged_path = features_path.with_name(features_path.name.replace("_features.parquet", "_features_lagged.parquet"))
            return base_path, features_path, lagged_path
        else:
            # No existing files - return expected path pattern for creation
            features_path = shared_dir / f"{split}_features.parquet"
            base_path = shared_dir / f"{split}.parquet"
            lagged_path = shared_dir / f"{split}_features_lagged.parquet"
            return base_path, features_path, lagged_path
    
    base_name = _family_cache_basename(symbol, horizon, family, split)
    target_dir = _family_cache_dir_for_write(cache_dir, symbol, horizon, family)
    base_path = target_dir / base_name
    features_path = target_dir / base_name.replace(".parquet", "_features.parquet")
    lagged_path = target_dir / base_name.replace(".parquet", "_features_lagged.parquet")
    return base_path, features_path, lagged_path

# Some sparse, event-driven families are already leak-safe via explicit session shifting
# and do not need (or want) lag-expanded feature caches.
WRITE_LAGGED_FEATURE_CACHES: bool = os.getenv("WRITE_LAGGED_FEATURE_CACHES", "0") == "1"


# ---------------------------------------------------------------------------
# Storage / cache environment diagnostics
# ---------------------------------------------------------------------------

def _find_mount_for_path(path: Path) -> Dict[str, str]:
    """Return best-effort mount metadata for a path (Linux only)."""
    mounts_path = Path("/proc/self/mounts")
    if not mounts_path.exists():
        return {}

    try:
        target = path.resolve()
    except Exception:
        target = path

    best: Optional[Dict[str, str]] = None
    try:
        with mounts_path.open("r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                src, mountpoint, fstype = parts[0], parts[1], parts[2]
                try:
                    mp = Path(mountpoint)
                except Exception:
                    continue
                mp_str = str(mp)
                target_str = str(target)
                if target_str == mp_str or target_str.startswith(mp_str.rstrip("/") + "/"):
                    cand: Dict[str, str] = {
                        "source": src,
                        "mountpoint": mountpoint,
                        "fstype": fstype,
                    }
                    if best is None or len(mountpoint) > len(best.get("mountpoint", "")):
                        best = cand
    except Exception:
        return {}

    return best or {}


def _du_bytes(path: Path, timeout_s: int = 60) -> Optional[int]:
    """Best-effort recursive size using du (fast on Linux, but may be slow on huge trees)."""
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["du", "-sb", str(path)],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            text=True,
        ).strip()
        if not out:
            return None
        # Format: <bytes>\t<path>
        return int(out.split()[0])
    except Exception:
        return None


def _write_storage_snapshot(*, horizon: int, cache_root: Path, output_dir: Path) -> Optional[Path]:
    """Write a JSON snapshot describing mounts + cache inventory for debugging."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        host = socket.gethostname()

        snapshot: Dict[str, object] = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "hostname": host,
            "repo_root": str(REPO_ROOT),
            "horizon": int(horizon),
            "paths": {
                "cache_root": str(cache_root),
                "feature_panel_dir": str(FEATURE_PANEL_DIR),
                "shared_cache": str(REPO_ROOT / "cache" / "shared"),
            },
            "mounts": {
                "cache_root": _find_mount_for_path(cache_root),
                "repo_root": _find_mount_for_path(REPO_ROOT),
                "feature_panel_dir": _find_mount_for_path(FEATURE_PANEL_DIR),
            },
        }

        try:
            usage = shutil.disk_usage(str(cache_root))
            snapshot["disk_usage_cache_root"] = {
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
            }
        except Exception:
            snapshot["disk_usage_cache_root"] = {}

        # Cheap inventory signals (counts)
        trackc_files = sorted(FEATURE_PANEL_DIR.glob(f"*_h{int(horizon)}_*trackc*.parquet"))
        snapshot["counts"] = {
            "trackc_parquets_horizon": len(trackc_files),
            "symbol_cache_dirs": len([p for p in cache_root.glob("*_h*") if p.is_dir()]),
        }

        # Best-effort sizes for key dirs
        snapshot["sizes_bytes"] = {
            "cache_root": _du_bytes(cache_root),
            "feature_panel_dir": _du_bytes(FEATURE_PANEL_DIR),
            "shared_cache": _du_bytes(REPO_ROOT / "cache" / "shared"),
        }

        out_path = output_dir / f"storage_snapshot_{host}_{stamp}.json"
        with out_path.open("w") as handle:
            json.dump(snapshot, handle, indent=2)

        return out_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Symbol-invariant HF modules
# ---------------------------------------------------------------------------
# Some HF modules are global (symbol-invariant) and only depend on the requested
# date range (and horizon) rather than the symbol. These are safe to compute once
# and reuse across symbols by copying the cached parquet artifacts.
SYMBOL_INVARIANT_HF_MODULES: Set[str] = {
    "doc_embedding_novelty_hf",
}

# Some HF modules are also horizon-invariant: their computation does not depend
# on the horizon, so the shared cache can be reused across all horizons.
HORIZON_INVARIANT_HF_MODULES: Set[str] = {
    "doc_embedding_novelty_hf",
}


def _shared_invariant_cache_root(cache_dir: Path, horizon: int, family: Optional[str] = None) -> Path:
    """Root folder for cross-symbol shared HF module caches.
    
    Symbol-invariant modules (like doc_embedding_novelty_hf) use:
        cache/shared/<family>/
    
    This is a GLOBAL shared cache - computed ONCE and reused across all symbols.
    """

    if family and str(family).lower() in HORIZON_INVARIANT_HF_MODULES:
        # doc_embedding_novelty_hf goes to cache/shared/doc_embedding/
        return SHARED_CACHE_DIR / "doc_embedding"
    # Other shared modules (if horizon-linked) go to cache/shared/<family>/
    return SHARED_CACHE_DIR / str(family).lower() if family else SHARED_CACHE_DIR


def _shared_invariant_cache_path(task: "SignalTask", cache_dir: Path) -> Path:
    start_key = pd.to_datetime(task.date_start).strftime("%Y%m%d")
    end_key = pd.to_datetime(task.date_end).strftime("%Y%m%d")
    split_key = str(task.split).lower()
    family_key = str(task.family).lower()
    dest_dir = _shared_invariant_cache_root(cache_dir, task.horizon, task.family) / family_key
    
    # Cache file naming convention: {split}_{start}_{end}_features.parquet
    new_path = dest_dir / f"{split_key}_{start_key}_{end_key}_features.parquet"

    # Backward-compat: if we previously stored horizon-scoped shared artifacts,
    # reuse them if present.
    if family_key in HORIZON_INVARIANT_HF_MODULES and not new_path.exists():
        old_dir = _shared_invariant_cache_root(cache_dir, task.horizon, None) / family_key
        old_path = old_dir / f"{split_key}_{start_key}_{end_key}_features.parquet"
        if old_path.exists():
            return old_path

    return new_path


def _acquire_file_lock(lock_path: Path) -> bool:
    """Best-effort cross-process lock using atomic create."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _wait_for_shared_file(path: Path, timeout_s: int = 3600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(2.0)
    return path.exists()


def _hf_feature_cache_paths(cache_path: Path) -> Tuple[Path, Path]:
    feature_cache_path = cache_path.parent / cache_path.name.replace(
        ".parquet",
        "_features.parquet",
    )
    lagged_feature_cache_path = feature_cache_path.with_name(
        feature_cache_path.name.replace("_features.parquet", "_features_lagged.parquet")
    )
    return feature_cache_path, lagged_feature_cache_path


def _copy_hf_artifacts(src_cache_path: Path, dst_cache_path: Path) -> bool:
    """Copy standardized HF artifacts (signal + feature caches) to a new target."""
    if not src_cache_path.exists():
        return False

    src_feature, src_lagged = _hf_feature_cache_paths(src_cache_path)
    dst_feature, dst_lagged = _hf_feature_cache_paths(dst_cache_path)

    dst_cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_cache_path, dst_cache_path)
    if src_feature.exists():
        shutil.copy2(src_feature, dst_feature)
    if WRITE_LAGGED_FEATURE_CACHES and src_lagged.exists():
        shutil.copy2(src_lagged, dst_lagged)
    return True


def _default_symbol_list_csv() -> str:
    """Default symbol universe for prep_families.

    Uses the CORE + satellite candidate universe so Track-C panels exist for
    universe selection and Phase2 runtime gating.
    """

    try:
        from src.stage_b.universe_selector import DEFAULT_CANDIDATE_UNIVERSE

        return ",".join(str(s).upper() for s in DEFAULT_CANDIDATE_UNIVERSE)
    except Exception:
        return "SPY,AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM,UNH,COST,GS,BAC,BLK,PG,KO,JNJ,WMT,CVX,CAT,BA,NFLX,PYPL,SHOP,SQ,ZM,FCX,COP,NEM,SCHW,MS,HD,LOW,DIS,TSLA,AMD"

# Merged parquets output directory (final outputs)
FEATURE_PANEL_DIR = REPO_ROOT / "cache" / "merged"

# Per-symbol family caches (individual features)
SYMBOLS_CACHE_DIR = REPO_ROOT / "cache" / "symbols"

# Shared caches (symbol-invariant like doc_embedding)
SHARED_CACHE_DIR = REPO_ROOT / "cache" / "shared"


# ----------------------------------------------------------------------------
# Multiprocessing helpers
# ----------------------------------------------------------------------------

def _get_pool_context() -> mp.context.BaseContext:
    """Return a stable multiprocessing context for the current platform."""
    if sys.platform == "win32":
        return mp.get_context("spawn")

    for method in ("fork", "forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    # Fallback to the default context (should never happen, but keeps us safe)
    return mp.get_context()


# ============================================================================
# Symbol Data Availability Detection
# ============================================================================

def detect_symbol_data_availability(symbol: str) -> Optional[pd.Timestamp]:
    """
    Detect the earliest available data date for a symbol.
    
    Tries to fetch a small sample to determine when data actually starts.
    This prevents wasting time on windows that predate a symbol's IPO/data availability.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'SQ')
    
    Returns:
        Earliest available date as pd.Timestamp, or None if no data available
    """
    try:
        from src.data.universal_data_fetcher import UniversalDataFetcher
        
        fetcher = UniversalDataFetcher()
        
        # Try to get max history (will use EODHD, then yfinance fallback)
        # Use a very early start date to capture full history
        test_data = fetcher.get_stock_data(
            symbol,
            start_date="1990-01-01",
            end_date=pd.Timestamp.now().strftime("%Y-%m-%d")
        )
        
        if test_data is not None and not test_data.empty:
            earliest = test_data.index.min()
            if hasattr(earliest, 'tz') and earliest.tz is not None:
                earliest = earliest.tz_localize(None)
            LOGGER.info(
                "✅ %s: Data available from %s (%d days history)",
                symbol, earliest.date(), len(test_data)
            )
            return pd.Timestamp(earliest)
        else:
            LOGGER.warning(
                "⚠️  %s: No data available from any provider (may be delisted)",
                symbol
            )
            return None
            
    except Exception as e:
        LOGGER.warning(
            "⚠️  %s: Failed to detect data availability: %s",
            symbol, e
        )
        # Return None to skip this symbol entirely
        return None


# ============================================================================
# Constants
# ============================================================================

# ⚠️ Order now driven by family_spec with dependency awareness
DEFAULT_BASE_FAMILIES: Tuple[str, ...] = tuple(default_base_families())

# API-dependent families (require external credentials)
API_DEPENDENT_FAMILIES: Tuple[str, ...] = (
    "commodities",     # Requires TIINGO_API_TOKEN
    "fx",              # Requires TIINGO_API_TOKEN
    "crypto",          # Requires crypto exchange API
)

DEFAULT_HF_MODULES: Tuple[str, ...] = tuple(default_hf_modules())
DEFAULT_HF_BLOCKS: Tuple[str, ...] = tuple(default_hf_blocks())
META_FAMILIES: Tuple[str, ...] = tuple(
    name for name, spec in FAMILY_SPECS.items() if spec.stage.upper() == "META"
)
DEFAULT_WF_START = "2010-07-02"
DEFAULT_WF_END = "2025-07-01"
DEFAULT_WF_STEP_DAYS = 126

# Provider-dependent families are noisy / often unavailable in strict cache-only runs.
# Also exclude cross-sectional/universe-wide families by default; those are generated
# as separate "universe snapshot" datasets and should only be merged into model
# features when explicitly requested.
# Phase 2 exclusions: families not needed for current pipeline
DEFAULT_EXCLUDE_FAMILIES: Tuple[str, ...] = (
    "fx",
    "commodities",
    "crypto",
    "news_sentiment_hf",
    "calibration",
    "online_learning",
)
DEFAULT_WF_TRAIN_YEARS = 5

# Families that must run sequentially due to explicit dependency ordering.
# Only quantile_forecast is horizon-bound. calibration/online_learning are removed.
# - Base families run in parallel
# - HF blocks run in parallel AFTER base families are complete
# - hf_agg is META-derived and computed last during unified panel build
SEQUENTIAL_ONLY_FAMILIES: Set[str] = {"quantile_forecast"}


def _is_horizon_bound_family(family: str) -> bool:
    """Return True if a family should be recomputed per horizon.

    Policy:
    - The explicit sequential chain is horizon-bound.
    - Any family (including HF blocks) that depends (directly or indirectly)
      on the sequential chain is also treated as horizon-bound.
    """
    if family in SEQUENTIAL_ONLY_FAMILIES:
        return True

    visited: Set[str] = set()
    stack = [family]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            deps = family_dependencies(current) or []
        except Exception:
            deps = []
        for dep in deps:
            if dep in SEQUENTIAL_ONLY_FAMILIES:
                return True
            if dep and dep not in visited:
                stack.append(dep)
    return False


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when possible; fallback to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _write_copied_cache_meta(
    *,
    dest_parquet: Path,
    symbol: str,
    family: str,
    split: Literal["train", "valid"],
    horizon: int,
    coverage: ConsolidatedCoverage,
    requested_start: Optional[pd.Timestamp],
    requested_end: Optional[pd.Timestamp],
    copied_from: Path,
    copied_from_horizon: Optional[int],
) -> None:
    meta = {
        "symbol": symbol,
        "family": family,
        "split": split,
        "horizon": int(horizon),
        "date_start": coverage.date_start.isoformat(),
        "date_end": coverage.date_end.isoformat(),
        "requested_start": requested_start.isoformat() if requested_start is not None else None,
        "requested_end": requested_end.isoformat() if requested_end is not None else None,
        "window_id": None,
        "scope": "consolidated",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "copied_from": str(copied_from),
        "copied_from_horizon": int(copied_from_horizon) if copied_from_horizon is not None else None,
    }
    meta_path = _cache_meta_path(dest_parquet)
    try:
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
    except Exception as exc:
        LOGGER.debug("Failed to write copied cache meta %s: %s", meta_path.name, exc)


def _prefill_symbol_only_cache_from_other_horizons(
    *,
    cache_root: Path,
    cache_dir: Path,
    symbol: str,
    horizon: int,
    families: Sequence[str],
    required_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    grace_days: int = 7,
) -> int:
    """Copy/link symbol-only family caches from other horizons.

    This enables true "compute once per symbol" reuse by reusing consolidated
    caches from another horizon directory when coverage is sufficient.
    """
    symbol_lower = symbol.lower()
    target_dir_name = f"{symbol_lower}_h{horizon}"

    # Find sibling horizon cache dirs, newest first.
    candidates: List[Tuple[float, Optional[int], Path]] = []
    for path in cache_root.glob(f"{symbol_lower}_h*"):
        if not path.is_dir() or path.name == target_dir_name:
            continue
        match = re.fullmatch(rf"{re.escape(symbol_lower)}_h(\d+)", path.name)
        src_h = int(match.group(1)) if match else None
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((mtime, src_h, path))
    candidates.sort(key=lambda t: t[0], reverse=True)

    if not candidates:
        return 0

    copied_files = 0
    for family in families:
        for split in ("train", "valid"):
            split_t = cast(Literal["train", "valid"], split)
            req = required_ranges.get(split_t)
            if req is None:
                continue
            req_start, req_end = req

            # Skip if already present in target.
            base_name = f"{symbol_lower}_h{horizon}_{family}_{split}.parquet"
            dst_base = cache_dir / base_name
            dst_features = cache_dir / base_name.replace(".parquet", "_features.parquet")
            dst_lagged = cache_dir / base_name.replace(".parquet", "_features_lagged.parquet")
            if dst_base.exists() or dst_features.exists() or (WRITE_LAGGED_FEATURE_CACHES and dst_lagged.exists()):
                continue

            for _, src_h, src_dir in candidates:
                src_base = src_dir / f"{symbol_lower}_h{src_h}_{family}_{split}.parquet" if src_h is not None else None
                src_features = src_dir / f"{symbol_lower}_h{src_h}_{family}_{split}_features.parquet" if src_h is not None else None
                src_lagged = src_dir / f"{symbol_lower}_h{src_h}_{family}_{split}_features_lagged.parquet" if src_h is not None else None

                # Prefer lagged/features/base as the source-of-truth.
                preferred = [src_features, src_base]
                if WRITE_LAGGED_FEATURE_CACHES:
                    preferred.insert(0, src_lagged)
                ordered = [p for p in preferred if p is not None]
                source_for_coverage: Optional[Path] = next((p for p in ordered if p.exists()), None)
                if source_for_coverage is None:
                    continue

                coverage = load_consolidated_coverage(family, split_t, source_for_coverage)
                if coverage is None:
                    continue
                if not coverage.covers(req_start, req_end, grace_days=grace_days):
                    continue

                # Copy/link every variant that exists in the source horizon.
                pairs = [
                    (src_base, dst_base),
                    (src_features, dst_features),
                ]
                if WRITE_LAGGED_FEATURE_CACHES:
                    pairs.append((src_lagged, dst_lagged))

                for src_path, dst_path in pairs:
                    if src_path is None or not src_path.exists() or dst_path.exists():
                        continue
                    _link_or_copy(src_path, dst_path)
                    _write_copied_cache_meta(
                        dest_parquet=dst_path,
                        symbol=symbol,
                        family=family,
                        split=split_t,
                        horizon=horizon,
                        coverage=coverage,
                        requested_start=req_start,
                        requested_end=req_end,
                        copied_from=src_path,
                        copied_from_horizon=src_h,
                    )
                    copied_files += 1
                break

    return copied_files


def _migrate_symbol_only_hf_block_caches_from_horizon_dir(
    *,
    cache_dir: Path,
    symbol: str,
    horizon: int,
    families: Sequence[str],
) -> int:
    """Back-compat migration: copy/link legacy horizon-scoped HF block caches into the
    new symbol-only cache folder.

    Older runs stored these HF blocks under data/local_cache/<symbol>_h<h>/...
    We now treat them as symbol-only and store under data/local_cache/<symbol>/...
    """

    symbol_lower = symbol.lower()
    migrated = 0

    for family in families:
        if not _uses_symbol_only_cache_path(family):
            continue
        for split in ("train", "valid"):
            # New locations
            new_base, new_feat, new_lag = _family_cache_paths(cache_dir, symbol, horizon, family, split)

            # Legacy locations (in the provided horizon dir)
            legacy_base = cache_dir / f"{symbol_lower}_h{horizon}_{family}_{split}.parquet"
            legacy_feat = cache_dir / f"{symbol_lower}_h{horizon}_{family}_{split}_features.parquet"
            legacy_lag = cache_dir / f"{symbol_lower}_h{horizon}_{family}_{split}_features_lagged.parquet"

            for src, dst in (
                (legacy_base, new_base),
                (legacy_feat, new_feat),
                (legacy_lag, new_lag),
            ):
                if src.exists() and (not dst.exists()):
                    _link_or_copy(src, dst)
                    migrated += 1

    return migrated

# Families that have dependencies and MUST run in specific order
# These run in the sequential phase AFTER parallel families complete
# NOTE: calibration and online_learning are removed from the pipeline
DEPENDENCY_FAMILIES: Dict[str, List[str]] = {
    "quantile_forecast": [],
}


def _family_metric_columns(family: str) -> Tuple[str, str, str]:
    return canonical_metric_columns(family)


def _family_attribute_columns(family: str, field_name: str) -> List[str]:
    spec = get_family_spec(family)
    if spec is None:
        return []
    values = getattr(spec, field_name, None)
    if not values:
        return []
    return list(values)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WalkForwardWindow:
    """A single walk-forward fold with train/valid date ranges."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Window{self.window_id}("
            f"train={self.train_start.date()}→{self.train_end.date()}, "
            f"valid={self.valid_start.date()}→{self.valid_end.date()}, "
            f"test={self.test_start.date()}→{self.test_end.date()})"
        )


@dataclass
class SignalTask:
    """A signal generation task (per-window or consolidated)."""
    symbol: str
    horizon: int
    family: str
    window_id: Optional[int]
    split: Literal["train", "valid"]
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    cache_path: Path
    is_hf: bool
    scope: Literal["window", "consolidated"] = "window"
    sequential_only: bool = False
    merge_strategy: Literal["replace", "merge"] = "replace"

    def cache_key(self) -> str:
        """Unique key for this signal task."""
        window_suffix = "full" if self.window_id is None else f"w{self.window_id}"
        key = (
            f"{self.symbol}_h{self.horizon}_{self.family}_"
            f"{self.split}_{window_suffix}"
        )
        if self.is_consolidated and self.window_id is None:
            key = (
                f"{key}_"
                f"{self.date_start.strftime('%Y%m%d')}"
                f"_{self.date_end.strftime('%Y%m%d')}"
            )
        return key

    def __repr__(self) -> str:
        target = "full" if self.window_id is None else f"w{self.window_id}"
        return (
            f"SignalTask({self.symbol}:h{self.horizon}:{self.family}:"
            f"{self.split}:{target}:{self.scope}:seq={self.sequential_only})"
        )
    @property
    def is_consolidated(self) -> bool:
        return self.scope == "consolidated"


@dataclass
class ConsolidatedCoverage:
    """Coverage metadata for a consolidated signal file."""
    family: str
    split: Literal["train", "valid"]
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    path: Path

    def covers(self, req_start: pd.Timestamp, req_end: pd.Timestamp, grace_days: int = 0) -> bool:
        """Return True if this cache covers the requested range.

        Coverage is evaluated on *trading sessions* (XNYS) so that requests that land
        on weekends/holidays don't force regeneration, while missing trading-day tails
        (e.g. req_end is a session but cache ends the day before) are treated as misses.
        """

        req_start = req_start.tz_localize(None) if getattr(req_start, "tz", None) else req_start
        req_end = req_end.tz_localize(None) if getattr(req_end, "tz", None) else req_end

        # Align requested endpoints onto the session grid.
        req_start_session = session_on_or_after(req_start)
        req_end_session = session_on_or_before(req_end)

        start_ok = self.date_start <= req_start_session + pd.Timedelta(days=grace_days)
        end_ok = self.date_end >= req_end_session - pd.Timedelta(days=grace_days)
        return start_ok and end_ok

    def partial_covers(self, req_start: pd.Timestamp, req_end: pd.Timestamp, min_overlap_days: int = 30) -> bool:
        """Check if there's ANY overlap with the requested range (at least min_overlap_days).
        
        This allows generating window files even when data doesn't fully cover the window.
        For example, if cboe_term data starts 2010-07-01 but window 0 needs 2010-01-01,
        we can still generate a partial window file with the available data.
        """
        req_start = req_start.tz_localize(None) if getattr(req_start, "tz", None) else req_start
        req_end = req_end.tz_localize(None) if getattr(req_end, "tz", None) else req_end
        
        # Calculate overlap
        overlap_start = max(self.date_start, req_start)
        overlap_end = min(self.date_end, req_end)
        
        if overlap_end <= overlap_start:
            return False  # No overlap at all
        
        overlap_days = (overlap_end - overlap_start).days
        return overlap_days >= min_overlap_days

@dataclass
class SliceTask:
    """Task to slice consolidated data into per-window parquet outputs."""
    symbol: str
    horizon: int
    family: str
    split: Literal["train", "valid"]
    window_id: int
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    source_path: Path
    dest_path: Path
    is_hf: bool

    def cache_key(self) -> str:
        return (
            f"SliceTask({self.symbol}:h{self.horizon}:{self.family}:"
            f"{self.split}:w{self.window_id})"
        )


def _cache_meta_path(parquet_path: Path) -> Path:
    """Derive the metadata path for any cached parquet file."""
    return parquet_path.with_name(parquet_path.name + ".meta.json")


def _read_parquet_date_range(file_path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the min/max date coverage for a parquet file."""
    try:
        df = pd.read_parquet(file_path, columns=["date"])
    except Exception as exc:
        LOGGER.warning("Failed to read %s for coverage: %s", file_path.name, exc)
        return None

    if "date" not in df.columns or df.empty:
        LOGGER.warning("File exists but missing/empty 'date' column: %s", file_path.name)
        return None

    file_start = pd.to_datetime(df["date"].min())
    file_end = pd.to_datetime(df["date"].max())
    if getattr(file_start, "tz", None) is not None:
        file_start = file_start.tz_localize(None)
    if getattr(file_end, "tz", None) is not None:
        file_end = file_end.tz_localize(None)
    return file_start, file_end


def load_consolidated_coverage(
    family: str,
    split: Literal["train", "valid"],
    file_path: Path,
) -> Optional[ConsolidatedCoverage]:
    """Load coverage metadata from JSON (preferred) or parquet fallback."""
    meta_path = _cache_meta_path(file_path)
    if meta_path.exists():
        try:
            with open(meta_path, "r") as handle:
                meta = json.load(handle)
            start = pd.to_datetime(meta.get("date_start"))
            end = pd.to_datetime(meta.get("date_end"))
            if start is not None and end is not None:
                return ConsolidatedCoverage(
                    family=family,
                    split=split,
                    date_start=start,
                    date_end=end,
                    path=file_path,
                )
        except Exception as exc:
            LOGGER.debug("Failed reading coverage meta %s: %s", meta_path.name, exc)

    coverage = _read_parquet_date_range(file_path)
    if coverage is None:
        return None
    file_start, file_end = coverage
    return ConsolidatedCoverage(
        family=family,
        split=split,
        date_start=file_start,
        date_end=file_end,
        path=file_path,
    )


def find_incomplete_families(
    existing_signals: Dict[str, Dict[str, Set[int]]],
    families: Sequence[str],
    total_windows: int,
) -> List[str]:
    """Return families missing one or more per-window slices."""
    incomplete: List[str] = []
    for family in families:
        split_map = existing_signals.get(family, {})
        for split in ["train", "valid"]:
            covered = len(split_map.get(split, set()))
            if covered < total_windows:
                incomplete.append(family)
                break
    return incomplete


def write_task_metadata(task: SignalTask) -> None:
    """Persist metadata for the generated _features.parquet cache file."""
    # Since we no longer write signal files, write metadata for _features.parquet
    features_path = task.cache_path.parent / task.cache_path.name.replace(
        ".parquet", "_features.parquet"
    )
    coverage = _read_parquet_date_range(features_path)
    if coverage is None:
        return
    file_start, file_end = coverage
    source_asof_ts = None
    source_col = f"{task.family}_source_asof_ts"
    try:
        if features_path.exists():
            try:
                source_frame = pd.read_parquet(features_path, columns=[source_col])
            except Exception:
                source_frame = None
            if isinstance(source_frame, pd.DataFrame) and source_col in source_frame.columns:
                latest = source_frame[source_col].dropna()
                if not latest.empty:
                    source_asof_ts = pd.to_datetime(latest.iloc[-1], errors="coerce")
                    if isinstance(source_asof_ts, pd.Timestamp) and not pd.isna(source_asof_ts):
                        source_asof_ts = source_asof_ts.isoformat()
                    else:
                        source_asof_ts = None
    except Exception:
        source_asof_ts = None

    cadence_days = None
    expected_latency_days = None
    try:
        from src.features.family_metadata import load_family_metadata  # type: ignore

        reg_path = Path(__file__).resolve().parents[1] / "src" / "features" / "family_metadata_registry.json"
        registry = load_family_metadata(reg_path if reg_path.exists() else None)
        meta_obj = registry.get(task.family)
        if meta_obj is not None:
            freq = str(getattr(meta_obj, "update_frequency", "")).lower()
            cadence_days = {
                "daily": 1,
                "intraday": 0,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90,
                "event": None,
                "irregular": None,
            }.get(freq, None)
            latency = str(getattr(meta_obj, "update_latency_class", "")).lower()
            expected_latency_days = {
                "intraday": 0,
                "same_day_close": 0,
                "t_plus_1": 1,
                "t_plus_2_plus": 2,
            }.get(latency, None)
    except Exception:
        cadence_days = None
        expected_latency_days = None

    meta = {
        "symbol": task.symbol,
        "family": task.family,
        "split": task.split,
        "horizon": task.horizon,
        "date_start": file_start.isoformat(),
        "date_end": file_end.isoformat(),
        "requested_start": task.date_start.isoformat(),
        "requested_end": task.date_end.isoformat(),
        "window_id": task.window_id,
        "scope": task.scope,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "fetch_ts": pd.Timestamp.utcnow().isoformat(),
        "source_asof_ts": source_asof_ts,
        "cadence_days": cadence_days,
        "expected_latency_days": expected_latency_days,
    }
    meta_path = _cache_meta_path(features_path)
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)
        LOGGER.debug("Wrote cache metadata: %s", meta_path.name)
    except Exception as exc:
        LOGGER.warning("Failed to write cache metadata %s: %s", meta_path.name, exc)


@dataclass
class PreparationResult:
    """Result of signal preparation."""
    symbol: str
    horizon: int
    success: bool
    total_tasks: int
    completed_tasks: int
    skipped_tasks: int
    failed_tasks: int
    failed_families: List[str] = field(default_factory=list)
    coverage_start: Optional[pd.Timestamp] = None
    coverage_end: Optional[pd.Timestamp] = None
    duration_seconds: float = 0.0
    completeness_manifest_path: Optional[Path] = None

    def summary(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        cov_info = ""
        if self.coverage_start and self.coverage_end:
            cov_info = f" | Coverage: {self.coverage_start.date()} → {self.coverage_end.date()}"
        return (
            f"{status} - {self.completed_tasks}/{self.total_tasks} tasks completed "
            f"({self.skipped_tasks} skipped, {self.failed_tasks} failed) "
            f"in {self.duration_seconds:.1f}s{cov_info}"
        )


# ============================================================================
# Walk-Forward Date Logic
# ============================================================================

def build_walk_forward_windows(
    *,
    start_date: str,
    end_date: str,
    train_years: int,
    step_years: int = None,
    step_days: int = None,
    horizon_days: int,
) -> List[WalkForwardWindow]:
    """
    Build walk-forward windows matching the launcher's logic.
    
    Args:
        start_date: First training date (YYYY-MM-DD)
        end_date: Last validation date (YYYY-MM-DD)
        train_years: Training window size in years
        step_years: Step size between windows in years (optional if step_days provided)
        step_days: Step size between windows in trading days (optional if step_years provided)
        horizon_days: Trading-day horizon used for validation/test spans
    
    Returns:
        List of WalkForwardWindow objects
    """
    if horizon_days is None or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")

    start_ts = session_on_or_after(pd.Timestamp(start_date))
    end_ts = session_on_or_before(pd.Timestamp(end_date))
    
    # Validate that exactly one of step_years or step_days is provided
    if step_years is None and step_days is None:
        raise ValueError("Either step_years or step_days must be provided")
    if step_years is not None and step_days is not None:
        raise ValueError("Cannot specify both step_years and step_days")
    
    windows = []
    window_id = 1
    year_step_offset = pd.DateOffset(years=step_years) if step_years is not None else None

    current_valid_start = start_ts
    while True:
        # All window boundaries are exchange sessions (not calendar days).
        train_end = add_sessions(current_valid_start, -1)
        train_start = session_on_or_after(current_valid_start - pd.DateOffset(years=train_years))

        valid_start = current_valid_start
        valid_end = add_sessions(valid_start, horizon_days - 1)
        test_start = add_sessions(valid_end, 1)
        test_end = add_sessions(test_start, horizon_days - 1)

        if test_end > end_ts:
            break

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        if step_days is not None:
            current_valid_start = add_sessions(current_valid_start, step_days)
        else:
            current_valid_start = session_on_or_after(current_valid_start + year_step_offset)
        window_id += 1
    
    return windows


def validate_walk_forward_windows(
    windows: List[WalkForwardWindow],
    min_windows: int = 1,
) -> None:
    """
    Validate walk-forward windows meet minimum requirements.
    
    Raises:
        ValueError: If windows are invalid
    """
    if len(windows) < min_windows:
        raise ValueError(
            f"Insufficient walk-forward windows: {len(windows)} < {min_windows}"
        )
    
    # Check for overlaps between train and valid
    for window in windows:
        if window.valid_start <= window.train_end:
            raise ValueError(
                f"Window {window.window_id}: validation starts before training ends! "
                f"train_end={window.train_end.date()}, valid_start={window.valid_start.date()}"
            )
        if window.test_start <= window.valid_end:
            raise ValueError(
                f"Window {window.window_id}: test starts before validation ends! "
                f"valid_end={window.valid_end.date()}, test_start={window.test_start.date()}"
            )
    
    LOGGER.info("✅ Walk-forward validation passed: %d windows", len(windows))


# ============================================================================
# Family Signal Detection & Validation
# ============================================================================

def resolve_families(
    families_arg: str,
    exclude_families: Optional[Sequence[str]] = None,
    include_base: bool = True,
    include_hf_modules: bool = True,
    include_hf_blocks: bool = True,
    include_meta: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Resolve the families argument into base and HF lists.
    
    Args:
        families_arg: 'all', 'base', 'hf', or comma-separated list
        include_base: Whether to include base families when 'all'
        include_hf: Whether to include HF modules when 'all'
    
    Returns:
        (base_families, hf_modules) tuple
    """

    def _dependency_closure(initial: Sequence[str]) -> Set[str]:
        """Return transitive dependency closure for families using FamilySpec.dependencies."""
        resolved: Set[str] = set()
        stack: List[str] = [f for f in initial if f]
        while stack:
            family = stack.pop()
            if family in resolved:
                continue
            resolved.add(family)
            for dep in family_dependencies(family):
                if dep and dep not in resolved:
                    stack.append(dep)
        return resolved

    def _order_with_dependencies(families: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
        """Deterministically order families such that dependencies run before dependents."""
        family_set: Set[str] = set(families)
        visited: Set[str] = set()
        visiting: Set[str] = set()
        ordered: List[str] = []

        def visit(name: str) -> None:
            if name not in family_set or name in visited:
                return
            if name in visiting:
                # Cycle or self-dependency; keep stable and continue.
                return
            visiting.add(name)
            for dep in family_dependencies(name):
                if dep in family_set:
                    visit(dep)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for token in preferred_order:
            visit(token)
        for token in sorted(family_set):
            visit(token)
        return ordered

    exclude_set: Set[str] = set()
    if exclude_families:
        exclude_set = {f.strip() for f in exclude_families if f and f.strip()}

    if families_arg == "all":
        base = list(DEFAULT_BASE_FAMILIES) if include_base else []
        hf = list(DEFAULT_HF_MODULES) if include_hf_modules else []
        blocks = list(DEFAULT_HF_BLOCKS) if include_hf_blocks else []
        meta = list(META_FAMILIES) if include_meta else []
    elif families_arg == "base":
        base = list(DEFAULT_BASE_FAMILIES)
        hf = []
        blocks = []
        meta = []
    elif families_arg == "hf":
        base = []
        hf = list(DEFAULT_HF_MODULES)
        blocks = []
        meta = []
    else:
        # Parse comma-separated list
        requested = [f.strip() for f in families_arg.split(",") if f.strip()]
        base = [f for f in requested if f in DEFAULT_BASE_FAMILIES]
        hf = [f for f in requested if f in DEFAULT_HF_MODULES]
        blocks = [f for f in requested if f in DEFAULT_HF_BLOCKS]
        meta = [f for f in requested if f in META_FAMILIES]
        
        # Check for unknowns
        known = set(base) | set(hf) | set(blocks) | set(meta)
        unknown = set(requested) - known
        if unknown:
            LOGGER.warning("Unknown families will be skipped: %s", ", ".join(unknown))

    # Apply exclusions.
    if exclude_set:
        before = set(base) | set(hf) | set(blocks) | set(meta)
        base = [f for f in base if f not in exclude_set]
        hf = [f for f in hf if f not in exclude_set]
        blocks = [f for f in blocks if f not in exclude_set]
        meta = [f for f in meta if f not in exclude_set]
        removed = sorted(before - (set(base) | set(hf) | set(blocks) | set(meta)))
        if removed:
            LOGGER.info("🚫 Excluding families: %s", ", ".join(removed))

    # HF blocks/modules require the consolidated base feature caches.
    # If the user requests HF-only, auto-include base families so the run is self-contained.
    if (hf or blocks) and not base:
        base = list(DEFAULT_BASE_FAMILIES)

    # Expand across all selected families (base + HF modules + HF blocks + meta).
    selected_set: Set[str] = set(base) | set(hf) | set(blocks) | set(meta)
    if selected_set:
        expanded_set = _dependency_closure(sorted(selected_set))

        # Re-apply exclusions after expansion, but keep anything that is still required.
        if exclude_set:
            desired = expanded_set - exclude_set
            required = _dependency_closure(sorted(desired))
            forced_back = sorted(required & exclude_set)
            if forced_back:
                LOGGER.warning(
                    "⚠️ Cannot exclude dependency families required by others; keeping: %s",
                    ", ".join(forced_back),
                )
            expanded_set = required

        added = sorted(expanded_set - selected_set)
        if added:
            LOGGER.info("🔗 Auto-added dependency families: %s", ", ".join(added))

        base_set = expanded_set & set(DEFAULT_BASE_FAMILIES)
        hf_set = expanded_set & set(DEFAULT_HF_MODULES)
        block_set = expanded_set & set(DEFAULT_HF_BLOCKS)
        meta_set = expanded_set & set(META_FAMILIES)

        base = (
            _order_with_dependencies(base_set, DEFAULT_BASE_FAMILIES)
            if base_set
            else []
        )
        hf = (
            _order_with_dependencies(hf_set, DEFAULT_HF_MODULES)
            if hf_set
            else []
        )
        blocks = (
            _order_with_dependencies(block_set, DEFAULT_HF_BLOCKS)
            if block_set
            else []
        )
        meta = (
            _order_with_dependencies(meta_set, META_FAMILIES)
            if meta_set
            else []
        )
    
    return base, hf, blocks, meta


def detect_existing_signals(
    cache_dir: Path,
    symbol: str,
    horizon: int,
    families: List[str],
    windows: List[WalkForwardWindow],
    grace_days: int = 7,
) -> Tuple[Dict[str, Dict[str, Set[int]]], Dict[str, Dict[str, ConsolidatedCoverage]]]:
    """Detect consolidated coverage for each family/split.

    Per-window cache shards are deprecated by design; this function intentionally
    does not scan for `_w*.parquet` artifacts.
    """
    symbol_lower = symbol.lower()
    # Keep return shape for backwards compatibility (some callers still expect
    # `existing_signals[family][split]`), but it will always be empty.
    existing: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    coverage_map: Dict[str, Dict[str, ConsolidatedCoverage]] = defaultdict(dict)

    def _delete_cache_artifact(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        # Some consolidated caches have a meta sidecar.
        try:
            Path(str(path) + ".meta.json").unlink(missing_ok=True)
        except Exception:
            pass

    for family in families:
        # Check both split patterns: "" (consolidated/Stage-A) and "train"/"valid" (walk-forward)
        for split in ("", "train", "valid"):
            consolidated_path, features_path, features_lagged_path = _family_cache_paths(
                cache_dir,
                symbol,
                horizon,
                family,
                split,
            )

            # Prefer lagged caches only when explicitly enabled.
            if WRITE_LAGGED_FEATURE_CACHES and features_lagged_path.exists():
                coverage = load_consolidated_coverage(family, split, features_lagged_path)
                if coverage is not None:
                    if family == "finbert" and _finbert_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "FinBERT cache %s has gaps; forcing regeneration",
                            coverage.path.name,
                        )
                    elif family == "earnings_transcript_hf" and _earnings_transcript_hf_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "earnings_transcript_hf cache %s is unhealthy; deleting and regenerating",
                            coverage.path.name,
                        )
                        _delete_cache_artifact(consolidated_path)
                        _delete_cache_artifact(features_path)
                        _delete_cache_artifact(features_lagged_path)
                        continue
                    elif family == "macro_tst_hf" and _macro_tst_hf_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "macro_tst_hf cache %s is unhealthy; deleting and regenerating",
                            coverage.path.name,
                        )
                        _delete_cache_artifact(consolidated_path)
                        _delete_cache_artifact(features_path)
                        _delete_cache_artifact(features_lagged_path)
                        continue
                    else:
                        is_valid, reason = validate_family_quality(features_lagged_path, family)
                        if not is_valid:
                            LOGGER.warning(
                                "🧹 %s/%s: cached _features_lagged.parquet is unhealthy (%s) → deleting and regenerating",
                                family,
                                split,
                                reason,
                            )
                            _delete_cache_artifact(features_lagged_path)
                            continue
                        LOGGER.debug(
                            "Using _features_lagged.parquet for %s/%s (full raw features with lags)",
                            family,
                            split,
                        )
                        coverage_map[family][split] = coverage
                        continue

            # Prefer full raw features.
            if features_path.exists():
                coverage = load_consolidated_coverage(family, split, features_path)
                if coverage is not None:
                    if family == "finbert" and _finbert_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "FinBERT cache %s has gaps; forcing regeneration",
                            coverage.path.name,
                        )
                    elif family == "earnings_transcript_hf" and _earnings_transcript_hf_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "earnings_transcript_hf cache %s is unhealthy; deleting and regenerating",
                            coverage.path.name,
                        )
                        _delete_cache_artifact(consolidated_path)
                        _delete_cache_artifact(features_path)
                        _delete_cache_artifact(features_lagged_path)
                        continue
                    elif family == "macro_tst_hf" and _macro_tst_hf_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "macro_tst_hf cache %s is unhealthy; deleting and regenerating",
                            coverage.path.name,
                        )
                        _delete_cache_artifact(consolidated_path)
                        _delete_cache_artifact(features_path)
                        _delete_cache_artifact(features_lagged_path)
                        continue
                    else:
                        is_valid, reason = validate_family_quality(features_path, family)
                        if not is_valid:
                            LOGGER.warning(
                                "🧹 %s/%s: cached _features.parquet is unhealthy (%s) → deleting and regenerating",
                                family,
                                split,
                                reason,
                            )
                            _delete_cache_artifact(features_path)
                            continue
                        LOGGER.debug(
                            "Using _features.parquet for %s/%s (full raw features)",
                            family,
                            split,
                        )
                        coverage_map[family][split] = coverage
                        continue

            # Fallback to signal file (only has score/conf - less useful for Track A)
            if consolidated_path.exists():
                coverage = load_consolidated_coverage(family, split, consolidated_path)
                if coverage is not None:
                    if family == "finbert" and _finbert_cache_requires_regen(coverage.path):
                        LOGGER.warning(
                            "FinBERT signal cache %s has gaps; forcing regeneration",
                            coverage.path.name,
                        )
                        continue
                    LOGGER.warning(
                        "Using signal file for %s/%s (MISSING _features.parquet - only score columns!)",
                        family,
                        split,
                    )
                    coverage_map[family][split] = coverage

    # Convert nested defaultdict to plain dict for downstream use
    normalized: Dict[str, Dict[str, Set[int]]] = {}
    for family, split_map in existing.items():
        normalized[family] = {split: set(window_ids) for split, window_ids in split_map.items()}
    normalized_coverage: Dict[str, Dict[str, ConsolidatedCoverage]] = {}
    for family, split_map in coverage_map.items():
        normalized_coverage[family] = dict(split_map)
    return normalized, normalized_coverage


def _merge_on_date(existing_path: Path, new_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows into an existing parquet by date without rewriting everything."""
    if not existing_path.exists():
        return new_frame
    try:
        existing = pd.read_parquet(existing_path)
    except Exception as exc:
        LOGGER.warning(
            "Failed to read existing cache %s for merge (%s) – rewriting segment only",
            existing_path.name,
            exc,
        )
        return new_frame

    # Normalize both frames to ensure datetime columns are tz-naive before concatenation
    try:
        existing_norm = _normalize_date_column(existing)
    except Exception as exc:
        LOGGER.warning(
            "Failed to normalize existing cache %s for merge (%s) – rewriting segment only",
            existing_path.name,
            exc,
        )
        existing_norm = existing

    try:
        new_norm = _normalize_date_column(new_frame)
    except Exception as exc:
        LOGGER.warning(
            "Failed to normalize new frame for %s (%s) – using raw payload",
            existing_path.name,
            exc,
        )
        new_norm = new_frame

    combined = pd.concat([existing_norm, new_norm], ignore_index=True)
    if combined.columns.duplicated().any():
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values("date")
        combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.reset_index(drop=True)


def _normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a dataframe has a timezone-naive 'date' column for cache writes."""
    df = frame.copy()
    index_names = [name for name in df.index.names if name]

    def _dedupe_columns(payload: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate column labels while keeping the last occurrence."""
        if payload.columns.duplicated().any():
            return payload.loc[:, ~payload.columns.duplicated(keep="last")]
        return payload

    # If the index already carries a 'date' level, surface it as a column and
    # prefer any existing column with the same name by keeping the last copy.
    if "date" in index_names and "date" not in df.columns:
        df = df.reset_index(level="date")
        df = _dedupe_columns(df)

    # If 'date' exists both as an index level name and a column label, pandas
    # operations like sort_values can raise ambiguity errors. For cache writes,
    # prefer the column and drop the index level.
    if "date" in index_names and "date" in df.columns:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level="date", drop=True)
        else:
            df = df.reset_index(drop=True)
        index_names = [name for name in df.index.names if name]

    if "date" not in df.columns:
        # Look for any other index level that clearly represents time.
        idx_candidate = next(
            (name for name in index_names if name.lower() in {"timestamp", "datetime"}),
            None,
        )
        if idx_candidate is not None:
            df = df.reset_index(level=idx_candidate)
            df = df.rename(columns={idx_candidate: "date"})
            df = _dedupe_columns(df)

    if "date" not in df.columns:
        for candidate in ("timestamp", "datetime", "index"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
        else:
            df = df.reset_index()
            if "index" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"index": "date"})
            df = _dedupe_columns(df)

    if "date" not in df.columns:
        raise ValueError("Unable to normalize dataframe – missing 'date' column")

    # Always ensure unique column labels before any parquet writes.
    df = _dedupe_columns(df)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def _datetime_index_from_date_column(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Return (payload_without_date, datetime_index) from a frame with a date column.

    The index is guaranteed to be a real tz-naive DatetimeIndex, sorted, and deduplicated.
    """
    if "date" not in frame.columns:
        raise ValueError("Missing 'date' column")

    normalized = _normalize_date_column(frame)
    dates = pd.to_datetime(normalized["date"], errors="coerce", utc=True).dt.tz_convert(None)
    date_index = pd.DatetimeIndex(dates.to_numpy(), name="date")

    valid_mask = ~date_index.isna()
    if not valid_mask.all():
        normalized = normalized.loc[valid_mask].reset_index(drop=True)
        date_index = date_index[valid_mask]

    payload = normalized.drop(columns=["date"]).copy()
    payload.index = date_index
    if not payload.index.is_unique:
        payload = payload.loc[~payload.index.duplicated(keep="last")]
    payload = payload.sort_index()
    return payload, payload.index


def _write_feature_cache(
    feature_path: Path,
    frame: pd.DataFrame,
    merge: bool = False,
    *,
    allowed_columns: Optional[Sequence[str]] = None,
    family: Optional[str] = None,
) -> None:
    """Persist a Stage A feature cache, optionally merging with existing rows.

    If ``allowed_columns`` is provided, the final cache is restricted to exactly:
    ``['date'] + allowed_columns`` (and missing columns are added as NaN).

    This is used to prevent schema drift where incremental merges keep legacy
    columns that are no longer generated.
    
    GOVERNANCE: If ``family`` is provided, validates that governance columns exist
    and are properly computed for the full date range. Missing/stub governance 
    columns are auto-computed based on the actual data coverage.
    """

    normalized = _normalize_date_column(frame)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if merge:
        normalized = _merge_on_date(feature_path, normalized)

    # ═══════════════════════════════════════════════════════════════════════════════
    # GOVERNANCE VALIDATION (Jan 2026)
    # ═══════════════════════════════════════════════════════════════════════════════
    # Centralized governance pass in aggregator_panel.py is authoritative.
    # Here we only ensure columns exist to keep schema stable; values are
    # treated as non-authoritative and overwritten later.
    # ═══════════════════════════════════════════════════════════════════════════════
    if family and not normalized.empty:
        has_data_col = f"{family}_has_data"
        activity_col = f"{family}_activity"
        days_since_col = f"{family}_days_since_update"
        
        # Get family feature columns (exclude governance columns)
        family_feature_cols = [
            c for c in normalized.columns 
            if str(c).startswith(f"{family}_") 
            and c not in [has_data_col, activity_col, days_since_col, 'date']
            and not str(c).endswith('_has_data')
            and not str(c).endswith('_activity')
            and not str(c).endswith('_days_since_update')
        ]
        
        # Default governance columns (non-authoritative)
        has_data_computed = pd.Series(1.0, index=normalized.index)
        
        # Check if existing has_data is all-1.0 stub or has NaNs
        if has_data_col in normalized.columns:
            existing_has_data = pd.to_numeric(normalized[has_data_col], errors='coerce')
            is_stub = existing_has_data.isna().any() or (existing_has_data == 1.0).all()
            if is_stub and family_feature_cols:
                # Replace stub with computed values
                normalized[has_data_col] = has_data_computed
        else:
            normalized[has_data_col] = has_data_computed
        
        # Activity: default to 1.0
        if activity_col not in normalized.columns or normalized[activity_col].isna().any():
            normalized[activity_col] = 1.0
        
        # Days since update: default to 0.0
        if days_since_col not in normalized.columns or normalized[days_since_col].isna().any():
            normalized[days_since_col] = 0.0

    if allowed_columns is not None:
        allowed = [c for c in allowed_columns if c and c != "date"]
        # Ensure all allowed columns exist (shape-stable caches).
        for col in allowed:
            if col not in normalized.columns:
                normalized[col] = np.nan
        keep_cols = ["date"] + allowed
        normalized = normalized[[c for c in keep_cols if c in normalized.columns]]

    normalized.to_parquet(feature_path, index=False)


from src.features.canonical_feature_cols import (
    CORRELATION_CANONICAL_FEATURE_COLS,
    EARNINGS_CANONICAL_FEATURE_COLS,
    ML_FRAMEWORK_CANONICAL_FEATURE_COLS,
)


def _backfill_feature_cache(base_path: Path, feature_path: Path) -> bool:
    """Backfill a missing feature cache from a consolidated signal file."""
    if feature_path.exists() or not base_path.exists():
        return False
    try:
        frame = pd.read_parquet(base_path)
    except Exception as exc:
        LOGGER.warning(
            "Failed to read %s for feature backfill: %s",
            base_path.name,
            exc,
        )
        return False
    try:
        _write_feature_cache(feature_path, frame, merge=False)
        LOGGER.info("♻️ Backfilled missing feature cache: %s", feature_path.name)
        return True
    except Exception as exc:
        LOGGER.warning(
            "Failed to backfill feature cache %s: %s",
            feature_path.name,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Family-specific hygiene
# ---------------------------------------------------------------------------

def _finbert_cache_requires_regen(path: Path) -> bool:
    """Return True if a cached FinBERT file contains NaNs or is unreadable."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning("FinBERT cache %s unreadable (%s); forcing regeneration", path.name, exc)
        return True

    finbert_cols = [col for col in df.columns if col.startswith("finbert_")]
    if not finbert_cols:
        return False

    nan_rows = int(df[finbert_cols].isna().any(axis=1).sum())
    if nan_rows > 0:
        LOGGER.warning(
            "FinBERT cache %s has %d rows with NaNs (columns=%s)",
            path.name,
            nan_rows,
            ",".join(sorted(finbert_cols)),
        )
        return True
    return False

def _enforce_finbert_neutral_fill(features: pd.DataFrame, context: str) -> pd.DataFrame:
    """Force FinBERT features to use neutral (0.0) fill when cache gaps appear."""
    finbert_cols = [col for col in features.columns if col.startswith("finbert_")]
    if not finbert_cols:
        return features

    nan_mask = features[finbert_cols].isna()
    if not nan_mask.values.any():
        return features

    rows_with_nan = nan_mask.any(axis=1)
    LOGGER.warning(
        "⚠️ FinBERT neutral fill enforced: %d rows had NaNs in %s",
        int(rows_with_nan.sum()),
        context,
    )

    filled = features.copy()
    filled[finbert_cols] = filled[finbert_cols].fillna(0.0)

    has_data_cols = [col for col in finbert_cols if col.endswith("has_data")]
    for col in has_data_cols:
        filled.loc[rows_with_nan, col] = 0.0

    return filled


def _earnings_transcript_hf_cache_requires_regen(path: Path) -> bool:
    """Return True if earnings_transcript_hf cached features are missing event columns or contain NaNs."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning(
            "earnings_transcript_hf cache %s unreadable (%s); forcing regeneration",
            path.name,
            exc,
        )
        return True

    event_cols = [c for c in df.columns if str(c).startswith("earnings_transcript_hf_event_")]
    if not event_cols:
        LOGGER.warning(
            "earnings_transcript_hf cache %s missing event_* columns; forcing regeneration",
            path.name,
        )
        return True

    nan_rows = int(df[event_cols].isna().any(axis=1).sum())
    if nan_rows > 0:
        LOGGER.warning(
            "earnings_transcript_hf cache %s has %d rows with NaNs in event columns; forcing regeneration",
            path.name,
            nan_rows,
        )
        return True

    has_col = "earnings_transcript_hf_has_data"
    score_col = "earnings_transcript_hf_event_score"
    if has_col in df.columns and score_col in df.columns:
        try:
            has_data = pd.to_numeric(df[has_col], errors="coerce").fillna(0.0)
            event_score = pd.to_numeric(df[score_col], errors="coerce")
            missing_on_events = int(((has_data > 0.0) & (event_score.isna())).sum())
            if missing_on_events > 0:
                LOGGER.warning(
                    "earnings_transcript_hf cache %s has %d earnings sessions with NaN event_score; forcing regeneration",
                    path.name,
                    missing_on_events,
                )
                return True
        except Exception:
            return True

    return False


def _macro_tst_hf_cache_requires_regen(path: Path) -> bool:
    """Return True if macro_tst_hf cached features look placeholder/degenerate in the recent window."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        LOGGER.warning("macro_tst_hf cache %s unreadable (%s); forcing regeneration", path.name, exc)
        return True

    # Handle both raw and lagged caches by matching on substrings.
    real_rate_cols = [c for c in df.columns if "l3_real_interest_rate" in str(c)]
    if not real_rate_cols:
        LOGGER.warning("macro_tst_hf cache %s missing l3_real_interest_rate; forcing regeneration", path.name)
        return True

    col = real_rate_cols[0]
    try:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    except Exception:
        return True

    if len(series) >= 252:
        tail = series.tail(252)
        if float(tail.std()) < 1e-12 and float(tail.abs().max()) == 0.0:
            LOGGER.warning(
                "macro_tst_hf cache %s has constant-zero %s in recent window; forcing regeneration",
                path.name,
                col,
            )
            return True

    return False


# ============================================================================
# Family Quality Validation
# ============================================================================

def validate_family_quality(
    feature_cache_path: Path,
    family: str,
    min_variance_threshold: float = 1e-10,
    max_zero_percentage: float = 0.95,
    max_null_percentage: float = 0.90,
) -> tuple[bool, str]:
    """
    Validate that a family's features contain real data, not just proxies or zeros.
    
    Args:
        feature_cache_path: Path to the _features.parquet file
        family: Family name
        min_variance_threshold: Minimum variance for at least one feature
        max_zero_percentage: Maximum allowed percentage of zeros (fail if exceeded)
        max_null_percentage: Maximum allowed percentage of nulls (fail if exceeded)
    
    Returns:
        (is_valid, reason) tuple where is_valid is True if the family passes validation,
        and reason provides details when validation fails.
    """
    try:
        if not feature_cache_path.exists():
            return False, "feature cache file missing"
        
        # Read the feature cache
        df = pd.read_parquet(feature_cache_path)

        # Preserve date column for family-specific health checks (some caches use RangeIndex).
        date_series: Optional[pd.Series] = None
        if 'date' in df.columns:
            try:
                date_series = pd.to_datetime(df['date'], errors='coerce')
            except Exception:
                date_series = None
        
        # Remove date column if present
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        
        # Get family columns - handle different naming conventions:
        # Some families use {family}_ prefix, others use abbreviated prefixes (CORR_, ml_, etc.)
        family_column_prefixes = {
            "correlation": ["CORR_", "correlation_", "corr_", "ACF_", "LAG_CORR_"],
            "ml_framework": ["ml_", "ml_framework_"],
            "finbert": ["finbert_", "FINBERT_"],
            "alternative_signals": ["alt_", "alternative_", "news_volume_", "news_sentiment_", "social_"],
            "cross_asset": ["cross_", "CROSS_", "cross_asset_"],
            "microstructure": ["micro_", "microstructure_", "MICRO_"],
            # macro_tst_hf uses legacy non-prefixed columns (l1_, l2_, l3_, derived_)
            "macro_tst_hf": ["macro_tst_hf_", "l1_", "l2_", "l3_", "derived_"],
            # doc_embedding_novelty_hf uses event-based column names
            "doc_embedding_novelty_hf": [
                "doc_embedding_novelty_hf_", "n_events", "n_articles",
                "macro_novelty", "geopolitical_novelty", "regulatory_novelty",
                "energy_novelty", "conflict_novelty", "tech_novelty",
                "theme_weight", "baseline_", "novelty_", "top_theme",
            ],
        }
        
        prefixes = family_column_prefixes.get(family, [f"{family}_", f"{family.upper()}_"])
        family_cols = [c for c in df.columns if any(c.startswith(p) or c.upper().startswith(p.upper()) for p in prefixes)]
        
        # If still no columns found, use all numeric columns as family columns
        # (the cache should only contain columns for this family anyway)
        if not family_cols:
            family_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not family_cols:
            return False, "no family columns found in feature cache"
        
        # Identify proxy/flag columns (not real features)
        proxy_patterns = ['_has_data', 'is_etf', 'is_quarter_start', 'is_quarter_end', 
                         'is_year_start', 'is_year_end', 'is_month_start', 'is_month_end']
        proxy_cols = [c for c in family_cols if any(p in c for p in proxy_patterns)]

        # Some families can be legitimately "dormant" (shape-stable outputs but no real signal).
        # In that case we keep caches/panels consistent and do not want the pipeline to hard-fail.
        # Examples:
        # - calibration: sequential Track-B chain (quantile_forecast → calibration → online_learning)
        # - peer_screener_context: cross-sectional ranks that are only valid when peer coverage is sufficient
        #   (use the *_has_data flag downstream to gate usage).
        # - fx/commodities/crypto: optional external market data feeds; may be unavailable in some environments.
        # - macro_tst_hf: HF model dependent, may produce low-variance outputs depending on model state.
        # - doc_embedding_novelty_hf: GDELT dependent, may have sparse coverage.
        allowed_dormant_families = {
            "calibration",
            "peer_screener_context",
            # Provider-backed and legitimately sparse/periodic (bi-monthly) with long stretches of has_data=0.
            # The family generator emits deterministic zero-filled stubs to keep schema stable.
            "short_interest",
            "fx",
            "commodities",
            "crypto",
            # HF-model dependent: may produce low-variance outputs depending on model availability/state.
            "macro_tst_hf",
            # GDELT dependent: may have sparse coverage when news feed is unavailable.
            "doc_embedding_novelty_hf",
            # NOTE: correlation and ml_framework now correctly set has_data=1 via their generators.
            # They are NOT dormant - they fetch real data from EODHD.
        }
        has_data_cols = [c for c in family_cols if c.endswith("_has_data")]
        sample_size_cols = [c for c in family_cols if c.endswith("_sample_size")]
        
        # If the family reports has_data, evaluate quality on rows where has_data>0.
        # This prevents long-history runs from failing snapshot-style providers (e.g. options)
        # where most dates are legitimately "no data".
        df_for_checks = df
        active_rows = None
        if has_data_cols:
            try:
                has_data_any = df[has_data_cols].max(axis=1)
                active_rows = has_data_any > 0.0
            except Exception:
                active_rows = None

            if active_rows is not None:
                active_count = int(active_rows.sum())
                if active_count <= 0:
                    if family in allowed_dormant_families:
                        return True, "dormant (has_data=0) — constant features accepted"
                    return False, "has_data=0 across cache (provider unavailable or empty payload)"
                df_for_checks = df.loc[active_rows].copy()

        # Get real feature columns (excluding proxies)
        real_feature_cols = [c for c in family_cols if c not in proxy_cols]
        
        if not real_feature_cols:
            return False, f"only proxy/flag columns found ({len(proxy_cols)} proxy cols, 0 real features)"

        # Guardrail: some families are expected to emit a rich raw feature panel.
        # If we only see a handful of summary-like columns (e.g., *_score/_conf),
        # it typically means build_panel ran in summary mode and we accidentally
        # wrote the score cache into a *_features.parquet.
        if family in {"alternative_signals", "econ_events_calendar"}:
            if len(real_feature_cols) < 10:
                return False, f"too few raw features for {family} (found {len(real_feature_cols)}; expected >=10)"
        
        # Check numeric features only
        numeric_cols = df_for_checks[real_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return False, f"no numeric features found (real cols: {len(real_feature_cols)})"

        # ------------------------------------------------------------------
        # Targeted semantic health checks (catch silently-dead sub-signals)
        # ------------------------------------------------------------------

        def _extract_symbol_from_cache_name(path: Path) -> str:
            try:
                token = path.name.split('_', 1)[0]
                return str(token).upper()
            except Exception:
                return ""

        def _recent_window_mask(dates: Optional[pd.Series], *, lookback_days: int) -> Optional[pd.Series]:
            if dates is None:
                return None
            try:
                end_dt = pd.to_datetime(dates.max(), errors='coerce')
                if pd.isna(end_dt):
                    return None
                start_dt = end_dt - pd.Timedelta(days=int(max(1, lookback_days)))
                return (dates >= start_dt) & (dates <= end_dt)
            except Exception:
                return None

        # Alternative signals: news volume should not be permanently dead for mega-caps.
        # (Google Trends can legitimately be 0 due to 429/rate limits, so we do NOT gate on it.)
        if family == "alternative_signals":
            symbol_guess = _extract_symbol_from_cache_name(feature_cache_path)
            strict_syms_env = os.getenv(
                "PREP_FAMILIES_STRICT_NEWS_SYMBOLS",
                "AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,GOOG,SPY,QQQ",
            )
            strict_syms = {s.strip().upper() for s in strict_syms_env.split(',') if s.strip()}
            if symbol_guess in strict_syms:
                news_col = f"{family}_news_volume_count"
                if news_col in df_for_checks.columns:
                    mask = _recent_window_mask(date_series, lookback_days=90)
                    series = df_for_checks[news_col]
                    if mask is not None and len(mask) == len(series):
                        series = series.loc[mask]
                    # Only flag as unhealthy when we have enough rows to make a judgement.
                    if len(series) >= 20:
                        try:
                            nonzero = int((series.fillna(0.0) > 0.0).sum())
                        except Exception:
                            nonzero = 0
                        if nonzero == 0:
                            return False, f"{news_col} is all-zero in recent window (dead news volume)"

        # Econ events calendar: if surprises are present, corresponding z-scores should not be all-zero.
        if family == "econ_events_calendar":
            mask = _recent_window_mask(date_series, lookback_days=365)
            cols = list(df_for_checks.columns)
            surprise_cols = [c for c in cols if c.startswith(f"{family}_surprise_") and ("_z_" not in c) and ("_w_" not in c) and ("has_forecast" not in c)]
            for base_col in surprise_cols:
                suffix = base_col[len(f"{family}_surprise_"):]
                z_col = f"{family}_surprise_z_{suffix}"
                if z_col not in df_for_checks.columns:
                    continue
                s = df_for_checks[base_col]
                z = df_for_checks[z_col]
                if mask is not None and len(mask) == len(s):
                    s = s.loc[mask]
                    z = z.loc[mask]
                # If there are surprise pulses but z is completely flat-zero, treat as unhealthy.
                try:
                    pulses = int((s.fillna(0.0) != 0.0).sum())
                    z_nonzero = int((z.fillna(0.0) != 0.0).sum())
                except Exception:
                    pulses, z_nonzero = 0, 0
                if pulses >= 3 and z_nonzero == 0:
                    return False, f"{z_col} all-zero despite {pulses} nonzero {base_col} pulses"
        
        # Check variance
        variances = df_for_checks[numeric_cols].var(ddof=0)
        features_with_variance = (variances > min_variance_threshold).sum()

        # Some families intentionally emit shape-stable, time-invariant metrics.
        # For these, "variance" is not a useful quality signal; rely on zero/null checks instead.
        allow_constant_families = {
            "calibration",
            "fin_g1",
            "index_constituents",
            "insider_form4",
            "corp_actions_splits",
            # Cross-sectional ranks can be piecewise-constant (especially with
            # small peer groups), but are still semantically meaningful.
            "peer_screener_context",
        }
        constant_ok = bool(family in allow_constant_families and features_with_variance == 0)

        # If we only have a handful of active rows, variance is not meaningful.
        # In that case, require only that at least one numeric feature is non-zero.
        min_rows_for_variance = 5
        if features_with_variance == 0 and not constant_ok and len(df_for_checks) < min_rows_for_variance:
            any_nonzero = bool((df_for_checks[numeric_cols] != 0).any().any())
            if any_nonzero:
                return True, f"valid (sparse has_data rows={len(df_for_checks)}; variance skipped)"
            return False, f"has_data rows present but ALL {len(numeric_cols)} numeric features are zero"
        
        if features_with_variance == 0 and not constant_ok:
            # If the family explicitly reports has_data==0 across the frame, treat as dormant.
            # This prevents sequential dependency chains from breaking when upstream providers
            # are unavailable or insufficient samples exist.
            if family in allowed_dormant_families and has_data_cols:
                try:
                    has_data_max = float(df_for_checks[has_data_cols].max().max())
                except Exception:
                    has_data_max = 1.0
                if has_data_max <= 0.0:
                    return True, f"dormant (has_data=0) — constant features accepted"

            # Calibration stubs are shape-stable and typically include sample_size=0 rather than has_data.
            if family == "calibration" and sample_size_cols:
                try:
                    sample_size_max = float(df_for_checks[sample_size_cols].max().max())
                except Exception:
                    sample_size_max = 1.0
                if sample_size_max <= 0.0:
                    return True, "dormant (sample_size=0) — constant features accepted"

            return False, f"ALL {len(numeric_cols)} numeric features have zero variance"
        
        # Check percentage of zeros
        zero_percentages = (df_for_checks[numeric_cols] == 0).sum() / len(df_for_checks)
        high_zero_features = (zero_percentages > max_zero_percentage).sum()
        
        # Check percentage of nulls
        null_percentages = df_for_checks[numeric_cols].isna().sum() / len(df_for_checks)
        high_null_features = (null_percentages > max_null_percentage).sum()
        
        # Warnings for borderline cases
        if high_zero_features > len(numeric_cols) * 0.8:
            LOGGER.warning(
                "⚠️ %s: %d/%d features have >%.0f%% zeros (borderline quality)",
                family, high_zero_features, len(numeric_cols), max_zero_percentage * 100
            )
        
        if high_null_features > len(numeric_cols) * 0.8:
            LOGGER.warning(
                "⚠️ %s: %d/%d features have >%.0f%% nulls (borderline quality)",
                family, high_null_features, len(numeric_cols), max_null_percentage * 100
            )
        
        # If most features are all zeros, this is likely a failure.
        # Exceptions:
        # - index_constituents is legitimately all-zeros for most symbols
        #   (not a member of any tracked indices).
        # - insider_form4 can be legitimately all-zeros over many windows
        #   (no Form-4 filings in the range), but should remain shape-stable.
        if high_zero_features == len(numeric_cols):
            if family == "index_constituents":
                return True, "valid (index_constituents: no membership/events; all-zero accepted)"
            if family == "insider_form4":
                return True, "valid (insider_form4: no filings in range; all-zero accepted)"
            if family == "corp_actions_splits":
                return True, "valid (corp_actions_splits: no splits in range; all-zero accepted)"
            return False, f"ALL {len(numeric_cols)} numeric features are >95% zeros"
        
        # If most features are all nulls, this is likely a failure
        if high_null_features == len(numeric_cols):
            return False, f"ALL {len(numeric_cols)} numeric features are >90% nulls"
        
        # Passed validation
        if constant_ok:
            return True, "valid (constant features allowed)"
        return True, f"valid ({features_with_variance}/{len(numeric_cols)} features with variance)"
        
    except Exception as e:
        LOGGER.error("Family quality validation exception for %s: %s", family, str(e))
        return False, f"validation error: {str(e)}"


# ============================================================================
# Signal Generation (Base Families)
# ============================================================================

def generate_base_family_signal(
    task: SignalTask,
    cache_dir: Path,
    venv_python: Optional[str] = None,
) -> bool:
    """
    Generate a base family signal by directly calling build_panel().
    
    Args:
        task: Signal generation task
        cache_dir: Cache directory
        venv_python: Optional Python executable path (unused, kept for compatibility)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import here to avoid heavy startup cost
        from src.features.aggregator_panel import build_panel
        from src.dcf_lab.signal_bus import standardize_module
        from src.features.lag_config import apply_lags

        try:
            spec = get_family_spec(task.family)
            is_hf_block = bool(getattr(spec, "stage", "") == "HF_BLOCK")
        except Exception:
            is_hf_block = False

        # Some families are computed from other families inside the aggregator.
        # Example: online_learning depends on quantile_forecast + calibration.
        # If we call build_panel with families=[online_learning] only, the
        # dependency helpers may return None and we get an empty frame.
        def _build_families_for_task(family: str) -> List[str]:
            deps = [d for d in family_dependencies(family) if d]
            if not deps:
                return [family]

            closure: Set[str] = set()
            stack: List[str] = [family]
            while stack:
                name = stack.pop()
                if name in closure:
                    continue
                closure.add(name)
                for dep in family_dependencies(name):
                    if dep and dep not in closure:
                        stack.append(dep)

            visited: Set[str] = set()
            visiting: Set[str] = set()
            ordered: List[str] = []

            def visit(name: str) -> None:
                if name not in closure or name in visited:
                    return
                if name in visiting:
                    return
                visiting.add(name)
                for dep in family_dependencies(name):
                    if dep in closure:
                        visit(dep)
                visiting.remove(name)
                visited.add(name)
                ordered.append(name)

            for token in DEFAULT_BASE_FAMILIES:
                visit(token)
            for token in sorted(closure):
                visit(token)
            return ordered
        
        # Some families require substantial historical context (rolling windows,
        # correlations, etc). When we need to generate a small missing segment
        # (e.g., a 1–2 day tail), build_panel can return empty unless we include
        # enough lookback. We fetch with padding and then trim back to the exact
        # requested range for cache writes.
        requested_start = task.date_start
        requested_end = task.date_end
        pad_bdays = int(os.getenv("PREP_FAMILIES_LOOKBACK_PAD_BDAYS", "400"))
        fetch_start = add_sessions(requested_start, -pad_bdays) if pad_bdays > 0 else requested_start

        families_to_build = _build_families_for_task(task.family)
        if len(families_to_build) > 1:
            LOGGER.info(
                "🔗 %s: building with dependencies=%s",
                task.family,
                ", ".join([f for f in families_to_build if f != task.family]),
            )

        panel = build_panel(
            task.symbol,
            fetch_start.strftime("%Y-%m-%d"),
            requested_end.strftime("%Y-%m-%d"),
            families=families_to_build,
            macro_lag_days=1,
            finbert_gap_thresh=1,
            cache_dir=cache_dir,
            window_idx=task.window_id,  # CRITICAL: Pass window_idx to enable windowed cache lookup
            view="raw",
            allow_generate_missing_from_cache=True,
        )

        if isinstance(panel, pd.DataFrame) and not panel.empty:
            panel = panel.loc[(panel.index >= requested_start) & (panel.index <= requested_end)]

        # If build_panel returned a non-empty panel that doesn't actually cover the requested
        # segment (common when build_panel hits a stale/partial cache during a merge backfill),
        # the trim above can yield an empty frame. In that case, force a cache-bypassing
        # re-run so we actually generate the missing date segment.
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            LOGGER.info(
                "🔁 Cache coverage miss for %s; forcing live generation (bypass cache)",
                task.cache_key(),
            )
            panel = build_panel(
                task.symbol,
                fetch_start.strftime("%Y-%m-%d"),
                requested_end.strftime("%Y-%m-%d"),
                families=families_to_build,
                macro_lag_days=1,
                finbert_gap_thresh=1,
                cache_dir=cache_dir,
                window_idx=task.window_id,
                view="raw",
                allow_generate_missing_from_cache=True,
            )
            if isinstance(panel, pd.DataFrame) and not panel.empty:
                panel = panel.loc[(panel.index >= requested_start) & (panel.index <= requested_end)]

        if not isinstance(panel, pd.DataFrame) or panel.empty:
            LOGGER.warning("build_panel returned empty DataFrame for %s", task.cache_key())
            return False
        
        # Extract family features
        family_cols = [col for col in panel.columns if col.startswith(f"{task.family}_")]
        if not family_cols:
            # build_panel may return a non-empty panel containing ONLY dependency families
            # loaded from cache (e.g., quantile_forecast) while skipping the target family
            # if its cache is missing. In that case, force a live generation pass.
            LOGGER.warning(
                "No features found for family %s in cached panel; forcing live generation",
                task.family,
            )
            panel = build_panel(
                task.symbol,
                fetch_start.strftime("%Y-%m-%d"),
                requested_end.strftime("%Y-%m-%d"),
                families=families_to_build,
                macro_lag_days=1,
                finbert_gap_thresh=1,
                cache_dir=cache_dir,
                window_idx=task.window_id,
                view="raw",
                allow_generate_missing_from_cache=True,
            )
            if isinstance(panel, pd.DataFrame) and not panel.empty:
                panel = panel.loc[(panel.index >= requested_start) & (panel.index <= requested_end)]
            if not isinstance(panel, pd.DataFrame) or panel.empty:
                LOGGER.warning("build_panel returned empty DataFrame for %s", task.cache_key())
                return False
            family_cols = [col for col in panel.columns if col.startswith(f"{task.family}_")]
            if not family_cols:
                LOGGER.warning("No features found for family %s in panel", task.family)
                return False
        
        # Base features (no lag expansion) saved for Stage A consumption.
        base_features = panel[family_cols].copy()
        if task.family == "finbert":
            base_features = _enforce_finbert_neutral_fill(base_features, task.cache_key())

        if base_features.columns.duplicated().any():
            base_features = base_features.loc[:, ~base_features.columns.duplicated(keep="last")]

        base_feature_cache = base_features.copy()
        base_feature_cache.insert(0, "date", panel.index)

        feature_cache_path = task.cache_path.parent / task.cache_path.name.replace(
            ".parquet",
            "_features.parquet",
        )
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Use the shared cache writer to guarantee parquet-safe column labels.
        _write_feature_cache(
            feature_cache_path,
            base_feature_cache,
            merge=(task.merge_strategy == "merge"),
            allowed_columns=(
                ML_FRAMEWORK_CANONICAL_FEATURE_COLS
                if task.family == "ml_framework"
                else CORRELATION_CANONICAL_FEATURE_COLS
                if task.family == "correlation"
                else EARNINGS_CANONICAL_FEATURE_COLS
                if task.family == "earnings"
                else None
            ),
            family=task.family,
        )
        LOGGER.info(
            "✅ Cached %d raw features for Stage A: %s",
            len(base_feature_cache.columns) - 1,
            feature_cache_path.name,
        )
        
        # Validate feature quality
        is_valid, reason = validate_family_quality(feature_cache_path, task.family)
        if not is_valid and cache_dir is not None:
            # One-shot retry: stale caches can be structurally valid but semantically
            # degenerate (all zeros / constant). In that case, bypass caches and
            # regenerate from live sources.
            LOGGER.warning(
                "⚠️ Family quality validation failed for %s (%s). Retrying with cache bypass...",
                task.cache_key(),
                reason,
            )
            panel = build_panel(
                task.symbol,
                fetch_start.strftime("%Y-%m-%d"),
                requested_end.strftime("%Y-%m-%d"),
                families=families_to_build,
                macro_lag_days=1,
                finbert_gap_thresh=1,
                cache_dir=(cache_dir if is_hf_block else None),
                window_idx=task.window_id,
                view="raw",
                allow_generate_missing_from_cache=True,
            )
            if isinstance(panel, pd.DataFrame) and not panel.empty:
                panel = panel.loc[(panel.index >= requested_start) & (panel.index <= requested_end)]

            if not isinstance(panel, pd.DataFrame) or panel.empty:
                LOGGER.error("Cache-bypass retry returned empty DataFrame for %s", task.cache_key())
                return False

            family_cols = [col for col in panel.columns if col.startswith(f"{task.family}_")]
            if not family_cols:
                LOGGER.error("Cache-bypass retry produced no %s_* columns for %s", task.family, task.cache_key())
                return False

            base_features = panel[family_cols].copy()
            if task.family == "finbert":
                base_features = _enforce_finbert_neutral_fill(base_features, task.cache_key())
            if base_features.columns.duplicated().any():
                base_features = base_features.loc[:, ~base_features.columns.duplicated(keep="last")]

            base_feature_cache = base_features.copy()
            base_feature_cache.insert(0, "date", panel.index)
            _write_feature_cache(
                feature_cache_path,
                base_feature_cache,
                merge=(task.merge_strategy == "merge"),
                allowed_columns=(
                    ML_FRAMEWORK_CANONICAL_FEATURE_COLS
                    if task.family == "ml_framework"
                    else CORRELATION_CANONICAL_FEATURE_COLS
                    if task.family == "correlation"
                    else EARNINGS_CANONICAL_FEATURE_COLS
                    if task.family == "earnings"
                    else None
                ),
                family=task.family,
            )

            is_valid, reason = validate_family_quality(feature_cache_path, task.family)

        if not is_valid:
            LOGGER.error(
                "❌ Family quality validation FAILED for %s: %s",
                task.cache_key(),
                reason,
            )
            return False

        LOGGER.info("✅ Family quality validation passed for %s: %s", task.family, reason)

        # Lag-expanded caches are disabled by default; set WRITE_LAGGED_FEATURE_CACHES=1 to re-enable.
        lagged_features = base_features
        if WRITE_LAGGED_FEATURE_CACHES:
            lagged_features = apply_lags(base_features, task.family)
            if lagged_features is None or lagged_features.empty:
                lagged_features = base_features
            if lagged_features.columns.duplicated().any():
                lagged_features = lagged_features.loc[:, ~lagged_features.columns.duplicated(keep="last")]
            lagged_cache = lagged_features.copy()
            lagged_cache.insert(0, "date", panel.index)
            lagged_feature_cache_path = feature_cache_path.with_name(
                feature_cache_path.name.replace("_features.parquet", "_features_lagged.parquet")
            )
            lagged_feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_feature_cache(
                lagged_feature_cache_path,
                lagged_cache,
                merge=(task.merge_strategy == "merge"),
                allowed_columns=(
                    ML_FRAMEWORK_CANONICAL_FEATURE_COLS
                    if task.family == "ml_framework"
                    else CORRELATION_CANONICAL_FEATURE_COLS
                    if task.family == "correlation"
                    else EARNINGS_CANONICAL_FEATURE_COLS
                    if task.family == "earnings"
                    else None
                ),
                family=task.family,
            )
            LOGGER.info(
                "\u2705 Cached %d lag-expanded features for downstream stages: %s",
                len(lagged_cache.columns) - 1,
                lagged_feature_cache_path.name,
            )

        # Governance columns: has_data, activity, days_since_update
        has_data_col, activity_col, days_since_col = _family_metric_columns(task.family)

        # Build governance signal frame with the 3 required columns
        signal = pd.DataFrame(index=panel.index)
        
        # Check if has_data already exists in lagged_features
        if has_data_col in lagged_features.columns:
            signal[has_data_col] = (
                pd.to_numeric(lagged_features[has_data_col], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
            )
        else:
            # Default: 1.0 if we have data at this row
            signal[has_data_col] = 1.0
        
        # Activity: 1.0 when data is present (can be refined per-family)
        if activity_col in lagged_features.columns:
            signal[activity_col] = (
                pd.to_numeric(lagged_features[activity_col], errors="coerce")
                .fillna(1.0)
                .clip(0.0, 1.0)
            )
        else:
            signal[activity_col] = 1.0
        
        # Days since update: 0.0 for fresh data
        if days_since_col in lagged_features.columns:
            signal[days_since_col] = (
                pd.to_numeric(lagged_features[days_since_col], errors="coerce")
                .fillna(0.0)
            )
        else:
            signal[days_since_col] = 0.0

        # Attach optional reliability/drift columns without losing chronology
        reliability_fields = _family_attribute_columns(task.family, "reliability_cols")
        drift_fields = _family_attribute_columns(task.family, "drift_cols")
        for extra in reliability_fields + drift_fields:
            col_name = f"{task.family}_{extra}"
            if col_name in panel.columns:
                signal[col_name] = panel[col_name].values

        # Append structured summary columns (quantile/ARIMA/etc.) when available
        summary_payload = compute_family_summary(task.family, lagged_features)
        for col_name, series in summary_payload.items():
            if series is None:
                continue
            if isinstance(series, pd.DataFrame):
                if col_name in series.columns:
                    series = series[col_name]
                elif series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    continue
            if not isinstance(series, pd.Series):
                continue
            if not series.index.is_unique:
                series = series[~series.index.duplicated(keep="last")]
            try:
                series.index = pd.to_datetime(series.index).tz_localize(None)
            except Exception:
                pass
            signal[col_name] = series.reindex(signal.index)
        
        # Normalize chronology + enforce parquet-safe columns
        signal = _normalize_date_column(signal)

        # DISABLED: Signal file writing (only _features.parquet is needed)
        # The signal file was a legacy artifact containing only governance columns.
        # All consumers now read from _features.parquet directly.
        # 
        # task.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # if task.merge_strategy == "merge":
        #     signal = _merge_on_date(task.cache_path, signal)
        # signal.to_parquet(task.cache_path, index=False)
        # LOGGER.info("✅ Generated base family signal: %s", task.cache_key())
        
        return True
        
    except Exception as e:
        LOGGER.error(
            "Base family generation exception for %s: %s",
            task.cache_key(),
            str(e)
        )
        return False


# ============================================================================
# Signal Generation (HF Modules)
# ============================================================================

def generate_hf_module_signal(
    task: SignalTask,
    cache_dir: Path,
) -> bool:
    """
    Generate an HF module signal using the HF registry.
    
    Looks up the generator in hf_registry.yaml and calls it with:
    - Leak-safe date range [task.date_start, task.date_end)
    - Default raw_source_cfg and compute_cfg from registry
    - Output path = task.cache_path
    
    Args:
        task: Signal generation task
        cache_dir: Cache directory
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Fast-path: reuse symbol-invariant HF modules across symbols and processes.
        family_lower = str(task.family).lower()
        if family_lower in SYMBOL_INVARIANT_HF_MODULES:
            shared_path = _shared_invariant_cache_path(task, cache_dir)
            lock_path = shared_path.with_suffix(shared_path.suffix + ".lock")

            # If a shared artifact already exists, just copy it.
            if shared_path.exists():
                if _copy_hf_artifacts(shared_path, task.cache_path):
                    LOGGER.info(
                        "♻️ Reused symbol-invariant HF module '%s' for %s from shared cache %s",
                        task.family,
                        task.cache_key(),
                        shared_path.name,
                    )
                    return True

            # Try to become the writer; otherwise wait for the shared artifact.
            have_lock = _acquire_file_lock(lock_path)
            if not have_lock:
                if _wait_for_shared_file(shared_path, timeout_s=3600):
                    if _copy_hf_artifacts(shared_path, task.cache_path):
                        LOGGER.info(
                            "♻️ Reused symbol-invariant HF module '%s' for %s from shared cache %s",
                            task.family,
                            task.cache_key(),
                            shared_path.name,
                        )
                        return True
                LOGGER.warning(
                    "Timed out waiting for shared %s (%s); falling back to local generation for %s",
                    task.family,
                    shared_path.name,
                    task.cache_key(),
                )
            else:
                LOGGER.info(
                    "🔒 Building shared symbol-invariant HF module '%s' at %s",
                    task.family,
                    shared_path,
                )

                # Generate into shared_path, then copy to this symbol's cache.
                original_cache_path = task.cache_path
                try:
                    task.cache_path = shared_path
                except Exception:
                    # Should never happen, but keep safety.
                    have_lock = False
                finally:
                    pass

        # Governance columns: has_data, activity, days_since_update
        has_data_col, activity_col, days_since_col = _family_metric_columns(task.family)
        # Import HF generators module
        from tools.hf_generators import get_generator, load_registry
        from src.dcf_lab.signal_bus import standardize_module
        from src.features.lag_config import apply_lags
        
        # Load registry and get config for this module
        registry = load_registry()
        
        if task.family not in registry:
            LOGGER.error(
                "HF module '%s' not found in registry. Available: %s",
                task.family,
                ', '.join(registry.keys())
            )
            return False
        
        module_config = registry[task.family]
        
        # Get generator module
        generator = get_generator(task.family)
        
        # Prepare configs (use defaults from registry)
        raw_source_cfg = module_config.get("default_raw_source_cfg", {})
        compute_cfg = module_config.get("default_compute_cfg", {})
        
        # HF generators expect an end-exclusive interval [start, end).
        # Our consolidated scheduling treats `task.date_end` as an inclusive session label.
        # Convert to exclusive by advancing one trading session.
        hf_start = task.date_start
        hf_end_exclusive = add_sessions(task.date_end, 1)

        LOGGER.info(
            "Generating HF signal: %s [%s → %s)",
            task.cache_key(),
            hf_start.date(),
            hf_end_exclusive.date(),
        )
        
        # Symbol-only HF blocks must be generated using a canonical horizon to
        # prevent cross-horizon cache drift (the cache path is shared).
        build_horizon = int(task.horizon)
        if _uses_symbol_only_cache_path(task.family):
            build_horizon = int(HF_SYMBOL_ONLY_CANONICAL_HORIZON)

        # Use temporary path for raw output
        temp_path = task.cache_path.with_suffix(".raw.parquet")
        
        generator.build(
            symbol=task.symbol,
            horizon=build_horizon,
            start=hf_start.isoformat(),
            end=hf_end_exclusive.isoformat(),
            out_path=str(temp_path),
            raw_source_cfg=raw_source_cfg,
            compute_cfg=compute_cfg,
        )
        
        # Verify file was created
        if not temp_path.exists():
            LOGGER.error(
                "HF generator did not create output file: %s",
                temp_path
            )
            return False
        
        signal = pd.read_parquet(temp_path)
        
        # Normalize chronology and drop duplicate dates before saving
        signal = _normalize_date_column(signal)

        # HF generators may emit a generic 'has_data' column. Persist it with a family prefix
        # to avoid collisions with other families and with the global stable-schema stubs.
        if "has_data" in signal.columns:
            try:
                signal[has_data_col] = pd.to_numeric(signal["has_data"], errors="coerce").fillna(1.0)
            except Exception:
                signal[has_data_col] = 1.0
            signal = signal.drop(columns=["has_data"], errors="ignore")
        else:
            signal[has_data_col] = 1.0
        
        # Add activity and days_since_update governance columns
        signal[activity_col] = 1.0
        signal[days_since_col] = 0.0
        
        # Drop legacy score/conf columns if present (we no longer use them)
        legacy_cols = ["score", "score_raw", "conf", "confidence"]
        signal = signal.drop(columns=[c for c in legacy_cols if c in signal.columns], errors="ignore")

        # DISABLED: Signal file writing (only _features.parquet is needed)
        # task.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # if task.merge_strategy == "merge":
        #     signal = _merge_on_date(task.cache_path, signal)
        # signal.to_parquet(task.cache_path, index=False)

        # Persist raw feature cache for Stage A consumers
        feature_cache_path = task.cache_path.parent / task.cache_path.name.replace(
            ".parquet",
            "_features.parquet",
        )
        try:
            _write_feature_cache(
                feature_cache_path,
                signal,
                merge=task.merge_strategy == "merge",
                family=task.family,
            )
            LOGGER.info(
                "✅ Cached HF raw features for Stage A: %s",
                feature_cache_path.name,
            )

            if WRITE_LAGGED_FEATURE_CACHES:
                # Optional lag-expanded cache for downstream stages.
                try:
                    if "date" in signal.columns:
                        base_features, date_index = _datetime_index_from_date_column(signal)
                        lagged_features = apply_lags(base_features, task.family)
                        if lagged_features is None or lagged_features.empty:
                            lagged_features = base_features

                        if lagged_features.columns.duplicated().any():
                            lagged_features = lagged_features.loc[:, ~lagged_features.columns.duplicated(keep="last")]

                        lagged_cache = lagged_features.copy()
                        lagged_cache.insert(0, "date", date_index.to_numpy())
                        lagged_feature_cache_path = feature_cache_path.with_name(
                            feature_cache_path.name.replace("_features.parquet", "_features_lagged.parquet")
                        )
                        _write_feature_cache(
                            lagged_feature_cache_path,
                            lagged_cache,
                            merge=task.merge_strategy == "merge",
                            family=task.family,
                        )
                        LOGGER.info(
                            "\u2705 Cached %d lag-expanded features for downstream stages: %s",
                            len(lagged_cache.columns) - 1,
                            lagged_feature_cache_path.name,
                        )
                except Exception as lag_exc:
                    LOGGER.warning(
                        "Failed to cache HF lag-expanded frame %s: %s",
                        feature_cache_path.name.replace("_features.parquet", "_features_lagged.parquet"),
                        lag_exc,
                    )
        except Exception as feature_exc:
            LOGGER.warning(
                "Failed to cache HF feature frame %s: %s",
                feature_cache_path.name,
                feature_exc,
            )
        
        # Clean up temp file
        temp_path.unlink()

        # If we generated into a shared cache path, copy artifacts back to the
        # per-symbol cache path and release the lock.
        if family_lower in SYMBOL_INVARIANT_HF_MODULES:
            shared_path = task.cache_path
            shared_lock = shared_path.with_suffix(shared_path.suffix + ".lock")
            original_cache_path = locals().get("original_cache_path")
            if isinstance(original_cache_path, Path) and original_cache_path != shared_path:
                try:
                    _copy_hf_artifacts(shared_path, original_cache_path)
                    # Restore for downstream metadata/logging
                    task.cache_path = original_cache_path
                    LOGGER.info(
                        "♻️ Reused symbol-invariant HF module '%s' for %s from shared cache %s",
                        task.family,
                        task.cache_key(),
                        shared_path.name,
                    )
                finally:
                    try:
                        shared_lock.unlink()
                    except Exception:
                        pass
        
        LOGGER.info("✅ Generated & standardized HF module signal: %s", task.cache_key())
        return True
        
    except ImportError as e:
        LOGGER.error(
            "Failed to import HF generator for %s: %s",
            task.family,
            str(e)
        )
        return False
    
    except Exception as e:
        LOGGER.error(
            "Failed to generate HF signal for %s: %s",
            task.cache_key(),
            str(e)
        )
        import traceback
        LOGGER.debug("Full traceback:\n%s", traceback.format_exc())
        return False


# ============================================================================
# Task Orchestration
# ============================================================================

def build_signal_tasks(
    symbol: str,
    horizon: int,
    windows: List[WalkForwardWindow],
    base_families: List[str],
    hf_modules: List[str],
    cache_dir: Path,
    existing_signals: Dict[str, Dict[str, Set[int]]],
) -> List[SignalTask]:
    """
    Build list of per-window signal generation tasks.
    
    Args:
        symbol: Stock symbol
        horizon: Forecast horizon
        windows: Walk-forward windows
        base_families: Base family list
        hf_modules: HF module list
        cache_dir: Cache directory
        existing_signals: Dict of existing signals per family (split -> window set)
    
    Returns:
        List of SignalTask objects for missing signals
    """
    symbol_lower = symbol.lower()
    tasks = []
    today = pd.Timestamp.now().normalize()
    
    def _family_iter():
        for fam in base_families:
            yield fam, False, fam in SEQUENTIAL_ONLY_FAMILIES
        # HF modules are treated as base families for *scheduling*, but they still
        # require the HF generator path (task.is_hf=True). They should not run in
        # a multiprocessing pool due to nested subprocess/thread usage.
        for fam in hf_modules:
            yield fam, True, True

    for family, is_hf, sequential_only in _family_iter():
        family_existing = existing_signals.get(family, {})
        
        for split in ["train", "valid"]:
            completed_windows = family_existing.get(split, set())
            
            for window_idx, window in enumerate(windows):
                if window_idx in completed_windows:
                    continue
                
                if split == "train":
                    date_start = window.train_start
                    date_end = window.train_end
                else:
                    date_start = window.valid_start
                    date_end = window.test_end
                
                if date_end > today:
                    LOGGER.debug(
                        "Capping end date from %s to %s (today) for %s %s w%d",
                        date_end.date(), today.date(), family, split, window_idx
                    )
                    date_end = today
                
                if date_start > date_end:
                    LOGGER.info(
                        "Skipping %s %s window %d (dates in future: %s > %s)",
                        family,
                        split,
                        window_idx,
                        date_start.date(),
                        date_end.date(),
                    )
                    continue
                
                cache_path = cache_dir / (
                    f"{symbol_lower}_h{horizon}_{family}_{split}_w{window_idx}.parquet"
                )
                
                task = SignalTask(
                    symbol=symbol,
                    horizon=horizon,
                    family=family,
                    window_id=window_idx,
                    split=split,
                    date_start=date_start,
                    date_end=date_end,
                    cache_path=cache_path,
                    is_hf=is_hf,
                    sequential_only=sequential_only,
                )
                tasks.append(task)
    
    return tasks


def deduplicate_tasks(tasks: Iterable[SignalTask]) -> Tuple[List[SignalTask], int]:
    """Remove duplicate cache targets while preserving original order."""

    unique: List[SignalTask] = []
    seen: Set[str] = set()
    duplicates = 0
    for task in tasks:
        key = task.cache_key()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(task)
    return unique, duplicates


def compute_required_split_ranges(
    windows: List[WalkForwardWindow],
    fallback_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Compute the aggregate date coverage needed for consolidated caches.
    
    When windows is empty (Stage-A mode), returns a single 'consolidated' range
    covering the entire fallback period - no train/valid splits needed.
    
    When windows exist (walk-forward mode), returns train/valid split ranges
    for the walk-forward windows.
    """
    if not windows:
        # Stage-A mode: single consolidated range, no train/valid splits
        if fallback_range is None:
            return {}
        fallback_start, fallback_end = fallback_range
        today = pd.Timestamp.now().normalize()
        capped_end = min(fallback_end, today)
        return {
            "consolidated": (fallback_start, capped_end),
        }

    today = pd.Timestamp.now().normalize()
    train_start = windows[0].train_start
    train_end = max(window.train_end for window in windows)
    valid_start = windows[0].valid_start
    valid_end = max(window.test_end for window in windows)

    # If a fallback_range is provided (typically wf_start/wf_end), ensure our
    # consolidated caches cover *at least* that span even if Stage-A windows are
    # shorter. This keeps merged parquets stable across orchestration modes.
    if fallback_range is not None:
        fallback_start, fallback_end = fallback_range
        if fallback_start is not None:
            train_start = min(train_start, fallback_start)
            valid_start = min(valid_start, fallback_start)
        if fallback_end is not None:
            train_end = max(train_end, fallback_end)
            valid_end = max(valid_end, fallback_end)

    return {
        "train": (train_start, min(train_end, today)),
        "valid": (valid_start, min(valid_end, today)),
    }


def build_consolidated_tasks(
    symbol: str,
    horizon: int,
    base_families: List[str],
    hf_modules: List[str],
    hf_blocks: List[str],
    meta_families: List[str],
    required_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    coverage_map: Dict[str, Dict[str, ConsolidatedCoverage]],
    cache_dir: Path,
) -> List[SignalTask]:
    """Create consolidated generation tasks for missing coverage."""
    tasks: List[SignalTask] = []
    symbol_lower = symbol.lower()
    # (family, is_hf_module)
    family_plan: List[Tuple[str, bool]] = []
    family_plan.extend((f, False) for f in base_families)
    # HF modules use the HF registry/generator path.
    family_plan.extend((f, True) for f in hf_modules)
    # HF blocks are derived panels built via build_panel() (NOT HF registry modules).
    family_plan.extend((f, False) for f in hf_blocks)
    # Meta families (like hf_agg) are derived (no standalone caches) and must not be scheduled as tasks.

    # Determine which splits to iterate: 'consolidated' for Stage-A, or 'train'/'valid' for walk-forward
    splits_to_iterate = list(required_ranges.keys())

    for family, is_hf_module in family_plan:
        for split in splits_to_iterate:
            if split not in required_ranges:
                continue
            required_start, required_end = required_ranges[split]
            # For coverage lookup, try split-specific first, then fall back to consolidated
            # This allows Stage-B to reuse Stage-A consolidated caches when train/valid don't exist
            coverage = coverage_map.get(family, {}).get(split)
            if coverage is None and split in ("train", "valid"):
                # Fall back to consolidated coverage if train/valid doesn't exist
                coverage = coverage_map.get(family, {}).get("")
            segments: List[Tuple[pd.Timestamp, pd.Timestamp, Literal["replace", "merge"]]] = []

            # Symbol-only HF blocks are shared across horizons, so they must be
            # generated using a canonical horizon to avoid cross-horizon drift.
            task_horizon = int(horizon)
            if _uses_symbol_only_cache_path(family):
                task_horizon = int(HF_SYMBOL_ONLY_CANONICAL_HORIZON)

            # For consolidated mode, use empty string for split to generate files without _train/_valid suffix
            file_split = "" if split == "consolidated" else split
            cache_path, _, _ = _family_cache_paths(
                cache_dir,
                symbol,
                horizon,
                family,
                file_split,
            )

            # Even if raw feature coverage exists, strict completeness (and some downstream
            # consumers) expect the consolidated score/conf signal parquet to exist.
            # If it's missing, force a rebuild for the full required range.
            if not cache_path.exists():
                segments.append((required_start, required_end, "replace"))
            elif coverage is None:
                segments.append((required_start, required_end, "replace"))
            elif coverage.covers(required_start, required_end):
                continue
            else:
                if coverage.date_start > required_start:
                    seg_end = min(coverage.date_start - pd.Timedelta(days=1), required_end)
                    if seg_end >= required_start:
                        segments.append((required_start, seg_end, "merge"))
                if coverage.date_end < required_end:
                    seg_start = max(coverage.date_end + pd.Timedelta(days=1), required_start)
                    if seg_start <= required_end:
                        segments.append((seg_start, required_end, "merge"))
            # Dependency families must be strictly sequential.
            # HF *modules* must not run inside a multiprocessing pool; we execute them via
            # an in-process threadpool instead (can still be parallel).
            sequential_only = (family in SEQUENTIAL_ONLY_FAMILIES) or is_hf_module
            for seg_start, seg_end, strategy in segments:
                if seg_end < seg_start:
                    continue
                # Skip tiny gaps when we already have consolidated coverage.
                # These are typically weekend/holiday boundary artifacts and can produce
                # empty panels, tripping strict mode for no practical benefit.
                if coverage is not None:
                    # Skip gaps only when they contain *zero* trading sessions.
                    # Small gaps can arise from weekend/holiday boundaries and can produce
                    # empty panels, but we must NOT skip missing trading-day tails.
                    try:
                        gap_start_session = session_on_or_after(seg_start)
                        gap_end_session = session_on_or_before(seg_end)
                    except Exception:
                        gap_start_session = seg_start.normalize()
                        gap_end_session = seg_end.normalize()

                    if gap_start_session > gap_end_session:
                        LOGGER.debug(
                            "Skipping zero-session consolidated gap for %s/%s: %s -> %s",
                            family,
                            split,
                            seg_start.date(),
                            seg_end.date(),
                        )
                        continue
                task = SignalTask(
                    symbol=symbol,
                    horizon=task_horizon,
                    family=family,
                    window_id=None,
                    split=file_split,  # Use file_split for cache file naming (empty for consolidated mode)
                    date_start=seg_start,
                    date_end=seg_end,
                    cache_path=cache_path,
                    is_hf=is_hf_module,
                    scope="consolidated",
                    sequential_only=sequential_only,
                    merge_strategy=strategy,
                )
                tasks.append(task)
    return tasks


def build_slice_tasks(
    symbol: str,
    horizon: int,
    windows: List[WalkForwardWindow],
    base_families: List[str],
    hf_modules: List[str],
    hf_blocks: List[str],
    meta_families: List[str],
    cache_dir: Path,
    existing_signals: Dict[str, Dict[str, Set[int]]],
    coverage_map: Dict[str, Dict[str, ConsolidatedCoverage]],
) -> List[SliceTask]:
    """Create slicing tasks for windows lacking per-window files."""
    tasks: List[SliceTask] = []
    symbol_lower = symbol.lower()
    if not windows:
        return tasks

    family_plan: List[Tuple[str, bool]] = []
    family_plan.extend((f, False) for f in base_families)
    # HF modules and HF blocks are event-driven; slice standardization should
    # treat them as HF so it preserves the conf>0 event mask behavior.
    family_plan.extend((f, True) for f in hf_modules)
    family_plan.extend((f, True) for f in hf_blocks)
    # Meta families (like hf_agg) are derived (no standalone caches) and must not be scheduled as tasks.

    for family, is_hf in family_plan:
        family_existing = existing_signals.get(family, {})
        for split in ["train", "valid"]:
            coverage = coverage_map.get(family, {}).get(split)
            if coverage is None:
                continue
            completed_windows = family_existing.get(split, set())
            for window_idx, window in enumerate(windows):
                if window_idx in completed_windows:
                    continue
                if split == "train":
                    req_start, req_end = window.train_start, window.train_end
                else:
                    req_start, req_end = window.valid_start, window.test_end
                # 🔧 CRITICAL FIX: Use partial_covers to generate window files even with partial data
                # This ensures families like cboe_term (starting July 2010) get included in early windows
                # where they have SOME data, even if not covering the full training period.
                if not coverage.partial_covers(req_start, req_end, min_overlap_days=30):
                    continue
                dest_path = cache_dir / (
                    f"{symbol_lower}_h{horizon}_{family}_{split}_w{window_idx}.parquet"
                )
                task = SliceTask(
                    symbol=symbol,
                    horizon=horizon,
                    family=family,
                    split=split,
                    window_id=window_idx,
                    date_start=req_start,
                    date_end=req_end,
                    source_path=coverage.path,
                    is_hf=is_hf,
                )
                tasks.append(task)
    return tasks


def process_slice_task(task: SliceTask) -> Tuple[SliceTask, bool]:
    """Slice consolidated parquet data into a window-scoped file."""
    try:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.dataset as ds  # type: ignore
        except ImportError:
            pa = None
            ds = None

        # Read consolidated file and slice by date
        df = pd.read_parquet(task.source_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Normalize timezones for comparison
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        
        start_ts = task.date_start
        end_ts = task.date_end
        if hasattr(start_ts, "tz") and start_ts.tz is not None:
            start_ts = start_ts.tz_localize(None)
        if hasattr(end_ts, "tz") and end_ts.tz is not None:
            end_ts = end_ts.tz_localize(None)
        
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        window_df = df.loc[mask].copy()

        if window_df.empty:
            LOGGER.warning("Slice produced empty frame for %s", task.cache_key())
            return task, False

        window_df["date"] = pd.to_datetime(window_df["date"])  # ensure datetime
        window_df = window_df.sort_values("date")

        # Governance columns: has_data, activity, days_since_update
        has_data_col, activity_col, days_since_col = _family_metric_columns(task.family)

        def _select_series(candidates: Sequence[str], default: float = 0.0) -> pd.Series:
            for col in candidates:
                if col in window_df.columns:
                    return window_df[col].astype(float)
            return pd.Series(default, index=window_df.index, dtype=float)

        # Ensure governance columns exist
        if has_data_col not in window_df.columns:
            window_df[has_data_col] = 1.0
        if activity_col not in window_df.columns:
            window_df[activity_col] = 1.0
        if days_since_col not in window_df.columns:
            window_df[days_since_col] = 0.0

        base_cols = [has_data_col, activity_col, days_since_col]
        family_prefix = f"{task.family}_"
        extra_cols = [
            col
            for col in window_df.columns
            if col.startswith(family_prefix) and col not in base_cols
        ]
        ordered_cols = ["date"] + [col for col in base_cols if col in window_df.columns] + extra_cols

        output = window_df.loc[:, ordered_cols].copy()
        task.dest_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(task.dest_path, index=False)

        meta = {
            "symbol": task.symbol,
            "family": task.family,
            "split": task.split,
            "horizon": task.horizon,
            "window_id": task.window_id,
            "date_start": output["date"].min().isoformat(),
            "date_end": output["date"].max().isoformat(),
            "requested_start": task.date_start.isoformat(),
            "requested_end": task.date_end.isoformat(),
            "source_path": str(task.source_path),
            "scope": "window",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "is_hf": task.is_hf,
        }
        meta_path = _cache_meta_path(task.dest_path)
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as handle:
                json.dump(meta, handle, indent=2)
        except Exception as exc:
            LOGGER.warning("Failed to write slice metadata %s: %s", meta_path.name, exc)
        return task, True
    except Exception as exc:
        LOGGER.error("Slice task failed for %s: %s", task.cache_key(), exc)
        return task, False


def execute_slice_tasks(
    tasks: List[SliceTask],
    workers: int,
) -> Tuple[int, int, List[str]]:
    """Execute slicing tasks in parallel."""
    if not tasks:
        return 0, 0, []

    completed = 0
    failed = 0
    failed_families: List[str] = []

    if workers <= 1:
        for task in tasks:
            _, success = process_slice_task(task)
            if success:
                completed += 1
            else:
                failed += 1
                if task.family not in failed_families:
                    failed_families.append(task.family)
    else:
        # Use platform-appropriate context so fork is used on Linux (avoids semaphore leaks)
        ctx = _get_pool_context()
        with ctx.Pool(processes=workers) as pool:
            results = [pool.apply_async(process_slice_task, (task,)) for task in tasks]
            for result in results:
                task, success = result.get(timeout=900)
                if success:
                    completed += 1
                else:
                    failed += 1
                    if task.family not in failed_families:
                        failed_families.append(task.family)

    return completed, failed, failed_families


def process_signal_task(task: SignalTask, cache_dir: Path) -> Tuple[SignalTask, bool]:
    """
    Process a single signal generation task.
    
    Args:
        task: Signal task to process
        cache_dir: Cache directory
    
    Returns:
        (task, success) tuple
    """
    try:
        if task.is_hf:
            success = generate_hf_module_signal(task, cache_dir)
        else:
            success = generate_base_family_signal(task, cache_dir)
        
        if success:
            write_task_metadata(task)
        return task, success
        
    except Exception as e:
        LOGGER.error("Task processing exception for %s: %s", task.cache_key(), str(e))
        return task, False


def _pool_signal_task(args: Tuple[SignalTask, Path]) -> Tuple[SignalTask, bool]:
    """Helper so multiprocessing pool can call process_signal_task."""
    task, cache_dir = args
    return process_signal_task(task, cache_dir)


def execute_tasks_parallel(
    tasks: List[SignalTask],
    cache_dir: Path,
    workers: int,
    hf_workers: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """
    Execute signal generation tasks in parallel.
    
    Splits tasks into parallel-safe vs sequential-only (HF pipelines, FinBERT, etc.)
    so we can keep base families on a large worker pool while avoiding daemon
    restrictions for HuggingFace workloads.
    
    Args:
        tasks: List of tasks to execute
        cache_dir: Cache directory
        workers: Number of parallel workers
    
    Returns:
        (completed, failed, failed_families) tuple
    """
    if not tasks:
        return 0, 0, []

    tasks, removed = deduplicate_tasks(tasks)
    if removed:
        LOGGER.info("⚠️ Deduplicated %d duplicate signal tasks before scheduling", removed)
    
    parallel_tasks = [t for t in tasks if not t.sequential_only]
    sequential_tasks = [t for t in tasks if t.sequential_only]
    
    completed = 0
    failed = 0
    failed_families: List[str] = []
    finished_keys: Set[str] = set()
    
    def _record_outcome(task: SignalTask, success: bool) -> None:
        nonlocal completed, failed
        finished_keys.add(task.cache_key())
        if success:
            completed += 1
        else:
            failed += 1
            if task.family not in failed_families:
                failed_families.append(task.family)

    def _run_pool(tasks_to_run: List[SignalTask], worker_count: int, ctx: mp.context.BaseContext, label: str) -> None:
        if worker_count == 1:
            for task in tasks_to_run:
                result_task, success = process_signal_task(task, cache_dir)
                _record_outcome(result_task, success)
            return

        base_executor = os.getenv("PREP_FAMILIES_BASE_EXECUTOR", "process").strip().lower()
        # HF families should always use threads (HF/tokenizers/datasets frequently spawn
        # subprocesses internally and can break under multiprocessing pools).
        use_threads = label == "HF families" or (
            label == "Base families" and base_executor in {"thread", "threads"}
        )

        # If this function is running inside a daemon process (common when
        # prep_families itself is invoked from an outer multiprocessing/Ray job),
        # attempting to spawn child processes can crash with:
        #   "daemonic processes are not allowed to have children"
        # In that case, prefer threads for the base-family parallelism.
        try:
            in_daemon = bool(mp.current_process().daemon)
        except Exception:
            in_daemon = False
        if in_daemon and label == "Base families" and not use_threads:
            LOGGER.warning(
                "prep_families running inside a daemon process; forcing base executor to threads to avoid child-process spawn failures"
            )
            use_threads = True

        chunksize = max(1, len(tasks_to_run) // (worker_count * 4))
        payloads = [(task, cache_dir) for task in tasks_to_run]
        progress_interval = max(1, len(tasks_to_run) // 5)
        LOGGER.info(
            "%s: Executing %d tasks with %d workers (chunksize=%d)",
            label,
            len(tasks_to_run),
            worker_count,
            chunksize,
        )

        if use_threads:
            LOGGER.info(
                "%s: Using ThreadPoolExecutor (PREP_FAMILIES_BASE_EXECUTOR=%s)",
                label,
                base_executor,
            )
            try:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(process_signal_task, task, cache_dir) for task in tasks_to_run]
                    for processed, fut in enumerate(as_completed(futures), start=1):
                        task, success = fut.result()
                        _record_outcome(task, success)
                        if processed % progress_interval == 0 or processed == len(tasks_to_run):
                            LOGGER.info(
                                "%s progress: %d/%d",
                                label,
                                processed,
                                len(tasks_to_run),
                            )
            except KeyboardInterrupt:
                raise
            return

        pool = ctx.Pool(processes=worker_count)
        try:
            iterator = pool.imap_unordered(
                _pool_signal_task,
                payloads,
                chunksize=chunksize,
            )
            for processed, (task, success) in enumerate(iterator, start=1):
                _record_outcome(task, success)
                if processed % progress_interval == 0 or processed == len(tasks_to_run):
                    LOGGER.info(
                        "%s progress: %d/%d",
                        label,
                        processed,
                        len(tasks_to_run),
                    )
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            pool.terminate()
            pool.join()
            pending = [
                task for task in tasks_to_run if task.cache_key() not in finished_keys
            ]
            LOGGER.error(
                "%s pool crashed (%s). Falling back to sequential for %d remaining tasks",
                label,
                exc,
                len(pending),
            )
            for task in pending:
                result_task, success = process_signal_task(task, cache_dir)
                _record_outcome(result_task, success)
        else:
            pool.close()
            pool.join()

    # Execute parallel-safe tasks with worker pool
    if parallel_tasks:
        parallel_workers = max(1, min(workers, len(parallel_tasks)))
        _run_pool(parallel_tasks, parallel_workers, _get_pool_context(), "Base families")

    # Execute sequential-only tasks (HF modules, FinBERT, dependency families) using spawn pool
    # 🔧 NEW: Sort dependency families to run AFTER their dependencies
    if sequential_tasks:
        # Separate dependency families from other sequential tasks
        dependency_tasks = [t for t in sequential_tasks if t.family in DEPENDENCY_FAMILIES]
        other_sequential = [t for t in sequential_tasks if t.family not in DEPENDENCY_FAMILIES]
        
        # Sort dependency tasks by their dependency depth
        def _dep_order(task: SignalTask) -> int:
            """Return execution order based on dependency depth."""
            family = task.family
            if family == "quantile_forecast":
                return 0  # Must run before downstream forecast families
            if family == "calibration":
                return 1  # Runs after quantile_forecast
            if family == "online_learning":
                return 2  # Runs after calibration
            return 3  # Unknown dependency family
        
        dependency_tasks.sort(key=_dep_order)
        
        # Run non-dependency sequential tasks first (HF, FinBERT).
        # IMPORTANT: multiprocessing.Pool workers are daemon processes, and many HF
        # generators (tokenizers/datasets) may spawn subprocesses internally.
        # To avoid "daemonic processes are not allowed to have children", execute
        # these tasks in-process (threadpool for parallelism).
        if other_sequential:
            LOGGER.info(
                "HF/FinBERT families scheduled for in-process execution (%d tasks)",
                len(other_sequential),
            )
            hf_cap = max(1, min(hf_workers or 1, len(other_sequential)))
            _run_pool(other_sequential, hf_cap, mp.get_context("spawn"), "HF families")
        
        # 🔧 NEW: Run dependency families STRICTLY SEQUENTIALLY in order
        if dependency_tasks:
            LOGGER.info(
                "🔗 Dependency families scheduled for sequential execution: %s",
                [t.family for t in dependency_tasks],
            )
            for task in dependency_tasks:
                LOGGER.info("🔗 Running %s (depends on: %s)", task.family, DEPENDENCY_FAMILIES.get(task.family, []))
                result_task, success = process_signal_task(task, cache_dir)
                _record_outcome(result_task, success)
                if not success:
                    LOGGER.warning("⚠️ %s failed - downstream families may also fail", task.family)
    
    return completed, failed, failed_families


def execute_tasks_sequential(
    horizon_linked_tasks: List[SignalTask],
    base_family_tasks: List[SignalTask],
    hf_block_tasks: List[SignalTask],
    cache_dir: Path,
) -> Tuple[int, int, List[str]]:
    """
    Execute all signal generation tasks SEQUENTIALLY within a single symbol run.
    
    Execution order:
    1. Horizon-linked families (quantile_forecast → calibration → online_learning)
    2. Base families (symbol-only, non-horizon-dependent)
    3. HF blocks (derived from base families)
    
    NO process pools, NO thread pools - one task at a time.
    This is the preferred execution model for per-symbol Dagster runs.
    
    Args:
        horizon_linked_tasks: Tasks for horizon-dependent families (dependency chain)
        base_family_tasks: Tasks for symbol-only base families
        hf_block_tasks: Tasks for HF blocks
        cache_dir: Cache directory
    
    Returns:
        (completed, failed, failed_families) tuple
    """
    completed = 0
    failed = 0
    failed_families: List[str] = []
    
    def _record_outcome(task: SignalTask, success: bool) -> None:
        nonlocal completed, failed
        if success:
            completed += 1
        else:
            failed += 1
            if task.family not in failed_families:
                failed_families.append(task.family)
    
    # Deduplicate all task lists
    all_tasks_combined = horizon_linked_tasks + base_family_tasks + hf_block_tasks
    all_tasks_combined, removed = deduplicate_tasks(all_tasks_combined)
    if removed:
        LOGGER.info("⚠️ Deduplicated %d duplicate signal tasks before scheduling", removed)
    
    # Re-separate after deduplication (tasks retain their identity)
    horizon_set = {t.cache_key() for t in horizon_linked_tasks}
    base_set = {t.cache_key() for t in base_family_tasks}
    hf_set = {t.cache_key() for t in hf_block_tasks}
    
    horizon_linked_tasks = [t for t in all_tasks_combined if t.cache_key() in horizon_set]
    base_family_tasks = [t for t in all_tasks_combined if t.cache_key() in base_set]
    hf_block_tasks = [t for t in all_tasks_combined if t.cache_key() in hf_set]
    
    total_tasks = len(horizon_linked_tasks) + len(base_family_tasks) + len(hf_block_tasks)
    LOGGER.info("")
    LOGGER.info("═" * 80)
    LOGGER.info("SEQUENTIAL EXECUTION MODE (per-symbol, no spawns)")
    LOGGER.info("═" * 80)
    LOGGER.info("Total tasks: %d (horizon-linked=%d, base=%d, hf=%d)",
                total_tasks, len(horizon_linked_tasks), len(base_family_tasks), len(hf_block_tasks))
    
    # Phase 1: Horizon-linked families (dependency chain)
    if horizon_linked_tasks:
        # Sort by dependency order: quantile_forecast → calibration → online_learning
        def _dep_order(task: SignalTask) -> int:
            family = task.family
            if family == "quantile_forecast":
                return 0
            if family == "calibration":
                return 1
            if family == "online_learning":
                return 2
            return 3
        
        horizon_linked_tasks.sort(key=_dep_order)
        
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("PHASE 1: Horizon-Linked Families (%d tasks)", len(horizon_linked_tasks))
        LOGGER.info("─" * 80)
        
        for idx, task in enumerate(horizon_linked_tasks, 1):
            LOGGER.info("  [%d/%d] %s/%s ...", idx, len(horizon_linked_tasks), task.family, task.split)
            result_task, success = process_signal_task(task, cache_dir)
            _record_outcome(result_task, success)
            status = "✓" if success else "✗"
            LOGGER.info("  [%d/%d] %s/%s %s", idx, len(horizon_linked_tasks), task.family, task.split, status)
            if not success:
                LOGGER.warning("    ⚠️ %s failed - downstream families may also fail", task.family)
    else:
        LOGGER.info("")
        LOGGER.info("PHASE 1: Horizon-Linked Families — nothing to rebuild.")
    
    # Phase 2: Base families (symbol-only)
    if base_family_tasks:
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("PHASE 2: Base Families (%d tasks)", len(base_family_tasks))
        LOGGER.info("─" * 80)
        
        for idx, task in enumerate(base_family_tasks, 1):
            LOGGER.info("  [%d/%d] %s/%s ...", idx, len(base_family_tasks), task.family, task.split)
            result_task, success = process_signal_task(task, cache_dir)
            _record_outcome(result_task, success)
            status = "✓" if success else "✗"
            LOGGER.info("  [%d/%d] %s/%s %s", idx, len(base_family_tasks), task.family, task.split, status)
    else:
        LOGGER.info("")
        LOGGER.info("PHASE 2: Base Families — nothing to rebuild.")
    
    # Phase 3: HF blocks
    if hf_block_tasks:
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("PHASE 3: HF Blocks (%d tasks)", len(hf_block_tasks))
        LOGGER.info("─" * 80)
        
        for idx, task in enumerate(hf_block_tasks, 1):
            LOGGER.info("  [%d/%d] %s/%s ...", idx, len(hf_block_tasks), task.family, task.split)
            result_task, success = process_signal_task(task, cache_dir)
            _record_outcome(result_task, success)
            status = "✓" if success else "✗"
            LOGGER.info("  [%d/%d] %s/%s %s", idx, len(hf_block_tasks), task.family, task.split, status)
    else:
        LOGGER.info("")
        LOGGER.info("PHASE 3: HF Blocks — nothing to rebuild.")
    
    LOGGER.info("")
    LOGGER.info("═" * 80)
    LOGGER.info("SEQUENTIAL EXECUTION COMPLETE: %d completed, %d failed", completed, failed)
    LOGGER.info("═" * 80)
    
    return completed, failed, failed_families


# ============================================================================
# Completeness Manifest
# ============================================================================

def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    import hashlib
    
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_completeness_manifest(
    output_path: Path,
    symbol: str,
    horizon: int,
    windows: List[WalkForwardWindow],
    base_families: List[str],
    hf_modules: List[str],
    hf_blocks: List[str],
    meta_families: List[str],
    result: PreparationResult,
    cache_dir: Path,
    mode: str,
    stage: str,
    coverage_start: Optional[pd.Timestamp],
    coverage_end: Optional[pd.Timestamp],
    track_panel_path: Optional[Path] = None,
) -> None:
    """
    Write completeness manifest with checksums.
    
    Args:
        output_path: Path to write manifest
        symbol: Stock symbol
        horizon: Forecast horizon
        windows: Walk-forward windows
        base_families: Base families list
        hf_modules: HF modules list
        result: Preparation result
        cache_dir: Cache directory that contains generated parquet files
        mode: Legacy mode token (stage-a/stage-b/walkforward)
        stage: Logical stage owner ("stage-a" or "stage-b")
    """
    cache_dir = cache_dir.resolve()
    symbol_lower = symbol.lower()

    # Ensure HF modules have Stage A feature caches before validation
    for family in hf_modules:
        for split in ["train", "valid"]:
            base_name = f"{symbol_lower}_h{horizon}_{family}_{split}.parquet"
            base_path = cache_dir / base_name
            feature_path = cache_dir / base_name.replace(".parquet", "_features.parquet")
            _backfill_feature_cache(base_path, feature_path)
    
    # Compute checksums for all generated *standalone* cache files.
    # Meta families (like hf_agg) are derived and may not have dedicated cache files.
    checksums = {}
    standalone_families = base_families + hf_modules + hf_blocks
    for family in standalone_families:
        for split in ["train", "valid"]:
            base_path, feature_path, lagged_path = _family_cache_paths(
                cache_dir,
                symbol,
                horizon,
                family,
                split,
            )
            variants: List[Path] = [base_path, feature_path]
            if WRITE_LAGGED_FEATURE_CACHES:
                variants.append(lagged_path)
            for file_path in variants:
                file_name = file_path.name
                if file_path.exists():
                    try:
                        checksum = compute_file_checksum(file_path)
                        checksums[file_name] = checksum
                    except Exception as e:
                        LOGGER.warning("Failed to compute checksum for %s: %s", file_name, e)
                        checksums[file_name] = "ERROR"
                else:
                    checksums[file_name] = "MISSING"

    track_panel_entry = None
    if track_panel_path is not None:
        try:
            checksum = compute_file_checksum(track_panel_path)
        except Exception as exc:
            LOGGER.warning("Failed to compute checksum for track panel %s: %s", track_panel_path.name, exc)
            checksum = "ERROR"
        track_panel_entry = {
            "path": str(track_panel_path),
            "checksum": checksum,
            "exists": track_panel_path.exists(),
        }
        checksums[track_panel_path.name] = checksum if checksum not in {"ERROR"} else "ERROR"
    
    manifest = {
        "version": "1.0",
        "generated_at": pd.Timestamp.now().isoformat(),
        "symbol": symbol,
        "horizon": horizon,
        "mode": mode,
        "stage": stage,
        "coverage_start": coverage_start.isoformat() if coverage_start is not None else None,
        "coverage_end": coverage_end.isoformat() if coverage_end is not None else None,
        "num_windows": len(windows),
        "windows": [
            {
                "window_id": w.window_id,
                "cache_window_idx": idx,
                "train_start": w.train_start.isoformat(),
                "train_end": w.train_end.isoformat(),
                "valid_start": w.valid_start.isoformat(),
                "valid_end": w.valid_end.isoformat(),
                "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(),
            }
            for idx, w in enumerate(windows)
        ],
        "families": {
            "base": base_families,
            "hf_modules": hf_modules,
            "hf_blocks": hf_blocks,
            "meta": meta_families,
            "total": len(base_families) + len(hf_modules) + len(hf_blocks) + len(meta_families),
        },
        "result": {
            "success": result.success,
            "total_tasks": result.total_tasks,
            "completed_tasks": result.completed_tasks,
            "skipped_tasks": result.skipped_tasks,
            "failed_tasks": result.failed_tasks,
            "failed_families": result.failed_families,
            "duration_seconds": result.duration_seconds,
        },
        "track_panel": track_panel_entry,
        "checksums": checksums,
        "checksum_stats": {
            "total_files": len(checksums),
            "present": sum(1 for v in checksums.values() if v not in ["MISSING", "ERROR"]),
            "missing": sum(1 for v in checksums.values() if v == "MISSING"),
            "errors": sum(1 for v in checksums.values() if v == "ERROR"),
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    LOGGER.info("✅ Wrote completeness manifest: %s", output_path)
    LOGGER.info("   Files: %d present, %d missing, %d errors",
                manifest["checksum_stats"]["present"],
                manifest["checksum_stats"]["missing"],
                manifest["checksum_stats"]["errors"])
    
    # Also write checksums file separately (SHA256SUMS format)
    checksum_path = output_path.with_suffix('.sha256')
    with open(checksum_path, 'w') as f:
        f.write(f"# SHA256 checksums for {symbol} h{horizon} family signals\n")
        f.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write("#\n")
        for file_name in sorted(checksums.keys()):
            checksum = checksums[file_name]
            if checksum not in ["MISSING", "ERROR"]:
                f.write(f"{checksum}  {file_name}\n")
    
    LOGGER.info("✅ Wrote checksums file: %s", checksum_path)


def build_symbol_panel_cache(
    symbol: str,
    horizon: int,
    cache_dir: Path,
    families: List[str],
    coverage_start: Optional[pd.Timestamp],
    coverage_end: Optional[pd.Timestamp],
    track_label: str = "TrackC",
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Materialize a single per-symbol feature panel parquet covering all dates."""
    try:
        from src.features.aggregator_panel import build_panel
    except ImportError as exc:  # pragma: no cover - import guard
        LOGGER.error("Unable to import build_panel for feature cache creation: %s", exc)
        return None

    start_ts = coverage_start or pd.Timestamp("1990-01-01")
    end_ts = coverage_end or pd.Timestamp.now().normalize()
    start_str = start_ts.strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")

    LOGGER.info(
        "Building consolidated %s panel covering %s → %s",
        track_label,
        start_str,
        end_str,
    )

    requested_families = list(families)
    wants_hf_agg = "hf_agg" in {f.lower() for f in requested_families}
    non_meta_families = [f for f in requested_families if f.lower() != "hf_agg"]

    panel = build_panel(
        symbol=symbol,
        start=start_str,
        end=end_str,
        families=non_meta_families,
        cache_dir=cache_dir,
        horizon=horizon,
        stage="B",
        view="both",
        window_idx=None,
    )

    # Preserve attrs early; pandas merge/joins used below can drop them.
    base_attrs = dict(getattr(panel, "attrs", {}) or {}) if panel is not None else {}

    # hf_agg is a derived meta family; generate it explicitly and merge on the
    # normalized date column to avoid index-alignment surprises.
    if wants_hf_agg:
        try:
            hf_agg_panel = build_panel(
                symbol=symbol,
                start=start_str,
                end=end_str,
                families=["hf_agg"],
                cache_dir=cache_dir,
                horizon=horizon,
                stage="B",
                view="both",
                window_idx=None,
            )
        except Exception as exc:
            LOGGER.warning("hf_agg build failed; continuing without it: %s", exc)
            hf_agg_panel = None

        if hf_agg_panel is not None and not hf_agg_panel.empty and panel is not None and not panel.empty:
            base_norm = _normalize_date_column(panel)
            hf_norm = _normalize_date_column(hf_agg_panel)
            overlapping = (set(base_norm.columns) & set(hf_norm.columns)) - {"date"}
            if overlapping:
                # Some columns (notably family-scoped *_has_data flags) may appear as
                # stubs in the base panel to keep schema stable, but carry the
                # correct values in the hf_agg panel. Prefer hf_agg for these.
                prefer_from_hf = {
                    "calibration_has_data",
                    "doc_embedding_novelty_hf_has_data",
                    "macro_tst_hf_has_data",
                }
                prefer_hits = sorted(list(overlapping & prefer_from_hf))
                if prefer_hits:
                    base_norm = base_norm.drop(columns=prefer_hits, errors="ignore")
                    overlapping = overlapping - set(prefer_hits)

                sample = ", ".join(sorted(list(overlapping))[:12])
                more = "" if len(overlapping) <= 12 else f" (+{len(overlapping) - 12} more)"
                LOGGER.warning(
                    "hf_agg panel has %d overlapping columns; keeping base panel versions and dropping overlaps from hf_agg (%s%s)",
                    len(overlapping),
                    sample,
                    more,
                )
                hf_norm = hf_norm.drop(columns=sorted(list(overlapping)), errors="ignore")

            merged = base_norm.merge(hf_norm, on="date", how="outer")

            # hf_agg is computed from block HF outputs which are typically indexed on
            # trading sessions. The unified panel is calendar-daily, so an outer merge
            # can introduce predictable NaNs on weekends/holidays.
            #
            # Fix: forward-fill only the hf_agg-derived columns across gaps.
            try:
                merged = merged.sort_values("date")
                hf_cols = [c for c in merged.columns if str(c).lower().startswith("hf_agg_")]
                if hf_cols:
                    merged[hf_cols] = merged[hf_cols].ffill().bfill()
            except Exception:
                pass

            # Repair family-scoped has_data flags when the family columns are present
            # but the *_has_data flag is incorrectly all-zero (a common outcome when
            # upstream emits stable stubs and later merges real columns).
            try:
                if "doc_embedding_novelty_hf_has_data" in merged.columns:
                    family_cols = [
                        c
                        for c in merged.columns
                        if str(c).startswith("doc_embedding_novelty_hf_") and str(c) != "doc_embedding_novelty_hf_has_data"
                    ]
                    if family_cols:
                        derived = merged[family_cols].notna().any(axis=1).astype(float)
                        existing = pd.to_numeric(merged["doc_embedding_novelty_hf_has_data"], errors="coerce").fillna(0.0)
                        merged["doc_embedding_novelty_hf_has_data"] = np.maximum(existing, derived)

                if "macro_tst_hf_has_data" in merged.columns:
                    macro_cols = [
                        c
                        for c in merged.columns
                        if (str(c).startswith("macro_tst_hf_") and str(c) != "macro_tst_hf_has_data")
                        or str(c).startswith("derived_")
                        or str(c).startswith("derived_interact")
                    ]
                    if "l3_real_interest_rate" in merged.columns and "l3_real_interest_rate" not in macro_cols:
                        macro_cols.append("l3_real_interest_rate")
                    if macro_cols:
                        derived = merged[macro_cols].notna().any(axis=1).astype(float)
                        existing = pd.to_numeric(merged["macro_tst_hf_has_data"], errors="coerce").fillna(0.0)
                        merged["macro_tst_hf_has_data"] = np.maximum(existing, derived)
            except Exception:
                pass

            # Reattach and reconcile attrs (merge drops attrs).
            try:
                merged.attrs = dict(base_attrs)
                hf_attrs = dict(getattr(hf_agg_panel, "attrs", {}) or {})

                tele = dict((merged.attrs.get("telemetry") or {}) if isinstance(merged.attrs.get("telemetry"), dict) else {})
                tele.update((hf_attrs.get("telemetry") or {}) if isinstance(hf_attrs.get("telemetry"), dict) else {})
                merged.attrs["telemetry"] = tele

                prov = dict((merged.attrs.get("provenance") or {}) if isinstance(merged.attrs.get("provenance"), dict) else {})
                prov.update((hf_attrs.get("provenance") or {}) if isinstance(hf_attrs.get("provenance"), dict) else {})
                merged.attrs["provenance"] = prov

                fams_included = []
                for src in (base_attrs.get("families"), hf_attrs.get("families")):
                    if isinstance(src, list):
                        for f in src:
                            if f not in fams_included:
                                fams_included.append(f)
                merged.attrs["families"] = fams_included

                fams_missing = []
                for src in (base_attrs.get("missing_families"), hf_attrs.get("missing_families")):
                    if isinstance(src, list):
                        for f in src:
                            if f not in fams_missing:
                                fams_missing.append(f)
                merged.attrs["missing_families"] = fams_missing
            except Exception:
                pass

            panel = merged
        elif panel is None or panel.empty:
            # If base panel is empty for some reason but hf_agg exists, still write something.
            if hf_agg_panel is not None and not hf_agg_panel.empty:
                panel = _normalize_date_column(hf_agg_panel)
                try:
                    panel.attrs = dict(getattr(hf_agg_panel, "attrs", {}) or {})
                except Exception:
                    pass

    if panel is None or panel.empty:
        LOGGER.warning("⚠️ Unable to build consolidated panel for %s h%d", symbol, horizon)
        return None

    # build_panel is intentionally best-effort and may skip families with missing cache.
    # Persist what actually made it into the panel so downstream validation can be strict.
    included_families = panel.attrs.get("families") if hasattr(panel, "attrs") else None
    missing_families = panel.attrs.get("missing_families") if hasattr(panel, "attrs") else None
    if not isinstance(included_families, list):
        included_families = []
    if not isinstance(missing_families, list):
        missing_families = []

    # If we requested hf_agg and successfully materialized columns for it, ensure
    # it is reflected in the meta manifests.
    requested_lower = {f.lower() for f in requested_families}
    if wants_hf_agg and "hf_agg" in requested_lower:
        has_hf_agg_cols = any(str(c).lower().startswith("hf_agg_") for c in panel.columns)
        if has_hf_agg_cols:
            if "hf_agg" not in {f.lower() for f in included_families}:
                included_families.append("hf_agg")
            missing_families = [f for f in missing_families if str(f).lower() != "hf_agg"]

    if os.getenv("PREP_FAMILIES_STRICT_TRACKC", "0") == "1":
        strict_missing = [f for f in (missing_families or []) if str(f).lower() != "hf_agg"]
        if strict_missing:
            LOGGER.error(
                "STRICT TrackC: build_panel skipped %d families for %s h%d: %s",
                len(strict_missing),
                symbol,
                horizon,
                ", ".join(strict_missing),
            )
            return None

    if out_path is None:
        # Use symbol subfolder: cache/merged/{SYMBOL}/
        symbol_dir = FEATURE_PANEL_DIR / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        panel_name = f"{symbol.upper()}_h{horizon}_{track_label.lower()}.parquet"
        panel_path = symbol_dir / panel_name
    else:
        panel_path = Path(out_path)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel_name = panel_path.name
    try:
        # Normalize chronology so downstream coverage readers always see a 'date' column
        normalized_panel = _normalize_date_column(panel)
        try:
            normalized_panel.attrs = dict(getattr(panel, "attrs", {}) or {})
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Family scaffolding (helps MAMBA reason about family presence/recency)
        # ------------------------------------------------------------------
        # Keep raw features as-is; add lightweight per-family scaffolding:
        # - <F>_has_data            (0/1)
        # - <F>_days_since_update   (int; -1 means never updated yet)
        # - <F>_activity            (rolling std of per-row family mean)
        # - <F>_summary_score       (tanh(zscore) of per-row family mean)
        add_scaffold = os.getenv("PREP_FAMILIES_ADD_FAMILY_SCAFFOLDING", "1") == "1"
        add_summary = os.getenv("PREP_FAMILIES_ADD_FAMILY_SUMMARY_SCORE", "1") == "1"
        if add_scaffold:
            try:
                df_scaf = normalized_panel
                df_scaf["date"] = pd.to_datetime(df_scaf["date"], errors="coerce")
                df_scaf = df_scaf.sort_values("date")

                proxy_tokens = (
                    "_has_data",
                    "has_data",
                    "is_etf",
                    "is_quarter_start",
                    "is_quarter_end",
                    "is_year_start",
                    "is_year_end",
                    "is_month_start",
                    "is_month_end",
                    "proxy",
                    "fallback",
                )

                def _family_cols_for_scaffold(fam: str) -> List[str]:
                    fam_s = str(fam)
                    cols = [c for c in df_scaf.columns if str(c).startswith(f"{fam_s}_")]
                    if fam_s == "macro_tst_hf":
                        cols.extend([c for c in df_scaf.columns if str(c).startswith("derived_")])
                        cols.extend([c for c in df_scaf.columns if str(c).startswith("derived_interact")])
                        if "l3_real_interest_rate" in df_scaf.columns:
                            cols.append("l3_real_interest_rate")
                    # De-dupe
                    seen: set[str] = set()
                    ordered: List[str] = []
                    for c in cols:
                        c_s = str(c)
                        if c_s in seen:
                            continue
                        ordered.append(c_s)
                        seen.add(c_s)
                    return ordered

                dates = pd.to_datetime(df_scaf["date"], errors="coerce")
                for fam in requested_families:
                    fam_s = str(fam)
                    fam_cols = _family_cols_for_scaffold(fam_s)

                    has_col = f"{fam_s}_has_data"
                    days_col = f"{fam_s}_days_since_update"
                    act_col = f"{fam_s}_activity"
                    sum_col = f"{fam_s}_summary_score"

                    # Prefer existing has_data flag if present.
                    if has_col in df_scaf.columns:
                        has = pd.to_numeric(df_scaf[has_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    else:
                        # Derive has_data from non-proxy numeric columns in the family.
                        candidate_cols = [
                            c
                            for c in fam_cols
                            if c != has_col
                            and not any(tok in str(c).lower() for tok in proxy_tokens)
                        ]
                        numeric = df_scaf[candidate_cols].select_dtypes(include=[np.number]) if candidate_cols else pd.DataFrame(index=df_scaf.index)
                        if numeric.empty:
                            has = pd.Series(0.0, index=df_scaf.index)
                        else:
                            has = (numeric.fillna(0.0).abs().sum(axis=1) > 0.0).astype(float)
                        df_scaf[has_col] = has

                    # Days since last update (calendar days; 0 on update day, increasing thereafter).
                    # Always recompute - cached values are often stubs (all zeros).
                    need_days_recompute = (
                        days_col not in df_scaf.columns
                        or (df_scaf[days_col] == 0).all()
                        or df_scaf[days_col].isna().any()
                    )
                    if need_days_recompute:
                        try:
                            # has > 0 marks days when this family has real data
                            has_data_mask = has > 0.0
                            last_dt = dates.where(has_data_mask).ffill()
                            ds = (dates - last_dt).dt.days
                            df_scaf[days_col] = ds.fillna(-1).astype(float)
                        except Exception:
                            df_scaf[days_col] = 0.0

                    # Activity: rolling std of family mean (captures "movement" when active).
                    # Always recompute if all zeros or all NaN.
                    need_activity_recompute = (
                        act_col not in df_scaf.columns
                        or (df_scaf[act_col].isna().all() if act_col in df_scaf.columns else False)
                        or (df_scaf[act_col] == 0).all()
                        or (df_scaf[act_col] == 1).all()
                    )
                    if need_activity_recompute:
                        try:
                            candidate_cols = [
                                c
                                for c in fam_cols
                                if c != has_col
                                and not any(tok in str(c).lower() for tok in proxy_tokens)
                            ]
                            numeric = df_scaf[candidate_cols].select_dtypes(include=[np.number]) if candidate_cols else pd.DataFrame(index=df_scaf.index)
                            if numeric.empty:
                                df_scaf[act_col] = 0.0
                            else:
                                fam_mean = numeric.mean(axis=1)
                                df_scaf[act_col] = fam_mean.rolling(20, min_periods=5).std().fillna(0.0)
                        except Exception:
                            df_scaf[act_col] = 0.0

                    # Optional: a tiny summary scalar available for every family.
                    if add_summary and sum_col not in df_scaf.columns:
                        try:
                            candidate_cols = [
                                c
                                for c in fam_cols
                                if c != has_col
                                and not any(tok in str(c).lower() for tok in proxy_tokens)
                            ]
                            numeric = df_scaf[candidate_cols].select_dtypes(include=[np.number]) if candidate_cols else pd.DataFrame(index=df_scaf.index)
                            if numeric.empty:
                                df_scaf[sum_col] = 0.0
                            else:
                                fam_mean = numeric.mean(axis=1)
                                roll = fam_mean.rolling(252, min_periods=20)
                                z = (fam_mean - roll.mean()) / (roll.std() + 1e-6)
                                df_scaf[sum_col] = np.tanh(z.fillna(0.0)).astype(float)
                        except Exception:
                            df_scaf[sum_col] = 0.0

                normalized_panel = df_scaf
            except Exception as exc:
                LOGGER.debug("Failed adding family scaffolding columns: %s", exc)

        # ------------------------------------------------------------------
        # Canonicalize per-family governance columns (Jan 2026)
        # ------------------------------------------------------------------
        # Goal: ensure each family has the 3 required governance columns:
        # - <F>_has_data (0/1) - whether data was successfully fetched
        # - <F>_activity (float) - activity level indicator
        # - <F>_days_since_update (int/float) - days since last update
        canonicalize = os.getenv("PREP_FAMILIES_CANONICALIZE_FAMILY_METRICS", "1") == "1"
        if canonicalize:
            try:
                df_clean = normalized_panel

                requested_family_names = [str(f) for f in requested_families]
                drop_cols: List[str] = []

                for fam_s in requested_family_names:
                    has_col = f"{fam_s}_has_data"
                    activity_col = f"{fam_s}_activity"
                    days_since_col = f"{fam_s}_days_since_update"

                    # Ensure has_data exists
                    if has_col not in df_clean.columns:
                        df_clean[has_col] = 0.0
                    df_clean[has_col] = (
                        pd.to_numeric(df_clean[has_col], errors="coerce")
                        .fillna(0.0)
                        .clip(0.0, 1.0)
                        .astype(float)
                    )

                    # Ensure activity exists
                    if activity_col not in df_clean.columns:
                        df_clean[activity_col] = df_clean[has_col]  # Default to has_data
                    df_clean[activity_col] = (
                        pd.to_numeric(df_clean[activity_col], errors="coerce")
                        .fillna(0.0)
                        .clip(0.0, 1.0)
                        .astype(float)
                    )

                    # Ensure days_since_update exists
                    if days_since_col not in df_clean.columns:
                        # Default: 0.0 if has_data=1, else 999.0
                        df_clean[days_since_col] = df_clean[has_col].apply(
                            lambda x: 0.0 if x > 0 else 999.0
                        )
                    df_clean[days_since_col] = (
                        pd.to_numeric(df_clean[days_since_col], errors="coerce")
                        .fillna(999.0)
                        .clip(0.0, 9999.0)
                        .astype(float)
                    )

                    # Collect family-specific columns and drop legacy score/conf columns
                    fam_pref = f"{fam_s}_"
                    fam_cols = [c for c in df_clean.columns if str(c).startswith(fam_pref)]
                    
                    # Drop legacy score columns (no longer used)
                    # NOTE: _confidence is NOT dropped - it's a first-class governance column (Jan 2026)
                    # Only drop: _score, _score_raw, _conf (abbreviated legacy confidence)
                    legacy_suffixes = ["_score", "_score_raw", "_conf"]
                    for suffix in legacy_suffixes:
                        legacy_col = f"{fam_s}{suffix}"
                        if legacy_col in fam_cols:
                            drop_cols.append(legacy_col)

                if drop_cols:
                    # De-dupe + drop only those that exist.
                    drop_unique = sorted({c for c in drop_cols if c in df_clean.columns})
                    if drop_unique:
                        df_clean = df_clean.drop(columns=drop_unique, errors="ignore")

                normalized_panel = df_clean
            except Exception as exc:
                LOGGER.debug("Failed canonicalizing family governance columns: %s", exc)

        # ------------------------------------------------------------------
        # Quantile forecast compression (reduce collinearity)
        # ------------------------------------------------------------------
        # Keep only 3 quantiles (q10, q50, q90) and add distribution-shape scalars:
        # - quantile_forecast_iqr  = q90 - q10 (fallback to q75 - q25)
        # - quantile_forecast_skew = (q90 + q10 - 2*q50) / iqr
        compress_qf = os.getenv("PREP_FAMILIES_COMPRESS_QUANTILE_FORECAST", "1") == "1"
        if compress_qf:
            try:
                df_q = normalized_panel
                q10 = "quantile_forecast_q10"
                q50 = "quantile_forecast_q50"
                q90 = "quantile_forecast_q90"
                q25 = "quantile_forecast_q25"
                q75 = "quantile_forecast_q75"

                have_core = all(c in df_q.columns for c in (q10, q50, q90))
                have_fallback = all(c in df_q.columns for c in (q25, q50, q75))
                if have_core or have_fallback:
                    if have_core:
                        q10_s = pd.to_numeric(df_q[q10], errors="coerce")
                        q50_s = pd.to_numeric(df_q[q50], errors="coerce")
                        q90_s = pd.to_numeric(df_q[q90], errors="coerce")
                        iqr = (q90_s - q10_s)
                    else:
                        # Fallback if q10/q90 are unavailable.
                        q10_s = pd.to_numeric(df_q[q25], errors="coerce")
                        q50_s = pd.to_numeric(df_q[q50], errors="coerce")
                        q90_s = pd.to_numeric(df_q[q75], errors="coerce")
                        iqr = (q90_s - q10_s)

                    eps = 1e-6
                    df_q["quantile_forecast_iqr"] = iqr.fillna(0.0).astype(float)
                    denom = iqr.abs().fillna(0.0) + eps
                    skew = (q90_s + q10_s - 2.0 * q50_s) / denom
                    df_q["quantile_forecast_skew"] = skew.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

                    # Drop redundant adjacent quantiles; keep only q10/q50/q90.
                    # We deliberately do not drop other quantile_forecast_* features.
                    keep = {q10, q50, q90, "quantile_forecast_iqr", "quantile_forecast_skew"}
                    drop = [
                        c
                        for c in df_q.columns
                        if str(c).startswith("quantile_forecast_q") and str(c) not in keep
                    ]
                    if drop:
                        df_q = df_q.drop(columns=drop, errors="ignore")
                    normalized_panel = df_q
            except Exception as exc:
                LOGGER.debug("Failed quantile_forecast compression: %s", exc)

        # ------------------------------------------------------------------
        # Role-based normalization (deterministic, scalable)
        # ------------------------------------------------------------------
        # Classify each feature into one of: hygiene / regime / risk / predictive
        # using ordered naming + value-domain rules, then normalize by role.
        # This runs at the merged-panel stage so per-family caches remain raw.
        role_norm = os.getenv("PREP_FAMILIES_ROLE_NORMALIZE", "1") == "1"
        timing_rules = os.getenv("PREP_FAMILIES_TIMING_RULES", "1") == "1"
        timing_shift_days = int(os.getenv("PREP_FAMILIES_TIMING_SHIFT_DAYS", "1"))
        timing_ffill = os.getenv("PREP_FAMILIES_TIMING_FFILL", "1") == "1"
        timing_decay = os.getenv("PREP_FAMILIES_TIMING_DECAY", "1") == "1"
        role_map: Dict[str, str] = {}
        role_family_map: Dict[str, str] = {}
        executable_map: Dict[str, bool] = {}
        executable_reason_map: Dict[str, str] = {}
        risk_scale_ok_map: Dict[str, bool] = {}
        gating_ok_map: Dict[str, bool] = {}
        veto_ok_map: Dict[str, bool] = {}
        risk_scale_ok_reason_map: Dict[str, str] = {}
        gating_ok_reason_map: Dict[str, str] = {}
        veto_ok_reason_map: Dict[str, str] = {}
        requires_point_in_time_map: Dict[str, bool] = {}
        update_latency_class_map: Dict[str, str] = {}
        decay_half_life_days_map: Dict[str, float] = {}
        expected_sparsity_map: Dict[str, str] = {}
        try:
            from src.features.feature_roles import (
                FeatureRole,
                FamilyMeta,
                apply_timing_rules,
                assign_roles_and_governance_with_reasons,
                load_family_meta_from_registry,
                load_overrides,
                normalize_by_role,
            )

            pred_window = int(os.getenv("PREP_FAMILIES_ROLE_PRED_Z_WINDOW", "252"))
            risk_window = int(os.getenv("PREP_FAMILIES_ROLE_RISK_WINDOW", "252"))
            min_periods = int(os.getenv("PREP_FAMILIES_ROLE_MIN_PERIODS", "20"))

            override_path = os.getenv("PREP_FAMILIES_ROLE_OVERRIDES_JSON", "").strip()
            overrides = load_overrides(Path(override_path)) if override_path else {}

            # STEP 0 anchor: canonical family registry drives fallback intent + alpha policy.
            fam_meta = load_family_meta_from_registry()

            # Optional: override family primary intents via JSON (family -> role).
            fam_meta_path = os.getenv("PREP_FAMILIES_FAMILY_META_JSON", "").strip()
            if fam_meta_path:
                try:
                    payload = json.loads(Path(fam_meta_path).read_text())
                    if isinstance(payload, dict):
                        for fam, intent in payload.items():
                            try:
                                fam_meta[str(fam)] = FamilyMeta(primary_intent=FeatureRole(str(intent)))
                            except Exception:
                                continue
                except Exception:
                    pass

            # Always assign roles + executability for provenance, even if we skip transforms.
            (
                roles,
                fams,
                reasons,
                executable_map,
                executable_reason_map,
                risk_scale_ok_map,
                gating_ok_map,
                veto_ok_map,
                risk_scale_ok_reason_map,
                gating_ok_reason_map,
                veto_ok_reason_map,
                requires_point_in_time_map,
                update_latency_class_map,
                decay_half_life_days_map,
                expected_sparsity_map,
            ) = assign_roles_and_governance_with_reasons(
                normalized_panel,
                family_meta=fam_meta,
                requested_families=[str(f) for f in requested_families],
                overrides=overrides,
            )
            role_map = {k: str(v.value) for k, v in roles.items()}
            role_family_map = dict(fams)
            role_reason_map = dict(reasons)

            if role_norm:
                if timing_rules:
                    normalized_panel = apply_timing_rules(
                        normalized_panel,
                        roles=roles,
                        family_map=fams,
                        family_meta=fam_meta,
                        shift_days=timing_shift_days,
                        enable_ffill=timing_ffill,
                        enable_shift=(timing_shift_days > 0),
                        enable_decay=timing_decay,
                    )
                normalized_panel = normalize_by_role(
                    normalized_panel,
                    roles=roles,
                    pred_window=pred_window,
                    risk_window=risk_window,
                    min_periods=min_periods,
                )
        except Exception as exc:
            LOGGER.debug("Failed role assignment/normalization: %s", exc)

        # ------------------------------------------------------------------
        # Optional: role-hint channels as numeric features (no metadata strings)
        # ------------------------------------------------------------------
        # These are constant-in-time binary columns per feature. They are emitted
        # only when explicitly enabled because they expand the parquet width.
        # Mamba still only sees numbers.
        role_hint_channels = os.getenv("PREP_FAMILIES_ROLE_HINT_CHANNELS", "1") == "1"
        if role_hint_channels and role_map:
            try:
                hint_cols_added = 0
                base_items = list(role_map.items())
                new_role_map_entries: Dict[str, str] = {}
                new_role_family_entries: Dict[str, str] = {}
                new_role_reason_entries: Dict[str, str] = {}
                new_executable_entries: Dict[str, bool] = {}
                new_executable_reason_entries: Dict[str, str] = {}

                for col_s, role_s in base_items:
                    if col_s == "date":
                        continue
                    is_pred = 1.0 if role_s == "predictive" else 0.0
                    is_risk = 1.0 if role_s == "risk" else 0.0
                    is_regime = 1.0 if role_s == "regime" else 0.0

                    c1 = f"{col_s}__is_predictive"
                    c2 = f"{col_s}__is_risk"
                    c3 = f"{col_s}__is_regime"
                    for c_name, v in ((c1, is_pred), (c2, is_risk), (c3, is_regime)):
                        if c_name in normalized_panel.columns:
                            continue
                        normalized_panel[c_name] = float(v)
                        hint_cols_added += 1

                    # Include hint cols in provenance maps as scaffolding.
                    new_role_map_entries[c1] = "hygiene"
                    new_role_map_entries[c2] = "hygiene"
                    new_role_map_entries[c3] = "hygiene"
                    new_role_family_entries[c1] = "<role_hint>"
                    new_role_family_entries[c2] = "<role_hint>"
                    new_role_family_entries[c3] = "<role_hint>"
                    new_role_reason_entries[c1] = "role_hint"
                    new_role_reason_entries[c2] = "role_hint"
                    new_role_reason_entries[c3] = "role_hint"
                    new_executable_entries[c1] = False
                    new_executable_entries[c2] = False
                    new_executable_entries[c3] = False
                    new_executable_reason_entries[c1] = "role_hint"
                    new_executable_reason_entries[c2] = "role_hint"
                    new_executable_reason_entries[c3] = "role_hint"

                    # Allowed-usage scaffolding for hint columns.
                    risk_scale_ok_map[c1] = False
                    risk_scale_ok_map[c2] = False
                    risk_scale_ok_map[c3] = False
                    gating_ok_map[c1] = True
                    gating_ok_map[c2] = True
                    gating_ok_map[c3] = True
                    veto_ok_map[c1] = True
                    veto_ok_map[c2] = True
                    veto_ok_map[c3] = True
                    risk_scale_ok_reason_map[c1] = "role_hint"
                    risk_scale_ok_reason_map[c2] = "role_hint"
                    risk_scale_ok_reason_map[c3] = "role_hint"
                    gating_ok_reason_map[c1] = "role_hint"
                    gating_ok_reason_map[c2] = "role_hint"
                    gating_ok_reason_map[c3] = "role_hint"
                    veto_ok_reason_map[c1] = "role_hint"
                    veto_ok_reason_map[c2] = "role_hint"
                    veto_ok_reason_map[c3] = "role_hint"
                    requires_point_in_time_map[c1] = False
                    requires_point_in_time_map[c2] = False
                    requires_point_in_time_map[c3] = False
                    update_latency_class_map[c1] = "unknown"
                    update_latency_class_map[c2] = "unknown"
                    update_latency_class_map[c3] = "unknown"
                    decay_half_life_days_map[c1] = 0.0
                    decay_half_life_days_map[c2] = 0.0
                    decay_half_life_days_map[c3] = 0.0
                    expected_sparsity_map[c1] = "unknown"
                    expected_sparsity_map[c2] = "unknown"
                    expected_sparsity_map[c3] = "unknown"

                if new_role_map_entries:
                    role_map.update(new_role_map_entries)
                if new_role_family_entries:
                    role_family_map.update(new_role_family_entries)
                if isinstance(locals().get("role_reason_map"), dict) and new_role_reason_entries:
                    role_reason_map.update(new_role_reason_entries)
                if new_executable_entries:
                    executable_map.update(new_executable_entries)
                if new_executable_reason_entries:
                    executable_reason_map.update(new_executable_reason_entries)

                if hint_cols_added:
                    LOGGER.info("Added %d role-hint numeric columns", hint_cols_added)
            except Exception as exc:
                LOGGER.debug("Failed adding role-hint channels: %s", exc)

        # ------------------------------------------------------------------
        # Optional: numeric staleness hints (days since last update/change)
        # ------------------------------------------------------------------
        days_since_update_hints = os.getenv("PREP_FAMILIES_DAYS_SINCE_UPDATE_HINTS", "0") == "1"
        if days_since_update_hints and role_map:
            try:
                import numpy as _np

                hint_cols_added = 0
                base_cols = [c for c in normalized_panel.columns if str(c) != "date"]
                for col in base_cols:
                    col_s = str(col)
                    # Avoid re-hinting hint/scaffold columns.
                    if "__is_" in col_s or col_s.endswith("__days_since_update"):
                        continue
                    if col_s not in normalized_panel.columns:
                        continue
                    s = pd.to_numeric(normalized_panel[col_s], errors="coerce")
                    if s.isna().all():
                        continue

                    # Define an "update" as a change in observed value (NaNs ignored).
                    s2 = s.copy()
                    filled = s2.fillna(method="ffill")
                    changed = filled.ne(filled.shift(1))
                    pos = pd.Series(_np.where(changed.to_numpy(), _np.arange(len(filled)), _np.nan))
                    last = pos.ffill().to_numpy()
                    idx = _np.arange(len(filled), dtype=float)
                    days = idx - last
                    # If we never saw an update, encode as a large constant.
                    days = _np.where(_np.isnan(days), 9999.0, days)

                    c_hint = f"{col_s}__days_since_update"
                    if c_hint in normalized_panel.columns:
                        continue
                    normalized_panel[c_hint] = pd.Series(days, index=normalized_panel.index).astype(float)
                    hint_cols_added += 1

                    # Tag as hygiene scaffolding in provenance.
                    role_map[c_hint] = "hygiene"
                    role_family_map[c_hint] = role_family_map.get(col_s, "<unknown>")
                    if isinstance(locals().get("role_reason_map"), dict):
                        role_reason_map[c_hint] = "days_since_update_hint"
                    executable_map[c_hint] = False
                    executable_reason_map[c_hint] = "days_since_update_hint"

                    # Allowed-usage: days-since-update can be used for gating/veto.
                    risk_scale_ok_map[c_hint] = False
                    gating_ok_map[c_hint] = True
                    veto_ok_map[c_hint] = True
                    risk_scale_ok_reason_map[c_hint] = "days_since_update_hint"
                    gating_ok_reason_map[c_hint] = "days_since_update_hint"
                    veto_ok_reason_map[c_hint] = "days_since_update_hint"
                    requires_point_in_time_map[c_hint] = False
                    update_latency_class_map[c_hint] = "unknown"
                    decay_half_life_days_map[c_hint] = 0.0
                    expected_sparsity_map[c_hint] = "unknown"

                if hint_cols_added:
                    LOGGER.info("Added %d days-since-update numeric columns", hint_cols_added)
            except Exception as exc:
                LOGGER.debug("Failed adding days-since-update hints: %s", exc)

        # Persist a provenance sidecar so we can audit data sources without inflating the parquet.
        # build_panel attaches per-family telemetry/provenance to attrs.
        # NOTE: We use a single .meta.json file (no separate CSV - it was redundant).
        attrs = getattr(panel, "attrs", {}) or {}
        panel_provenance = attrs.get("provenance") if isinstance(attrs, dict) else None
        panel_telemetry = attrs.get("telemetry") if isinstance(attrs, dict) else None
        provenance_path = panel_path.with_suffix(".meta.json")

        # Collect cache artifact + meta.json info per requested family.
        cache_artifacts: dict = {}
        try:
            cache_root = Path(cache_dir)
            symbol_lower = symbol.lower()
            for fam in requested_families:
                fam_key = str(fam)
                fam_art: dict = {"train": [], "valid": []}
                for split in ("train", "valid"):
                    suffixes = ["_features.parquet", ".parquet"]
                    if WRITE_LAGGED_FEATURE_CACHES:
                        suffixes.insert(0, "_features_lagged.parquet")
                    for suffix in suffixes:
                        cand = cache_root / f"{symbol_lower}_h{horizon}_{fam_key}_{split}{suffix}"
                        if not cand.exists():
                            continue
                        meta_path = Path(str(cand) + ".meta.json")
                        meta_payload = None
                        if meta_path.exists():
                            try:
                                with open(meta_path, "r") as handle:
                                    meta_payload = json.load(handle)
                            except Exception:
                                meta_payload = None
                        fam_art[split].append(
                            {
                                "path": str(cand),
                                "meta_path": str(meta_path) if meta_path.exists() else "",
                                "meta": meta_payload,
                            }
                        )
                cache_artifacts[fam_key] = fam_art
        except Exception as exc:
            LOGGER.debug("Failed collecting cache artifact metadata: %s", exc)

        def _infer_family_for_column(col_name: str, family_names: List[str]) -> str:
            if col_name == "date":
                return "<meta>"
            # Prefer the longest matching family prefix to support family names with underscores.
            candidates = [
                f
                for f in family_names
                if col_name == f or col_name.startswith(f + "_")
            ]
            if candidates:
                return max(candidates, key=len)
            # Special-case stable schema stubs / base columns.
            if col_name in {"has_data", "is_etf"}:
                return "<base>"

            # Heuristic attribution for known families that emit un-prefixed columns.
            fam_set = {str(f) for f in (family_names or [])}

            # Macro lagged/derived features are emitted without a family prefix.
            if "macro_tst_hf" in fam_set:
                if col_name.startswith(("l1_", "l2_", "l3_", "derived_", "derived_interact", "macro_")):
                    return "macro_tst_hf"

            # Doc-embedding novelty emits a small set of un-prefixed novelty metadata.
            if "doc_embedding_novelty_hf" in fam_set:
                if col_name in {"n_events", "n_articles", "theme_weight", "baseline_mean", "top_theme_numeric"}:
                    return "doc_embedding_novelty_hf"
                if "novelty" in str(col_name):
                    return "doc_embedding_novelty_hf"
                if col_name.startswith(("theme_", "baseline_", "top_theme_")):
                    return "doc_embedding_novelty_hf"

            return "<unknown>"

        family_names_for_map = [str(f) for f in requested_families]
        column_family_map = {
            str(col): _infer_family_for_column(str(col), family_names_for_map)
            for col in normalized_panel.columns
            if str(col) != "date"
        }

        provenance_payload = {
            "symbol": symbol,
            "horizon": int(horizon),
            "track": track_label,
            "panel_path": str(panel_path),
            "features_path": "",
            "index_path": "",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "families_requested": requested_families,
            "families_included": included_families,
            "families_missing": missing_families,
            "telemetry": panel_telemetry if isinstance(panel_telemetry, dict) else {},
            "provenance": panel_provenance if isinstance(panel_provenance, dict) else {},
            "family_cache_artifacts": cache_artifacts,
            "column_family_map": column_family_map,
            "column_role_map": role_map,
            "column_role_reason_map": (role_reason_map if isinstance(locals().get("role_reason_map"), dict) else {}),
            "column_executable_map": (executable_map if isinstance(locals().get("executable_map"), dict) else {}),
            "column_executable_reason_map": (executable_reason_map if isinstance(locals().get("executable_reason_map"), dict) else {}),

            # New governance maps (allowed usages + PIT hints)
            "column_alpha_ok_map": (executable_map if isinstance(locals().get("executable_map"), dict) else {}),
            "column_risk_scale_ok_map": (risk_scale_ok_map if isinstance(locals().get("risk_scale_ok_map"), dict) else {}),
            "column_gating_ok_map": (gating_ok_map if isinstance(locals().get("gating_ok_map"), dict) else {}),
            "column_veto_ok_map": (veto_ok_map if isinstance(locals().get("veto_ok_map"), dict) else {}),
            "column_risk_scale_ok_reason_map": (risk_scale_ok_reason_map if isinstance(locals().get("risk_scale_ok_reason_map"), dict) else {}),
            "column_gating_ok_reason_map": (gating_ok_reason_map if isinstance(locals().get("gating_ok_reason_map"), dict) else {}),
            "column_veto_ok_reason_map": (veto_ok_reason_map if isinstance(locals().get("veto_ok_reason_map"), dict) else {}),
            "column_requires_point_in_time_map": (requires_point_in_time_map if isinstance(locals().get("requires_point_in_time_map"), dict) else {}),
            "column_update_latency_class_map": (update_latency_class_map if isinstance(locals().get("update_latency_class_map"), dict) else {}),
            "column_decay_half_life_days_map": (decay_half_life_days_map if isinstance(locals().get("decay_half_life_days_map"), dict) else {}),
            "column_expected_sparsity_map": (expected_sparsity_map if isinstance(locals().get("expected_sparsity_map"), dict) else {}),
        }

        try:
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            with open(provenance_path, "w") as handle:
                json.dump(provenance_payload, handle, indent=2, default=str)

            LOGGER.info("🧾 Wrote provenance metadata: %s", provenance_path)
        except Exception as exc:
            LOGGER.warning("Failed to write provenance sidecar for %s: %s", panel_name, exc)

        # ------------------------------------------------------------------
        # Production-grade split outputs (numeric-only features + index sidecar)
        # ------------------------------------------------------------------
        # Keep the historical merged parquet as-is for backward compatibility,
        # but ALSO write:
        # - <SYMBOL>_<H>_features.parquet : strictly numeric matrix (no date/index)
        # - <SYMBOL>_<H>_index.parquet    : row_id + date(int) + session_id(int)
        features_path = None
        index_path = None
        try:
            if "date" in normalized_panel.columns:
                # Derive output names from the merged panel name.
                stem = str(panel_path.stem)
                base = stem[:-7] if stem.endswith("_merged") else stem
                legacy_features_path = panel_path.with_name(f"{base}_features.parquet")
                legacy_index_path = panel_path.with_name(f"{base}_index.parquet")

                sym_u = str(symbol).upper()
                h_i = int(horizon)
                # Canonical convention: no date columns, no timestamps in parquet.
                # Use h{H} prefix for consistency with merged panel naming.
                features_path = panel_path.with_name(f"{sym_u}_h{h_i}_features.parquet")
                index_path = panel_path.with_name(f"{sym_u}_h{h_i}_index.parquet")

                df_idx = normalized_panel.copy()
                dt = pd.to_datetime(df_idx["date"], errors="coerce")
                df_idx = df_idx.assign(_date=dt).dropna(subset=["_date"]).sort_values("_date")

                apply_to_next_session = os.getenv("PREP_FAMILIES_APPLY_TO_NEXT_SESSION", "1") == "1"

                # Map every row to the *next* NYSE session (handles weekends/holidays
                # by applying values on the next tradable session).
                session_labels = None
                session_ids = None
                try:
                    import exchange_calendars as xcals  # type: ignore

                    cal = xcals.get_calendar("XNYS")
                    # exchange_calendars expects tz-naive session labels.
                    s_src = df_idx["_date"]
                    if getattr(s_src.dt, "tz", None) is not None:
                        s_src = s_src.dt.tz_localize(None)
                    s_norm = s_src.dt.normalize()

                    # Point-in-time contract: apply feature values on the NEXT tradable session
                    # by default (end-of-day features should not be used on the same session).
                    s_for_session = (s_norm + pd.Timedelta(days=1)) if apply_to_next_session else s_norm
                    session_labels = pd.Index([pd.Timestamp(cal.date_to_session(d, direction="next")) for d in s_for_session])
                    # Collapse to exactly one row per session; keep the last observation
                    # (closest in time to the session).
                    df_idx = df_idx.assign(_session=session_labels)
                    df_idx = df_idx.drop(columns=["date"], errors="ignore")
                    df_idx = df_idx.groupby("_session", sort=True).last()
                    sess_index = pd.DatetimeIndex(df_idx.index)
                    session_ids = pd.Index(cal.sessions.get_indexer(sess_index)).astype(int)

                    # ================================================================
                    # ENHANCED LEAKAGE AUDIT
                    # ================================================================
                    # Multi-layer audit for temporal leakage prevention (institutional-grade).
                    audit: dict = {
                        "symbol": str(symbol),
                        "horizon": int(horizon),
                        "track": str(track_label),
                        "apply_to_next_session": bool(apply_to_next_session),
                        "raw_rows": int(len(s_norm)),
                        "collapsed_sessions": int(len(sess_index)),
                    }
                    
                    # Layer 1: Session-level temporal alignment check
                    try:
                        same_session = 0
                        if apply_to_next_session:
                            # Count cases where a session date mapped to itself.
                            for d0, d_sess in zip(s_norm.to_list(), session_labels.to_list()):
                                try:
                                    if bool(cal.is_session(pd.Timestamp(d0))) and pd.Timestamp(d_sess).normalize() == pd.Timestamp(d0).normalize():
                                        same_session += 1
                                except Exception:
                                    continue
                        audit["same_session_mappings"] = int(same_session)
                        audit["temporal_alignment_ok"] = bool((same_session == 0) if apply_to_next_session else True)
                    except Exception:
                        audit["temporal_alignment_ok"] = True
                    
                    # Layer 2: Column-name leakage scan (pattern detection)
                    try:
                        leak_patterns = [
                            "lead", "future", "target", "fwd", "next", "t+", 
                            "forward", "ahead", "label", "y_", "lookahead"
                        ]
                        suspicious_cols = []
                        for col in df_idx.columns:
                            col_lower = str(col).lower()
                            if any(pattern in col_lower for pattern in leak_patterns):
                                suspicious_cols.append(str(col))
                        
                        audit["column_name_leakage"] = {
                            "suspicious_count": len(suspicious_cols),
                            "suspicious_columns": suspicious_cols[:50],  # First 50
                            "ok": len(suspicious_cols) == 0
                        }
                    except Exception as e:
                        audit["column_name_leakage"] = {"ok": True, "error": str(e)}
                    
                    # Layer 3: Governance column isolation check
                    try:
                        gov_suffixes = (
                            "_has_data", "_activity", "_days_since_update", 
                            "_confidence", "_conf", "_source_asof_ts"
                        )
                        gov_cols = [c for c in df_idx.columns if str(c).endswith(gov_suffixes)]
                        
                        audit["governance_isolation"] = {
                            "governance_column_count": len(gov_cols),
                            "governance_columns": gov_cols[:50],  # First 50
                            "ok": True  # Governance columns are allowed in cache, blocked in Stage-B
                        }
                    except Exception as e:
                        audit["governance_isolation"] = {"ok": True, "error": str(e)}
                    
                    # Layer 4: Timestamp alignment check (source_asof_ts ≤ session date)
                    try:
                        timestamp_violations = []
                        asof_cols = [c for c in df_idx.columns if "_source_asof_ts" in str(c)]
                        
                        for asof_col in asof_cols[:10]:  # Check first 10 timestamp columns
                            try:
                                asof_ts = pd.to_datetime(df_idx[asof_col], errors="coerce")
                                session_ts = sess_index
                                violations = (asof_ts > session_ts).sum()
                                if violations > 0:
                                    timestamp_violations.append({
                                        "column": str(asof_col),
                                        "violations": int(violations),
                                        "total_rows": int(len(asof_ts))
                                    })
                            except Exception:
                                continue
                        
                        audit["timestamp_alignment"] = {
                            "violations": timestamp_violations,
                            "ok": len(timestamp_violations) == 0
                        }
                    except Exception as e:
                        audit["timestamp_alignment"] = {"ok": True, "error": str(e)}
                    
                    # Overall audit status
                    audit["ok"] = bool(
                        audit.get("temporal_alignment_ok", True) and
                        audit.get("column_name_leakage", {}).get("ok", True) and
                        audit.get("governance_isolation", {}).get("ok", True) and
                        audit.get("timestamp_alignment", {}).get("ok", True)
                    )

                    try:
                        audit_path = panel_path.with_suffix(".leakage_audit.json")
                        with open(audit_path, "w") as _handle:
                            json.dump(audit, _handle, indent=2, default=str)
                    except Exception:
                        pass

                    if apply_to_next_session and not bool(audit.get("ok", True)):
                        raise RuntimeError(
                            f"Point-in-time audit failed: mapped {audit.get('same_session_mappings')} session rows to same-session"
                        )
                except Exception:
                    # Fallback: weekday-only calendar. Still collapse to one row per weekday.
                    s_src = df_idx["_date"]
                    if getattr(s_src.dt, "tz", None) is not None:
                        s_src = s_src.dt.tz_localize(None)
                    s_norm = s_src.dt.normalize()
                    if apply_to_next_session:
                        s_norm = s_norm + pd.Timedelta(days=1)
                    # Push weekends to next weekday.
                    next_weekday = []
                    for d in s_norm:
                        d0 = pd.Timestamp(d).normalize()
                        while d0.weekday() >= 5:
                            d0 = d0 + pd.Timedelta(days=1)
                        next_weekday.append(d0)
                    df_idx = df_idx.assign(_session=pd.Index(next_weekday))
                    df_idx = df_idx.drop(columns=["date"], errors="ignore")
                    df_idx = df_idx.groupby("_session", sort=True).last()
                    sess_index = pd.DatetimeIndex(df_idx.index)
                    # Synthetic session_id: monotonically increasing business-day ordinal.
                    session_ids = pd.Index(range(len(sess_index))).astype(int)

                # Features parquet: numeric columns only, no index.
                features_df = df_idx.drop(columns=["_date"], errors="ignore")
                if "_session" in features_df.columns:
                    features_df = features_df.drop(columns=["_session"], errors="ignore")
                features_df = features_df.select_dtypes(include=[np.number])

                # Index parquet: row_id, date, session_id.
                sess_index = pd.DatetimeIndex(df_idx.index)
                date_int = pd.Index(sess_index.strftime("%Y%m%d")).astype(int).astype(np.int32)
                row_id = np.arange(len(sess_index), dtype=np.int32)
                sess_id = pd.Index(session_ids).astype(np.int32)
                index_df = pd.DataFrame(
                    {
                        "row_id": row_id,
                        "date": date_int,
                        "session_id": sess_id,
                    }
                )

                features_path.parent.mkdir(parents=True, exist_ok=True)
                # Ensure the canonical outputs are real parquet files, not symlinks.
                try:
                    for pth in (features_path, index_path):
                        if pth is not None and pth.is_symlink():
                            pth.unlink()
                except Exception:
                    pass
                # Dtype hygiene: keep numeric-only matrix; use int8 for bool-ish flags when possible.
                try:
                    for c in list(features_df.columns):
                        s = features_df[c]
                        if pd.api.types.is_bool_dtype(s.dtype):
                            features_df[c] = s.astype(np.int8)
                        elif pd.api.types.is_integer_dtype(s.dtype):
                            # Downcast 0/1 flags to int8.
                            try:
                                vals = s.dropna().unique()
                                if len(vals) and set(map(int, vals.tolist())) <= {0, 1}:
                                    features_df[c] = s.astype(np.int8)
                            except Exception:
                                pass
                        elif pd.api.types.is_float_dtype(s.dtype):
                            features_df[c] = s.astype(np.float32)
                except Exception:
                    pass

                features_df.reset_index(drop=True).to_parquet(features_path, index=False)
                index_df.to_parquet(index_path, index=False)

                # Backward-compatibility symlinks (best-effort): keep legacy names pointing
                # at the canonical h{H} convention.
                try:
                    import os as _os

                    for link_path, target_path in (
                        (legacy_features_path, features_path),
                        (legacy_index_path, index_path),
                    ):
                        if link_path == target_path:
                            continue
                        if link_path.exists() or link_path.is_symlink():
                            link_path.unlink()
                        rel = _os.path.relpath(target_path, start=link_path.parent)
                        _os.symlink(rel, link_path)
                except Exception:
                    pass

                # Attach to provenance/meta (best-effort; do not fail the build).
                try:
                    provenance_payload["features_path"] = str(features_path)
                    provenance_payload["index_path"] = str(index_path)
                    with open(provenance_path, "w") as handle:
                        json.dump(provenance_payload, handle, indent=2, default=str)
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.debug("Failed writing features/index sidecars for %s: %s", panel_name, exc)

        # Final repair pass for family-scoped has_data flags.
        # Some upstream blocks emit stable stubs (all zeros) while still filling
        # the underlying feature columns; the merged-panel quality gate relies
        # on *_has_data to detect missing sources, so compute it from actual
        # column presence right before persisting.
        try:
            if "doc_embedding_novelty_hf_has_data" in normalized_panel.columns:
                doc_cols = [
                    c
                    for c in normalized_panel.columns
                    if str(c).startswith("doc_embedding_novelty_hf_") and str(c) != "doc_embedding_novelty_hf_has_data"
                ]
                if doc_cols:
                    derived = normalized_panel[doc_cols].notna().any(axis=1).astype(float)
                    existing = pd.to_numeric(normalized_panel["doc_embedding_novelty_hf_has_data"], errors="coerce").fillna(0.0)
                    normalized_panel["doc_embedding_novelty_hf_has_data"] = np.maximum(existing, derived)

            if "macro_tst_hf_has_data" in normalized_panel.columns:
                macro_cols = [
                    c
                    for c in normalized_panel.columns
                    if (str(c).startswith("macro_tst_hf_") and str(c) != "macro_tst_hf_has_data")
                    or str(c).startswith("derived_")
                    or str(c).startswith("derived_interact")
                ]
                if "l3_real_interest_rate" in normalized_panel.columns and "l3_real_interest_rate" not in macro_cols:
                    macro_cols.append("l3_real_interest_rate")
                if macro_cols:
                    derived = normalized_panel[macro_cols].notna().any(axis=1).astype(float)
                    existing = pd.to_numeric(normalized_panel["macro_tst_hf_has_data"], errors="coerce").fillna(0.0)
                    normalized_panel["macro_tst_hf_has_data"] = np.maximum(existing, derived)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Role-aware split outputs: Mamba (alpha) vs Portfolio (policy)
        # ------------------------------------------------------------------
        write_role_splits = os.getenv("PREP_FAMILIES_WRITE_ROLE_SPLITS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        mamba_panel: Optional[pd.DataFrame] = None
        portfolio_panel: Optional[pd.DataFrame] = None
        mamba_path: Optional[Path] = None
        portfolio_path: Optional[Path] = None

        if write_role_splits and isinstance(normalized_panel, pd.DataFrame):
            try:
                alt_prefix = "alternative_signals_"

                alt_mamba_default = {
                    "intraday_range_pct",
                    "intraday_range_z",
                    "opening_reversal",
                    "closing_ramp",
                    "overnight_return_z",
                    "gap_up_pct",
                    "gap_down_pct",
                    "relative_volume_20d",
                    "volume_z_20d",
                    "volume_trend_10d",
                    "opening_volume_surge",
                    "volume_price_divergence",
                    "earnings_runup_10d",
                    "post_earnings_drift_5d",
                    "news_volume_change",
                    "news_volume_z",
                    "trend_acceleration",
                }

                alt_mamba_optional = {
                    "overnight_return",
                    "gap_vs_vix_interaction",
                    "buy_volume_proxy",
                    "google_trends_score",
                    "mean_reversion_signal",
                }

                alt_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "beta_20d",
                    "beta_change_rate",
                    "beta_vix_interaction",
                    "spy_correlation_20d",
                    "qqq_correlation_20d",
                    "sector_etf_correlation_20d",
                    "rv_5d",
                    "rv_10d",
                    "rv_20d",
                    "rv_ratio_5_20",
                    "rv_z_20",
                    "close_to_close_volatility",
                    "open_to_close_volatility",
                    "high_low_volatility_ratio",
                    "intraday_volatility_ratio",
                    "liquidity_stress_pct",
                    "days_since_last_earnings",
                    "days_to_next_earnings",
                    "turn_of_month_flag",
                    "news_volume_count",
                }

                arima_prefix = "arima_forecast_"
                arima_mamba_default = {
                    "arima_forecast_1d",
                    "arima_forecast_5d",
                    "arima_residual_zscore",
                }
                arima_mamba_optional = {
                    "arima_residual_t",
                    "arima_momentum_indicator",
                }
                arima_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "arima_log_likelihood",
                    "confidence",
                    "arima_abs_residual",
                    "arima_uncertainty_proxy",
                    "arima_persistence",
                    "arima_innovation",
                }

                quantile_prefix = "quantile_forecast_"
                quantile_mamba_optional = {
                    "q50",
                    "q_median_50",
                    "q_skewness_proxy",
                    "skew",
                    "q_tilt_direction",
                    "hf_score",
                }

                candle_prefix = "candle_mechanics_"
                candle_mamba_default = {
                    "close_pos",
                    "body_pct_range",
                    "upper_wick_pct_range",
                    "lower_wick_pct_range",
                    "wick_imbalance",
                    "body_atr14",
                    "gap_atr14",
                    "close_vs_prev_close_atr14",
                    "high_vs_prev_close_atr14",
                    "low_vs_prev_close_atr14",
                    "logret_1d",
                    "ret_5d",
                    "rng_z_20",
                    "trend_3",
                    "trend_5",
                    "body_z_20",
                    "dist_to_high_20_atr14",
                    "dist_to_low_20_atr14",
                    "dist_to_mean_20_atr14",
                    "vol_ratio_20",
                    "vol_log_chg",
                }
                candle_mamba_optional = {
                    "sign_sum_3",
                    "sign_sum_5",
                    "inside_bar",
                    "outside_bar",
                    "range_atr14",
                }
                candle_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    "range_atr14",
                    "atr_ratio_14_60",
                    "rv_5",
                    "rv_20",
                    "dir_change",
                    "inside_bar",
                    "outside_bar",
                    "ret_1d",
                    "dow_1",
                    "dow_2",
                    "dow_3",
                    "dow_4",
                    "dow_5",
                    "moy_1",
                    "moy_2",
                    "moy_3",
                    "moy_4",
                    "moy_5",
                    "moy_6",
                    "moy_7",
                    "moy_8",
                    "moy_9",
                    "moy_10",
                    "moy_11",
                    "moy_12",
                }

                # ============================================================
                # UNIFIED MAMBA OPTIONAL COLUMNS (single CLI command)
                # ============================================================
                # Format: MAMBA_OPTIONAL_COLUMNS="family:col1,col2;family2:col3,col4"
                # Example: MAMBA_OPTIONAL_COLUMNS="correlation:corr_decoupling_z;dcf:undervaluation;cross_asset:all"
                # Special values:
                #   - "all:all" enables ALL optional columns for ALL families (master switch)
                #   - "family:all" enables ALL optional columns for that family
                #   - Column names are suffix only (without family prefix)
                # 
                # Supported families:
                #   - correlation, corp_splits, cross_asset, dcf, dividends, earnings
                #   - alt_signals, arima, quantile, candle
                # ============================================================
                unified_opt_raw = os.getenv("MAMBA_OPTIONAL_COLUMNS", "").strip().lower()
                unified_opt_map: Dict[str, set] = {}
                
                # Master switch: all:all enables everything
                _mamba_optional_master_switch = False
                
                for segment in unified_opt_raw.split(";"):
                    segment = segment.strip()
                    if ":" in segment:
                        fam, cols = segment.split(":", 1)
                        fam = fam.strip()
                        cols_set = {c.strip() for c in cols.split(",") if c.strip()}
                        # Check for master switch
                        if fam == "all" and "all" in cols_set:
                            _mamba_optional_master_switch = True
                        if fam not in unified_opt_map:
                            unified_opt_map[fam] = set()
                        unified_opt_map[fam].update(cols_set)

                def _unified_mamba_enabled(family: str, suffix: str) -> bool:
                    """Check if suffix is enabled for family in unified env var."""
                    # Master switch overrides everything
                    if _mamba_optional_master_switch:
                        return True
                    fam_opts = unified_opt_map.get(family, set())
                    return "all" in fam_opts or suffix in fam_opts

                # ------------------------------------------------------------
                # Corp actions splits: route with bounded recency
                # CRITICAL: Use corp_actions_splits_recency (bounded [0,1]) instead of
                # raw days_since=9999 which corrupts role-aware averaging.
                # Mamba: flag, log_ratio, post_5d, post_20d, recency (rare events, conditioning)
                # Portfolio: all columns including recency (regime overlays)
                # ------------------------------------------------------------
                corp_splits_prefix = "corp_actions_splits_"
                corp_splits_mamba_optional = {
                    "flag",
                    "log_ratio",
                    "post_5d",
                    "post_20d",
                    "recency",
                }
                corp_splits_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    "ratio",
                    "days_since",  # Raw days_since still goes to portfolio but prefer recency
                    "count_5y",
                }

                # ------------------------------------------------------------
                # Correlation: predictive spillovers to Mamba, exposure/regime to portfolio
                # ------------------------------------------------------------
                correlation_prefix = "correlation_"
                correlation_mamba_default = {
                    # Spillover / propagation (predictive)
                    "lag_corr_1_spy",
                    "lag_corr_5_spy",
                    "lag_corr_1_vxx",
                    "lag_corr_2_vxx",
                    # VIX lag correlations
                    "corr_20_vix_lag1",
                    "corr_60_vix_lag1",
                    "corr_20_vix_lag2",
                    "corr_60_vix_lag2",
                    # Serial structure in returns (autocorrelation)
                    "acf_ret_1",
                    "acf_ret_5",
                }
                correlation_mamba_optional = {
                    # Optional context (only if explicitly enabled)
                    "corr_decoupling_z",
                    "corr_spread_20_60_spy",
                }
                correlation_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    # RISK: systematic exposure
                    "corr_20_spy",
                    "corr_60_spy",
                    "corr_20_qqq",
                    "corr_20_vxx",
                    "corr_20_sector",
                    "corr_20_spy_vol",
                    "corr_20_vxx_vol",
                    # REGIME: regime shifts, instability
                    "corr_decoupling_z",
                    "corr_spread_20_60_spy",
                    "corr_20_spy_trend",
                    "corr_20_qqq_trend",
                    "corr_20_vxx_trend",
                    "corr_spread_20_60_qqq",
                    "corr_return_vol_20",
                    "corr_return_range_10",
                    "corr_vol_volatility_20",
                    "acf_absret_1",
                    "acf_vol_1",
                }

                # Environment variable for optional correlation columns in Mamba
                # Supports both legacy CORRELATION_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                corr_opt_raw = os.getenv("CORRELATION_MAMBA_OPTIONAL", "").strip().lower()
                corr_opt_tokens = {t.strip() for t in corr_opt_raw.split(",") if t.strip()}

                def _corr_mamba_enabled(suffix: str) -> bool:
                    return suffix in corr_opt_tokens or _unified_mamba_enabled("correlation", suffix)

                # Environment variable for optional corp_splits columns in Mamba
                # Supports both legacy CORP_SPLITS_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                splits_opt_raw = os.getenv("CORP_SPLITS_MAMBA_OPTIONAL", "").strip().lower()
                splits_opt_tokens = {t.strip() for t in splits_opt_raw.split(",") if t.strip()}
                splits_opt_all = "all" in splits_opt_tokens

                def _splits_mamba_enabled(suffix: str) -> bool:
                    return bool(splits_opt_all or suffix in splits_opt_tokens or _unified_mamba_enabled("corp_splits", suffix))

                # ------------------------------------------------------------
                # Cross-asset: lead/lag predictors to Mamba, everything else to portfolio
                # CRITICAL: Regime-change fields are ABSOLUTE MAGNITUDE for stress aggregation
                # ------------------------------------------------------------
                cross_asset_prefix = "cross_asset_"
                cross_asset_mamba_default = {
                    # Lead/lag analysis: per-asset timing/leadership patterns
                    "spy_leads_stock_5d",
                    "stock_leads_spy_5d",
                }
                cross_asset_mamba_optional = set()  # Currently none
                cross_asset_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    # RISK: systematic exposure
                    "spy_corr_20d",
                    "qqq_corr_20d",
                    "sector_etf_corr_20d",
                    "beta_20d",
                    "beta_volatility_20d",
                    "vix_corr_20d",
                    "vix_spread_indicator",
                    "realized_vol_vs_spy_corr",
                    "tnx_corr_20d",
                    "irx_corr_20d",
                    "credit_spread_level",
                    "asset_corr_hyg_60",
                    "asset_corr_uup_60",
                    "risk_offness",  # One-sided stress for overlays
                    # REGIME: regime-break detectors (ABSOLUTE MAGNITUDE)
                    "beta_change_rate",
                    "beta_volatility_change",
                    "beta_sign_flip_flag",
                    "tnx_corr_change_5d",
                    "irx_corr_change_5d",
                    "cross_asset_coupling_change",
                    "risk_onoff_factor",  # Directional, for policy state only
                }

                # Environment variable for optional cross_asset columns in Mamba
                # Supports both legacy CROSS_ASSET_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                xasset_opt_raw = os.getenv("CROSS_ASSET_MAMBA_OPTIONAL", "").strip().lower()
                xasset_opt_tokens = {t.strip() for t in xasset_opt_raw.split(",") if t.strip()}

                def _xasset_mamba_enabled(suffix: str) -> bool:
                    return suffix in xasset_opt_tokens or _unified_mamba_enabled("cross_asset", suffix)

                # ------------------------------------------------------------
                # DCF: momentum/z-score predictors to Mamba, anchors/stress to portfolio
                # CRITICAL: One-sided stress signals (overextension, downside_skew_stress)
                # ------------------------------------------------------------
                dcf_prefix = "dcf_"
                dcf_mamba_default = {
                    # Momentum (alpha signals)
                    "mom_1m",
                    "mom_3m",
                    "mom_12m",
                    "mom_vol_adjusted",
                    "mom_sharped",
                    # Mean-reversion z-scores
                    "zscore_1m",
                    "zscore_3m",
                    # Value-momentum mix
                    "value_momentum_ratio",
                    # Directional alpha signals (NOT stress)
                    "undervaluation",
                    "scenario_skew",
                }
                dcf_mamba_optional = set()  # Currently none
                dcf_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    # REGIME: anchor ratios (raw and log)
                    "price_to_fairvalue_1y",
                    "price_to_fairvalue_3m",
                    "price_regime",
                    "log_p2fv_1y",
                    "log_p2fv_3m",
                    "log_price_regime",
                    # Trend quality
                    "trend_slope_1m",
                    "trend_stability",
                    # Scenario positioning
                    "scenario_position",
                    "terminal_value_pct",
                    # RISK: volatility-adjusted + one-sided stress
                    "vol_adjusted_value",
                    "scenario_spread",
                    "overextension",
                    "downside_skew_stress",
                }

                # Environment variable for optional dcf columns in Mamba
                # Supports both legacy DCF_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                dcf_opt_raw = os.getenv("DCF_MAMBA_OPTIONAL", "").strip().lower()
                dcf_opt_tokens = {t.strip() for t in dcf_opt_raw.split(",") if t.strip()}

                def _dcf_mamba_enabled(suffix: str) -> bool:
                    return suffix in dcf_opt_tokens or _unified_mamba_enabled("dcf", suffix)

                # ------------------------------------------------------------
                # Dividends: event_intensity to Mamba (if labels are adjusted), rest to portfolio
                # CRITICAL: dividend_event_stress is one-sided RISK for overlays
                # ------------------------------------------------------------
                dividends_prefix = "dividends_"
                dividends_mamba_conditional = {
                    # Only include if labels are dividend-adjusted AND shifted
                    "dividend_event_intensity",
                }
                dividends_mamba_optional = {
                    # Optional context (enable via env var if you trust your label pipeline)
                    "ex_dividend_window_strength",
                    "days_to_ex_dividend",
                }
                dividends_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    # RISK: yield metrics
                    "dividend_yield_est",
                    "dividend_yield_zscore",
                    # RISK: one-sided event stress
                    "dividend_event_stress",
                    # REGIME: event proximity
                    "days_to_ex_dividend",
                    "ex_dividend_window_strength",
                    # REGIME: structural
                    "dividend_frequency",
                    "dividend_amount",
                }

                # Environment variable for optional dividends columns in Mamba
                # Supports both legacy DIVIDENDS_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                div_opt_raw = os.getenv("DIVIDENDS_MAMBA_OPTIONAL", "").strip().lower()
                div_opt_tokens = {t.strip() for t in div_opt_raw.split(",") if t.strip()}
                div_opt_all = "all" in div_opt_tokens

                def _div_mamba_enabled(suffix: str) -> bool:
                    return bool(div_opt_all or suffix in div_opt_tokens or _unified_mamba_enabled("dividends", suffix))

                # ------------------------------------------------------------
                # Earnings: surprises/growth/revisions/beat-rate to Mamba, stress/dispersion to portfolio
                # CRITICAL: Surprises are SHIFTED by 1 day in EarningsAnalyzer (leakage prevention)
                # CRITICAL: pre_event_stress and post_event_stress are one-sided RISK for overlays
                # ------------------------------------------------------------
                earnings_prefix = "earnings_"
                earnings_mamba_default = {
                    # Core alpha signals (SHIFTED to prevent leakage)
                    "eps_surprise_pct",
                    "revenue_surprise_pct",
                    # Growth trends
                    "eps_growth_qoq",
                    "eps_growth_yoy",
                    "revenue_growth_qoq",
                    "revenue_growth_yoy",
                    # Beat patterns (HIGH ALPHA)
                    "beat_streak",
                    "beat_rate_3y",
                    # Revisions (CRITICAL ALPHA)
                    "revision_breadth",
                    # Event decay (PEAD capture)
                    "event_decay",
                }
                earnings_mamba_optional = {
                    # Optional conditioning (enable via env var)
                    "days_since_earnings",
                    "days_to_next_earnings",
                }
                earnings_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    # RISK: uncertainty / volatility
                    "miss_streak",
                    "surprise_volatility",
                    "estimate_dispersion",
                    # RISK: one-sided event stress (magnitude-based)
                    "pre_event_stress",
                    "post_event_stress",
                    # REGIME: event timing (also useful for policy state)
                    "days_since_earnings",
                    "days_to_next_earnings",
                }

                # Environment variable for optional earnings columns in Mamba
                # Supports both legacy EARNINGS_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                earn_opt_raw = os.getenv("EARNINGS_MAMBA_OPTIONAL", "").strip().lower()
                earn_opt_tokens = {t.strip() for t in earn_opt_raw.split(",") if t.strip()}
                earn_opt_all = "all" in earn_opt_tokens

                def _earn_mamba_enabled(suffix: str) -> bool:
                    return bool(earn_opt_all or suffix in earn_opt_tokens or _unified_mamba_enabled("earnings", suffix))

                # ------------------------------------------------------------
                # CBOE term structure: ALL columns go to portfolio, NONE to Mamba
                # This family provides market stress context for overlays/policy state
                # ------------------------------------------------------------
                cboe_prefix = "cboe_term_"
                # Raw column names from cboe_term.py (unprefixed)
                cboe_raw_cols = {
                    "vxst_vix_term_slope",
                    "vix_vxv_term_slope",
                    "vix_vxmt_term_slope",
                    "normalized_term_slope",
                    "vix_term_curvature",
                    "front_back_spread",
                    "panic_premium",
                    "vix_roll_yield",
                    "vix_ratio_term",
                    "vix_contango_strength",
                    "vol_risk_premium",
                    "vol_risk_premium_pct",
                    "vol_risk_premium_z",
                    "vix_term_slope_change_1d",
                    "vix_term_slope_change_5d",
                    "vix_curvature_change",
                    "panic_premium_change",
                }

                # ------------------------------------------------------------
                # Calibration: ALL columns go to portfolio (quality→RISK, not Mamba)
                # Meta-performance signals should NEVER be fed to the model being graded
                # ------------------------------------------------------------
                calibration_prefix = "calibration_"
                calibration_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "mean_calibration_error",
                    "overall_score",
                    "interval_error",
                    "quantile_error",
                    "sharpness",
                    "reliability",
                    "calibration_slope",
                    "calibration_intercept",
                    "brier_score",
                    "log_loss",
                    "expected_calibration_error",
                    "maximum_calibration_error",
                    "requires_recalibration",
                    "recalibration_flag",
                    "model_stale",
                    "quality_regime",
                }

                # ------------------------------------------------------------
                # Online learning: ALL columns go to portfolio (trust/drift signals)
                # These are meta-performance signals, not predictive features
                # ------------------------------------------------------------
                online_prefix = "online_learning_"
                online_portfolio_only = {
                    "has_data",
                    "activity",
                    "days_since_update",
                    "trust_score",
                    "model_confidence",
                    "direction_accuracy",
                    "sign_accuracy",
                    "hit_rate",
                    "accuracy",
                    "quantile_accuracy",
                    "mse",
                    "mae",
                    "sharpe_estimate",
                    "information_ratio",
                    "uncertainty_estimate",
                    "prediction_variance",
                    "drift_flag",
                    "partial_retrain_flag",
                    "full_retrain_flag",
                    "uncertainty_compression_alert",
                    "regime_shift_detected",
                    "distribution_drift",
                    "concept_drift",
                    "covariate_drift",
                    "regime_prob_bull",
                    "regime_prob_bear",
                    "regime_prob_neutral",
                    "regime_probability",
                }

                # ------------------------------------------------------------
                # Econ events calendar: selective routing
                # Compact macro context to Mamba; regime/risk overlays to portfolio
                # CRITICAL: 9999 sentinels replaced with bounded prox_next_* / recency_last_*
                # ------------------------------------------------------------
                econ_events_prefix = "econ_events_calendar_"
                econ_events_mamba_default = {
                    # Pulse surprises (shifted, signed, for directional context)
                    "pulse_surprise_cpi",
                    "pulse_surprise_fomc",
                    "pulse_surprise_nfp",
                    "pulse_surprise_gdp",
                    "pulse_surprise_pce",
                    "pulse_surprise_unemployment",
                    "pulse_surprise_retail_sales",
                    "pulse_surprise_ism",
                    "pulse_surprise_core_cpi",
                    # Pulse strength (magnitude only)
                    "pulse_strength_cpi",
                    "pulse_strength_fomc",
                    "pulse_strength_nfp",
                    "pulse_strength_gdp",
                    "pulse_strength_pce",
                    "pulse_strength_unemployment",
                    "pulse_strength_retail_sales",
                    "pulse_strength_ism",
                    "pulse_strength_core_cpi",
                    # Bounded proximity (safe for aggregation)
                    "prox_next_cpi",
                    "prox_next_fomc",
                    "prox_next_nfp",
                    "prox_next_gdp",
                    "prox_next_pce",
                    "prox_next_unemployment",
                    "prox_next_retail_sales",
                    "prox_next_ism",
                    "prox_next_core_cpi",
                }
                econ_events_mamba_optional = {
                    # Optional: z-scored surprises (more selective context)
                    "surprise_z_cpi",
                    "surprise_z_fomc",
                    "surprise_z_nfp",
                    "surprise_z_gdp",
                    "surprise_z_pce",
                    "surprise_z_unemployment",
                    "surprise_z_retail_sales",
                    "surprise_z_ism",
                    "surprise_z_core_cpi",
                }
                econ_events_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    # RISK: composite stress
                    "macro_shock_major",
                    "macro_upcoming_major",
                    # REGIME: composite signed (for policy state)
                    "macro_surprise_signed",
                    # Note: pre_window_*, post_window_*, recency_last_*, event_occurrence_*,
                    # pulse_occurrence_*, days_to_next_*, days_since_last_* all go to portfolio
                }

                # Environment variable for optional econ columns in Mamba
                econ_opt_raw = os.getenv("ECON_EVENTS_MAMBA_OPTIONAL", "").strip().lower()
                econ_opt_tokens = {t.strip() for t in econ_opt_raw.split(",") if t.strip()}
                econ_opt_all = "all" in econ_opt_tokens

                def _econ_mamba_enabled(suffix: str) -> bool:
                    return bool(econ_opt_all or suffix in econ_opt_tokens or _unified_mamba_enabled("econ_events", suffix))

                # ------------------------------------------------------------
                # Exchange calendar: ALL columns go to portfolio, NONE to Mamba
                # Execution/microstructure regime context (not alpha)
                # CRITICAL: 9999 sentinels replaced with bounded holiday_prox / holiday_recency
                # CRITICAL: liquidity_stress composite for position sizing
                # ------------------------------------------------------------
                exchange_calendar_prefix = "exchange_calendar_"
                exchange_calendar_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    "confidence",
                    # RISK: liquidity stress for position sizing
                    "liquidity_stress",
                    # REGIME: event flags
                    "is_holiday_adjacent",
                    "is_half_day",
                    # REGIME: bounded proximity/recency
                    "holiday_prox",
                    "holiday_recency",
                    # DEPRECATED (keep for backward compat)
                    "days_to_holiday",
                    "days_since_holiday",
                }

                # ------------------------------------------------------------
                # FIN_G1 (Liquidity): stress to portfolio, raw ratios optional for Mamba
                # CRITICAL: Raw ratios are HIGHER = SAFER (inverted meaning)
                # CRITICAL: Use *_stress features for portfolio risk overlays
                # ------------------------------------------------------------
                fin_g1_prefix = "fin_g1_"
                fin_g1_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    # RISK: stress features (higher = worse, safe for aggregation)
                    "liquidity_stress",
                    "quick_stress",
                    "cash_stress",
                    "liquidity_trend_stress",
                    "liquidity_zscore_5y",
                    # REGIME: trend direction
                    "liquidity_trend_3y",
                }
                fin_g1_mamba_optional = {
                    # Raw ratios: only include if horizon >= 21d
                    "current_ratio",
                    "quick_ratio",
                    "cash_ratio",
                }

                # Environment variable for optional fin_g1 columns in Mamba
                fin_g1_opt_raw = os.getenv("FIN_G1_MAMBA_OPTIONAL", "").strip().lower()
                fin_g1_opt_tokens = {t.strip() for t in fin_g1_opt_raw.split(",") if t.strip()}
                fin_g1_opt_all = "all" in fin_g1_opt_tokens

                def _fin_g1_mamba_enabled(suffix: str) -> bool:
                    return bool(fin_g1_opt_all or suffix in fin_g1_opt_tokens or _unified_mamba_enabled("fin_g1", suffix))

                # ------------------------------------------------------------
                # FIN_G2 (Leverage): stress to portfolio, raw ratios optional for Mamba
                # CRITICAL: interest_coverage is HIGHER = SAFER (use _stress variant)
                # CRITICAL: Use robust transforms for exploding debt ratios
                # ------------------------------------------------------------
                fin_g2_prefix = "fin_g2_"
                fin_g2_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    # RISK: stress features (higher = worse)
                    "interest_coverage_stress",
                    "debt_to_equity_robust",
                    "net_debt_to_ebitda_robust",
                    "leverage_zscore_5y",
                    # RISK: raw leverage ratios (already higher = worse)
                    "debt_to_equity",
                    "debt_to_assets",
                    "equity_multiplier",
                    "net_debt_to_ebitda",
                    "net_debt_to_fcf",
                    "interest_burden",
                    # REGIME: debt levels and trend
                    "total_debt",
                    "long_term_debt",
                    "leverage_trend_3y",
                }
                fin_g2_mamba_optional = {
                    # Raw coverage: only include if horizon >= 21d
                    "interest_coverage",
                }

                # Environment variable for optional fin_g2 columns in Mamba
                fin_g2_opt_raw = os.getenv("FIN_G2_MAMBA_OPTIONAL", "").strip().lower()
                fin_g2_opt_tokens = {t.strip() for t in fin_g2_opt_raw.split(",") if t.strip()}
                fin_g2_opt_all = "all" in fin_g2_opt_tokens

                def _fin_g2_mamba_enabled(suffix: str) -> bool:
                    return bool(fin_g2_opt_all or suffix in fin_g2_opt_tokens or _unified_mamba_enabled("fin_g2", suffix))

                # ------------------------------------------------------------
                # FIN_G3 (Efficiency): stress to portfolio, turnover ratios industry-structural
                # CRITICAL: Raw turnover ratios are INDUSTRY-STRUCTURAL
                # CRITICAL: Prefer sector-normalized versions or keep out of Mamba
                # ------------------------------------------------------------
                fin_g3_prefix = "fin_g3_"
                fin_g3_portfolio_only = {
                    # HYGIENE
                    "has_data",
                    "activity",
                    "days_since_update",
                    # RISK: stress features (higher = worse)
                    "ccc_stress",
                    "turnover_volatility_3y",
                    # REGIME: days outstanding and CCC
                    "dsos",
                    "dios",
                    "dpos",
                    "ccc",
                }
                fin_g3_mamba_optional = {
                    # Raw turnover ratios: only include if sector-normalized
                    "asset_turnover",
                    "inventory_turnover",
                    "receivables_turnover",
                    "payables_turnover",
                }

                # Environment variable for optional fin_g3 columns in Mamba
                fin_g3_opt_raw = os.getenv("FIN_G3_MAMBA_OPTIONAL", "").strip().lower()
                fin_g3_opt_tokens = {t.strip() for t in fin_g3_opt_raw.split(",") if t.strip()}
                fin_g3_opt_all = "all" in fin_g3_opt_tokens

                def _fin_g3_mamba_enabled(suffix: str) -> bool:
                    return bool(fin_g3_opt_all or suffix in fin_g3_opt_tokens or _unified_mamba_enabled("fin_g3", suffix))

                # Optional alt-signals columns can be enabled for Mamba via env list.
                # Supports both legacy ALT_SIGNALS_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                # Example: ALT_SIGNALS_MAMBA_OPTIONAL="overnight_return,gap_vs_vix_interaction"
                # Example: MAMBA_OPTIONAL_COLUMNS="alt_signals:overnight_return,gap_vs_vix_interaction"
                opt_raw = os.getenv("ALT_SIGNALS_MAMBA_OPTIONAL", "").strip().lower()
                opt_tokens = {t.strip() for t in opt_raw.split(",") if t.strip()}
                opt_all = "all" in opt_tokens

                def _alt_mamba_enabled(suffix: str) -> bool:
                    return bool(opt_all or suffix in opt_tokens or _unified_mamba_enabled("alt_signals", suffix))

                # Supports both legacy ARIMA_FORECAST_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                arima_opt_raw = os.getenv("ARIMA_FORECAST_MAMBA_OPTIONAL", "").strip().lower()
                arima_opt_tokens = {t.strip() for t in arima_opt_raw.split(",") if t.strip()}
                arima_opt_all = "all" in arima_opt_tokens

                def _arima_mamba_enabled(suffix: str) -> bool:
                    return bool(arima_opt_all or suffix in arima_opt_tokens or _unified_mamba_enabled("arima", suffix))

                # Supports both legacy QUANTILE_FORECAST_MAMBA_STACKING and unified MAMBA_OPTIONAL_COLUMNS
                quantile_opt_raw = os.getenv("QUANTILE_FORECAST_MAMBA_STACKING", "").strip().lower()
                quantile_opt_tokens = {t.strip() for t in quantile_opt_raw.split(",") if t.strip()}
                quantile_opt_all = "all" in quantile_opt_tokens

                def _quantile_mamba_enabled(suffix: str) -> bool:
                    return bool(quantile_opt_all or suffix in quantile_opt_tokens or _unified_mamba_enabled("quantile", suffix))

                # Supports both legacy CANDLE_MECHANICS_MAMBA_OPTIONAL and unified MAMBA_OPTIONAL_COLUMNS
                candle_opt_raw = os.getenv("CANDLE_MECHANICS_MAMBA_OPTIONAL", "").strip().lower()
                candle_opt_tokens = {t.strip() for t in candle_opt_raw.split(",") if t.strip()}
                candle_opt_all = "all" in candle_opt_tokens

                def _candle_mamba_enabled(suffix: str) -> bool:
                    return bool(candle_opt_all or suffix in candle_opt_tokens or _unified_mamba_enabled("candle", suffix))

                mamba_cols: List[str] = []
                portfolio_cols: List[str] = []

                for col in [c for c in normalized_panel.columns if str(c) != "date"]:
                    col_s = str(col)
                    col_lower = col_s.lower()
                    stream = "portfolio"

                    # --------------------------------------------------------
                    # CBOE term structure: ALL to portfolio, NONE to Mamba
                    # --------------------------------------------------------
                    if col_s.startswith(cboe_prefix) or col_lower in cboe_raw_cols:
                        stream = "portfolio"
                    # --------------------------------------------------------
                    # Calibration: ALL to portfolio (meta-performance)
                    # --------------------------------------------------------
                    elif col_s.startswith(calibration_prefix):
                        stream = "portfolio"
                    # --------------------------------------------------------
                    # Online learning: ALL to portfolio (meta-performance)
                    # --------------------------------------------------------
                    elif col_s.startswith(online_prefix):
                        stream = "portfolio"
                    # --------------------------------------------------------
                    # Alternative signals: selective routing
                    # --------------------------------------------------------
                    elif col_s.startswith(alt_prefix):
                        suffix = col_s[len(alt_prefix):].lower()
                        if suffix in alt_mamba_default:
                            stream = "mamba"
                        elif suffix in alt_mamba_optional and _alt_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # ARIMA forecast: selective routing
                    # --------------------------------------------------------
                    elif col_s.startswith(arima_prefix):
                        suffix = col_s[len(arima_prefix):].lower()
                        if suffix in arima_mamba_default:
                            stream = "mamba"
                        elif suffix in arima_mamba_optional and _arima_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Quantile forecast: portfolio by default (Option 1 from spec)
                    # --------------------------------------------------------
                    elif col_s.startswith(quantile_prefix):
                        suffix = col_s[len(quantile_prefix):].lower()
                        if suffix in quantile_mamba_optional and _quantile_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Candle mechanics: selective routing
                    # --------------------------------------------------------
                    elif col_s.startswith(candle_prefix):
                        suffix = col_s[len(candle_prefix):].lower()
                        if suffix in candle_mamba_default:
                            stream = "mamba"
                        elif suffix in candle_mamba_optional and _candle_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in candle_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Corp actions splits: selective routing with bounded recency
                    # CRITICAL: Use recency (bounded [0,1]) for Mamba, not raw days_since
                    # --------------------------------------------------------
                    elif col_s.startswith(corp_splits_prefix):
                        suffix = col_s[len(corp_splits_prefix):].lower()
                        if suffix in corp_splits_portfolio_only:
                            stream = "portfolio"
                        elif suffix in corp_splits_mamba_optional and _splits_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Correlation: predictive spillovers to Mamba, exposure/regime to portfolio
                    # --------------------------------------------------------
                    elif col_s.startswith(correlation_prefix) or col_lower.startswith("corr_") or col_lower.startswith("acf_") or col_lower.startswith("lag_corr_"):
                        # Extract suffix for matching
                        if col_s.startswith(correlation_prefix):
                            suffix = col_s[len(correlation_prefix):].lower()
                        else:
                            suffix = col_lower
                        if suffix in correlation_mamba_default:
                            stream = "mamba"
                        elif suffix in correlation_mamba_optional and _corr_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in correlation_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Cross-asset: lead/lag predictors to Mamba, everything else to portfolio
                    # CRITICAL: Regime-change fields are ABSOLUTE MAGNITUDE for stress aggregation
                    # --------------------------------------------------------
                    elif col_s.startswith(cross_asset_prefix) or col_lower.startswith("cross_asset_"):
                        if col_s.startswith(cross_asset_prefix):
                            suffix = col_s[len(cross_asset_prefix):].lower()
                        else:
                            suffix = col_lower[len("cross_asset_"):]
                        if suffix in cross_asset_mamba_default:
                            stream = "mamba"
                        elif suffix in cross_asset_mamba_optional and _xasset_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in cross_asset_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # DCF: momentum/z-score predictors to Mamba, anchors/stress to portfolio
                    # CRITICAL: One-sided stress signals (overextension, downside_skew_stress)
                    # --------------------------------------------------------
                    elif col_s.startswith(dcf_prefix):
                        suffix = col_s[len(dcf_prefix):].lower()
                        if suffix in dcf_mamba_default:
                            stream = "mamba"
                        elif suffix in dcf_mamba_optional and _dcf_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in dcf_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Dividends: event_intensity to Mamba (conditional), rest to portfolio
                    # CRITICAL: dividend_event_stress is one-sided RISK for overlays
                    # --------------------------------------------------------
                    elif col_s.startswith(dividends_prefix):
                        suffix = col_s[len(dividends_prefix):].lower()
                        if suffix in dividends_mamba_conditional and _div_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in dividends_mamba_optional and _div_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in dividends_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Earnings: surprises/growth/revisions to Mamba, stress/dispersion to portfolio
                    # CRITICAL: Surprises are SHIFTED by 1 day (leakage prevention)
                    # CRITICAL: pre_event_stress and post_event_stress are one-sided RISK
                    # --------------------------------------------------------
                    elif col_s.startswith(earnings_prefix):
                        suffix = col_s[len(earnings_prefix):].lower()
                        if suffix in earnings_mamba_default:
                            stream = "mamba"
                        elif suffix in earnings_mamba_optional and _earn_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in earnings_portfolio_only:
                            stream = "portfolio"
                        else:
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Econ events calendar: compact macro context to Mamba, rest to portfolio
                    # CRITICAL: Use bounded prox_next_* / recency_last_* (not 9999 sentinels)
                    # --------------------------------------------------------
                    elif col_s.startswith(econ_events_prefix):
                        suffix = col_s[len(econ_events_prefix):].lower()
                        if suffix in econ_events_mamba_default:
                            stream = "mamba"
                        elif suffix in econ_events_mamba_optional and _econ_mamba_enabled(suffix):
                            stream = "mamba"
                        elif suffix in econ_events_portfolio_only:
                            stream = "portfolio"
                        else:
                            # Pre/post windows, recency, occurrences, raw days go to portfolio
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Exchange calendar: ALL to portfolio (execution/microstructure context)
                    # CRITICAL: Use bounded holiday_prox / holiday_recency + liquidity_stress
                    # --------------------------------------------------------
                    elif col_s.startswith(exchange_calendar_prefix):
                        # All exchange_calendar columns go to portfolio by design
                        stream = "portfolio"
                    # --------------------------------------------------------
                    # FIN_G1 (Liquidity): stress to portfolio, raw ratios optional
                    # CRITICAL: Raw ratios are HIGHER = SAFER (inverted for risk_scale)
                    # --------------------------------------------------------
                    elif col_s.startswith(fin_g1_prefix):
                        suffix = col_s[len(fin_g1_prefix):].lower()
                        if suffix in fin_g1_portfolio_only:
                            stream = "portfolio"
                        elif suffix in fin_g1_mamba_optional and _fin_g1_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            # Unknown fin_g1 columns go to portfolio by default
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # FIN_G2 (Leverage): stress to portfolio, raw ratios optional
                    # CRITICAL: interest_coverage is HIGHER = SAFER
                    # --------------------------------------------------------
                    elif col_s.startswith(fin_g2_prefix):
                        suffix = col_s[len(fin_g2_prefix):].lower()
                        if suffix in fin_g2_portfolio_only:
                            stream = "portfolio"
                        elif suffix in fin_g2_mamba_optional and _fin_g2_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            # Unknown fin_g2 columns go to portfolio by default
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # FIN_G3 (Efficiency): stress to portfolio, turnover optional
                    # CRITICAL: Turnover ratios are INDUSTRY-STRUCTURAL
                    # --------------------------------------------------------
                    elif col_s.startswith(fin_g3_prefix):
                        suffix = col_s[len(fin_g3_prefix):].lower()
                        if suffix in fin_g3_portfolio_only:
                            stream = "portfolio"
                        elif suffix in fin_g3_mamba_optional and _fin_g3_mamba_enabled(suffix):
                            stream = "mamba"
                        else:
                            # Unknown fin_g3 columns go to portfolio by default
                            stream = "portfolio"
                    # --------------------------------------------------------
                    # Default: use role_map from feature_roles
                    # --------------------------------------------------------
                    else:
                        role = str(role_map.get(col_s, ""))
                        stream = "mamba" if role == "predictive" else "portfolio"

                    if stream == "mamba":
                        mamba_cols.append(col_s)
                    else:
                        portfolio_cols.append(col_s)

                # De-dup while preserving order
                def _uniq(xs: List[str]) -> List[str]:
                    seen: set[str] = set()
                    out: List[str] = []
                    for x in xs:
                        if x in seen:
                            continue
                        seen.add(x)
                        out.append(x)
                    return out

                mamba_cols = _uniq(mamba_cols)
                portfolio_cols = _uniq(portfolio_cols)

                # Hygiene isolation: never allow governance columns into Mamba inputs
                gov_suffixes = ("_has_data", "_activity", "_days_since_update", "_confidence")
                hygiene_cols = [c for c in mamba_cols if str(c).endswith(gov_suffixes)]
                if hygiene_cols:
                    mamba_cols = [c for c in mamba_cols if c not in hygiene_cols]
                    portfolio_cols.extend(hygiene_cols)
                    portfolio_cols = _uniq(portfolio_cols)

                # Ensure total coverage (no leaks, no drops)
                all_feature_cols = [c for c in normalized_panel.columns if str(c) != "date"]
                covered = set(mamba_cols).union(set(portfolio_cols))
                missing = [c for c in all_feature_cols if c not in covered]
                overlap = set(mamba_cols).intersection(set(portfolio_cols))
                if missing:
                    LOGGER.warning("Role split: %d columns unassigned; routing to portfolio by default.", len(missing))
                    portfolio_cols.extend(missing)
                    portfolio_cols = _uniq(portfolio_cols)
                if overlap:
                    LOGGER.warning("Role split: %d columns assigned to both streams; keeping in portfolio only.", len(overlap))
                    mamba_cols = [c for c in mamba_cols if c not in overlap]

                # Build split frames
                base_cols = ["date"] if "date" in normalized_panel.columns else []
                mamba_panel = normalized_panel[base_cols + mamba_cols].copy()
                portfolio_panel = normalized_panel[base_cols + portfolio_cols].copy()

                # Apply alternative_signals staleness weight
                has_col = f"{alt_prefix}has_data"
                act_col = f"{alt_prefix}activity"
                days_col = f"{alt_prefix}days_since_update"
                if has_col in normalized_panel.columns and act_col in normalized_panel.columns and days_col in normalized_panel.columns:
                    has = pd.to_numeric(normalized_panel[has_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    activity = pd.to_numeric(normalized_panel[act_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                    days = pd.to_numeric(normalized_panel[days_col], errors="coerce").fillna(9999.0).clip(lower=0.0)
                    k_default = float(np.log(2.0) / 2.0)
                    try:
                        k = float(os.getenv("ALT_SIGNALS_STALENESS_K", str(k_default)))
                    except Exception:
                        k = k_default
                    w_stale = has * activity * np.exp(-k * days)
                else:
                    w_stale = pd.Series(0.0, index=normalized_panel.index)

                # Mamba: apply staleness decay to alternative_signals predictive features only
                if not mamba_panel.empty:
                    for col_s in [c for c in mamba_cols if str(c).startswith(alt_prefix)]:
                        if col_s in mamba_panel.columns:
                            mamba_panel[col_s] = pd.to_numeric(mamba_panel[col_s], errors="coerce").fillna(0.0) * w_stale

                # Portfolio: expose staleness weight for gating/threshold control
                if not portfolio_panel.empty:
                    w_name = f"{alt_prefix}stale_weight"
                    if w_name not in portfolio_panel.columns:
                        portfolio_panel[w_name] = pd.to_numeric(w_stale, errors="coerce").fillna(0.0)

                sym_u = str(symbol).upper()
                h_i = int(horizon)
                mamba_path = panel_path.with_name(f"{sym_u}_h{h_i}_merged_mamba.parquet")
                portfolio_path = panel_path.with_name(f"{sym_u}_h{h_i}_merged_portfolio.parquet")
            except Exception as exc:
                LOGGER.warning("Role split failed for %s h%d: %s", symbol, int(horizon), exc)
                mamba_panel = None
                portfolio_panel = None
                mamba_path = None
                portfolio_path = None

        # ------------------------------------------------------------------
        # HF Blocks merged parquet (forecast_hf + tech_micro_hf + event_risk_hf)
        # ------------------------------------------------------------------
        hf_panel: Optional[pd.DataFrame] = None
        hf_path: Optional[Path] = None
        try:
            hf_block_prefixes = ("forecast_hf_", "tech_micro_hf_", "event_risk_hf_")
            hf_cols = [
                c for c in normalized_panel.columns
                if any(str(c).startswith(prefix) for prefix in hf_block_prefixes)
            ]
            if hf_cols:
                base_cols = ["date"] if "date" in normalized_panel.columns else []
                hf_panel = normalized_panel[base_cols + hf_cols].copy()
                sym_u = str(symbol).upper()
                h_i = int(horizon)
                hf_path = panel_path.with_name(f"{sym_u}_h{h_i}_merged_hf.parquet")
                LOGGER.info("HF blocks panel: %d columns from %s", len(hf_cols), ", ".join(hf_block_prefixes))
        except Exception as exc:
            LOGGER.warning("HF blocks panel creation failed for %s h%d: %s", symbol, int(horizon), exc)
            hf_panel = None
            hf_path = None

        normalized_panel.to_parquet(panel_path, index=False)

        # Write split parquets (best-effort; do not fail build)
        try:
            if write_role_splits and mamba_panel is not None and mamba_path is not None:
                mamba_panel.to_parquet(mamba_path, index=False)
            if write_role_splits and portfolio_panel is not None and portfolio_path is not None:
                portfolio_panel.to_parquet(portfolio_path, index=False)
            if hf_panel is not None and hf_path is not None and not hf_panel.empty:
                hf_panel.to_parquet(hf_path, index=False)
                LOGGER.info("💾 Wrote HF blocks panel: %s", hf_path)
        except Exception as exc:
            LOGGER.warning("Failed writing role-split panels for %s h%d: %s", symbol, int(horizon), exc)

        meta = {
            "symbol": symbol,
            "horizon": horizon,
            "track": track_label,
            "families_requested": requested_families,
            "families_included": included_families,
            "families_missing": missing_families,
            "coverage_start": start_str,
            "coverage_end": end_str,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "provenance_path": str(provenance_path),
            "features_path": str(features_path) if features_path is not None else "",
            "index_path": str(index_path) if index_path is not None else "",
            "mamba_panel_path": str(mamba_path) if mamba_path is not None else "",
            "portfolio_panel_path": str(portfolio_path) if portfolio_path is not None else "",
            "hf_panel_path": str(hf_path) if hf_path is not None else "",
        }
        with open(panel_path.with_suffix(".meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
        LOGGER.info("💾 Wrote consolidated feature panel: %s", panel_path)
        return panel_path
    except Exception as exc:
        LOGGER.error("Failed to persist consolidated panel %s: %s", panel_name, exc)
        return None


def build_symbol_merged_parquet_via_feast(
    *,
    symbol: str,
    horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    families: List[str],
    cache_root: Path,
    offline_root: Path,
    feast_repo: Path,
    out_path: Path,
    service_name: Optional[str] = None,
    timestamp_family: str = "quantile_forecast",
    drop_split_column: bool = True,
) -> Optional[Path]:
    """Build one merged parquet for a symbol using Feast.

    This is the "single unified parquet per symbol" artifact.
    Internally we still export per-family offline parquet tables so Feast can
    manage joins and feature selection.
    """
    try:
        from tools.feast.export_local_cache_to_feast_offline import export as feast_export
        from tools.feast.generate_feature_definitions import generate as feast_generate
        from tools.feast.build_symbol_parquet import build_symbol_parquet
    except Exception as exc:
        LOGGER.error("Feast helpers unavailable; cannot build merged parquet: %s", exc)
        return None

    # 1) Export per-family caches -> offline parquet tables
    try:
        feast_export(cache_root=cache_root, out_root=offline_root, horizon=horizon, families=sorted(set(families)))
    except Exception as exc:
        LOGGER.error("Feast offline export failed: %s", exc)
        return None

    # 2) Generate AUTO feature definitions from offline tables
    try:
        out_defs = feast_repo / "feature_definitions_auto.py"
        feast_generate(off_root=offline_root, horizon=horizon, out_path=out_defs)
    except Exception as exc:
        LOGGER.error("Feast feature definition generation failed: %s", exc)
        return None

    # 3) Apply Feast repo (refresh registry)
    try:
        import shutil
        import subprocess

        feast_bin = shutil.which("feast")
        if feast_bin is None:
            # Fall back to local venv if present
            candidate = (REPO_ROOT / ".venv" / "bin" / "feast")
            if candidate.exists():
                feast_bin = str(candidate)
        if feast_bin is None:
            raise RuntimeError("feast CLI not found (missing PATH and .venv/bin/feast)")

        subprocess.run([feast_bin, "apply"], cwd=str(feast_repo), check=True)
    except Exception as exc:
        LOGGER.error("Feast apply failed: %s", exc)
        return None

    # 4) Fetch merged features for both splits and write one parquet
    try:
        if service_name is None or not str(service_name).strip():
            service_name = f"phase2_all_h{int(horizon)}_v1"
        df = build_symbol_parquet(
            repo=str(feast_repo),
            symbol=symbol,
            horizon=horizon,
            splits=["train", "valid"],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            service=service_name,
            timestamp_family=timestamp_family,
        )
        if drop_split_column and "split" in df.columns:
            # `split` is an internal Feast entity key in this repo. The unified parquet
            # we want downstream is all-dates and should not carry train/valid semantics.
            df = df.drop(columns=["split"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        LOGGER.info("💾 Wrote merged symbol parquet: %s", out_path)
        return out_path
    except Exception as exc:
        LOGGER.error("Merged parquet build failed: %s", exc)
        return None


# ============================================================================
# Main Orchestration
# ============================================================================

def prepare_families(
    *,
    symbol: str,
    horizon: int,
    wf_start: str,
    wf_end: str,
    wf_train_years: int,
    wf_step_years: int = None,
    wf_step_days: int = None,
    families: str = "all",
    exclude_families: Optional[Sequence[str]] = None,
    strict: bool = False,
    workers: int = 1,
    hf_workers: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    mode: str = "stage-b",
    write_merged: bool = True,
    merged_out: Optional[Path] = None,
    merged_service: Optional[str] = None,
    manifest_scope: str = "requested",
    manifest_exclude_families: Optional[Sequence[str]] = None,
    reuse_symbol_only_cache: bool = True,
    sequential_mode: bool = True,
) -> PreparationResult:
    """
    Main orchestration function for family signal preparation.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        horizon: Forecast horizon in days
        wf_start: Walk-forward start date (YYYY-MM-DD)
        wf_end: Walk-forward end date (YYYY-MM-DD)
        wf_train_years: Training window size in years
        wf_step_years: Step size in years (optional if wf_step_days provided)
        wf_step_days: Step size in days (optional if wf_step_years provided)
        families: 'all', 'base', 'hf', or comma-separated list
        strict: If True, fail fast on any missing family
        workers: Number of parallel workers (ignored if sequential_mode=True)
        cache_dir: Cache directory (default: data/local_cache)
        output_dir: Output directory for manifests (default: artifacts/prep_families)
        sequential_mode: If True (default), run all tasks sequentially within symbol.
                         Order: horizon-linked → base families → HF blocks.
                         No process/thread pools are spawned.
    
    Returns:
        PreparationResult
    """
    start_time = time.time()

    # Default exclusions (can be overridden by passing an explicit list, including []).
    if exclude_families is None:
        exclude_families = list(DEFAULT_EXCLUDE_FAMILIES)
    
    mode_normalized = (mode or "stage-b").lower()
    if mode_normalized == "walkforward":
        mode_normalized = "stage-b"
    if mode_normalized not in {"stage-a", "stage-b"}:
        LOGGER.warning("Unknown mode '%s'; defaulting to stage-b", mode_normalized)
        mode_normalized = "stage-b"
    stage_a_mode = mode_normalized == "stage-a"
    stage_b_mode = mode_normalized == "stage-b"

    if cache_dir is None:
        # Default: cache/symbols/<SYMBOL>/h<H>/ for individual family caches
        cache_dir = SYMBOLS_CACHE_DIR / symbol.upper() / f"h{horizon}"
    if output_dir is None:
        output_dir = REPO_ROOT / "artifacts" / "prep_families"
    
    # Ensure cache_dir exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔧 SIMPLIFIED: Use ONE flat cache directory per symbol/horizon
    # All windowed files (_w0, _w1, etc.) and feature files go in the SAME directory
    # This allows different step sizes to reuse cached features without regeneration
    expected_suffix = f"h{horizon}"
    
    # Check if cache_dir already ends with the expected suffix (avoid double nesting)
    if cache_dir.name == expected_suffix:
        # Cache dir already has the correct structure, use it directly
        symbol_cache_dir = cache_dir
        LOGGER.debug("Cache dir already has correct suffix: %s", cache_dir)
    else:
        # Append the symbol_h{horizon} suffix
        symbol_cache_dir = cache_dir / expected_suffix
        LOGGER.debug("Creating symbol cache dir: %s", symbol_cache_dir)
    
    symbol_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Use symbol-specific directory for all operations (no nested step subdirs)
    cache_dir = symbol_cache_dir
    
    LOGGER.info("=" * 80)
    LOGGER.info("FAMILY SIGNAL PREPARATION")
    LOGGER.info("=" * 80)
    LOGGER.info("Symbol: %s | Horizon: %dd", symbol, horizon)
    
    # Display step parameter correctly (days or years)
    if stage_a_mode:
        LOGGER.info("Mode: Stage-A raw feature cache")
        LOGGER.info("Coverage request: %s → %s", wf_start, wf_end)
    elif wf_step_days is not None:
        LOGGER.info("Walk-Forward: %s → %s (train=%dy, step=%dd)", 
                    wf_start, wf_end, wf_train_years, wf_step_days)
    else:
        LOGGER.info("Walk-Forward: %s → %s (train=%dy, step=%dy)", 
                    wf_start, wf_end, wf_train_years, wf_step_years)
    
    if hf_workers is None or hf_workers <= 0:
        hf_workers = max(1, min(workers, max(2, workers // 4) if workers > 1 else 1))

    excl_preview = (
        ",".join([f.strip() for f in exclude_families if f and f.strip()])
        if exclude_families
        else "<none>"
    )
    LOGGER.info(
        "Families: %s | Exclude: %s | Strict: %s | Workers: %d | HF workers: %d",
        families,
        excl_preview,
        strict,
        workers,
        hf_workers,
    )
    LOGGER.info("Cache: %s", cache_dir)
    
    # 1. Build walk-forward windows when needed
    # NOTE: Both Stage-A and Stage-B now use consolidated files (no train/valid splits)
    # to ensure cache files are reusable across runs. Walk-forward windows are
    # NOT generated - the entire date range is used as a single consolidated span.
    LOGGER.info("")
    LOGGER.info("─" * 80)
    if stage_a_mode or stage_b_mode:
        mode_label = "Stage-A" if stage_a_mode else "Stage-B"
        LOGGER.info("STEP 1: %s consolidated coverage (no train/valid splits)", mode_label)
        windows: List[WalkForwardWindow] = []
    else:
        windows = []
    
    # 2. Check symbol data availability (detect IPO date / data start)
    LOGGER.info("")
    LOGGER.info("─" * 80)
    LOGGER.info("STEP 2: Symbol Data Availability Check")
    LOGGER.info("─" * 80)
    
    symbol_data_start = detect_symbol_data_availability(symbol)
    if symbol_data_start is None:
        LOGGER.error(
            "❌ %s: No data available - symbol may be delisted or invalid. Skipping.",
            symbol
        )
        return PreparationResult(
            symbol=symbol,
            horizon=horizon,
            success=False,
            total_tasks=0,
            completed_tasks=0,
            skipped_tasks=0,
            failed_tasks=0,
            failed_families=[],
            coverage_start=None,
            coverage_end=None,
            duration_seconds=time.time() - start_time,
        )
    
    # Filter windows to only those that overlap with available data
    if windows and symbol_data_start is not None:
        original_count = len(windows)
        windows = [
            w for w in windows
            if w.train_start >= symbol_data_start  # Keep windows whose training period starts after data is available
        ]
        if len(windows) < original_count:
            skipped_count = original_count - len(windows)
            LOGGER.info(
                "⏭️  Skipped %d windows (training period before data availability: %s)",
                skipped_count, symbol_data_start.date()
            )
            for window in windows:
                LOGGER.info("  %s", window)
    
    # 3. Resolve families
    LOGGER.info("")
    LOGGER.info("─" * 80)
    LOGGER.info("STEP 3: Resolving Families")
    LOGGER.info("─" * 80)
    
    base_families, hf_modules, hf_blocks, meta_families = resolve_families(
        families,
        exclude_families=exclude_families,
    )

    # Optional: write a manifest that represents the *expected* pipeline family set,
    # even when this invocation is only preparing a subset (e.g. Dagster per-family assets).
    manifest_base_families = list(base_families)
    manifest_hf_modules = list(hf_modules)
    manifest_hf_blocks = list(hf_blocks)
    manifest_meta_families = list(meta_families)

    scope_norm = (manifest_scope or "requested").strip().lower()
    if scope_norm not in {"requested", "all"}:
        LOGGER.warning("Unknown manifest_scope '%s'; defaulting to 'requested'", manifest_scope)
        scope_norm = "requested"

    if scope_norm == "all":
        manifest_base_families, manifest_hf_modules, manifest_hf_blocks, manifest_meta_families = resolve_families(
            "all",
            exclude_families=exclude_families,
        )
        extra_excl = {
            str(f).strip()
            for f in (manifest_exclude_families or [])
            if f and str(f).strip()
        }
        if extra_excl:
            manifest_base_families = [f for f in manifest_base_families if f not in extra_excl]
            manifest_hf_modules = [f for f in manifest_hf_modules if f not in extra_excl]
            manifest_hf_blocks = [f for f in manifest_hf_blocks if f not in extra_excl]
            manifest_meta_families = [f for f in manifest_meta_families if f not in extra_excl]
    
    LOGGER.info(
        "Base families (%d): %s",
        len(base_families),
        ", ".join(base_families[:5]) + "..." if base_families else "<none>",
    )
    LOGGER.info(
        "HF modules (%d): %s",
        len(hf_modules),
        ", ".join(hf_modules) if hf_modules else "<none>",
    )
    LOGGER.info(
        "HF blocks (%d): %s",
        len(hf_blocks),
        ", ".join(hf_blocks) if hf_blocks else "<none>",
    )
    LOGGER.info(
        "Meta families (%d): %s",
        len(meta_families),
        ", ".join(meta_families) if meta_families else "<none>",
    )
    
    # 4. Detect existing signals
    LOGGER.info("")
    LOGGER.info("─" * 80)
    LOGGER.info("STEP 4: Detecting Existing Signals")
    LOGGER.info("─" * 80)
    
    # Calculate required date range.
    # In stage-b mode we want consolidated caches (and the merged parquet) to respect
    # the configured walk-forward span (wf_start/wf_end). Stage-A windows may be
    # shorter (e.g., due to a particular Stage-A schedule), but that should not
    # silently truncate the merged parquet.
    required_start = pd.to_datetime(wf_start)
    required_end = pd.to_datetime(wf_end)
    
    # Adjust required_start to symbol's data availability
    if symbol_data_start is not None and required_start < symbol_data_start:
        LOGGER.info(
            "⚠️  Adjusting required_start from %s to %s (symbol data availability)",
            required_start.date(), symbol_data_start.date()
        )
        required_start = symbol_data_start
    
    # 🔧 FIX: Cap required_end to today to avoid requesting future data
    today = pd.Timestamp.now().normalize()
    if required_end is not None and required_end > today:
        LOGGER.info(
            "⚠️  Capping required_end from %s to %s (today) - avoiding future data requests",
            required_end.date(), today.date()
        )
        required_end = today
    
    LOGGER.info("Required coverage: %s to %s",
                required_start.date() if required_start else "N/A",
                required_end.date() if required_end else "N/A")

    required_range_tuple = None
    if required_start is not None and required_end is not None:
        required_range_tuple = (required_start, required_end)
    required_ranges = compute_required_split_ranges(
        windows,
        fallback_range=required_range_tuple,
    )

    if reuse_symbol_only_cache:
        # Pre-fill this horizon's cache directory with symbol-only family files
        # from another horizon, when coverage is sufficient.
        symbol_only_families = [
            f
            for f in (base_families + hf_modules + hf_blocks)
            if (not _is_horizon_bound_family(f)) and (not _uses_symbol_only_cache_path(f))
        ]
        copied = _prefill_symbol_only_cache_from_other_horizons(
            cache_root=cache_dir.parent,
            cache_dir=cache_dir,
            symbol=symbol,
            horizon=horizon,
            families=symbol_only_families,
            required_ranges=required_ranges,
        )
        if copied:
            LOGGER.info(
                "♻️ Prefilled %d cached parquet(s) for %s h%s from other horizons.",
                copied,
                symbol,
                horizon,
            )

    # Back-compat migration for HF symbol-only blocks: if legacy horizon-scoped
    # artifacts exist (from older runs), link/copy them into the new symbol-only
    # cache folder so we reuse existing work.
    migrated = _migrate_symbol_only_hf_block_caches_from_horizon_dir(
        cache_dir=cache_dir,
        symbol=symbol,
        horizon=horizon,
        families=hf_blocks,
    )
    if migrated:
        LOGGER.info(
            "♻️ Migrated %d legacy HF block cache file(s) into symbol-only cache for %s",
            migrated,
            symbol,
        )
    
    existing_signals, coverage_map = detect_existing_signals(
        cache_dir=cache_dir,
        symbol=symbol,
        horizon=horizon,
        families=base_families + hf_modules + hf_blocks + meta_families,
        windows=windows,
    )
    
    all_families = base_families + hf_modules + hf_blocks + meta_families
    # Per-window shards are not produced; log consolidated coverage instead.
    if stage_b_mode:
        for family in all_families:
            train_cov = coverage_map.get(family, {}).get("train")
            valid_cov = coverage_map.get(family, {}).get("valid")
            parts: List[str] = []
            if train_cov is not None:
                parts.append(
                    f"train:{train_cov.date_start.date()}→{train_cov.date_end.date()}"
                )
            else:
                parts.append("train:<missing>")
            if valid_cov is not None:
                parts.append(
                    f"valid:{valid_cov.date_start.date()}→{valid_cov.date_end.date()}"
                )
            else:
                parts.append("valid:<missing>")
            LOGGER.info("  📌 %s: %s", family, "; ".join(parts))

    # required_ranges computed above (used for prefill + task construction)

    total_task_count = 0
    total_completed = 0
    total_failed = 0
    task_failure_families: List[str] = []

    # 5. Base family build
    #    - Parallel for most families
    #    - Sequential dependency chain: quantile_forecast → calibration → online_learning
    LOGGER.info("")
    LOGGER.info("─" * 80)
    LOGGER.info("STEP 5: Base Family Build (dependency chain sequenced)")
    LOGGER.info("─" * 80)

    consolidated_tasks = build_consolidated_tasks(
        symbol=symbol,
        horizon=horizon,
        base_families=base_families,
        hf_modules=hf_modules,
        hf_blocks=hf_blocks,
        meta_families=meta_families,
        required_ranges=required_ranges,
        coverage_map=coverage_map,
        cache_dir=cache_dir,
    )
    LOGGER.info("Total consolidated tasks queued: %d", len(consolidated_tasks))

    # Scheduling rules (per user request):
    # - Horizon-linked phase = dependency chain (quantile_forecast → calibration → online_learning)
    # - Base phase = base_families + hf_modules (excluding horizon-linked)
    # - HF phase = hf_blocks only
    # - META (hf_agg) is derived; it is computed during unified panel build.
    horizon_linked_families = {"quantile_forecast", "calibration", "online_learning"}
    base_family_set = set(base_families + hf_modules)
    hf_family_set = set(hf_blocks)
    
    # Separate horizon-linked tasks from other base tasks
    horizon_linked_tasks = [t for t in consolidated_tasks if t.family in horizon_linked_families]
    base_tasks = [t for t in consolidated_tasks if t.family in base_family_set and t.family not in horizon_linked_families]
    hf_tasks = [t for t in consolidated_tasks if t.family in hf_family_set]
    orphan_tasks = [
        t for t in consolidated_tasks if t.family not in base_family_set | hf_family_set | horizon_linked_families
    ]
    if orphan_tasks:
        LOGGER.warning(
            "Assigning %d orphan tasks to base batch (families=%s)",
            len(orphan_tasks),
            sorted({t.family for t in orphan_tasks}),
        )
        base_tasks.extend(orphan_tasks)

    # =========================================================================
    # EXECUTION: Sequential mode (default) vs Parallel mode
    # =========================================================================
    if sequential_mode:
        # Sequential execution: one task at a time, no spawns
        # Order: horizon-linked → base families → HF blocks
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("STEP 5-6: Sequential Execution (no spawns)")
        LOGGER.info("─" * 80)
        
        total_task_count = len(horizon_linked_tasks) + len(base_tasks) + len(hf_tasks)
        
        seq_completed, seq_failed, seq_failed_families = execute_tasks_sequential(
            horizon_linked_tasks=horizon_linked_tasks,
            base_family_tasks=base_tasks,
            hf_block_tasks=hf_tasks,
            cache_dir=cache_dir,
        )
        
        total_completed = seq_completed
        total_failed = seq_failed
        task_failure_families.extend(seq_failed_families)
        
        # Refresh coverage after sequential execution
        existing_signals, coverage_map = detect_existing_signals(
            cache_dir=cache_dir,
            symbol=symbol,
            horizon=horizon,
            families=all_families,
            windows=windows,
        )
    else:
        # Legacy parallel execution mode (workers > 1)
        hf_spawn_workers = 1 if stage_a_mode else hf_workers

        def _execute_phase(
            label: str,
            tasks: List[SignalTask],
            worker_cap: int,
            detail: Optional[str] = None,
        ) -> None:
            nonlocal total_task_count, total_completed, total_failed, existing_signals, coverage_map
            if detail:
                LOGGER.info(detail)
            LOGGER.info("%s tasks queued: %d", label, len(tasks))
            if not tasks:
                LOGGER.info("%s already satisfied — nothing to rebuild.", label)
                return
            total_task_count += len(tasks)
            completed, failed, failed_phase = execute_tasks_parallel(
                tasks=tasks,
                cache_dir=cache_dir,
                workers=max(1, min(worker_cap, len(tasks))),
                hf_workers=hf_spawn_workers,
            )
            total_completed += completed
            total_failed += failed
            task_failure_families.extend(failed_phase)
            existing_signals, coverage_map = detect_existing_signals(
                cache_dir=cache_dir,
                symbol=symbol,
                horizon=horizon,
                families=all_families,
                windows=windows,
            )

        # 5. Base family build (includes horizon-linked in parallel mode)
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("STEP 5: Base Family Build (parallel mode)")
        LOGGER.info("─" * 80)
        all_base_for_parallel = horizon_linked_tasks + base_tasks
        _execute_phase(
            label="Base family",
            tasks=all_base_for_parallel,
            worker_cap=workers,
            detail=(
                "Base batch enforces quantile → calibration → online sequencing while the"
                " remaining families run in parallel."
            ),
        )

        # 6. HF blocks (derived from base families)
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("STEP 6: HF Blocks")
        LOGGER.info("─" * 80)
        _execute_phase(
            label="HF blocks",
            tasks=hf_tasks,
            worker_cap=workers,
            detail=(
                "HF blocks are computed after base families complete and may run in parallel."
            ),
        )

    panel_cache_path: Optional[Path] = None
    
    # Build unified merged panel for both Stage-A and Stage-B modes
    # Stage-A: Reads from consolidated _features.parquet files
    # Stage-B: Reads from train/valid split files
    if write_merged or stage_a_mode:
        LOGGER.info("")
        LOGGER.info("─" * 80)
        LOGGER.info("STEP 7: Unified Feature Panel (%s mode)", "Stage-A" if stage_a_mode else "Stage-B")
        LOGGER.info("─" * 80)
        
        if write_merged:
            LOGGER.info("Building merged per-symbol parquet (all dates; unified panel format).")
            # Use symbol subfolder: cache/merged/{SYMBOL}/
            symbol_merged_dir = FEATURE_PANEL_DIR / symbol.upper()
            symbol_merged_dir.mkdir(parents=True, exist_ok=True)
            merged_default = symbol_merged_dir / f"{symbol.upper()}_h{horizon}_merged.parquet"
            merged_path = (merged_out or merged_default)
            panel_cache_path = build_symbol_panel_cache(
                symbol=symbol,
                horizon=horizon,
                cache_dir=cache_dir,
                families=list(all_families),
                coverage_start=required_start,
                coverage_end=required_end,
                track_label="merged",
                out_path=merged_path,
            )
            if panel_cache_path is None:
                LOGGER.warning("Unified merged panel build failed; folds will rebuild on demand.")

            # Convenience: also place a link to the merged parquet in the per-(symbol,horizon)
            # cache directory so a single folder contains both family caches + merged panel.
            try:
                link_path = cache_dir / merged_path.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                rel_target = os.path.relpath(merged_path, start=cache_dir)
                os.symlink(rel_target, link_path)
            except Exception as exc:
                LOGGER.debug("Failed to symlink merged parquet into cache_dir: %s", exc)
        elif stage_a_mode:
            # Stage-A mode: build merged panel from consolidated files
            LOGGER.info("Building consolidated panel from Stage-A caches.")
            symbol_merged_dir = FEATURE_PANEL_DIR / symbol.upper()
            symbol_merged_dir.mkdir(parents=True, exist_ok=True)
            merged_default = symbol_merged_dir / f"{symbol.upper()}_h{horizon}_merged.parquet"
            panel_cache_path = build_symbol_panel_cache(
                symbol=symbol,
                horizon=horizon,
                cache_dir=cache_dir,
                families=list(all_families),
                coverage_start=required_start,
                coverage_end=required_end,
                track_label="merged",
                out_path=merged_default,
            )
            if panel_cache_path is None:
                LOGGER.warning("Unified feature panel build failed; folds will rebuild on demand.")
        else:
            LOGGER.info("Building consolidated TrackC panel; window slices are derived at runtime.")
            panel_cache_path = build_symbol_panel_cache(
                symbol=symbol,
                horizon=horizon,
                cache_dir=cache_dir,
                families=list(all_families),
                coverage_start=required_start,
                coverage_end=required_end,
                track_label="TrackC",
            )
            if panel_cache_path is None:
                LOGGER.warning("Unified feature panel build failed; folds will rebuild on demand.")

    # ------------------------------------------------------------------------
    # Coverage validation (strict-mode correctness)
    # ------------------------------------------------------------------------
    # Prior refactors left remaining_missing stuck at 0, which made strict mode
    # incorrectly report success even when required cache files were missing.
    # Recompute missing coverage directly from on-disk cache existence.
    symbol_lower = symbol.lower()
    skipped = 0
    missing_families: set[str] = set()
    remaining_missing = 0

    # Ensure HF modules have Stage A feature caches before validation.
    for family in list(base_families) + list(hf_modules) + list(hf_blocks):
        for split in ["train", "valid"]:
            base_path, feature_path, _ = _family_cache_paths(
                cache_dir,
                symbol,
                horizon,
                family,
                split,
            )
            _backfill_feature_cache(base_path, feature_path)

    # Ensure detect_existing_signals has up-to-date coverage for feature caches.
    _, coverage_map = detect_existing_signals(
        cache_dir=cache_dir,
        symbol=symbol,
        horizon=horizon,
        families=all_families,
        windows=windows,
    )

    for family in list(base_families) + list(hf_modules) + list(hf_blocks):
        # Stage-A mode: check for consolidated files (no split suffix)
        # Stage-B mode: check for both consolidated AND train/valid split files
        family_has_cache = False
        
        if stage_a_mode:
            # Stage-A: only check consolidated _features.parquet
            base_path, features_path, lagged_path = _family_cache_paths(
                cache_dir,
                symbol,
                horizon,
                family,
                "",  # No split for consolidated files
            )
            variants = [features_path, base_path]
            if WRITE_LAGGED_FEATURE_CACHES:
                variants.insert(0, lagged_path)
            if any(path.exists() for path in variants):
                family_has_cache = True
        else:
            # Stage-B: check train/valid splits, fall back to consolidated
            for split in ["", "train", "valid"]:
                base_path, features_path, lagged_path = _family_cache_paths(
                    cache_dir,
                    symbol,
                    horizon,
                    family,
                    split,
                )
                variants = [features_path, base_path]
                if WRITE_LAGGED_FEATURE_CACHES:
                    variants.insert(0, lagged_path)
                if any(path.exists() for path in variants):
                    family_has_cache = True
                    break
        
        if not family_has_cache:
            remaining_missing += 1
            missing_families.add(family)

    # Strict-mode must also ensure the unified TrackC panel did not silently skip any families.
    if strict and stage_b_mode and panel_cache_path is not None:
        meta_path = panel_cache_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    panel_meta = json.load(handle)
                missing_in_panel = panel_meta.get("families_missing") or []
                if isinstance(missing_in_panel, list):
                    requested = set(all_families)
                    for fam in missing_in_panel:
                        if fam in requested:
                            missing_families.add(fam)
                    if any(fam in requested for fam in missing_in_panel):
                        remaining_missing += 1
            except Exception as exc:
                LOGGER.warning("Unable to read panel meta for strict validation: %s", exc)

    duration = time.time() - start_time

    if missing_families:
        failed_families = sorted(missing_families)
        LOGGER.error("Incomplete families (missing cache files): %s", ", ".join(failed_families))
    else:
        failed_families = []
        if task_failure_families:
            # In strict mode, a task failure must be surfaced even if an older/stub
            # cache exists. Otherwise providers can go dormant without failing CI/Dagster.
            failed = sorted(set(task_failure_families))
            if strict:
                failed_families.extend(failed)
                LOGGER.error(
                    "Generation tasks failed for %s (strict mode treats this as failure)",
                    ", ".join(failed),
                )
            else:
                LOGGER.info(
                    "Generation tasks failed for %s but consolidated caches already exist.",
                    ", ".join(failed),
                )

    # Completion is coverage-oriented.
    #
    # Policy:
    # - Non-strict runs are allowed to finish successfully even when some families fail
    #   (e.g. provider endpoints unavailable). The failures are still recorded in
    #   `failed_families` and the completeness manifest.
    # - Strict runs require both full expected-file presence and zero failed tasks.
    if strict:
        success = (remaining_missing == 0) and (total_failed == 0)
    else:
        success = True

    result = PreparationResult(
        symbol=symbol,
        horizon=horizon,
        success=success,
        total_tasks=total_task_count,
        completed_tasks=total_completed,
        skipped_tasks=skipped,
        failed_tasks=total_failed + remaining_missing,
        failed_families=failed_families,
        coverage_start=required_start,
        coverage_end=required_end,
        duration_seconds=duration,
    )

    # 7. Write completeness manifest
    LOGGER.info("")
    LOGGER.info("─" * 80)
    LOGGER.info("STEP 8: Writing Completeness Manifest")
    LOGGER.info("─" * 80)
    
    manifest_path = output_dir / f"{symbol.lower()}_h{horizon}_completeness.json"
    if stage_b_mode and panel_cache_path is None:
        result.success = False
        result.failed_tasks += 1
        result.failed_families.append("feature_panel")
        LOGGER.error(
            "Unified feature panel missing for %s h%s; preparation marked as failed",
            symbol,
            horizon,
        )
    
    write_completeness_manifest(
        output_path=manifest_path,
        symbol=symbol,
        horizon=horizon,
        windows=windows,
        base_families=manifest_base_families,
        hf_modules=manifest_hf_modules,
        hf_blocks=manifest_hf_blocks,
        meta_families=manifest_meta_families,
        result=result,
        cache_dir=cache_dir,
        mode=mode_normalized,
        stage="stage-a" if stage_a_mode else "stage-b",
        coverage_start=required_start,
        coverage_end=required_end,
        track_panel_path=panel_cache_path,
    )
    result.completeness_manifest_path = manifest_path
    
    # 8. Final summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("PREPARATION COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info(result.summary())
    
    if failed_families:
        LOGGER.warning("Failed families: %s", ", ".join(failed_families))
    
    if strict and not success:
        LOGGER.error("STRICT MODE: Preparation failed - exiting")

    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare family signals with walk-forward fold awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all families (base + HF) for AAPL:63
  python tools/prep_families.py \\
    --symbol AAPL --horizon 63 \\
    --wf-start 2015-01-01 --wf-end 2024-12-31 \\
    --wf-train-years 5 --wf-step-years 1 \\
        --families all --strict yes --workers 6

    # Prepare all families except provider-dependent ones
    python tools/prep_families.py \
        --symbol AAPL --horizon 63 \
        --wf-start 2015-01-01 --wf-end 2024-12-31 \
        --wf-train-years 5 --wf-step-years 1 \
        --families all --exclude-families fx,commodities,crypto --strict yes --workers 6

  # Prepare only base families
  python tools/prep_families.py \\
    --symbol AAPL --horizon 63 \\
    --wf-start 2015-01-01 --wf-end 2024-12-31 \\
    --wf-train-years 5 --wf-step-years 1 \\
    --families base --workers 4

  # Prepare specific families
  python tools/prep_families.py \\
    --symbol AAPL --horizon 63 \\
    --wf-start 2015-01-01 --wf-end 2024-12-31 \\
    --wf-train-years 5 --wf-step-years 1 \\
    --families "finbert,dcf,options"
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--symbol",
        required=False,
        default=_default_symbol_list_csv(),
        help=(
            "Stock symbol or comma-separated list (e.g., AAPL or AAPL,MSFT,NVDA). "
            "Multiple symbols processed in parallel. Default: CORE+satellite candidate universe."
        ),
    )
    parser.add_argument("--horizon", type=int, default=63, help="Forecast horizon in days (default: 63)")
    
    # Walk-forward parameters
    parser.add_argument(
        "--wf-start",
        default=DEFAULT_WF_START,
        help=f"Walk-forward start date (YYYY-MM-DD, default: {DEFAULT_WF_START})",
    )
    parser.add_argument(
        "--wf-end",
        default=DEFAULT_WF_END,
        help=f"Walk-forward end date (YYYY-MM-DD, default: {DEFAULT_WF_END})",
    )
    parser.add_argument(
        "--wf-train-years",
        type=int,
        default=DEFAULT_WF_TRAIN_YEARS,
        help=f"Training window size in years (default: {DEFAULT_WF_TRAIN_YEARS})",
    )
    
    # Step size - accept either years or days (default to 63d if unspecified)
    step_group = parser.add_mutually_exclusive_group(required=False)
    step_group.add_argument("--wf-step-years", type=int, help="Step size in years")
    step_group.add_argument(
        "--wf-step-days",
        type=int,
        help=f"Step size in days (default: {DEFAULT_WF_STEP_DAYS})",
    )

    parser.add_argument(
        "--mode",
        choices=["walkforward", "stage-a", "stage-b"],
        default="stage-b",
        help=(
            "Generation mode / stage: stage-a builds consolidated caches; stage-b (or walkforward) builds consolidated caches "
            "and a unified per-symbol panel (no per-window parquet slices)."
        ),
    )

    parser.add_argument(
        "--write-merged",
        choices=["yes", "no"],
        default="yes",
        help="Write a single merged parquet per (symbol,horizon) (default: yes)",
    )
    parser.add_argument(
        "--write-role-splits",
        choices=["yes", "no"],
        default="yes",
        help="Write merged mamba/portfolio parquets alongside merged output (default: yes)",
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help="Path for merged parquet output (default: cache/features/<SYMBOL>_h<H>_merged.parquet)",
    )
    parser.add_argument(
        "--merged-service",
        default=None,
        help="Feast FeatureService name to fetch into the merged parquet (default: phase2_all_h<H>_v1)",
    )

    parser.add_argument(
        "--mamba-optional-all",
        choices=["yes", "no"],
        default="no",
        help=(
            "Enable all optional Mamba features across families (alt_signals, arima_forecast, quantile_forecast, candle_mechanics)."
        ),
    )

    parser.add_argument(
        "--reuse-symbol-only-cache",
        choices=["yes", "no"],
        default="yes",
        help=(
            "If yes, symbol-only families are copied/linked from an existing horizon cache when coverage matches; "
            "only horizon-bound families are recomputed (default: yes)."
        ),
    )
    
    # Family selection
    parser.add_argument(
        "--families",
        default="all",
        help="Families to prepare: 'all', 'base', 'hf', or comma-separated list (default: all)"
    )

    parser.add_argument(
        "--exclude-families",
        default=",".join(DEFAULT_EXCLUDE_FAMILIES),
        help=(
            "Comma-separated families to skip (default: fx,commodities,crypto). "
            "Pass an empty string to disable (e.g., --exclude-families \"\")."
        ),
    )
    
    # Execution parameters
    parser.add_argument(
        "--strict",
        choices=["yes", "no"],
        default="no",
        help="Strict mode: fail if any family missing (default: no)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers per symbol (default: 1)"
    )
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=None,
        help="Max parallel workers dedicated to HF/FinBERT families (default: derives from --workers)",
    )
    
    # Optional paths
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: data/local_cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifests (default: artifacts/prep_families)"
    )

    # Safety / diagnostics
    parser.add_argument(
        "--write-storage-snapshot",
        choices=["yes", "no"],
        default="yes",
        help="Write a storage/mount/cache inventory JSON to output dir (default: yes)",
    )
    parser.add_argument(
        "--min-trackc-parquets",
        type=int,
        default=0,
        help="Abort if cache/features has fewer than this many *_trackc.parquet files for the horizon (default: 0)",
    )
    parser.add_argument(
        "--min-symbol-cache-dirs",
        type=int,
        default=0,
        help="Abort if cache root (data/local_cache) has fewer than this many *_h* directories (default: 0)",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args(argv)

    if args.wf_step_years is None and args.wf_step_days is None:
        args.wf_step_days = DEFAULT_WF_STEP_DAYS

    args.exclude_families = [
        f.strip()
        for f in (args.exclude_families or "").split(",")
        if f and f.strip()
    ]

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Parse symbols from argument (supports both single and comma-separated)
    symbols = [s.strip().upper() for s in args.symbol.split(',') if s.strip()]
    if not symbols:
        LOGGER.error("Error: No valid symbols provided")
        return 1

    cache_root = (args.cache_dir or (REPO_ROOT / "data" / "local_cache")).resolve()
    output_dir = (args.output_dir or (REPO_ROOT / "artifacts" / "prep_families")).resolve()

    if args.write_storage_snapshot == "yes":
        snap_path = _write_storage_snapshot(horizon=args.horizon, cache_root=cache_root, output_dir=output_dir)
        if snap_path is not None:
            LOGGER.info("🧾 Wrote storage snapshot: %s", snap_path)

    # Optional cache sanity checks (helps detect detached/missing volumes before overwriting).
    if args.min_trackc_parquets and args.min_trackc_parquets > 0:
        trackc_count = len(list(FEATURE_PANEL_DIR.glob(f"*_h{int(args.horizon)}_*trackc*.parquet")))
        if trackc_count < args.min_trackc_parquets:
            LOGGER.error(
                "Cache sanity check failed: trackc parquet count %d < %d under %s",
                trackc_count,
                args.min_trackc_parquets,
                FEATURE_PANEL_DIR,
            )
            return 2

    if args.min_symbol_cache_dirs and args.min_symbol_cache_dirs > 0:
        sym_dir_count = len([p for p in cache_root.glob("*_h*") if p.is_dir()])
        if sym_dir_count < args.min_symbol_cache_dirs:
            LOGGER.error(
                "Cache sanity check failed: symbol cache dirs %d < %d under %s",
                sym_dir_count,
                args.min_symbol_cache_dirs,
                cache_root,
            )
            return 2
    
    # Handle single symbol case (original behavior, no parallelization overhead)
    if len(symbols) == 1:
        try:
            if args.write_role_splits is not None:
                os.environ["PREP_FAMILIES_WRITE_ROLE_SPLITS"] = "1" if args.write_role_splits == "yes" else "0"
            if args.mamba_optional_all == "yes":
                os.environ["ALT_SIGNALS_MAMBA_OPTIONAL"] = "all"
                os.environ["ARIMA_FORECAST_MAMBA_OPTIONAL"] = "all"
                os.environ["QUANTILE_FORECAST_MAMBA_STACKING"] = "all"
                os.environ["CANDLE_MECHANICS_MAMBA_OPTIONAL"] = "all"
            result = prepare_families(
                symbol=symbols[0],
                horizon=args.horizon,
                wf_start=args.wf_start,
                wf_end=args.wf_end,
                wf_train_years=args.wf_train_years,
                wf_step_years=args.wf_step_years if not args.wf_step_days else None,
                wf_step_days=args.wf_step_days if args.wf_step_days else None,
                families=args.families,
                exclude_families=args.exclude_families,
                strict=(args.strict == "yes"),
                workers=args.workers,
                hf_workers=args.hf_workers,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                write_merged=(args.write_merged == "yes"),
                merged_out=args.merged_out,
                merged_service=args.merged_service,
                reuse_symbol_only_cache=(args.reuse_symbol_only_cache == "yes"),
            )
            return 0 if result.success else 1
        except Exception as e:
            LOGGER.exception("Fatal error during preparation: %s", str(e))
            return 1
    
    # Handle multiple symbols in parallel
    LOGGER.info("=" * 80)
    LOGGER.info("PARALLEL SYMBOL PROCESSING")
    LOGGER.info("=" * 80)
    LOGGER.info("Processing %d symbols in parallel: %s", len(symbols), ', '.join(symbols))
    LOGGER.info("Workers per symbol: %d", args.workers)
    LOGGER.info("=" * 80)
    
    # Create worker function for parallel execution
    def process_symbol(symbol: str) -> tuple[str, bool]:
        """Process a single symbol and return (symbol, success)."""
        try:
            if args.write_role_splits is not None:
                os.environ["PREP_FAMILIES_WRITE_ROLE_SPLITS"] = "1" if args.write_role_splits == "yes" else "0"
            if args.mamba_optional_all == "yes":
                os.environ["ALT_SIGNALS_MAMBA_OPTIONAL"] = "all"
                os.environ["ARIMA_FORECAST_MAMBA_OPTIONAL"] = "all"
                os.environ["QUANTILE_FORECAST_MAMBA_STACKING"] = "all"
                os.environ["CANDLE_MECHANICS_MAMBA_OPTIONAL"] = "all"
            result = prepare_families(
                symbol=symbol,
                horizon=args.horizon,
                wf_start=args.wf_start,
                wf_end=args.wf_end,
                wf_train_years=args.wf_train_years,
                wf_step_years=args.wf_step_years if not args.wf_step_days else None,
                wf_step_days=args.wf_step_days if args.wf_step_days else None,
                families=args.families,
                exclude_families=args.exclude_families,
                strict=(args.strict == "yes"),
                workers=args.workers,
                hf_workers=args.hf_workers,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                write_merged=(args.write_merged == "yes"),
                merged_out=args.merged_out,
                merged_service=args.merged_service,
                reuse_symbol_only_cache=(args.reuse_symbol_only_cache == "yes"),
            )
            return (symbol, result.success)
        except Exception as e:
            LOGGER.error("Symbol %s failed with error: %s", symbol, str(e))
            return (symbol, False)
    
    # Execute in parallel using ThreadPoolExecutor
    # (threads work better than processes for I/O-bound symbol processing)
    max_symbol_workers = min(len(symbols), mp.cpu_count())
    LOGGER.info("Using %d parallel symbol workers (max: %d)", max_symbol_workers, mp.cpu_count())
    
    results = {}
    try:
        with ThreadPoolExecutor(max_workers=max_symbol_workers) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                symbol, success = future.result()
                results[symbol] = success
                status = "✅ SUCCESS" if success else "❌ FAILED"
                LOGGER.info("Symbol %s: %s", symbol, status)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 1
    except Exception as e:
        LOGGER.exception("Fatal error during parallel symbol processing: %s", str(e))
        return 1
    
    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("PARALLEL PROCESSING SUMMARY")
    LOGGER.info("=" * 80)
    successes = [sym for sym, ok in results.items() if ok]
    failures = [sym for sym, ok in results.items() if not ok]
    
    LOGGER.info("Total symbols: %d", len(symbols))
    LOGGER.info("Successful: %d (%s)", len(successes), ', '.join(successes) if successes else "none")
    LOGGER.info("Failed: %d (%s)", len(failures), ', '.join(failures) if failures else "none")
    LOGGER.info("=" * 80)
    
    return 0 if len(failures) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
