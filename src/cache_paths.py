"""Centralized cache path definitions.

This module defines the canonical cache layout for the entire project.
All other modules should import paths from here instead of hardcoding them.

Directory Structure:
    cache/
    ├── merged/                           # Final merged parquets (one per symbol/horizon)
    │   ├── <SYMBOL>_h<H>_merged.parquet
    │   ├── <SYMBOL>_h<H>_merged.meta.json
    │   └── <SYMBOL>_h<H>_merged.leakage_audit.json
    │
    ├── symbols/                          # Per-symbol feature caches
    │   └── <SYMBOL>/
    │       └── h<H>/                     # Horizon-specific families ONLY
    │           ├── tech.parquet
    │           ├── fundamental.parquet
    │           └── ...                   # Only horizon-dependent families
    │
    ├── shared/                           # Symbol-invariant caches (shared across all symbols)
    │   ├── doc_embedding/                # Document embedding novelty (GDELT-based, symbol-independent)
    │   │   └── doc_embedding_novelty_hf.parquet
    │   └── macro/                        # Macro indicators
    │
    ├── data_sources/                     # External data provider caches
    │   ├── eodhd/
    │   │   ├── fundamentals/<SYMBOL>.json
    │   │   ├── delisted_companies_US.parquet
    │   │   └── group_map.csv
    │   └── gdelt_global/merged/<year>/<month>.parquet
    │
    └── universe/                         # Universe registries
        ├── registry_YYYYMMDD.parquet
        └── two_tier/snapshot_YYYYMMDD_h<H>.parquet

    artifacts/                            # Training outputs (NOT caches)
        ├── optuna/
        ├── stage_a/
        ├── backtests/
        └── models/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# Repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# MAIN CACHE DIRECTORIES
# =============================================================================

# Main cache directory
CACHE_ROOT = REPO_ROOT / "cache"

# Merged parquets directory (final outputs)
MERGED_CACHE_ROOT = CACHE_ROOT / "merged"

# Symbol-level caches (individual families)
SYMBOLS_CACHE_ROOT = CACHE_ROOT / "symbols"

# External data source caches
DATA_SOURCES_CACHE_ROOT = CACHE_ROOT / "data_sources"

# Shared (symbol-invariant) caches - doc_embedding, macro, etc.
SHARED_CACHE_ROOT = CACHE_ROOT / "shared"

# Doc embedding specifically (symbol-independent, shared across all)
DOC_EMBEDDING_SHARED_DIR = SHARED_CACHE_ROOT / "doc_embedding"

# Universe registries
UNIVERSE_CACHE_ROOT = CACHE_ROOT / "universe"

# Artifacts directory (training outputs, not caches)
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

# =============================================================================
# LEGACY PATHS (for backward compatibility during migration)
# =============================================================================

LEGACY_FEATURE_PANEL_DIR = CACHE_ROOT / "features"
LEGACY_LOCAL_CACHE_DIR = REPO_ROOT / "data" / "local_cache"
LEGACY_EODHD_CACHE_DIR = REPO_ROOT / "data" / "cache" / "eodhd"
LEGACY_GDELT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "gdelt"
LEGACY_GDELT_GLOBAL_CACHE_DIR = REPO_ROOT / "data" / "cache" / "gdelt_global"
LEGACY_UNIVERSE_CACHE_DIR = REPO_ROOT / "data" / "cache" / "universe"
LEGACY_SHORT_INTEREST_CACHE_DIR = REPO_ROOT / "data" / "short_interest_cache"


# =============================================================================
# DATA SOURCE CACHE PATHS
# =============================================================================

# EODHD
EODHD_CACHE_ROOT = DATA_SOURCES_CACHE_ROOT / "eodhd"
EODHD_FUNDAMENTALS_DIR = EODHD_CACHE_ROOT / "fundamentals"


def eodhd_fundamentals_path(symbol: str) -> Path:
    """Return path to cached EODHD fundamentals JSON for a symbol."""
    return EODHD_FUNDAMENTALS_DIR / f"{symbol.upper()}.json"


def eodhd_delisting_path() -> Path:
    """Return path to delisted companies parquet."""
    return EODHD_CACHE_ROOT / "delisted_companies_US.parquet"


def eodhd_group_map_path() -> Path:
    """Return path to symbol→sector group map CSV."""
    return EODHD_CACHE_ROOT / "group_map.csv"


# GDELT
GDELT_CACHE_ROOT = DATA_SOURCES_CACHE_ROOT / "gdelt"
GDELT_GLOBAL_CACHE_ROOT = DATA_SOURCES_CACHE_ROOT / "gdelt_global"


def gdelt_events_dir(date_str: str) -> Path:
    """Return path to GDELT events directory for a date."""
    return GDELT_CACHE_ROOT / date_str


def gdelt_global_merged_dir() -> Path:
    """Return path to GDELT global merged cache."""
    return GDELT_GLOBAL_CACHE_ROOT / "merged"


# =============================================================================
# EMBEDDING CACHE PATHS (now under shared/)
# =============================================================================

# FinBERT and doc_novelty embeddings are symbol-specific, stored under shared/embeddings/
EMBEDDINGS_CACHE_ROOT = SHARED_CACHE_ROOT / "embeddings"
FINBERT_CACHE_ROOT = EMBEDDINGS_CACHE_ROOT / "finbert"
DOC_NOVELTY_CACHE_ROOT = EMBEDDINGS_CACHE_ROOT / "doc_novelty"


def finbert_embeddings_dir(symbol: str) -> Path:
    """Return path to FinBERT embeddings cache for a symbol."""
    return FINBERT_CACHE_ROOT / symbol.upper()


def doc_novelty_embeddings_dir(symbol: str) -> Path:
    """Return path to document novelty embeddings for a symbol."""
    return DOC_NOVELTY_CACHE_ROOT / symbol.upper()


# =============================================================================
# SYMBOL-LEVEL CACHE PATHS
# =============================================================================

def symbol_cache_dir(symbol: str) -> Path:
    """Return the root cache directory for a symbol."""
    return SYMBOLS_CACHE_ROOT / symbol.upper()


def symbol_transcripts_dir(symbol: str) -> Path:
    """Return path to earnings transcripts cache for a symbol."""
    return symbol_cache_dir(symbol) / "transcripts"


def symbol_short_interest_path(symbol: str) -> Path:
    """Return path to short interest history JSON for a symbol."""
    return symbol_cache_dir(symbol) / "short_interest" / "history.json"


def symbol_horizon_dir(symbol: str, horizon: int) -> Path:
    """Return the horizon-specific cache directory for a symbol."""
    return symbol_cache_dir(symbol) / f"h{horizon}"


def symbol_hf_dir(symbol: str) -> Path:
    """Return the horizon-invariant HF cache directory for a symbol."""
    return symbol_cache_dir(symbol) / "hf"


def symbol_families_dir(symbol: str, horizon: int) -> Path:
    """Return the individual family caches directory (horizon-specific families only)."""
    return symbol_horizon_dir(symbol, horizon)


# -----------------------------------------------------------------------------
# Merged parquet paths (final outputs in cache/merged/)
# -----------------------------------------------------------------------------

def merged_parquet_path(symbol: str, horizon: int) -> Path:
    """Return the path to the merged parquet for a symbol/horizon."""
    return MERGED_CACHE_ROOT / f"{symbol.upper()}_h{horizon}_merged.parquet"


def merged_meta_path(symbol: str, horizon: int) -> Path:
    """Return the path to the merged provenance JSON."""
    return MERGED_CACHE_ROOT / f"{symbol.upper()}_h{horizon}_merged.meta.json"


def merged_leakage_audit_path(symbol: str, horizon: int) -> Path:
    """Return the path to the leakage audit JSON."""
    return MERGED_CACHE_ROOT / f"{symbol.upper()}_h{horizon}_merged.leakage_audit.json"


def features_parquet_path(symbol: str, horizon: int) -> Path:
    """Return the path to the numeric-only features parquet."""
    return symbol_horizon_dir(symbol, horizon) / "features.parquet"


def index_parquet_path(symbol: str, horizon: int) -> Path:
    """Return the path to the index sidecar parquet."""
    return symbol_horizon_dir(symbol, horizon) / "index.parquet"


def trackc_parquet_path(symbol: str, horizon: int) -> Path:
    """Return the path to the Track-C parquet."""
    return symbol_horizon_dir(symbol, horizon) / "trackc.parquet"


def trackc_meta_path(symbol: str, horizon: int) -> Path:
    """Return the path to the Track-C metadata."""
    return symbol_horizon_dir(symbol, horizon) / "trackc.meta.json"


# -----------------------------------------------------------------------------
# Family cache paths
# -----------------------------------------------------------------------------

def family_cache_path(symbol: str, horizon: int, family: str, split: str = "") -> Path:
    """Return the path to an individual family cache parquet.
    
    Note: split parameter is deprecated and ignored. Using single file per family.
    """
    return symbol_families_dir(symbol, horizon) / f"{family}.parquet"


def family_features_cache_path(symbol: str, horizon: int, family: str, split: str = "") -> Path:
    """Return the path to an individual family features cache parquet.
    
    Note: split parameter is deprecated and ignored. Using single file per family.
    """
    return symbol_families_dir(symbol, horizon) / f"{family}_features.parquet"


# -----------------------------------------------------------------------------
# Shared cache paths (symbol-invariant, used across ALL symbols)
# -----------------------------------------------------------------------------

def doc_embedding_shared_path() -> Path:
    """Return the path to the shared doc_embedding_novelty_hf parquet.
    
    This is SYMBOL-INDEPENDENT - computed once from GDELT and shared across all symbols.
    """
    return DOC_EMBEDDING_SHARED_DIR / "doc_embedding_novelty_hf.parquet"


def shared_hf_block_path(hf_block: str) -> Path:
    """Return the path to a shared (symbol-invariant) HF block cache.
    
    Use this for blocks like doc_embedding_novelty_hf that don't depend on symbol.
    """
    return SHARED_CACHE_ROOT / hf_block / f"{hf_block}.parquet"


# -----------------------------------------------------------------------------
# Universe paths
# -----------------------------------------------------------------------------

def universe_registry_path(date_str: str) -> Path:
    """Return the path to a universe registry parquet."""
    return UNIVERSE_CACHE_ROOT / f"registry_{date_str}.parquet"


def universe_two_tier_path(date_str: str, horizon: int) -> Path:
    """Return the path to a two-tier universe snapshot."""
    return UNIVERSE_CACHE_ROOT / "two_tier" / f"snapshot_{date_str}_h{horizon}.parquet"


# -----------------------------------------------------------------------------
# Artifact paths
# -----------------------------------------------------------------------------

def optuna_study_path(study_name: str) -> Path:
    """Return the path to an Optuna study SQLite DB."""
    return ARTIFACTS_ROOT / "optuna_studies" / f"{study_name}.db"


def optuna_winner_path(study_name: str) -> Path:
    """Return the path to an Optuna winner JSON."""
    return ARTIFACTS_ROOT / "optuna" / f"{study_name}_best.json"


def stage_a_weights_path(horizon: int, date_str: str) -> Path:
    """Return the path to Stage-A family weights."""
    return ARTIFACTS_ROOT / "stage_a" / f"h{horizon}" / date_str / "family_weights_best.json"


def stage_a_output_dir(symbol: str, horizon: int) -> Path:
    """Return the output directory for Stage-A selector runs."""
    return ARTIFACTS_ROOT / "stage_a" / f"{symbol.upper()}_h{horizon}"


def stage_a_latest_weights_path(symbol: str, horizon: int) -> Path:
    """Return the path to the latest Stage-A weights for a symbol/horizon."""
    return stage_a_output_dir(symbol, horizon) / "family_weights_best.json"


def stage_b_study_db_path(horizon: int, label_id: str = "base") -> Path:
    """Return the path to a Stage-B Phase2 Optuna study DB."""
    return ARTIFACTS_ROOT / "optuna_studies" / f"phase2_v16_full_h{horizon}__label_{label_id}.db"


def stage_b_best_trial_path(study_name: str) -> Path:
    """Return the path to the Stage-B best trial JSON export."""
    return ARTIFACTS_ROOT / "optuna_studies" / f"{study_name}_best_trial.json"


def three_pillar_cache_path(cache_key: str) -> Path:
    """Return the path to a three-pillar cache pickle."""
    return ARTIFACTS_ROOT / "three_pillar_cache" / f"three_pillar_{cache_key}.pkl"


def group_map_path() -> Path:
    """Return the path to the Phase2 group map CSV."""
    return ARTIFACTS_ROOT / "meta_optimizer" / "phase2_symbol_group_map_gic_sector.csv"


# -----------------------------------------------------------------------------
# Legacy path resolution (backward compatibility)
# -----------------------------------------------------------------------------

def resolve_merged_parquet_legacy(symbol: str, horizon: int) -> Optional[Path]:
    """Find merged parquet in legacy locations (for migration)."""
    sym_upper = symbol.upper()
    
    # Check new location first
    new_path = merged_parquet_path(symbol, horizon)
    if new_path.exists():
        return new_path
    
    # Check legacy location
    legacy_path = LEGACY_FEATURE_PANEL_DIR / f"{sym_upper}_h{horizon}_merged.parquet"
    if legacy_path.exists():
        return legacy_path
    
    return None


def resolve_merged_meta_legacy(symbol: str, horizon: int) -> Optional[Path]:
    """Find merged meta/provenance in legacy locations (for migration)."""
    sym_upper = symbol.upper()
    
    # Check new location first
    new_path = merged_meta_path(symbol, horizon)
    if new_path.exists():
        return new_path
    
    # Check legacy JSON location
    legacy_json = LEGACY_FEATURE_PANEL_DIR / f"{sym_upper}_h{horizon}_merged.provenance.json"
    if legacy_json.exists():
        return legacy_json
    
    return None


def resolve_eodhd_cache_dir() -> Path:
    """Return the EODHD cache directory, checking legacy first."""
    if LEGACY_EODHD_CACHE_DIR.exists():
        return LEGACY_EODHD_CACHE_DIR
    return EODHD_CACHE_ROOT


def resolve_gdelt_cache_dir() -> Path:
    """Return the GDELT cache directory, checking legacy first."""
    if LEGACY_GDELT_CACHE_DIR.exists():
        return LEGACY_GDELT_CACHE_DIR
    return GDELT_CACHE_ROOT


def resolve_gdelt_global_cache_dir() -> Path:
    """Return the GDELT global cache directory, checking legacy first."""
    if LEGACY_GDELT_GLOBAL_CACHE_DIR.exists():
        return LEGACY_GDELT_GLOBAL_CACHE_DIR
    return GDELT_GLOBAL_CACHE_ROOT


def resolve_short_interest_cache_dir() -> Path:
    """Return the short interest cache directory, checking legacy first."""
    if LEGACY_SHORT_INTEREST_CACHE_DIR.exists():
        return LEGACY_SHORT_INTEREST_CACHE_DIR
    return SYMBOLS_CACHE_ROOT  # Per-symbol storage in new layout


# -----------------------------------------------------------------------------
# Directory creation helpers
# -----------------------------------------------------------------------------

def ensure_symbol_dirs(symbol: str, horizon: int) -> None:
    """Create all necessary directories for a symbol/horizon."""
    symbol_horizon_dir(symbol, horizon).mkdir(parents=True, exist_ok=True)
    symbol_families_dir(symbol, horizon).mkdir(parents=True, exist_ok=True)
    symbol_hf_dir(symbol).mkdir(parents=True, exist_ok=True)
    symbol_transcripts_dir(symbol).mkdir(parents=True, exist_ok=True)
    (symbol_cache_dir(symbol) / "short_interest").mkdir(parents=True, exist_ok=True)


def ensure_data_source_dirs() -> None:
    """Create all data source cache directories."""
    EODHD_FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    GDELT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    gdelt_global_merged_dir().mkdir(parents=True, exist_ok=True)
    FINBERT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    DOC_NOVELTY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SHARED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (SHARED_CACHE_ROOT / "macro").mkdir(parents=True, exist_ok=True)


def ensure_artifact_dirs() -> None:
    """Create all necessary artifact directories."""
    (ARTIFACTS_ROOT / "optuna_studies").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "optuna").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "stage_a").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "three_pillar_cache").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "meta_optimizer").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "backtests").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "prediction_tapes").mkdir(parents=True, exist_ok=True)


def ensure_universe_dirs() -> None:
    """Create universe cache directories."""
    UNIVERSE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (UNIVERSE_CACHE_ROOT / "two_tier").mkdir(parents=True, exist_ok=True)
