"""Stage B Mamba-only pipeline implementation.

This module assembles three feature tracks (A/B/C) and runs a Mamba-like
sequence model for Tier-2 predictions.

Tier-1 tree models (LightGBM, XGBoost, ElasticNet) have been removed.

Sequence Views:
- seq_raw: Raw Stage-A features with adaptive selection
- seq_hf: High-frequency feature blocks + Stage-B summaries  
- seq_hybrid: Combined seq_raw + seq_hf features
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from src.dcf_lab.utils.trading_calendar import add_sessions, session_on_or_after
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pa = None  # type: ignore
    pq = None  # type: ignore
    PYARROW_AVAILABLE = False

try:  # Optional gradient boosted tree dependencies
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dep
    lgb = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - optional dep
    xgb = None  # type: ignore

try:  # Optuna optional dependency
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

# 🚀 Ray Tune integration for distributed hyperparameter tuning
try:
    import ray
    import ray.train
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter
    RAY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    tune = None  # type: ignore
    OptunaSearch = None  # type: ignore
    ConcurrencyLimiter = None  # type: ignore
    RAY_AVAILABLE = False

from src.features.aggregator_panel import build_panel, _fetch_price_data  # type: ignore
from src.features.canonical_feature_cols import FAMILY_FORBIDDEN_COLS, FAMILY_REQUIRED_COLS, META_COL_SUFFIXES
from src.features.family_spec import default_hf_blocks
from src.stage_b.backtest import BacktestEngine
from src.stage_b.sequence_models import build_sequence_data, train_mamba_fold, predict_mamba_on_data
from src.stage_b.stage_b_export import collect_fold_predictions, save_prediction_tape

# Optuna optimizer (optional - lazy loaded when enabled)
try:
    from src.stage_b.optuna_optimizer import (
        OptunaConfig, OptimizedParams, StageBOptunaOptimizer,
        run_optuna_optimization, run_optuna_optimization_multi_symbol, apply_optimized_params,
    )
    OPTUNA_OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTUNA_OPTIMIZER_AVAILABLE = False

# Import centralized cache paths
from src.cache_paths import (
    CACHE_ROOT,
    LEGACY_FEATURE_PANEL_DIR,
    LEGACY_LOCAL_CACHE_DIR,
    ARTIFACTS_ROOT,
    merged_parquet_path,
    merged_meta_path,
    trackc_parquet_path,
    features_parquet_path,
    index_parquet_path,
    resolve_merged_parquet_legacy,
)

LOGGER = logging.getLogger("stage_b")
DEFAULT_RETURN_SCALE = 0.04
REPO_ROOT = Path(__file__).resolve().parents[2]
PREP_FAMILIES_SCRIPT = REPO_ROOT / "tools" / "prep_families.py"
DEFAULT_PREP_CACHE_ROOT = LEGACY_LOCAL_CACHE_DIR  # Use legacy path for now
DEFAULT_PREP_OUTPUT_DIR = ARTIFACTS_ROOT / "prep_families"
DEFAULT_STAGE_A_WF_START = "2010-07-02"
DEFAULT_STAGE_A_WF_END = "2025-07-01"
DEFAULT_STAGE_A_WF_TRAIN_YEARS = 8
DEFAULT_STAGE_B_STEP_DAYS = 63
FEATURE_PANEL_DIR = LEGACY_FEATURE_PANEL_DIR  # Use legacy path for backward compatibility
FEATURE_PANEL_TRACK = "trackc"
DEFAULT_PARQUET_ROW_GROUP_SIZE = 1000

# Fixed global universe for multi-symbol pooled sequence training.
# This is intentionally explicit (not inferred from CLI symbol list).
GLOBAL_OPTUNA_SYMBOLS: Tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "JPM",
    "XOM",
    "UNH",
    "COST",
    "AMD",
    "SPY",
)


# ---------------------------------------------------------------------------
# Specifications and configuration
# ---------------------------------------------------------------------------


STAGE_A_FAMILIES: Tuple[str, ...] = (
    "alternative_signals",
    "cboe_term",
    "correlation",
    "cross_asset",
    "index_constituents",
    "dcf",
    "dividends",
    "doc_embedding_novelty_hf",
    "earnings",
    "earnings_transcript_hf",
    "fin_g2",
    "fin_g3",
    "fin_g4",
    "fin_g5",
    "fin_g6",
    "fin_g7",
    "finbert",
    "garch_iv",
    "econ_events_calendar",
    "macro_tst_hf",
    "microstructure",
    "candle_mechanics",
    "ml_framework",
    "multiasset",
    "options",
    "options_anchoring",
    "peer_screener_context",
    "regime",
    "short_interest",
    "subsidiary",
    "tft_features",
)

STAGE_B_BASE_FAMILIES: Tuple[str, ...] = (
    "quantile_forecast",
    "calibration",
    "online_learning",
    "arima_forecast",
)

HF_BLOCK_FAMILIES: Tuple[str, ...] = tuple(default_hf_blocks())

TRACK_A_CORE_FAMILIES: Tuple[str, ...] = STAGE_A_FAMILIES + STAGE_B_BASE_FAMILIES

STAGE_B_FAMILIES: Tuple[str, ...] = STAGE_B_BASE_FAMILIES + HF_BLOCK_FAMILIES + (
    "hf_agg",
    "correlation",
)

DEFAULT_STAGE_A_TAGS: Dict[str, List[str]] = {
    "alternative_signals": ["alt", "flow"],
    "cboe_term": ["vol", "derivatives"],
    "correlation": ["regime", "risk"],
    "cross_asset": ["macro", "spread"],
    "index_constituents": ["fundamental", "index"],
    "dcf": ["fundamental"],
    "dividends": ["fundamental"],
    "doc_embedding_novelty_hf": ["nlp", "hf"],
    "earnings": ["fundamental"],
    "earnings_transcript_hf": ["nlp", "hf"],
    "fin_g2": ["fundamental"],
    "fin_g3": ["fundamental"],
    "fin_g4": ["fundamental"],
    "fin_g5": ["fundamental"],
    "fin_g6": ["fundamental"],
    "fin_g7": ["fundamental"],
    "finbert": ["nlp"],
    "garch_iv": ["vol"],
    "econ_events_calendar": ["macro", "calendar"],
    "macro_tst_hf": ["macro", "hf"],
    "microstructure": ["microstructure"],
    "ml_framework": ["ml"],
    "multiasset": ["exposure"],
    "options": ["derivatives"],
    "options_anchoring": ["derivatives", "regime"],
    "peer_screener_context": ["fundamental", "peers", "context"],
    "regime": ["regime"],
    "short_interest": ["positioning"],
    "subsidiary": ["fundamental"],
    "tft_features": ["sequence"],
}

STAGE_B_TAGS: Dict[str, List[str]] = {
    "quantile_forecast": ["probability", "short_horizon"],
    "calibration": ["probability", "reliability"],
    "online_learning": ["probability", "drift"],
    "arima_forecast": ["ts", "regime"],
    "correlation": ["regime", "risk"],
    "hf_agg": ["meta", "hf"],
    "tech_micro_hf": ["hf", "microstructure", "technical"],
    "forecast_hf": ["hf", "signals", "ensemble"],
    "vol_deriv_hf": ["hf", "vol", "derivatives"],
    "macro_regime_hf": ["hf", "macro", "regime"],
    "fundamental_val_hf": ["hf", "fundamental", "valuation"],
    "news_nlp_hf": ["hf", "nlp", "news"],
}


@dataclass
class FamilySpec:
    name: str
    stage: str  # "A" or "B"
    tags: List[str]
    columns: List[str] = field(default_factory=list)


@dataclass
class PrepFamiliesSettings:
    """Configuration for orchestrating tools/prep_families before Stage B runs."""

    enabled: bool = False
    wf_start: Optional[str] = None
    wf_end: Optional[str] = None
    wf_train_years: Optional[int] = None
    wf_step_years: Optional[int] = None
    wf_step_days: Optional[int] = None
    family_selector: str = "all"
    families: Optional[List[str]] = None
    mode: str = "stage-b"
    workers: int = 4
    hf_workers: Optional[int] = None
    strict: bool = False
    cache_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    log_level: str = "INFO"
    force: bool = False


def _default_prep_settings() -> PrepFamiliesSettings:
    return PrepFamiliesSettings(
        enabled=True,  # ✅ ENABLED: Use pre-generated windowed caches with lag features
        family_selector="resolved",
        wf_start=DEFAULT_STAGE_A_WF_START,
        wf_end=DEFAULT_STAGE_A_WF_END,
        wf_train_years=DEFAULT_STAGE_A_WF_TRAIN_YEARS,
        wf_step_days=DEFAULT_STAGE_B_STEP_DAYS,
    )


@dataclass
class StageBConfig:
    symbol: str
    horizons: List[int]
    horizon_clusters: Dict[str, List[int]]
    run_lstm_for_clusters: List[str]
    min_samples_for_lstm: int
    max_lstm_folds: int
    lstm_trials_per_cluster: int
    top_k_families_for_seq: int  # Legacy - use adaptive if 0
    performance_bar_for_lstm: float  # Legacy
    sequence_model_type: str = "mamba"  # Mamba-only Stage B
    seq_raw_features_per_family: int = 0  # 0 = adaptive M_f = floor(0.25 * features_per_family)
    seq_max_feature_dim: Optional[int] = None  # No hard cap - adaptive selection
    # Adaptive LSTM feature selection (Step 1-5)
    # 🔧 DISABLED by default; Track A/B feed full feature sets into Optuna
    lstm_adaptive_selection_enabled: bool = False
    lstm_adaptive_family_ratio: float = 0.75  # Used only when adaptive selection is explicitly enabled
    lstm_adaptive_feature_ratio: float = 0.65  # Used only when adaptive selection is explicitly enabled
    lstm_always_include_stage_b: bool = True  # Never drop Stage-B families
    window_cache_enabled: bool = True
    window_parallel_load: int = 40  # Workers for parallel cache loading (0=seq, 8=safe, 16=fast, 32=aggressive)
    start: Optional[str] = None
    end: Optional[str] = None
    stage_a_artifact_dir: Path = Path("artifacts/stage_a")
    stage_a_weights_filename: str = "family_weights_best.json"
    families: Optional[List[str]] = None
    include_stage_a: bool = True
    cache_dir: Optional[Path] = None
    tabular_folds: int = 5
    tabular_max_depth: int = 6
    tabular_learning_rate: float = 0.05
    tabular_max_iter: int = 800
    random_state: int = 17
    lstm_cluster_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    family_meta_window: int = 63
    prep: Optional[PrepFamiliesSettings] = field(default_factory=_default_prep_settings)

    # 🚀 Tier-1 (LightGBM/XGBoost/ElasticNet) has been removed
    # Mamba-like Tier-2 model is used
    
    # =======================================================================
    # MAMBA (SEQUENCE MODEL) DEFAULTS
    # =======================================================================
    mamba_seq_len: int = 128
    mamba_d_model: int = 128
    mamba_n_layers: int = 4
    mamba_ssm_dim: int = 96
    mamba_expand_factor: float = 2.0
    mamba_activation: str = "silu"
    mamba_norm_type: str = "rmsnorm"
    mamba_norm_strategy: str = "pre"
    mamba_dropout: float = 0.10
    mamba_resid_dropout: float = 0.0
    mamba_ssm_dropout: float = 0.0
    mamba_gate_dropout: float = 0.0
    mamba_optimizer: str = "adamw"
    mamba_learning_rate: float = 1e-3
    mamba_weight_decay: float = 1e-4
    mamba_grad_clip: float = 1.0
    mamba_lr_scheduler: str = "cosine"
    mamba_warmup_steps: int = 100
    mamba_max_epochs: int = 10
    mamba_batch_size: int = 32
    mamba_loss_fn: str = "smooth_l1"
    mamba_head_type: str = "linear"
    mamba_head_hidden_dim: int = 128
    mamba_head_num_layers: int = 1
    mamba_head_dropout: float = 0.0

    # =======================================================================
    # LEGACY LSTM CONFIG (kept for compatibility; not used in Mamba-only mode)
    # =======================================================================
    lstm_seq_len: int = 30  # legacy
    lstm_hidden_dim: int = 96  # 64-128 range
    lstm_layers: int = 2  # 1-2 layers
    lstm_dropout: float = 0.2  # 0.1-0.3 range
    lstm_learning_rate: float = 1e-3  # 0.0005-0.002 range
    lstm_weight_decay: float = 1e-4
    lstm_epochs: int = 15
    lstm_batch_size: int = 32
    lstm_early_stop_patience: int = 3
    lstm_warm_start: bool = True
    lstm_use_amp: bool = True
    lstm_min_samples: int = 1000  # 🔧 Reduced from 2000 to activate LSTM more readily
    # lstm_feature_cap removed - using adaptive selection via lstm_adaptive_feature_ratio
    lstm_scheduler: str = "cosine_decay"
    ray_lstm_parallel: bool = True
    ray_lstm_per_fold_gpu: int = 1
    ray_lstm_folds_parallel: int = 4  # 🚀 NEW: Number of folds to train in parallel (0=sequential)
    reliability_cal_weight: float = 1.35
    reliability_drift_weight: float = 1.05
    reliability_alpha_weight: float = 0.9
    reliability_bias: float = 0.0
    # 🔧 SIGNAL THRESHOLDS: Filter weak predictions for cleaner Sharpe
    # Only trade when |prediction| > threshold (avoids noisy signals)
    signal_threshold_upper: float = 0.02   # Go long if pred >= this
    signal_threshold_lower: float = -0.02  # Go short if pred <= this
    # If pred is between lower and upper, direction = 0 (neutral/no trade)
    
    # 🔧 VALIDATION WINDOW SIZE: Ensure val_frame >= seq_len for proper OOS validation
    # If val_frame < seq_len, LSTM falls back to in-sample validation (overfitting risk!)
    min_val_window_size: int = 300  # Minimum validation window size (>= max seq_len)
    
    # 🔧 COVERAGE CONSTRAINTS: Ensure sufficient signal coverage per fold
    # Percentile-based thresholds for adaptive signal generation
    signal_upper_percentile: float = 0.70  # Go long if pred > 70th percentile
    signal_lower_percentile: float = 0.30  # Go short if pred < 30th percentile
    min_coverage_per_fold: float = 0.25    # Minimum 25% of samples should have non-neutral signals
    
    # =======================================================================
    # 🚀 STEP 7: Regime-Aware Threshold Adjustment (Optuna-tuned)
    # =======================================================================
    # Base threshold T: |pred| < T_regime → do nothing
    # Regime multipliers adjust T based on market regime:
    # - T_bull = T * bull_mult (lower = more aggressive in uptrends)
    # - T_bear = T * bear_mult (higher = more conservative in downtrends)
    # - T_crisis = T * crisis_mult (much higher = very conservative in crises)
    regime_threshold_base: float = 0.10      # Base threshold T
    regime_threshold_bull_mult: float = 0.8  # T_bull = T * 0.8 (more aggressive)
    regime_threshold_bear_mult: float = 1.5  # T_bear = T * 1.5 (more conservative)
    regime_threshold_crisis_mult: float = 3.0  # T_crisis = T * 3.0 (very conservative)
    regime_aware_thresholds_enabled: bool = False  # Auto-enabled when Optuna provides params
    
    # =======================================================================
    # 🚀 OPTUNA OPTIMIZATION (4-Tier Feature Selection & Hyperparameter Tuning)
    # =======================================================================
    # When enabled, Optuna will run BEFORE LSTM to optimize:
    # - Tier 1: Family inclusion (which Stage-A families to use)
    # - Tier 2: Per-family dimensionality (PCA/AE latent dims)
    # - Tier 3: Track-A vs Track-B weighting
    # - Tier 4: LSTM hyperparameters
    optuna_enabled: bool = True  # Set False to disable Optuna pre-optimization
    optuna_n_trials: int = 150  # Number of Optuna trials
    optuna_timeout: int = 7200  # 2 hours timeout per horizon (for 150 trials)
    # Default to sequential Optuna execution for sequence models to avoid GPU/VRAM contention.
    # (Trial-level parallelism is handled explicitly via Ray Tune when enabled.)
    optuna_n_jobs: int = 1
    optuna_objective_sharpe_weight: float = 0.65
    optuna_objective_rwa_weight: float = 0.20
    optuna_objective_stability_weight: float = 0.15
    optuna_max_total_dims: int = 500  # Track C max dimensions (target ~450-500)
    optuna_track_a_dim_range: Tuple[int, int] = (80, 120)  # Track A dimension bounds
    optuna_track_b_dim_range: Tuple[int, int] = (15, 30)  # Track B dimension bounds
    optuna_track_weight_range: Tuple[float, float] = (0.5, 1.5)  # Track weight bounds
    optuna_params_cache_dir: Path = Path("artifacts/optuna")  # Cache for best params
    optuna_use_cached_params: bool = True  # Reuse cached params if available
    optuna_use_ray: bool = True
    optuna_use_ray_tune: bool = True  # Use Ray Tune for trial-level parallelism
    # Default to 1 concurrent trial to give Mamba full GPU/VRAM.
    optuna_ray_tune_concurrent_trials: int = 1
    optuna_ray_cpus_per_trial: int = 0  # 0 = auto-split CPUs across concurrent trials
    optuna_ray_max_concurrency: Optional[int] = None
    optuna_lstm_batch_sizes: Tuple[int, ...] = (32, 64, 128, 192, 256)
    optuna_lstm_amp_choices: Tuple[bool, ...] = (True, False)
    optuna_ray_fold_parallelism: int = 1  # 1 fold at a time for single GPU
    optuna_ray_fold_gpu_fraction: Optional[float] = 1.0  # Full GPU per fold (RTX 3070)
    optuna_fold_gpu_memory_gb: float = 8.0  # RTX 3070 has 8GB VRAM
    optuna_fold_gpu_reserve_gb: float = 1.0  # Keep 1GB reserved for system/CUDA

    # =======================================================================
    # 🚀 GLOBAL MULTI-SYMBOL OPTUNA MODE
    # =======================================================================
    # When enabled, Optuna trials train ONE shared sequence model per fold on a
    # pooled batch (concat on batch axis only) across this fixed symbol set.
    # Scoring uses equal-weight mean returns across symbols.
    optuna_global_multi_symbol: bool = True
    optuna_global_symbols: Tuple[str, ...] = GLOBAL_OPTUNA_SYMBOLS

    # Optional Optuna study pinning (Ray Tune / OptunaSearch)
    # If set, forces reuse of this exact study (SQLite DB) across runs.
    optuna_study_name: Optional[str] = None
    # When True, do not auto-create a new "*_reset_*" study on search-space mismatch.
    # Instead, fail fast so the user can explicitly manage study names.
    optuna_disable_study_auto_reset: bool = False
    
    strategy: Dict[str, Any] = field(
        default_factory=lambda: {
            "signal_rule": "hybrid",
            "k": 1.0,
            "leverage": 1.0,
            "max_exposure": 1.0,
            "fee_bp": 0.0,
            "slippage_bp": 0.0,
            "holding_period_days": 63,
            "overlap": True,
            "confidence_weighting": True,
        }
    )
    backtest_output_dir: Path = Path("artifacts/backtests")


@dataclass
class TrackData:
    name: str
    frame: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSummary:
    name: str
    model_type: str
    metrics: Dict[str, float]
    params: Dict[str, Any] = field(default_factory=dict)


# TierOneResult removed - LSTM-only mode (no Tier-1 tree models)


@dataclass
class TierTwoResult:
    track: str
    task: str
    best_model: Optional[ModelSummary]
    top_models: List[ModelSummary]
    predictions: Dict[str, pd.Series]
    composite_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCandidate:
    track: str
    tier: str
    model_name: str
    model_type: str
    task: str
    composite: float
    reliability: float
    metrics: Dict[str, float]
    predictions: pd.Series
    features_used: List[str]
    tags: List[str]
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageBResult:
    """Result from Stage B pipeline (LSTM-only mode)."""
    horizon: int
    tracks: Dict[str, TrackData]
    tier2: Dict[str, TierTwoResult]  # LSTM results only
    allow_lstm: bool
    lstm_features: Optional[Dict[str, pd.DataFrame]]
    candidates: List[ModelCandidate]
    primary_model: Optional[ModelCandidate]
    ensemble: Optional[Dict[str, Any]]
    outputs: Optional[pd.DataFrame]
    per_model_predictions: Dict[str, pd.Series]
    family_specs: Dict[str, FamilySpec]
    stage_a_priors: Dict[str, float]
    backtest_equity: Optional[pd.DataFrame] = None
    backtest_metrics: Optional[Dict[str, Any]] = None
    window_id: Optional[int] = None
    window_meta: Optional[Dict[str, Any]] = None
    window_history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _build_family_registry() -> Dict[str, FamilySpec]:
    registry: Dict[str, FamilySpec] = {}
    for fam in STAGE_A_FAMILIES:
        tags = DEFAULT_STAGE_A_TAGS.get(fam, ["base"])
        registry[fam] = FamilySpec(name=fam, stage="A", tags=tags)
    for fam, tags in STAGE_B_TAGS.items():
        registry[fam] = FamilySpec(name=fam, stage="B", tags=tags)
    return registry


FAMILIES = _build_family_registry()


# ---------------------------------------------------------------------------
# 🚀 Ray Tune helper functions for distributed hyperparameter tuning
# ---------------------------------------------------------------------------


def _calculate_optimal_ray_resources(
    gpu_fraction_per_trial: float = 0.0125,
    cpus_per_trial: int = 4,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Calculate optimal Ray Tune resources based on available hardware.
    
    Auto-detects available CPUs and GPUs, then calculates the maximum number
    of concurrent trials that can run without resource contention.
    
    Args:
        gpu_fraction_per_trial: GPU fraction per trial (0.05 = 20 parallel trials on 1 GPU)
        cpus_per_trial: Number of CPUs allocated per trial
        use_gpu: Whether the model uses GPU (XGBoost CUDA)
    
    Returns:
        Dict with 'max_concurrent', 'cpus_per_trial', 'gpu_per_trial', and 'resources_per_trial'
    """
    if not RAY_AVAILABLE or ray is None:
        return {
            "max_concurrent": 1,
            "cpus_per_trial": 1,
            "gpu_per_trial": 0.0,
            "resources_per_trial": {"cpu": 1},
        }
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Get available resources
    cluster_resources = ray.cluster_resources()
    total_cpus = int(cluster_resources.get("CPU", 1))
    total_gpus = int(cluster_resources.get("GPU", 0))
    
    # Calculate max concurrent based on CPU constraint
    max_by_cpu = total_cpus // cpus_per_trial if cpus_per_trial > 0 else total_cpus
    
    # Calculate max concurrent based on GPU constraint (if GPU is used)
    if use_gpu and total_gpus > 0 and gpu_fraction_per_trial > 0:
        max_by_gpu = int(total_gpus / gpu_fraction_per_trial)
        max_concurrent = min(max_by_cpu, max_by_gpu)
        gpu_per_trial = gpu_fraction_per_trial
    else:
        max_concurrent = max_by_cpu
        gpu_per_trial = 0.0
    
    # Ensure at least 1 trial can run
    max_concurrent = max(1, max_concurrent)
    
    # Build resources dict
    resources_per_trial = {"cpu": cpus_per_trial}
    if gpu_per_trial > 0:
        resources_per_trial["gpu"] = gpu_per_trial
    
    LOGGER.info(
        "🚀 Ray resources: %d CPUs, %d GPUs → %d max concurrent trials "
        "(%.2f GPU/trial, %d CPU/trial)",
        total_cpus, total_gpus, max_concurrent, gpu_per_trial, cpus_per_trial
    )
    
    return {
        "max_concurrent": max_concurrent,
        "cpus_per_trial": cpus_per_trial,
        "gpu_per_trial": gpu_per_trial,
        "resources_per_trial": resources_per_trial,
        "total_cpus": total_cpus,
        "total_gpus": total_gpus,
    }


def _create_ray_tree_trainable(
    X_ref: Any,  # ray.ObjectRef
    y_ref: Any,  # ray.ObjectRef
    actuals_ref: Any,  # ray.ObjectRef
    actual_returns_ref: Any,  # ray.ObjectRef
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    task: str,
    horizon: int,
    base_params: Dict[str, Any],
    lightgbm_available: bool,
    xgboost_available: bool,
):
    """Create a Ray Tune trainable function for tree model hyperparameter tuning.
    
    This function returns a trainable that:
    1. Receives data via Ray object store (X_ref, y_ref, etc.)
    2. Samples hyperparameters from the config dict (provided by OptunaSearch)
    3. Runs cross-validation over the provided folds
    4. Reports the mean composite score to Ray Tune
    
    Args:
        X_ref: Ray ObjectRef to feature DataFrame
        y_ref: Ray ObjectRef to target Series
        actuals_ref: Ray ObjectRef to actual values Series
        actual_returns_ref: Ray ObjectRef to actual returns Series
        folds: List of (train_idx, val_idx) tuples for CV
        model_name: 'lightgbm' or 'xgboost'
        task: 'regression' or 'classification'
        horizon: Prediction horizon
        base_params: Base model parameters to merge with tuned params
        lightgbm_available: Whether LightGBM is available
        xgboost_available: Whether XGBoost is available
    
    Returns:
        A trainable function compatible with tune.run()
    """
    import numpy as np
    
    def trainable(config: Dict[str, Any]) -> Dict[str, float]:
        """Trainable function executed by Ray Tune workers."""
        # Retrieve data from object store
        X = ray.get(X_ref)
        y = ray.get(y_ref)
        actuals = ray.get(actuals_ref)
        actual_returns = ray.get(actual_returns_ref)
        
        # Merge base params with tuned params
        params = base_params.copy()
        params.update(config)
        
        fold_scores: List[float] = []
        
        for train_idx, val_idx in folds:
            # Instantiate model
            model = _instantiate_tree_model(
                model_name, task, params,
                lightgbm_available, xgboost_available
            )
            if model is None:
                return {"mean_composite": float("nan")}
            
            # Train and predict
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            if task == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X.iloc[val_idx])
                    preds = preds[:, 1] if preds.ndim > 1 else preds
                else:
                    preds = model.predict(X.iloc[val_idx])
            else:
                preds = model.predict(X.iloc[val_idx])
            
            # Compute metrics
            metrics = _compute_cv_metrics(
                preds, actuals.iloc[val_idx], actual_returns.iloc[val_idx],
                task, horizon
            )
            fold_scores.append(metrics.get("composite", float("nan")))
        
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        return {"mean_composite": mean_score}
    
    return trainable


def _instantiate_tree_model(
    name: str,
    task: str,
    params: Dict[str, Any],
    lightgbm_available: bool,
    xgboost_available: bool,
):
    """Instantiate a tree model (LightGBM or XGBoost) for Ray Tune workers."""
    if name == "lightgbm":
        if not lightgbm_available:
            return None
        import lightgbm as lgb_local
        model_params = params.copy()
        n_estimators = model_params.pop("n_estimators", 600)
        model_params.setdefault("verbose", -1)
        model_params.setdefault("force_col_wise", True)
        model_params.setdefault("n_jobs", 1)  # 🚀 1 thread per model = max trial parallelism
        if task == "classification":
            model_params.setdefault("objective", "binary")
            model_params.setdefault("metric", "binary_logloss")
            return lgb_local.LGBMClassifier(n_estimators=n_estimators, **model_params)
        else:
            return lgb_local.LGBMRegressor(n_estimators=n_estimators, **model_params)
    
    elif name == "xgboost":
        if not xgboost_available:
            return None
        import xgboost as xgb_local
        model_params = params.copy()
        n_estimators = model_params.pop("n_estimators", 600)
        # Ensure GPU params are set for XGBoost CUDA
        model_params.setdefault("tree_method", "hist")
        model_params.setdefault("device", "cuda")
        if task == "classification":
            model_params.setdefault("objective", "binary:logistic")
            model_params.setdefault("eval_metric", "logloss")
            return xgb_local.XGBClassifier(n_estimators=n_estimators, **model_params)
        else:
            return xgb_local.XGBRegressor(n_estimators=n_estimators, **model_params)
    
    return None


def _compute_cv_metrics(
    preds: np.ndarray,
    actuals: pd.Series,
    actual_returns: pd.Series,
    task: str,
    horizon: int,
) -> Dict[str, float]:
    """Compute cross-validation metrics for Ray Tune workers.
    
    Uses unified scoring formula for all tracks (A/B/C):
        composite = 0.6 * Sharpe + 0.15 * RWA + 0.15 * (1 - ECE) + 0.10 * stability
    
    This ensures all tracks are directly comparable during HPO.
    """
    actuals_arr = np.asarray(actuals)
    returns_arr = np.asarray(actual_returns)
    preds_arr = np.asarray(preds)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(actuals_arr) | np.isnan(preds_arr))
    if valid_mask.sum() == 0:
        return {"composite": float("nan")}
    
    actuals_clean = actuals_arr[valid_mask]
    preds_clean = preds_arr[valid_mask]
    returns_clean = returns_arr[valid_mask] if len(returns_arr) == len(actuals_arr) else returns_arr[valid_mask]
    
    # --- Sharpe ratio ---
    # Strategy returns: sign(pred) * actual_return
    # 🔧 FIX: For horizons > 1, use non-overlapping samples to avoid autocorrelation inflation
    strategy_returns = np.sign(preds_clean) * returns_clean
    if strategy_returns.size > 0 and np.std(strategy_returns) > 1e-9:
        if horizon > 1 and len(strategy_returns) >= horizon:
            # Sample every H-th return to get independent observations
            non_overlapping = strategy_returns[::horizon]
            if len(non_overlapping) >= 2 and np.std(non_overlapping) > 1e-9:
                periods_per_year = max(1, 252 // horizon)
                scale = np.sqrt(periods_per_year)
                sharpe = float(np.mean(non_overlapping) / (np.std(non_overlapping) + 1e-9) * scale)
            else:
                sharpe = 0.0
        else:
            scale = np.sqrt(max(1, 252 // max(1, horizon)))
            sharpe = float(np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * scale)
    else:
        sharpe = 0.0
    
    # --- RWA (Return-Weighted Accuracy / Directional Accuracy) ---
    # 🔧 FIX: Also use non-overlapping for RWA when horizon > 1
    hits = (np.sign(preds_clean) == np.sign(returns_clean)).astype(float)
    if horizon > 1 and len(hits) >= horizon:
        non_overlapping_hits = hits[::horizon]
        rwa = float(np.mean(non_overlapping_hits)) if len(non_overlapping_hits) > 0 else 0.0
    else:
        rwa = float(np.mean(hits)) if hits.size > 0 else 0.0
    
    # --- ECE (Expected Calibration Error, normalized) ---
    mae = np.mean(np.abs(preds_clean - actuals_clean))
    denom = np.std(actuals_clean) + 1e-6
    ece = float(np.clip(mae / denom, 0.0, 1.0))
    
    # --- Stability (prediction consistency) ---
    if preds_clean.size > 0:
        stability = float(1.0 - np.clip(np.std(preds_clean) / (np.abs(np.mean(preds_clean)) + 1e-6), 0.0, 1.0))
    else:
        stability = 0.0
    
    # --- Unified Composite Score ---
    # score = 0.6 * Sharpe + 0.15 * RWA + 0.15 * (1 - ECE) + 0.10 * stability
    composite = 0.6 * sharpe + 0.15 * rwa + 0.15 * (1.0 - ece) + 0.10 * stability
    
    return {
        "sharpe": sharpe,
        "rwa": rwa,
        "ece": ece,
        "stability": stability,
        "composite": composite,
    }


def _get_optuna_search_space(model_name: str) -> Dict[str, Any]:
    """Get Optuna search space for Ray Tune's OptunaSearch.
    
    Returns a dict of parameter distributions compatible with Optuna.
    
    🔧 OPTIMIZED: Tightened ranges to avoid over-regularization that causes:
    - LightGBM: "No further splits with positive gain" warnings (3,177 in last run)
    - XGBoost: Negative/low composite scores (73 trials < 0.3 in last run)
    
    Key constraints for ~2000 samples with ~1000 features:
    - min_data_in_leaf ≤ 50: Ensures enough leaves possible (2000/50 = 40 leaves)
    - min_gain_to_split ≤ 0.3: High values block all splits
    - lambda_l1/l2 ≤ 2.0: Heavy regularization kills feature contributions
    - gamma ≤ 1.0: High gamma requires unrealistic loss reduction per split
    - n_estimators ≤ 800: Reduces wasted iterations when early stopping kicks in
    """
    if model_name == "lightgbm":
        return {
            "num_leaves": optuna.distributions.IntDistribution(16, 96),  # Was 128, cap for stability
            "max_depth": optuna.distributions.IntDistribution(3, 8),     # Was 10, avoid overfitting
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.10, log=True),
            "feature_fraction": optuna.distributions.FloatDistribution(0.5, 0.95),  # Cap at 0.95
            "bagging_fraction": optuna.distributions.FloatDistribution(0.6, 0.95),  # Min 0.6 for stability
            "bagging_freq": optuna.distributions.IntDistribution(1, 7),   # Was 10, reduce
            "min_data_in_leaf": optuna.distributions.IntDistribution(10, 50),  # 🔧 Was 100, cap at 50
            "lambda_l1": optuna.distributions.FloatDistribution(0.0, 2.0),     # 🔧 Was 5.0, cap at 2.0
            "lambda_l2": optuna.distributions.FloatDistribution(0.0, 2.0),     # 🔧 Was 5.0, cap at 2.0
            "min_gain_to_split": optuna.distributions.FloatDistribution(0.0, 0.3),  # 🔧 Was 1.0, cap at 0.3
            "n_estimators": optuna.distributions.IntDistribution(300, 800),    # 🔧 Was 1200, cap at 800
        }
    elif model_name == "xgboost":
        return {
            "max_depth": optuna.distributions.IntDistribution(3, 8),      # Was 10, avoid overfitting
            "eta": optuna.distributions.FloatDistribution(0.01, 0.10, log=True),
            "subsample": optuna.distributions.FloatDistribution(0.6, 0.95),    # Min 0.6 for stability
            "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 0.95),
            "min_child_weight": optuna.distributions.FloatDistribution(1.0, 5.0),  # 🔧 Was 10, cap at 5
            "reg_alpha": optuna.distributions.FloatDistribution(0.0, 2.0),     # 🔧 Was 5.0, cap at 2.0
            "reg_lambda": optuna.distributions.FloatDistribution(0.0, 2.0),    # 🔧 Was 5.0, cap at 2.0
            "gamma": optuna.distributions.FloatDistribution(0.0, 1.0),         # 🔧 Was 5.0, cap at 1.0
            "n_estimators": optuna.distributions.IntDistribution(300, 800),    # 🔧 Was 1200, cap at 800
        }
    return {}


# ---------------------------------------------------------------------------
# 🚀 Ray remote LSTM fold training for parallel execution
# ---------------------------------------------------------------------------


def _ray_train_lstm_fold_remote(
    sequences: np.ndarray,
    targets: np.ndarray,
    timestamps: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Dict[str, Any],
    warm_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ray remote function for training a single LSTM fold.
    
    This function is designed to be called via ray.remote() for parallel
    fold execution. Each worker trains an independent LSTM on its fold.
    
    Args:
        sequences: Numpy array of shape (n_samples, seq_len, n_features)
        targets: Numpy array of shape (n_samples,)
        timestamps: Numpy array of timestamp labels
        train_idx: Indices for training
        val_idx: Indices for validation
        cfg: Configuration dict (from vars(self.config))
        warm_state: Optional state dict for warm starting
    
    Returns:
        Dict with 'preds', 'timestamps', 'state_dict', 'val_loss', 'fold_id'
    """
    # Re-import inside ray worker (isolated environment)
    from src.stage_b.sequence_models import SequenceData, train_lstm_fold
    
    # Reconstruct SequenceData
    seq_data = SequenceData(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
    )
    
    # Train the fold
    result = train_lstm_fold(
        seq_data,
        train_idx,
        val_idx,
        cfg,
        warm_state=warm_state,
    )
    return result


def _run_lstm_folds_parallel_ray(
    seq_data: "SequenceData",
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    max_parallel: int = 4,
    use_warm_start: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Run LSTM folds in parallel using Ray.
    
    🚀 PERFORMANCE: With 4 folds and 4 parallel workers:
    - Sequential: 4 × ~30s = 120s
    - Parallel:   ~35s (each fold on separate GPU fraction)
    
    Args:
        seq_data: SequenceData object with sequences, targets, timestamps
        folds: List of (train_idx, val_idx) tuples
        cfg: Configuration dict
        max_parallel: Maximum parallel folds (default 4)
        use_warm_start: If True, chains folds sequentially (disables parallelism)
        logger: Optional logger
    
    Returns:
        List of result dicts from each fold
    """
    if not RAY_AVAILABLE or ray is None:
        if logger:
            logger.warning("⚠️ Ray not available, falling back to sequential fold training")
        return None  # Signal to use sequential fallback
    
    if use_warm_start:
        if logger:
            logger.info("📝 Warm start enabled - using sequential fold training")
        return None  # Warm start requires sequential
    
    if len(folds) < 2:
        return None  # Not worth parallelizing single fold
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # Calculate resources per fold
    cluster_resources = ray.cluster_resources()
    total_gpus = cluster_resources.get("GPU", 0)
    num_folds = min(len(folds), max_parallel)
    
    # Each fold gets a fraction of GPU
    gpu_per_fold = (total_gpus / num_folds) if total_gpus > 0 else 0.0
    
    if logger:
        logger.info(
            "🚀 Ray parallel LSTM: %d folds × %.2f GPU/fold = %.1f total GPU",
            num_folds, gpu_per_fold, gpu_per_fold * num_folds
        )
    
    # Create Ray remote function with resources
    @ray.remote(num_gpus=gpu_per_fold, num_cpus=1)
    def train_fold_task(
        sequences: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        cfg: Dict[str, Any],
        fold_id: int,
    ) -> Dict[str, Any]:
        """Remote task for training a single LSTM fold."""
        result = _ray_train_lstm_fold_remote(
            sequences, targets, timestamps, train_idx, val_idx, cfg, None
        )
        result["fold_id"] = fold_id
        return result
    
    # Put data in object store (shared across workers)
    sequences_ref = ray.put(seq_data.sequences)
    targets_ref = ray.put(seq_data.targets)
    timestamps_ref = ray.put(seq_data.timestamps)
    cfg_ref = ray.put(cfg)
    
    # Launch parallel tasks
    futures = []
    for fold_id, (train_idx, val_idx) in enumerate(folds[:max_parallel]):
        future = train_fold_task.remote(
            sequences_ref, targets_ref, timestamps_ref,
            train_idx, val_idx, cfg_ref, fold_id
        )
        futures.append(future)
    
    # Collect results
    try:
        results = ray.get(futures)
        if logger:
            logger.info("✅ Ray parallel LSTM completed: %d folds", len(results))
        return results
    except Exception as e:
        if logger:
            logger.warning("⚠️ Ray parallel LSTM failed: %s, falling back to sequential", e)
        return None


# ---------------------------------------------------------------------------
# Stage B pipeline implementation
# ---------------------------------------------------------------------------


class StageBPipeline:
    """Constructs feature tracks and executes Tier-1 models per horizon."""

    def __init__(self, config: StageBConfig):
        self.config = config
        self.logger = logging.getLogger(f"stage_b.{config.symbol.lower()}")

        if getattr(self.config, "sequence_model_type", "mamba") != "mamba":
            raise ValueError(
                f"StageBPipeline is Mamba-only; got sequence_model_type={self.config.sequence_model_type!r}"
            )

        self.family_specs = self._clone_family_specs()
        self._price_cache: Dict[int, pd.DataFrame] = {}
        self._prep_completed: Set[str] = set()
        self._forced_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        # 🔧 CRITICAL: Track current train indices to prevent data leakage
        # When set, block summaries are computed ONLY on train portion
        self._current_train_idx: Optional[np.ndarray] = None
        self._current_validation_idx: Optional[np.ndarray] = None
        self._current_test_idx: Optional[np.ndarray] = None
        # 🔧 NEW: Store window metadata for proper date-based train/val splits
        self._current_window_meta: Optional[Dict[str, Any]] = None
        # Unified per-horizon panels loaded from cache/features or built on demand
        self._unified_panels: Dict[int, pd.DataFrame] = {}
        self._panel_sources: Dict[int, Path] = {}
        self._window_coverage: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
        self._optuna_track_cache: Dict[int, List[str]] = {}
        self._index_position_cache: Dict[int, pd.DatetimeIndex] = {}
        # Global per-horizon seq_len cap derived from fold geometry (no per-window clamping)
        self._mamba_seq_len_cap: Dict[int, int] = {}
        # 🎯 Stage C: Collect LSTM predictions for prediction tape export
        self._prediction_tape_folds: Dict[int, List[pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[int, StageBResult]:
        results: Dict[int, StageBResult] = {}
        priors_map, metadata_map = self._load_stage_a_artifacts()

        for horizon in self.config.horizons:
            priors = priors_map.get(horizon, priors_map.get("global", {}))
            stage_a_meta = metadata_map.get(horizon, metadata_map.get("global", {}))

            families = self._resolve_families()
            cache_dir = self._resolve_cache_dir(horizon)
            self._ensure_feature_cache(horizon, cache_dir, families)
            base_panel = self._load_unified_panel(horizon, families, cache_dir)
            if base_panel is not None and horizon not in self._index_position_cache:
                self._index_position_cache[horizon] = pd.DatetimeIndex(base_panel.index)

            windows = self._load_walkforward_windows(horizon)
            if not windows:
                panel = self._build_panel(horizon)
                result = self._execute_panel(panel, horizon, priors, stage_a_meta)
                result.window_meta = {
                    "mode": "consolidated",
                    "train_start": panel.index.min(),
                    "train_end": panel.index.max(),
                }
                result.window_history = [self._summarize_window_result(result, result.window_meta)]
                results[horizon] = result
                continue

            # ----------------------------------------------------------------
            # 🌍 GLOBAL MULTI-SYMBOL WALK-FORWARD (POST-OPTUNA) MODE
            # ----------------------------------------------------------------
            # In this mode, after Optuna produces the best cached params (GLOBAL13),
            # we run the main walk-forward Mamba engine pooled across all symbols:
            # - one shared model per fold
            # - concatenate samples on batch axis only
            # - fold geometry (manifest windows) shared across symbols
            # - per-symbol prediction tapes written for Stage C
            if bool(getattr(self.config, "optuna_global_multi_symbol", False)):
                if self.config.prep is None or not self.config.prep.enabled:
                    raise ValueError("Global multi-symbol mode requires prep.enabled=True (manifest windows)")

                # Ensure Optuna is run (or cached) so best params are available.
                if base_panel is None or base_panel.empty:
                    base_panel = self._build_panel(horizon)
                column_families = self._infer_column_families(base_panel.columns)
                self._attach_columns_to_specs(column_families)
                block_summaries = self._compute_block_summaries(base_panel)
                labels = self._construct_labels(horizon, base_panel.index)
                self._run_optuna_optimization(
                    panel=base_panel,
                    column_families=column_families,
                    labels=labels,
                    block_summaries=block_summaries,
                    priors=priors,
                    horizon=horizon,
                )

                cache_path = self.config.optuna_params_cache_dir / f"GLOBAL13_h{horizon}_optuna.json"
                if not cache_path.exists():
                    raise ValueError(f"Global Optuna cache missing at {cache_path}")
                if not OPTUNA_OPTIMIZER_AVAILABLE:
                    raise ValueError("Global pooled Tier-2 requires optuna optimizer module available")

                from src.stage_b.optuna_optimizer import generate_prediction_tapes_multi_symbol_mamba

                optuna_params = StageBOptunaOptimizer().load_params(cache_path)
                trial_number = int(getattr(optuna_params, "trial_number", -1))
                best_score = float(getattr(optuna_params, "best_score", float("-inf")))
                # Ray Tune-backed OptunaSearch does not provide stable Optuna trial numbers in
                # the cached params payload (trial_number may be -1). Use best_score as the
                # authoritative signal that the cache is usable.
                if trial_number < 0 and not (np.isfinite(best_score) and best_score > 0):
                    raise ValueError(
                        "Global Optuna cache exists but no usable best result is recorded yet "
                        f"(trial_number={trial_number}, best_score={best_score})."
                    )
                if trial_number < 0:
                    self.logger.warning(
                        "Global Optuna cache has no Optuna trial_number (likely Ray Tune). "
                        "Proceeding with best_score=%.4f.",
                        best_score,
                    )

                self._apply_optuna_lstm_params(optuna_params)

                symbols = [str(s).upper() for s in (getattr(self.config, "optuna_global_symbols", GLOBAL_OPTUNA_SYMBOLS) or GLOBAL_OPTUNA_SYMBOLS)]
                self.logger.info(
                    "🌍 Running GLOBAL pooled Mamba walk-forward after Optuna: symbols=%d folds=%d",
                    len(symbols),
                    len(windows),
                )

                from dataclasses import replace as _replace
                pipelines: Dict[str, "StageBPipeline"] = {}
                panels_by_symbol: Dict[str, pd.DataFrame] = {}
                labels_by_symbol: Dict[str, pd.DataFrame] = {}
                block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}

                for sym in symbols:
                    pipelines[sym] = StageBPipeline(_replace(self.config, symbol=sym))
                    sym_panel = pipelines[sym]._build_panel(horizon)
                    sym_labels = pipelines[sym]._construct_labels(horizon, sym_panel.index)
                    valid_mask = sym_labels["forward_return"].notna()
                    sym_panel = sym_panel.loc[valid_mask]
                    sym_labels = sym_labels.loc[valid_mask]
                    panels_by_symbol[sym] = sym_panel
                    labels_by_symbol[sym] = sym_labels

                # Align to anchor symbol columns for consistent encoders across symbols.
                # IMPORTANT: Do NOT intersect columns across symbols. Intersection will drop
                # entire families (e.g. fin_g2..fin_g7) if a single symbol (like SPY) lacks
                # those columns, which incorrectly prevents Optuna from weighting them for
                # symbols that do have the data.
                anchor_cols = list(panels_by_symbol[symbols[0]].columns)

                for sym in symbols:
                    # Ensure all panels share the same feature space. Missing columns are
                    # filled with zeros (not NaNs) to keep downstream scalers/encoders stable.
                    panels_by_symbol[sym] = panels_by_symbol[sym].reindex(columns=anchor_cols, fill_value=0.0)
                    block_summaries_by_symbol[sym] = pipelines[sym]._compute_block_summaries(panels_by_symbol[sym])

                anchor_column_families = pipelines[symbols[0]]._infer_column_families(anchor_cols)

                # Build multi-symbol folds from manifest windows.
                walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
                for w in windows:
                    train_start = w.get("train_start")
                    train_end = w.get("train_end")
                    valid_start = w.get("valid_start")
                    test_end = w.get("test_end") or w.get("valid_end")
                    if train_start is None or train_end is None or valid_start is None or test_end is None:
                        continue
                    fold_entry: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    for sym in symbols:
                        idx = panels_by_symbol[sym].index
                        train_mask = (idx >= pd.Timestamp(train_start)) & (idx <= pd.Timestamp(train_end))
                        out_mask = (idx >= pd.Timestamp(valid_start)) & (idx <= pd.Timestamp(test_end))
                        fold_entry[sym] = (np.flatnonzero(train_mask), np.flatnonzero(out_mask))
                    walk_forward_folds_multi.append(fold_entry)

                tape_folds_by_symbol, meta = generate_prediction_tapes_multi_symbol_mamba(
                    panels_by_symbol=panels_by_symbol,
                    column_families=anchor_column_families,
                    labels_by_symbol=labels_by_symbol,
                    block_summaries_by_symbol=block_summaries_by_symbol,
                    walk_forward_folds_multi=walk_forward_folds_multi,
                    horizon=horizon,
                    params=optuna_params,
                    max_total_dims=int(getattr(self.config, "optuna_max_total_dims", 500)),
                )

                from src.stage_b.stage_b_export import save_prediction_tape

                written: Dict[str, Path] = {}
                for sym, fold_dfs in tape_folds_by_symbol.items():
                    if not fold_dfs:
                        self.logger.warning("No prediction folds generated for %s H%s", sym, horizon)
                        continue
                    out_path = save_prediction_tape(
                        fold_dfs=fold_dfs,
                        symbol=sym,
                        horizon=horizon,
                    )
                    written[sym] = out_path
                self.logger.info(
                    "🎯 Global pooled prediction tapes written: %d/%d symbols (portfolio_sharpe=%.4f)",
                    len(written),
                    len(symbols),
                    float(meta.get("portfolio_sharpe_sign_pred", 0.0)),
                )

                # Return a minimal result for the anchor symbol (CLI expects a StageBResult).
                results[horizon] = StageBResult(
                    horizon=horizon,
                    tracks={},
                    tier2={},
                    allow_lstm=True,
                    lstm_features=None,
                    candidates=[],
                    primary_model=None,
                    ensemble=None,
                    outputs=None,
                    per_model_predictions={},
                    family_specs=self.family_specs,
                    stage_a_priors=priors,
                    backtest_metrics={
                        "mode": "global_multi_symbol",
                        "n_symbols": int(len(symbols)),
                        "portfolio_sharpe_sign_pred": float(meta.get("portfolio_sharpe_sign_pred", 0.0)),
                        "mamba_seq_len": int(meta.get("mamba_seq_len", getattr(self.config, "mamba_seq_len", 0))),
                    },
                    window_meta={
                        "mode": "global_multi_symbol",
                        "symbol_count": int(len(symbols)),
                        "window_count": int(len(walk_forward_folds_multi)),
                    },
                )
                continue

            train_starts = [pd.Timestamp(w["train_start"]) for w in windows]
            test_ends = [
                pd.Timestamp(w.get("test_end") or w.get("valid_end"))
                for w in windows
            ]
            self._window_coverage[horizon] = (
                min(ts.tz_localize(None) if getattr(ts, "tz", None) else ts for ts in train_starts),
                max(ts.tz_localize(None) if getattr(ts, "tz", None) else ts for ts in test_ends),
            )

            self._ensure_optuna_track_ready(
                horizon=horizon,
                priors=priors,
                stage_a_meta=stage_a_meta,
                families=families,
                cache_dir=cache_dir,
                coverage_override=self._window_coverage[horizon],
            )

            # ----------------------------------------------------------------
            # Parallel cache loading: Pre-load all window panels concurrently
            # ----------------------------------------------------------------
            window_panels: Dict[str, Tuple[Dict[str, Any], pd.DataFrame]] = {}
            parallel_workers = self.config.window_parallel_load
            
            if parallel_workers > 0 and len(windows) > 1:
                self.logger.info(
                    "🚀 Parallel cache loading: %d windows with %d workers",
                    len(windows),
                    parallel_workers,
                )
                
                def load_window_panel(window: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[pd.DataFrame]]:
                    """Thread worker: Load panel for a single window."""
                    window_meta = dict(window)
                    window_id = window_meta.get("window_id", "unknown")
                    try:
                        panel = self._build_panel_for_window(horizon, window_meta)
                        return (window_id, window_meta, panel)
                    except Exception as e:
                        self.logger.warning(
                            "⚠️ Failed to load window %s: %s",
                            window_id,
                            str(e),
                        )
                        return (window_id, window_meta, None)
                
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    futures = {
                        executor.submit(load_window_panel, w): w.get("window_id", f"w{i}")
                        for i, w in enumerate(windows)
                    }
                    for future in as_completed(futures):
                        window_id, window_meta, panel = future.result()
                        if panel is not None and not panel.empty:
                            window_panels[window_id] = (window_meta, panel)
                        else:
                            self.logger.warning(
                                "Stage B window %s has no panel data for %s H%s (parallel load)",
                                window_id,
                                self.config.symbol,
                                horizon,
                            )
                
                self.logger.info(
                    "✅ Parallel cache loading complete: %d/%d panels loaded",
                    len(window_panels),
                    len(windows),
                )
            else:
                # Sequential fallback
                for window in windows:
                    window_meta = dict(window)
                    window_id = window_meta.get("window_id", "unknown")
                    panel = self._build_panel_for_window(horizon, window_meta)
                    if panel is not None and not panel.empty:
                        window_panels[window_id] = (window_meta, panel)
                    else:
                        self.logger.warning(
                            "Stage B window %s has no panel data for %s H%s",
                            window_id,
                            self.config.symbol,
                            horizon,
                        )
            
            # ----------------------------------------------------------------
            # Sequential model training: Process each panel in order
            # (Model training uses all cores, so sequential is optimal)
            # ----------------------------------------------------------------
            window_results: List[StageBResult] = []
            window_histories: List[Dict[str, Any]] = []

            # ----------------------------------------------------------------
            # ✅ Global seq_len cap for ALL folds (no per-window clamping)
            # ----------------------------------------------------------------
            # Compute once BEFORE training begins, matching Optuna's cap logic:
            #   cap = min(min_train_len - 20, min_oos_len - 5)
            #   cap = max(cap, 8)
            # where oos_len is the combined validation+test span for each window.
            precomputed_splits: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            min_train_len: Optional[int] = None
            # We treat "val_len" as the full out-of-sample span available after training.
            # In this pipeline, that's validation + test, because test predictions need
            # contiguous context for sequence construction.
            min_val_len: Optional[int] = None
            for window in windows:
                window_id = window.get("window_id", "unknown")
                if window_id not in window_panels:
                    continue
                window_meta, panel = window_panels[window_id]
                train_idx, val_idx, test_idx = self._window_split_indices(panel.index, window_meta, horizon)
                if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                    continue
                precomputed_splits[window_id] = (train_idx, val_idx, test_idx)
                t_len = int(len(train_idx))
                val_len = int(len(val_idx) + len(test_idx))
                min_train_len = t_len if min_train_len is None else min(min_train_len, t_len)
                min_val_len = val_len if min_val_len is None else min(min_val_len, val_len)

            if min_train_len is None or min_val_len is None:
                raise ValueError(
                    f"No usable walk-forward windows for {self.config.symbol} H{horizon}; cannot compute seq_len cap"
                )

            seq_cap = min(int(min_train_len) - 20, int(min_val_len) - 5)
            seq_cap = max(int(seq_cap), 8)
            self._mamba_seq_len_cap[horizon] = int(seq_cap)
            if int(getattr(self.config, "mamba_seq_len", 0)) > int(seq_cap):
                raise ValueError(
                    f"mamba_seq_len={int(self.config.mamba_seq_len)} exceeds global seq_len cap={int(seq_cap)} "
                    f"for {self.config.symbol} H{horizon} (min_train_len={min_train_len}, min_val_len={min_val_len}). "
                    "Fix by lowering mamba_seq_len or regenerating windows with longer train/val spans."
                )
            self.logger.info(
                "✅ Global seq_len cap (no clamping): cap=%d using min_train_len=%d min_val_len=%d",
                int(seq_cap),
                int(min_train_len),
                int(min_val_len),
            )
            
            # Process windows in original order for reproducibility
            for window in windows:
                window_id = window.get("window_id", "unknown")
                if window_id not in window_panels:
                    continue
                    
                window_meta, panel = window_panels[window_id]
                if window_id not in precomputed_splits:
                    train_idx, val_idx, test_idx = self._window_split_indices(panel.index, window_meta, horizon)
                else:
                    train_idx, val_idx, test_idx = precomputed_splits[window_id]
                window_meta["train_samples"] = int(len(train_idx))
                window_meta["valid_samples"] = int(len(val_idx))
                window_meta["test_samples"] = int(len(test_idx))
                if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                    self.logger.warning(
                        "Stage B window %s insufficient samples (train=%d, val=%d, test=%d) for %s H%s",
                        window_meta.get("window_id"),
                        len(train_idx),
                        len(val_idx),
                        len(test_idx),
                        self.config.symbol,
                        horizon,
                    )
                    continue
                self._forced_splits = [(train_idx, val_idx)]
                # 🔧 CRITICAL FIX: Pass train indices to prevent data leakage
                # Block summaries will be computed ONLY on train portion
                self._current_train_idx = train_idx
                self._current_validation_idx = val_idx
                self._current_test_idx = test_idx
                # 🔧 NEW: Store window metadata for date-based HPO splits
                self._current_window_meta = window_meta
                try:
                    result = self._execute_panel(panel, horizon, priors, stage_a_meta)
                    result = self._restrict_outputs_to_test_span(panel, result, test_idx)
                finally:
                    self._forced_splits = None
                    self._current_train_idx = None
                    self._current_validation_idx = None
                    self._current_test_idx = None
                    self._current_window_meta = None
                result.window_id = window_meta.get("window_id")
                result.window_meta = window_meta
                window_summary = self._summarize_window_result(result, window_meta)
                result.window_history = [window_summary]
                window_histories.append(window_summary)
                if result.outputs is not None and not result.outputs.empty:
                    result.outputs = result.outputs.assign(window_id=result.window_id)
                window_results.append(result)

            if not window_results:
                raise ValueError(
                    f"No usable walk-forward windows for {self.config.symbol} H{horizon}; "
                    "cannot run Stage B"
                )

            aggregated = self._aggregate_window_results(horizon, window_results)
            aggregated.window_meta = {
                "mode": "windowed",
                "window_count": len(window_results),
            }
            aggregated.window_history = window_histories
            results[horizon] = aggregated
            
            # 🎯 Stage C: Export prediction tape after all windows are processed
            if horizon in self._prediction_tape_folds and self._prediction_tape_folds[horizon]:
                try:
                    tape_path = save_prediction_tape(
                        fold_dfs=self._prediction_tape_folds[horizon],
                        symbol=self.config.symbol,
                        horizon=horizon,
                    )
                    self.logger.info(
                        "🎯 Prediction tape exported: %s (%d folds)",
                        tape_path,
                        len(self._prediction_tape_folds[horizon]),
                    )
                except Exception as exc:
                    self.logger.warning("Failed to save prediction tape: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Panel + artifacts
    # ------------------------------------------------------------------
    def _build_panel(self, horizon: int) -> pd.DataFrame:
        families = self._resolve_families()
        cache_dir = self._resolve_cache_dir(horizon)
        self._ensure_feature_cache(horizon, cache_dir, families)
        coverage_override = None
        if self.config.start and self.config.end:
            coverage_override = (
                pd.Timestamp(self.config.start),
                pd.Timestamp(self.config.end),
            )
        panel = self._load_unified_panel(
            horizon=horizon,
            families=families,
            cache_dir=cache_dir,
            coverage_override=coverage_override,
        )
        if panel is not None:
            return panel
        # Final fallback - rebuild directly for requested coverage
        fallback_panel = build_panel(
            symbol=self.config.symbol,
            start=self.config.start,
            end=self.config.end,
            families=families,
            cache_dir=cache_dir,
            horizon=horizon,
            stage="B",
        )
        return self._normalize_panel_frame(fallback_panel)

    def _panel_cache_candidates(
        self,
        horizon: int,
        track_label: str = FEATURE_PANEL_TRACK,
    ) -> List[Path]:
        token = (track_label or FEATURE_PANEL_TRACK).lower()
        candidates: List[Path] = []
        seen: Set[str] = set()
        for sym in (self.config.symbol.upper(), self.config.symbol.lower()):
            candidate = FEATURE_PANEL_DIR / f"{sym}_h{horizon}_{token}.parquet"
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _merged_panel_candidates(self, horizon: int) -> List[Path]:
        """Preferred unified panel artifact (single merged parquet per symbol).

        Produced by `tools/prep_families.py --write-merged yes` via Feast.
        """
        variant = os.getenv("STAGE_B_MERGED_PANEL_VARIANT", "auto").strip().lower()
        if variant in {"alpha"}:
            variant = "mamba"
        if variant in {"policy"}:
            variant = "portfolio"

        order: List[str]
        if variant == "mamba":
            order = ["merged_mamba", "merged"]
        elif variant == "portfolio":
            order = ["merged_portfolio", "merged"]
        elif variant == "merged":
            order = ["merged"]
        else:
            # auto: prefer mamba if available, then fall back to merged
            order = ["merged_mamba", "merged"]

        candidates: List[Path] = []
        seen: Set[str] = set()
        for sym in (self.config.symbol.upper(), self.config.symbol.lower()):
            for token in order:
                if token == "merged":
                    candidate = FEATURE_PANEL_DIR / f"{sym}_h{horizon}_merged.parquet"
                elif token == "merged_mamba":
                    candidate = FEATURE_PANEL_DIR / f"{sym}_h{horizon}_merged_mamba.parquet"
                elif token == "merged_portfolio":
                    candidate = FEATURE_PANEL_DIR / f"{sym}_h{horizon}_merged_portfolio.parquet"
                else:
                    continue
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _panel_variant_from_path(self, path: Path) -> str:
        name = str(path.name).lower()
        if "_merged_mamba" in name:
            return "mamba"
        if "_merged_portfolio" in name:
            return "portfolio"
        return "merged"

    def _split_panel_candidates(self, horizon: int) -> List[Tuple[Path, Path]]:
        """Production-grade split artifacts (numeric-only features + index sidecar).

        Supports both naming conventions:
        - <SYMBOL>_h<H>_features.parquet + <SYMBOL>_h<H>_index.parquet
        - <SYMBOL>_<H>_features.parquet + <SYMBOL>_<H>_index.parquet
        - <SYMBOL>_h<H>__features.parquet + <SYMBOL>_h<H>__index.parquet (Dagster/Phase2 variant)
        - <SYMBOL>_<H>__features.parquet + <SYMBOL>_<H>__index.parquet (Dagster/Phase2 variant)
        """
        candidates: List[Tuple[Path, Path]] = []
        seen: Set[str] = set()
        for sym in (self.config.symbol.upper(), self.config.symbol.lower()):
            for stem in (f"{sym}_h{horizon}", f"{sym}_{horizon}"):
                for infix in ("_", "__"):
                    feat = FEATURE_PANEL_DIR / f"{stem}{infix}features.parquet"
                    idx = FEATURE_PANEL_DIR / f"{stem}{infix}index.parquet"
                    key = f"{feat}||{idx}"
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append((feat, idx))
        return candidates

    def _peek_unified_panel_columns(self, panel_path: Path) -> List[str]:
        """Read parquet column names without loading full data when possible."""
        if PYARROW_AVAILABLE:
            try:
                pf = pq.ParquetFile(panel_path)
                return list(pf.schema_arrow.names)
            except Exception:
                # Fall back to pandas read below.
                pass
        try:
            frame = pd.read_parquet(panel_path)
            return list(frame.columns)
        except Exception:
            return []

    def _validate_cached_panel_schema(
        self,
        feature_cols: Sequence[str],
        families: Sequence[str],
        panel_variant: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Validate cached panel feature columns using contract enforcement.

        Policy: Accept-and-Warn + Contract Enforcement.
        - Fail only if required columns are missing or forbidden columns are present.
        - Allow extra columns (log drift for non-meta extras).
        """

        variant = str(panel_variant or "").strip().lower()
        if variant in {"mamba", "portfolio"}:
            return True, f"skip_contract:{variant}"

        requested = {str(f).lower() for f in (families or [])}
        contract_families = requested.intersection(FAMILY_REQUIRED_COLS.keys())
        if not contract_families:
            return True, "no contract families requested"

        col_list = [str(c) for c in feature_cols]
        column_families = self._infer_column_families(col_list)

        violations: List[str] = []
        drift_msgs: List[str] = []

        def _is_meta(col: str) -> bool:
            return any(str(col).endswith(suf) for suf in META_COL_SUFFIXES)

        for fam in sorted(contract_families):
            required = set(FAMILY_REQUIRED_COLS.get(fam, set()))
            forbidden = set(FAMILY_FORBIDDEN_COLS.get(fam, set()))
            present = {c for c in col_list if column_families.get(c) == fam}

            missing = sorted(required - present)
            forbidden_present = sorted(forbidden.intersection(present))
            if missing or forbidden_present:
                parts: List[str] = []
                if missing:
                    parts.append(f"missing_required={len(missing)}")
                if forbidden_present:
                    parts.append(f"forbidden_present={len(forbidden_present)}")
                violations.append(f"{fam}({', '.join(parts)})")
                continue

            extra = sorted(present - required)
            extra_non_meta = [c for c in extra if not _is_meta(c)]
            if extra_non_meta:
                drift_msgs.append(f"{fam}(extra_non_meta={len(extra_non_meta)})")

        if drift_msgs:
            self.logger.warning(
                "⚠️ Cached panel schema drift detected for %s H%s: %s (extras allowed)",
                self.config.symbol,
                getattr(self.config, "horizon", "?"),
                "; ".join(drift_msgs),
            )

        if violations:
            return False, "; ".join(violations)
        return True, "ok"

    def _delete_panel_artifact(self, panel_path: Path) -> None:
        try:
            panel_path.unlink(missing_ok=True)
        except Exception:
            pass
        # Legacy trackc artifacts have a sidecar meta json.
        meta_path = panel_path.with_suffix(".meta.json")
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _delete_split_panel_artifacts(self, features_path: Path, index_path: Path) -> None:
        for p in (features_path, index_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def _panel_artifact_ready_and_valid(self, horizon: int, families: Sequence[str]) -> bool:
        """Return True iff a cached unified panel exists and passes schema validation.

        Policy: Accept-and-Warn + Contract Enforcement.

        If a cached panel violates the contract, it is ignored.
        Automatic deletion is intentionally disabled (operator-initiated only).
        """
        prefer_split = os.environ.get("STAGE_B_PREFER_SPLIT_PANEL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        # Prefer Feast/Dagster merged per-symbol parquet by default.
        candidates = self._merged_panel_candidates(horizon) + self._panel_cache_candidates(horizon)
        for candidate in candidates:
            if not candidate.exists():
                continue
            cols = self._peek_unified_panel_columns(candidate)
            if not cols:
                self.logger.warning("⚠️ Cached panel exists but columns unreadable: %s", candidate)
                continue
            # Remove non-feature columns (Feast merged + legacy trackc)
            feature_cols = [
                c
                for c in cols
                if c not in {"event_timestamp", "date", "symbol", "split"}
            ]
            variant = self._panel_variant_from_path(candidate)
            ok, reason = self._validate_cached_panel_schema(feature_cols, families, panel_variant=variant)
            if not ok:
                self.logger.warning(
                    "⚠️ Cached panel contract violated for %s H%s (%s) → ignoring %s (no delete)",
                    self.config.symbol,
                    horizon,
                    reason,
                    candidate,
                )
                continue
            return True

        # Fallback: production-grade split artifacts (numeric-only features + index sidecar).
        # These can exist from older runs; use only if no merged/unified candidate is valid.
        for feat_path, idx_path in self._split_panel_candidates(horizon):
            if not (feat_path.exists() and idx_path.exists()):
                continue
            cols = self._peek_unified_panel_columns(feat_path)
            if not cols:
                self.logger.warning("⚠️ Cached split panel exists but columns unreadable: %s", feat_path)
                continue
            ok, reason = self._validate_cached_panel_schema([str(c) for c in cols], families, panel_variant=None)
            if not ok:
                self.logger.warning(
                    "⚠️ Cached split panel contract violated for %s H%s (%s) → ignoring %s and %s (no delete)",
                    self.config.symbol,
                    horizon,
                    reason,
                    feat_path,
                    idx_path,
                )
                continue
            if prefer_split:
                return True
            # If prefer_split is false, split is only a last resort and we've already exhausted
            # merged/unified candidates above; still consider this valid for readiness.
            return True
        return False

    def _normalize_panel_frame(self, frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return frame
        normalized = frame.copy()
        if not isinstance(normalized.index, pd.DatetimeIndex):
            normalized.index = pd.to_datetime(normalized.index)
        if getattr(normalized.index, "tz", None) is not None:
            normalized.index = normalized.index.tz_localize(None)
        normalized.index = normalized.index.astype("datetime64[ns]")
        normalized = normalized.sort_index().replace([np.inf, -np.inf], np.nan).ffill().bfill()
        float_cols = normalized.select_dtypes(include=[np.float64]).columns
        if len(float_cols) > 0:
            normalized[float_cols] = normalized[float_cols].astype(np.float32)
        return normalized

    def _slice_panel_by_range(
        self,
        panel: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        if panel is None or panel.empty:
            return panel
        idx = panel.index
        start = pd.Timestamp(start_ts).tz_localize(None) if getattr(start_ts, "tz", None) else pd.Timestamp(start_ts)
        end = pd.Timestamp(end_ts).tz_localize(None) if getattr(end_ts, "tz", None) else pd.Timestamp(end_ts)
        start_pos = int(np.clip(idx.searchsorted(start, side="left"), 0, len(idx)))
        end_pos = int(np.clip(idx.searchsorted(end, side="right"), 0, len(idx)))
        if start_pos >= end_pos:
            return panel.iloc[0:0]
        return panel.iloc[start_pos:end_pos]

    def _persist_unified_panel(self, horizon: int) -> Optional[Path]:
        panel = self._unified_panels.get(horizon)
        panel = self._normalize_panel_frame(panel)
        if panel is None or panel.empty:
            self.logger.debug(
                "_persist_unified_panel skipped: no data for %s H%s",
                self.config.symbol,
                horizon,
            )
            return None

        FEATURE_PANEL_DIR.mkdir(parents=True, exist_ok=True)
        panel_path = self._panel_cache_candidates(horizon)[0]
        save_df = panel.reset_index()
        if "index" in save_df.columns:
            save_df = save_df.rename(columns={"index": "date"})
        if PYARROW_AVAILABLE:
            table = pa.Table.from_pandas(save_df, preserve_index=False)
            pq.write_table(
                table,
                panel_path,
                row_group_size=DEFAULT_PARQUET_ROW_GROUP_SIZE,
                compression="snappy",
            )
        else:
            save_df.to_parquet(panel_path, index=False)

        coverage = self._window_coverage.get(horizon)
        if coverage is None:
            start_ts = panel.index.min()
            end_ts = panel.index.max()
        else:
            start_ts, end_ts = coverage
        start_ts = start_ts.tz_localize(None) if getattr(start_ts, "tz", None) else start_ts
        end_ts = end_ts.tz_localize(None) if getattr(end_ts, "tz", None) else end_ts

        meta = {
            "symbol": self.config.symbol,
            "horizon": horizon,
            "track": FEATURE_PANEL_TRACK,
            "columns": len(panel.columns),
            "coverage_start": start_ts.strftime("%Y-%m-%d"),
            "coverage_end": end_ts.strftime("%Y-%m-%d"),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        }
        with open(panel_path.with_suffix(".meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)

        self._panel_sources[horizon] = panel_path
        self.logger.info(
            "💾 Persisted unified panel for %s H%s → %s (%d rows × %d cols)",
            self.config.symbol,
            horizon,
            panel_path,
            len(panel),
            len(panel.columns),
        )
        return panel_path

    def _merge_optuna_track_c_into_unified_panel(
        self,
        horizon: int,
        track_c_frame: Optional[pd.DataFrame],
    ) -> None:
        if track_c_frame is None or track_c_frame.empty:
            return
        master = self._unified_panels.get(horizon)
        if master is None or master.empty:
            self.logger.warning(
                "Cannot merge Optuna Track C for %s H%s: unified panel not loaded",
                self.config.symbol,
                horizon,
            )
            return

        track_c_frame = track_c_frame.sort_index()
        overlap_index = master.index.intersection(track_c_frame.index)
        if overlap_index.empty:
            self.logger.warning(
                "Optuna Track C indices do not overlap unified panel for %s H%s",
                self.config.symbol,
                horizon,
            )
            return

        updates = track_c_frame.loc[overlap_index]
        for col in updates.columns:
            if col not in master.columns:
                master[col] = np.nan
            master.loc[overlap_index, col] = updates[col].values

        self._unified_panels[horizon] = master
        self._optuna_track_cache[horizon] = list(updates.columns)
        self.logger.info(
            "🧩 Merged Optuna Track C slice (%d rows × %d cols) into unified panel for %s H%s",
            len(updates),
            len(updates.columns),
            self.config.symbol,
            horizon,
        )
        self._persist_unified_panel(horizon)

    def _ensure_optuna_track_ready(
        self,
        horizon: int,
        priors: Dict[str, float],
        stage_a_meta: Dict[str, Any],
        families: List[str],
        cache_dir: Optional[Path],
        coverage_override: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> None:
        if not (self.config.optuna_enabled and OPTUNA_OPTIMIZER_AVAILABLE):
            return
        if horizon in self._optuna_track_cache:
            return

        panel = self._load_unified_panel(
            horizon=horizon,
            families=families,
            cache_dir=cache_dir,
            coverage_override=coverage_override,
        )
        if panel is None or panel.empty:
            return

        column_families = self._infer_column_families(panel.columns)
        self._attach_columns_to_specs(column_families)
        block_summaries = self._compute_block_summaries(panel)
        labels = self._construct_labels(horizon, panel.index)

        optuna_params = self._run_optuna_optimization(
            panel=panel,
            column_families=column_families,
            labels=labels,
            block_summaries=block_summaries,
            priors=priors,
            horizon=horizon,
        )
        if optuna_params is None:
            return

        self._apply_optuna_lstm_params(optuna_params)
        track_c_frame, _ = self._build_optuna_track_c(
            panel=panel,
            column_families=column_families,
            block_summaries=block_summaries,
            optuna_params=optuna_params,
        )
        if track_c_frame is not None:
            self._merge_optuna_track_c_into_unified_panel(horizon, track_c_frame)

    def _load_unified_panel(
        self,
        horizon: int,
        families: List[str],
        cache_dir: Optional[Path],
        coverage_override: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> Optional[pd.DataFrame]:
        if horizon in self._unified_panels:
            return self._unified_panels[horizon]

        FEATURE_PANEL_DIR.mkdir(parents=True, exist_ok=True)

        prefer_split = os.environ.get("STAGE_B_PREFER_SPLIT_PANEL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        # Prefer Feast/Dagster merged per-symbol parquet (single artifact) by default.
        for candidate in self._merged_panel_candidates(horizon):
            if candidate.exists():
                # Validate schema before trusting this cached artifact.
                cols = self._peek_unified_panel_columns(candidate)
                feature_cols = [
                    c
                    for c in cols
                    if c not in {"event_timestamp", "date", "symbol", "split"}
                ]
                variant = self._panel_variant_from_path(candidate)
                ok, reason = self._validate_cached_panel_schema(feature_cols, families, panel_variant=variant)
                if not ok:
                    self.logger.warning(
                        "⚠️ Merged cached panel contract violated for %s H%s (%s) → ignoring %s (no delete)",
                        self.config.symbol,
                        horizon,
                        reason,
                        candidate,
                    )
                    continue
                panel = self._read_unified_panel(candidate)
                self._unified_panels[horizon] = panel
                self._panel_sources[horizon] = candidate
                self._index_position_cache[horizon] = pd.DatetimeIndex(panel.index)
                return panel

        # Prefer split (numeric-only) artifacts only if explicitly requested or as a fallback.
        for feat_path, idx_path in self._split_panel_candidates(horizon):
            if not (feat_path.exists() and idx_path.exists()):
                continue
            cols = self._peek_unified_panel_columns(feat_path)
            if cols:
                ok, reason = self._validate_cached_panel_schema([str(c) for c in cols], families, panel_variant=None)
                if not ok:
                    self.logger.warning(
                        "⚠️ Split cached panel contract violated for %s H%s (%s) → ignoring %s and %s (no delete)",
                        self.config.symbol,
                        horizon,
                        reason,
                        feat_path,
                        idx_path,
                    )
                    continue
            try:
                panel = self._read_split_panel(feat_path, idx_path)
            except Exception as exc:
                self.logger.warning("⚠️ Failed reading split panel %s: %s", feat_path, exc)
                continue
            if not prefer_split and any(p.exists() for p in self._merged_panel_candidates(horizon)):
                self.logger.info(
                    "ℹ️ Using split panel fallback for %s H%s even though a merged panel exists. "
                    "Set STAGE_B_PREFER_SPLIT_PANEL=1 to make split-first.",
                    self.config.symbol,
                    horizon,
                )
            self._unified_panels[horizon] = panel
            self._panel_sources[horizon] = feat_path
            self._index_position_cache[horizon] = pd.DatetimeIndex(panel.index)
            return panel

        for candidate in self._panel_cache_candidates(horizon):
            if candidate.exists():
                panel = self._read_unified_panel(candidate)
                self._unified_panels[horizon] = panel
                self._panel_sources[horizon] = candidate
                self._index_position_cache[horizon] = pd.DatetimeIndex(panel.index)

                # Validate that the unified panel actually contains the expected families.
                # This is a common failure mode when prep_families ran under quota limits or
                # a different family selector, leading to silent per-window fallbacks later.
                try:
                    expected_families = set(families or [])
                    if expected_families:
                        column_families = self._infer_column_families(panel.columns)
                        present_families = set(column_families.values())
                        missing_families = sorted(expected_families - present_families)
                        unmapped_cols = [c for c in panel.columns if c not in column_families]
                        if missing_families:
                            self.logger.warning(
                                "⚠️ Unified panel for %s H%s is missing %d/%d expected families (e.g. %s). "
                                "This may cause weaker models or trigger expensive live fallbacks if coverage is thin.",
                                self.config.symbol,
                                horizon,
                                len(missing_families),
                                len(expected_families),
                                ", ".join(missing_families[:6]) + ("..." if len(missing_families) > 6 else ""),
                            )
                        if unmapped_cols:
                            self.logger.warning(
                                "⚠️ Unified panel for %s H%s has %d columns that do not map to known families (e.g. %s).",
                                self.config.symbol,
                                horizon,
                                len(unmapped_cols),
                                ", ".join([str(c) for c in unmapped_cols[:6]]) + ("..." if len(unmapped_cols) > 6 else ""),
                            )
                        require_all = os.environ.get("STAGE_B_REQUIRE_ALL_FAMILIES", "0").strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                        }
                        if require_all and (missing_families or unmapped_cols):
                            raise ValueError(
                                f"Unified panel validation failed for {self.config.symbol} H{horizon}: "
                                f"missing_families={len(missing_families)}, unmapped_cols={len(unmapped_cols)}. "
                                "Re-run tools/prep_families.py (or set prep.force=True) to regenerate caches."
                            )
                except Exception as exc:
                    self.logger.debug("Unified panel validation skipped/failed for %s H%s: %s", self.config.symbol, horizon, exc)
                self.logger.info(
                    "✅ Loaded unified feature panel for %s H%s from %s (%d rows × %d cols)",
                    self.config.symbol,
                    horizon,
                    candidate,
                    len(panel),
                    len(panel.columns),
                )
                return panel

        coverage = coverage_override or self._window_coverage.get(horizon)
        prep_cfg = self.config.prep
        prep_enabled = bool(prep_cfg and prep_cfg.enabled)

        if prep_enabled:
            if coverage is not None:
                start_ts, end_ts = coverage
                start_ts = start_ts.tz_localize(None) if getattr(start_ts, "tz", None) else start_ts
                end_ts = end_ts.tz_localize(None) if getattr(end_ts, "tz", None) else end_ts
                start_str = start_ts.strftime("%Y-%m-%d")
                end_str = end_ts.strftime("%Y-%m-%d")
                raise FileNotFoundError(
                    "Unified feature panel missing for %s H%s. Expected prep_families to generate %s (coverage %s→%s)."
                    % (self.config.symbol, horizon, self._panel_cache_candidates(horizon)[0], start_str, end_str)
                )
            raise FileNotFoundError(
                "Unified feature panel missing for %s H%s and coverage unknown. Run tools/prep_families.py before Stage B."
                % (self.config.symbol, horizon)
            )

        # Prep disabled → build panel on the fly (development fallback)
        if coverage is None:
            start_ts = pd.Timestamp(self.config.start) if self.config.start else pd.Timestamp("1990-01-01")
            end_ts = pd.Timestamp(self.config.end) if self.config.end else pd.Timestamp.now().normalize()
        else:
            start_ts, end_ts = coverage
            start_ts = start_ts.tz_localize(None) if getattr(start_ts, "tz", None) else start_ts
            end_ts = end_ts.tz_localize(None) if getattr(end_ts, "tz", None) else end_ts

        start_str = start_ts.strftime("%Y-%m-%d")
        end_str = end_ts.strftime("%Y-%m-%d")
        self.logger.warning(
            "⚠️ Prep disabled; building unified panel for %s H%s live (%s → %s)",
            self.config.symbol,
            horizon,
            start_str,
            end_str,
        )
        panel = build_panel(
            symbol=self.config.symbol,
            start=start_str,
            end=end_str,
            families=families,
            cache_dir=cache_dir,
            horizon=horizon,
            stage="B",
        )
        if panel is None or panel.empty:
            raise ValueError(
                f"❌ Failed to build unified panel for {self.config.symbol} H{horizon} with prep disabled"
            )
        panel = self._normalize_panel_frame(panel)
        self._unified_panels[horizon] = panel
        self._index_position_cache[horizon] = pd.DatetimeIndex(panel.index)
        return panel

    def _read_unified_panel(self, panel_path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(panel_path)
        # Support both legacy TrackC (`date`) and Feast merged (`event_timestamp`).
        ts_col: Optional[str] = None
        if "event_timestamp" in frame.columns:
            ts_col = "event_timestamp"
        elif "date" in frame.columns:
            ts_col = "date"

        if ts_col is not None:
            dates = pd.to_datetime(frame[ts_col])
            if getattr(dates.dt, "tz", None) is not None:
                dates = dates.dt.tz_localize(None)
            frame = frame.drop(columns=[ts_col])
            frame.index = dates
        else:
            frame.index = pd.to_datetime(frame.index)
            if getattr(frame.index, "tz", None) is not None:
                frame.index = frame.index.tz_localize(None)

        # Feast merged artifacts include string columns that should not be model inputs.
        for col in ("symbol", "split"):
            if col in frame.columns:
                frame = frame.drop(columns=[col])
        return self._normalize_panel_frame(frame)

    def _read_split_panel(self, features_path: Path, index_path: Path) -> pd.DataFrame:
        """Read split artifacts and reconstruct a DateTimeIndex in-memory."""
        idx = pd.read_parquet(index_path)
        if idx is None or idx.empty:
            raise ValueError(f"Index parquet empty: {index_path}")
        if "row_id" not in idx.columns:
            raise ValueError(f"Index parquet missing row_id: {index_path}")
        idx = idx.sort_values("row_id").reset_index(drop=True)

        feat = pd.read_parquet(features_path)
        if feat is None or feat.empty:
            raise ValueError(f"Features parquet empty: {features_path}")
        if len(feat) != len(idx):
            raise ValueError(
                f"Split panel row mismatch for {features_path.name}: features_rows={len(feat)} index_rows={len(idx)}"
            )

        if "date" not in idx.columns:
            raise ValueError(f"Index parquet missing date: {index_path}")
        dates = pd.to_datetime(idx["date"].astype(str), format="%Y%m%d", errors="coerce")
        if dates.isna().any():
            raise ValueError(f"Index parquet has invalid dates: {index_path}")
        dates = dates.dt.tz_localize(None)
        feat.index = pd.DatetimeIndex(dates)

        # Enforce numeric-only model inputs.
        feat = feat.select_dtypes(include=["number", "bool"]).astype(float)
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self._normalize_panel_frame(feat)


    def _build_panel_for_window(self, horizon: int, window: Dict[str, Any]) -> pd.DataFrame:
        families = self._resolve_families()
        cache_dir = self._resolve_cache_dir(horizon)
        self._ensure_feature_cache(horizon, cache_dir, families)
        train_start = pd.Timestamp(window["train_start"])
        panel_end = pd.Timestamp(window.get("test_end") or window["valid_end"])
        start_str = train_start.strftime("%Y-%m-%d")
        end_str = panel_end.strftime("%Y-%m-%d")
        window_idx = window.get("cache_window_idx")
        if window_idx is None:
            window_idx = window.get("cache_idx")

        train_start = train_start.tz_localize(None) if getattr(train_start, "tz", None) else train_start
        panel_end = panel_end.tz_localize(None) if getattr(panel_end, "tz", None) else panel_end

        unified_panel = self._load_unified_panel(horizon, families, cache_dir)
        if unified_panel is not None:
            window_slice = self._slice_panel_by_range(unified_panel, train_start, panel_end)
            if not window_slice.empty:
                self.logger.debug(
                    "Window %s slice: %d rows from consolidated panel (%s → %s)",
                    window_idx,
                    len(window_slice),
                    window_slice.index.min().date(),
                    window_slice.index.max().date(),
                )
                return window_slice.copy()
            self.logger.warning(
                "⚠️ Unified panel %s lacks coverage for window %s (%s → %s); falling back to live build",
                str(self._panel_sources.get(horizon, Path("<unknown>"))),
                window_idx,
                start_str,
                end_str,
            )

            # Optional hard-stop to prevent per-window live builds that can explode API calls.
            # Use this when you want Stage B to be strictly cache-driven.
            prep_cfg = self.config.prep
            if prep_cfg is not None and prep_cfg.enabled:
                no_live = os.environ.get("STAGE_B_NO_LIVE_FALLBACK", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
                if no_live:
                    raise FileNotFoundError(
                        f"Unified panel lacks coverage for window {window_idx} ({start_str}→{end_str}) for {self.config.symbol} H{horizon}. "
                        "Live per-window build is disabled via STAGE_B_NO_LIVE_FALLBACK=1; rerun tools/prep_families.py "
                        "with sufficient coverage/lookback padding to regenerate TrackC panel."
                    )

        self.logger.info(
            "🔍 Stage B window %s: cache_dir=%s, window_idx=%s, families=%d",
            window.get("window_id"),
            cache_dir,
            window_idx,
            len(families),
        )
        panel = build_panel(
            symbol=self.config.symbol,
            start=start_str,
            end=end_str,
            families=families,
            cache_dir=cache_dir,
            horizon=horizon,
            stage="B",
            window_idx=window_idx,
        )
        
        # CRITICAL VALIDATION: Check panel before proceeding
        if panel is None or panel.empty:
            raise ValueError(f"❌ Panel is empty for window {window_idx} - cannot proceed with training")
        
        if len(panel.columns) == 0:
            raise ValueError(f"❌ Panel has no columns for window {window_idx} - cannot proceed with training")
        
        non_null_cols = panel.notna().any(axis=0).sum()
        if non_null_cols == 0:
            raise ValueError(f"❌ All panel columns are null for window {window_idx} - cannot proceed with training")
        
        # 🔧 NEW: Validate lag features are present
        lag_cols = [c for c in panel.columns if '_lag' in c.lower()]
        self.logger.info(
            "✅ Panel validation passed: %d rows × %d cols (%d non-null, %d lag features) for window %s",
            len(panel),
            len(panel.columns),
            non_null_cols,
            len(lag_cols),
            window_idx,
        )
        
        # 🔧 NEW: Validate family data by checking for family-specific columns
        families = self._resolve_families()
        missing_families = []
        for fam in families:
            fam_cols = [c for c in panel.columns if c.startswith(fam) or c.startswith(f"{fam}_")]
            if len(fam_cols) == 0:
                missing_families.append(fam)
        
        if missing_families:
            self.logger.warning(
                "⚠️ Families with no columns in panel for window %s: %s",
                window_idx,
                ", ".join(missing_families[:10]) + (f" (+{len(missing_families)-10} more)" if len(missing_families) > 10 else ""),
            )
        
        return self._normalize_panel_frame(panel)

    def _discover_windows_from_cache(self, horizon: int, cache_dir: Path) -> List[Dict[str, Any]]:
        """
        🔧 AUTO-DISCOVER windows from cache file metadata.
        
        Scans cache directory for windowed parquet files and extracts window
        definitions from their .meta.json files. This ensures Stage B always
        uses windows that match the actual cached data.
        """
        self.logger.debug(
            "Per-window cache discovery is disabled; relying on completeness manifests. (%s H%s)",
            self.config.symbol,
            horizon,
        )
        return []

    def _update_manifest_from_cache(
        self,
        horizon: int,
        windows: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """
        🔧 AUTO-UPDATE manifest to reflect discovered cache windows.
        
        This ensures the manifest always matches the actual cached data,
        preventing cache misses due to manifest/cache mismatch.
        """
        # Deprecated: per-window cache shards are no longer a supported input.
        return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{self.config.symbol.lower()}_h{horizon}_completeness.json"
        
        # Build manifest payload
        payload = {
            "symbol": self.config.symbol.upper(),
            "horizon": horizon,
            "mode": "stage-b",
            "stage": "stage-b",
            "auto_discovered": True,
            "discovered_at": pd.Timestamp.utcnow().isoformat(),
            "window_count": len(windows),
            "windows": [
                {
                    "window_id": w["window_id"],
                    "cache_window_idx": w["cache_window_idx"],
                    "train_start": w["train_start"].isoformat() if hasattr(w["train_start"], "isoformat") else str(w["train_start"]),
                    "train_end": w["train_end"].isoformat() if hasattr(w["train_end"], "isoformat") else str(w["train_end"]),
                    "valid_start": w["valid_start"].isoformat() if hasattr(w["valid_start"], "isoformat") else str(w["valid_start"]),
                    "valid_end": w["valid_end"].isoformat() if hasattr(w["valid_end"], "isoformat") else str(w["valid_end"]),
                    "test_start": (
                        w.get("test_start").isoformat()
                        if hasattr(w.get("test_start"), "isoformat")
                        else (str(w.get("test_start")) if w.get("test_start") is not None else None)
                    ),
                    "test_end": (
                        w.get("test_end").isoformat()
                        if hasattr(w.get("test_end"), "isoformat")
                        else (str(w.get("test_end")) if w.get("test_end") is not None else None)
                    ),
                }
                for w in windows
            ],
        }
        
        try:
            with open(manifest_path, "w") as f:
                json.dump(payload, f, indent=2)
            self.logger.info(
                "✅ Auto-updated manifest at %s with %d windows from cache",
                manifest_path,
                len(windows),
            )
        except Exception as e:
            self.logger.warning("Failed to update manifest %s: %s", manifest_path, e)

    def _load_walkforward_windows(self, horizon: int) -> List[Dict[str, Any]]:
        prep_cfg = self.config.prep
        self.logger.info(
            "🔍 _load_walkforward_windows called: prep_cfg=%s, enabled=%s",
            "None" if prep_cfg is None else "exists",
            prep_cfg.enabled if prep_cfg else False,
        )
        if prep_cfg is None or not prep_cfg.enabled:
            self.logger.warning("⚠️ Prep disabled or not configured - will use consolidated panel mode")
            return []
        
        cache_dir = self._resolve_cache_dir(horizon)
        output_dir = Path(prep_cfg.output_dir).expanduser() if prep_cfg.output_dir else DEFAULT_PREP_OUTPUT_DIR
        manifest = output_dir / f"{self.config.symbol.lower()}_h{horizon}_completeness.json"
        
        # 🔧 OPTIMIZED: Trust manifest first (fast path), only scan cache if manifest missing
        # Manifest is auto-updated after cache discovery, so it should be reliable
        self.logger.info("🔍 Looking for manifest at: %s", manifest)
        if manifest.exists():
            try:
                with open(manifest, "r") as handle:
                    payload = json.load(handle)
                
                stage_token = str(payload.get("stage") or payload.get("mode") or "stage-b").lower()
                if stage_token == "stage-a":
                    self.logger.info(
                        "Completeness manifest %s recorded for Stage A; Stage B will rebuild consolidated panel",
                        manifest,
                    )
                    return []
                if stage_token not in {"stage-b", "walkforward"}:
                    self.logger.warning(
                        "Unknown completeness manifest stage '%s' at %s; ignoring window metadata",
                        stage_token,
                        manifest,
                    )
                    return []
                
                windows_raw = payload.get("windows") or []
                if windows_raw:
                    resolved = self._parse_manifest_windows(windows_raw, horizon)
                    if resolved:
                        resolved = self._filter_windows_by_date(resolved)
                        self.logger.info(
                            "✅ Loaded %d walk-forward windows for %s H%s from manifest (fast path)",
                            len(resolved),
                            self.config.symbol,
                            horizon,
                        )
                        return resolved
            except Exception as exc:
                self.logger.warning(
                    "Failed to parse completeness manifest %s for %s H%s: %s",
                    manifest,
                    self.config.symbol,
                    horizon,
                    exc,
                )
        
        # Per-window cache scanning is intentionally disabled; prep_families writes
        # a completeness manifest with window metadata.
        if cache_dir and cache_dir.exists():
            self.logger.info(
                "📂 Manifest missing/empty; skipping per-window cache discovery (deprecated)"
            )
        
        self.logger.info(
            "No completeness manifest or cache files found for %s H%s; falling back to consolidated panel",
            self.config.symbol,
            horizon,
        )
        return []

    def _parse_manifest_windows(
        self,
        windows_raw: List[Dict[str, Any]],
        horizon: int,
    ) -> List[Dict[str, Any]]:
        """Parse windows from manifest JSON into internal format."""
        resolved: List[Dict[str, Any]] = []
        for idx, window in enumerate(windows_raw):
            try:
                parsed = {
                    key: pd.to_datetime(window.get(key))
                    for key in ("train_start", "train_end", "valid_start", "valid_end")
                }
            except Exception as exc:
                self.logger.debug("Skipping malformed window %s: %s", window, exc)
                continue
            if any(parsed[key] is None for key in parsed):
                continue
            normalized: Dict[str, pd.Timestamp] = {}
            for key, value in parsed.items():
                ts_value = value
                if getattr(ts_value, "tzinfo", None) is not None:
                    ts_value = ts_value.tz_localize(None)
                normalized[key] = ts_value
            test_start = window.get("test_start")
            test_end = window.get("test_end")
            test_start_ts = pd.to_datetime(test_start) if test_start else None
            test_end_ts = pd.to_datetime(test_end) if test_end else None
            if test_start_ts is None:
                test_start_ts = add_sessions(normalized["valid_end"], 1)
            if test_end_ts is None:
                test_end_ts = add_sessions(test_start_ts, max(1, horizon) - 1)
            if getattr(test_start_ts, "tzinfo", None) is not None:
                test_start_ts = test_start_ts.tz_localize(None)
            if getattr(test_end_ts, "tzinfo", None) is not None:
                test_end_ts = test_end_ts.tz_localize(None)
            cache_idx = window.get("cache_window_idx")
            if cache_idx is None:
                cache_idx = window.get("cache_idx")
            resolved_idx = int(cache_idx) if cache_idx is not None else idx
            resolved.append(
                {
                    "window_id": int(window.get("window_id") or idx + 1),
                    "cache_idx": resolved_idx,
                    "cache_window_idx": resolved_idx,
                    "train_start": normalized["train_start"],
                    "train_end": normalized["train_end"],
                    "valid_start": normalized["valid_start"],
                    "valid_end": normalized["valid_end"],
                    "test_start": test_start_ts,
                    "test_end": test_end_ts,
                }
            )
        return resolved

    def _filter_windows_by_date(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter windows by config start/end date range."""
        start_filter = pd.to_datetime(self.config.start) if self.config.start else None
        end_filter = pd.to_datetime(self.config.end) if self.config.end else None
        if getattr(start_filter, "tzinfo", None) is not None:
            start_filter = start_filter.tz_localize(None)
        if getattr(end_filter, "tzinfo", None) is not None:
            end_filter = end_filter.tz_localize(None)
        if not start_filter and not end_filter:
            return windows
        filtered: List[Dict[str, Any]] = []
        for window in windows:
            window_test_end = window.get("test_end", window["valid_end"])
            if start_filter is not None and window_test_end < start_filter:
                continue
            if end_filter is not None and window["train_start"] > end_filter:
                continue
            filtered.append(window)
        return filtered

    def _get_cached_optuna_seq_len(self, horizon: int) -> int:
        """Load cached Optuna seq_len for a horizon if available.
        
        Returns the cached seq_len or config default if not found.
        This allows validation windows to be sized appropriately.
        """
        cache_tag = "GLOBAL13" if bool(getattr(self.config, "optuna_global_multi_symbol", False)) else self.config.symbol
        cache_path = self.config.optuna_params_cache_dir / f"{cache_tag}_h{horizon}_optuna.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    seq_len = data.get("mamba_seq_len", data.get("lstm_seq_len", self.config.mamba_seq_len))
                    return int(seq_len)
            except Exception:
                pass
        return self.config.mamba_seq_len

    def _window_split_indices(
        self,
        index: pd.Index,
        window: Dict[str, Any],
        horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split panel index into train/validation/test indices for walk-forward.

        This is a pure date/position-based split using the manifest window geometry.
        Sequence model `seq_len` feasibility is enforced separately via a global
        fold-derived cap (no per-window clamping).
        """
        ts_index = pd.DatetimeIndex(index)
        if getattr(ts_index, "tz", None) is not None:
            ts_index = ts_index.tz_localize(None)
        if not ts_index.is_monotonic_increasing:
            raise ValueError("Panel index must be sorted ascending for positional slicing")
        if horizon is not None:
            self._index_position_cache[horizon] = ts_index
        train_start = pd.Timestamp(window["train_start"])
        train_end = pd.Timestamp(window["train_end"])
        valid_start = pd.Timestamp(window["valid_start"])
        valid_end = pd.Timestamp(window["valid_end"])
        test_start = pd.Timestamp(window.get("test_start") or valid_end)
        test_end = pd.Timestamp(window.get("test_end") or valid_end)
        
        for ts_name, ts_value in (
            ("train_start", train_start),
            ("train_end", train_end),
            ("valid_start", valid_start),
            ("valid_end", valid_end),
            ("test_start", test_start),
            ("test_end", test_end),
        ):
            normalized = ts_value
            if getattr(ts_value, "tzinfo", None) is not None:
                normalized = ts_value.tz_localize(None)
            window[ts_name] = normalized

        def _pos_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> np.ndarray:
            start_pos = int(np.clip(ts_index.searchsorted(start_ts, side="left"), 0, len(ts_index)))
            end_pos = int(np.clip(ts_index.searchsorted(end_ts, side="right"), 0, len(ts_index)))
            if start_pos >= end_pos:
                return np.empty(0, dtype=int)
            return np.arange(start_pos, end_pos, dtype=int)

        train_idx = _pos_range(window["train_start"], window["train_end"])
        val_idx = _pos_range(window["valid_start"], window["valid_end"])
        test_idx = _pos_range(window["test_start"], window["test_end"])

        return train_idx, val_idx, test_idx

    def _restrict_outputs_to_test_span(
        self,
        panel: pd.DataFrame,
        result: StageBResult,
        test_idx: np.ndarray,
    ) -> StageBResult:
        """Limit outputs/per-model predictions to rows contained in the test span."""
        if result.outputs is None or len(test_idx) == 0:
            return result

        test_dates = panel.index[test_idx]
        test_dates = pd.DatetimeIndex(test_dates)
        if test_dates.tz is not None:
            test_dates = test_dates.tz_localize(None)

        available = result.outputs.index.intersection(test_dates)
        if available.empty:
            self.logger.warning(
                "⚠️ No test-date rows present in outputs for window %s",
                result.window_id,
            )
            return result

        result.outputs = result.outputs.loc[available]
        if result.per_model_predictions:
            for key, frame in list(result.per_model_predictions.items()):
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    overlap = frame.index.intersection(test_dates)
                    result.per_model_predictions[key] = frame.loc[overlap]

        return result

    @staticmethod
    def _ts_to_iso(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            ts_value = pd.to_datetime(value)
        except Exception:
            return str(value)
        if getattr(ts_value, "tzinfo", None) is not None:
            ts_value = ts_value.tz_localize(None)
        return ts_value.isoformat()

    def _summarize_window_result(
        self,
        result: StageBResult,
        window_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "window_id": None,
            "train_start": None,
            "train_end": None,
            "valid_start": None,
            "valid_end": None,
            "primary_model": None,
            "primary_track": None,
            "primary_composite": None,
            "allow_lstm": result.allow_lstm,
            "backtest_metrics": result.backtest_metrics or {},
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        if window_meta:
            summary.update(
                {
                    "window_id": window_meta.get("window_id"),
                    "train_start": self._ts_to_iso(window_meta.get("train_start")),
                    "train_end": self._ts_to_iso(window_meta.get("train_end")),
                    "valid_start": self._ts_to_iso(window_meta.get("valid_start")),
                    "valid_end": self._ts_to_iso(window_meta.get("valid_end")),
                    "train_samples": window_meta.get("train_samples"),
                    "valid_samples": window_meta.get("valid_samples"),
                    "test_start": self._ts_to_iso(window_meta.get("test_start")),
                    "test_end": self._ts_to_iso(window_meta.get("test_end")),
                    "test_samples": window_meta.get("test_samples"),
                }
            )
        if result.primary_model is not None:
            summary["primary_model"] = result.primary_model.model_name
            summary["primary_track"] = result.primary_model.track
            summary["primary_composite"] = result.primary_model.metrics.get("composite")
        if result.outputs is not None:
            summary["output_rows"] = int(result.outputs.shape[0])
        return summary

    def _aggregate_window_results(
        self,
        horizon: int,
        window_results: List[StageBResult],
    ) -> StageBResult:
        ordered = sorted(
            window_results,
            key=lambda res: (
                (
                    res.window_meta.get("test_end")
                    if res.window_meta and res.window_meta.get("test_end") is not None
                    else res.window_meta.get("valid_end")
                    if res.window_meta
                    else pd.Timestamp.min
                )
            ),
        )
        base = ordered[-1]
        frames: List[pd.DataFrame] = []
        for res in ordered:
            if res.outputs is not None and not res.outputs.empty:
                frames.append(res.outputs.assign(window_id=res.window_id))
        combined_outputs = None
        if frames:
            combined_outputs = pd.concat(frames, axis=0)
            combined_outputs = combined_outputs[~combined_outputs.index.duplicated(keep="last")]
            combined_outputs = combined_outputs.sort_index()
        aggregate = replace(
            base,
            outputs=combined_outputs,
            backtest_equity=base.backtest_equity,
            backtest_metrics=base.backtest_metrics,
        )
        aggregate.per_model_predictions = base.per_model_predictions
        aggregate.horizon = horizon
        aggregate.window_id = base.window_id
        aggregate.window_meta = base.window_meta
        return aggregate

    def _execute_panel(
        self,
        panel: pd.DataFrame,
        horizon: int,
        priors: Dict[str, float],
        stage_a_meta: Dict[str, Any],
    ) -> StageBResult:
        if panel is None or panel.empty:
            raise ValueError(f"Stage B panel empty for horizon {horizon}")

        panel = self._normalize_panel_frame(panel)
        if horizon not in self._unified_panels:
            self._unified_panels[horizon] = panel.copy()
            self._index_position_cache[horizon] = pd.DatetimeIndex(panel.index)
        if horizon not in self._window_coverage and not panel.empty:
            self._window_coverage[horizon] = (
                panel.index.min(),
                panel.index.max(),
            )

        column_families = self._infer_column_families(panel.columns)
        self._attach_columns_to_specs(column_families)
        block_summaries = self._compute_block_summaries(panel)
        tracks = self._construct_tracks(
            panel,
            block_summaries,
            column_families,
            priors,
            stage_a_meta,
            horizon,
        )
        labels = self._construct_labels(horizon, panel.index)

        # 🚀 LSTM-ONLY MODE: Tier-1 (LightGBM/XGBoost/ElasticNet) removed
        # Only LSTM Tier-2 models are trained and evaluated
        self.logger.info("🚀 LSTM-ONLY MODE: Running LSTM Tier-2 models only")
        # 🚀 LSTM-ONLY MODE: No Tier-1 models

        allow_lstm = self._allow_lstm(
            horizon=horizon,
            n_samples=len(panel),
        )

        # =======================================================================
        # 🚀 OPTUNA PRE-OPTIMIZATION (when enabled)
        # Runs BEFORE LSTM to optimize feature selection and hyperparameters
        # ONE trial = ENTIRE walk-forward, encoders cached, Track C exported
        # =======================================================================
        optuna_params = None
        optuna_track_c = None
        optuna_encoders = None

        can_use_cached_track = (
            self.config.optuna_enabled
            and allow_lstm
            and horizon in self._optuna_track_cache
            and horizon in self._unified_panels
        )
        if can_use_cached_track:
            track_cols = self._optuna_track_cache.get(horizon, [])
            master_panel = self._unified_panels.get(horizon)
            if master_panel is not None and all(col in master_panel.columns for col in track_cols):
                try:
                    cached_slice = master_panel.loc[panel.index, track_cols].copy()
                    if not cached_slice.isnull().all().all():
                        optuna_track_c = cached_slice
                        self.logger.info(
                            "✅ Using cached Optuna Track C for %s H%s (columns=%d)",
                            self.config.symbol,
                            horizon,
                            len(track_cols),
                        )
                except KeyError:
                    optuna_track_c = None

        if (
            self.config.optuna_enabled
            and allow_lstm
            and OPTUNA_OPTIMIZER_AVAILABLE
            and optuna_track_c is None
        ):
            optuna_params = self._run_optuna_optimization(
                panel=panel,
                column_families=column_families,
                labels=labels,
                block_summaries=block_summaries,
                priors=priors,
                horizon=horizon,
            )
            if optuna_params is not None:
                self._apply_optuna_lstm_params(optuna_params)
                optuna_track_c, optuna_encoders = self._build_optuna_track_c(
                    panel=panel,
                    column_families=column_families,
                    block_summaries=block_summaries,
                    optuna_params=optuna_params,
                )
                self.logger.info(
                    "✅ Optuna Track C built: shape=%s, encoders=%d",
                    optuna_track_c.shape if optuna_track_c is not None else None,
                    len(optuna_encoders) if optuna_encoders else 0,
                )
                self._merge_optuna_track_c_into_unified_panel(horizon, optuna_track_c)

        # Build LSTM features - use Optuna Track C if available, else legacy views
        if optuna_track_c is not None:
            # 🚀 OPTUNA MODE: Use Track C directly (no Track A/B views)
            lstm_features = {"track_c": optuna_track_c}
            self.logger.info(
                "🚀 Using Optuna Track C for LSTM: dim=%d (Track A/B bypassed)",
                optuna_track_c.shape[1]
            )
        elif allow_lstm:
            # Legacy mode: build views from scratch
            lstm_features = self._prepare_lstm_features(
                panel=panel,
                block_summaries=block_summaries,
                column_families=column_families,
                priors=priors,
                stage_a_meta=stage_a_meta,
            )
        else:
            lstm_features = None

        tier2 = self._run_tier_two_models(
            tracks=tracks,
            labels=labels,
            horizon=horizon,
            allow_lstm=allow_lstm,
            lstm_features=lstm_features,
            train_idx=self._current_train_idx,  # 🔧 Pass train indices for walk-forward
        )

        ranking, primary_model, selection_meta = self._select_best_models(
            tier2_results=tier2,
            tracks=tracks,
        )

        ensemble = self._build_local_ensemble(
            ranking,
            tracks.get("B").frame if "B" in tracks else pd.DataFrame(),
        )

        outputs = self._generate_outputs(
            horizon=horizon,
            tracks=tracks,
            ranking=ranking,
            primary=primary_model,
            ensemble=ensemble,
            selection_meta=selection_meta,
            labels=labels,
        )

        backtest_equity, backtest_metrics = (None, None)
        if outputs is not None and not outputs.empty:
            backtest_equity, backtest_metrics = self._run_backtest_for_outputs(horizon, outputs)

        return StageBResult(
            horizon=horizon,
            tracks=tracks,
            tier2=tier2,
            allow_lstm=allow_lstm,
            lstm_features=lstm_features,
            candidates=ranking,
            primary_model=primary_model,
            ensemble=ensemble,
            outputs=outputs,
            per_model_predictions=selection_meta.get("per_model_predictions", {}),
            family_specs=self.family_specs,
            stage_a_priors=priors,
            backtest_equity=backtest_equity,
            backtest_metrics=backtest_metrics,
        )

    def _resolve_cache_dir(self, horizon: int) -> Optional[Path]:
        prep_cfg = self.config.prep
        # If prep is enabled, always use prep-managed cache structure
        if prep_cfg is not None and prep_cfg.enabled:
            cache_root = Path(prep_cfg.cache_root).expanduser() if prep_cfg.cache_root else DEFAULT_PREP_CACHE_ROOT
            cache_root.mkdir(parents=True, exist_ok=True)
            symbol_token = self.config.symbol.lower()
            # 🔧 SIMPLIFIED: Use ONE flat folder per symbol/horizon
            # No more nested step subdirs - all windowed files go in same place
            folder = f"{symbol_token}_h{horizon}"
            target = cache_root / folder
            target.mkdir(parents=True, exist_ok=True)
            return target
        # If prep is disabled, use config.cache_dir directly
        if self.config.cache_dir is not None:
            path = Path(self.config.cache_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path
        # Fallback: no cache dir
        return None

    def _cache_suffix_token(self, prep_cfg: PrepFamiliesSettings) -> str:
        """Legacy method - kept for backwards compatibility but no longer used for cache paths."""
        mode = (prep_cfg.mode or "stage-b").lower()
        if mode == "stage-a":
            return "stage_a"
        if mode not in {"stage-b", "walkforward"}:
            return mode.replace("-", "_")
        if prep_cfg.wf_step_days:
            return f"step{int(prep_cfg.wf_step_days)}d"
        if prep_cfg.wf_step_years:
            return f"step{int(prep_cfg.wf_step_years)}y"
        return "wf"

    def _ensure_feature_cache(
        self,
        horizon: int,
        cache_dir: Optional[Path],
        families: List[str],
    ) -> None:
        prep_cfg = self.config.prep
        self.logger.info("🔍 _ensure_feature_cache called: prep_cfg=%s, enabled=%s", 
                        "None" if prep_cfg is None else "exists",
                        prep_cfg.enabled if prep_cfg else False)
        if prep_cfg is None or not prep_cfg.enabled:
            self.logger.info("⚠️ _ensure_feature_cache: prep disabled, skipping")
            return
        cache_dir = cache_dir or self._resolve_cache_dir(horizon)
        if cache_dir is None:
            self.logger.warning(
                "prep_families enabled but cache_dir unresolved for %s H%s",
                self.config.symbol,
                horizon,
            )
            return
        self.logger.info("🔍 _ensure_feature_cache: cache_dir=%s", cache_dir)
        if not PREP_FAMILIES_SCRIPT.exists():
            raise FileNotFoundError(f"prep_families.py not found at {PREP_FAMILIES_SCRIPT}")
        cache_key = f"{cache_dir.resolve()}::h{horizon}"
        
        # Check if prep_families was already run for this horizon (success or failure)
        if cache_key in self._prep_completed:
            self.logger.info("⚠️ _ensure_feature_cache: cache_key already in _prep_completed, skipping")
            return
        
        # Check if completeness manifest exists and is recent
        output_dir = Path(prep_cfg.output_dir).expanduser() if prep_cfg.output_dir else DEFAULT_PREP_OUTPUT_DIR
        manifest = output_dir / f"{self.config.symbol.lower()}_h{horizon}_completeness.json"
        panel_ready = self._panel_artifact_ready_and_valid(horizon, families)
        windows_ready = False
        self.logger.info("🔍 _ensure_feature_cache: checking manifest at %s (exists=%s)", manifest, manifest.exists())
        if manifest.exists():
            try:
                with open(manifest, "r") as f:
                    payload = json.load(f)
                windows_ready = bool(payload.get("windows"))
                if windows_ready:
                    self.logger.info(
                        "Completeness manifest lists %d windows for %s H%s",
                        len(payload.get("windows", [])),
                        self.config.symbol,
                        horizon,
                    )
                else:
                    self.logger.warning("Manifest found but windows list empty - will re-run prep_families")
            except Exception as exc:
                self.logger.debug("Failed to parse manifest %s: %s", manifest, exc)
                windows_ready = False

        if panel_ready and windows_ready and not prep_cfg.force:
            self.logger.info(
                "Unified feature panel and manifest already exist for %s H%s - skipping prep_families",
                self.config.symbol,
                horizon,
            )
            self._prep_completed.add(cache_key)
            return
        
        prep_params = self._prep_params(prep_cfg, horizon)
        if prep_params is None:
            return
        if prep_params is None:
            return
        wf_start, wf_end, wf_train_years, wf_step_years, wf_step_days = prep_params
        command = self._build_prep_command(
            prep_cfg=prep_cfg,
            cache_dir=cache_dir,
            families=families,
            horizon=horizon,
            wf_start=wf_start,
            wf_end=wf_end,
            wf_train_years=wf_train_years,
            wf_step_years=wf_step_years,
            wf_step_days=wf_step_days,
        )
        self.logger.info(
            "Ensuring per-window caches via prep_families for %s H%s",
            self.config.symbol,
            horizon,
        )
        self.logger.debug("prep_families command: %s", " ".join(command))
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH")
        repo_path = str(REPO_ROOT)
        if python_path:
            if repo_path not in python_path.split(os.pathsep):
                env["PYTHONPATH"] = os.pathsep.join([repo_path, python_path])
        else:
            env["PYTHONPATH"] = repo_path
        try:
            subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=env)
            self._prep_completed.add(cache_key)
        except subprocess.CalledProcessError as exc:
            self.logger.warning(
                "prep_families failed for %s H%s (exit=%d) - will fall back to live generation where needed",
                self.config.symbol,
                horizon,
                exc.returncode,
            )
            # Don't crash - let build_panel fall back to live generation for missing families
            return

    def _prep_params(
        self,
        prep_cfg: PrepFamiliesSettings,
        horizon: int,
    ) -> Optional[Tuple[str, str, int, Optional[int], Optional[int]]]:
        wf_start = prep_cfg.wf_start or self.config.start or DEFAULT_STAGE_A_WF_START
        wf_end = prep_cfg.wf_end or self.config.end or DEFAULT_STAGE_A_WF_END
        train_years = (
            prep_cfg.wf_train_years
            if prep_cfg.wf_train_years is not None
            else DEFAULT_STAGE_A_WF_TRAIN_YEARS
        )
        step_years = prep_cfg.wf_step_years
        step_days = (
            prep_cfg.wf_step_days
            if prep_cfg.wf_step_days is not None
            else DEFAULT_STAGE_B_STEP_DAYS
        )
        missing: List[str] = []
        if not wf_start:
            missing.append("wf_start")
        if not wf_end:
            missing.append("wf_end")
        if train_years is None:
            missing.append("wf_train_years")
        if step_years is None and step_days is None:
            step_days = int(horizon)
        if step_years is not None and step_days is not None:
            self.logger.error(
                "prep_families misconfigured for %s H%s: both step_years and step_days provided",
                self.config.symbol,
                horizon,
            )
            return None
        if missing:
            self.logger.warning(
                "prep_families skipped for %s H%s; missing %s",
                self.config.symbol,
                horizon,
                ", ".join(missing),
            )
            return None
        return (
            str(wf_start),
            str(wf_end),
            int(train_years),
            int(step_years) if step_years is not None else None,
            int(step_days) if step_days is not None else None,
        )

    def _format_family_selector(
        self,
        prep_cfg: PrepFamiliesSettings,
        families: List[str],
    ) -> str:
        if prep_cfg.families:
            cleaned = sorted({fam.strip() for fam in prep_cfg.families if fam.strip()})
            return ",".join(cleaned) if cleaned else "all"
        token = (prep_cfg.family_selector or "all").strip()
        if token.lower() == "resolved":
            return ",".join(families)
        return token or "all"

    def _build_prep_command(
        self,
        prep_cfg: PrepFamiliesSettings,
        cache_dir: Path,
        families: List[str],
        horizon: int,
        wf_start: str,
        wf_end: str,
        wf_train_years: int,
        wf_step_years: Optional[int],
        wf_step_days: Optional[int],
    ) -> List[str]:
        output_dir = Path(prep_cfg.output_dir).expanduser() if prep_cfg.output_dir else DEFAULT_PREP_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        family_arg = self._format_family_selector(prep_cfg, families)
        cmd = [
            sys.executable,
            str(PREP_FAMILIES_SCRIPT),
            "--symbol",
            self.config.symbol.upper(),
            "--horizon",
            str(horizon),
            "--wf-start",
            wf_start,
            "--wf-end",
            wf_end,
            "--wf-train-years",
            str(wf_train_years),
        ]
        if wf_step_days is not None:
            cmd.extend(["--wf-step-days", str(wf_step_days)])
        elif wf_step_years is not None:
            cmd.extend(["--wf-step-years", str(wf_step_years)])
        mode = (prep_cfg.mode or "stage-b").lower()
        if mode in {"walkforward", "stage-b"}:
            normalized_mode = "stage-b"
        elif mode == "stage-a":
            normalized_mode = "stage-a"
        else:
            normalized_mode = mode
        cmd.extend(["--mode", normalized_mode])
        cmd.extend(["--families", family_arg])
        cmd.extend(["--strict", "yes" if prep_cfg.strict else "no"])
        workers = max(1, int(prep_cfg.workers or 1))
        cmd.extend(["--workers", str(workers)])
        if prep_cfg.hf_workers:
            cmd.extend(["--hf-workers", str(max(1, int(prep_cfg.hf_workers)))])
        cmd.extend(["--cache-dir", str(cache_dir)])
        cmd.extend(["--output-dir", str(output_dir)])
        log_level = (prep_cfg.log_level or "INFO").upper()
        cmd.extend(["--log-level", log_level])
        return cmd

    def _resolve_families(self) -> List[str]:
        if self.config.families:
            return [fam.strip() for fam in self.config.families if fam]
        fams: List[str] = []
        if self.config.include_stage_a:
            fams.extend(STAGE_A_FAMILIES)
        fams.extend(STAGE_B_FAMILIES)
        seen = set()
        ordered: List[str] = []
        stage_b_order = list(STAGE_B_FAMILIES)
        for fam in stage_b_order + fams:
            if fam not in seen:
                ordered.append(fam)
                seen.add(fam)
        # CRITICAL FIX: hf_agg is computed in-pipeline, not by prep_families
        # Remove it from the list passed to prep_families
        ordered = [f for f in ordered if f != "hf_agg"]
        return ordered

    def _load_stage_a_artifacts(self) -> Tuple[Dict[Any, Dict[str, float]], Dict[Any, Dict[str, Any]]]:
        priors: Dict[Any, Dict[str, float]] = {}
        metadata: Dict[Any, Dict[str, Any]] = {}
        targets: List[Any] = list(self.config.horizons)
        targets.append("global")
        for token in targets:
            if token == "global":
                path = self.config.stage_a_artifact_dir / self.config.symbol.upper() / self.config.stage_a_weights_filename
            else:
                path = self.config.stage_a_artifact_dir / f"{self.config.symbol.upper()}_h{token}" / self.config.stage_a_weights_filename
            priors[token] = self._read_stage_a_weights(path)
            metadata[token] = self._read_stage_a_meta(path)
        return priors, metadata

    def _read_stage_a_weights(self, path: Path) -> Dict[str, float]:
        """Read family weights from Stage-A family_weights_best.json.
        
        Uses family_normalized_scores (global.normalized) if available,
        otherwise falls back to family_weights.
        """
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - IO guard
            self.logger.warning("Failed to read Stage A weights from %s: %s", path, exc)
            return {}
        # Prefer family_normalized_scores (global.normalized) for ranking
        weights = payload.get("family_normalized_scores", {})
        if not weights:
            weights = payload.get("family_weights", {})
        return {fam: float(val) for fam, val in weights.items() if isinstance(val, (int, float))}

    def _read_stage_a_meta(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception:  # pragma: no cover
            return {}
        return {
            "family_scores": payload.get("family_normalized_scores", payload.get("family_scores", {})),
            "family_stability": payload.get("family_stability", {}),
            "family_coverage": payload.get("family_coverage", {}),
            "family_feature_details": payload.get("family_feature_details", {}),
            "family_average_scores": payload.get("family_average_scores", {}),
            "family_logits": payload.get("family_logits", {}),
        }

    def _infer_column_families(self, columns: Iterable[str]) -> Dict[str, str]:
        """
        Infer which family each column belongs to.
        
        Matches columns against ALL known family names (Stage A, Stage B, HF blocks).
        Handles family names with underscores like 'alternative_signals', 'cross_asset', etc.
        Also handles special prefix mappings for columns that don't follow standard naming.
        
        IMPORTANT: Only known families are included. Unknown columns are skipped
        to avoid creating bogus 1-3 column "families" that waste 3-pillar analysis time.
        """
        # Build list of all known family names, sorted by length (longest first)
        # This ensures 'doc_embedding_novelty_hf' matches before 'doc' would
        all_known_families = sorted(
            set(STAGE_A_FAMILIES) | set(STAGE_B_FAMILIES) | set(HF_BLOCK_FAMILIES),
            key=len,
            reverse=True,
        )
        
        # Special prefix mappings for columns that don't follow family_* pattern
        # Maps prefix -> target family
        SPECIAL_PREFIX_MAPPINGS = {
            # GDELT / Global Events features -> map to existing global events family
            # NOTE: These are emitted by the doc-embedding novelty module and are un-prefixed.
            # They should not be attributed to the EODHD technical bundle (ml_framework).
            "n_events": "doc_embedding_novelty_hf",
            "n_articles": "doc_embedding_novelty_hf",
            "macro_novelty": "doc_embedding_novelty_hf",
            "geopolitical_novelty": "doc_embedding_novelty_hf",
            "regulatory_novelty": "doc_embedding_novelty_hf",
            "energy_novelty": "doc_embedding_novelty_hf",
            "conflict_novelty": "doc_embedding_novelty_hf",
            "tech_novelty": "doc_embedding_novelty_hf",
            "theme_weight": "doc_embedding_novelty_hf",
            "baseline_mean": "doc_embedding_novelty_hf",
            "novelty_spike_flag": "doc_embedding_novelty_hf",
            "novelty_persistence": "doc_embedding_novelty_hf",
            "top_theme_numeric": "doc_embedding_novelty_hf",

            # Base panel hygiene columns (un-prefixed)
            "has_data": "listing_status",
            "is_etf": "listing_status",

            # Feature module prefixes that don't match Stage-A family names
            # (keep mapping stable so Phase-2 weights can apply deterministically).
            "econ_events_calendar": "econ_events_calendar",
            "insider_form4": "alternative_signals",
            "corp_actions_splits": "dividends",
            "marketcap_history": "dcf",
            "exchange_calendar": "microstructure",
            "index_constituents_": "index_constituents",
            "news_sentiment_hf": "doc_embedding_novelty_hf",
            "fin_g1": "fin_g2",
            
            # Macro lagged features (l1_, l2_, l3_, derived_) -> macro_tst_hf
            "l1_": "macro_tst_hf",
            "l2_": "macro_tst_hf",
            "l3_": "macro_tst_hf",
            "derived_": "macro_tst_hf",
            
            # Quantile features without prefix -> quantile_forecast
            "q_loc": "quantile_forecast",
            "q_spread": "quantile_forecast",
            "q_vol": "quantile_forecast",
            "q_skew": "quantile_forecast",
            "q_tail": "quantile_forecast",
            "quantile_hf_": "quantile_forecast",
            "eff_q_": "quantile_forecast",
            
            # Calibration features
            "cal_quality": "calibration",
            
            # Online learning features
            "ol_mae": "online_learning",
            "ol_rmse": "online_learning",
            "ol_mape": "online_learning",
            "ol_dir_acc": "online_learning",
            "drift_flag": "online_learning",
            
            # AR/ARIMA features -> arima_forecast
            "ar_forecast": "arima_forecast",
            "ar_momentum": "arima_forecast",
            "ar_persistence": "arima_forecast",
        }
        
        # Columns to completely skip (metadata, not features)
        SKIP_COLUMNS = {"date", "symbol", "ticker", "index"}
        
        mapping: Dict[str, str] = {}
        skipped_cols = []
        
        for col in columns:
            col_lower = col.lower()
            
            # Skip metadata columns
            if col_lower in SKIP_COLUMNS:
                continue
            
            matched = False
            
            # First try standard family prefix matching (longest first)
            for family in all_known_families:
                # Column should start with family name followed by underscore
                if col_lower.startswith(f"{family.lower()}_"):
                    mapping[col] = family
                    matched = True
                    break
            
            # If not matched, try special prefix mappings
            if not matched:
                for prefix, target_family in SPECIAL_PREFIX_MAPPINGS.items():
                    if col_lower.startswith(prefix.lower()) or col_lower == prefix.lower():
                        mapping[col] = target_family
                        matched = True
                        break
            
            # Track unmatched columns for debugging
            if not matched:
                skipped_cols.append(col)
        
        # Log skipped columns for debugging
        if skipped_cols:
            self.logger.debug(f"Skipped {len(skipped_cols)} columns not matching known families: {skipped_cols[:10]}...")
        
        return mapping

    def _attach_columns_to_specs(self, column_families: Dict[str, str]) -> None:
        for col, fam in column_families.items():
            spec = self.family_specs.get(fam)
            if spec is None:
                spec = FamilySpec(name=fam, stage="A", tags=["unknown"])
                self.family_specs[fam] = spec
            if col not in spec.columns:
                spec.columns.append(col)

    # ------------------------------------------------------------------
    # Track data validation
    # ------------------------------------------------------------------
    def _validate_track_data_requirements(
        self,
        panel: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        🔧 Validate data requirements for each track BEFORE construction.
        
        NOTE: Track A now includes ALL numeric columns from panel (not just STAGE_A_FAMILIES).
        This validation reflects the ACTUAL track construction logic.
        """
        validation = {}
        
        # Get families present in panel
        panel_families = set(column_families.values())
        
        # Count ALL numeric columns (this is what Track A actually gets)
        numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
        numeric_non_null = panel[numeric_cols].notna().any(axis=0).sum() if numeric_cols else 0
        
        # Count lag features specifically
        lag_cols = [c for c in numeric_cols if '_lag' in c.lower()]
        
        # === Track A validation ===
        # Track A / seq_raw uses ONLY Stage-A families (27 families)
        # Adaptive selection may reduce this further
        stage_a_families_present = panel_families & set(STAGE_A_FAMILIES)
        stage_a_cols = [c for c in numeric_cols if column_families.get(c) in STAGE_A_FAMILIES]
        stage_a_non_null = panel[stage_a_cols].notna().any(axis=0).sum() if stage_a_cols else 0
        stage_a_lag_cols = [c for c in stage_a_cols if '_lag' in c.lower()]
        
        validation["A"] = {
            "is_valid": len(stage_a_cols) > 0 and stage_a_non_null > 0,
            "available_families": sorted(stage_a_families_present),  # Only Stage-A families
            "missing_families": sorted(set(STAGE_A_FAMILIES) - stage_a_families_present),
            "available_columns": len(stage_a_cols),
            "non_null_columns": stage_a_non_null,
            "lag_columns": len(stage_a_lag_cols),
            "summary_status": {"combined": not block_summaries.get("combined", pd.DataFrame()).empty},
        }
        
        # === Track B validation ===
        # Track B needs: HF blocks + summaries (quantile, calibration, online, arima)
        hf_block_families = set(HF_BLOCK_FAMILIES)
        track_b_available_hf = panel_families & hf_block_families
        track_b_missing_hf = hf_block_families - panel_families
        
        # Check summary availability
        summary_status = {}
        for key in ["quantile", "calibration", "online", "arima"]:
            df = block_summaries.get(key, pd.DataFrame())
            has_data = not df.empty and df.notna().any(axis=0).sum() > 0
            summary_status[key] = has_data
        
        # Track B families that generate summaries
        summary_families = {"quantile_forecast", "calibration", "online_learning", "arima_forecast"}
        track_b_available_sum = panel_families & summary_families
        track_b_missing_sum = summary_families - panel_families
        
        # Track B columns from panel
        track_b_cols = [c for c in panel.columns 
                       if column_families.get(c) in (hf_block_families | summary_families)]
        track_b_non_null = panel[track_b_cols].notna().any(axis=0).sum() if track_b_cols else 0
        
        # Track B also gets block summaries (computed in-pipeline)
        summary_cols = sum(df.shape[1] for df in block_summaries.values() if not df.empty)
        
        # Track B is valid if it has HF data OR summary data
        track_b_valid = (
            (len(track_b_available_hf) > 0 or any(summary_status.values())) 
            and (track_b_non_null > 0 or summary_cols > 0)
        )
        
        validation["B"] = {
            "is_valid": track_b_valid,
            "available_families": sorted(track_b_available_hf | track_b_available_sum),
            "missing_families": sorted(track_b_missing_hf | track_b_missing_sum),
            "available_columns": len(track_b_cols),
            "non_null_columns": track_b_non_null,
            "summary_status": summary_status,
        }
        
        # === Track C validation ===
        # Track C combines A + B, so it's valid if either is valid
        track_c_available = validation["A"]["available_families"] + validation["B"]["available_families"]
        track_c_missing = list(set(validation["A"]["missing_families"]) & set(validation["B"]["missing_families"]))
        
        validation["C"] = {
            "is_valid": validation["A"]["is_valid"] or validation["B"]["is_valid"],
            "available_families": sorted(set(track_c_available)),
            "missing_families": sorted(track_c_missing),
            "available_columns": validation["A"]["available_columns"] + validation["B"]["available_columns"],
            "non_null_columns": validation["A"]["non_null_columns"] + validation["B"]["non_null_columns"],
            "summary_status": validation["B"]["summary_status"],
        }
        
        # Log validation results
        self._log_track_validation(validation)
        
        return validation

    def _log_track_validation(self, validation: Dict[str, Dict[str, Any]]) -> None:
        """Log track validation results."""
        self.logger.info("=" * 60)
        self.logger.info("📊 TRACK DATA VALIDATION")
        self.logger.info("=" * 60)
        
        for track_name, v in validation.items():
            status = "✅" if v["is_valid"] else "❌"
            self.logger.info(
                "%s Track %s: %d families, %d cols (%d non-null)",
                status,
                track_name,
                len(v["available_families"]),
                v["available_columns"],
                v["non_null_columns"],
            )
            
            # Track A: Show lag columns and summary status
            if track_name == "A":
                lag_cols = v.get("lag_columns", 0)
                self.logger.info("   📈 All numeric features: %d total, %d lag columns", 
                               v["available_columns"], lag_cols)
                sum_status = v.get("summary_status", {})
                if sum_status:
                    sum_str = ", ".join(f"{k}={'✓' if s else '✗'}" for k, s in sum_status.items())
                    self.logger.info("   📋 Summary blocks: %s", sum_str)
            
            if v["missing_families"]:
                missing_str = ", ".join(v["missing_families"][:5])
                if len(v["missing_families"]) > 5:
                    missing_str += f" (+{len(v['missing_families'])-5} more)"
                self.logger.info("   Missing: %s", missing_str)
            
            if track_name == "B":
                sum_status = v.get("summary_status", {})
                sum_str = ", ".join(f"{k}={'✓' if s else '✗'}" for k, s in sum_status.items())
                self.logger.info("   Summaries: %s", sum_str)
        
        self.logger.info("=" * 60)

    def _get_missing_families_for_regeneration(
        self,
        validation: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Get list of families that need to be regenerated.
        
        Prioritizes families that are missing from multiple tracks
        or that generate critical summaries.
        """
        # Critical families that generate summaries for Track B
        critical_families = {"quantile_forecast", "calibration", "online_learning", "arima_forecast"}
        
        # Collect all missing families
        all_missing = set()
        for track_name, v in validation.items():
            all_missing.update(v["missing_families"])
        
        # Prioritize critical families
        missing_critical = all_missing & critical_families
        missing_other = all_missing - critical_families
        
        # Return critical first, then others
        return list(missing_critical) + list(missing_other)

    # ------------------------------------------------------------------
    # Track construction
    # ------------------------------------------------------------------
    def _construct_tracks(
        self,
        panel: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
        priors: Dict[str, float],
        stage_a_meta: Dict[str, Any],
        horizon: int,
    ) -> Dict[str, TrackData]:
        # 🔧 NEW: Validate track data requirements before construction
        validation = self._validate_track_data_requirements(panel, block_summaries, column_families)
        
        numeric = panel.select_dtypes(include=[np.number]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        stage_a_scaled = self._apply_family_scaling(numeric, column_families, priors)
        stage_a_cols = [c for c in numeric.columns if column_families.get(c) in STAGE_A_FAMILIES]
        stage_a_block = stage_a_scaled[stage_a_cols] if stage_a_cols else pd.DataFrame(index=panel.index)

        family_meta_static = self._build_family_meta_from_stage_a(panel.index, stage_a_meta, top_k=None)
        family_meta_dynamic = self._build_family_dynamic_meta(stage_a_block, column_families, priors, block_summaries)
        family_meta_all = self._sanitize_frame(pd.concat([family_meta_static, family_meta_dynamic], axis=1))
        top_families = self._select_top_families(priors, stage_a_meta, self.config.top_k_families_for_seq)
        if not self.config.lstm_adaptive_selection_enabled:
            family_meta_top = family_meta_all
        else:
            family_meta_top = self._family_meta_subset(family_meta_dynamic, top_families)
            if family_meta_top is None or family_meta_top.empty:
                family_meta_top = self._family_meta_subset(family_meta_static, top_families)
            family_meta_top = self._sanitize_frame(family_meta_top)

        combined_summary = block_summaries.get("combined", pd.DataFrame(index=panel.index))
        hf_block = self._build_hf_agg_block(
            panel=panel,
            column_families=column_families,
            priors=priors,
            block_summaries=block_summaries,
            stage_a_meta=stage_a_meta,
        )
        for col in hf_block.columns:
            column_families[col] = "hf_agg"
        hf_block_features = self._build_hf_block_features(panel, column_families)
        hf_inputs = self._sanitize_frame(pd.concat([hf_block_features, hf_block], axis=1))

        # 🔧 CRITICAL FIX: Track A should include ALL raw features from ALL families
        # Not just STAGE_A_FAMILIES and STAGE_B_BASE_FAMILIES
        # This ensures LightGBM sees all features (786 cols from panel, not just 333)
        all_raw_features = numeric.copy()  # ALL numeric columns from panel
        
        # Log what we're including
        self.logger.info(
            "🔧 Track A: Including ALL %d raw features from panel (not filtered by family)",
            len(all_raw_features.columns),
        )
        
        # Build Track A from ALL raw features + meta + summaries
        track_a_sources = [all_raw_features, family_meta_all, combined_summary]
        track_a_frame = self._sanitize_frame(pd.concat(track_a_sources, axis=1))
        track_a_families = self._families_from_columns(track_a_frame.columns, column_families)
        track_a_tags = sorted(
            set(self._tags_for_families(track_a_families) + ["stage_a_priors", "macro_blend", "rich"])
        )

        quant_summaries = block_summaries.get("quantile", pd.DataFrame(index=panel.index))
        calibration_summary = block_summaries.get("calibration", pd.DataFrame(index=panel.index))
        online_summary = block_summaries.get("online", pd.DataFrame(index=panel.index))
        arima_summary = block_summaries.get("arima", pd.DataFrame(index=panel.index))
        track_b_sources = [hf_inputs, quant_summaries, calibration_summary, online_summary, arima_summary, family_meta_top]
        track_b_frame = self._sanitize_frame(pd.concat(track_b_sources, axis=1))
        track_b_families = self._families_from_columns(track_b_frame.columns, column_families)
        track_b_tags = sorted(
            set(self._tags_for_families(track_b_families) + ["hf", "hf_meta", "summary", "drift"])
        )

        track_c_extra = self._derive_track_c(track_b_frame, family_meta_top)
        track_c_frame = self._sanitize_frame(pd.concat([track_a_frame, track_b_frame, track_c_extra], axis=1))
        track_c_families = self._families_from_columns(track_c_frame.columns, column_families)
        if not track_c_families:
            track_c_families = ["synthetic"]
        track_c_tags = sorted(
            set(self._tags_for_families(track_c_families) + ["classification", "gating", "hybrid", "drift"])
        )

        return {
            "A": TrackData(
                name="TrackA",
                frame=track_a_frame,
                metadata={
                    "task": "regression",
                    "horizon": horizon,
                    "families": track_a_families,
                    "tags": track_a_tags,
                },
            ),
            "B": TrackData(
                name="TrackB",
                frame=track_b_frame,
                metadata={
                    "task": "regression",
                    "horizon": horizon,
                    "families": track_b_families,
                    "tags": track_b_tags,
                },
            ),
            "C": TrackData(
                name="TrackC",
                frame=track_c_frame,
                metadata={
                    "task": "regression",  # 🔧 FIX: Same as A/B for comparable scoring
                    "horizon": horizon,
                    "families": track_c_families,
                    "tags": track_c_tags,
                },
            ),
        }

    def _apply_family_scaling(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        priors: Dict[str, float],
    ) -> pd.DataFrame:
        if not priors:
            return panel.copy()
        scaled = panel.copy()
        for col in scaled.columns:
            fam = column_families.get(col)
            if fam is None:
                continue
            prior = priors.get(fam)
            if prior is None:
                continue
            weight = np.sqrt(max(prior, 1e-6))
            scaled[col] = scaled[col] * weight
        return scaled

    def _build_family_meta_from_stage_a(
        self,
        index: pd.Index,
        stage_a_meta: Dict[str, Any],
        top_k: Optional[int],
    ) -> pd.DataFrame:
        scores = stage_a_meta.get("family_scores", {})
        stability = stage_a_meta.get("family_stability", {})
        coverage = stage_a_meta.get("family_coverage", {})
        if top_k:
            sorted_fams = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: top_k]
            families = [fam for fam, _ in sorted_fams]
        else:
            families = list(scores.keys()) or list(STAGE_A_FAMILIES)
        meta = pd.DataFrame(index=index)
        for fam in families:
            prior = float(scores.get(fam, 0.0))
            confidence = float(stability.get(fam, coverage.get(fam, 0.0)))
            meta[f"family_score_prior_static_{fam}"] = prior
            meta[f"family_conf_static_{fam}"] = confidence
        return meta

    def _build_family_dynamic_meta(
        self,
        stage_a_frame: pd.DataFrame,
        column_families: Dict[str, str],
        priors: Dict[str, float],
        block_summaries: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build dynamic family meta columns with full specification.
        
        Creates the following columns for each family f:
        1. family_score_{f}(t) = mean of family features
        2. family_conf_{f}(t) = 1 / (1 + rolling_std)
        3. family_score_prior_{f}(t) = sqrt(global.normalized[f]) * family_score
        4. family_conf_prior_{f}(t) = sqrt(global.normalized[f]) * family_conf
        5. family_reliability_{f}(t) = family_score_prior * (1 - drift_severity) * (1 - cal_ECE)
        6. family_weight_const_{f} = global.normalized[f] (constant reference)
        7. family_score_zscore_{f}(t) = z-score of family_score
        8. family_conf_zscore_{f}(t) = z-score of family_conf
        """
        if stage_a_frame.empty:
            return pd.DataFrame(index=stage_a_frame.index)
        
        block_summaries = block_summaries or {}
        meta = pd.DataFrame(index=stage_a_frame.index)
        grouped = self._group_columns_by_family(stage_a_frame.columns, column_families)
        window = max(5, int(self.config.family_meta_window))
        min_periods = max(5, window // 3)
        
        # Get drift and calibration metrics from block_summaries
        index = stage_a_frame.index
        online_summary = block_summaries.get("online", pd.DataFrame(index=index))
        calibration_summary = block_summaries.get("calibration", pd.DataFrame(index=index))
        drift_severity = online_summary.get("drift_severity", pd.Series(0.0, index=index)).reindex(index).fillna(0.0)
        cal_ece = calibration_summary.get("cal_ECE", pd.Series(0.2, index=index)).reindex(index).fillna(0.2)
        
        for fam, cols in grouped.items():
            if fam not in STAGE_A_FAMILIES or not cols:
                continue
            fam_values = stage_a_frame[cols].astype(float)
            score = fam_values.mean(axis=1)
            rolling_std = score.rolling(window=window, min_periods=min_periods).std()
            conf = 1.0 / (1.0 + rolling_std.fillna(method="ffill").fillna(method="bfill").abs())
            
            # Prior weight: sqrt of normalized weight from Stage-A
            weight_const = max(priors.get(fam, 0.0), 0.0)
            weight = np.sqrt(weight_const)
            
            # 1-4: Score, conf, and their prior-weighted versions
            meta[f"family_score_{fam}"] = score
            meta[f"family_conf_{fam}"] = conf
            meta[f"family_score_prior_{fam}"] = score * weight
            meta[f"family_conf_prior_{fam}"] = conf * weight
            
            # 5: Reliability = score_prior * (1 - drift) * (1 - cal_ECE)
            meta[f"family_reliability_{fam}"] = (
                meta[f"family_score_prior_{fam}"] 
                * (1.0 - drift_severity.clip(0.0, 1.0)) 
                * (1.0 - cal_ece.clip(0.0, 1.0))
            )
            
            # 6: Constant weight reference
            meta[f"family_weight_const_{fam}"] = weight_const
            
            # 7-8: Z-scores for score and conf
            score_mean = score.rolling(window=window, min_periods=min_periods).mean()
            score_std = score.rolling(window=window, min_periods=min_periods).std().replace(0, 1e-6)
            meta[f"family_score_zscore_{fam}"] = ((score - score_mean) / score_std).fillna(0.0)
            
            conf_mean = conf.rolling(window=window, min_periods=min_periods).mean()
            conf_std = conf.rolling(window=window, min_periods=min_periods).std().replace(0, 1e-6)
            meta[f"family_conf_zscore_{fam}"] = ((conf - conf_mean) / conf_std).fillna(0.0)
            
            # Register all columns to their family
            column_families[f"family_score_{fam}"] = fam
            column_families[f"family_conf_{fam}"] = fam
            column_families[f"family_score_prior_{fam}"] = fam
            column_families[f"family_conf_prior_{fam}"] = fam
            column_families[f"family_reliability_{fam}"] = fam
            column_families[f"family_weight_const_{fam}"] = fam
            column_families[f"family_score_zscore_{fam}"] = fam
            column_families[f"family_conf_zscore_{fam}"] = fam
            
        return meta

    def _select_top_families(
        self,
        priors: Dict[str, float],
        stage_a_meta: Dict[str, Any],
        top_k: int,
    ) -> List[str]:
        """Select top families for LSTM using Stage-A weights.
        
        🔧 CRITICAL: Adaptive selection ONLY applies to Stage-A raw families.
        Stage-B families (quantile_forecast, calibration, etc.) are NEVER filtered.
        
        Uses Stage-A family_weights_best.json to rank families by their contribution
        to prediction performance.
        
        Args:
            priors: Family weights from Stage-A (family_weights_best.json)
            stage_a_meta: Stage-A metadata including family_scores
            top_k: Legacy fixed limit (0 = use adaptive ratio)
        
        Returns:
            List of selected Stage-A family names (Stage-B handled separately)
        """
        if not self.config.lstm_adaptive_selection_enabled:
            self.logger.info(
                "🔧 Adaptive Stage-A family selection disabled: using ALL %d families",
                len(STAGE_A_FAMILIES),
            )
            return list(STAGE_A_FAMILIES)

        # 🔧 CRITICAL: Only filter Stage-A families - Stage-B is always included
        # Filter priors to only include Stage-A families
        stage_a_priors = {fam: weight for fam, weight in priors.items() 
                         if fam in STAGE_A_FAMILIES}
        
        # Fallback to family_scores from Stage-A meta if priors empty
        if not stage_a_priors:
            stage_scores = stage_a_meta.get("family_scores", {})
            stage_a_priors = {fam: score for fam, score in stage_scores.items()
                             if fam in STAGE_A_FAMILIES}
        
        # If still empty, return all Stage-A families (no filtering)
        if not stage_a_priors:
            self.logger.warning(
                "⚠️ No Stage-A weights found - using ALL %d Stage-A families",
                len(STAGE_A_FAMILIES)
            )
            return list(STAGE_A_FAMILIES)
        
        # Sort Stage-A families by weight (descending)
        ranked = sorted(stage_a_priors.items(), key=lambda kv: kv[1], reverse=True)
        total_stage_a = len(ranked)
        
        # Adaptive K selection: K = floor(ratio * num_stage_a_families)
        if top_k <= 0:
            adaptive_k = max(1, int(self.config.lstm_adaptive_family_ratio * total_stage_a))
            k = min(total_stage_a, adaptive_k)
            self.logger.info(
                "🔧 Adaptive Stage-A family selection: K=%d of %d (ratio=%.2f)",
                k, total_stage_a, self.config.lstm_adaptive_family_ratio
            )
        else:
            # Legacy fixed limit
            k = min(top_k, total_stage_a)
        
        top = [fam for fam, _ in ranked[:k]]
        
        # Log which families were selected vs dropped
        if len(top) < total_stage_a:
            dropped = [fam for fam, _ in ranked[k:]]
            self.logger.debug(
                "Selected Stage-A families: %s", ", ".join(top[:5]) + ("..." if len(top) > 5 else "")
            )
            self.logger.debug(
                "Dropped Stage-A families: %s", ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else "")
            )
        
        return top if top else list(STAGE_A_FAMILIES)

    def _family_meta_subset(
        self,
        family_meta: pd.DataFrame,
        families: Sequence[str],
    ) -> pd.DataFrame:
        if family_meta.empty:
            return family_meta
        cols: List[str] = []
        # All 8 meta column suffixes per family
        suffixes = (
            "score", "conf", "score_prior", "conf_prior",
            "reliability", "weight_const", "score_zscore", "conf_zscore"
        )
        for fam in families:
            for suffix in suffixes:
                col = f"family_{suffix}_{fam}"
                if col in family_meta.columns:
                    cols.append(col)
                    continue
                static_col = f"family_{suffix}_static_{fam}"
                if static_col in family_meta.columns:
                    cols.append(static_col)
        return family_meta.reindex(columns=cols)

    def _families_from_columns(self, columns: Sequence[str], column_families: Dict[str, str]) -> List[str]:
        """Get unique families represented in the columns.
        
        NOTE: For Track B family counting, we exclude meta columns (family_score_*, family_conf_*, etc.)
        as these are summary statistics, not actual raw features from the family.
        """
        # All meta column prefixes to skip
        meta_prefixes = (
            "family_score_", "family_conf_", "family_score_prior_", "family_conf_prior_",
            "family_reliability_", "family_weight_const_", "family_score_zscore_", "family_conf_zscore_",
            "block_",  # Also skip block summaries
        )
        families: List[str] = []
        for col in columns:
            # Skip meta columns when counting families
            if col.startswith(meta_prefixes):
                continue
            fam = column_families.get(col)
            if fam:
                families.append(fam)
        return sorted(set(families))

    def _group_columns_by_family(
        self,
        columns: Sequence[str],
        column_families: Dict[str, str],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for col in columns:
            fam = column_families.get(col)
            if fam is None:
                continue
            grouped.setdefault(fam, []).append(col)
        return grouped

    def _tags_for_families(self, families: Sequence[str]) -> List[str]:
        tags: List[str] = []
        for fam in families:
            spec = self.family_specs.get(fam)
            if spec:
                tags.extend(spec.tags)
        return sorted(set(tags))

    def _build_hf_agg_block(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        priors: Dict[str, float],
        block_summaries: Dict[str, pd.DataFrame],
        stage_a_meta: Dict[str, Any],
    ) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame(index=panel.index)
        grouped = self._group_columns_by_family(panel.columns, column_families)
        index = panel.index
        score = pd.Series(0.0, index=index)
        weight_sum = pd.Series(0.0, index=index)
        drift_weighted = pd.Series(0.0, index=index)
        quality_weighted = pd.Series(0.0, index=index)
        online_summary = block_summaries.get("online", pd.DataFrame(index=index))
        calibration_summary = block_summaries.get("calibration", pd.DataFrame(index=index))
        default_drift = online_summary.get("drift_severity", pd.Series(0.0, index=index)).reindex(index).fillna(0.0)
        default_cal = calibration_summary.get("cal_ECE", pd.Series(0.2, index=index)).reindex(index).fillna(0.2)
        default_quality = calibration_summary.get("cal_quality", pd.Series(0.5, index=index)).reindex(index).fillna(0.5)
        family_scores = stage_a_meta.get("family_scores", {})
        total_prior = sum(max(priors.get(fam, family_scores.get(fam, 0.0)), 0.0) for fam in grouped.keys())
        total_prior = max(total_prior, 1e-6)

        for family, cols in grouped.items():
            if not cols or family == "hf_agg":
                continue
            fam_cols = [c for c in cols if panel[c].dtype.kind in {"f", "i"}]
            if not fam_cols:
                continue
            signal = panel[fam_cols].astype(float).mean(axis=1)
            prior = float(priors.get(family, family_scores.get(family, 0.0)))
            if prior <= 0:
                continue
            drift = default_drift
            cal = default_cal
            quality = default_quality
            weight = prior * (1.0 - drift.clip(0.0, 1.0)) * (1.0 - cal.clip(0.0, 1.0))
            score = score.add(weight * signal, fill_value=0.0)
            weight_sum = weight_sum.add(weight, fill_value=0.0)
            drift_weighted = drift_weighted.add(weight * drift, fill_value=0.0)
            quality_weighted = quality_weighted.add(weight * quality, fill_value=0.0)

        conf = (weight_sum / total_prior).clip(0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hf_drift = drift_weighted / weight_sum.replace(0.0, np.nan)
            hf_quality = quality_weighted / weight_sum.replace(0.0, np.nan)
        hf_drift = hf_drift.fillna(default_drift)
        hf_quality = hf_quality.fillna(default_quality)
        block = pd.DataFrame(
            {
                "hf_agg_score": score,
                "hf_agg_conf": conf,
                "hf_agg_drift": hf_drift,
                "hf_agg_quality": hf_quality,
            }
        )
        return block

    def _build_hf_block_features(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for family in HF_BLOCK_FAMILIES:
            fam_cols = [c for c in panel.columns if column_families.get(c) == family]
            if not fam_cols:
                continue
            fam_df = panel[fam_cols]
            block = pd.DataFrame(index=panel.index)
            score_col = self._find_column(fam_df, contains="score")
            conf_col = self._find_column(fam_df, contains="conf")
            score_series = fam_df[score_col] if score_col else fam_df.mean(axis=1)
            conf_series = fam_df[conf_col] if conf_col else pd.Series(0.5, index=panel.index)
            block_score_col = f"block_score_{family}"
            block_conf_col = f"block_conf_{family}"
            block[block_score_col] = score_series
            block[block_conf_col] = conf_series
            momentum_col = f"block_momentum_{family}"
            block[momentum_col] = score_series.diff().fillna(0.0)
            column_families[block_score_col] = family
            column_families[block_conf_col] = family
            column_families[momentum_col] = family
            frames.append(block)
        if not frames:
            return pd.DataFrame(index=panel.index)
        return self._sanitize_frame(pd.concat(frames, axis=1))

    def _derive_track_c(self, track_b_frame: pd.DataFrame, meta_top: pd.DataFrame) -> pd.DataFrame:
        cols_of_interest = [
            "q_loc",
            "q_spread",
            "q_vol",
            "q_skew_proxy",
            "eff_q_loc",
            "eff_q_spread",
            "cal_ECE",
            "cal_quality",
            "ol_dir_acc",
            "ol_mae",
            "ol_rmse",
            "drift_severity",
            "drift_flag",
            "time_since_last_drift",
            "alpha_weight",
            "bias_correction_strength",
            "ar_forecast_5d",
            "ar_resid_vol",
            "ar_momentum_sign",
            "ar_persistence",
            "hf_agg_score",
            "hf_agg_conf",
        ]
        frame = track_b_frame.reindex(columns=cols_of_interest)
        frame = frame.copy()
        frame["quantile_signal_strength"] = frame.get("q_loc", 0.0) / (frame.get("q_vol", 1.0) + 1e-6)
        frame["drift_pressure"] = frame.get("drift_severity", 0.0) * (frame.get("drift_flag", 0.0) + 1.0)
        frame["calibration_health"] = 1.0 - frame.get("cal_ECE", 0.0)
        meta_subset = meta_top.iloc[:, : min(2 * self.config.top_k_families_for_seq, meta_top.shape[1])]
        frame = pd.concat([frame, meta_subset], axis=1)
        return self._sanitize_frame(frame)

    def _sanitize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        df = df.loc[:, ~df.columns.duplicated()]
        nunique = df.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1]
        keep_cols = nunique[nunique > 1].index
        if len(constant_cols) > 0:
            self.logger.debug(
                "🔧 _sanitize_frame: removing %d constant columns (keeping %d)",
                len(constant_cols), len(keep_cols)
            )
        if not len(keep_cols):
            return df
        return df[keep_cols]

    # ------------------------------------------------------------------
    # Block summaries
    # ------------------------------------------------------------------
    def _compute_block_summaries(self, panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute block summaries for Track B.
        
        🔧 CRITICAL FIX: When _current_train_idx is set (walk-forward mode),
        summaries are computed ONLY on train portion to prevent data leakage.
        The summaries are then forward-filled to validation portion to simulate
        what would be available at prediction time.
        """
        summaries: Dict[str, pd.DataFrame] = {}
        
        # 🔧 CRITICAL: Use only train data for summary computation to prevent leakage
        if self._current_train_idx is not None and len(self._current_train_idx) > 0:
            train_panel = panel.iloc[self._current_train_idx]
            full_index = panel.index
            use_train_only = True
            self.logger.info(
                "🛡️ LEAKAGE PREVENTION: Computing block summaries on train-only data (%d rows)",
                len(train_panel),
            )
        else:
            train_panel = panel
            full_index = panel.index
            use_train_only = False
        
        quant_cols = [c for c in train_panel.columns if c.startswith("quantile_forecast_")]
        cal_cols = [c for c in train_panel.columns if c.startswith("calibration_")]
        ol_cols = [c for c in train_panel.columns if c.startswith("online_learning_")]
        ar_cols = [c for c in train_panel.columns if c.startswith("arima_forecast_")]
        
        # 🔧 NEW: Log column availability for summaries
        self.logger.info(
            "📊 Block summary columns: quantile=%d, calibration=%d, online=%d, arima=%d",
            len(quant_cols), len(cal_cols), len(ol_cols), len(ar_cols),
        )
        
        if quant_cols:
            summaries["quantile"] = self._summarize_quantiles(train_panel[quant_cols])
        else:
            self.logger.warning("⚠️ No quantile_forecast columns - Track B quantile summary will be empty")
            
        if cal_cols:
            summaries["calibration"] = self._summarize_calibration(train_panel[cal_cols])
        else:
            self.logger.warning("⚠️ No calibration columns - Track B calibration summary will be empty")
            
        if ol_cols:
            summaries["online"] = self._summarize_online(train_panel[ol_cols])
        else:
            self.logger.warning("⚠️ No online_learning columns - Track B online summary will be empty")
            
        if ar_cols:
            summaries["arima"] = self._summarize_arima(train_panel)
        else:
            self.logger.warning("⚠️ No arima_forecast columns - Track B arima summary will be empty")
        
        # 🔧 CRITICAL: If using train-only data, expand summaries to full panel index
        # Use forward-fill to simulate what would be available at prediction time
        if use_train_only:
            expanded_summaries: Dict[str, pd.DataFrame] = {}
            for key, df in summaries.items():
                if df.empty:
                    expanded_summaries[key] = pd.DataFrame(index=full_index)
                    continue
                # Reindex to full panel, forward-fill from train data
                # This ensures val period sees only information available up to train_end
                expanded = df.reindex(full_index)
                expanded = expanded.ffill()  # Forward-fill train values into val period
                expanded = expanded.bfill()  # Backfill any leading NaNs
                expanded_summaries[key] = expanded
            summaries = expanded_summaries
            self.logger.info(
                "🛡️ Expanded train-only summaries to full panel index (forward-filled)"
            )
            
        combined_pieces = [df for key, df in summaries.items() if key != "combined"]
        if combined_pieces:
            summaries["combined"] = self._sanitize_frame(pd.concat(combined_pieces, axis=1))
        else:
            summaries["combined"] = pd.DataFrame(index=full_index)
            
        # 🔧 NEW: Validate summary quality
        self._validate_block_summaries(summaries)
        
        return summaries

    def _validate_block_summaries(self, summaries: Dict[str, pd.DataFrame]) -> None:
        """Validate that block summaries have meaningful data."""
        empty_summaries = []
        for name, df in summaries.items():
            if name == "combined":
                continue
            if df.empty or df.shape[1] == 0:
                empty_summaries.append(name)
            elif df.notna().any(axis=0).sum() == 0:
                empty_summaries.append(f"{name}(all-null)")
                
        if empty_summaries:
            self.logger.warning(
                "⚠️ Track B data quality issue - empty/null summaries: %s",
                ", ".join(empty_summaries),
            )

    def _summarize_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame(index=df.index)
        def pick(suffix: str) -> Optional[str]:
            cols = [c for c in df.columns if c.endswith(suffix)]
            return cols[0] if cols else None

        q05, q25, q50, q75, q95 = (pick("q05"), pick("q25"), pick("q50"), pick("q75"), pick("q95"))
        if q50:
            summary["q_loc"] = df[q50]
        if q95 and q05:
            summary["q_spread"] = df[q95] - df[q05]
            summary["q_vol"] = summary["q_spread"] / 1.645
        if q75 and q25 and q50:
            summary["q_skew_proxy"] = df[q75] + df[q25] - 2 * df[q50]
        if q05:
            summary["q_tail_low"] = df[q05]
        hf_score = [c for c in df.columns if "hf_score" in c]
        hf_conf = [c for c in df.columns if "hf_conf" in c]
        if hf_score:
            summary["quantile_hf_score"] = df[hf_score[0]]
        if hf_conf:
            conf = df[hf_conf[0]].clip(0.0, 1.0)
            summary["quantile_hf_conf"] = conf
            if "q_loc" in summary:
                summary["eff_q_loc"] = summary["q_loc"] * conf
            if "q_spread" in summary:
                summary["eff_q_spread"] = summary["q_spread"] * (1 - conf)
            if "q_vol" in summary:
                summary["eff_q_vol"] = summary["q_vol"] * (1 - conf)
        return summary

    def _summarize_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame(index=df.index)
        summary["cal_ECE"] = df.filter(like="expected_calibration_error", axis=1).mean(axis=1)
        summary["cal_ECE_max"] = df.filter(like="max_calibration_error", axis=1).mean(axis=1)
        summary["cal_quality"] = df.filter(like="overall_score", axis=1).mean(axis=1)
        recal_cols = df.filter(like="requires_recalibration", axis=1).columns
        if len(recal_cols):
            summary["cal_recal_flag"] = df[recal_cols].max(axis=1)
        return summary

    def _summarize_online(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = pd.DataFrame(index=df.index)
        for metric in ["mae", "rmse", "mape"]:
            col = self._find_column(df, contains=f"_{metric}")
            if col:
                summary[f"ol_{metric}"] = df[col]
        dir_col = self._find_column(df, contains="direction_accuracy")
        if dir_col:
            summary["ol_dir_acc"] = df[dir_col]
        drift_cols = df.filter(like="drift", axis=1).columns
        if len(drift_cols):
            severity_cols = [c for c in drift_cols if "severity" in c]
            flag_cols = [c for c in drift_cols if "flag" in c]
            if severity_cols:
                summary["drift_severity"] = df[severity_cols].max(axis=1)
            if flag_cols:
                summary["drift_flag"] = df[flag_cols].max(axis=1)
        tsl_col = self._find_column(df, contains="time_since_last_drift")
        if tsl_col:
            summary["time_since_last_drift"] = df[tsl_col]
        weight_col = self._find_column(df, contains="alpha_weight")
        bias_col = self._find_column(df, contains="bias_correction")
        if weight_col:
            summary["alpha_weight"] = df[weight_col]
        if bias_col:
            summary["bias_correction_strength"] = df[bias_col]
        if "alpha_weight" in summary and "eff_q_loc" in summary:
            summary["eff_q_loc"] = summary["eff_q_loc"] * summary["alpha_weight"]
        return summary

    def _summarize_arima(self, panel: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in panel.columns if c.startswith("arima_forecast_")]
        if not cols:
            return pd.DataFrame(index=panel.index)
        df = panel[cols]
        summary = pd.DataFrame(index=df.index)
        summary["ar_forecast_5d"] = df.filter(like="5d", axis=1).mean(axis=1)
        summary["ar_resid_vol"] = df.filter(like="residual_vol", axis=1).mean(axis=1)
        summary["ar_momentum_sign"] = np.sign(summary["ar_forecast_5d"].fillna(0.0))
        summary["ar_persistence"] = df.filter(like="persistence", axis=1).mean(axis=1)
        return summary

    def _summarize_correlation(self, panel: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in panel.columns if c.startswith("correlation_")]
        if not cols:
            return pd.DataFrame(index=panel.index)
        df = panel[cols]
        summary = pd.DataFrame(index=df.index)
        summary["corr_avg"] = df.mean(axis=1)
        summary["corr_vol"] = df.std(axis=1)
        return summary

    # ------------------------------------------------------------------
    # Labels and modeling
    # ------------------------------------------------------------------
    def _construct_labels(self, horizon: int, index: pd.Index) -> pd.DataFrame:
        start = self.config.start
        end = self.config.end
        if end is not None:
            end = (pd.to_datetime(end) + pd.Timedelta(days=horizon + 5)).strftime("%Y-%m-%d")
        price_df = _fetch_price_data(self.config.symbol, start, end)
        if price_df is None or "close" not in price_df.columns:
            raise ValueError("Price history unavailable for label construction")
        self._price_cache[horizon] = price_df
        close = price_df["close"].astype(float).sort_index()
        
        # 🔧 CRITICAL: Normalize price index timezone (remove tz if present)
        if hasattr(close.index, 'tz') and close.index.tz is not None:
            close = close.tz_localize(None)
        
        # Base label: forward log return
        # y_base(t) = log(P_{t+H} / P_t)
        future = close.shift(-horizon)
        forward_return = np.log(future / close)

        # Vol-adjusted label: forward log return divided by realized daily log-vol.
        # sigma_real(t) = std( log(P_{t-i}/P_{t-i-1}), i=1..N ), using only information up to t-1.
        # y_voladj(t) = y_base(t) / max(sigma_real(t), eps)
        vol_window = 21
        eps = 1e-4
        daily_log_ret = np.log(close / close.shift(1))
        sigma_real = daily_log_ret.rolling(window=int(vol_window), min_periods=int(vol_window)).std().shift(1)
        sigma_real = sigma_real.clip(lower=float(eps))
        forward_return_voladj = forward_return / sigma_real

        forward_direction = np.sign(forward_return).replace(0.0, 0.0)
        labels = pd.DataFrame(
            {
                "forward_return": forward_return,
                "forward_return_voladj": forward_return_voladj,
                "forward_direction": forward_direction,
            }
        )
        
        # 🔧 CRITICAL: Normalize panel index timezone for matching
        norm_index = pd.DatetimeIndex(index)
        if hasattr(norm_index, 'tz') and norm_index.tz is not None:
            norm_index = norm_index.tz_localize(None)
        
        # 🔧 DEBUG: Log index comparison before reindex
        self.logger.info(
            "🔍 Label alignment check: labels index %s to %s, panel index %s to %s",
            labels.index.min().strftime("%Y-%m-%d") if len(labels) > 0 else "N/A",
            labels.index.max().strftime("%Y-%m-%d") if len(labels) > 0 else "N/A",
            norm_index.min().strftime("%Y-%m-%d") if len(norm_index) > 0 else "N/A",
            norm_index.max().strftime("%Y-%m-%d") if len(norm_index) > 0 else "N/A",
        )
        
        # Reindex labels to panel dates, then forward-fill
        labels_reindexed = labels.reindex(norm_index).ffill()
        valid_labels = labels_reindexed["forward_return"].notna().sum()
        
        # 🔧 If still no valid labels, try date-only matching (ignore time component)
        if valid_labels == 0 and len(labels) > 0:
            self.logger.warning("⚠️ No label overlap - trying date-only matching...")
            # Convert both to date-only for matching
            labels_dates = labels.copy()
            labels_dates.index = pd.to_datetime(labels_dates.index.date)
            panel_dates = pd.to_datetime(norm_index.date)
            
            # Reindex by date
            labels_by_date = labels_dates.reindex(panel_dates).ffill()
            labels_by_date.index = norm_index  # Restore original index
            valid_labels = labels_by_date["forward_return"].notna().sum()
            
            if valid_labels > 0:
                self.logger.info("✅ Date-only matching recovered %d labels", valid_labels)
                labels_reindexed = labels_by_date
        
        self.logger.info(
            "🏷️ Labels constructed: price_df=%d rows, close range=%s to %s, index=%d rows, valid_labels=%d (%.1f%%)",
            len(price_df),
            close.index.min().strftime("%Y-%m-%d") if len(close) > 0 else "N/A",
            close.index.max().strftime("%Y-%m-%d") if len(close) > 0 else "N/A",
            len(norm_index),
            valid_labels,
            100.0 * valid_labels / len(norm_index) if len(norm_index) > 0 else 0,
        )
        
        # Restore original index to labels
        labels_reindexed.index = index
        return labels_reindexed

    def _get_price_data_for_horizon(self, horizon: int) -> Optional[pd.DataFrame]:
        """Retrieve cached price data for the given horizon."""
        return self._price_cache.get(horizon)

    def _run_tier_one_suite(
        self,
        tracks: Dict[str, TrackData],
        labels: pd.DataFrame,
        horizon: int,
    ) -> Tuple[Dict[str, TierOneResult], float, Optional[Dict[str, Any]]]:
        tier1: Dict[str, TierOneResult] = {}
        tree_scores: List[Tuple[float, str, str]] = []  # score, track, model
        best_tree_context: Optional[Dict[str, Any]] = None

        for track_name, track in tracks.items():
            result = self._run_models_for_track(track, labels, horizon)
            tier1[track_name] = result
            for model_name, score in result.composite_scores.items():
                if model_name in {"lightgbm", "xgboost"}:
                    tree_scores.append((score, track_name, model_name))
            if tree_scores:
                best_score, best_track, best_model = max(tree_scores, key=lambda x: x[0])
                pred_series = tier1[best_track].predictions.get(best_model)
                if pred_series is not None:
                    best_tree_context = {
                        "track": best_track,
                        "model": best_model,
                        "preds": pred_series,
                        "actuals": tier1[best_track].actuals.reindex(pred_series.index),
                        "n_samples": len(pred_series.dropna()),
                    }
        s_tree_best = max([score for score, _, _ in tree_scores], default=float("-inf"))
        return tier1, s_tree_best, best_tree_context

    def _collect_candidates(
        self,
        tier2_results: Dict[str, TierTwoResult],
        tracks: Dict[str, TrackData],
    ) -> Tuple[List[ModelCandidate], Dict[str, pd.Series]]:
        """Collect LSTM Tier-2 model candidates.
        
        🚀 LSTM-ONLY MODE: Only collects Tier-2 (LSTM) candidates.
        Tier-1 (LightGBM/XGBoost/ElasticNet) has been removed.
        """
        candidates: List[ModelCandidate] = []
        per_model_predictions: Dict[str, pd.Series] = {}

        for track_name, result in tier2_results.items():
            for model_name, preds in result.predictions.items():
                summary = next((m for m in result.top_models if m.name == model_name), result.best_model)
                if summary is None:
                    continue
                
                metadata = tracks.get(track_name).metadata if track_name in tracks else {}
                tags = list(metadata.get("tags", [])) if metadata else []
                if summary.params.get("view"):
                    tags = sorted(set(tags + [str(summary.params["view"]).lower()]))
                feature_names: Sequence[str] = summary.params.get("feature_names", [])
                feature_list = list(feature_names[:32]) if isinstance(feature_names, Sequence) else []
                if not feature_list:
                    feature_list = list(tags)
                
                reliability = self._candidate_reliability(summary.metrics)
                candidate = ModelCandidate(
                    track=track_name,
                    tier="tier2",
                    model_name=model_name,
                    model_type=summary.model_type,
                    task=result.task,
                    composite=summary.metrics.get("composite", float("nan")),
                    reliability=reliability,
                    metrics=summary.metrics,
                    predictions=preds,
                    features_used=feature_list,
                    tags=list(tags),
                    extras={"folds": summary.params.get("folds"), "tier": "tier2"},
                )
                label = f"{track_name}_tier2_{model_name}"
                per_model_predictions[label] = preds
                candidates.append(candidate)

        return candidates, per_model_predictions

    def _select_best_models(
        self,
        tier2_results: Dict[str, TierTwoResult],
        tracks: Dict[str, TrackData],
    ) -> Tuple[List[ModelCandidate], Optional[ModelCandidate], Dict[str, Any]]:
        """Select best LSTM models from Tier-2 results.
        
        🚀 LSTM-ONLY MODE: Simplified to only consider Tier-2 (LSTM) candidates.
        """
        candidates, per_model_predictions = self._collect_candidates(tier2_results, tracks)
        ranking = sorted(
            candidates,
            key=lambda c: (float("-inf") if not np.isfinite(c.composite) else c.composite),
            reverse=True,
        )
        primary = ranking[0] if ranking else None

        def pick_best(filter_fn):
            for cand in ranking:
                if filter_fn(cand):
                    return cand
            return None

        best_regression = pick_best(lambda c: c.task == "regression")
        best_classification = pick_best(lambda c: c.task == "classification")
        best_lstm_by_view: Dict[str, Optional[ModelCandidate]] = {}
        for view in ["seq_raw", "seq_hf", "seq_hybrid"]:
            best_lstm_by_view[view] = pick_best(
                lambda c, v=view: v in c.model_name.lower()
            )

        selection_meta = {
            "per_model_predictions": per_model_predictions,
            "best_regression": best_regression,
            "best_classification": best_classification,
            "best_lstm_by_view": best_lstm_by_view,
        }
        return ranking, primary, selection_meta

    def _validate_track_data_before_training(
        self,
        track: TrackData,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[bool, str]:
        """
        🔧 NEW: Validate track data quality before training tree models.
        
        Ensures:
        1. X has non-zero columns
        2. X has non-null values
        3. y has non-null values
        4. Sufficient rows for training
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check minimum rows
        min_rows = 100  # Minimum rows for meaningful training
        if len(X) < min_rows:
            errors.append(f"Insufficient rows: {len(X)} < {min_rows}")
        
        # Check feature columns
        if X.shape[1] == 0:
            errors.append("No feature columns in X")
        
        # Check for non-null columns
        non_null_cols = X.notna().any(axis=0).sum()
        if non_null_cols == 0:
            errors.append("All feature columns are null")
        elif non_null_cols < X.shape[1] * 0.5:
            errors.append(f"Only {non_null_cols}/{X.shape[1]} columns have non-null values")
        
        # Check for rows with all null values
        null_rows = X.isna().all(axis=1).sum()
        if null_rows > len(X) * 0.5:
            errors.append(f"{null_rows}/{len(X)} rows are all null")
        
        # Check target variable
        y_non_null = y.notna().sum()
        if y_non_null == 0:
            errors.append("Target variable y has no non-null values")
        elif y_non_null < len(y) * 0.5:
            errors.append(f"Only {y_non_null}/{len(y)} target values are non-null")
        
        if errors:
            error_msg = f"Track {track.name} validation failed: " + "; ".join(errors)
            self.logger.warning("❌ %s", error_msg)
            return False, error_msg
        
        # Log successful validation
        self.logger.info("✅ Track %s data validated: %d rows, %d cols (%d non-null), %d target values",
                        track.name, len(X), X.shape[1], non_null_cols, y_non_null)
        return True, ""

    def _run_models_for_track(
        self,
        track: TrackData,
        labels: pd.DataFrame,
        horizon: int,
    ) -> TierOneResult:
        frame = track.frame.sort_index()
        
        # 🔧 DEBUG: Log feature counts at each stage
        self.logger.debug("🔍 %s frame before join: %d rows × %d cols", track.name, len(frame), frame.shape[1])
        
        # 🔧 DEBUG: Check label quality before join
        labels_subset = labels[["forward_return", "forward_direction"]]
        label_non_null = labels_subset.notna().all(axis=1).sum()
        self.logger.info(
            "🔍 %s labels check: %d total rows, %d with valid labels (%.1f%%)",
            track.name,
            len(labels_subset),
            label_non_null,
            100.0 * label_non_null / len(labels_subset) if len(labels_subset) > 0 else 0,
        )
        
        dataset = frame.join(labels[["forward_return", "forward_direction"]]).dropna()
        
        self.logger.info("🔍 %s after dropna join: %d rows × %d cols (dropped %d rows)",
                         track.name, len(dataset), dataset.shape[1], len(frame) - len(dataset))
        
        if dataset.empty:
            self.logger.warning("❌ %s: Empty dataset after dropna! Check label alignment.", track.name)
            empty_preds: Dict[str, pd.Series] = {}
            return TierOneResult(
                track=track.name,
                task=track.metadata.get("task", "regression"),
                best_model=None,
                top_models=[],
                composite_scores={},
                predictions=empty_preds,
                actuals=pd.Series(dtype=float),
                model_summaries={},
            )

        task = track.metadata.get("task", "regression")
        actual_returns = dataset["forward_return"].astype(float)
        if task == "classification":
            actuals = dataset["forward_direction"].astype(float)
            y = (actuals > 0).astype(int)
        else:
            actuals = actual_returns
            y = actuals

        X = dataset[frame.columns]
        
        # 🔧 DEBUG: Check for constant columns in X that LightGBM will ignore
        nunique = X.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        varying_cols = nunique[nunique > 1].index.tolist()
        self.logger.info(
            "🔍 %s features: %d total, %d varying, %d constant (will be ignored by LightGBM)",
            track.name, X.shape[1], len(varying_cols), len(constant_cols)
        )
        if constant_cols and len(constant_cols) <= 10:
            self.logger.debug("Constant cols: %s", constant_cols)
        
        # 🔧 NEW: Validate track data before training
        is_valid, error_msg = self._validate_track_data_before_training(track, X, y)
        if not is_valid:
            self.logger.warning("⚠️ Skipping %s due to data validation failure", track.name)
            empty_preds: Dict[str, pd.Series] = {}
            return TierOneResult(
                track=track.name,
                task=task,
                best_model=None,
                top_models=[],
                composite_scores={},
                predictions=empty_preds,
                actuals=pd.Series(dtype=float),
                model_summaries={},
            )
        
        model_specs: List[Tuple[str, str]] = []
        if task == "regression":
            model_specs = [
                ("lightgbm", "tree"),
                ("xgboost", "tree"),
                ("elasticnet", "linear"),
            ]
        else:
            model_specs = [
                ("lightgbm", "tree"),
                ("xgboost", "tree"),
                ("logistic", "linear"),
            ]

        predictions: Dict[str, pd.Series] = {}
        summaries: List[ModelSummary] = []
        composite_scores: Dict[str, float] = {}
        model_summaries: Dict[str, ModelSummary] = {}

        for name, model_type in model_specs:
            try:
                summary, preds = self._train_model(
                    name,
                    model_type,
                    X,
                    y,
                    actuals,
                    actual_returns,
                    task,
                    horizon,
                )
            except Exception as exc:
                self.logger.warning("%s training failed on %s (%s): %s", name, track.name, task, exc)
                continue
            if summary is None or preds is None:
                continue
            summaries.append(summary)
            predictions[name] = preds
            composite_scores[name] = summary.metrics.get("composite", float("nan"))
            model_summaries[name] = summary

        best_model = None
        if summaries:
            best_model = max(summaries, key=lambda s: s.metrics.get("composite", float("-inf")))
        top_models = sorted(summaries, key=lambda s: s.metrics.get("composite", float("-inf")), reverse=True)[:3]

        return TierOneResult(
            track=track.name,
            task=task,
            best_model=best_model,
            top_models=top_models,
            composite_scores=composite_scores,
            predictions=predictions,
            actuals=actuals,
            model_summaries=model_summaries,
        )

    def _train_model(
        self,
        name: str,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        actuals: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
    ) -> Tuple[Optional[ModelSummary], Optional[pd.Series]]:
        if (
            model_type == "tree"
            and self.config.tree_optuna_backtest
            and optuna is not None
        ):
            summary, preds = self._train_tree_with_optuna(
                name,
                model_type,
                X,
                y,
                actuals,
                actual_returns,
                task,
                horizon,
            )
            if summary is not None and preds is not None:
                return summary, preds
        return self._train_model_standard(
            name,
            model_type,
            X,
            y,
            actuals,
            actual_returns,
            task,
            horizon,
        )

    def _train_model_standard(
        self,
        name: str,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        actuals: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
    ) -> Tuple[Optional[ModelSummary], Optional[pd.Series]]:
        folds = self._time_splits(len(X))
        if not folds:
            folds = [(np.arange(len(X)), np.arange(len(X)))]
        fold_scores: List[float] = []

        for train_idx, val_idx in folds:
            model = self._instantiate_model(name, task)
            if model is None:
                return None, None
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = self._predict(model, X.iloc[val_idx], task)
            fold_metrics = self._compute_metrics(
                preds,
                actuals.iloc[val_idx],
                actual_returns.iloc[val_idx],
                task,
                horizon,
            )
            fold_scores.append(fold_metrics["composite"])

        model = self._instantiate_model(name, task)
        if model is None:
            return None, None
        model.fit(X, y)
        preds_all = self._predict(model, X, task)

        metrics = self._compute_metrics(preds_all, actuals, actual_returns, task, horizon)
        metrics["cv_mean"] = float(np.nanmean(fold_scores)) if fold_scores else metrics["composite"]
        summary = ModelSummary(
            name=name,
            model_type=model_type,
            metrics=metrics,
            params={"task": task, "folds": len(fold_scores)},
        )
        return summary, pd.Series(preds_all, index=X.index)

    def _train_tree_with_optuna(
        self,
        name: str,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        actuals: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
    ) -> Tuple[Optional[ModelSummary], Optional[pd.Series]]:
        # 🔧 CRITICAL FIX: Pass DataFrame index for proper date-based split mapping
        folds = self._time_splits_with_index(len(X), X.index)
        if not folds:
            return self._train_model_standard(name, model_type, X, y, actuals, actual_returns, task, horizon)

        # 🚀 Use Ray Tune if enabled and available
        if (
            self.config.ray_tree_tuning_enabled
            and RAY_AVAILABLE
            and ray is not None
            and tune is not None
            and OptunaSearch is not None
        ):
            return self._train_tree_with_ray_tune(
                name, model_type, X, y, actuals, actual_returns, task, horizon, folds
            )
        
        # Fallback to native Optuna with n_jobs parallelism
        return self._train_tree_with_native_optuna(
            name, model_type, X, y, actuals, actual_returns, task, horizon, folds
        )

    def _train_tree_with_ray_tune(
        self,
        name: str,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        actuals: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[ModelSummary], Optional[pd.Series]]:
        """Train tree model using Ray Tune + OptunaSearch for distributed HPO.
        
        This method:
        1. Puts data in Ray object store for efficient sharing
        2. Uses OptunaSearch with ConcurrencyLimiter for smart parallel search
        3. Auto-detects optimal resources if configured
        4. Returns best model trained on full data
        """
        # Determine if this model uses GPU and set appropriate CPU allocation
        use_gpu = (name == "xgboost" and self.config.xgboost_params.get("device") == "cuda")
        
        # 🚀 Different CPU allocation: LightGBM=1 (80 parallel), XGBoost=2 (with GPU)
        if name == "lightgbm":
            cpus_per_trial = self.config.ray_lightgbm_cpus_per_trial  # Default: 1
        else:
            cpus_per_trial = self.config.ray_xgboost_cpus_per_trial  # Default: 2
        
        # Calculate optimal resources
        if self.config.ray_auto_detect_resources:
            resources = _calculate_optimal_ray_resources(
                gpu_fraction_per_trial=self.config.ray_gpu_fraction_per_trial if use_gpu else 0.0,
                cpus_per_trial=cpus_per_trial,
                use_gpu=use_gpu,
            )
            max_concurrent = self.config.ray_max_concurrent_trials or resources["max_concurrent"]
        else:
            max_concurrent = self.config.ray_max_concurrent_trials or self.config.tree_optuna_parallelism
            resources = {
                "cpus_per_trial": cpus_per_trial,
                "gpu_per_trial": self.config.ray_gpu_fraction_per_trial if use_gpu else 0.0,
                "resources_per_trial": {
                    "cpu": cpus_per_trial,
                    **({"gpu": self.config.ray_gpu_fraction_per_trial} if use_gpu else {}),
                },
            }
        
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        
        # Put data in Ray object store for efficient sharing across workers
        X_ref = ray.put(X)
        y_ref = ray.put(y)
        actuals_ref = ray.put(actuals)
        actual_returns_ref = ray.put(actual_returns)
        
        # Get base params and search space
        base_params = self.config.lightgbm_params.copy() if name == "lightgbm" else self.config.xgboost_params.copy()
        search_space = _get_optuna_search_space(name)
        
        if not search_space:
            self.logger.warning("No search space defined for %s, falling back to native Optuna", name)
            return self._train_tree_with_native_optuna(
                name, model_type, X, y, actuals, actual_returns, task, horizon, folds
            )
        
        # Create trainable function
        trainable = _create_ray_tree_trainable(
            X_ref=X_ref,
            y_ref=y_ref,
            actuals_ref=actuals_ref,
            actual_returns_ref=actual_returns_ref,
            folds=folds,
            model_name=name,
            task=task,
            horizon=horizon,
            base_params=base_params,
            lightgbm_available=(lgb is not None),
            xgboost_available=(xgb is not None),
        )
        
        # 🚀 Use Ray Tune's native search space for TRUE parallel execution
        # OptunaSearch was limiting parallelism even with n_startup_trials
        # Ray's native random search can run all trials in parallel
        # 🔧 OPTIMIZED: Tightened ranges to avoid over-regularization
        if name == "lightgbm":
            ray_search_space = {
                "num_leaves": tune.randint(16, 97),       # Was 129, cap at 96
                "max_depth": tune.randint(3, 9),          # Was 11, cap at 8
                "learning_rate": tune.loguniform(0.01, 0.10),
                "feature_fraction": tune.uniform(0.5, 0.95),  # Cap at 0.95
                "bagging_fraction": tune.uniform(0.6, 0.95),  # Min 0.6
                "bagging_freq": tune.randint(1, 8),       # Was 11, cap at 7
                "min_data_in_leaf": tune.randint(10, 51), # 🔧 Was 101, cap at 50
                "lambda_l1": tune.uniform(0.0, 2.0),      # 🔧 Was 5.0, cap at 2.0
                "lambda_l2": tune.uniform(0.0, 2.0),      # 🔧 Was 5.0, cap at 2.0
                "min_gain_to_split": tune.uniform(0.0, 0.3),  # 🔧 Was 1.0, cap at 0.3
                "n_estimators": tune.randint(300, 801),   # 🔧 Was 1201, cap at 800
            }
        else:  # xgboost
            ray_search_space = {
                "max_depth": tune.randint(3, 9),          # Was 11, cap at 8
                "eta": tune.loguniform(0.01, 0.10),
                "subsample": tune.uniform(0.6, 0.95),     # Min 0.6
                "colsample_bytree": tune.uniform(0.5, 0.95),
                "min_child_weight": tune.uniform(1.0, 5.0),   # 🔧 Was 10, cap at 5
                "reg_alpha": tune.uniform(0.0, 2.0),      # 🔧 Was 5.0, cap at 2.0
                "reg_lambda": tune.uniform(0.0, 2.0),     # 🔧 Was 5.0, cap at 2.0
                "gamma": tune.uniform(0.0, 1.0),          # 🔧 Was 5.0, cap at 1.0
                "n_estimators": tune.randint(300, 801),   # 🔧 Was 1201, cap at 800
            }
        
        self.logger.info(
            "🚀 Ray Tune: %s with %d trials, %d concurrent (GPU=%.2f, CPU=%d per trial)",
            name,
            self.config.tree_optuna_trials,
            max_concurrent,
            resources.get("gpu_per_trial", 0.0),
            resources.get("cpus_per_trial", 2),
        )
        
        # Run Ray Tune with native random search (TRUE parallel execution)
        try:
            tuner = tune.Tuner(
                tune.with_resources(trainable, resources=resources["resources_per_trial"]),
                tune_config=tune.TuneConfig(
                    num_samples=self.config.tree_optuna_trials,
                    max_concurrent_trials=max_concurrent,  # 🚀 Enable parallel trial execution
                ),
                param_space=ray_search_space,
            )
            results = tuner.fit()
            
            # Get best result
            best_result = results.get_best_result(metric="mean_composite", mode="max")
            if best_result is None or best_result.config is None:
                self.logger.warning("Ray Tune failed to find best result, falling back to native Optuna")
                return self._train_tree_with_native_optuna(
                    name, model_type, X, y, actuals, actual_returns, task, horizon, folds
                )
            
            best_params = best_result.config
            best_score = best_result.metrics.get("mean_composite", float("nan"))
            
            self.logger.info("🏆 Ray Tune best: %s score=%.4f", name, best_score)
            
        except Exception as e:
            self.logger.warning("Ray Tune failed (%s), falling back to native Optuna", str(e))
            return self._train_tree_with_native_optuna(
                name, model_type, X, y, actuals, actual_returns, task, horizon, folds
            )
        
        # Train final model with best params on full data
        final_params = base_params.copy()
        final_params.update(best_params)
        model = self._instantiate_model(name, task, params_override=final_params)
        if model is None:
            return None, None
        model.fit(X, y)
        preds_all = self._predict(model, X, task)
        metrics = self._compute_metrics(preds_all, actuals, actual_returns, task, horizon)
        metrics["cv_mean"] = best_score
        
        summary = ModelSummary(
            name=name,
            model_type=model_type,
            metrics=metrics,
            params={
                "task": task,
                "folds": len(folds),
                "optuna_trials": self.config.tree_optuna_trials,
                "ray_tune": True,
                "max_concurrent": max_concurrent,
            },
        )
        return summary, pd.Series(preds_all, index=X.index)

    def _train_tree_with_native_optuna(
        self,
        name: str,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        actuals: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Optional[ModelSummary], Optional[pd.Series]]:
        """Train tree model using native Optuna with n_jobs parallelism (fallback)."""
        
        def objective(trial: "optuna.trial.Trial") -> float:
            params = self._sample_tree_params(name, trial)
            fold_scores: List[float] = []
            for train_idx, val_idx in folds:
                model = self._instantiate_model(name, task, params_override=params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = self._predict(model, X.iloc[val_idx], task)
                metrics = self._compute_metrics(
                    preds,
                    actuals.iloc[val_idx],
                    actual_returns.iloc[val_idx],
                    task,
                    horizon,
                )
                fold_scores.append(metrics.get("composite", float("nan")))
            score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            trial.set_user_attr("mean_composite", score)
            trial.set_user_attr("params", params)
            return score

        study = optuna.create_study(direction="maximize")  # type: ignore[union-attr]
        
        # 🚀 PARALLEL TRIALS: Use n_jobs to run multiple trials concurrently
        # For GPU models (XGBoost), each trial uses ~100-200MB VRAM, so we can run
        # multiple in parallel on the same GPU (8GB VRAM / 200MB ≈ 40 parallel trials max)
        # For CPU models (LightGBM), use all available cores
        n_parallel = min(self.config.tree_optuna_parallelism, self.config.tree_optuna_trials)
        if n_parallel <= 0:
            n_parallel = -1  # Use all available cores
        
        self.logger.info(
            "🚀 Optuna tuning %s with %d trials, %d parallel workers",
            name, self.config.tree_optuna_trials, n_parallel if n_parallel > 0 else "all"
        )
        
        study.optimize(
            objective,
            n_trials=self.config.tree_optuna_trials,
            timeout=self.config.tree_optuna_timeout,
            show_progress_bar=False,
            n_jobs=n_parallel,  # 🚀 Enable parallel trial execution
        )
        if study.best_trial is None:
            return self._train_model_standard(name, model_type, X, y, actuals, actual_returns, task, horizon)
        best_params = study.best_trial.user_attrs.get("params")
        model = self._instantiate_model(name, task, params_override=best_params)
        if model is None:
            return None, None
        model.fit(X, y)
        preds_all = self._predict(model, X, task)
        metrics = self._compute_metrics(preds_all, actuals, actual_returns, task, horizon)
        metrics["cv_mean"] = float(study.best_value)
        summary = ModelSummary(
            name=name,
            model_type=model_type,
            metrics=metrics,
            params={
                "task": task,
                "folds": len(folds),
                "optuna_trials": len(study.trials),
            },
        )
        return summary, pd.Series(preds_all, index=X.index)

    def _sample_tree_params(self, name: str, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        """Sample hyperparameters for tree models using Optuna.
        
        Search spaces are optimized for financial time series:
        - LightGBM: includes min_gain_to_split for drift regime stability
        - XGBoost: includes gamma for split control under volatility
        """
        if name == "lightgbm":
            params = self.config.lightgbm_params.copy()
            params.update(
                {
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),  # Bagging frequency
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                    "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                    "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),  # Critical for drift regimes
                    "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                }
            )
            return params
        if name == "xgboost":
            params = self.config.xgboost_params.copy()
            params.update(
                {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "eta": trial.suggest_float("eta", 0.01, 0.10, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),  # Split control under volatility
                    "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                }
            )
            return params
        return {}

    def _instantiate_model(self, name: str, task: str, params_override: Optional[Dict[str, Any]] = None):
        if name == "lightgbm":
            if lgb is None:
                raise RuntimeError("lightgbm not installed")
            params = self.config.lightgbm_params.copy()
            if params_override:
                params.update(params_override)
            if task == "classification":
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)
        if name == "xgboost":
            if xgb is None:
                raise RuntimeError("xgboost not installed")
            params = self.config.xgboost_params.copy()
            if params_override:
                params.update(params_override)
            if task == "classification":
                return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
            return xgb.XGBRegressor(**params)
        if name == "elasticnet":
            params = self.config.elasticnet_params
            return ElasticNet(l1_ratio=params["l1_ratio"], alpha=params["alpha"], max_iter=params["max_iter"], random_state=self.config.random_state)
        if name == "logistic":
            params = self.config.logistic_params
            return LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=params["l1_ratio"],
                C=params["C"],
                max_iter=params["max_iter"],
                random_state=self.config.random_state,
            )
        raise ValueError(f"Unknown model {name}")

    def _predict(self, model, X: pd.DataFrame, task: str) -> np.ndarray:
        if task == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def _time_splits_with_index(
        self, n_samples: int, data_index: pd.Index
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-based CV splits with proper date mapping.
        
        🔧 CRITICAL FIX: When _forced_splits is set from walk-forward windows,
        the indices are positional from the ORIGINAL panel. After dropna(),
        rows may be removed causing position shifts.
        
        This method uses the window metadata's train_end date to properly split:
        1. Get train_end date from _current_window_meta
        2. Create mask: rows <= train_end are train, rows > train_end are validation
        3. Return new positional indices that correctly correspond to train/val dates
        
        This prevents data leakage from incorrect train/val splits.
        """
        # 🔧 Use window metadata dates for splitting (PREFERRED)
        if self._current_window_meta is not None:
            train_end_str = self._current_window_meta.get("train_end")
            if train_end_str:
                try:
                    # Parse train_end date from window metadata
                    train_end = pd.Timestamp(train_end_str)
                    if train_end.tz is not None:
                        train_end = train_end.tz_localize(None)
                    
                    # Convert index to DatetimeIndex
                    ts_index = pd.DatetimeIndex(data_index)
                    if ts_index.tz is not None:
                        ts_index = ts_index.tz_localize(None)
                    
                    # Create date-based mask: train = dates <= train_end
                    train_mask = ts_index <= train_end
                    new_train = np.where(train_mask)[0]
                    new_val = np.where(~train_mask)[0]
                    
                    if len(new_train) > 0 and len(new_val) > 0:
                        self.logger.info(
                            "🛡️ DATE-BASED walk-forward split using train_end=%s: "
                            "train=%d (%.1f%%), val=%d (%.1f%%) of %d total samples",
                            train_end_str,
                            len(new_train), 100.0 * len(new_train) / n_samples,
                            len(new_val), 100.0 * len(new_val) / n_samples,
                            n_samples
                        )
                        return [(new_train, new_val)]
                    else:
                        self.logger.warning(
                            "⚠️ Date-based split produced empty train=%d or val=%d, "
                            "falling back to ratio-based split",
                            len(new_train), len(new_val)
                        )
                except Exception as e:
                    self.logger.warning("⚠️ Date-based split failed: %s, falling back", str(e))
        
        # 🔧 FALLBACK: Use ratio from _forced_splits if window meta not available
        if self._forced_splits is not None:
            valid_splits: List[Tuple[np.ndarray, np.ndarray]] = []
            
            for train_idx, val_idx in self._forced_splits:
                # Use the original train/val split ratio to determine the split point
                total_orig = len(train_idx) + len(val_idx)
                train_ratio = len(train_idx) / total_orig if total_orig > 0 else 0.8
                
                # Apply the same ratio to the new dataset
                split_point = int(n_samples * train_ratio)
                split_point = max(1, min(split_point, n_samples - 1))
                
                new_train = np.arange(0, split_point)
                new_val = np.arange(split_point, n_samples)
                
                if len(new_train) > 0 and len(new_val) > 0:
                    valid_splits.append((new_train, new_val))
                    self.logger.info(
                        "🛡️ RATIO-BASED walk-forward split (fallback): "
                        "train=%d (%.1f%%), val=%d (%.1f%%) of %d total samples",
                        len(new_train), 100.0 * len(new_train) / n_samples,
                        len(new_val), 100.0 * len(new_val) / n_samples,
                        n_samples
                    )
            
            if valid_splits:
                return valid_splits
            
            self.logger.warning("⚠️ Could not create valid splits, falling back to TimeSeriesSplit")
        
        # Fallback to standard TimeSeriesSplit
        return self._time_splits(n_samples)

    def _time_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self._forced_splits is not None:
            # 🔧 CRITICAL FIX: Validate that forced splits indices are within bounds
            # After dropna(), the dataset may have fewer rows than original panel
            valid_splits: List[Tuple[np.ndarray, np.ndarray]] = []
            for train_idx, val_idx in self._forced_splits:
                # Filter out indices that are >= n_samples (out of bounds)
                train_valid = train_idx[train_idx < n_samples]
                val_valid = val_idx[val_idx < n_samples]
                
                if len(train_valid) > 0 and len(val_valid) > 0:
                    valid_splits.append((train_valid, val_valid))
                else:
                    # If indices don't work, create proportional splits based on original ratio
                    orig_train_ratio = len(train_idx) / (len(train_idx) + len(val_idx))
                    train_end = int(n_samples * orig_train_ratio)
                    train_end = max(1, min(train_end, n_samples - 1))
                    
                    new_train = np.arange(0, train_end)
                    new_val = np.arange(train_end, n_samples)
                    
                    if len(new_train) > 0 and len(new_val) > 0:
                        valid_splits.append((new_train, new_val))
                        self.logger.debug(
                            "🔧 Remapped forced splits: orig train=%d val=%d → new train=%d val=%d (n_samples=%d)",
                            len(train_idx), len(val_idx), len(new_train), len(new_val), n_samples
                        )
            
            if valid_splits:
                return valid_splits
            # Fall through to default splits if forced splits couldn't be used
            self.logger.warning("⚠️ Forced splits invalid for n_samples=%d, using default splits", n_samples)
        
        if n_samples < 30:
            return []
        n_splits = max(2, min(self.config.tabular_folds, n_samples // 60))
        splitter = TimeSeriesSplit(n_splits=n_splits)
        indices = np.arange(n_samples)
        return [(indices[train], indices[val]) for train, val in splitter.split(indices)]

    def _build_sequence_folds(self, n_samples: int, horizon: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build folds for LSTM sequence training.
        
        🔧 CRITICAL: When _forced_splits is set, we need to MAP panel indices to sequence indices.
        Sequence data has different length than panel due to windowing (seq_len lookback).
        
        For walk-forward mode:
        - _forced_splits contains panel indices
        - We need to convert to sequence indices based on the seq_len offset
        """
        # NOTE: _forced_splits are panel indices, but n_samples is sequence count
        # The sequence at index i corresponds to panel index (i + seq_len)
        # So we cannot directly use _forced_splits here - build fresh folds for seq data
        
        # In walk-forward mode with single window, use simple train/val split on seq data
        if self._forced_splits is not None and len(self._forced_splits) == 1:
            # Single forced split - use simple proportion for sequence data
            val_ratio = 0.25  # Use ~25% for validation
            pivot = max(5, int(n_samples * (1 - val_ratio)))
            train_idx = np.arange(0, pivot)
            val_idx = np.arange(pivot, n_samples)
            self.logger.debug(
                "🛡️ LSTM walk-forward mode: seq train=%d, seq val=%d (total=%d)",
                len(train_idx), len(val_idx), n_samples
            )
            return [(train_idx, val_idx)]
        
        if n_samples <= horizon or n_samples < 20:
            return []
        val_span = max(5, min(horizon, n_samples // 4))
        step = max(1, val_span // 2)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        end = val_span
        while end < n_samples:
            train_end = end - val_span
            if train_end <= 0:
                end += step
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, min(train_end + val_span, n_samples))
            folds.append((train_idx, val_idx))
            if len(folds) >= max(1, self.config.max_lstm_folds):
                break
            end += step
        return folds

    # ------------------------------------------------------------------
    # STEP 7: Regime Detection and Threshold Application
    # ------------------------------------------------------------------
    def _detect_regime_from_returns(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Detect market regime from returns.
        
        STEP 7: Regime is an INPUT FEATURE (not optimization target).
        Uses rolling statistics to classify each timestamp:
        - 0: Bull (positive returns, low volatility)
        - 1: Bear (negative returns, moderate volatility)
        - 2: Crisis (high volatility, regardless of direction)
        
        Args:
            returns: Array of returns
            window: Rolling window for statistics
            
        Returns:
            Array of regime labels (0=bull, 1=bear, 2=crisis)
        """
        if len(returns) < window:
            # Not enough data - assume bull
            return np.zeros(len(returns), dtype=int)
        
        # Pad for rolling calculation
        returns_series = pd.Series(returns)
        
        # Rolling statistics
        roll_mean = returns_series.rolling(window=window, min_periods=1).mean()
        roll_vol = returns_series.rolling(window=window, min_periods=1).std()
        
        # Volatility percentiles for regime classification
        vol_median = roll_vol.median()
        vol_high = roll_vol.quantile(0.85)
        
        regime = np.zeros(len(returns), dtype=int)
        
        for i in range(len(returns)):
            vol = roll_vol.iloc[i] if not np.isnan(roll_vol.iloc[i]) else vol_median
            ret = roll_mean.iloc[i] if not np.isnan(roll_mean.iloc[i]) else 0
            
            if vol > vol_high:
                # Crisis: high volatility
                regime[i] = 2
            elif ret < 0:
                # Bear: negative trend
                regime[i] = 1
            else:
                # Bull: positive/neutral trend, normal volatility
                regime[i] = 0
        
        return regime
    
    def _apply_regime_thresholds(
        self,
        predictions: np.ndarray,
        regime: np.ndarray,
        threshold: float,
        bull_mult: float,
        bear_mult: float,
        crisis_mult: float,
    ) -> np.ndarray:
        """Apply regime-aware thresholds to get directional signals.
        
        STEP 7 Final Direction Logic:
        - |pred| < T_regime → 0 (do nothing / neutral)
        - pred >= T_regime → +1 (long)
        - pred <= -T_regime → -1 (short)
        
        Args:
            predictions: Raw model predictions
            regime: Regime labels (0=bull, 1=bear, 2=crisis)
            threshold: Base threshold T
            bull_mult: Multiplier for bull regime
            bear_mult: Multiplier for bear regime
            crisis_mult: Multiplier for crisis regime
            
        Returns:
            Directional signals: -1, 0, or +1
        """
        # Compute regime-specific thresholds
        T_bull = threshold * bull_mult
        T_bear = threshold * bear_mult
        T_crisis = threshold * crisis_mult
        
        # Map regime to threshold
        regime_thresholds = np.where(
            regime == 0, T_bull,
            np.where(regime == 1, T_bear, T_crisis)
        )
        
        # Apply threshold logic
        signals = np.where(
            predictions >= regime_thresholds, 1.0,
            np.where(predictions <= -regime_thresholds, -1.0, 0.0)
        )
        
        return signals

    # ------------------------------------------------------------------
    # Metrics & scoring
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        preds: Sequence[float],
        actual_targets: Sequence[float],
        actual_returns: Sequence[float],
        task: str,
        horizon: int,
    ) -> Dict[str, float]:
        actual_ret_series = actual_returns if isinstance(actual_returns, pd.Series) else pd.Series(actual_returns)
        preds_series = preds if isinstance(preds, pd.Series) else pd.Series(preds, index=actual_ret_series.index)
        actual_target_series = (
            actual_targets
            if isinstance(actual_targets, pd.Series)
            else pd.Series(actual_targets, index=actual_ret_series.index)
        )
        preds_series = preds_series.reindex(actual_ret_series.index)
        actual_target_series = actual_target_series.reindex(actual_ret_series.index)
        inline_metrics = self._inline_backtest_metrics(preds_series, actual_target_series, actual_ret_series, task, horizon)
        if inline_metrics is not None:
            return inline_metrics

        preds_arr = np.asarray(preds_series, dtype=float)
        actuals_arr = np.asarray(actual_target_series, dtype=float)
        if task == "classification":
            strategy = (preds_arr - 0.5) * np.sign(actuals_arr)
        else:
            # ===================================================================
            # STEP 7: Regime-Aware Threshold Direction Logic
            # ===================================================================
            if self.config.regime_aware_thresholds_enabled:
                # Detect regime from actual returns
                regime = self._detect_regime_from_returns(actuals_arr)
                
                # Apply regime-specific thresholds
                direction = self._apply_regime_thresholds(
                    preds_arr,
                    regime,
                    self.config.regime_threshold_base,
                    self.config.regime_threshold_bull_mult,
                    self.config.regime_threshold_bear_mult,
                    self.config.regime_threshold_crisis_mult,
                )
            else:
                # Legacy: simple threshold-based direction
                upper_thresh = self.config.signal_threshold_upper
                lower_thresh = self.config.signal_threshold_lower
                
                direction = np.where(
                    preds_arr >= upper_thresh, 1.0,
                    np.where(preds_arr <= lower_thresh, -1.0, 0.0)
                )
            
            # strategy = direction * actual_return (neutral positions contribute 0)
            strategy = direction * actuals_arr
        sharpe = self._sharpe_ratio(strategy, horizon)
        rwa = self._rwa(preds_arr, actuals_arr, task)
        calib = self._normalized_calibration_error(preds_arr, actuals_arr, task)
        stability = self._stability(preds_arr)
        composite = 0.6 * sharpe + 0.15 * rwa + 0.15 * (1 - calib) + 0.1 * stability
        return {
            "sharpe": sharpe,
            "rwa": rwa,
            "norm_calib_err": calib,
            "stability": stability,
            "composite": composite,
        }

    def _candidate_reliability(self, metrics: Dict[str, float]) -> float:
        cal = metrics.get("norm_calib_err", 0.5)
        stability = metrics.get("stability", 0.0)
        return float(np.clip(0.6 * (1 - cal) + 0.4 * stability, 0.0, 1.0))

    def _aggregate_fold_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {
                "sharpe": 0.0,
                "rwa": 0.0,
                "norm_calib_err": 1.0,
                "stability": 0.0,
                "composite": 0.0,
            }
        keys = set().union(*metrics_list)
        aggregated: Dict[str, float] = {}
        for key in keys:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if not values:
                continue
            aggregated[key] = float(np.nanmean(values))
        return aggregated

    def _inline_backtest_metrics(
        self,
        preds: pd.Series,
        actual_targets: pd.Series,
        actual_returns: pd.Series,
        task: str,
        horizon: int,
    ) -> Optional[Dict[str, float]]:
        price_data = self._get_price_data_for_horizon(horizon)
        if price_data is None or price_data.empty:
            return None
        strategy_cfg = dict(self.config.strategy or {})
        fee = float(strategy_cfg.get("fee_bp", 0.0)) / 10000.0
        slippage = float(strategy_cfg.get("slippage_bp", 0.0)) / 10000.0
        engine = BacktestEngine(price_data, fee=fee, slippage_bp=slippage)
        idx = actual_returns.index
        preds_df = pd.DataFrame(index=idx)
        preds_df["actual_return"] = actual_returns.astype(float)
        preds_df = preds_df.dropna(subset=["actual_return"])
        if preds_df.empty:
            return None
        preds_aligned = preds.reindex(preds_df.index).astype(float).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        if task == "classification":
            probs = preds_aligned.clip(0.0, 1.0)
            preds_df["p_up"] = probs
            preds_df["mu_hat"] = (probs - 0.5) * DEFAULT_RETURN_SCALE
        else:
            preds_df["mu_hat"] = preds_aligned
            denom = actual_returns.reindex(preds_df.index).abs().rolling(window=max(5, horizon)).std().fillna(DEFAULT_RETURN_SCALE)
            logits = (preds_aligned / denom.replace(0.0, np.nan).fillna(DEFAULT_RETURN_SCALE)).clip(-8, 8)
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds_df["p_up"] = probs
        sigma_proxy = actual_returns.reindex(preds_df.index).rolling(window=max(5, horizon)).std().fillna(DEFAULT_RETURN_SCALE)
        preds_df["sigma_hat"] = sigma_proxy.clip(lower=1e-4)
        preds_df["rho"] = 1.0
        preds_df["drift_flag"] = 0.0
        try:
            # 🔧 FIX: Pass symbol so backtest can load Optuna regime thresholds
            _, metrics = engine.run(preds_df, horizon, strategy_cfg, symbol=self.config.symbol)
        except Exception as exc:  # pragma: no cover - diagnostics only
            self.logger.debug("Inline backtest metrics failed for %s H%s: %s", self.config.symbol, horizon, exc)
            return None
        return {
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "rwa": float(metrics.get("rwa", 0.0)),
            "norm_calib_err": float(metrics.get("ece", 0.0)),
            "stability": float(metrics.get("stability", 0.0)),
            "composite": float(metrics.get("composite", 0.0)),
        }

    def _candidate_to_mu(self, candidate: Optional[ModelCandidate]) -> Optional[pd.Series]:
        if candidate is None:
            return None
        preds = candidate.predictions
        if candidate.task == "regression":
            return preds
        prob = preds.clip(0.0, 1.0)
        return (prob - 0.5) * DEFAULT_RETURN_SCALE

    def _candidate_to_prob(
        self,
        candidate: Optional[ModelCandidate],
        vol_proxy: Optional[pd.Series] = None,
    ) -> Optional[pd.Series]:
        if candidate is None:
            return None
        preds = candidate.predictions
        if candidate.task == "classification":
            return preds.clip(0.0, 1.0)
        mu = preds
        if vol_proxy is not None and not vol_proxy.empty:
            vol = vol_proxy.reindex(mu.index).fillna(method="ffill").fillna(method="bfill").fillna(0.02)
            denom = (vol.abs() + 1e-3).clip(lower=1e-3)
        else:
            denom = pd.Series(DEFAULT_RETURN_SCALE, index=mu.index)
        logits = (mu / denom).clip(-8, 8)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return pd.Series(probs, index=mu.index)

    def _extract_mu_series(
        self,
        primary: Optional[ModelCandidate],
        selection_meta: Dict[str, Any],
    ) -> Optional[pd.Series]:
        mu = self._candidate_to_mu(primary)
        if mu is not None:
            return mu
        fallback = selection_meta.get("best_regression")
        return self._candidate_to_mu(fallback)

    def _extract_prob_series(
        self,
        primary: Optional[ModelCandidate],
        selection_meta: Dict[str, Any],
        vol_proxy: pd.Series,
    ) -> Optional[pd.Series]:
        probs = self._candidate_to_prob(primary, vol_proxy)
        if probs is not None:
            return probs
        fallback = selection_meta.get("best_classification")
        return self._candidate_to_prob(fallback, vol_proxy)

    def _estimate_reliability_series(
        self,
        track_b_frame: pd.DataFrame,
        primary: Optional[ModelCandidate],
    ) -> pd.Series:
        if not track_b_frame.empty:
            index = track_b_frame.index
        elif primary is not None:
            index = primary.predictions.index
        else:
            index = pd.Index([])
        if index.empty:
            return pd.Series(dtype=float)
        base = float(primary.reliability if primary else 0.5)
        cal_component = (1.0 - track_b_frame.get("cal_ECE", pd.Series(0.5, index=index))).reindex(index).fillna(0.5)
        drift_component = (1.0 - track_b_frame.get("drift_severity", pd.Series(0.0, index=index))).reindex(index).fillna(1.0).clip(0.0, 1.0)
        alpha_component = track_b_frame.get("alpha_weight", pd.Series(0.5, index=index)).reindex(index).fillna(0.5).clip(0.0, 1.0)
        logits = (
            self.config.reliability_bias
            + self.config.reliability_cal_weight * cal_component
            + self.config.reliability_drift_weight * drift_component
            + self.config.reliability_alpha_weight * alpha_component
            + (base - 0.5)
        )
        reliability = 1.0 / (1.0 + np.exp(-logits))
        return pd.Series(reliability, index=index).clip(0.0, 1.0)

    def _sharpe_ratio(self, strategy_returns: Sequence[float], horizon: int) -> float:
        """Compute Sharpe ratio with non-overlapping samples for horizon > 1.
        
        🔧 CRITICAL FIX: For multi-day horizons, overlapping returns cause
        autocorrelation that deflates std and inflates Sharpe by ~sqrt(horizon).
        Sample every H-th return to get independent observations.
        
        Returns:
            Annualized Sharpe ratio capped at ±5 to prevent noisy outliers.
        """
        MAX_SHARPE = 5.0  # Cap to prevent extreme values from noisy low-sample estimates
        
        arr = np.asarray(strategy_returns, dtype=float)
        if arr.size == 0:
            return 0.0
        
        # 🔧 FIX: For horizons > 1, use non-overlapping samples
        if horizon > 1 and len(arr) >= horizon:
            non_overlapping = arr[::horizon]
            if len(non_overlapping) < 2 or np.allclose(non_overlapping.std(), 0):
                return 0.0
            periods_per_year = max(1, 252 // horizon)
            scale = np.sqrt(periods_per_year)
            raw_sharpe = float(np.mean(non_overlapping) / (np.std(non_overlapping) + 1e-9) * scale)
            return float(np.clip(raw_sharpe, -MAX_SHARPE, MAX_SHARPE))
        else:
            # Daily horizon or insufficient samples
            if np.allclose(arr.std(), 0):
                return 0.0
            scale = np.sqrt(max(1, 252 // max(1, horizon)))
            raw_sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * scale)
            return float(np.clip(raw_sharpe, -MAX_SHARPE, MAX_SHARPE))

    def _rwa(self, preds: np.ndarray, actuals: np.ndarray, task: str) -> float:
        """Return-Weighted Accuracy using threshold-based direction.
        
        STEP 7: Uses regime-aware thresholds when enabled.
        Only counts trades where prediction exceeds threshold.
        """
        if preds.size == 0:
            return 0.0
        if task == "classification":
            direction = np.sign(actuals)
            hits = (np.where(preds >= 0.5, 1.0, -1.0) == np.sign(direction)).astype(float)
            return float(np.mean(hits))
        
        # ===================================================================
        # STEP 7: Regime-Aware Threshold Direction for RWA
        # ===================================================================
        if self.config.regime_aware_thresholds_enabled:
            # Detect regime and apply regime-specific thresholds
            regime = self._detect_regime_from_returns(actuals)
            pred_direction = self._apply_regime_thresholds(
                preds,
                regime,
                self.config.regime_threshold_base,
                self.config.regime_threshold_bull_mult,
                self.config.regime_threshold_bear_mult,
                self.config.regime_threshold_crisis_mult,
            )
        else:
            # Legacy: simple threshold-based direction
            upper_thresh = self.config.signal_threshold_upper
            lower_thresh = self.config.signal_threshold_lower
            
            pred_direction = np.where(
                preds >= upper_thresh, 1.0,
                np.where(preds <= lower_thresh, -1.0, 0.0)
            )
        
        actual_direction = np.sign(actuals)
        
        # Only count accuracy for trades we actually take (non-zero direction)
        trade_mask = pred_direction != 0
        if not trade_mask.any():
            return 0.5  # No trades = neutral accuracy
        
        hits = (pred_direction[trade_mask] == actual_direction[trade_mask]).astype(float)
        return float(np.mean(hits))

    def _normalized_calibration_error(self, preds: np.ndarray, actuals: np.ndarray, task: str) -> float:
        if preds.size == 0:
            return 1.0
        if task == "classification":
            actual_binary = (actuals > 0).astype(float)
            mae = np.mean(np.abs(preds - actual_binary))
            return float(np.clip(mae / 0.5, 0.0, 1.0))
        mae = np.mean(np.abs(preds - actuals))
        denom = np.std(actuals) + 1e-6
        return float(np.clip(mae / (denom if denom else 1.0), 0.0, 1.0))

    def _stability(self, preds: np.ndarray) -> float:
        if preds.size == 0:
            return 0.0
        return float(1.0 - np.clip(np.std(preds) / (np.abs(np.mean(preds)) + 1e-6), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 🚀 Optuna Pre-Optimization Methods
    # ------------------------------------------------------------------
    def _run_optuna_optimization(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        priors: Dict[str, float],
        horizon: int,
    ) -> Optional["OptimizedParams"]:
        """Run Optuna optimization for feature selection and LSTM hyperparameters.
        
        ONE TRIAL = ENTIRE WALK-FORWARD (~60 folds)
        
        This runs BEFORE LSTM training to optimize:
        - Tier 1: Family inclusion (which Stage-A families to use)
        - Tier 2: Per-family dimensionality (PCA/AE latent dims)
        - Tier 3: Track-A vs Track-B weighting
        - Tier 4: LSTM hyperparameters
        
        Encoders are fitted ONCE on first fold's training data and cached
        for all subsequent folds within a trial.
        
        Returns:
            OptimizedParams if optimization successful, None otherwise
        """
        if not OPTUNA_OPTIMIZER_AVAILABLE:
            self.logger.warning("Optuna optimizer not available - skipping optimization")
            return None
        
        # Check for cached params
        cache_tag = "GLOBAL13" if bool(getattr(self.config, "optuna_global_multi_symbol", False)) else self.config.symbol
        cache_path = self.config.optuna_params_cache_dir / f"{cache_tag}_h{horizon}_optuna.json"
        if self.config.optuna_use_cached_params and cache_path.exists():
            try:
                self.logger.info("📂 Loading cached Optuna params from %s", cache_path)
                optimizer = StageBOptunaOptimizer()
                params = optimizer.load_params(cache_path)
                if params and params.best_score > 0:
                    self.logger.info(
                        "✅ Using cached Optuna params (score=%.4f, trial=%d)",
                        params.best_score, params.trial_number
                    )
                    return params
            except Exception as e:
                self.logger.warning("Failed to load cached params: %s", e)

        # ===========================================================
        # GLOBAL MULTI-SYMBOL OPTUNA MODE
        # ===========================================================
        if bool(getattr(self.config, "optuna_global_multi_symbol", False)):
            if self.config.prep is None or not self.config.prep.enabled:
                self.logger.warning(
                    "❌ Global multi-symbol Optuna requires prep_families window manifests (config.prep.enabled=True)"
                )
                return None

            symbols = [str(s).upper() for s in (getattr(self.config, "optuna_global_symbols", GLOBAL_OPTUNA_SYMBOLS) or GLOBAL_OPTUNA_SYMBOLS)]
            self.logger.info(
                "🌍 Global multi-symbol Optuna enabled: %d symbols (%s ...)",
                len(symbols),
                ", ".join(symbols[:5]),
            )

            # Build per-symbol panels + labels (aligned to valid label coverage).
            # Keep column intersection so Track A/B encoders remain consistent.
            from dataclasses import replace as _replace

            pipelines: Dict[str, "StageBPipeline"] = {}
            panels_by_symbol: Dict[str, pd.DataFrame] = {}
            labels_by_symbol: Dict[str, pd.DataFrame] = {}
            block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]] = {}

            # Stage-A weights: average across symbols when available; fallback to the anchor priors.
            priors_accum: Dict[str, List[float]] = {}

            for sym in symbols:
                pipelines[sym] = StageBPipeline(_replace(self.config, symbol=sym))
                sym_panel = pipelines[sym]._build_panel(horizon)
                sym_labels = pipelines[sym]._construct_labels(horizon, sym_panel.index)

                valid_mask = sym_labels["forward_return"].notna()
                if int(valid_mask.sum()) == 0:
                    self.logger.warning("❌ Global Optuna skipped: no valid labels for %s H%d", sym, horizon)
                    return None

                sym_panel = sym_panel.loc[valid_mask]
                sym_labels = sym_labels.loc[valid_mask]

                panels_by_symbol[sym] = sym_panel
                labels_by_symbol[sym] = sym_labels

                try:
                    priors_map, _meta_map = pipelines[sym]._load_stage_a_artifacts()
                    sym_priors = priors_map.get(horizon, priors_map.get("global", {})) or {}
                    for k, v in sym_priors.items():
                        try:
                            priors_accum.setdefault(str(k), []).append(float(v))
                        except Exception:
                            continue
                except Exception:
                    # Non-fatal; fallback later
                    pass

            # Align to anchor symbol columns for consistent encoders across symbols.
            # IMPORTANT: Do NOT intersect columns across symbols. Intersection will drop
            # entire families if any symbol lacks them (e.g. SPY missing fin_g2..fin_g7),
            # which incorrectly forces Optuna weights for those families to 0.
            anchor_cols = list(panels_by_symbol[symbols[0]].columns)

            for sym in symbols:
                panels_by_symbol[sym] = panels_by_symbol[sym].reindex(columns=anchor_cols, fill_value=0.0)
                # Recompute summaries on the aligned panel to keep indices/columns consistent.
                block_summaries_by_symbol[sym] = pipelines[sym]._compute_block_summaries(panels_by_symbol[sym])

            # Column family mapping from anchor symbol.
            anchor_column_families = pipelines[symbols[0]]._infer_column_families(anchor_cols)

            # Build manifest-aligned walk-forward folds (date windows) shared across symbols.
            windows_anchor = pipelines[symbols[0]]._load_walkforward_windows(horizon)
            if not windows_anchor:
                self.logger.warning("❌ Global Optuna failed: no walk-forward windows loaded from manifest")
                return None

            walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
            for w in windows_anchor:
                train_start = w.get("train_start")
                train_end = w.get("train_end")
                valid_start = w.get("valid_start")
                # IMPORTANT: Use the same out-of-sample span as the post-Optuna
                # pooled walk-forward (valid_start -> test_end). This prevents
                # optimizer fold geometry from being much shorter than the
                # later evaluation/tape-generation geometry.
                valid_end = w.get("test_end") or w.get("valid_end")
                if train_start is None or train_end is None or valid_start is None or valid_end is None:
                    continue
                fold_entry: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                for sym in symbols:
                    idx = panels_by_symbol[sym].index
                    train_mask = (idx >= pd.Timestamp(train_start)) & (idx <= pd.Timestamp(train_end))
                    val_mask = (idx >= pd.Timestamp(valid_start)) & (idx <= pd.Timestamp(valid_end))
                    fold_entry[sym] = (np.flatnonzero(train_mask), np.flatnonzero(val_mask))
                walk_forward_folds_multi.append(fold_entry)

            if len(walk_forward_folds_multi) < 3:
                self.logger.warning(
                    "Insufficient manifest-based folds for global Optuna (%d folds)",
                    len(walk_forward_folds_multi),
                )
                return None

            # Derive horizon-aware seq_len band (same logic as single-symbol).
            lstm_seq_len_min = max(10, int(1.0 * horizon))
            lstm_seq_len_max = min(500, int(3.0 * horizon))
            if lstm_seq_len_max < lstm_seq_len_min:
                lstm_seq_len_max = lstm_seq_len_min

            # Average priors across symbols (fallback to provided priors if empty)
            if priors_accum:
                global_priors = {k: float(np.mean(vs)) for k, vs in priors_accum.items() if vs}
            else:
                global_priors = priors

            config = OptunaConfig(
                n_trials=self.config.optuna_n_trials,
                timeout=self.config.optuna_timeout,
                n_jobs=self.config.optuna_n_jobs,
                sharpe_weight=self.config.optuna_objective_sharpe_weight,
                rwa_weight=self.config.optuna_objective_rwa_weight,
                stability_weight=self.config.optuna_objective_stability_weight,
                max_total_dims=self.config.optuna_max_total_dims,
                min_track_a_dims=self.config.optuna_track_a_dim_range[0],
                max_track_a_dims=self.config.optuna_track_a_dim_range[1],
                min_track_b_dims=self.config.optuna_track_b_dim_range[0],
                max_track_b_dims=self.config.optuna_track_b_dim_range[1],
                track_a_weight_min=self.config.optuna_track_weight_range[0],
                track_a_weight_max=self.config.optuna_track_weight_range[1],
                track_b_weight_min=self.config.optuna_track_weight_range[0],
                track_b_weight_max=self.config.optuna_track_weight_range[1],
                horizon=horizon,
                seq_len_horizon_mult_min=lstm_seq_len_min / max(horizon, 1),
                seq_len_horizon_mult_max=lstm_seq_len_max / max(horizon, 1),
                lstm_batch_sizes=self.config.optuna_lstm_batch_sizes,
                seed=self.config.random_state,
                use_ray=self.config.optuna_use_ray,
                use_ray_tune=self.config.optuna_use_ray_tune,
                ray_tune_concurrent_trials=self.config.optuna_ray_tune_concurrent_trials,
                ray_cpus_per_trial=self.config.optuna_ray_cpus_per_trial,
                ray_max_concurrent_trials=self.config.optuna_ray_max_concurrency,
                ray_fold_parallelism=0,  # Global multi-symbol runs sequential folds within trial
                ray_fold_gpu_fraction=0.0,
                fold_gpu_memory_gb=self.config.optuna_fold_gpu_memory_gb,
                fold_gpu_reserve_gb=self.config.optuna_fold_gpu_reserve_gb,
                sequence_model_type=getattr(self.config, "sequence_model_type", "mamba"),
                global_multi_symbol=True,
                global_symbols=tuple(symbols),
                study_name_override=getattr(self.config, "optuna_study_name", None),
                disable_study_auto_reset=bool(
                    getattr(self.config, "optuna_disable_study_auto_reset", False)
                ),
            )

            self.logger.info(
                "🚀 Starting GLOBAL multi-symbol Optuna for H%d: %d trials, %d folds/trial, symbols=%d",
                horizon,
                config.n_trials,
                len(walk_forward_folds_multi),
                len(symbols),
            )

            try:
                params = run_optuna_optimization_multi_symbol(
                    panels_by_symbol=panels_by_symbol,
                    column_families=anchor_column_families,
                    labels_by_symbol=labels_by_symbol,
                    block_summaries_by_symbol=block_summaries_by_symbol,
                    walk_forward_folds_multi=walk_forward_folds_multi,
                    horizon=horizon,
                    config=config,
                    stage_a_weights=global_priors,
                    output_path=cache_path,
                )
                if params and params.best_score > 0:
                    self.logger.info(
                        "✅ Global Optuna complete: score=%.4f, included_families=%d",
                        params.best_score,
                        sum(params.included_families.values()),
                    )
                    return params
                self.logger.warning("Global Optuna failed to find good params")
                return None
            except Exception as exc:
                self.logger.error("Global Optuna optimization failed: %s", exc)
                import traceback

                self.logger.debug(traceback.format_exc())
                return None
        
        # ===========================================================
        # ALIGN PANEL/LABELS TO VALID LABEL COVERAGE
        # ===========================================================
        valid_label_mask = labels["forward_return"].notna()
        valid_rows = int(valid_label_mask.sum())
        total_rows = len(labels)

        if valid_rows == 0:
            self.logger.warning(
                "❌ Optuna skipped: no valid labels available for %s H%d",
                self.config.symbol,
                horizon,
            )
            return None

        if valid_rows < total_rows:
            coverage_pct = 100.0 * valid_rows / total_rows
            self.logger.info(
                "🎯 Optuna label alignment: using %d/%d rows with valid labels (%.1f%%)",
                valid_rows,
                total_rows,
                coverage_pct,
            )

            # Restrict panel and labels to rows that have forward returns
            panel = panel.loc[valid_label_mask]
            labels = labels.loc[valid_label_mask]

            # Reindex block summaries to the filtered panel for consistency
            aligned_summaries: Dict[str, pd.DataFrame] = {}
            for name, summary in block_summaries.items():
                if summary is None or summary.empty:
                    aligned_summaries[name] = summary
                    continue
                aligned = summary.reindex(panel.index)
                aligned_summaries[name] = aligned.ffill().bfill()
            block_summaries = aligned_summaries

        # ===========================================================
        # BUILD WALK-FORWARD FOLDS for Optuna
        # ONE trial runs ENTIRE walk-forward (~60 windows)
        # ===========================================================
        n_samples = len(panel)
        walk_forward_folds = self._build_optuna_walk_forward_folds(n_samples, horizon)
        
        if len(walk_forward_folds) < 3:
            self.logger.warning(
                "Insufficient walk-forward folds for Optuna optimization (%d folds)",
                len(walk_forward_folds)
            )
            return None
        
        self.logger.info(
            "🔄 Walk-forward folds for Optuna: %d folds (n_samples=%d)",
            len(walk_forward_folds), n_samples
        )
        
        # Derive horizon-aware LSTM window search bounds
        # seq_len = H × [1.0, 3.0] - lookback window scaled to forecast horizon
        # For H=63: seq_len in [63, 189]
        lstm_seq_len_min = max(10, int(1.0 * horizon))
        lstm_seq_len_max = min(500, int(3.0 * horizon))
        if lstm_seq_len_max < lstm_seq_len_min:
            lstm_seq_len_max = lstm_seq_len_min
        self.logger.info(
            "🧮 Optuna LSTM seq_len band for H%d: [%d, %d]",
            horizon, lstm_seq_len_min, lstm_seq_len_max
        )

        # Create Optuna config
        # Pass explicit seq_len bounds computed above
        config = OptunaConfig(
            n_trials=self.config.optuna_n_trials,
            timeout=self.config.optuna_timeout,
            n_jobs=self.config.optuna_n_jobs,
            sharpe_weight=self.config.optuna_objective_sharpe_weight,
            rwa_weight=self.config.optuna_objective_rwa_weight,
            stability_weight=self.config.optuna_objective_stability_weight,
            max_total_dims=self.config.optuna_max_total_dims,
            min_track_a_dims=self.config.optuna_track_a_dim_range[0],
            max_track_a_dims=self.config.optuna_track_a_dim_range[1],
            min_track_b_dims=self.config.optuna_track_b_dim_range[0],
            max_track_b_dims=self.config.optuna_track_b_dim_range[1],
            track_a_weight_min=self.config.optuna_track_weight_range[0],
            track_a_weight_max=self.config.optuna_track_weight_range[1],
            track_b_weight_min=self.config.optuna_track_weight_range[0],
            track_b_weight_max=self.config.optuna_track_weight_range[1],
            # LSTM: set horizon for dynamic bounds (seq_len, hidden_dim, dropout)
            horizon=horizon,
            # Override seq_len multipliers to use our computed bounds
            seq_len_horizon_mult_min=lstm_seq_len_min / max(horizon, 1),
            seq_len_horizon_mult_max=lstm_seq_len_max / max(horizon, 1),
            lstm_batch_sizes=self.config.optuna_lstm_batch_sizes,
            seed=self.config.random_state,
            use_ray=self.config.optuna_use_ray,
            use_ray_tune=self.config.optuna_use_ray_tune,
            ray_tune_concurrent_trials=self.config.optuna_ray_tune_concurrent_trials,
            ray_cpus_per_trial=self.config.optuna_ray_cpus_per_trial,
            ray_max_concurrent_trials=self.config.optuna_ray_max_concurrency,
            ray_fold_parallelism=self.config.optuna_ray_fold_parallelism,
            ray_fold_gpu_fraction=self.config.optuna_ray_fold_gpu_fraction,
            fold_gpu_memory_gb=self.config.optuna_fold_gpu_memory_gb,
            fold_gpu_reserve_gb=self.config.optuna_fold_gpu_reserve_gb,
            sequence_model_type=getattr(self.config, "sequence_model_type", "mamba"),
            study_name_override=getattr(self.config, "optuna_study_name", None),
            disable_study_auto_reset=bool(
                getattr(self.config, "optuna_disable_study_auto_reset", False)
            ),
        )
        
        self.logger.info(
            "🚀 Starting Optuna walk-forward optimization for %s H%d: "
            "%d trials, %d folds/trial, timeout=%ds",
            self.config.symbol, horizon, config.n_trials, 
            len(walk_forward_folds), config.timeout
        )
        
        try:
            params = run_optuna_optimization(
                panel=panel,
                column_families=column_families,
                labels=labels,
                block_summaries=block_summaries,
                walk_forward_folds=walk_forward_folds,
                horizon=horizon,
                config=config,
                stage_a_weights=priors,
                output_path=cache_path,
            )
            
            if params and params.best_score > 0:
                self.logger.info(
                    "✅ Optuna walk-forward optimization complete: score=%.4f, included_families=%d",
                    params.best_score, sum(params.included_families.values())
                )
                return params
            else:
                self.logger.warning("Optuna optimization failed to find good params")
                return None
                
        except Exception as e:
            self.logger.error("Optuna optimization failed: %s", e)
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _build_optuna_walk_forward_folds(
        self,
        n_samples: int,
        horizon: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Build walk-forward folds for Optuna optimization.
        
        Creates expanding-window folds similar to production walk-forward:
        - Initial training window: ~3-5 years of data
        - Step size: ~1 year (252 trading days)
        - Validation window: ~1 year
        
        For ~10 years of data (~2520 samples):
        - Initial train: 756 samples (3 years)
        - Step: 252 samples (1 year)  
        - This creates ~(2520-756)/252 ≈ 7 folds
        
        For shorter periods, adjust proportionally to get ~20-60 folds.
        
        Args:
            n_samples: Total number of samples
            horizon: Forecast horizon
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        if n_samples < 500:
            # Too few samples for meaningful walk-forward
            self.logger.warning(
                "Insufficient samples for walk-forward Optuna (%d < 500)",
                n_samples
            )
            # Fallback: simple train/val split
            pivot = int(n_samples * 0.7)
            return [(np.arange(0, pivot), np.arange(pivot, n_samples))]
        
        # Walk-forward parameters
        # Minimum training window: at least 500 samples or 30% of data
        min_train_samples = max(500, int(n_samples * 0.3))
        
        # Step size: aim for ~60 folds over the data
        # If we have 2520 samples and want 60 folds after min_train:
        # step = (2520 - min_train) / 60 ≈ 33 samples
        target_folds = min(60, max(10, (n_samples - min_train_samples) // 50))
        step_size = max(20, (n_samples - min_train_samples) // target_folds)
        
        # Validation window: follow walk-forward step size without additional padding
        val_size = step_size
        
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        
        train_end = min_train_samples
        while train_end + val_size <= n_samples:
            train_idx = np.arange(0, train_end)  # Expanding window
            val_start = train_end
            val_end = min(train_end + val_size, n_samples)
            val_idx = np.arange(val_start, val_end)
            
            if len(train_idx) >= 100 and len(val_idx) >= 10:
                folds.append((train_idx, val_idx))
            
            train_end += step_size
        
        # Ensure at least 5 folds
        if len(folds) < 5:
            self.logger.warning(
                "Only %d walk-forward folds generated, using proportional splits",
                len(folds)
            )
            # Create 10 proportional splits
            folds = []
            for i in range(10):
                split_point = int(n_samples * (0.3 + 0.05 * i))
                train_idx = np.arange(0, split_point)
                val_size_prop = min(int(n_samples * 0.1), n_samples - split_point)
                val_idx = np.arange(split_point, split_point + val_size_prop)
                if len(train_idx) >= 50 and len(val_idx) >= 10:
                    folds.append((train_idx, val_idx))
        
        self.logger.debug(
            "Built %d walk-forward folds: min_train=%d, step=%d, val=%d",
            len(folds), min_train_samples, step_size, val_size
        )
        
        return folds
    
    def _apply_optuna_lstm_params(self, params: "OptimizedParams"):
        """Apply optimized sequence hyperparameters and threshold params to config.
        
        This updates self.config with Optuna-optimized:
        - LSTM hyperparameters
        - STEP 7: Regime-aware threshold parameters
        
        The optimized feature selection is applied separately in _prepare_lstm_features.
        """
        self.logger.info("🔧 Applying Optuna-optimized parameters")
        
        # Apply legacy LSTM hyperparameters (kept for compatibility; not used in Mamba-only mode)
        if params.lstm_layers > 0:
            self.config.lstm_layers = params.lstm_layers
        if params.lstm_hidden_dim > 0:
            self.config.lstm_hidden_dim = params.lstm_hidden_dim
        if params.lstm_dropout >= 0:
            self.config.lstm_dropout = params.lstm_dropout
        if params.lstm_seq_len > 0:
            self.config.lstm_seq_len = params.lstm_seq_len
        if params.lstm_batch_size > 0:
            self.config.lstm_batch_size = params.lstm_batch_size
        if params.lstm_learning_rate > 0:
            self.config.lstm_learning_rate = params.lstm_learning_rate
        self.config.lstm_use_amp = params.lstm_use_amp
        
        self.logger.info(
            "📊 LSTM params: layers=%d, hidden=%d, dropout=%.2f, seq_len=%d, batch=%d, lr=%.2e",
            self.config.lstm_layers, self.config.lstm_hidden_dim, self.config.lstm_dropout,
            self.config.lstm_seq_len, self.config.lstm_batch_size, self.config.lstm_learning_rate
        )

        # Apply Mamba hyperparameters (Stage-B runtime)
        # These are stored in the Optuna cache JSON and must drive the post-Optuna walk-forward engine.
        try:
            if getattr(params, "mamba_seq_len", None):
                self.config.mamba_seq_len = int(params.mamba_seq_len)
            if getattr(params, "mamba_d_model", None):
                self.config.mamba_d_model = int(params.mamba_d_model)
            if getattr(params, "mamba_n_layers", None):
                self.config.mamba_n_layers = int(params.mamba_n_layers)
            if getattr(params, "mamba_ssm_dim", None):
                self.config.mamba_ssm_dim = int(params.mamba_ssm_dim)
            if getattr(params, "mamba_expand_factor", None) is not None:
                self.config.mamba_expand_factor = float(params.mamba_expand_factor)
            if getattr(params, "mamba_activation", None):
                self.config.mamba_activation = str(params.mamba_activation)
            if getattr(params, "mamba_norm_type", None):
                self.config.mamba_norm_type = str(params.mamba_norm_type)
            if getattr(params, "mamba_norm_strategy", None):
                self.config.mamba_norm_strategy = str(params.mamba_norm_strategy)
            if getattr(params, "mamba_dropout", None) is not None:
                self.config.mamba_dropout = float(params.mamba_dropout)
            if getattr(params, "mamba_resid_dropout", None) is not None:
                self.config.mamba_resid_dropout = float(params.mamba_resid_dropout)
            if getattr(params, "mamba_ssm_dropout", None) is not None:
                self.config.mamba_ssm_dropout = float(params.mamba_ssm_dropout)
            if getattr(params, "mamba_gate_dropout", None) is not None:
                self.config.mamba_gate_dropout = float(params.mamba_gate_dropout)
            if getattr(params, "mamba_optimizer", None):
                self.config.mamba_optimizer = str(params.mamba_optimizer)
            if getattr(params, "mamba_learning_rate", None) is not None:
                self.config.mamba_learning_rate = float(params.mamba_learning_rate)
            if getattr(params, "mamba_weight_decay", None) is not None:
                self.config.mamba_weight_decay = float(params.mamba_weight_decay)
            if getattr(params, "mamba_grad_clip", None) is not None:
                self.config.mamba_grad_clip = float(params.mamba_grad_clip)
            if getattr(params, "mamba_lr_scheduler", None):
                self.config.mamba_lr_scheduler = str(params.mamba_lr_scheduler)
            if getattr(params, "mamba_warmup_steps", None) is not None:
                self.config.mamba_warmup_steps = int(params.mamba_warmup_steps)
            if getattr(params, "mamba_max_epochs", None) is not None:
                self.config.mamba_max_epochs = int(params.mamba_max_epochs)
            if getattr(params, "mamba_batch_size", None) is not None:
                self.config.mamba_batch_size = int(params.mamba_batch_size)
            if getattr(params, "mamba_loss_fn", None):
                self.config.mamba_loss_fn = str(params.mamba_loss_fn)
            if getattr(params, "mamba_head_type", None):
                self.config.mamba_head_type = str(params.mamba_head_type)
            if getattr(params, "mamba_head_hidden_dim", None) is not None:
                self.config.mamba_head_hidden_dim = int(params.mamba_head_hidden_dim)
            if getattr(params, "mamba_head_num_layers", None) is not None:
                self.config.mamba_head_num_layers = int(params.mamba_head_num_layers)
            if getattr(params, "mamba_head_dropout", None) is not None:
                self.config.mamba_head_dropout = float(params.mamba_head_dropout)

            self.logger.info(
                "📊 Mamba params: seq_len=%d, d_model=%d, layers=%d, ssm_dim=%d, lr=%.2e, batch=%d",
                int(getattr(self.config, "mamba_seq_len", 0)),
                int(getattr(self.config, "mamba_d_model", 0)),
                int(getattr(self.config, "mamba_n_layers", 0)),
                int(getattr(self.config, "mamba_ssm_dim", 0)),
                float(getattr(self.config, "mamba_learning_rate", 0.0)),
                int(getattr(self.config, "mamba_batch_size", 0)),
            )
        except Exception as exc:
            self.logger.warning("Failed to apply Optuna Mamba params: %s", exc)
        
        # ===================================================================
        # STEP 7: Apply Regime-Aware Threshold Parameters
        # ===================================================================
        if params.threshold > 0:
            self.config.regime_threshold_base = params.threshold
            self.config.regime_threshold_bull_mult = params.bull_mult
            self.config.regime_threshold_bear_mult = params.bear_mult
            self.config.regime_threshold_crisis_mult = params.crisis_mult
            self.config.regime_aware_thresholds_enabled = True
            
            # Also update legacy thresholds for backwards compatibility
            # These will be overridden by regime logic when enabled
            T_base = params.threshold
            self.config.signal_threshold_upper = T_base * params.bull_mult
            self.config.signal_threshold_lower = -T_base * params.bull_mult
            
            self.logger.info(
                "📊 STEP 7 Threshold params: T=%.3f, bull=%.2f, bear=%.2f, crisis=%.2f",
                params.threshold, params.bull_mult, params.bear_mult, params.crisis_mult
            )
            self.logger.info(
                "   → T_bull=%.3f, T_bear=%.3f, T_crisis=%.3f",
                T_base * params.bull_mult,
                T_base * params.bear_mult, 
                T_base * params.crisis_mult
            )
    
    def _build_optuna_track_c(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        block_summaries: Dict[str, pd.DataFrame],
        optuna_params: "OptimizedParams",
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Build Track C using Optuna-optimized parameters.
        
        This replaces the legacy _prepare_lstm_features when Optuna is enabled.
        
        Track C = concat(Track A × weight_a, Track B × weight_b)
        where:
        - Track A: Stage-A families → Encoders (PCA/AE) → Compressed features
        - Track B: HF blocks + Model-family summaries
        
        Encoders are fitted on training data and cached.
        
        Args:
            panel: Full feature panel
            column_families: Column to family mapping
            block_summaries: Stage-B block summaries
            optuna_params: Optimized parameters from Optuna
            
        Returns:
            Tuple of (track_c_frame, fitted_encoders) or (None, None) on failure
        """
        try:
            from src.stage_b.optuna_optimizer import apply_optimized_params
            
            # Determine training indices for encoder fitting
            # Use first 70% of data or current train indices if available
            if self._current_train_idx is not None and len(self._current_train_idx) > 0:
                train_idx = self._current_train_idx
            else:
                n_samples = len(panel)
                train_idx = np.arange(int(n_samples * 0.7))
            
            # Build Track C using apply_optimized_params
            track_c, encoders, track_meta = apply_optimized_params(
                panel=panel,
                column_families=column_families,
                block_summaries=block_summaries,
                params=optuna_params,
                train_idx=train_idx,
            )
            
            if track_c is None or track_c.empty:
                self.logger.warning("Optuna Track C is empty")
                return None, None
            
            # Sanitize the frame
            track_c = self._sanitize_frame(track_c)
            track_c = self._apply_optuna_track_weights(track_c, track_meta)
            
            self.logger.info(
                "✅ Built Optuna Track C: shape=%s, Track A weight=%.2f, Track B weight=%.2f",
                track_c.shape, optuna_params.track_a_weight, optuna_params.track_b_weight
            )
            
            return track_c, encoders
            
        except Exception as e:
            self.logger.error("Failed to build Optuna Track C: %s", e)
            import traceback
            self.logger.debug(traceback.format_exc())
            return None, None

    def _apply_optuna_track_weights(
        self,
        frame: Optional[pd.DataFrame],
        track_meta: Optional[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        """Apply Optuna-determined Track A/B weights after feature sanitization."""
        if frame is None or frame.empty or not track_meta:
            return frame
        adjusted = frame.copy()

        def _apply(cols_key: str, weight_key: str) -> None:
            cols = [c for c in track_meta.get(cols_key, []) if c in adjusted.columns]
            if not cols:
                return
            weight = track_meta.get(weight_key)
            try:
                weight_val = float(weight)
            except (TypeError, ValueError):
                return
            adjusted.loc[:, cols] = adjusted.loc[:, cols] * weight_val

        _apply("track_a_cols", "track_a_weight")
        _apply("track_b_cols", "track_b_weight")
        return adjusted

    # ------------------------------------------------------------------
    # Tier-2 gating and LSTM prep
    # ------------------------------------------------------------------
    def _allow_lstm(
        self,
        horizon: int,
        n_samples: int,
    ) -> bool:
        """Determine if LSTM (Tier-2) models should run.
        
        🚀 LSTM-ONLY MODE: Simplified - only checks:
        1. Sufficient samples (min_samples_for_lstm)
        2. Horizon cluster is enabled
        """
        if n_samples < self.config.min_samples_for_lstm:
            self.logger.info(
                "LSTM disabled: insufficient samples (%d < %d)",
                n_samples, self.config.min_samples_for_lstm
            )
            return False
        cluster = self._resolve_cluster(horizon)
        if cluster is None or cluster not in self.config.run_lstm_for_clusters:
            self.logger.info("LSTM disabled: cluster %s not in enabled list", cluster)
            return False
        self.logger.info("✅ LSTM enabled for horizon %d (cluster=%s, samples=%d)", horizon, cluster, n_samples)
        return True

    def _residuals_exhibit_autocorr(self, context: Dict[str, Any]) -> bool:
        preds = context.get("preds")
        actuals = context.get("actuals")
        if preds is None or actuals is None:
            return False
        residuals = actuals - preds
        residuals = residuals.fillna(0.0)
        if len(residuals) < 50:
            return False
        series = residuals.values
        series = series - np.mean(series)
        numerator = np.sum(series[1:] * series[:-1])
        denominator = np.sum(series[:-1] ** 2) + 1e-9
        autocorr = numerator / denominator
        return float(np.abs(autocorr)) > 0.1

    def _prepare_lstm_features(
        self,
        panel: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
        priors: Dict[str, float],
        stage_a_meta: Dict[str, Any],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        numeric = panel.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return None

        # PCA/AE encoders (Track A/C) run strictly after lag validation and numeric downcasting,
        # so every sequence view here already reflects the sanitized float32 panel.

        # =======================================================================
        # 🔧 ADAPTIVE SELECTION: Only applies to Stage-A raw families
        # Stage-B families (quantile_forecast, calibration, etc.) are NEVER filtered
        # Uses Stage-A config (family_weights_best.json) to rank raw families
        # =======================================================================
        
        # Separate Stage-A and Stage-B columns
        stage_a_cols = [c for c in numeric.columns if column_families.get(c) in STAGE_A_FAMILIES]
        stage_a_block = numeric[stage_a_cols] if stage_a_cols else pd.DataFrame(index=panel.index)
        
        # Stage-B families are ALWAYS included (never filtered)
        stage_b_families = STAGE_B_BASE_FAMILIES + HF_BLOCK_FAMILIES + ("hf_agg",)
        stage_b_cols = [c for c in numeric.columns if column_families.get(c) in stage_b_families]
        stage_b_block = numeric[stage_b_cols] if stage_b_cols else pd.DataFrame(index=panel.index)

        adaptive_enabled = self.config.lstm_adaptive_selection_enabled

        # 🔧 BYPASS: When adaptive selection is disabled or ratios hit 1.0, keep everything
        use_all_stage_a_features = (
            (not adaptive_enabled) or
            (
                self.config.lstm_adaptive_family_ratio >= 1.0 and 
                self.config.lstm_adaptive_feature_ratio >= 1.0
            )
        )
        
        # Build family meta for ALL families (needed for summaries)
        family_meta_static = self._build_family_meta_from_stage_a(panel.index, stage_a_meta, top_k=None)
        family_meta_dynamic = self._build_family_dynamic_meta(stage_a_block, column_families, priors, block_summaries)
        family_meta_all = self._sanitize_frame(pd.concat([family_meta_static, family_meta_dynamic], axis=1))
        
        # Select top Stage-A families using Stage-A weights (priors)
        if use_all_stage_a_features:
            # Use ALL Stage-A families - no selection
            top_stage_a_families = list(STAGE_A_FAMILIES)
            if adaptive_enabled:
                self.logger.info(
                    "🔧 LSTM adaptive selection bypassed (ratios=1.0): using ALL %d Stage-A families",
                    len(top_stage_a_families)
                )
            else:
                self.logger.info(
                    "🔧 LSTM adaptive selection disabled: using ALL %d Stage-A families",
                    len(top_stage_a_families)
                )
        else:
            # Apply adaptive selection ONLY to Stage-A families using Stage-A weights
            top_stage_a_families = self._select_top_families(priors, stage_a_meta, self.config.top_k_families_for_seq)
            
            # 🔧 Log the Stage-A weights being used for family selection
            stage_a_priors = {fam: weight for fam, weight in priors.items() if fam in STAGE_A_FAMILIES}
            if stage_a_priors:
                sorted_priors = sorted(stage_a_priors.items(), key=lambda kv: -kv[1])[:5]
                self.logger.info(
                    "🔧 Stage-A weights (top 5): %s",
                    ", ".join([f"{fam}={w:.3f}" for fam, w in sorted_priors])
                )
            
            self.logger.info(
                "🔧 LSTM adaptive selection: %d/%d Stage-A families (family_ratio=%.2f), feature_ratio=%.2f",
                len(top_stage_a_families), len(STAGE_A_FAMILIES), 
                self.config.lstm_adaptive_family_ratio, self.config.lstm_adaptive_feature_ratio
            )
        
        if not adaptive_enabled:
            family_meta_top = family_meta_all
        else:
            # family_meta_top is subset for selected Stage-A families only
            family_meta_top = self._family_meta_subset(family_meta_all, top_stage_a_families)
            if family_meta_top is None or family_meta_top.empty:
                family_meta_top = self._family_meta_subset(family_meta_static, top_stage_a_families)
            family_meta_top = self._sanitize_frame(family_meta_top)

        hf_block = self._build_hf_agg_block(
            panel=panel,
            column_families=column_families,
            priors=priors,
            block_summaries=block_summaries,
            stage_a_meta=stage_a_meta,
        )
        for col in hf_block.columns:
            column_families[col] = "hf_agg"
        hf_block_features = self._build_hf_block_features(panel, column_families)
        hf_inputs = self._sanitize_frame(pd.concat([hf_block_features, hf_block], axis=1))

        # =======================================================================
        # Build seq_raw_stage_a: adaptively selected Stage-A features
        # =======================================================================
        if use_all_stage_a_features:
            # Use ALL Stage-A features directly - same as Track A
            seq_raw_stage_a = stage_a_block
            family_feature_map = {fam: [c for c in stage_a_block.columns if column_families.get(c) == fam] 
                                  for fam in top_stage_a_families if fam in STAGE_A_FAMILIES}
            self.logger.info(
                "🔧 LSTM: Using ALL %d Stage-A features (no adaptive feature filtering)",
                len(stage_a_block.columns)
            )
        else:
            # Apply adaptive feature selection within each selected Stage-A family
            seq_feature_details = stage_a_meta.get("family_feature_details", {}) or {}
            raw_family_cols, family_feature_map = self._collect_sequence_family_features(
                stage_a_block,
                column_families,
                seq_feature_details,
                top_stage_a_families,  # Only selected Stage-A families
            )
            seq_raw_stage_a = stage_a_block[raw_family_cols] if raw_family_cols else pd.DataFrame(index=panel.index)
            self.logger.info(
                "🔧 LSTM: Adaptive feature selection reduced Stage-A from %d to %d features (ratio=%.2f)",
                len(stage_a_block.columns), len(raw_family_cols), self.config.lstm_adaptive_feature_ratio
            )

        quant_summary = block_summaries.get("quantile", pd.DataFrame(index=panel.index))
        calibration_summary = block_summaries.get("calibration", pd.DataFrame(index=panel.index))
        online_summary = block_summaries.get("online", pd.DataFrame(index=panel.index))
        arima_summary = block_summaries.get("arima", pd.DataFrame(index=panel.index))

        prior_stats = pd.DataFrame(index=panel.index)
        prior_values = list(priors.values())
        prior_stats["prior_weight_mean"] = np.mean(prior_values) if prior_values else 0.0
        prior_stats["prior_weight_std"] = np.std(prior_values) if prior_values else 0.0

        # Get combined summary (from block_summaries)
        combined_summary = block_summaries.get("combined", pd.DataFrame(index=panel.index))

        # =====================================================================
        # 🔧 FIX: Build LSTM views with adaptive selection ONLY for raw Stage-A features
        # =====================================================================
        # 
        # Track A (seq_raw): ADAPTIVELY SELECTED Stage-A raw features + family_meta_all
        #                    NO summaries (those belong to Track B)
        #
        # Track B (seq_hf):  HF inputs + Stage-B summaries + family_meta_all (UNFILTERED)
        #                    = hf_inputs + quant + calibration + online + arima + family_meta_all
        #
        # Track C (seq_hybrid): Track A + Track B (deduplicated on family_meta_all)
        # =====================================================================
        
        seq_views: Dict[str, pd.DataFrame] = {}
        
        # ---------------------------------------------------------------------
        # seq_raw = Track A: ONLY Stage-A raw features (NO family_meta_all)
        # 🔧 FIX: Track A is purely raw Stage-A features - family_meta_all stays in Track B
        # ---------------------------------------------------------------------
        seq_raw_sources = [seq_raw_stage_a]  # NO family_meta_all
        seq_raw_frame = self._sanitize_frame(pd.concat(seq_raw_sources, axis=1))
        # Remove duplicate columns
        seq_raw_frame = seq_raw_frame.loc[:, ~seq_raw_frame.columns.duplicated()]
        if not seq_raw_frame.empty:
            seq_views["seq_raw"] = seq_raw_frame

        # ---------------------------------------------------------------------
        # seq_hf = Track B: HF inputs + Stage-B summaries + family_meta_all
        # family_meta_all (~216 cols) provides family-level aggregated signals
        # ---------------------------------------------------------------------
        seq_hf_sources = [hf_inputs, quant_summary, calibration_summary, 
                         online_summary, arima_summary, family_meta_all]
        seq_hf_frame = self._sanitize_frame(pd.concat(seq_hf_sources, axis=1))
        # Remove duplicate columns
        seq_hf_frame = seq_hf_frame.loc[:, ~seq_hf_frame.columns.duplicated()]
        if not seq_hf_frame.empty:
            seq_views["seq_hf"] = seq_hf_frame

        # ---------------------------------------------------------------------
        # seq_hybrid = Track C: Track A + Track B (NO deduplication)
        # Track A has raw Stage-A features, Track B has HF+summaries+family_meta_all
        # Combined gives full feature set with family_meta_all appearing once (from B)
        # ---------------------------------------------------------------------
        if "seq_raw" in seq_views and "seq_hf" in seq_views:
            # Simple concatenation - A has raw features, B has HF+summaries+family_meta
            seq_hybrid_frame = self._sanitize_frame(pd.concat([seq_views["seq_raw"], seq_views["seq_hf"]], axis=1))
        else:
            seq_hybrid_frame = self._sanitize_frame(pd.concat(seq_raw_sources + seq_hf_sources, axis=1))
        if not seq_hybrid_frame.empty:
            seq_views["seq_hybrid"] = seq_hybrid_frame

        if seq_views:
            # Log final LSTM input dimensions with accurate track mapping
            for view_name, view_frame in seq_views.items():
                if view_name == "seq_raw":
                    self.logger.info(
                        "LSTM %s view: D=%d features (= Track A: Stage-A raw features ONLY)",
                        view_name, view_frame.shape[1]
                    )
                elif view_name == "seq_hf":
                    self.logger.info(
                        "LSTM %s view: D=%d features (= Track B: HF + summaries + family_meta_all)",
                        view_name, view_frame.shape[1]
                    )
                elif view_name == "seq_hybrid":
                    self.logger.info(
                        "LSTM %s view: D=%d features (= Track C: Track A + Track B, no dedup)",
                        view_name, view_frame.shape[1]
                    )
            seq_views["__meta__"] = {
                "top_families": top_stage_a_families,
                "family_feature_map": family_feature_map,
                "track_a_cols": seq_raw_frame.columns.tolist() if "seq_raw" in seq_views else [],
                "track_b_cols": seq_hf_frame.columns.tolist() if "seq_hf" in seq_views else [],
            }

        return seq_views or None

    def _limit_sequence_dim_preserve_protected(
        self,
        frame: pd.DataFrame,
        protected_cols: List[str],
    ) -> pd.DataFrame:
        """Limit sequence dimension while preserving protected (Stage-B) columns."""
        if frame is None or frame.empty:
            return frame
        limit = self.config.seq_max_feature_dim
        if limit is None:
            return frame
        
        # Separate protected vs non-protected
        protected = [c for c in protected_cols if c in frame.columns]
        non_protected = [c for c in frame.columns if c not in protected]
        
        if len(non_protected) == 0:
            return frame
        
        # Only limit non-protected columns
        available_for_non_protected = max(0, limit - len(protected))
        if len(non_protected) <= available_for_non_protected:
            return frame
        
        # Rank non-protected by variance and select top
        non_protected_frame = frame[non_protected]
        variance = non_protected_frame.var().abs().sort_values(ascending=False)
        keep_non_protected = list(variance.head(available_for_non_protected).index)
        
        # Combine protected + selected non-protected
        final_cols = protected + keep_non_protected
        self.logger.debug(
            "Dim limit with protection: %d protected + %d selected = %d total",
            len(protected), len(keep_non_protected), len(final_cols)
        )
        return frame.reindex(columns=final_cols)

    def _collect_sequence_family_features(
        self,
        stage_a_block: pd.DataFrame,
        column_families: Dict[str, str],
        feature_details: Dict[str, Any],
        top_families: Sequence[str],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Collect features from Stage-A families for LSTM using Stage-A importance.
        
        🔧 CRITICAL: This function is ONLY called for Stage-A families.
        Stage-B families are handled separately and NEVER filtered.
        
        Uses feature_details from Stage-A artifacts (family_feature_details) to rank
        features within each family by their importance contribution.
        
        Args:
            stage_a_block: DataFrame with only Stage-A family columns
            column_families: Mapping of column name to family
            feature_details: Stage-A family_feature_details (from artifacts)
            top_families: Selected Stage-A families (from _select_top_families)
        
        Returns:
            Tuple of (selected_column_names, family_feature_map)
        """
        selected: List[str] = []
        family_feature_map: Dict[str, List[str]] = {}
        
        # Legacy fixed limit or adaptive
        use_adaptive = self.config.seq_raw_features_per_family <= 0
        
        # Track which families used Stage-A importance vs variance fallback
        families_with_stage_a_importance = []
        families_using_variance_fallback = []
        
        for fam in top_families:
            # Only process Stage-A families
            if fam not in STAGE_A_FAMILIES:
                continue
                
            cols = [c for c in stage_a_block.columns if column_families.get(c) == fam]
            if not cols:
                continue
            
            # Adaptive M_f = min(all_features, floor(ratio * feature_count_per_family))
            if use_adaptive:
                family_feature_count = len(cols)
                adaptive_m = max(1, int(self.config.lstm_adaptive_feature_ratio * family_feature_count))
                limit = min(family_feature_count, adaptive_m)
            else:
                limit = max(1, int(self.config.seq_raw_features_per_family))
            
            ranked: List[str] = []
            used_stage_a_importance = False
            
            # Strategy 1: Use Stage-A feature importance (from Stage-A artifacts) - PREFERRED
            details = feature_details.get(fam, {}).get("features", []) if feature_details else []
            sorted_details = sorted(details, key=lambda item: item.get("mean_importance", 0.0), reverse=True)
            for item in sorted_details:
                name = item.get("feature")
                if name in cols:
                    ranked.append(name)
                elif name and name in stage_a_block.columns:
                    ranked.append(name)
                if len(ranked) >= limit:
                    break
            
            if ranked:
                used_stage_a_importance = True
                families_with_stage_a_importance.append(fam)
            
            # Strategy 2: Feature volatility / informativeness (variance ranking) - FALLBACK
            if len(ranked) < limit:
                remaining = [c for c in cols if c not in ranked]
                if remaining:
                    variance_rank = stage_a_block[remaining].std().abs().sort_values(ascending=False)
                    fallback_count = limit - len(ranked)
                    ranked.extend(list(variance_rank.head(fallback_count).index))
                    
                    # Track families using variance fallback
                    if not used_stage_a_importance:
                        families_using_variance_fallback.append(fam)
            
            ranked = ranked[:limit]
            if not ranked:
                continue
            family_feature_map[fam] = ranked
            selected.extend(ranked)
        
        # Log summary of feature selection method usage
        total_families = len(family_feature_map)
        n_with_importance = len(families_with_stage_a_importance)
        n_variance_only = len(families_using_variance_fallback)
        
        if n_variance_only > 0:
            self.logger.warning(
                "⚠️ Feature selection: %d/%d families using variance fallback (no Stage-A importance): %s",
                n_variance_only, total_families, 
                ", ".join(families_using_variance_fallback[:5]) + ("..." if n_variance_only > 5 else "")
            )
        
        # 🔍 DIAGNOSTIC: Count lag columns in selected vs total
        total_cols = list(stage_a_block.columns)
        total_lag_cols = sum(1 for c in total_cols if "_lag_" in c)
        selected_lag_cols = sum(1 for c in selected if "_lag_" in c)
        
        self.logger.info(
            "📊 Adaptive feature selection: %d features from %d families (Stage-A importance: %d, variance fallback: %d, ratio=%.2f)",
            len(selected), total_families, n_with_importance, n_variance_only, self.config.lstm_adaptive_feature_ratio
        )
        self.logger.info(
            "📈 LAG ANALYSIS: Total=%d (%d lags, %.1f%%), Selected=%d (%d lags, %.1f%%)",
            len(total_cols), total_lag_cols, 100*total_lag_cols/len(total_cols) if total_cols else 0,
            len(selected), selected_lag_cols, 100*selected_lag_cols/len(selected) if selected else 0
        )
        
        return selected, family_feature_map

    def _limit_sequence_dim(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Optionally limit sequence dimension. No hard cap by default (seq_max_feature_dim=None).
        
        With adaptive selection, final LSTM input dim D will normally land between 250-800
        depending on symbol/horizon. LSTM remains trainable and stable.
        """
        if frame is None or frame.empty:
            return frame
        limit = self.config.seq_max_feature_dim
        # No cap if limit is None - trust adaptive selection
        if limit is None:
            self.logger.debug("No sequence dim cap - using adaptive selection (dim=%d)", frame.shape[1])
            return frame
        if frame.shape[1] <= limit:
            return frame
        # Only apply if explicitly set and exceeded
        self.logger.debug("Applying sequence dim cap: %d -> %d", frame.shape[1], limit)
        variance = frame.var().abs().sort_values(ascending=False)
        keep = variance.head(limit).index
        return frame.reindex(columns=keep)

    def _compress_track_c_train_val(
        self,
        train_frame: pd.DataFrame,
        val_frame: Optional[pd.DataFrame],
        track_a_cols: List[str],
        track_b_cols: List[str],
        family_feature_map: Dict[str, List[str]],
        raw_passthrough_families: Optional[Set[str]],
        max_total_dims: int,
        target_variance: float,
        family_dim_min: int,
        family_dim_max: int,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
        """Leakage-safe Track C compression for seq_hybrid.

        - Keep Track B columns verbatim.
        - PCA-compress Track A per Stage-A family (fit on train only).
        - Choose k_req per family as the smallest k reaching target_variance (or max available).
        - Enforce total dim cap (max_total_dims) by shrinking if needed.
        - If there is remaining budget, *add extra components* (variance stays >= target) to
          land near the cap (desired ~450–500 total dims).
        """
        if train_frame is None or train_frame.empty:
            return train_frame, val_frame, {
                "track_b_dims": 0,
                "track_a_raw_dims": 0,
                "track_a_pca_dims": 0,
                "total_dims": 0,
                "max_total_dims": int(max_total_dims),
                "families": 0,
                "families_hit_target_at_k_req": 0,
            }

        track_b_cols_in = [c for c in track_b_cols if c in train_frame.columns]
        track_a_cols_in = [c for c in track_a_cols if c in train_frame.columns and c not in track_b_cols_in]

        max_total_dims_i = int(max_total_dims)
        track_b_dims = len(track_b_cols_in)
        raw_fams = set(raw_passthrough_families or set())

        fam_to_cols: Dict[str, List[str]] = {}
        raw_to_cols: Dict[str, List[str]] = {}
        for fam, cols in (family_feature_map or {}).items():
            keep = [c for c in cols if c in track_a_cols_in]
            if keep:
                if fam in raw_fams:
                    raw_to_cols[fam] = keep
                else:
                    fam_to_cols[fam] = keep

        covered = set(c for cols in fam_to_cols.values() for c in cols)
        covered |= set(c for cols in raw_to_cols.values() for c in cols)
        leftovers = [c for c in track_a_cols_in if c not in covered]
        if leftovers:
            fam_to_cols.setdefault("__misc__", []).extend(leftovers)

        # Raw passthrough Track-A dims are fixed and count against the cap.
        raw_track_a_cols: List[str] = []
        for fam, cols in raw_to_cols.items():
            raw_track_a_cols.extend(cols)
        raw_track_a_cols = [c for c in raw_track_a_cols if c in train_frame.columns and c not in track_b_cols_in]
        raw_track_a_cols = list(dict.fromkeys(raw_track_a_cols))
        raw_a_dims = len(raw_track_a_cols)

        budget_a = max(0, max_total_dims_i - track_b_dims - raw_a_dims)
        if not track_a_cols_in:
            train_out = self._sanitize_frame(train_frame[track_b_cols_in])
            val_out = self._sanitize_frame(val_frame[track_b_cols_in]) if val_frame is not None else None
            return train_out, val_out, {
                "track_b_dims": track_b_dims,
                "track_a_raw_dims": 0,
                "track_a_pca_dims": 0,
                "total_dims": track_b_dims,
                "max_total_dims": max_total_dims_i,
                "families": 0,
                "families_hit_target_at_k_req": 0,
            }

        if budget_a <= 0:
            self.logger.warning(
                "⚠️ Track C PCA budget exhausted by raw passthrough: track_b=%d, track_a_raw=%d, cap=%d. Using raw only.",
                track_b_dims,
                raw_a_dims,
                max_total_dims_i,
            )
            parts_train: List[pd.DataFrame] = []
            if track_b_cols_in:
                parts_train.append(train_frame[track_b_cols_in])
            if raw_track_a_cols:
                parts_train.append(train_frame[raw_track_a_cols])
            train_out = self._sanitize_frame(pd.concat(parts_train, axis=1))
            train_out = train_out.loc[:, ~train_out.columns.duplicated()]
            val_out = None
            if val_frame is not None:
                parts_val: List[pd.DataFrame] = []
                if track_b_cols_in:
                    parts_val.append(val_frame[track_b_cols_in])
                if raw_track_a_cols:
                    parts_val.append(val_frame[raw_track_a_cols])
                val_out = self._sanitize_frame(pd.concat(parts_val, axis=1))
                val_out = val_out.loc[:, ~val_out.columns.duplicated()]
            return train_out, val_out, {
                "track_b_dims": track_b_dims,
                "track_a_raw_dims": raw_a_dims,
                "track_a_pca_dims": 0,
                "total_dims": int(train_out.shape[1]),
                "max_total_dims": max_total_dims_i,
                "target_variance": float(target_variance),
                "budget_a": int(budget_a),
                "families": int(0),
                "families_hit_target_at_k_req": int(0),
                "pca_families": int(0),
                "raw_passthrough_families": int(len(raw_to_cols)),
            }

        fam_models: Dict[str, Dict[str, Any]] = {}
        for fam, cols in fam_to_cols.items():
            x_train = train_frame[cols].astype(float).to_numpy(copy=True)
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples = x_train.shape[0]
            n_features = x_train.shape[1]
            if n_features <= 1 or n_samples <= 2:
                fam_models[fam] = {
                    "mode": "identity",
                    "cols": cols,
                    "k_req": min(n_features, 1),
                    "k_max": min(n_features, 1),
                    "achieved_var": 1.0,
                }
                continue

            mu = x_train.mean(axis=0)
            sigma = x_train.std(axis=0)
            sigma = np.where(sigma < 1e-6, 1e-6, sigma)
            xz = (x_train - mu) / sigma

            n_comp_max = min(n_features, max(1, n_samples - 1), int(family_dim_max))
            if n_comp_max <= 1:
                fam_models[fam] = {
                    "mode": "identity",
                    "cols": cols,
                    "k_req": 1,
                    "k_max": 1,
                    "achieved_var": 1.0,
                }
                continue

            pca = PCA(
                n_components=n_comp_max,
                svd_solver="full",
                random_state=getattr(self.config, "random_state", 42),
            )
            pca.fit(xz)
            csum = np.cumsum(pca.explained_variance_ratio_)
            k_req = int(np.searchsorted(csum, float(target_variance)) + 1)
            k_req = max(int(family_dim_min), min(int(k_req), int(n_comp_max)))
            achieved = float(csum[min(k_req, len(csum)) - 1]) if len(csum) else 0.0
            fam_models[fam] = {
                "mode": "pca",
                "cols": cols,
                "mu": mu,
                "sigma": sigma,
                "pca": pca,
                "k_req": k_req,
                "k_max": int(n_comp_max),
                "achieved_var": achieved,
            }

        alloc: Dict[str, int] = {fam: int(m["k_req"]) for fam, m in fam_models.items()}
        total_alloc = sum(alloc.values())

        # Enforce budget by shrinking if necessary.
        if total_alloc > budget_a:
            self.logger.warning(
                "⚠️ Track C PCA budget shortfall: requested=%d > budget=%d (track_b=%d, cap=%d). Shrinking dims.",
                total_alloc,
                budget_a,
                track_b_dims,
                max_total_dims_i,
            )
            shrink_order = sorted(alloc.keys(), key=lambda f: alloc[f], reverse=True)
            for fam in shrink_order:
                if total_alloc <= budget_a:
                    break
                min_k = 1 if fam_models[fam]["mode"] == "identity" else int(family_dim_min)
                while alloc[fam] > min_k and total_alloc > budget_a:
                    alloc[fam] -= 1
                    total_alloc -= 1

        # Fill remaining budget (keeps variance >= target).
        remaining = budget_a - sum(alloc.values())
        if remaining > 0:
            fill_order = sorted(
                alloc.keys(),
                key=lambda f: (len(fam_models[f]["cols"]), alloc[f]),
                reverse=True,
            )
            for fam in fill_order:
                if remaining <= 0:
                    break
                k_max = int(fam_models[fam].get("k_max", alloc[fam]))
                while alloc[fam] < k_max and remaining > 0:
                    alloc[fam] += 1
                    remaining -= 1

        def _transform(frame: pd.DataFrame) -> pd.DataFrame:
            parts: List[pd.DataFrame] = []
            if track_b_cols_in:
                parts.append(frame[track_b_cols_in])

            if raw_track_a_cols:
                parts.append(frame[raw_track_a_cols])

            for fam, model in fam_models.items():
                k_use = int(alloc.get(fam, 0))
                if k_use <= 0:
                    continue
                cols = model["cols"]
                x = frame[cols].astype(float).to_numpy(copy=True)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                if model["mode"] == "identity":
                    out = x[:, :k_use]
                    col_names = [f"tc_id:{fam}:{i}" for i in range(out.shape[1])]
                    parts.append(pd.DataFrame(out, index=frame.index, columns=col_names))
                    continue
                mu = model["mu"]
                sigma = model["sigma"]
                pca = model["pca"]
                xz = (x - mu) / sigma
                z = pca.transform(xz)
                z = z[:, :k_use]
                col_names = [f"tc_pca:{fam}:{i}" for i in range(z.shape[1])]
                parts.append(pd.DataFrame(z, index=frame.index, columns=col_names))

            out_frame = self._sanitize_frame(pd.concat(parts, axis=1))
            out_frame = out_frame.loc[:, ~out_frame.columns.duplicated()]
            return out_frame

        train_out = _transform(train_frame)
        val_out = _transform(val_frame) if val_frame is not None else None

        hit = 0
        pca_fams = 0
        for fam, model in fam_models.items():
            if model.get("mode") == "pca":
                pca_fams += 1
                if float(model.get("achieved_var", 0.0)) >= float(target_variance) - 1e-6:
                    hit += 1

        meta = {
            "track_b_dims": int(track_b_dims),
            "track_a_raw_dims": int(raw_a_dims),
            "track_a_pca_dims": int(max(0, train_out.shape[1] - track_b_dims - raw_a_dims)),
            "total_dims": int(train_out.shape[1]),
            "max_total_dims": int(max_total_dims_i),
            "target_variance": float(target_variance),
            "budget_a": int(budget_a),
            "families": int(len(fam_models)),
            "families_hit_target_at_k_req": int(hit),
            "pca_families": int(pca_fams),
            "raw_passthrough_families": int(len(raw_to_cols)),
        }
        return train_out, val_out, meta

    def _run_tier_two_models(
        self,
        tracks: Dict[str, TrackData],
        labels: pd.DataFrame,
        horizon: int,
        allow_lstm: bool,
        lstm_features: Optional[Dict[str, pd.DataFrame]],
        train_idx: Optional[np.ndarray] = None,
    ) -> Dict[str, TierTwoResult]:
        """Run Tier-2 LSTM sequence models on Track C.
        
        🚀 OPTUNA MODE: When Optuna is enabled, lstm_features contains only "track_c"
                        raw_passthrough_families = {fam for fam in family_feature_map.keys() if str(fam).endswith("_hf")}
        which is the optimized Track C from Optuna (not separate Track A/B views).
        
        Track C = concat(Track A × weight_a, Track B × weight_b)
        - Track A: Stage-A families → PCA/AE encoders → compressed features
        - Track B: HF blocks + Model-family summaries
        
                            raw_passthrough_families=raw_passthrough_families,
        LSTM trains ONLY on Track C with Optuna-optimized hyperparameters.
        
        🔧 CRITICAL for walk-forward: When train_idx is provided, LSTM is trained
        ONLY on train portion and evaluated on validation portion.
        """
        if not allow_lstm or not lstm_features:
                            "🧩 Track C compression applied: track_b=%d, track_a_raw=%d, track_a_pca=%d, total=%d (cap=%d, hit95=%d/%d)",

                            compress_meta.get("track_a_raw_dims", 0),
        # Extract views (in Optuna mode: only "track_c")
        view_items = [
            (name, frame)
            for name, frame in lstm_features.items()
            if not name.startswith("__") and frame is not None and not frame.empty
        ]
        if not view_items:
            return {}
        
        # Check if we're in Optuna mode (single track_c view)
        is_optuna_mode = len(view_items) == 1 and view_items[0][0] == "track_c"
        if is_optuna_mode:
            self.logger.info(
                "🚀 LSTM Tier-2 (Optuna Mode): Training on Track C only (dim=%d)",
                view_items[0][1].shape[1]
            )

        actual_returns = labels["forward_return"].astype(float)
        summaries: List[ModelSummary] = []
        predictions: Dict[str, pd.Series] = {}
        composite_scores: Dict[str, float] = {}
        tier2_metadata: Dict[str, Any] = {
            "fold_metrics": {},
            "views": [name for name, _ in view_items],
            "optuna_mode": is_optuna_mode,
            "walk_forward_mode": train_idx is not None,
        }

        for view_name, feature_frame in view_items:

            meta = lstm_features.get("__meta__", {}) if isinstance(lstm_features, dict) else {}
            track_a_cols = list(meta.get("track_a_cols", []) or [])
            track_b_cols = list(meta.get("track_b_cols", []) or [])
            family_feature_map = meta.get("family_feature_map", {}) or {}
            
            # 🔧 WALK-FORWARD MODE: Use only train portion for building sequence data
            if train_idx is not None and len(train_idx) > 0:
                train_feature_frame = feature_frame.iloc[train_idx]
                train_returns = actual_returns.iloc[train_idx]
                
                # 🔧 NO-GAP: Validation starts immediately after train (contiguous)
                # Use all indices after the last train index
                last_train_pos = train_idx[-1]
                total_samples = len(feature_frame)
                val_idx_arr = np.arange(last_train_pos + 1, total_samples)
                
                val_feature_frame = feature_frame.iloc[val_idx_arr] if len(val_idx_arr) > 0 else None
                val_returns = actual_returns.iloc[val_idx_arr] if len(val_idx_arr) > 0 else None

                # Track C compression for seq_hybrid only (leakage-safe):
                # Fit PCA per Stage-A family on TRAIN ONLY; keep Track-B columns verbatim.
                if (
                    view_name == "seq_hybrid"
                    and not is_optuna_mode
                    and track_a_cols
                    and track_b_cols
                    and family_feature_map
                ):
                    try:
                        train_feature_frame, val_feature_frame, compress_meta = self._compress_track_c_train_val(
                            train_frame=train_feature_frame,
                            val_frame=val_feature_frame,
                            track_a_cols=track_a_cols,
                            track_b_cols=track_b_cols,
                            family_feature_map=family_feature_map,
                            max_total_dims=int(getattr(self.config, "optuna_max_total_dims", 500)),
                            target_variance=0.95,
                            family_dim_min=4,
                            family_dim_max=128,
                        )
                        self.logger.info(
                            "🧩 Track C compression applied: track_b=%d, track_a_pca=%d, total=%d (cap=%d, hit95=%d/%d)",
                            compress_meta.get("track_b_dims", 0),
                            compress_meta.get("track_a_pca_dims", 0),
                            compress_meta.get("total_dims", 0),
                            compress_meta.get("max_total_dims", 0),
                            compress_meta.get("families_hit_target_at_k_req", 0),
                            compress_meta.get("families", 0),
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Track C compression failed (falling back to raw seq_hybrid): %s",
                            exc,
                        )
                
                self.logger.info(
                    "🛡️ Mamba walk-forward (no-gap): train=%d [0:%d], val=%d [%d:%d] for %s",
                    len(train_feature_frame), last_train_pos,
                    len(val_idx_arr), last_train_pos + 1, total_samples,
                    view_name
                )

                # ✅ No per-window clamping. Enforce global seq_len cap derived from fold geometry.
                seq_len_used = int(self.config.mamba_seq_len)
                seq_cap = self._mamba_seq_len_cap.get(horizon)
                if seq_cap is not None and seq_len_used > int(seq_cap):
                    raise ValueError(
                        f"mamba_seq_len={seq_len_used} exceeds global seq_len cap={int(seq_cap)} for H{horizon}. "
                        "Per-window seq_len clamping is disabled by design."
                    )

                # Build sequence data from TRAIN portion only
                seq_result = build_sequence_data(train_feature_frame, train_returns, seq_len_used)
                if seq_result is None:
                    self.logger.warning("Insufficient train data for %s view on horizon %s", view_name, horizon)
                    continue
                seq_data, scaler_stats = seq_result
                if len(seq_data) < 10:
                    self.logger.warning("Insufficient train data for %s view on horizon %s", view_name, horizon)
                    continue
                
                # Split train sequences for internal validation during training
                n_train_seq = len(seq_data)
                internal_split = int(n_train_seq * 0.8)
                internal_train_idx = np.arange(internal_split)
                internal_val_idx = np.arange(internal_split, n_train_seq)
                
                # Build validation sequence data if we have enough val samples
                # 🔧 FIX: Pass scaler_stats from training to validation for consistent normalization
                val_seq_data = None
                if val_feature_frame is None or len(val_feature_frame) <= seq_len_used:
                    raise ValueError(
                        f"Walk-forward geometry invalid for {view_name} H{horizon}: "
                        f"val_frame={0 if val_feature_frame is None else len(val_feature_frame)} <= seq_len={seq_len_used}. "
                        "Global seq_len cap should have prevented this; check window construction/alignment."
                    )
                val_result = build_sequence_data(
                    val_feature_frame,
                    val_returns,
                    seq_len_used,
                    scaler_stats=scaler_stats,
                )
                if val_result is None:
                    raise ValueError(
                        f"Failed to build val sequence data for {view_name} H{horizon} with seq_len={seq_len_used}"
                    )
                val_seq_data, _ = val_result
                self.logger.info(
                    "✅ Mamba %s: Built val_seq_data with %d sequences (val_frame=%d, seq_len=%d)",
                    view_name,
                    len(val_seq_data),
                    int(len(val_feature_frame)),
                    int(seq_len_used),
                )
                
                # Train Mamba with internal train/val split for early stopping
                try:
                    cfg_override = vars(self.config).copy()
                    cfg_override["mamba_seq_len"] = seq_len_used

                    result = train_mamba_fold(
                        seq_data,
                        internal_train_idx,  # 80% of train for training
                        internal_val_idx,    # 20% of train for early stopping
                        cfg_override,
                        warm_state=None,
                    )
                except ImportError as exc:
                    self.logger.warning("Sequence stack unavailable: %s", exc)
                    return {}
                
                # 🔧 Run inference on OUT-OF-SAMPLE validation data
                fold_predictions: List[pd.Series] = []
                fold_metrics: List[Dict[str, float]] = []
                
                if val_seq_data is not None and len(val_seq_data) > 0:
                    # Run inference on validation sequence data using trained model
                    try:
                        val_result = predict_mamba_on_data(
                            val_seq_data,
                            result["state_dict"],
                            cfg_override,
                        )
                        val_timestamps = pd.to_datetime(val_result["timestamps"])
                        val_preds_series = pd.Series(
                            val_result["preds"], index=pd.Index(val_timestamps)
                        ).sort_index()
                        fold_predictions.append(val_preds_series)
                        
                        # Compute metrics on validation predictions
                        aligned_returns = actual_returns.reindex(val_preds_series.index)
                        
                        # 🔍 DEBUG: Check prediction and return distributions
                        pred_mean = val_preds_series.mean()
                        pred_std = val_preds_series.std()
                        pred_sign_pct = (val_preds_series > 0).mean() * 100
                        ret_mean = aligned_returns.mean()
                        ret_std = aligned_returns.std()
                        self.logger.info(
                            "🔍 Mamba %s DEBUG: preds mean=%.4f std=%.4f sign%%=%.1f%% | returns mean=%.4f std=%.4f",
                            view_name, pred_mean, pred_std, pred_sign_pct, ret_mean, ret_std
                        )
                        
                        val_metrics = self._compute_metrics(
                            val_preds_series,
                            aligned_returns,
                            aligned_returns,
                            "regression",
                            horizon,
                        )
                        fold_metrics.append(val_metrics)
                        self.logger.info(
                            "✅ LSTM %s validation: %d predictions, Sharpe=%.4f",
                            view_name, len(val_preds_series), val_metrics.get("sharpe", float("nan"))
                        )
                        
                        # 🎯 Stage C: Collect predictions for prediction tape export
                        if horizon not in self._prediction_tape_folds:
                            self._prediction_tape_folds[horizon] = []
                        current_fold_id = len(self._prediction_tape_folds[horizon])
                        fold_tape_df = collect_fold_predictions(
                            fold_id=current_fold_id,
                            timestamps=val_timestamps,
                            y_true=aligned_returns.values,
                            y_pred=val_preds_series.values,
                            metadata={"view": view_name, "sharpe": val_metrics.get("sharpe", float("nan"))},
                        )
                        self._prediction_tape_folds[horizon].append(fold_tape_df)
                        
                    except Exception as exc:
                        self.logger.warning("Failed to run LSTM validation inference: %s", exc)
                        # Fallback to internal validation predictions
                        train_timestamps = pd.to_datetime(result["timestamps"])
                        preds_series = pd.Series(result["preds"], index=pd.Index(train_timestamps)).sort_index()
                        fold_predictions.append(preds_series)
                        fold_metrics.append(self._compute_metrics(
                            preds_series,
                            actual_returns.reindex(preds_series.index),
                            actual_returns.reindex(preds_series.index),
                            "regression",
                            horizon,
                        ))
                else:
                    # No val data - use internal validation predictions
                    train_timestamps = pd.to_datetime(result["timestamps"])
                    preds_series = pd.Series(result["preds"], index=pd.Index(train_timestamps)).sort_index()
                    fold_predictions.append(preds_series)
                    fold_metrics.append(self._compute_metrics(
                        preds_series,
                        actual_returns.reindex(preds_series.index),
                        actual_returns.reindex(preds_series.index),
                        "regression",
                        horizon,
                    ))
            else:
                # Standard mode - internal cross-validation
                if (
                    view_name == "seq_hybrid"
                    and not is_optuna_mode
                    and track_a_cols
                    and track_b_cols
                    and family_feature_map
                ):
                    try:
                        raw_passthrough_families = {fam for fam in family_feature_map.keys() if str(fam).endswith("_hf")}
                        feature_frame, _, compress_meta = self._compress_track_c_train_val(
                            train_frame=feature_frame,
                            val_frame=None,
                            track_a_cols=track_a_cols,
                            track_b_cols=track_b_cols,
                            family_feature_map=family_feature_map,
                            raw_passthrough_families=raw_passthrough_families,
                            max_total_dims=int(getattr(self.config, "optuna_max_total_dims", 500)),
                            target_variance=0.95,
                            family_dim_min=4,
                            family_dim_max=128,
                        )
                        self.logger.info(
                            "🧩 Track C compression applied (standard mode): track_b=%d, track_a_raw=%d, track_a_pca=%d, total=%d (cap=%d)",
                            compress_meta.get("track_b_dims", 0),
                            compress_meta.get("track_a_raw_dims", 0),
                            compress_meta.get("track_a_pca_dims", 0),
                            compress_meta.get("total_dims", 0),
                            compress_meta.get("max_total_dims", 0),
                        )
                    except Exception as exc:
                        self.logger.warning("Track C compression failed in standard mode: %s", exc)
                seq_len_used = self.config.mamba_seq_len
                seq_result = build_sequence_data(feature_frame, actual_returns, seq_len_used)
                if seq_result is None:
                    self.logger.warning("Insufficient data for %s view on horizon %s", view_name, horizon)
                    continue
                seq_data, scaler_stats = seq_result
                if len(seq_data) < 10:
                    self.logger.warning("Insufficient data for %s view on horizon %s", view_name, horizon)
                    continue
                folds = self._build_sequence_folds(len(seq_data), horizon)
                if not folds:
                    pivot = max(5, len(seq_data) // 3)
                    train_seq_idx = np.arange(0, max(1, pivot))
                    val_seq_idx = np.arange(max(1, pivot), len(seq_data))
                    folds = [(train_seq_idx, val_seq_idx)]
                fold_predictions: List[pd.Series] = []
                fold_metrics: List[Dict[str, float]] = []
                
                # 📝 Sequential fold training (Mamba-only)
                for train_seq_idx, val_seq_idx in folds:
                    try:
                        result = train_mamba_fold(
                            seq_data,
                            np.asarray(train_seq_idx),
                            np.asarray(val_seq_idx),
                            vars(self.config),
                            warm_state=None,
                        )
                    except ImportError as exc:
                        self.logger.warning("Sequence stack unavailable: %s", exc)
                        return {}
                    timestamps = pd.to_datetime(result["timestamps"])
                    preds_series = pd.Series(result["preds"], index=pd.Index(timestamps)).sort_index()
                    fold_predictions.append(preds_series)
                    metrics = self._compute_metrics(
                        preds_series,
                        actual_returns.reindex(preds_series.index),
                        actual_returns.reindex(preds_series.index),
                        "regression",
                        horizon,
                    )
                    fold_metrics.append(metrics)
            
            combined_preds = pd.concat(fold_predictions).sort_index() if fold_predictions else pd.Series(dtype=float)
            aggregate = self._aggregate_fold_metrics(fold_metrics)
            model_name = f"mamba_{view_name}"
            summary = ModelSummary(
                name=model_name,
                model_type="sequence",
                metrics=aggregate,
                params={
                    "folds": len(fold_predictions),
                    "seq_len": seq_len_used,
                    "feature_dim": seq_data.feature_dim,
                    "view": view_name,
                    "feature_names": feature_frame.columns.tolist(),
                    "walk_forward": train_idx is not None,
                },
            )
            summaries.append(summary)
            predictions[model_name] = combined_preds
            composite_scores[model_name] = aggregate.get("composite", float("nan"))
            tier2_metadata.setdefault("fold_metrics", {})[view_name] = fold_metrics
            tier2_metadata.setdefault("view_dims", {})[view_name] = seq_data.feature_dim
            tier2_metadata.setdefault("feature_counts", {})[view_name] = feature_frame.shape[1]

        if not summaries:
            return {}

        best_model = max(summaries, key=lambda s: s.metrics.get("composite", float("-inf")))
        top_models = sorted(summaries, key=lambda s: s.metrics.get("composite", float("-inf")), reverse=True)

        tier2 = TierTwoResult(
            track="TrackC",
            task="regression",
            best_model=best_model,
            top_models=top_models,
            predictions=predictions,
            composite_scores=composite_scores,
            metadata=tier2_metadata,
        )
        return {"TrackC": tier2}

    def _build_local_ensemble(
        self,
        ranking: List[ModelCandidate],
        track_b_frame: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        if not ranking:
            return None
        members = {
            "tree": next((c for c in ranking if c.model_type == "tree"), None),
            "seq": next((c for c in ranking if c.tier == "tier2"), None),
            "hf": next((c for c in ranking if c.track == "B"), None),
        }
        weights_raw: Dict[str, float] = {}
        for label, cand in members.items():
            if cand is None:
                continue
            score = max(cand.composite, 0.0)
            weights_raw[label] = max(score * cand.reliability, 0.0)
        total = sum(weights_raw.values())
        if total <= 0:
            return None
        weights = {label: weights_raw.get(label, 0.0) / total for label in members.keys()}
        series_list = [cand.predictions for cand in members.values() if cand is not None]
        index = self._union_indices(series_list)
        if index.empty:
            return None
        vol_proxy = track_b_frame.get("q_vol") if "q_vol" in track_b_frame else None
        if vol_proxy is None:
            vol_proxy = pd.Series(DEFAULT_RETURN_SCALE, index=index)
        else:
            vol_proxy = vol_proxy.reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(DEFAULT_RETURN_SCALE)
        mu_components: List[pd.Series] = []
        prob_components: List[pd.Series] = []
        ensemble_reliability = 0.0
        for label, cand in members.items():
            if cand is None or weights.get(label, 0.0) == 0:
                continue
            mu = self._candidate_to_mu(cand)
            if mu is not None:
                mu_components.append(weights[label] * mu.reindex(index).fillna(method="ffill").fillna(method="bfill"))
            prob = self._candidate_to_prob(cand, vol_proxy)
            if prob is not None:
                prob_components.append(weights[label] * prob.reindex(index).fillna(method="ffill").fillna(method="bfill"))
            ensemble_reliability += weights[label] * cand.reliability
        combined_mu = sum(mu_components) if mu_components else None
        combined_prob = sum(prob_components) if prob_components else None
        return {
            "weights": weights,
            "mu": combined_mu,
            "prob": combined_prob,
            "reliability": float(ensemble_reliability),
            "members": members,
        }

    def _generate_outputs(
        self,
        horizon: int,
        tracks: Dict[str, TrackData],
        ranking: List[ModelCandidate],
        primary: Optional[ModelCandidate],
        ensemble: Optional[Dict[str, Any]],
        selection_meta: Dict[str, Any],
        labels: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        track_b_frame = tracks.get("B").frame if "B" in tracks else pd.DataFrame()
        index = track_b_frame.index if not track_b_frame.empty else None
        if (index is None or len(index) == 0) and primary is not None:
            index = primary.predictions.index
        if index is None:
            return None
        index = pd.Index(index)
        outputs = pd.DataFrame(index=index)
        vol_proxy = track_b_frame.get("q_vol") if "q_vol" in track_b_frame else None
        if vol_proxy is None:
            vol_proxy = pd.Series(DEFAULT_RETURN_SCALE, index=index)
        else:
            vol_proxy = vol_proxy.reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(DEFAULT_RETURN_SCALE)

        mu_series = self._extract_mu_series(primary, selection_meta)
        if mu_series is None:
            mu_series = pd.Series(0.0, index=index)
        else:
            mu_series = mu_series.reindex(index).fillna(method="ffill").fillna(method="bfill")
        outputs[f"mu_hat_{horizon}d"] = mu_series

        prob_series = self._extract_prob_series(primary, selection_meta, vol_proxy)
        if prob_series is None:
            prob_series = pd.Series(0.5, index=index)
        else:
            prob_series = prob_series.reindex(index).fillna(method="ffill").fillna(method="bfill")
        outputs[f"p_up_{horizon}d"] = prob_series.clip(0.0, 1.0)

        q50 = track_b_frame.get("q_loc", mu_series).reindex(index).fillna(mu_series)
        vol = vol_proxy
        spread_proxy = 1.2815 * vol
        q10 = q50 - spread_proxy
        q90 = q50 + spread_proxy
        outputs[f"q10_{horizon}d"] = q10
        outputs[f"q50_{horizon}d"] = q50
        outputs[f"q90_{horizon}d"] = q90
        outputs[f"sigma_hat_{horizon}d"] = vol

        reliability = self._estimate_reliability_series(track_b_frame, primary).reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(0.5)
        outputs["reliability"] = reliability

        actual_returns = labels.get("forward_return", pd.Series(index=index, data=np.nan)).reindex(index)
        actual_returns = actual_returns.fillna(method="ffill").fillna(method="bfill")
        outputs[f"actual_{horizon}d"] = actual_returns
        actual_dir = labels.get("forward_direction", pd.Series(index=index, data=0.0)).reindex(index)
        actual_dir = actual_dir.fillna(method="ffill").fillna(method="bfill")
        outputs[f"actual_direction_{horizon}d"] = actual_dir

        for col in ("hf_agg_score", "hf_agg_conf"):
            if col in track_b_frame:
                outputs[col] = track_b_frame[col].reindex(index).fillna(method="ffill").fillna(method="bfill")

        outputs["model_type"] = primary.model_type if primary else "NA"
        outputs["track"] = primary.track if primary else "NA"
        features_used = primary.features_used if primary else []
        outputs["features_used"] = "|".join(features_used)
        outputs["model_id"] = (
            f"{primary.track}_{primary.tier}_{primary.model_name}" if primary else "NA"
        )

        for col, default in {
            "drift_severity": 0.0,
            "drift_flag": 0.0,
            "time_since_last_drift": 0.0,
            "ol_dir_acc": 0.5,
            "cal_ECE": 0.5,
        }.items():
            if col in track_b_frame:
                outputs[col] = track_b_frame[col].reindex(index).fillna(method="ffill").fillna(method="bfill")
            else:
                outputs[col] = default

        if ensemble:
            weights = ensemble.get("weights", {})
            outputs["w_tree"] = weights.get("tree", 0.0)
            outputs["w_seq"] = weights.get("seq", 0.0)
            outputs["w_hf"] = weights.get("hf", 0.0)
            if ensemble.get("mu") is not None:
                outputs[f"mu_hat_{horizon}d_ensemble"] = (
                    ensemble["mu"].reindex(index).fillna(method="ffill").fillna(method="bfill")
                )
            if ensemble.get("prob") is not None:
                outputs[f"p_up_{horizon}d_ensemble"] = (
                    ensemble["prob"].reindex(index).fillna(method="ffill").fillna(method="bfill")
                )
            outputs["reliability"] = (
                0.5 * outputs["reliability"] + 0.5 * float(ensemble.get("reliability", 0.5))
            )
        else:
            outputs["w_tree"] = 1.0
            outputs["w_seq"] = 0.0
            outputs["w_hf"] = 0.0

        for idx_rank, cand in enumerate(ranking[:3], start=1):
            col_name = f"pred_rank{idx_rank}_{cand.track}_{cand.model_name}"
            outputs[col_name] = cand.predictions.reindex(index).fillna(method="ffill").fillna(method="bfill")

        return outputs

    def _prepare_backtest_inputs(self, outputs: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if outputs is None or outputs.empty:
            return pd.DataFrame()
        index = outputs.index
        df = pd.DataFrame(index=index)

        def _first_existing(columns: Sequence[str]) -> Optional[pd.Series]:
            for col in columns:
                if col in outputs:
                    return outputs[col]
            return None

        mu_series = _first_existing([f"mu_hat_{horizon}d_ensemble", f"mu_hat_{horizon}d"])
        sigma_series = outputs.get(f"sigma_hat_{horizon}d")
        p_series = _first_existing([f"p_up_{horizon}d_ensemble", f"p_up_{horizon}d"])
        actual_series = outputs.get(f"actual_{horizon}d")
        if mu_series is None or sigma_series is None or actual_series is None:
            return pd.DataFrame()

        mu_vals = mu_series.astype(float)
        sigma_vals = sigma_series.astype(float)
        sigma_fallback = sigma_vals.replace(0.0, np.nan).median()
        if sigma_fallback is None or not np.isfinite(sigma_fallback):
            sigma_fallback = DEFAULT_RETURN_SCALE
        df["mu_hat"] = mu_vals
        df["sigma_hat"] = sigma_vals.replace(0.0, np.nan).fillna(sigma_fallback)
        df["p_up"] = (p_series.astype(float) if p_series is not None else pd.Series(0.5, index=index)).clip(0.0, 1.0)
        df["rho"] = outputs.get("reliability", pd.Series(0.5, index=index)).astype(float).clip(0.0, 1.0)
        df["actual_return"] = actual_series.astype(float)
        df["actual_direction"] = outputs.get(f"actual_direction_{horizon}d", pd.Series(0.0, index=index)).astype(float)
        for quant in ("q10", "q50", "q90"):
            col = f"{quant}_{horizon}d"
            if col in outputs:
                df[quant] = outputs[col].astype(float)
        for col in ("hf_agg_score", "hf_agg_conf", "drift_flag", "drift_severity"):
            if col in outputs:
                df[col] = outputs[col]
        return df.dropna(subset=["mu_hat", "sigma_hat", "actual_return"])

    def _run_backtest_for_outputs(
        self,
        horizon: int,
        outputs: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        bt_inputs = self._prepare_backtest_inputs(outputs, horizon)
        if bt_inputs.empty:
            return None, None
        price_data = self._get_price_data_for_horizon(horizon)
        if price_data is None or price_data.empty:
            self.logger.warning("Price data unavailable for backtest on %s H%s", self.config.symbol, horizon)
            return None, None
        strategy_cfg = dict(self.config.strategy or {})
        fee = float(strategy_cfg.get("fee_bp", 0.0)) / 10000.0
        slippage = float(strategy_cfg.get("slippage_bp", 0.0)) / 10000.0
        engine = BacktestEngine(price_data, fee=fee, slippage_bp=slippage)
        try:
            # 🔧 FIX: Pass symbol so backtest can load Optuna regime thresholds
            equity, metrics = engine.run(bt_inputs, horizon, strategy_cfg, symbol=self.config.symbol)
        except Exception as exc:  # pragma: no cover - safeguard
            self.logger.warning("Backtest failed for %s H%s: %s", self.config.symbol, horizon, exc)
            return None, None
        self._persist_backtest_artifacts(horizon, equity, metrics)
        return equity, metrics

    def _persist_backtest_artifacts(
        self,
        horizon: int,
        equity: Optional[pd.DataFrame],
        metrics: Optional[Dict[str, Any]],
    ) -> None:
        if equity is None or metrics is None:
            return
        outdir = self.config.backtest_output_dir / self.config.symbol.upper() / f"h{horizon}"
        outdir.mkdir(parents=True, exist_ok=True)
        equity_path = outdir / "bt_equity.parquet"
        try:
            equity.to_parquet(equity_path)
        except Exception:  # pragma: no cover - parquet fallback
            equity.to_csv(outdir / "bt_equity.csv")
        metrics_path = outdir / "bt_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, default=lambda obj: float(obj)))
        summary_path = outdir / "bt_fold_summary.csv"
        row = {"symbol": self.config.symbol, "horizon": horizon}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()})
        summary_df = pd.DataFrame([row])
        if summary_path.exists():
            summary_df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)

        # Benchmark-relative tracking (EODHD-backed) written next to bt_equity.
        # This is intended to be used by Phase2 v2 / stateful runs as well,
        # since they persist bt_equity via the same artifact writer.
        try:
            import os

            from src.analytics.benchmarking import BenchmarkFramework, compute_benchmark_timeseries, summarize_benchmark
            from src.analytics.eodhd_benchmark_data import fetch_eodhd_adjusted_close, prices_to_returns

            if "net_return" not in equity.columns:
                return

            bench_ticker = str(os.getenv("STAGEB_BENCHMARK_TICKER", "SPY")).strip().upper() or "SPY"
            bench_suffix = str(os.getenv("STAGEB_BENCHMARK_EXCHANGE_SUFFIX", ".US")).strip() or ".US"
            use_adj = str(os.getenv("STAGEB_BENCHMARK_USE_ADJUSTED", "1")).strip() not in {"0", "false", "False"}

            windows_raw = str(os.getenv("STAGEB_BENCHMARK_WINDOWS", "20,63,126"))
            windows = tuple(int(x.strip()) for x in windows_raw.split(",") if x.strip())

            rp = pd.to_numeric(equity["net_return"], errors="coerce").dropna().sort_index()
            if not isinstance(rp.index, pd.DatetimeIndex) or rp.empty:
                return

            start = rp.index.min()
            end = rp.index.max()
            px = fetch_eodhd_adjusted_close(
                ticker=bench_ticker,
                start=start,
                end=end,
                exchange_suffix=bench_suffix,
                use_adjusted_close=use_adj,
            )
            rb = prices_to_returns(px)

            ts = compute_benchmark_timeseries(
                portfolio_returns=rp,
                benchmark_returns=rb,
                windows=windows,
                risk_free=float(os.getenv("STAGEB_BENCHMARK_RF_ANNUAL", "0.0")),
                trading_days=252,
                cvar_alpha=float(os.getenv("STAGEB_BENCHMARK_CVAR_ALPHA", "0.05")),
                target_vol_annual=(
                    float(os.getenv("STAGEB_BENCHMARK_TARGET_VOL", "0"))
                    if float(os.getenv("STAGEB_BENCHMARK_TARGET_VOL", "0")) > 0
                    else None
                ),
                use_excess_alpha=str(os.getenv("STAGEB_BENCHMARK_USE_EXCESS_ALPHA", "0")).strip() in {"1", "true", "True"},
            )
            summary = summarize_benchmark(ts=ts, windows=windows)

            ts_path = outdir / "bt_benchmark_timeseries.parquet"
            ts.to_parquet(ts_path)
            (outdir / "bt_benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            fw = BenchmarkFramework(primary=bench_ticker, secondary=(), trading_days=252)
            (outdir / "bt_benchmark_framework.json").write_text(json.dumps(fw.__dict__, indent=2, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Benchmark tracking failed for %s H%s: %s", self.config.symbol, horizon, exc)

    def _union_indices(self, series_list: Sequence[pd.Series]) -> pd.Index:
        if not series_list:
            return pd.Index([])
        index = series_list[0].index
        for series in series_list[1:]:
            index = index.union(series.index)
        return index

    def _resolve_cluster(self, horizon: int) -> Optional[str]:
        for cluster, members in self.config.horizon_clusters.items():
            if horizon in members:
                return cluster
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _clone_family_specs(self) -> Dict[str, FamilySpec]:
        cloned: Dict[str, FamilySpec] = {}
        for name, spec in FAMILIES.items():
            cloned[name] = replace(spec, columns=list(spec.columns))
        return cloned

    def _find_column(self, df: pd.DataFrame, contains: Optional[str] = None, endswith: Optional[str] = None) -> Optional[str]:
        for col in df.columns:
            if contains and contains not in col:
                continue
            if endswith and not col.endswith(endswith):
                continue
            return col
        return None


class StageBRunner:
    """High-level orchestrator that wraps :class:`StageBPipeline` with the API
    outlined in the design doc. It keeps lightweight caches so each step can be
    inspected independently during development or debugging."""

    def __init__(
        self,
        cfg: StageBConfig,
        stage_a_assets: Optional[Dict[str, Any]] = None,
        family_registry: Optional[Dict[str, FamilySpec]] = None,
    ) -> None:
        self.cfg = cfg
        self.stage_a_assets = stage_a_assets or {}
        self.family_registry = family_registry or FAMILIES
        self.logger = logging.getLogger("stage_b.runner")
        self._symbol_assets: Dict[str, Tuple[Dict[Any, Dict[str, float]], Dict[Any, Dict[str, Any]]]] = {}
        self._contexts: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core helper routines
    # ------------------------------------------------------------------
    def build_panel(self, symbol: str, horizon: int) -> pd.DataFrame:
        pipeline = self._spawn_pipeline(symbol)
        priors_map, meta_map = self._ensure_symbol_assets(symbol, pipeline)
        panel = pipeline._build_panel(horizon)
        ctx = self._contexts.setdefault((symbol, horizon), {})
        ctx.update(
            {
                "pipeline": pipeline,
                "panel": panel,
                "priors_map": priors_map,
                "meta_map": meta_map,
            }
        )
        return panel

    def build_tracks(self, symbol: str, horizon: int) -> Dict[str, TrackData]:
        ctx = self._get_context(symbol, horizon)
        pipeline: StageBPipeline = ctx["pipeline"]
        panel: pd.DataFrame = ctx.get("panel") or self.build_panel(symbol, horizon)
        column_families = pipeline._infer_column_families(panel.columns)
        pipeline._attach_columns_to_specs(column_families)
        block_summaries = pipeline._compute_block_summaries(panel)
        priors = self._horizon_asset(ctx["priors_map"], horizon)
        stage_a_meta = self._horizon_asset(ctx["meta_map"], horizon)
        tracks = pipeline._construct_tracks(panel, block_summaries, column_families, priors, stage_a_meta, horizon)
        ctx.update(
            {
                "tracks": tracks,
                "column_families": column_families,
                "priors": priors,
                "stage_a_meta": stage_a_meta,
                "block_summaries": block_summaries,
            }
        )
        return tracks

    def run_tier1_models(self, symbol: str, horizon: int) -> Dict[str, TierOneResult]:
        ctx = self._get_context(symbol, horizon)
        pipeline: StageBPipeline = ctx["pipeline"]
        tracks: Dict[str, TrackData] = ctx.get("tracks") or self.build_tracks(symbol, horizon)
        panel: pd.DataFrame = ctx.get("panel") or self.build_panel(symbol, horizon)
        labels = pipeline._construct_labels(horizon, panel.index)
        tier1, s_tree_best, best_tree_context = pipeline._run_tier_one_suite(tracks, labels, horizon)
        ctx.update(
            {
                "tier1": tier1,
                "s_tree_best": s_tree_best,
                "best_tree_context": best_tree_context,
            }
        )
        return tier1

    def run_global_lstm_search(self) -> Dict[str, Any]:
        """Placeholder for the pooled LSTM search logic.

        The intention is to iterate over horizon clusters, pool data, and tune
        sequence models (e.g., via Optuna). The function currently returns the
        existing config to show where the results would be stored."""

        self.logger.info("Global LSTM search not implemented; returning existing configs")
        return self.cfg.lstm_cluster_configs

    def run_tier2_models(self, symbol: str, horizon: int) -> Dict[str, TierTwoResult]:
        ctx = self._get_context(symbol, horizon)
        pipeline: StageBPipeline = ctx["pipeline"]
        tier1: Dict[str, TierOneResult] = ctx.get("tier1") or self.run_tier1_models(symbol, horizon)
        tracks: Dict[str, TrackData] = ctx.get("tracks") or self.build_tracks(symbol, horizon)
        s_tree_best = ctx.get("s_tree_best", float("-inf"))
        best_tree_context = ctx.get("best_tree_context")
        allow_lstm = pipeline._allow_lstm(horizon, s_tree_best, tier1, best_tree_context)
        ctx["allow_lstm"] = allow_lstm
        lstm_features = (
            pipeline._prepare_lstm_features(
                panel=ctx.get("panel") or pipeline._build_panel(horizon),
                block_summaries=ctx.get("block_summaries", {}),
                column_families=ctx.get("column_families", {}),
                priors=ctx.get("priors", {}),
                stage_a_meta=ctx.get("stage_a_meta", {}),
            )
            if allow_lstm
            else None
        )
        ctx["lstm_features"] = lstm_features
        panel: pd.DataFrame = ctx.get("panel") or self.build_panel(symbol, horizon)
        ctx["panel"] = panel
        labels = pipeline._construct_labels(horizon, panel.index)
        tier2 = pipeline._run_tier_two_models(tracks, labels, horizon, allow_lstm, lstm_features)
        ctx["tier2"] = tier2
        return tier2

    def select_best_models(self, symbol: str, horizon: int) -> Dict[str, Any]:
        ctx = self._get_context(symbol, horizon)
        pipeline: StageBPipeline = ctx["pipeline"]
        tier1: Dict[str, TierOneResult] = ctx.get("tier1") or self.run_tier1_models(symbol, horizon)
        tier2: Dict[str, TierTwoResult] = ctx.get("tier2") or self.run_tier2_models(symbol, horizon)
        tracks: Dict[str, TrackData] = ctx.get("tracks") or self.build_tracks(symbol, horizon)
        panel: pd.DataFrame = ctx.get("panel") or self.build_panel(symbol, horizon)
        labels = pipeline._construct_labels(horizon, panel.index)
        ranking, primary, selection_meta = pipeline._select_best_models(tier1, tier2, tracks)
        ensemble = pipeline._build_local_ensemble(ranking, tracks.get("B").frame if "B" in tracks else pd.DataFrame())
        outputs = pipeline._generate_outputs(horizon, tracks, ranking, primary, ensemble, selection_meta, labels)
        backtest_equity, backtest_metrics = (None, None)
        if outputs is not None and not outputs.empty:
            backtest_equity, backtest_metrics = pipeline._run_backtest_for_outputs(horizon, outputs)
        result = {
            "ranking": ranking,
            "primary": primary,
            "ensemble": ensemble,
            "outputs": outputs,
            "backtest_equity": backtest_equity,
            "backtest_metrics": backtest_metrics,
            "selection_meta": selection_meta,
        }
        ctx.update(result)
        return result

    def export_outputs(
        self,
        symbol: str,
        horizon: int,
        best_models: Dict[str, Any],
        outdir: Optional[Path] = None,
    ) -> Optional[Dict[str, Path]]:
        outputs: Optional[pd.DataFrame] = best_models.get("outputs")
        if outputs is None or outputs.empty:
            self.logger.warning("No Stage-B outputs available for %s H%s", symbol, horizon)
            return None
        outdir = outdir or Path("artifacts") / "stage_b" / f"{symbol.upper()}_h{horizon}"
        outdir.mkdir(parents=True, exist_ok=True)
        parquet_path = outdir / "stage_b_outputs.parquet"
        outputs.to_parquet(parquet_path)
        primary = best_models.get("primary")
        metadata = {
            "symbol": symbol,
            "horizon": horizon,
            "primary_model": primary.model_name if primary else None,
            "primary_track": primary.track if primary else None,
            "ensemble_weights": best_models.get("ensemble", {}).get("weights"),
        }
        metadata_path = outdir / "stage_b_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        return {"parquet": parquet_path, "metadata": metadata_path}

    # ------------------------------------------------------------------
    # Batch execution helpers
    # ------------------------------------------------------------------
    def run(self, symbols: Sequence[str]) -> Dict[str, Dict[int, StageBResult]]:
        """Execute the full Stage-B pipeline for every symbol requested."""

        outputs: Dict[str, Dict[int, StageBResult]] = {}
        for symbol in symbols:
            pipeline = self._spawn_pipeline(symbol)
            results = pipeline.run()
            outputs[symbol] = results
            for horizon, stage_result in results.items():
                ctx = self._contexts.setdefault((symbol, horizon), {})
                ctx.update(
                    {
                        "pipeline": pipeline,
                        "tracks": stage_result.tracks,
                        "tier1": getattr(stage_result, "tier1", {}),  # tier1 not present in LSTM-only mode
                        "tier2": stage_result.tier2,
                        "ranking": stage_result.candidates,
                        "primary": stage_result.primary_model,
                        "ensemble": stage_result.ensemble,
                        "outputs": stage_result.outputs,
                        "backtest_equity": stage_result.backtest_equity,
                        "backtest_metrics": stage_result.backtest_metrics,
                    }
                )
        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _spawn_pipeline(self, symbol: str) -> StageBPipeline:
        cfg = replace(self.cfg, symbol=symbol)
        return StageBPipeline(cfg)

    def _ensure_symbol_assets(
        self,
        symbol: str,
        pipeline: StageBPipeline,
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[Any, Dict[str, Any]]]:
        if symbol not in self._symbol_assets:
            self._symbol_assets[symbol] = pipeline._load_stage_a_artifacts()
        return self._symbol_assets[symbol]

    def _get_context(self, symbol: str, horizon: int) -> Dict[str, Any]:
        key = (symbol, horizon)
        if key not in self._contexts:
            self.build_panel(symbol, horizon)
        return self._contexts[key]

    def _horizon_asset(self, asset_map: Dict[Any, Dict[str, Any]], horizon: int) -> Dict[str, Any]:
        return asset_map.get(horizon, asset_map.get("global", {}))


__all__ = [
    "FamilySpec",
    "PrepFamiliesSettings",
    "StageBConfig",
    "StageBPipeline",
    "StageBRunner",
    "StageBResult",
    "TierTwoResult",
    "ModelCandidate",
    "TrackData",
]
