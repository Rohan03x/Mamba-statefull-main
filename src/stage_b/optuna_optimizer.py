"""Stage-B Optuna Optimizer for LSTM Feature Selection & Hyperparameter Tuning.

This module implements the comprehensive 4-tier Optuna optimization plan:

TIER 1 - Feature Selection (Adaptive Family Inclusion)
TIER 2 - Dimensionality Selection (Per-Family Latent Size via PCA/AE)
TIER 3 - Track-A vs Track-B Weighting (Fusion Control)
TIER 4 - LSTM Hyperparameters

Track Structure:
- Track A: Stage-A Raw Families -> Encoders -> Compressed Features (~100-120 dims)
- Track B: HF Blocks + Model-Family Summaries (~15-25 features)
- Track C: concat(Track A, Track B) -> Final LSTM Input (120-150 dims max)

Objective:
    score = 0.65 * sharpe + 0.20 * rwa + 0.15 * stability
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import subprocess

import numpy as np
import pandas as pd


def _apply_ray_runtime_env_envvars(init_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Inject selected env vars into Ray runtime_env so workers see them.

    This is important for profiling flags (e.g., NVTX enablement) because Ray
    worker processes may not inherit the driver's environment in all modes.
    """
    env_vars: Dict[str, str] = {}
    for key in ("DCF_ENABLE_NVTX",):
        value = os.environ.get(key)
        if value is not None:
            env_vars[key] = value

    if not env_vars:
        return init_kwargs

    runtime_env = dict(init_kwargs.get("runtime_env") or {})
    runtime_env_envvars = dict(runtime_env.get("env_vars") or {})
    runtime_env_envvars.update(env_vars)
    runtime_env["env_vars"] = runtime_env_envvars
    init_kwargs["runtime_env"] = runtime_env
    return init_kwargs

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    Trial = None
    OPTUNA_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    PCA_AVAILABLE = True
except ImportError:
    PCA = None
    StandardScaler = None
    PCA_AVAILABLE = False


# Module-level gamma function for TPE sampler (must be at module level for Ray pickling)
def _tpe_custom_gamma(n: int) -> int:
    """Custom gamma function: use top 20% of trials as 'good' instead of default sqrt(n)*0.1.
    
    Default only uses 1-2 trials as 'good' which makes TPE unable to learn.
    With this fix: 50 trials -> top 10 good, 100 trials -> top 20 good.
    """
    return min(20, max(3, int(0.2 * n)))

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    RAY_AVAILABLE = True
    RAY_TUNE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    tune = None  # type: ignore
    OptunaSearch = None  # type: ignore
    ASHAScheduler = None  # type: ignore
    PlacementGroupSchedulingStrategy = None  # type: ignore
    RAY_AVAILABLE = False
    RAY_TUNE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Ray Tune + Optuna trial mapping helper
# -----------------------------------------------------------------------------
# Ray's OptunaSearch keeps an internal mapping from Ray Tune's `trial_id` to the
# underlying Optuna trial object. Inject the Optuna trial number into the Ray
# config so it shows up in `params.json`, `result.json`, and worker logs.
if OptunaSearch is not None:
    class StageBOptunaSearch(OptunaSearch):
        def suggest(self, trial_id: str) -> Optional[Dict]:
            params = super().suggest(trial_id)
            if params is None:
                return None
            try:
                ot_trial = getattr(self, "_ot_trials", {}).get(trial_id)
                if ot_trial is not None and hasattr(ot_trial, "number"):
                    params["_optuna_trial_number"] = int(getattr(ot_trial, "number"))
            except Exception:
                # Best-effort only; never fail trial suggestion due to logging.
                pass
            return params
else:
    StageBOptunaSearch = None  # type: ignore


LOGGER = logging.getLogger("stage_b.optuna")

from src.stage_b.sequence_models import build_sequence_data, train_lstm_fold, train_mamba_fold
from src.stage_b.meta_optimizer import MetaOptimizer, MetaOptimizerConfig, create_meta_optimizer

import os


# -----------------------------------------------------------------------------
# Ray Actor for Memory-Efficient Trial Execution
# -----------------------------------------------------------------------------
# By using an Actor, the heavy imports (PyTorch, numpy, etc.) happen ONCE per
# worker, not once per trial. This reduces memory from ~2.5GB/trial to ~400MB/trial
# since the Python environment is shared across trials in the same actor.

if RAY_AVAILABLE and ray is not None:
    @ray.remote
    class TrialEvaluatorActor:
        """Persistent Ray Actor that evaluates multiple trials efficiently.
        
        This actor is created once per worker slot. The heavy imports happen
        during __init__, and then multiple trials reuse the same Python process.
        Memory savings: ~2GB per concurrent trial (28 trials = 56GB saved).
        """
        
        def __init__(
            self,
            panel,
            column_families,
            labels,
            block_summaries,
            walk_forward_folds,
            family_columns: Dict[str, List[str]],
            family_sizes: Dict[str, int],
            stage_a_weights: Dict[str, float],
            config_dict: Dict[str, Any],
            horizon: int,
        ):
            """Initialize actor with shared data - imports happen here ONCE."""
            import logging
            self.logger = logging.getLogger("stage_b.optuna.ray_actor")
            self.logger.info("TrialEvaluatorActor initializing...")
            
            # Data is already materialized by Ray when passed to actor __init__
            # (Ray auto-deserializes ObjectRefs for actor constructor args)
            self.panel = panel
            self.column_families = column_families
            self.labels = labels
            self.block_summaries = block_summaries
            self.walk_forward_folds = walk_forward_folds
            
            # Store metadata
            self.family_columns = family_columns
            self.family_sizes = family_sizes
            self.stage_a_weights = stage_a_weights
            self.config_dict = config_dict
            self.horizon = horizon
            
            # Create optimizer ONCE per actor (not per trial)
            self.trial_config = OptunaConfig(
                ray_fold_parallelism=0,  # Sequential folds within trial
                ray_fold_gpu_fraction=0.0,
                fold_gpu_reserve_gb=config_dict["fold_gpu_reserve_gb"],
                max_total_dims=config_dict["max_total_dims"],
                sharpe_weight=config_dict["sharpe_weight"],
                rwa_weight=config_dict["rwa_weight"],
                stability_weight=config_dict["stability_weight"],
                sequence_model_type=config_dict.get("sequence_model_type", "lstm"),
            )
            self.optimizer = StageBOptunaOptimizer(config=self.trial_config, logger=self.logger)
            self.optimizer.family_columns = family_columns
            self.optimizer.family_sizes = family_sizes
            self.optimizer.stage_a_weights = stage_a_weights
            
            self.logger.info("TrialEvaluatorActor ready - optimizer created")
        
        def evaluate_trial(self, config: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate a single trial config. Called multiple times per actor."""
            import time as _time
            _t_start = _time.time()
            
            try:
                # Build family_params from config
                family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
                for family in STAGE_A_FAMILIES:
                    weight_key = f"weight_{family}"
                    dim_type_key = f"{family}_dim_type"
                    weight = config.get(weight_key, 0.0)
                    include = weight >= self.optimizer.config.family_weight_clip_min

                    # HF families must remain raw (passthrough), regardless of any dim_type keys.
                    if _is_raw_passthrough_family(family):
                        fam_size = int(self.family_sizes.get(family, 0) or 0)
                        if fam_size <= 0:
                            family_params[family] = (False, 0, "none", 0.0, {})
                        else:
                            w = max(float(weight), float(self.optimizer.config.family_weight_clip_min))
                            family_params[family] = (True, 0, "passthrough", w, {})
                        continue

                    method = config.get(dim_type_key, "pca")
                    
                    if method == "ae":
                        dim_key = f"{family}_ae_dim"
                    else:
                        dim_key = f"{family}_pca_dim"
                    dim = config.get(dim_key, 4) if include else 0
                    
                    ae_params: Dict[str, Any] = {}
                    if method == "ae":
                        ae_params = {
                            "ae_layers": config.get(f"{family}_ae_layers", 2),
                            "ae_activation": config.get(f"{family}_ae_activation", "relu"),
                            "ae_dropout": config.get(f"{family}_ae_dropout", 0.0),
                            "ae_lr": config.get(f"{family}_ae_lr", 1e-3),
                        }
                    
                    family_params[family] = (include, dim, method, weight if include else 0.0, ae_params)
                
                sequence_model_type = self.optimizer.config.sequence_model_type

                if sequence_model_type == "lstm":
                    lstm_params = {
                        "lstm_layers": config.get("lstm_layers", 2),
                        "lstm_hidden_dim": config.get("lstm_hidden_dim", 96),
                        "lstm_dropout": config.get("lstm_dropout", 0.2),
                        "lstm_seq_len": config.get("lstm_seq_len", 30),
                        "lstm_batch_size": config.get("lstm_batch_size", 64),
                        "lstm_learning_rate": config.get("lstm_learning_rate", 1e-3),
                        "lstm_optimizer": config.get("lstm_optimizer", "AdamW"),
                        "lstm_activation": config.get("lstm_activation", "relu"),
                        "lstm_use_amp": config.get("lstm_use_amp", True),
                    }
                    mamba_params: Dict[str, Any] = {}
                elif sequence_model_type == "mamba":
                    mamba_params = {
                        "mamba_d_model": config.get("mamba_d_model"),
                        "mamba_n_layers": config.get("mamba_n_layers"),
                        "mamba_ssm_dim": config.get("mamba_ssm_dim"),
                        "mamba_expand_factor": config.get("mamba_expand_factor"),
                        "mamba_seq_len": config.get("mamba_seq_len"),
                        "mamba_activation": config.get("mamba_activation", "silu"),
                        "mamba_norm_type": config.get("mamba_norm_type", "rmsnorm"),
                        "mamba_norm_strategy": config.get("mamba_norm_strategy", "pre"),
                        "mamba_dropout": config.get("mamba_dropout", 0.1),
                        "mamba_resid_dropout": config.get("mamba_resid_dropout", 0.0),
                        "mamba_ssm_dropout": config.get("mamba_ssm_dropout", 0.0),
                        "mamba_gate_dropout": config.get("mamba_gate_dropout", 0.0),
                        "mamba_optimizer": config.get("mamba_optimizer", "adamw"),
                        "mamba_learning_rate": config.get("mamba_learning_rate", 1e-3),
                        "mamba_weight_decay": config.get("mamba_weight_decay", 1e-4),
                        "mamba_grad_clip": config.get("mamba_grad_clip", 1.0),
                        "mamba_lr_scheduler": config.get("mamba_lr_scheduler", "cosine"),
                        "mamba_warmup_steps": config.get("mamba_warmup_steps", 100),
                        "mamba_max_epochs": config.get("mamba_max_epochs", 10),
                        "mamba_batch_size": config.get("mamba_batch_size", 32),
                        "mamba_loss_fn": config.get("mamba_loss_fn", "smooth_l1"),
                        "mamba_head_type": config.get("mamba_head_type", "linear"),
                        "mamba_head_hidden_dim": config.get("mamba_head_hidden_dim", 128),
                        "mamba_head_num_layers": config.get("mamba_head_num_layers", 1),
                        "mamba_head_dropout": config.get("mamba_head_dropout", 0.0),
                    }
                    lstm_params = {}
                else:
                    raise ValueError(f"Unknown sequence_model_type: {sequence_model_type}")
                
                # Build other params
                threshold_params = {
                    "threshold": config.get("threshold", 0.10),
                    "bull_long_mult": config.get("bull_long_mult", 1.0),
                    "bull_short_mult": config.get("bull_short_mult", 1.0),
                    "bear_long_mult": config.get("bear_long_mult", 1.5),
                    "bear_short_mult": config.get("bear_short_mult", 0.5),
                    "crisis_long_mult": config.get("crisis_long_mult", 3.0),
                    "crisis_short_mult": config.get("crisis_short_mult", 3.0),
                    "conf_threshold": config.get("conf_threshold", 0.5),
                    "vol_scaler": config.get("vol_scaler", 1.0),
                }
                pipeline_params = {
                    "smoothing_type": config.get("smoothing_type", "none"),
                    "smoothing_window": config.get("smoothing_window", 3),
                    "train_fraction": config.get("train_fraction", 0.8),
                }

                # Track-B per-family weights (separate category)
                track_b_family_weights: Dict[str, float] = {}
                for family in HF_BLOCK_FAMILIES:
                    track_b_family_weights[family] = float(config.get(f"weight_b_{family}", 1.0))
                for summary_name in TRACK_B_SUMMARY_BLOCKS:
                    track_b_family_weights[summary_name] = float(config.get(f"weight_b_{summary_name}", 1.0))
                
                # Evaluate (reuses self.optimizer which has heavy imports cached)
                score, metadata = self.optimizer._evaluate_trial_params(
                    self.panel,
                    self.column_families,
                    self.labels,
                    self.block_summaries,
                    self.walk_forward_folds,
                    family_params,
                    config.get("track_a_weight", 1.0),
                    config.get("track_b_weight", 1.0),
                    sequence_model_type,
                    lstm_params,
                    mamba_params,
                    threshold_params,
                    pipeline_params,
                    self.horizon,
                    track_b_family_weights=track_b_family_weights,
                    progress_callback=None,  # Actor doesn't support ASHA callbacks directly
                )
                
                elapsed = _time.time() - _t_start
                self.logger.info(f"Trial completed: score={score:.4f} in {elapsed:.1f}s")
                
                return {
                    "score": score,
                    "avg_score": metadata.get("avg_score", score),
                    "score_std": metadata.get("score_std", 0.0),
                    "n_successful_folds": metadata.get("n_successful_folds", 0),
                    "track_a_dims": metadata.get("track_a_dims", 0),
                    "track_b_dims": metadata.get("track_b_dims", 0),
                    "track_c_dims": metadata.get("track_c_dims", 0),
                    "elapsed": elapsed,
                    "done": True,
                }
                
            except Exception as e:
                import traceback
                self.logger.error(f"Trial failed: {e}\n{traceback.format_exc()}")
                return {"score": float("-inf"), "done": True, "error": str(e)}


# -----------------------------------------------------------------------------
# GPU helpers
# -----------------------------------------------------------------------------


def _detect_gpu_memory() -> Optional[Dict[str, float]]:
    """Return GPU memory (total/used/free) using nvidia-smi if available."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None

    totals: List[float] = []
    free_vals: List[float] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            total_mb = float(parts[0])
            used_mb = float(parts[1]) if len(parts) > 1 else 0.0
        except (ValueError, IndexError):
            continue

        free_mb = max(0.0, total_mb - used_mb)
        totals.append(total_mb / 1024.0)
        free_vals.append(free_mb / 1024.0)

    if not totals:
        return None

    stats: Dict[str, float] = {
        "per_device_gb": float(np.mean(totals)),
        "device_count": float(len(totals)),
        "total_gb": float(np.sum(totals)),
    }
    if free_vals:
        stats.update(
            {
                "per_device_free_gb": float(np.mean(free_vals)),
                "min_free_gb": float(np.min(free_vals)),
                "max_free_gb": float(np.max(free_vals)),
                "total_free_gb": float(np.sum(free_vals)),
            }
        )
    return stats


# =============================================================================
# 3-PILLAR DIMENSIONALITY ANALYZER
# =============================================================================
# Computes optimal latent dimension per family using:
#   1. PCA eigenvalue decay (linear signal) → dim floor
#   2. AE reconstruction elbow (nonlinear structure) → manifold dimension
#   3. Predictive screening (optional) → alpha-maximizing dimension
# Final: weighted_median([k_pca, k_ae, k_pred], weights=[1, 2, 3])

@dataclass
class FamilyDimConfig:
    """Optimal dimension configuration for a single family."""
    family_name: str
    k_pca: int              # PCA intrinsic dimension (90% variance)
    k_ae: int               # AE elbow dimension
    k_final: int            # Final weighted dimension
    method: str             # Recommended method: "pca" or "ae"
    pca_explained_90: float # Variance explained at k_pca
    ae_elbow_loss: float    # Reconstruction loss at elbow
    family_size: int        # Original feature count
    # New: refinement metrics
    noise_penalty: float = 0.0      # Noise penalty applied (0-1)
    ae_advantage: float = 0.0       # How much better AE is than PCA
    pca_weight: int = 1             # Dynamic weight for PCA in median
    ae_weight: int = 2              # Dynamic weight for AE in median


# Module-level worker function for ProcessPoolExecutor (must be at module level for pickling)
def _analyze_family_worker(
    family: str,
    X: np.ndarray,
    cols: List[str],
    device: str,
    pca_variance_threshold: float,
    dim_min: int,
    dim_max: int,
    ae_elbow_threshold: float,
) -> FamilyDimConfig:
    """Worker function for parallel family analysis using ProcessPoolExecutor.
    
    This is a module-level function to allow pickling for multiprocessing.
    """
    # Import here to ensure fresh state in subprocess
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    
    if TORCH_AVAILABLE:
        torch.set_num_threads(2)
    
    family_size = X.shape[1]

    # Raw passthrough families must not be reduced (enforced here for safety).
    if _is_raw_passthrough_family(family):
        passthrough_dim = int(family_size)
        return FamilyDimConfig(
            family_name=family,
            k_pca=passthrough_dim,
            k_ae=passthrough_dim,
            k_final=passthrough_dim,
            method="passthrough",
            pca_explained_90=1.0,
            ae_elbow_loss=0.0,
            family_size=passthrough_dim,
        )
    
    # Standardize
    if PCA_AVAILABLE:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # PILLAR 1: PCA intrinsic dimension
    k_pca, pca_explained, pca_noise_ratio = _compute_pca_dimension(
        X_scaled,
        family_size,
        variance_threshold=pca_variance_threshold,
        dim_min=dim_min,
        dim_max=dim_max,
    )
    
    # PILLAR 2: AE dimension (must also meet variance target)
    k_ae, ae_loss, ae_loss_at_k_pca = _compute_ae_dimension(
        X_scaled,
        family_size,
        k_pca,
        device,
        variance_threshold=pca_variance_threshold,
        ae_elbow_threshold=ae_elbow_threshold,
        dim_min=dim_min,
        dim_max=dim_max,
    )
    
    # Compute refinements
    ae_advantage = (k_pca - k_ae) / max(k_pca, 1)
    
    if ae_advantage > 0.2:
        pca_weight, ae_weight = 1, 3
    elif ae_advantage > 0:
        pca_weight, ae_weight = 1, 2
    elif ae_advantage > -0.2:
        pca_weight, ae_weight = 1, 1
    else:
        pca_weight, ae_weight = 2, 1
    
    noise_penalty = max(0, 1.0 - pca_explained) * pca_noise_ratio
    k_final = _weighted_median([k_pca, k_ae], [pca_weight, ae_weight])
    
    DIM_MIN, DIM_MAX = dim_min, dim_max
    if noise_penalty > 0.3:
        noise_reduction = int(k_final * noise_penalty * 0.3)
        k_final = max(DIM_MIN, k_final - noise_reduction)
    
    k_final = max(DIM_MIN, min(DIM_MAX, k_final))
    k_final = min(k_final, family_size)
    
    # Choose method: AE wins if it compresses even a little better
    if k_ae < k_pca:
        method = "ae"
    else:
        method = "pca"
    
    return FamilyDimConfig(
        family_name=family,
        k_pca=k_pca,
        k_ae=k_ae,
        k_final=k_final,
        method=method,
        pca_explained_90=pca_explained,
        ae_elbow_loss=ae_loss,
        family_size=family_size,
        noise_penalty=noise_penalty,
        ae_advantage=ae_advantage,
        pca_weight=pca_weight,
        ae_weight=ae_weight,
    )
def _compute_pca_dimension(
    X: np.ndarray,
    family_size: int,
    *,
    variance_threshold: float = 0.90,
    dim_min: int = 4,
    dim_max: int = 24,
) -> Tuple[int, float, float]:
    """Compute PCA intrinsic dimension with noise ratio estimation.

    Notes:
        This function is used by the multiprocessing worker path, so it must stay
        module-level and accept simple, pickle-friendly arguments.
    """
    DIM_MIN, DIM_MAX = dim_min, dim_max
    PCA_VARIANCE_THRESHOLD = float(variance_threshold)

    if not PCA_AVAILABLE or family_size < 2:
        return min(8, family_size), 0.0, 0.5

    n_components = min(family_size, X.shape[0] - 1)
    if n_components < 1:
        return min(8, family_size), 0.0, 0.5

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k_pca = np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1
    k_pca = max(DIM_MIN, min(DIM_MAX, int(k_pca)))
    k_pca = min(k_pca, family_size)

    explained = cumvar[min(k_pca - 1, len(cumvar) - 1)] if len(cumvar) > 0 else 0.0

    eigenvalues = pca.explained_variance_ratio_
    if len(eigenvalues) > 2:
        mid = len(eigenvalues) // 2
        top_var = float(np.sum(eigenvalues[:mid]))
        bottom_var = float(np.sum(eigenvalues[mid:]))
        noise_ratio = bottom_var / (top_var + 1e-8)
        noise_ratio = min(1.0, float(noise_ratio))
    else:
        noise_ratio = 0.3

    return int(k_pca), float(explained), float(noise_ratio)


def _compute_ae_dimension(
    X: np.ndarray,
    family_size: int,
    k_pca: int,
    device: str,
    *,
    variance_threshold: float = 0.95,
    ae_elbow_threshold: float = 0.02,
    dim_min: int = 4,
    dim_max: int = 24,
) -> Tuple[int, float, float]:
    """Compute AE dimension.

    Primary selection rule:
        Choose the smallest latent dim whose reconstruction preserves at least
        `variance_threshold` of variance (measured as 1 - SSE/SST).

    Fallback:
        If AE cannot meet the variance target within bounds, fall back to k_pca.
    """
    AE_DIM_CANDIDATES = (4, 6, 8, 10, 12, 16, 20, 24, 32)
    AE_EPOCHS = 15
    AE_LR = 1e-3
    AE_BATCH_SIZE = 64
    VARIANCE_THRESHOLD = float(variance_threshold)
    AE_ELBOW_THRESHOLD = float(ae_elbow_threshold)
    DIM_MIN, DIM_MAX = dim_min, dim_max
    
    if not TORCH_AVAILABLE or family_size < 4:
        return min(8, family_size), 0.0, 0.0
    
    candidates = [k for k in AE_DIM_CANDIDATES if k < family_size]
    if not candidates:
        return min(8, family_size), 0.0, 0.0
    
    if k_pca not in candidates and k_pca < family_size:
        candidates = sorted(set(candidates) | {k_pca})
    
    losses: Dict[int, float] = {}
    device_obj = torch.device(device)
    X_tensor = torch.FloatTensor(X)
    
    for k in candidates:
        try:
            loss = _train_ae_get_loss(X_tensor, k, family_size, device_obj, AE_EPOCHS, AE_LR, AE_BATCH_SIZE)
            losses[k] = loss
        except Exception:
            continue
    
    if not losses:
        return min(8, family_size), 0.0, 0.0
    
    # Sort dimensions from smallest to largest
    sorted_k = sorted(losses.keys())

    # Prefer smallest k achieving variance target (same threshold as PCA)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    ss_tot = float(np.sum(X_centered * X_centered))
    ss_tot = max(ss_tot, 1e-12)

    explained: Dict[int, float] = {}
    for k, mse in losses.items():
        sse = float(mse) * float(X.shape[0]) * float(X.shape[1])
        var_expl = 1.0 - (sse / ss_tot)
        explained[k] = float(np.clip(var_expl, 0.0, 1.0))

    eligible = [k for k in sorted_k if explained.get(k, 0.0) >= VARIANCE_THRESHOLD]
    if eligible:
        k_ae = eligible[0]
        k_ae = max(DIM_MIN, min(DIM_MAX, k_ae))
        k_ae = min(k_ae, family_size)
        ae_loss_at_k_pca = losses.get(k_pca, losses.get(sorted_k[-1], 0.0))
        return k_ae, losses.get(k_ae, 0.0), ae_loss_at_k_pca

    # If AE can't hit variance target, fall back to k_pca to enforce the same constraint.
    k_ae = max(DIM_MIN, min(DIM_MAX, int(k_pca)))
    k_ae = min(k_ae, family_size)
    ae_loss_at_k_pca = losses.get(k_pca, losses.get(sorted_k[-1], 0.0))
    return k_ae, losses.get(k_ae, ae_loss_at_k_pca), ae_loss_at_k_pca
    
    # (Unreachable) variance-target selection returns above; keep elbow logic as a dead-simple fallback.


def _train_ae_get_loss(
    X_tensor: "torch.Tensor",
    latent_dim: int,
    input_dim: int,
    device: "torch.device",
    epochs: int,
    lr: float,
    batch_size: int,
) -> float:
    """Train a simple autoencoder and return final reconstruction loss."""
    # Simple 2-layer AE
    encoder = nn.Sequential(
        nn.Linear(input_dim, max(latent_dim * 2, 16)),
        nn.ReLU(),
        nn.Linear(max(latent_dim * 2, 16), latent_dim),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, max(latent_dim * 2, 16)),
        nn.ReLU(),
        nn.Linear(max(latent_dim * 2, 16), input_dim),
    )
    
    encoder.to(device)
    decoder.to(device)
    X_tensor = X_tensor.to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    n_samples = X_tensor.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = X_tensor[batch_idx]
            
            optimizer.zero_grad()
            encoded = encoder(batch)
            decoded = decoder(encoded)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
    
    # Final loss on full data
    with torch.no_grad():
        encoded = encoder(X_tensor)
        decoded = decoder(encoded)
        final_loss = criterion(decoded, X_tensor).item()
    
    return final_loss


def _weighted_median(values: List[int], weights: List[int]) -> int:
    """Compute weighted median of values."""
    expanded = []
    for v, w in zip(values, weights):
        expanded.extend([v] * w)
    expanded.sort()
    return expanded[len(expanded) // 2]


class ThreePillarDimensionalityAnalyzer:
    """Computes optimal latent dimensions for each family using 3-pillar system.
    
    This is run ONCE before optimization to pre-compute the optimal dimension
    for each family, replacing the wide Optuna search space with targeted dims.
    
    The 3 pillars:
        1. PCA: Find smallest k where cumulative_variance >= 0.95
        2. AE: Find elbow in reconstruction loss curve
        3. Predictive: (Optional) Test dims in mini-LSTM for validation Sharpe
    
    Final dimension: weighted_median([k_pca, k_ae], weights=[1, 2])
    Clamped to [4, 32] for stability.
    
    Uses parallel processing across families for faster analysis.
    """
    
    # AE candidate dimensions for elbow detection
    AE_DIM_CANDIDATES: Tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 32)
    
    # AE training config (fast for analysis)
    AE_EPOCHS: int = 15
    AE_LR: float = 1e-3
    AE_BATCH_SIZE: int = 64
    
    # Dimension bounds
    DIM_MIN: int = 4
    DIM_MAX: int = 32
    
    # PCA variance threshold
    PCA_VARIANCE_THRESHOLD: float = 0.95
    
    # AE elbow threshold (relative to loss at smallest dim)
    AE_ELBOW_THRESHOLD: float = 0.02
    
    # Parallel processing settings (using ProcessPoolExecutor)
    MAX_PARALLEL_WORKERS: int = 12
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or LOGGER
        self.family_configs: Dict[str, FamilyDimConfig] = {}
        
        # Set global thread limits to prevent explosion
        if TORCH_AVAILABLE:
            torch.set_num_threads(2)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # Already set or parallel work started
    
    def analyze_all_families(
        self,
        panel: pd.DataFrame,
        family_columns: Dict[str, List[str]],
        device: str = "cpu",
    ) -> Dict[str, FamilyDimConfig]:
        """Analyze all families and compute optimal dimensions.
        
        Uses parallel processing for speed.
        
        Args:
            panel: Feature panel DataFrame
            family_columns: Dict mapping family name -> list of column names
            device: Device for AE training ("cpu" or "cuda")
        
        Returns:
            Dict mapping family name -> FamilyDimConfig
        """
        # Set thread limits to prevent contention with parallel workers
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        
        self.logger.info("=" * 60)
        self.logger.info("3-PILLAR DIMENSIONALITY ANALYSIS (Parallel)")
        self.logger.info("=" * 60)
        
        # Prepare family data for parallel processing
        family_data = []
        
        for family, cols in family_columns.items():
            if not cols:
                continue

            # Enforce raw passthrough policy (no dimensionality reduction)
            if _is_raw_passthrough_family(family):
                passthrough_dim = int(len(cols))
                self.family_configs[family] = FamilyDimConfig(
                    family_name=family,
                    k_pca=passthrough_dim,
                    k_ae=passthrough_dim,
                    k_final=passthrough_dim,
                    method="passthrough",
                    pca_explained_90=1.0,
                    ae_elbow_loss=0.0,
                    family_size=passthrough_dim,
                )
                continue
            
            # Prepare data for this family
            # fillna handles pd.NA from pandas 2.x nullable dtypes before numpy conversion
            family_panel = panel[cols].fillna(0.0)
            X = family_panel.values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            family_data.append(
                (
                    family,
                    X,
                    cols,
                    device,
                    float(self.PCA_VARIANCE_THRESHOLD),
                    int(self.DIM_MIN),
                    int(self.DIM_MAX),
                    float(self.AE_ELBOW_THRESHOLD),
                )
            )
        
        # Process families in parallel using ProcessPoolExecutor
        # Process-based parallelism avoids GIL and PyTorch threading issues
        if family_data:
            self.logger.info(f"  Analyzing {len(family_data)} families (parallel, {self.MAX_PARALLEL_WORKERS} workers)...")
            import sys
            sys.stdout.flush()
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            
            # Use 'spawn' context to avoid fork issues with PyTorch
            ctx = mp.get_context('spawn')
            
            with ProcessPoolExecutor(max_workers=self.MAX_PARALLEL_WORKERS, mp_context=ctx) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(_analyze_family_worker, *args): args[0]
                    for args in family_data
                }
                
                completed = 0
                for future in as_completed(futures):
                    family = futures[future]
                    completed += 1
                    try:
                        config = future.result(timeout=300)  # 5 min timeout per family
                        self.family_configs[family] = config
                        self.logger.info(
                            f"  [{completed}/{len(family_data)}] {family}: size={config.family_size} -> "
                            f"k_pca={config.k_pca}, k_ae={config.k_ae}, k_final={config.k_final}, method={config.method}"
                        )
                        sys.stdout.flush()
                    except Exception as e:
                        self.logger.warning(f"  [{completed}/{len(family_data)}] {family}: analysis failed ({e}), using fallback dim=8")
                        # family_data tuples are:
                        #   (family, X, cols, device, pca_variance, dim_min, dim_max, ae_elbow)
                        cols_for_family = next((args[2] for args in family_data if args[0] == family), [])
                        fallback = FamilyDimConfig(
                            family_name=family,
                            k_pca=8,
                            k_ae=8,
                            k_final=8,
                            method="pca",
                            pca_explained_90=0.0,
                            ae_elbow_loss=0.0,
                            family_size=len(cols_for_family) if cols_for_family else 8,
                        )
                        self.family_configs[family] = fallback
        
        # Summary statistics
        total_dims = sum(c.k_final for c in self.family_configs.values())
        avg_dim = total_dims / len(self.family_configs) if self.family_configs else 0
        self.logger.info(f"Total dims across {len(self.family_configs)} families: {total_dims} (avg={avg_dim:.1f})")
        
        return self.family_configs
    
    def _analyze_family_parallel(
        self,
        family: str,
        X: np.ndarray,
        cols: List[str],
        device: str,
    ) -> FamilyDimConfig:
        """Analyze a single family - for parallel execution."""
        # Limit torch threads to avoid contention in parallel execution
        if TORCH_AVAILABLE:
            torch.set_num_threads(2)
        
        family_size = X.shape[1]
        
        # Standardize
        if PCA_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # PILLAR 1: PCA intrinsic dimension
        k_pca, pca_explained, pca_noise_ratio = self._compute_pca_dimension_with_noise(X_scaled, family_size)
        
        # PILLAR 2: AE elbow detection
        k_ae, ae_loss, ae_loss_at_k_pca = self._compute_ae_dimension_with_comparison(
            X_scaled, family_size, k_pca, device
        )
        
        # Compute refinements (same as before)
        ae_advantage = (k_pca - k_ae) / max(k_pca, 1)
        
        if ae_advantage > 0.2:
            pca_weight, ae_weight = 1, 3
        elif ae_advantage > 0:
            pca_weight, ae_weight = 1, 2
        elif ae_advantage > -0.2:
            pca_weight, ae_weight = 1, 1
        else:
            pca_weight, ae_weight = 2, 1
        
        noise_penalty = max(0, 1.0 - pca_explained) * pca_noise_ratio
        k_final = self._weighted_median([k_pca, k_ae], [pca_weight, ae_weight])
        
        if noise_penalty > 0.3:
            noise_reduction = int(k_final * noise_penalty * 0.3)
            k_final = max(self.DIM_MIN, k_final - noise_reduction)
        
        k_final = max(self.DIM_MIN, min(self.DIM_MAX, k_final))
        k_final = min(k_final, family_size)
        
        dim_diff = abs(k_pca - k_ae)
        if dim_diff <= 2:
            method = "pca"
        elif k_ae < k_pca - 2:
            method = "ae"
        else:
            method = "pca"
        
        return FamilyDimConfig(
            family_name=family,
            k_pca=k_pca,
            k_ae=k_ae,
            k_final=k_final,
            method=method,
            pca_explained_90=pca_explained,
            ae_elbow_loss=ae_loss,
            family_size=family_size,
            noise_penalty=noise_penalty,
            ae_advantage=ae_advantage,
            pca_weight=pca_weight,
            ae_weight=ae_weight,
        )
    
    def _compute_pca_dimension_with_noise(self, X: np.ndarray, family_size: int) -> Tuple[int, float, float]:
        """Compute PCA intrinsic dimension with noise ratio estimation.
        
        Returns:
            (k_pca, explained_variance, noise_ratio)
            noise_ratio: fraction of variance in tail (eigenvalue decay flatness)
        """
        if not PCA_AVAILABLE or family_size < 2:
            return min(8, family_size), 0.0, 0.5
        
        # Run full PCA
        n_components = min(family_size, X.shape[0] - 1)
        if n_components < 1:
            return min(8, family_size), 0.0, 0.5
        
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)
        
        # Find smallest k where cumulative variance >= 90%
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k_pca = np.searchsorted(cumvar, self.PCA_VARIANCE_THRESHOLD) + 1
        k_pca = max(self.DIM_MIN, min(self.DIM_MAX, k_pca))
        k_pca = min(k_pca, family_size)
        
        explained = cumvar[min(k_pca - 1, len(cumvar) - 1)] if len(cumvar) > 0 else 0.0
        
        # Compute noise ratio: how flat is the eigenvalue tail?
        # Flat tail = more noise, steep decay = more signal
        eigenvalues = pca.explained_variance_ratio_
        if len(eigenvalues) > 2:
            # Noise ratio = variance in bottom 50% / variance in top 50%
            mid = len(eigenvalues) // 2
            top_var = np.sum(eigenvalues[:mid])
            bottom_var = np.sum(eigenvalues[mid:])
            noise_ratio = bottom_var / (top_var + 1e-8)
            noise_ratio = min(1.0, noise_ratio)  # Cap at 1.0
        else:
            noise_ratio = 0.3  # Default for small families
        
        return k_pca, float(explained), float(noise_ratio)
    
    def _compute_ae_dimension_with_comparison(
        self,
        X: np.ndarray,
        family_size: int,
        k_pca: int,
        device: str,
    ) -> Tuple[int, float, float]:
        """Compute AE elbow dimension with comparison to PCA dimension.
        
        Returns:
            (k_ae, ae_loss_at_elbow, ae_loss_at_k_pca)
            ae_loss_at_k_pca: Reconstruction loss when using same dim as PCA
        """
        if not TORCH_AVAILABLE or family_size < 4:
            return min(8, family_size), 0.0, 0.0
        
        # Filter candidates to those smaller than family size
        candidates = [k for k in self.AE_DIM_CANDIDATES if k < family_size]
        if not candidates:
            return min(8, family_size), 0.0, 0.0
        
        # Ensure k_pca is in candidates for comparison
        if k_pca not in candidates and k_pca < family_size:
            candidates = sorted(set(candidates) | {k_pca})
        
        # Train AE at each candidate dim and record reconstruction loss
        losses: Dict[int, float] = {}
        
        torch.set_num_threads(2)  # Limit threads
        device_obj = torch.device(device)
        
        X_tensor = torch.FloatTensor(X)
        
        for k in candidates:
            try:
                loss = self._train_ae_get_loss(X_tensor, k, device_obj)
                losses[k] = loss
            except Exception:
                continue
        
        if not losses:
            return min(8, family_size), 0.0, 0.0
        
        # Prefer the smallest latent dim that meets the same variance target as PCA.
        # Variance explained is measured as 1 - SSE/SST, where SSE is derived from MSE.
        sorted_k = sorted(losses.keys())
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        ss_tot = float(np.sum(X_centered * X_centered))
        ss_tot = max(ss_tot, 1e-12)

        explained: Dict[int, float] = {}
        for k, mse in losses.items():
            sse = float(mse) * float(X.shape[0]) * float(X.shape[1])
            var_expl = 1.0 - (sse / ss_tot)
            explained[k] = float(np.clip(var_expl, 0.0, 1.0))

        eligible = [k for k in sorted_k if explained.get(k, 0.0) >= float(self.PCA_VARIANCE_THRESHOLD)]
        if eligible:
            k_ae = eligible[0]
        else:
            # If AE can't meet the variance target within bounds, fall back to k_pca.
            k_ae = int(k_pca)
        
        k_ae = max(self.DIM_MIN, min(self.DIM_MAX, k_ae))
        k_ae = min(k_ae, family_size)
        
        # Get loss at k_pca for comparison
        ae_loss_at_k_pca = losses.get(k_pca, losses.get(sorted_k[-1], 0.0))
        
        return k_ae, losses.get(k_ae, 0.0), ae_loss_at_k_pca
    
    def _compute_ae_dimension(self, X: np.ndarray, family_size: int, device: str) -> Tuple[int, float]:
        """Legacy: Compute AE elbow dimension by testing multiple latent sizes."""
        k_ae, ae_loss, _ = self._compute_ae_dimension_with_comparison(X, family_size, 8, device)
        return k_ae, ae_loss
    
    def _train_ae_get_loss(self, X_tensor: "torch.Tensor", latent_dim: int, device: "torch.device") -> float:
        """Train a simple AE and return final reconstruction loss."""
        input_dim = X_tensor.shape[1]
        
        model = SimpleAutoEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_layers=2,
            activation="relu",
            dropout=0.0,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.AE_LR)
        criterion = nn.MSELoss()
        
        X_dev = X_tensor.to(device)
        dataset = torch.utils.data.TensorDataset(X_dev, X_dev)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.AE_BATCH_SIZE, shuffle=True, num_workers=0
        )
        
        model.train()
        final_loss = 0.0
        for epoch in range(self.AE_EPOCHS):
            epoch_loss = 0.0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                _, decoded = model(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(loader)
        
        # Cleanup
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        return final_loss
    
    def _weighted_median(self, values: List[int], weights: List[int]) -> int:
        """Compute weighted median of values."""
        if not values:
            return 8
        
        # Expand values by weights
        expanded = []
        for v, w in zip(values, weights):
            expanded.extend([v] * w)
        
        expanded.sort()
        mid = len(expanded) // 2
        
        if len(expanded) % 2 == 0:
            return (expanded[mid - 1] + expanded[mid]) // 2
        else:
            return expanded[mid]
    
    def get_family_dim(self, family: str) -> int:
        """Get the optimal dimension for a family."""
        if family in self.family_configs:
            return self.family_configs[family].k_final
        return 8  # Default fallback
    
    def get_family_method(self, family: str) -> str:
        """Get the recommended method (pca/ae) for a family."""
        if family in self.family_configs:
            return self.family_configs[family].method
        return "pca"  # Default fallback


# =============================================================================
# Configuration and Constants
# =============================================================================

# Stage-A families (engineering baseline + pipeline additions)
STAGE_A_FAMILIES: Tuple[str, ...] = (
    "alternative_signals", "cboe_term", "correlation", "cross_asset", "dcf",
    "index_constituents",
    "dividends", "doc_embedding_novelty_hf", "earnings", "earnings_transcript_hf",
    "fin_g2", "fin_g3", "fin_g4", "fin_g5", "fin_g6", "fin_g7", "finbert",
    "garch_iv", "macro_tst_hf", "microstructure", "candle_mechanics", "ml_framework", "multiasset",
    "options", "options_anchoring", "peer_screener_context", "regime", "short_interest", "subsidiary",
    "tft_features",
)

# HF Block families (6 total - always included)
HF_BLOCK_FAMILIES: Tuple[str, ...] = (
    "tech_micro_hf", "forecast_hf", "vol_deriv_hf", 
    "macro_regime_hf", "fundamental_val_hf", "news_nlp_hf",
)

# Model-based families (4 total - always included in Track B)
MODEL_FAMILIES: Tuple[str, ...] = (
    "quantile_forecast", "arima_forecast", "calibration", "online_learning",
)

# Track-B summary blocks (4 total - always included in Track B)
# NOTE: These are the keys used in `block_summaries` (not the MODEL_FAMILIES names).
TRACK_B_SUMMARY_BLOCKS: Tuple[str, ...] = (
    "quantile",
    "calibration",
    "online",
    "arima",
)

# REMOVED: Hardcoded AE_FAMILIES and ENCODING_CONFIG
# Optuna now chooses dim_reduction_type per family per trial
# See OptunaConfig for the new hyperparameter bounds:
#   - dim_reduction_type: "pca" or "ae" (Optuna chooses)
#   - pca_components: [10, 60]
#   - ae_latent_dim: [8, 64]
#   - ae_layers: [1, 3]
#   - ae_activation: relu, leakyrelu, gelu
#   - ae_dropout: [0.0, 0.4]
#   - ae_learning_rate: [1e-5, 1e-3]


def _is_raw_passthrough_family(family: str) -> bool:
    """Families that must remain raw (no PCA/AE compression).

    Policy: ALL families use raw features (dimensionality reduction disabled)
    """
    # Return True for all families - no dimensionality reduction
    return True


@dataclass
class OptunaConfig:
    """Configuration for Optuna optimization."""

    # Optional symbol tag for per-symbol studies.
    # When set, it is included in the Optuna study fingerprint/name so
    # different symbols do not accidentally share the same sqlite-backed study.
    symbol: Optional[str] = None
    
    # Optimization settings
    n_trials: int = 100  # Can be overridden via CLI
    timeout: int = 7200  # 2 hours for 150 trials
    # Default to sequential Optuna execution to avoid GPU/VRAM contention.
    # (Parallel trial execution is handled explicitly via Ray Tune when enabled.)
    n_jobs: int = 1

    # Multi-symbol pooled mode: minimum participating symbols required for a fold
    # to be included in scoring/cap calculations.
    #
    # Policy B (minimum-coverage gating): a fold only participates if it has at
    # least this many symbols with enough labeled validation rows to be scorable
    # for the chosen seq_len.
    min_symbols_per_fold: int = 8
    
    # =========================================================================
    # PIPELINE-LEVEL HYPERPARAMETERS (NOT LSTM-specific)
    # =========================================================================
    # These are tuned at the pipeline level, not inside LSTM training
    
    # 1. Feature Smoothing (applied in PCM / Feature Preprocessing)
    # Smooths raw features before they enter the model
    smoothing_types: Tuple[str, ...] = ("none", "ema", "sma", "gaussian")
    smoothing_window_min: int = 3
    smoothing_window_max: int = 21
    smoothing_window_step: int = 3
    
    # 2. Train Fraction (applied in Walk-Forward Data Splitter)
    # Controls train/val split within each walk-forward window
    train_fraction_min: float = 0.6
    train_fraction_max: float = 0.9
    
    # Objective weights
    sharpe_weight: float = 0.65
    rwa_weight: float = 0.20
    stability_weight: float = 0.15
    
    # Feature constraints
    max_total_dims: int = 500  # Track C max dimensions (target ~450-500)
    min_track_a_dims: int = 80
    max_track_a_dims: int = 120
    min_track_b_dims: int = 15
    max_track_b_dims: int = 30
    
    # Track weighting bounds
    track_a_weight_min: float = 0.5
    track_a_weight_max: float = 1.5
    track_b_weight_min: float = 0.5
    track_b_weight_max: float = 1.5
    
    # =========================================================================
    # SEQUENCE MODEL (LSTM with optional attention)
    # =========================================================================
    # TFT removed - using LSTM with comprehensive attention instead
    
    # Horizon for dynamic hyperparameter formulas (set by pipeline)
    horizon: int = 21  # Default H=21, overridden by pipeline
    
    # =========================================================================
    # SEQUENCE MODEL TYPE (determines which model to optimize)
    # =========================================================================
    # "lstm" = optimize LSTM architecture
    # "mamba" = optimize Mamba (SSM) architecture
    # These are separate optimization runs, not competing choices in same trial
    sequence_model_type: str = "mamba"  # Default to Mamba (Mamba-only Stage B)

    # =========================================================================
    # GLOBAL MULTI-SYMBOL MODE
    # =========================================================================
    # When enabled, a trial trains ONE shared sequence model per fold on a pooled
    # batch (concat on batch axis only) across multiple symbols.
    # This is primarily used for global multi-symbol Optuna runs.
    global_multi_symbol: bool = False
    # If True (and global_multi_symbol=True), sample family weights separately per
    # symbol (symbol-prefixed weight keys) while keeping other params global/static.
    global_per_symbol_weights: bool = False
    global_symbols: Tuple[str, ...] = ()
    
    # =========================================================================
    # HORIZON-BASED FORMULAS (dynamic bounds computed from H)
    # =========================================================================
    # seq_len: H*1.0 to H*1.5 (reduced from 3.0 to avoid seq_len > validation size)
    seq_len_horizon_mult_min: float = 1.0
    seq_len_horizon_mult_max: float = 1.5
    
    # hidden_dim: (32 + 0.8*H) to (64 + 2.0*H)
    hidden_base_min: int = 32
    hidden_base_max: int = 64
    hidden_horizon_mult_min: float = 0.8
    hidden_horizon_mult_max: float = 2.0
    
    # dropout: 0.05 to min(0.6, 0.05 + 0.004*H)
    dropout_base: float = 0.05
    dropout_horizon_mult: float = 0.004
    dropout_max_cap: float = 0.6
    
    # =========================================================================
    # LSTM-SPECIFIC HYPERPARAMETERS
    # =========================================================================
    # 
    # HORIZON-DEPENDENT (formula-based, computed from H):
    #   - lstm_seq_len: H × [1.0, 3.0]
    #   - lstm_hidden_dim: [32 + 0.8H, 64 + 2.0H]
    #   - lstm_dropout: [0.05, min(0.6, 0.05 + 0.004H)]
    #   - lstm_attention_heads: [log2(H), log2(H) + 3] (if attention enabled)
    #
    # GLOBAL (Optuna-tuned, H-independent):
    #   - layers, activation, cell_type
    #   - regularization (recurrent_dropout, input_dropout, grad_clip)
    #   - enhancements (layer_norm, residual)
    #   - output (fc_layers, fc_hidden, output_dropout)
    #   - optimization (optimizer, lr, batch_size)
    #
    # HARD-CODED (not optimized):
    #   - lstm_bidirectional = False (forecasting uses causal only)
    #   - lstm_attention = False (use TFT for attention)
    #   - lstm_use_amp = True (always use mixed precision)
    # =========================================================================
    
    # 2.1 Core Architecture (Global Optuna)
    lstm_layers_min: int = 1
    lstm_layers_max: int = 5  # Extended range for deeper networks
    lstm_cell_types: Tuple[str, ...] = ("lstm", "gru")
    lstm_activations: Tuple[str, ...] = ("relu", "gelu", "tanh", "selu")  # No leaky_relu for LSTM gates
    
    # 2.1b Initialization (critical for LSTM stability)
    lstm_recurrent_kernel_inits: Tuple[str, ...] = ("xavier_uniform", "xavier_normal", "orthogonal")
    lstm_hidden_state_inits: Tuple[str, ...] = ("zeros", "learned", "normal")
    
    # 2.2 Regularization (Global Optuna)
    lstm_recurrent_dropout_min: float = 0.0
    lstm_recurrent_dropout_max: float = 0.4
    lstm_input_dropout_min: float = 0.0
    lstm_input_dropout_max: float = 0.3
    lstm_grad_clip_min: float = 0.0
    lstm_grad_clip_max: float = 5.0
    
    # 2.2b Time-wise dropout (dropout over time dimension)
    lstm_time_dropout_min: float = 0.0
    lstm_time_dropout_max: float = 0.3
    
    # 2.2c L2 recurrent weight regularization
    lstm_recurrent_weight_decay_min: float = 0.0
    lstm_recurrent_weight_decay_max: float = 1e-3
    
    # 2.3 Enhancements (Global Optuna)
    lstm_layer_norm_choices: Tuple[bool, ...] = (True, False)
    lstm_residual_choices: Tuple[bool, ...] = (True, False)
    lstm_skip_connect_choices: Tuple[bool, ...] = (True, False)  # Skip connections between LSTM layers
    
    # 2.4 Output Layers (Global Optuna)
    lstm_fc_layers_min: int = 0
    lstm_fc_layers_max: int = 2
    lstm_fc_hidden_min: int = 32
    lstm_fc_hidden_max: int = 256
    lstm_output_dropout_min: float = 0.0
    lstm_output_dropout_max: float = 0.5
    
    # 2.4b Output activation (for normalized returns vs direction prob)
    lstm_output_activations: Tuple[str, ...] = ("none", "tanh", "sigmoid")
    
    # 2.5 Optimization (Global Optuna)
    # Extended optimizer list including advanced options (Ranger, Novograd, Lookahead, RAdam, NAdam)
    lstm_optimizers: Tuple[str, ...] = (
        "Adam", "AdamW", "SGD", "RMSprop",
        "Ranger", "Novograd", "Lookahead", "RAdam", "NAdam"
    )
    lstm_lr_min: float = 1e-5
    lstm_lr_max: float = 3e-3
    lstm_batch_sizes: Tuple[int, ...] = (32, 64, 128, 192, 256)  # max 256
    
    # 2.5b Gradient accumulation (stabilizes training with small batch)
    lstm_gradient_accumulation_min: int = 1
    lstm_gradient_accumulation_max: int = 4
    
    # 2.5c Optimizer momentum (only for SGD/RMSprop)
    lstm_momentum_min: float = 0.5
    lstm_momentum_max: float = 0.99
    
    # =========================================================================
    # 3. CNN FRONT-END PARAMETERS (STAGE B FRONT)
    # Dilated residual CNN used at Jane Street, Jump, Two Sigma
    # =========================================================================
    
    # 3.1 CNN Enable/Disable
    cnn_frontend_enabled: Tuple[bool, ...] = (True, False)
    
    # 3.2 CNN Structural Parameters
    cnn_blocks_min: int = 1
    cnn_blocks_max: int = 3
    cnn_filters: Tuple[int, ...] = (32, 48, 64, 96, 128, 192)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9)
    cnn_dilations: Tuple[int, ...] = (1, 2, 4)
    cnn_pooling: Tuple[Optional[int], ...] = (None, 2, 4)
    cnn_strides: Tuple[int, ...] = (1, 2)
    cnn_activations: Tuple[str, ...] = ("relu", "gelu", "mish", "silu")
    
    # 3.3 CNN Regularization
    cnn_dropout_min: float = 0.05
    cnn_dropout_max: float = 0.5
    cnn_batch_norm_choices: Tuple[bool, ...] = (True, False)
    cnn_layer_norm_choices: Tuple[bool, ...] = (True, False)
    
    # 3.4 CNN Residual Connections (critical for deep networks)
    cnn_residual_choices: Tuple[bool, ...] = (True, False)
    
    # =========================================================================
    # 3.5 BIDIRECTIONAL LSTM (NEW - tunable instead of hard-coded)
    # =========================================================================
    # Note: Bidirectional uses future info, may be okay for some use cases
    lstm_bidirectional_choices: Tuple[bool, ...] = (True, False)
    
    # =========================================================================
    # 3.6 ADVANCED REGULARIZATION (NEW)
    # =========================================================================
    # Label smoothing: prevents overconfident predictions
    label_smoothing_min: float = 0.0
    label_smoothing_max: float = 0.2
    
    # Stochastic depth: randomly drop layers during training (ResNet/ViT style)
    stochastic_depth_min: float = 0.0
    stochastic_depth_max: float = 0.2
    
    # Mixout: interpolates between dropout and keeping original weights
    mixout_prob_min: float = 0.0
    mixout_prob_max: float = 0.2
    
    # =========================================================================
    # 3.7 ATTENTION ENHANCEMENTS (NEW)
    # =========================================================================
    # Multi-layer attention: stack multiple attention layers
    attn_layers_min: int = 1
    attn_layers_max: int = 3
    
    # Rotary positional embedding (RoPE): modern position encoding
    rotary_embedding_choices: Tuple[bool, ...] = (True, False)
    
    # Feedforward dimension in attention block (MLP expansion)
    feedforward_dim_min: int = 64
    feedforward_dim_max: int = 512
    
    # =========================================================================
    # 3.8 HEAD ARCHITECTURE (NEW)
    # =========================================================================
    # Dense/FC activation choices
    dense_activations: Tuple[str, ...] = ("relu", "gelu", "mish", "silu", "tanh")
    
    # Batch normalization in output head
    batch_norm_head_choices: Tuple[bool, ...] = (True, False)
    
    # =========================================================================
    # 3.9 ADDITIONAL OPTIMIZERS (NEW)
    # =========================================================================
    # Extended optimizer list including advanced options
    lstm_optimizers_extended: Tuple[str, ...] = (
        "Adam", "AdamW", "SGD", "RMSprop",
        "Ranger", "Novograd", "Lookahead", "RAdam", "NAdam"
    )
    
    # =========================================================================
    # 3.10 ADDITIONAL LR SCHEDULERS (NEW)
    # =========================================================================
    # Extended scheduler list
    lr_schedulers_extended: Tuple[str, ...] = (
        "cosine", "plateau", "none", "step", "warmup_cosine", "cyclical", "one_cycle"
    )
    
    # Step scheduler params
    step_lr_step_size_min: int = 10
    step_lr_step_size_max: int = 50
    step_lr_gamma: float = 0.1
    
    # Cyclical LR params
    cyclical_base_lr_min: float = 1e-5
    cyclical_base_lr_max: float = 1e-4
    cyclical_max_lr_min: float = 1e-3
    cyclical_max_lr_max: float = 1e-2
    cyclical_step_size_min: int = 100
    cyclical_step_size_max: int = 1000
    
    # One cycle params
    one_cycle_pct_start: float = 0.3  # Fraction of cycle spent increasing LR
    
    # =========================================================================
    # 4. ADDITIONAL LSTM OPTIMIZATIONS (NEW)
    # 4.1 Recurrent Weight Dropout (AWD-LSTM style)
    lstm_weight_dropout_min: float = 0.0
    lstm_weight_dropout_max: float = 0.5
    
    # 4.2 Zoneout (RNN stabilizer, improves gradient flow)
    lstm_zoneout_min: float = 0.0
    lstm_zoneout_max: float = 0.2
    
    # 4.3 Layer-wise Learning Rate Scaling (deeper layers learn slower)
    lstm_lr_multiplier_min: float = 0.5
    lstm_lr_multiplier_max: float = 2.0
    
    # 4.4 Sequence Noise Injection (improves generalization)
    lstm_sequence_noise_min: float = 0.0
    lstm_sequence_noise_max: float = 0.03
    
    # =========================================================================
    # LSTM ATTENTION HYPERPARAMETERS (Global Optuna, not horizon-dependent)
    # =========================================================================
    # 5.1 Attention Type Selection
    # none = no attention (pure LSTM)
    # bahdanau = additive attention (original seq2seq)
    # luong = dot-product attention (simpler, faster)
    # scaled_dot = scaled dot-product (transformer-style, lightweight)
    lstm_attention_types: Tuple[str, ...] = ("none", "bahdanau", "luong", "scaled_dot")
    
    # 5.2 Attention Hidden Dimension (MLP expressiveness)
    lstm_attn_hidden_dim_min: int = 32
    lstm_attn_hidden_dim_max: int = 256
    
    # 5.3 Attention Dropout (prevents attention collapse)
    lstm_attn_dropout_min: float = 0.0
    lstm_attn_dropout_max: float = 0.4
    
    # 5.4 Attention Normalization
    # softmax = standard
    # sparsemax = sparse attention (good for noisy indicators)
    # entmax = hybrid smooth-sparse (SOTA for finance)
    lstm_attn_normalization_types: Tuple[str, ...] = ("softmax", "sparsemax", "entmax")
    
    # 5.5 Number of Attention Heads (lightweight multi-head, not full Transformer)
    lstm_attn_heads_min: int = 1
    lstm_attn_heads_max: int = 4
    
    # 5.6 Attention Scoring Function
    # dot = Luong simple
    # general = linear transform before dot
    # concat = Bahdanau-style concatenation
    lstm_attn_score_functions: Tuple[str, ...] = ("dot", "general", "concat")
    
    # 5.7 Context Vector Merge Type
    # concat = concatenate context with LSTM output
    # add = additive fusion
    # gate = learnable gating (best for noisy features)
    lstm_attn_merge_types: Tuple[str, ...] = ("concat", "add", "gate")
    
    # 5.8 Positional Encoding for Attention (helps for long sequences)
    lstm_attn_positional_encoding_choices: Tuple[bool, ...] = (True, False)
    
    # 5.9 Attention Temperature (stabilizes gradients)
    # smaller = sharper attention, larger = smoother attention
    lstm_attn_temperature_min: float = 0.3
    lstm_attn_temperature_max: float = 2.0
    
    # 5.10 Attention Regularizers (improve generalization in regime shifts)
    # Entropy regularization: penalty for overly sharp attention
    lstm_attn_entropy_reg_min: float = 0.0
    lstm_attn_entropy_reg_max: float = 0.2
    # Distance regularization: penalty for attending only to recent timesteps
    lstm_attn_distance_reg_min: float = 0.0
    lstm_attn_distance_reg_max: float = 0.1
    
    # 5.11 Attention Context Window (how many timesteps attention sees)
    # Range is [5, seq_len] but seq_len computed dynamically
    lstm_attn_context_length_min: int = 5
    lstm_attn_context_length_ratio_max: float = 1.0  # Fraction of seq_len
    
    # 5.12 Key/Value Projection Dimensions (for scaled_dot attention)
    lstm_attn_key_dim_min: int = 16
    lstm_attn_key_dim_max: int = 128
    lstm_attn_value_dim_min: int = 16
    lstm_attn_value_dim_max: int = 128
    
    # =========================================================================
    # 4.3 LOSS FUNCTION CHOICE (Optuna-tuned)
    # =========================================================================
    # Huber stabilizes rare shocks, Quantile for directional probability,
    # Gaussian NLL gives uncertainty estimation
    loss_functions: Tuple[str, ...] = ("mse", "huber", "quantile", "nll_gauss")
    huber_delta_min: float = 0.5   # Huber delta parameter
    huber_delta_max: float = 2.0
    quantile_alpha_min: float = 0.1  # Quantile loss alpha (asymmetry)
    quantile_alpha_max: float = 0.9
    nll_min_var: float = 1e-4  # Minimum variance for NLL stability
    
    # =========================================================================
    # 4.8 REGULARIZATION & NOISE INJECTION (Optuna-tuned)
    # =========================================================================
    # Small noise on input improves generalization sharply
    input_noise_std_min: float = 0.0
    input_noise_std_max: float = 0.05
    weight_decay_min: float = 0.0
    weight_decay_max: float = 1e-3
    # Note: lstm_dropout already exists above, used for dropout_rate
    
    # =========================================================================
    # 4.9 LEARNING RATE SCHEDULER (Optuna-tuned)
    # =========================================================================
    # Affects convergence more than people think
    # Extended scheduler list including step, warmup_cosine, cyclical, one_cycle
    lr_schedulers: Tuple[str, ...] = (
        "cosine", "plateau", "none", "step", "warmup_cosine", "cyclical", "one_cycle"
    )
    # Cosine params
    cosine_t_max_min: int = 10   # Min T_max for cosine annealing
    cosine_t_max_max: int = 100  # Max T_max
    cosine_eta_min: float = 1e-6  # Minimum LR for cosine
    # Plateau params
    plateau_patience_min: int = 3
    plateau_patience_max: int = 15
    plateau_factor: float = 0.5  # LR reduction factor
    
    # =========================================================================
    # 4.10 TRAINING CONTROL (Critical for preventing overtraining)
    # =========================================================================
    # Early stopping: THE main control - stops when validation stops improving
    early_stopping_patience_min: int = 5
    early_stopping_patience_max: int = 30
    
    # Max epochs: upper bound CAP, not target (most folds converge in 20-80)
    max_epochs_min: int = 50
    max_epochs_max: int = 300
    
    # Warmup steps: prevents exploding gradients early, helps AdamW adapt
    warmup_steps_min: int = 0
    warmup_steps_max: int = 500
    
    # =========================================================================
    # 4.11 TRAINING STABILITY PARAMETERS (Hedge Fund Best Practices)
    # =========================================================================
    # EMA decay for model weights (stabilizes returns forecasts)
    ema_decay_min: float = 0.90
    ema_decay_max: float = 0.9999
    ema_enabled: Tuple[bool, ...] = (True, False)
    
    # AMP precision configuration
    # bf16 for forward/backward pass, fp32 for critical numerics
    amp_precision_choices: Tuple[str, ...] = ("fp32", "fp16", "bf16")
    
    # Max gradient norm for clipping (stability)
    max_grad_norm_min: float = 0.5
    max_grad_norm_max: float = 2.0
    
    # Training sequence length (lookback window in days)
    train_seq_length_min: int = 90
    train_seq_length_max: int = 365
    
    # Target sequence length (forecast horizon)
    target_seq_length_min: int = 1
    target_seq_length_max: int = 5
    
    # =========================================================================
    # 4.12 REGIME-AWARE PARAMETERS (Stage C Enhanced)
    # =========================================================================
    # Volatility regime detection window (days)
    volatility_regime_window_min: int = 20
    volatility_regime_window_max: int = 60
    
    # Market regime detection model
    market_regime_models: Tuple[str, ...] = ("HMM", "VIX-feature", "PCA-regime")
    
    # =========================================================================
    # STEP 7: Stage-C Threshold & Regime Adjustment Parameters (UPGRADED)
    # =========================================================================
    # Base threshold bounds (|pred| < T_regime -> do nothing)
    # Renamed for clarity: threshold_base = threshold_min/max
    threshold_min: float = 0.05
    threshold_max: float = 0.25
    # Alias for clarity
    threshold_base_min: float = 0.05
    threshold_base_max: float = 0.25
    
    # Regime-aware multipliers - SEPARATE LONG/SHORT per regime
    # Bull regime: aggressive longs, normal shorts
    # threshold_multiplier_bull = [0.5, 1.2] per your spec
    bull_long_mult_min: float = 0.5
    bull_long_mult_max: float = 1.2
    bull_short_mult_min: float = 0.8
    bull_short_mult_max: float = 1.5
    # Alias for threshold_multiplier_bull
    threshold_multiplier_bull_min: float = 0.5
    threshold_multiplier_bull_max: float = 1.2
    
    # Bear regime: conservative longs, aggressive shorts
    # threshold_multiplier_bear = [1.0, 2.0] per your spec
    bear_long_mult_min: float = 1.0
    bear_long_mult_max: float = 2.0
    bear_short_mult_min: float = 0.6
    bear_short_mult_max: float = 1.2
    # Alias for threshold_multiplier_bear
    threshold_multiplier_bear_min: float = 1.0
    threshold_multiplier_bear_max: float = 2.0
    
    # Crisis regime: very conservative both sides
    # threshold_multiplier_crisis = [2.0, 4.0] per your spec
    crisis_long_mult_min: float = 2.0
    crisis_long_mult_max: float = 4.0
    crisis_short_mult_min: float = 1.5
    crisis_short_mult_max: float = 3.0
    # Alias for threshold_multiplier_crisis
    threshold_multiplier_crisis_min: float = 2.0
    threshold_multiplier_crisis_max: float = 4.0
    
    # Confidence gating threshold (pred_conf < conf_threshold -> signal = 0)
    conf_threshold_min: float = 0.3
    conf_threshold_max: float = 0.8
    
    # Volatility scaler: pred_norm = pred / (vol20/vol252)**vol_scaler
    vol_scaler_min: float = 0.0
    vol_scaler_max: float = 1.0
    
    # =========================================================================
    # FAMILY WEIGHT SYSTEM (replaces boolean toggles)
    # =========================================================================
    # Weight range for Stage-A families (continuous, not boolean)
    family_weight_min: float = 0.0  # Minimum weight (0 = excluded)
    family_weight_max: float = 1.0  # Maximum weight
    family_weight_clip_min: float = 0.01  # Clip to prevent zero-coverage
    
    # Stage-B mandatory families minimum weight (quantile_forecast, arima, etc.)
    stage_b_family_min_weight: float = 0.5  # Mandatory families get at least 0.5
    
    # Weight normalization settings
    weight_normalization: str = "sparsemax"  # "sparsemax", "softmax", "sum", or "none"
    weight_temperature: float = 1.0  # Temperature for normalization (lower = more sparse)
    
    # =========================================================================
    # DIMENSION REDUCTION (3-PILLAR SYSTEM)
    # =========================================================================
    # NEW: 3-pillar dimensionality analysis replaces wide Optuna search
    # Computes optimal dim per family using PCA + AE + weighted combination
    use_three_pillar_dims: bool = True  # Enable 3-pillar pre-computed dimensions

    # =========================================================================
    # WEIGHTS-ONLY OPTIMIZATION MODE
    # =========================================================================
    # When enabled, Optuna/Ray Tune will tune ONLY per-family weights (and Track-B
    # per-family weights). All other hyperparameters (model, thresholds, pipeline,
    # dims/encoders) are frozen to `fixed_params`.
    tune_weights_only: bool = False
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    
    # 3-pillar bounds (final k_final is clamped to these)
    three_pillar_dim_min: int = 4   # Minimum dim (avoid underspecification)
    three_pillar_dim_max: int = 32  # Maximum dim (allows hitting ~95% variance on larger families)
    three_pillar_pca_variance: float = 0.95  # Variance threshold for PCA
    three_pillar_ae_elbow: float = 0.02  # Elbow threshold for AE
    
    # =========================================================================
    # 🔵 MAMBA SEQUENCE MODEL HYPERPARAMETERS (NEW)
    # =========================================================================
    # Mamba (Structured State Space Model) is an alternative to LSTM/Transformers
    # Optimized for long-range dependencies with O(n) complexity vs O(n²) attention
    # Particularly effective for financial time series with regime shifts
    
    # =========================================================================
    # A. CORE MAMBA ARCHITECTURE (HIGHEST IMPACT)
    # =========================================================================
    # These determine model capacity, memory depth, regime learning
    
    # Model dimension (hidden state size)
    mamba_d_model_choices: Tuple[int, ...] = (96, 128, 160, 192, 224, 256, 288)
    
    # Number of Mamba layers (stacked SSM blocks)
    mamba_n_layers_choices: Tuple[int, ...] = (3, 4, 5, 6, 8)
    
    # SSM state dimension (internal memory)
    mamba_ssm_dim_choices: Tuple[int, ...] = (64, 96, 128, 160)
    
    # Expansion factor (MLP intermediate dimension = d_model * expand_factor)
    mamba_expand_factor_choices: Tuple[float, ...] = (2.0, 2.5, 3.0, 4.0)
    
    # Sequence length bounds for Mamba
    mamba_seq_len_choices: Tuple[int, ...] = (128, 160, 192, 224, 256, 288, 315)
    
    # Activation function for gates/projections
    mamba_activation_choices: Tuple[str, ...] = ("silu", "gelu")
    
    # =========================================================================
    # B. NORMALIZATION & DROPOUT (HIGH IMPACT)
    # =========================================================================
    # Controls stability, noise suppression, generalization
    
    # Normalization type
    mamba_norm_type_choices: Tuple[str, ...] = ("rmsnorm", "layernorm")
    
    # Normalization strategy (pre-norm more stable for SSMs)
    mamba_norm_strategy_choices: Tuple[str, ...] = ("pre", "post")
    
    # Global dropout (applied to MLP and residual connections)
    mamba_dropout_min: float = 0.05
    mamba_dropout_max: float = 0.30
    
    # Residual dropout (applied after each block)
    mamba_resid_dropout_choices: Tuple[float, ...] = (0.0, 0.05, 0.10)
    
    # SSM-specific dropout (applied to state space computations)
    mamba_ssm_dropout_choices: Tuple[float, ...] = (0.0, 0.02, 0.05)
    
    # Gate dropout (applied to gating mechanisms)
    mamba_gate_dropout_choices: Tuple[float, ...] = (0.0, 0.02, 0.05)
    
    # =========================================================================
    # C. TRAINING DYNAMICS (HIGH IMPACT)
    # =========================================================================
    # Determines convergence quality and Sharpe stability
    
    # Optimizer choice (Lion often wins for finance)
    mamba_optimizer_choices: Tuple[str, ...] = ("adamw", "lion")
    
    # Learning rate (log-uniform sampling)
    mamba_lr_min: float = 1e-4
    mamba_lr_max: float = 3e-3
    
    # Weight decay (log-uniform sampling)
    mamba_weight_decay_min: float = 1e-6
    mamba_weight_decay_max: float = 5e-4
    
    # Gradient clipping (critical for SSM stability)
    mamba_grad_clip_choices: Tuple[float, ...] = (0.5, 1.0, 2.0)
    
    # =========================================================================
    # D. TRAINING SCHEDULE (MEDIUM-HIGH IMPACT)
    # =========================================================================
    # Finance models need stable warmup & decay
    
    # LR scheduler type
    mamba_lr_scheduler_choices: Tuple[str, ...] = ("cosine", "one_cycle", "linear_warmup_cosine")
    
    # Warmup steps (prevents early instability)
    mamba_warmup_steps_choices: Tuple[int, ...] = (50, 100, 200)
    
    # Max epochs for Stage-B folds
    mamba_max_epochs_choices: Tuple[int, ...] = (8, 10, 12, 15)
    
    # Batch size (CRITICAL: smaller = noisier gradients = higher Sharpe)
    mamba_batch_size_choices: Tuple[int, ...] = (16, 32, 48, 64)
    
    # =========================================================================
    # E. LOSS & OUTPUT STRUCTURE (HIGH IMPACT FOR SHARPE)
    # =========================================================================
    # Determines how Mamba turns hidden state into trading signal
    
    # Loss function
    mamba_loss_fn_choices: Tuple[str, ...] = ("smooth_l1", "huber", "bce_logits", "mse")
    
    # Output head type (MLP often improves Sharpe)
    mamba_head_type_choices: Tuple[str, ...] = ("linear", "mlp")
    
    # MLP head configuration (when head_type="mlp")
    mamba_head_hidden_dim_min: int = 64
    mamba_head_hidden_dim_max: int = 256
    mamba_head_num_layers_choices: Tuple[int, ...] = (1, 2, 3)
    mamba_head_dropout_min: float = 0.0
    mamba_head_dropout_max: float = 0.3
    
    # =========================================================================
    # F. THRESHOLD & SIGNAL PARAMETERS (ALREADY EXIST ABOVE)
    # =========================================================================
    # These are shared across all sequence models (LSTM/Mamba):
    #   - threshold_min/max (base_threshold)
    #   - bull_long_mult_min/max, bull_short_mult_min/max
    #   - bear_long_mult_min/max, bear_short_mult_min/max
    #   - crisis_long_mult_min/max, crisis_short_mult_min/max
    # These parameters are CRITICAL FOR SHARPE and already defined in the
    # threshold section above (lines ~1580-1620)
    
    # Legacy: Optuna dim search (used when use_three_pillar_dims=False)
    dim_reduction_types: Tuple[str, ...] = ("pca", "ae")  # Optuna picks one per family
    
    # PCA hyperparameters (legacy)
    pca_components_min: int = 4    # Aligned with 3-pillar min
    pca_components_max: int = 24   # Aligned with 3-pillar max
    
    # AutoEncoder hyperparameters (Optuna-tuned)
    ae_latent_dim_min: int = 4     # Aligned with 3-pillar min
    ae_latent_dim_max: int = 24    # Aligned with 3-pillar max
    ae_layers_choices: Tuple[int, ...] = (1, 2)  # Reduced: 1-2 layers (simpler is better)
    ae_activation_choices: Tuple[str, ...] = ("relu", "gelu")  # Reduced: relu, gelu
    ae_dropout_min: float = 0.0    # No dropout
    ae_dropout_max: float = 0.3    # Reduced max dropout
    ae_lr_min: float = 1e-4        # Tightened LR range
    ae_lr_max: float = 1e-3        # Tightened LR range
    ae_epochs: int = 20            # Training epochs for AE (reduced for speed)
    
    # Random state
    seed: int = 42

    # =========================================================================
    # META-OPTIMIZER (Self-Learning System)
    # =========================================================================
    use_meta_optimizer: bool = True  # Enable self-learning meta-optimizer by default
    meta_optimizer_history_size: int = 200  # Trial history size
    meta_optimizer_elite_percentile: float = 0.10  # Top 10% = elite
    meta_optimizer_min_trials: int = 20  # Min trials before adaptation
    meta_optimizer_save_dir: Optional[Path] = None  # Directory to persist meta-optimizer state

    # Ray integration
    use_ray: bool = False
    use_ray_tune: bool = True  # Use Ray Tune for trial-level parallelism (recommended)
    # Default to 1 concurrent trial to give the sequence model full GPU/VRAM.
    ray_tune_concurrent_trials: int = 1
    ray_cpus_per_trial: int = 0  # 0 = auto-split CPUs across concurrent trials
    # Request the full GPU by default; the effective gpu_per_trial is still capped by
    # 1 / concurrent_trials in the Ray Tune trainable setup.
    ray_gpus_per_trial: float = 1.0
    ray_max_concurrent_trials: Optional[int] = None
    ray_address: Optional[str] = None
    ray_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    ray_fold_parallelism: int = 1  # 1 fold at a time for single GPU
    ray_fold_gpu_fraction: Optional[float] = 1.0  # Full GPU per fold
    fold_gpu_memory_gb: float = 8.0  # RTX 3070 has 8GB VRAM
    fold_gpu_reserve_gb: float = 1.0  # Keep 1GB reserved for system/CUDA overhead

    # =====================================================================
    # OPTUNA STUDY PERSISTENCE / OVERRIDES
    # =====================================================================
    # If provided, forces OptunaSearch to reuse this exact study name (and DB file)
    # rather than computing a new search-space fingerprinted name.
    # Can also be provided via env var STAGE_B_OPTUNA_STUDY_NAME.
    study_name_override: Optional[str] = None
    # When True (or when study_name_override is set), do NOT create a new “reset_*”
    # study on dynamic search-space errors; instead fail fast so the user can
    # intentionally manage study names.
    disable_study_auto_reset: bool = False


@dataclass
class OptimizedParams:
    """Container for optimized parameters from Optuna."""
    
    # =========================================================================
    # PIPELINE-LEVEL HYPERPARAMETERS (NOT LSTM-specific)
    # =========================================================================
    # Feature Smoothing (applied in PCM / Feature Preprocessing)
    smoothing_type: str = "none"  # none, ema, sma, gaussian
    smoothing_window: int = 5  # [3, 21] step 3
    
    # Train Fraction (applied in Walk-Forward Data Splitter)
    train_fraction: float = 0.8  # [0.6, 0.9]
    
    # =========================================================================
    # Tier 1: Family inclusion (legacy - kept for backward compatibility)
    # =========================================================================
    included_families: Dict[str, bool] = field(default_factory=dict)
    
    # Tier 1 NEW: Family weights (continuous, replaces boolean toggles)
    # Weight in [0.0, 1.0] - applied to encoded features before concatenation
    family_weights: Dict[str, float] = field(default_factory=dict)

    # Optional: per-symbol family weights for GLOBAL pooled mode.
    # Keys are upper-case symbols. Values mirror family_weights / track_b_family_weights.
    family_weights_by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Tier 1B: Track-B family weights (separate category from Stage-A families)
    # Keys:
    # - HF block families: e.g. "tech_micro_hf"
    # - Track-B summaries: "quantile", "calibration", "online", "arima"
    track_b_family_weights: Dict[str, float] = field(default_factory=dict)

    # Optional: per-symbol Track-B weights for GLOBAL pooled mode.
    track_b_family_weights_by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Tier 2: Per-family dimensionality
    family_dimensions: Dict[str, int] = field(default_factory=dict)
    family_encoders: Dict[str, str] = field(default_factory=dict)  # 'pca' or 'ae'
    
    # Tier 3: Track weighting
    track_a_weight: float = 1.0
    track_b_weight: float = 1.0
    
    # Tier 4: LSTM hyperparameters
    # =========================================================================
    # HORIZON-DEPENDENT (formula-based):
    #   lstm_seq_len, lstm_hidden_dim, lstm_dropout, lstm_attention_heads
    # GLOBAL (Optuna-tuned):
    #   layers, activation, cell_type, regularization, output, optimization
    # HARD-CODED (not optimized):
    #   bidirectional=False, attention=False, use_amp=True
    # =========================================================================
    
    # Core architecture (Global Optuna)
    lstm_layers: int = 2
    lstm_cell_type: str = "lstm"  # "lstm" or "gru"
    lstm_activation: str = "relu"
    lstm_recurrent_kernel_init: str = "orthogonal"  # xavier_uniform, xavier_normal, orthogonal
    lstm_hidden_state_init: str = "zeros"  # zeros, learned, normal
    
    # Horizon-dependent (formula-based)
    lstm_seq_len: int = 30
    lstm_hidden_dim: int = 96
    lstm_dropout: float = 0.2
    
    # Regularization (Global Optuna)
    lstm_recurrent_dropout: float = 0.0
    lstm_input_dropout: float = 0.0
    lstm_grad_clip: float = 1.0
    lstm_time_dropout: float = 0.0  # Dropout over time dimension
    lstm_recurrent_weight_decay: float = 0.0  # L2 recurrent weight regularization
    
    # Advanced regularization (NEW - Section 4)
    lstm_weight_dropout: float = 0.0  # AWD-LSTM style
    lstm_zoneout: float = 0.0  # RNN stabilizer
    lstm_lr_multiplier: float = 1.0  # Layer-wise LR scaling
    lstm_sequence_noise_std: float = 0.0  # Sequence noise injection
    
    # Enhancements (Global Optuna)
    lstm_layer_norm: bool = False
    lstm_residual: bool = False
    lstm_skip_connect: bool = False  # Skip connections between LSTM layers
    
    # Output layers (Global Optuna)
    lstm_fc_layers: int = 1
    lstm_fc_hidden: int = 64
    lstm_output_dropout: float = 0.2
    lstm_output_activation: str = "none"  # none, tanh, sigmoid
    
    # Optimization (Global Optuna)
    lstm_optimizer: str = "AdamW"
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 64
    lstm_gradient_accumulation: int = 1  # Gradient accumulation steps
    lstm_momentum: float = 0.9  # Only for SGD/RMSprop
    
    # HARD-CODED (not in Optuna search)
    lstm_bidirectional: bool = False  # Forecasting = causal only
    lstm_use_amp: bool = True  # Always use mixed precision
    
    # =========================================================================
    # Tier 4a: LSTM ATTENTION HYPERPARAMETERS (Global Optuna)
    # =========================================================================
    # Attention type selection
    lstm_attention_type: str = "none"  # none, bahdanau, luong, scaled_dot
    
    # Attention architecture
    lstm_attn_hidden_dim: int = 64  # MLP expressiveness [32, 256]
    lstm_attn_dropout: float = 0.1  # Prevents attention collapse [0, 0.4]
    lstm_attn_heads: int = 1  # Lightweight multi-head [1, 4]
    
    # Attention scoring and normalization
    lstm_attn_normalization: str = "softmax"  # softmax, sparsemax, entmax
    lstm_attn_score_fn: str = "dot"  # dot, general, concat
    lstm_attn_merge: str = "concat"  # concat, add, gate
    
    # Attention enhancements
    lstm_attn_positional_encoding: bool = False  # Helps for long sequences
    lstm_attn_temperature: float = 1.0  # [0.3, 2.0] stabilizes gradients
    
    # Attention regularizers (improve generalization in regime shifts)
    lstm_attn_entropy_reg: float = 0.0  # [0, 0.2] penalty for sharp attention
    lstm_attn_distance_reg: float = 0.0  # [0, 0.1] penalty for recency bias
    
    # Attention context window and projections
    lstm_attn_context_length: int = 30  # How many timesteps attention sees [5, seq_len]
    lstm_attn_key_dim: int = 64  # Key projection dim for scaled_dot [16, 128]
    lstm_attn_value_dim: int = 64  # Value projection dim for scaled_dot [16, 128]
    
    # =========================================================================
    # Tier 5: TRAINING HYPERPARAMETERS
    # =========================================================================
    # 4.3 Loss function choice
    loss_fn: str = "mse"  # mse, huber, quantile, nll_gauss
    huber_delta: float = 1.0  # Delta for Huber loss
    quantile_alpha: float = 0.5  # Alpha for quantile loss
    
    # 4.8 Regularization & noise injection
    input_noise_std: float = 0.0  # Gaussian noise on inputs
    weight_decay: float = 0.0  # L2 regularization
    
    # 4.9 Learning rate scheduler
    lr_scheduler: str = "none"  # cosine, plateau, none
    cosine_t_max: int = 50  # T_max for cosine annealing
    plateau_patience: int = 5  # Patience for plateau scheduler
    
    # 4.10 Training control (critical for preventing overtraining)
    early_stopping_patience: int = 10  # Main control: stop when validation stops improving [5, 30]
    max_epochs: int = 100  # Upper bound CAP, not target [50, 300]
    warmup_steps: int = 100  # Prevents exploding gradients early [0, 500]

    # =========================================================================
    # Tier 4b: MAMBA HYPERPARAMETERS (Global Optuna)
    # =========================================================================
    # NOTE: Stage-B is Mamba-only at runtime, but we keep the legacy LSTM fields
    # for backwards compatibility with older cached artifacts.
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
    
    # Step 7: Stage-C Threshold & Regime Adjustment
    # Base threshold: |pred| < T_regime -> do nothing
    threshold: float = 0.10
    bull_mult: float = 0.8   # T_bull = T * bull_mult (lower = more aggressive)
    bear_mult: float = 1.5   # T_bear = T * bear_mult (higher = more conservative)
    crisis_mult: float = 3.0 # T_crisis = T * crisis_mult (much higher = very conservative)
    
    # Metadata
    best_score: float = 0.0
    trial_number: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            # Pipeline-level hyperparameters
            "smoothing_type": self.smoothing_type,
            "smoothing_window": self.smoothing_window,
            "train_fraction": self.train_fraction,
            # Family params
            "included_families": self.included_families,
            "family_weights": self.family_weights,
            "family_weights_by_symbol": self.family_weights_by_symbol,
            "track_b_family_weights": self.track_b_family_weights,
            "track_b_family_weights_by_symbol": self.track_b_family_weights_by_symbol,
            "family_dimensions": self.family_dimensions,
            "family_encoders": self.family_encoders,
            "track_a_weight": self.track_a_weight,
            "track_b_weight": self.track_b_weight,
            # LSTM/shared params
            "lstm_layers": self.lstm_layers,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_dropout": self.lstm_dropout,
            "lstm_seq_len": self.lstm_seq_len,
            "lstm_batch_size": self.lstm_batch_size,
            "lstm_learning_rate": self.lstm_learning_rate,
            "lstm_optimizer": self.lstm_optimizer,
            "lstm_activation": self.lstm_activation,
            "lstm_use_amp": self.lstm_use_amp,
            # LSTM Architecture
            "lstm_bidirectional": self.lstm_bidirectional,
            "lstm_cell_type": self.lstm_cell_type,
            "lstm_recurrent_kernel_init": self.lstm_recurrent_kernel_init,
            "lstm_hidden_state_init": self.lstm_hidden_state_init,
            # LSTM Regularization
            "lstm_recurrent_dropout": self.lstm_recurrent_dropout,
            "lstm_input_dropout": self.lstm_input_dropout,
            "lstm_grad_clip": self.lstm_grad_clip,
            "lstm_time_dropout": self.lstm_time_dropout,
            "lstm_recurrent_weight_decay": self.lstm_recurrent_weight_decay,
            "lstm_weight_dropout": self.lstm_weight_dropout,
            "lstm_zoneout": self.lstm_zoneout,
            "lstm_sequence_noise_std": self.lstm_sequence_noise_std,
            # LSTM Enhancements
            "lstm_layer_norm": self.lstm_layer_norm,
            "lstm_residual": self.lstm_residual,
            "lstm_skip_connect": self.lstm_skip_connect,
            "lstm_lr_multiplier": self.lstm_lr_multiplier,
            # LSTM Attention (new comprehensive attention system)
            "lstm_attention_type": self.lstm_attention_type,
            "lstm_attn_hidden_dim": self.lstm_attn_hidden_dim,
            "lstm_attn_dropout": self.lstm_attn_dropout,
            "lstm_attn_heads": self.lstm_attn_heads,
            "lstm_attn_normalization": self.lstm_attn_normalization,
            "lstm_attn_score_fn": self.lstm_attn_score_fn,
            "lstm_attn_merge": self.lstm_attn_merge,
            "lstm_attn_positional_encoding": self.lstm_attn_positional_encoding,
            "lstm_attn_temperature": self.lstm_attn_temperature,
            "lstm_attn_entropy_reg": self.lstm_attn_entropy_reg,
            "lstm_attn_distance_reg": self.lstm_attn_distance_reg,
            "lstm_attn_context_length": self.lstm_attn_context_length,
            "lstm_attn_key_dim": self.lstm_attn_key_dim,
            "lstm_attn_value_dim": self.lstm_attn_value_dim,
            # LSTM Output
            "lstm_fc_layers": self.lstm_fc_layers,
            "lstm_fc_hidden": self.lstm_fc_hidden,
            "lstm_output_dropout": self.lstm_output_dropout,
            "lstm_output_activation": self.lstm_output_activation,
            # LSTM Optimization
            "lstm_gradient_accumulation": self.lstm_gradient_accumulation,
            "lstm_momentum": self.lstm_momentum,
            # 4.3 Loss function
            "loss_fn": self.loss_fn,
            "huber_delta": self.huber_delta,
            "quantile_alpha": self.quantile_alpha,
            # 4.8 Regularization
            "input_noise_std": self.input_noise_std,
            "weight_decay": self.weight_decay,
            # 4.9 LR scheduler
            "lr_scheduler": self.lr_scheduler,
            "cosine_t_max": self.cosine_t_max,
            "plateau_patience": self.plateau_patience,
            # 4.10 Training control
            "early_stopping_patience": self.early_stopping_patience,
            "max_epochs": self.max_epochs,
            "warmup_steps": self.warmup_steps,

            # Mamba params
            "mamba_seq_len": self.mamba_seq_len,
            "mamba_d_model": self.mamba_d_model,
            "mamba_n_layers": self.mamba_n_layers,
            "mamba_ssm_dim": self.mamba_ssm_dim,
            "mamba_expand_factor": self.mamba_expand_factor,
            "mamba_activation": self.mamba_activation,
            "mamba_norm_type": self.mamba_norm_type,
            "mamba_norm_strategy": self.mamba_norm_strategy,
            "mamba_dropout": self.mamba_dropout,
            "mamba_resid_dropout": self.mamba_resid_dropout,
            "mamba_ssm_dropout": self.mamba_ssm_dropout,
            "mamba_gate_dropout": self.mamba_gate_dropout,
            "mamba_optimizer": self.mamba_optimizer,
            "mamba_learning_rate": self.mamba_learning_rate,
            "mamba_weight_decay": self.mamba_weight_decay,
            "mamba_grad_clip": self.mamba_grad_clip,
            "mamba_lr_scheduler": self.mamba_lr_scheduler,
            "mamba_warmup_steps": self.mamba_warmup_steps,
            "mamba_max_epochs": self.mamba_max_epochs,
            "mamba_batch_size": self.mamba_batch_size,
            "mamba_loss_fn": self.mamba_loss_fn,
            "mamba_head_type": self.mamba_head_type,
            "mamba_head_hidden_dim": self.mamba_head_hidden_dim,
            "mamba_head_num_layers": self.mamba_head_num_layers,
            "mamba_head_dropout": self.mamba_head_dropout,

            # Step 7: Threshold & Regime
            "threshold": self.threshold,
            "bull_mult": self.bull_mult,
            "bear_mult": self.bear_mult,
            "crisis_mult": self.crisis_mult,
            "best_score": self.best_score,
            "trial_number": self.trial_number,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizedParams":
        """Create from dictionary with backward-compatible coercions."""

        raw_family_encoders = data.get("family_encoders", {}) or {}
        family_encoders: Dict[str, str] = {}
        for family, raw_value in raw_family_encoders.items():
            if isinstance(raw_value, str) and raw_value:
                family_encoders[family] = raw_value
            else:
                # Legacy artifacts stored integers here; default to PCA.
                family_encoders[family] = "pca"

        return cls(
            # Pipeline-level hyperparameters
            smoothing_type=data.get("smoothing_type", "none"),
            smoothing_window=data.get("smoothing_window", 5),
            train_fraction=data.get("train_fraction", 0.8),
            # Family params
            included_families=data.get("included_families", {}),
            family_weights=data.get("family_weights", {}) or {},
            family_weights_by_symbol=data.get("family_weights_by_symbol", {}) or {},
            track_b_family_weights=data.get("track_b_family_weights", {}) or {},
            track_b_family_weights_by_symbol=data.get("track_b_family_weights_by_symbol", {}) or {},
            family_dimensions=data.get("family_dimensions", {}),
            family_encoders=family_encoders,
            track_a_weight=data.get("track_a_weight", 1.0),
            track_b_weight=data.get("track_b_weight", 1.0),
            # LSTM/shared params
            lstm_layers=data.get("lstm_layers", 2),
            lstm_hidden_dim=data.get("lstm_hidden_dim", 96),
            lstm_dropout=data.get("lstm_dropout", 0.2),
            lstm_seq_len=data.get("lstm_seq_len", 30),
            lstm_batch_size=data.get("lstm_batch_size", 64),
            lstm_learning_rate=data.get("lstm_learning_rate", 1e-3),
            lstm_optimizer=data.get("lstm_optimizer", "AdamW"),
            lstm_activation=data.get("lstm_activation", "relu"),
            lstm_use_amp=data.get("lstm_use_amp", True),
            # LSTM Architecture
            lstm_bidirectional=data.get("lstm_bidirectional", False),
            lstm_cell_type=data.get("lstm_cell_type", "lstm"),
            lstm_recurrent_kernel_init=data.get("lstm_recurrent_kernel_init", "orthogonal"),
            lstm_hidden_state_init=data.get("lstm_hidden_state_init", "zeros"),
            # LSTM Regularization
            lstm_recurrent_dropout=data.get("lstm_recurrent_dropout", 0.0),
            lstm_input_dropout=data.get("lstm_input_dropout", 0.0),
            lstm_grad_clip=data.get("lstm_grad_clip", 1.0),
            lstm_time_dropout=data.get("lstm_time_dropout", 0.0),
            lstm_recurrent_weight_decay=data.get("lstm_recurrent_weight_decay", 0.0),
            lstm_weight_dropout=data.get("lstm_weight_dropout", 0.0),
            lstm_zoneout=data.get("lstm_zoneout", 0.0),
            lstm_sequence_noise_std=data.get("lstm_sequence_noise_std", 0.0),
            # LSTM Enhancements
            lstm_layer_norm=data.get("lstm_layer_norm", False),
            lstm_residual=data.get("lstm_residual", False),
            lstm_skip_connect=data.get("lstm_skip_connect", False),
            lstm_lr_multiplier=data.get("lstm_lr_multiplier", 1.0),
            # LSTM Attention (new comprehensive attention system)
            lstm_attention_type=data.get("lstm_attention_type", "none"),
            lstm_attn_hidden_dim=data.get("lstm_attn_hidden_dim", 64),
            lstm_attn_dropout=data.get("lstm_attn_dropout", 0.1),
            lstm_attn_heads=data.get("lstm_attn_heads", 1),
            lstm_attn_normalization=data.get("lstm_attn_normalization", "softmax"),
            lstm_attn_score_fn=data.get("lstm_attn_score_fn", "dot"),
            lstm_attn_merge=data.get("lstm_attn_merge", "concat"),
            lstm_attn_positional_encoding=data.get("lstm_attn_positional_encoding", False),
            lstm_attn_temperature=data.get("lstm_attn_temperature", 1.0),
            lstm_attn_entropy_reg=data.get("lstm_attn_entropy_reg", 0.0),
            lstm_attn_distance_reg=data.get("lstm_attn_distance_reg", 0.0),
            lstm_attn_context_length=data.get("lstm_attn_context_length", 30),
            lstm_attn_key_dim=data.get("lstm_attn_key_dim", 64),
            lstm_attn_value_dim=data.get("lstm_attn_value_dim", 64),
            # LSTM Output
            lstm_fc_layers=data.get("lstm_fc_layers", 1),
            lstm_fc_hidden=data.get("lstm_fc_hidden", 64),
            lstm_output_dropout=data.get("lstm_output_dropout", 0.2),
            lstm_output_activation=data.get("lstm_output_activation", "none"),
            # LSTM Optimization
            lstm_gradient_accumulation=data.get("lstm_gradient_accumulation", 1),
            lstm_momentum=data.get("lstm_momentum", 0.9),
            # 4.3 Loss function
            loss_fn=data.get("loss_fn", "mse"),
            huber_delta=data.get("huber_delta", 1.0),
            quantile_alpha=data.get("quantile_alpha", 0.5),
            # 4.8 Regularization
            input_noise_std=data.get("input_noise_std", 0.0),
            weight_decay=data.get("weight_decay", 0.0),
            # 4.9 LR scheduler
            lr_scheduler=data.get("lr_scheduler", "none"),
            cosine_t_max=data.get("cosine_t_max", 50),
            plateau_patience=data.get("plateau_patience", 5),
            # 4.10 Training control
            early_stopping_patience=data.get("early_stopping_patience", 10),
            max_epochs=data.get("max_epochs", 100),
            warmup_steps=data.get("warmup_steps", 100),

            # Mamba params
            mamba_seq_len=data.get("mamba_seq_len", 128),
            mamba_d_model=data.get("mamba_d_model", 128),
            mamba_n_layers=data.get("mamba_n_layers", 4),
            mamba_ssm_dim=data.get("mamba_ssm_dim", 96),
            mamba_expand_factor=data.get("mamba_expand_factor", 2.0),
            mamba_activation=data.get("mamba_activation", "silu"),
            mamba_norm_type=data.get("mamba_norm_type", "rmsnorm"),
            mamba_norm_strategy=data.get("mamba_norm_strategy", "pre"),
            mamba_dropout=data.get("mamba_dropout", 0.10),
            mamba_resid_dropout=data.get("mamba_resid_dropout", 0.0),
            mamba_ssm_dropout=data.get("mamba_ssm_dropout", 0.0),
            mamba_gate_dropout=data.get("mamba_gate_dropout", 0.0),
            mamba_optimizer=data.get("mamba_optimizer", "adamw"),
            mamba_learning_rate=data.get("mamba_learning_rate", 1e-3),
            mamba_weight_decay=data.get("mamba_weight_decay", 1e-4),
            mamba_grad_clip=data.get("mamba_grad_clip", 1.0),
            mamba_lr_scheduler=data.get("mamba_lr_scheduler", "cosine"),
            mamba_warmup_steps=data.get("mamba_warmup_steps", 100),
            mamba_max_epochs=data.get("mamba_max_epochs", 10),
            mamba_batch_size=data.get("mamba_batch_size", 32),
            mamba_loss_fn=data.get("mamba_loss_fn", "smooth_l1"),
            mamba_head_type=data.get("mamba_head_type", "linear"),
            mamba_head_hidden_dim=data.get("mamba_head_hidden_dim", 128),
            mamba_head_num_layers=data.get("mamba_head_num_layers", 1),
            mamba_head_dropout=data.get("mamba_head_dropout", 0.0),

            # Step 7: Threshold & Regime
            threshold=data.get("threshold", 0.10),
            bull_mult=data.get("bull_mult", 0.8),
            bear_mult=data.get("bear_mult", 1.5),
            crisis_mult=data.get("crisis_mult", 3.0),
            best_score=data.get("best_score", 0.0),
            trial_number=data.get("trial_number", -1),
        )


# =============================================================================
# Encoder Classes
# =============================================================================

class SimpleAutoEncoder(nn.Module):
    """Configurable AutoEncoder for family feature compression.
    
    Supports Optuna-tuned architecture:
        - n_layers: 1, 2, or 3 layers in encoder/decoder
        - activation: relu, leakyrelu, gelu
        - dropout: [0.0, 0.4] for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.activation_name = activation
        self.dropout_rate = dropout
        
        if hidden_dim is None:
            hidden_dim = max(latent_dim * 2, (input_dim + latent_dim) // 2)
        
        # Get activation function
        act_fn = self._get_activation(activation)
        
        # Build encoder layers
        encoder_layers = []
        dims = self._compute_layer_dims(input_dim, latent_dim, hidden_dim, n_layers)
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Not the last layer
                encoder_layers.append(act_fn())
                if dims[i + 1] > 1:  # BatchNorm needs >1 features
                    encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))
                if dropout > 0:
                    encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (reverse)
        decoder_layers = []
        decoder_dims = list(reversed(dims))
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # Not the last layer
                decoder_layers.append(act_fn())
                if decoder_dims[i + 1] > 1:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name: str):
        """Get activation function class by name."""
        activations = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        return activations.get(name.lower(), nn.ReLU)
    
    def _compute_layer_dims(self, input_dim: int, latent_dim: int, hidden_dim: int, n_layers: int) -> List[int]:
        """Compute dimensions for each layer."""
        if n_layers == 1:
            return [input_dim, latent_dim]
        elif n_layers == 2:
            return [input_dim, hidden_dim, latent_dim]
        else:  # 3+ layers
            # Gradual compression: input -> hidden -> hidden/2 -> latent
            mid_dim = max(latent_dim, hidden_dim // 2)
            if n_layers == 3:
                return [input_dim, hidden_dim, mid_dim, latent_dim]
            else:
                # For 4+ layers, interpolate
                dims = [input_dim]
                for i in range(1, n_layers):
                    ratio = i / n_layers
                    dim = int(input_dim * (1 - ratio) + latent_dim * ratio)
                    dims.append(max(dim, latent_dim))
                dims.append(latent_dim)
                return dims
    
    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class FamilyEncoder:
    """Encoder wrapper for PCA or AutoEncoder compression.
    
    Supports Optuna-tuned hyperparameters:
        - method: "pca" or "ae" (Optuna chooses per family)
        - n_components: PCA components or AE latent dim
        - ae_layers: 1, 2, or 3 layers
        - ae_activation: relu, leakyrelu, gelu
        - ae_dropout: [0.0, 0.4]
        - ae_lr: [1e-5, 1e-3]
    """
    
    def __init__(
        self,
        family_name: str,
        n_components: int,
        method: str = "pca",
        device: Optional[str] = None,
        # AE-specific hyperparameters (Optuna-tuned)
        ae_layers: int = 2,
        ae_activation: str = "relu",
        ae_dropout: float = 0.0,
        ae_lr: float = 1e-3,
    ):
        self.family_name = family_name
        self.n_components = n_components
        self.method = method.lower()
        # 🔧 Use CPU for AutoEncoder to avoid GPU memory overhead
        # AE families are small (3-6 cols) so GPU isn't beneficial
        self.device = device or "cpu"
        
        # AE hyperparameters (Optuna-tuned)
        self.ae_layers = ae_layers
        self.ae_activation = ae_activation
        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        
        self.scaler = StandardScaler() if PCA_AVAILABLE else None
        self.pca = None
        self.ae_model = None
        self.fitted = False
    
    def fit(self, X: np.ndarray, epochs: int = 50) -> "FamilyEncoder":
        """Fit the encoder on data."""
        if X.shape[1] <= self.n_components:
            # No compression needed
            self.n_components = X.shape[1]
            self.method = "passthrough"
            self.fitted = True
            return self
        
        # Standardize
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        if self.method == "pca" and PCA_AVAILABLE:
            self.pca = PCA(n_components=self.n_components, random_state=42)
            self.pca.fit(X_scaled)
            
        elif self.method == "ae" and TORCH_AVAILABLE:
            self._fit_autoencoder(X_scaled, epochs)
        
        else:
            # Fallback to PCA if AE not available
            if PCA_AVAILABLE:
                self.pca = PCA(n_components=self.n_components, random_state=42)
                self.pca.fit(X_scaled)
                self.method = "pca"
        
        self.fitted = True
        return self
    
    def _fit_autoencoder(self, X: np.ndarray, epochs: int = 50):
        """Train AutoEncoder with Optuna-tuned hyperparameters."""
        import sys
        import time as _time
        _t0 = _time.time()
        
        # Limit threads to avoid contention in parallel trials
        torch.set_num_threads(2)
        
        device = torch.device(self.device)
        input_dim = X.shape[1]
        
        print(f"[TRACE:AE] {self.family_name}: creating model (layers={self.ae_layers}, act={self.ae_activation}, dropout={self.ae_dropout:.2f}, lr={self.ae_lr:.2e})...", file=sys.stderr, flush=True)
        self.ae_model = SimpleAutoEncoder(
            input_dim=input_dim,
            latent_dim=self.n_components,
            n_layers=self.ae_layers,
            activation=self.ae_activation,
            dropout=self.ae_dropout,
        ).to(device)
        optimizer = torch.optim.Adam(self.ae_model.parameters(), lr=self.ae_lr)
        criterion = nn.MSELoss()
        
        print(f"[TRACE:AE] {self.family_name}: creating dataloader...", file=sys.stderr, flush=True)
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        
        print(f"[TRACE:AE] {self.family_name}: starting training {epochs} epochs, {len(loader)} batches...", file=sys.stderr, flush=True)
        self.ae_model.train()
        for epoch in range(epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                _, decoded = self.ae_model(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
            if epoch == 0:
                print(f"[TRACE:AE] {self.family_name}: epoch 0 done in {_time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
        
        print(f"[TRACE:AE] {self.family_name}: training done in {_time.time()-_t0:.1f}s", file=sys.stderr, flush=True)
        self.ae_model.eval()
        
        # 🔧 Move model to CPU to free GPU memory if it was trained on GPU
        if self.device != "cpu" and torch.cuda.is_available():
            self.ae_model = self.ae_model.cpu()
            self.device = "cpu"  # Update device for transform
            torch.cuda.empty_cache()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted encoder."""
        if not self.fitted:
            raise RuntimeError(f"Encoder for {self.family_name} not fitted")
        
        if self.method == "passthrough":
            return X
        
        # Standardize
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if self.method == "pca" and self.pca is not None:
            return self.pca.transform(X_scaled)
        
        elif self.method == "ae" and self.ae_model is not None:
            device = torch.device(self.device)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                encoded = self.ae_model.encode(X_tensor)
            return encoded.cpu().numpy()
        
        return X_scaled[:, :self.n_components]
    
    def fit_transform(self, X: np.ndarray, epochs: int = 50) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, epochs)
        return self.transform(X)


# =============================================================================
# Optuna Optimizer
# =============================================================================

class StageBOptunaOptimizer:
    """Optuna optimizer for Stage-B LSTM feature selection and hyperparameters."""
    
    def __init__(
        self,
        config: Optional[OptunaConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or OptunaConfig()
        self.logger = logger or LOGGER
        
        # Family metadata (populated during optimization)
        self.family_sizes: Dict[str, int] = {}
        self.family_columns: Dict[str, List[str]] = {}
        self.stage_a_weights: Dict[str, float] = {}
        
        # 3-Pillar dimensionality analyzer (computed once before optimization)
        self.dim_analyzer: Optional[ThreePillarDimensionalityAnalyzer] = None
        self.family_optimal_dims: Dict[str, FamilyDimConfig] = {}
        
        # Encoder cache: keyed by (family, latent_dim, method, ae_params_tuple) -> FamilyEncoder
        # ae_params_tuple is empty for PCA, or sorted tuple of AE hyperparams for AE
        self._encoder_cache: Dict[Tuple, FamilyEncoder] = {}
        
        # Legacy (for backwards compatibility)
        self.encoders: Dict[str, FamilyEncoder] = {}
        
        # Best params
        self.best_params: Optional[OptimizedParams] = None
        self.study: Optional["optuna.Study"] = None

        # Ray execution helpers
        self._ray_enabled: bool = False
        self._ray_workers: List[Any] = []
        self._ray_worker_cursor: int = 0
        self._ray_max_workers: int = 1
        
        # Meta-optimizer for self-learning
        self.meta_optimizer: Optional[MetaOptimizer] = None
        self._meta_optimizer_enabled: bool = False

        # Global fold-geometry-derived caps (set when folds are known).
        # This keeps trial semantics consistent across folds and avoids per-fold clamping.
        self._mamba_seq_len_cap_for_folds: Optional[int] = None
        
        # Auto-enable meta-optimizer if configured
        if self.config.use_meta_optimizer:
            # Default save_dir to artifacts/meta_optimizer if not specified
            save_dir = self.config.meta_optimizer_save_dir
            if save_dir is None:
                save_dir = Path("artifacts/meta_optimizer")
                save_dir.mkdir(parents=True, exist_ok=True)
            
            self.enable_meta_optimizer(
                save_dir=save_dir,
                history_size=self.config.meta_optimizer_history_size,
                elite_percentile=self.config.meta_optimizer_elite_percentile,
                min_trials_before_adaptation=self.config.meta_optimizer_min_trials,
            )

    @staticmethod
    def _compute_mamba_seq_len_cap_for_folds(
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
        *,
        min_seq_len: int = 8,
        train_buffer: int = 20,
        val_buffer: int = 5,
    ) -> int:
        """Compute a global maximum seq_len that is valid for *scoring* folds.

        Core rule: sequence length must be constrained only by folds that
        actually contribute labeled samples to scoring.

        For single-symbol folds, a fold is considered usable if it has
        len(val_idx) > 0.

        IMPORTANT: We anchor the cap on validation geometry (val window size)
        rather than training geometry. Training geometry is enforced as a
        runtime guard per fold.
        """
        if not walk_forward_folds:
            return int(min_seq_len)

        # Only folds with labeled validation samples constrain the cap.
        usable_val_lens = [int(len(val_idx)) for _, val_idx in walk_forward_folds if int(len(val_idx)) > 0]
        if not usable_val_lens:
            return int(min_seq_len)

        min_val_window = min(usable_val_lens)
        cap = int(min_val_window) - int(val_buffer)
        return max(int(min_seq_len), int(cap))

    @staticmethod
    def _compute_mamba_seq_len_cap_for_multi_folds(
        walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        *,
        min_seq_len: int = 8,
        train_buffer: int = 20,
        val_buffer: int = 5,
        min_symbols_per_fold: int = 1,
        min_val_len_for_participation: Optional[int] = None,
    ) -> int:
        """Compute a global maximum seq_len valid for scoring folds/symbols only.

        Policy B (minimum-coverage gating):
          - A fold constrains the cap only if it has at least `min_symbols_per_fold`
            participating symbols.
          - A symbol participates if it has labeled validation rows, and (optionally)
            meets `min_val_len_for_participation`.
          - For a participating fold, the effective validation length is the Nth-largest
            validation window length (N = min_symbols_per_fold). This prevents a single
            short-history symbol from collapsing the global cap when we only require
            scoring on a minimum-coverage subset.

        Training geometry is enforced as a runtime guard per fold.
        """
        if not walk_forward_folds_multi:
            return int(min_seq_len)

        cap_raw: Optional[int] = None
        min_symbols = max(1, int(min_symbols_per_fold))
        min_val_len_req = int(min_val_len_for_participation) if min_val_len_for_participation is not None else None

        for fold in walk_forward_folds_multi:
            if not fold:
                continue
            # Compute per-symbol validation lengths eligible to participate.
            eligible_val_lens: List[int] = []
            for _sym, (_train_idx, val_idx) in fold.items():
                v = int(len(val_idx))
                if v <= 0:
                    continue
                if min_val_len_req is not None and v < int(min_val_len_req):
                    continue
                eligible_val_lens.append(v)

            if int(len(eligible_val_lens)) < int(min_symbols):
                continue

            # Nth-largest val length (equivalently: min over the top-N symbols).
            eligible_val_lens.sort(reverse=True)
            val_len_eff = int(eligible_val_lens[int(min_symbols) - 1])
            candidate = int(val_len_eff) - int(val_buffer)
            cap_raw = candidate if cap_raw is None else min(int(cap_raw), int(candidate))

        if cap_raw is None:
            return int(min_seq_len)

        return max(int(min_seq_len), int(cap_raw))

    def _get_effective_mamba_seq_len_choices(self) -> List[int]:
        """Return seq_len choices capped by fold geometry (if known)."""
        base_choices = [int(v) for v in self.config.mamba_seq_len_choices]
        cap = self._mamba_seq_len_cap_for_folds
        if cap is None:
            return base_choices

        capped = [v for v in base_choices if v <= int(cap)]
        if capped:
            return capped

        # If all configured choices exceed the fold-derived cap, fall back to the cap.
        # This keeps the run feasible without silently skipping folds.
        return [int(cap)]
    
    def _analyze_families(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        stage_a_weights: Optional[Dict[str, float]] = None,
    ):
        """Analyze family structure from panel data.
        
        If 3-pillar dimensionality is enabled, also computes optimal dims per family.
        """
        self.stage_a_weights = stage_a_weights or {}
        
        for col in panel.columns:
            family = column_families.get(col)
            if family:
                if family not in self.family_columns:
                    self.family_columns[family] = []
                self.family_columns[family].append(col)
        
        for family, cols in self.family_columns.items():
            self.family_sizes[family] = len(cols)
        
        self.logger.info(
            "Analyzed %d families: %s",
            len(self.family_sizes),
            ", ".join(f"{f}={s}" for f, s in sorted(self.family_sizes.items(), key=lambda x: -x[1])[:5])
        )

        # Visibility: log which families are forced passthrough (no PCA/AE compression).
        passthrough_track_a = sorted(
            [
                fam
                for fam, size in self.family_sizes.items()
                if int(size or 0) > 0 and _is_raw_passthrough_family(fam)
            ]
        )
        self.logger.info(
            "Passthrough (uncompressed) Track-A families present: %s",
            ", ".join(passthrough_track_a) if passthrough_track_a else "(none)",
        )
        self.logger.info(
            "Passthrough (uncompressed) Track-B families (always): %s",
            ", ".join(sorted(set(HF_BLOCK_FAMILIES + MODEL_FAMILIES))),
        )
        
        # Run 3-pillar dimensionality analysis if enabled
        if self.config.use_three_pillar_dims:
            self._run_three_pillar_analysis(panel)
    
    def _run_three_pillar_analysis(self, panel: pd.DataFrame):
        """Run 3-pillar dimensionality analysis to compute optimal dims per family.
        
        This is called ONCE before optimization starts. The results are stored
        in self.family_optimal_dims and used by _suggest_family_config to
        constrain the search space.
        """
        self.logger.info("Running 3-pillar dimensionality analysis...")
        
        # Create analyzer with config bounds
        self.dim_analyzer = ThreePillarDimensionalityAnalyzer(self.logger)
        self.dim_analyzer.DIM_MIN = self.config.three_pillar_dim_min
        self.dim_analyzer.DIM_MAX = self.config.three_pillar_dim_max
        self.dim_analyzer.PCA_VARIANCE_THRESHOLD = self.config.three_pillar_pca_variance
        self.dim_analyzer.AE_ELBOW_THRESHOLD = self.config.three_pillar_ae_elbow
        
        # Determine device for AE analysis
        device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Use CPU for analysis to avoid conflicts with training
            device = "cpu"
        
        # Run analysis
        self.family_optimal_dims = self.dim_analyzer.analyze_all_families(
            panel, self.family_columns, device
        )
        
        # Log summary
        total_dims = sum(c.k_final for c in self.family_optimal_dims.values())
        pca_families = sum(1 for c in self.family_optimal_dims.values() if c.method == "pca")
        ae_families = sum(1 for c in self.family_optimal_dims.values() if c.method == "ae")
        
        self.logger.info(
            f"3-pillar analysis complete: {len(self.family_optimal_dims)} families, "
            f"total_dims={total_dims}, pca={pca_families}, ae={ae_families}"
        )
    
    def enable_meta_optimizer(
        self,
        save_dir: Optional[Path] = None,
        history_size: int = 200,
        elite_percentile: float = 0.10,
        enable_interactions: bool = True,
        min_trials_before_adaptation: int = 20,
    ):
        """Enable the self-learning meta-optimizer layer.
        
        The meta-optimizer tracks trial history and learns:
        - Which parameters are helpful vs harmful (attribution)
        - Parameter interactions (via LightGBM)
        - Elite trial characteristics
        
        Args:
            save_dir: Directory to save/load meta-optimizer state
            history_size: Number of trials to keep in memory
            elite_percentile: Top percentile for elite selection
            enable_interactions: Enable LightGBM interaction learning
            min_trials_before_adaptation: Min trials before adapting search space
        """
        config = MetaOptimizerConfig(
            history_size=history_size,
            elite_percentile=elite_percentile,
            enable_interaction_learning=enable_interactions,
            min_trials_before_adaptation=min_trials_before_adaptation,
            save_path=save_dir,
        )
        
        self.meta_optimizer = MetaOptimizer(config, self.logger)
        self._meta_optimizer_enabled = True
        
        # Try to load existing state
        if save_dir and Path(save_dir).exists():
            self.meta_optimizer.load()
            self.logger.info(f"Meta-optimizer enabled (loaded {self.meta_optimizer.memory.size} trials)")
        else:
            self.logger.info("Meta-optimizer enabled (fresh start)")
    
    def get_meta_summary(self) -> Dict[str, Any]:
        """Get summary from meta-optimizer if enabled."""
        if not self._meta_optimizer_enabled or self.meta_optimizer is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "memory": self.meta_optimizer.get_memory_summary(),
            "attribution": self.meta_optimizer.get_attribution_summary(),
            "interactions": self.meta_optimizer.get_interaction_summary(),
        }
    
    def clear_encoder_cache(self):
        """Clear the encoder cache. Call this when training data changes."""
        cache_size = len(self._encoder_cache)
        self._encoder_cache.clear()
        self.logger.info(f"Cleared encoder cache ({cache_size} entries)")
    
    def get_encoder_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the encoder cache."""
        by_family = {}
        by_method = {"pca": 0, "ae": 0, "hybrid": 0}
        
        for (family, dim, method), encoder in self._encoder_cache.items():
            if family not in by_family:
                by_family[family] = []
            by_family[family].append((dim, method))
            if method in by_method:
                by_method[method] += 1
        
        return {
            "total_cached": len(self._encoder_cache),
            "families_cached": len(by_family),
            "by_method": by_method,
            "cache_keys": list(self._encoder_cache.keys()),
        }

    def _shutdown_ray_workers(self):
        """Terminate Ray workers if they were created."""
        if not self._ray_workers:
            self._ray_enabled = False
            return

        if not RAY_AVAILABLE or ray is None:
            self._ray_workers = []
            self._ray_enabled = False
            return

        for worker in self._ray_workers:
            try:
                ray.kill(worker)
            except Exception:
                continue

        self._ray_workers = []
        self._ray_enabled = False

    def _next_ray_worker(self):
        if not self._ray_workers:
            raise RuntimeError("Ray workers not initialized")
        worker = self._ray_workers[self._ray_worker_cursor]
        self._ray_worker_cursor = (self._ray_worker_cursor + 1) % len(self._ray_workers)
        return worker

    def _setup_ray_workers(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Initialize Ray workers for distributed trial execution."""
        self._shutdown_ray_workers()

        if not (self.config.use_ray and RAY_AVAILABLE and ray is not None):
            if self.config.use_ray and (not RAY_AVAILABLE or ray is None):
                self.logger.warning("Ray requested for Optuna but not available; running locally")
            self._ray_enabled = False
            return

        # When fold parallelism is enabled, we DON'T create trial workers.
        # Instead, trials run sequentially in the main process, and fold tasks
        # are spawned via Ray. This is the only working configuration because
        # Optuna's n_jobs spawns separate processes that can't share a Ray cluster.
        fold_parallelism = max(1, self.config.ray_fold_parallelism) if self.config.ray_fold_parallelism > 0 else 1
        if fold_parallelism > 1:
            self.logger.info(
                f"Ray fold parallelism enabled ({fold_parallelism} folds concurrent) - "
                f"running trials SEQUENTIALLY in main process with Ray fold tasks"
            )
            # Initialize Ray but don't create trial workers
            if not ray.is_initialized():
                init_kwargs = dict(self.config.ray_init_kwargs or {})
                if self.config.ray_address:
                    init_kwargs["address"] = self.config.ray_address
                init_kwargs = _apply_ray_runtime_env_envvars(init_kwargs)
                ray.init(ignore_reinit_error=True, log_to_driver=False, **init_kwargs)
            self._ray_enabled = False  # No trial workers, but Ray is available for fold tasks
            return

        if not ray.is_initialized():
            init_kwargs = dict(self.config.ray_init_kwargs or {})
            if self.config.ray_address:
                init_kwargs["address"] = self.config.ray_address
            init_kwargs = _apply_ray_runtime_env_envvars(init_kwargs)
            ray.init(ignore_reinit_error=True, log_to_driver=False, **init_kwargs)

        cluster_resources = ray.cluster_resources()
        total_cpus = float(cluster_resources.get("CPU", 1))
        total_gpus = float(cluster_resources.get("GPU", 0))
        cpus_per_trial = max(1, int(self.config.ray_cpus_per_trial))

        max_by_cpu = max(1, int(total_cpus // cpus_per_trial))
        target_workers = max_by_cpu

        if self.config.ray_max_concurrent_trials:
            target_workers = min(target_workers, int(self.config.ray_max_concurrent_trials))

        # Calculate GPU limits based on fold parallelism
        # Each trial spawns fold_parallelism concurrent fold tasks, each needing gpu_per_fold
        gpu_per_fold = self.config.ray_fold_gpu_fraction if self.config.ray_fold_gpu_fraction else 0.167
        effective_gpu_per_trial = gpu_per_fold * fold_parallelism
        
        if total_gpus > 0:
            max_by_gpu = max(1, int(total_gpus / effective_gpu_per_trial))
            target_workers = min(target_workers, max_by_gpu)
            self.logger.info(
                f"Ray GPU limit: {total_gpus} GPUs, fold_parallelism={fold_parallelism}, "
                f"gpu_per_fold={gpu_per_fold} -> {effective_gpu_per_trial:.2f} GPU/trial -> max {max_by_gpu} concurrent trial workers"
            )

        self._ray_max_workers = max(1, target_workers)

        shared_refs = {
            "panel": ray.put(panel),
            "column_families": ray.put(column_families),
            "labels": ray.put(labels),
            "block_summaries": ray.put(block_summaries),
            "walk_forward_folds": ray.put(walk_forward_folds),
        }

        config_dict = asdict(self.config)
        self._ray_workers = []
        self._ray_worker_cursor = 0

        self.logger.info(f"Setting up Ray workers with {len(self.family_columns)} families: {list(self.family_columns.keys())[:5]}...")
        
        # Trial workers don't need GPU - the fold tasks get GPU access directly
        # This avoids GPU resource conflicts between trial workers and fold tasks
        trial_worker_gpu = 0.0
        
        for _ in range(self._ray_max_workers):
            worker = _OptunaTrialWorker.options(
                num_cpus=cpus_per_trial,
                num_gpus=trial_worker_gpu,
            ).remote(
                shared_refs,
                config_dict,
                self.family_columns,
                self.family_sizes,
                self.stage_a_weights,
                self.logger.level,
            )
            self._ray_workers.append(worker)

        self._ray_enabled = True
        self.logger.info(
            "Initialized %d Ray Optuna workers (cpus_per_trial=%d, fold_parallelism=%d, gpu_per_fold=%.3f)",
            self._ray_max_workers,
            cpus_per_trial,
            fold_parallelism,
            gpu_per_fold,
        )
    def _evaluate_trial_params(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
        family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]],
        weight_a: float,
        weight_b: float,
        sequence_model_type: str,
        lstm_params: Dict[str, Any],
        mamba_params: Dict[str, Any],
        threshold_params: Dict[str, float],
        pipeline_params: Dict[str, Any],
        horizon: int,
        track_b_family_weights: Optional[Dict[str, float]] = None,
        trial: Optional["Trial"] = None,  # For Hyperband pruning
        progress_callback: Optional[Callable[[int, float], bool]] = None,  # For Ray Tune ASHA: (step, score) -> should_stop
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute a single Optuna trial with the provided parameters."""
        import sys
        import time
        _start_eval = time.time()
        print(f"[TRACE] _evaluate_trial_params START: families_enabled={sum(1 for f,(inc,_,_,w,_) in family_params.items() if inc)}, total_weight={sum(w for _,(inc,_,_,w,_) in family_params.items() if inc):.2f}", file=sys.stderr, flush=True)

        self.logger.info(f"_evaluate_trial: panel={panel.shape}, folds={len(walk_forward_folds)}, families_enabled={sum(1 for f,(inc,_,_,w,_) in family_params.items() if inc)}")
        
        first_train_idx = walk_forward_folds[0][0]

        print(f"[TRACE] Building Track A...", file=sys.stderr, flush=True)
        _t0 = time.time()
        track_a, _ = self._build_track_a(
            panel, column_families, family_params, first_train_idx
        )
        print(f"[TRACE] Track A complete in {time.time()-_t0:.1f}s, shape={track_a.shape}", file=sys.stderr, flush=True)
        track_a_dims = track_a.shape[1]
        self.logger.info(f"Track A: shape={track_a.shape}, family_cols_count={len(self.family_columns)}")
        
        track_b = self._build_track_b(
            panel,
            column_families,
            block_summaries,
            track_b_family_weights=track_b_family_weights,
        )
        track_b_dims = track_b.shape[1]
        self.logger.info(f"Track B: shape={track_b.shape}")
        
        print(f"[TRACE] Building Track C...", file=sys.stderr, flush=True)
        track_c = self._build_track_c(track_a, track_b, weight_a, weight_b)
        track_c_dims = track_c.shape[1]
        print(f"[TRACE] Track C complete: shape={track_c.shape}, dims={track_c_dims}", file=sys.stderr, flush=True)
        self.logger.info(f"Track C: shape={track_c.shape}")
        
        # Free track_a and track_b immediately - they're merged into track_c
        del track_a
        del track_b
        gc.collect()

        if track_c.empty:
            self.logger.warning("Track C is empty - returning -inf")
            return float("-inf"), {"reject_reason": "track_c_empty"}
        # Dimension cap: allow disabling by setting max_total_dims <= 0.
        if int(getattr(self.config, "max_total_dims", 0) or 0) > 0 and track_c_dims > int(self.config.max_total_dims):
            self.logger.warning(f"Track C dims {track_c_dims} > max {self.config.max_total_dims} - returning -inf")
            return float("-inf"), {"reject_reason": "track_c_too_large", "dims": track_c_dims}

        actual_returns = labels["forward_return"].astype(float)
        fold_scores: List[float] = []
        fold_losses: List[float] = []
        fold_mean_preds: List[float] = []  # For sign-stability penalty across folds
        fold_sharpes: List[float] = []  # Per-fold Sharpe for robustness objective (Mamba)
        
        # 🔧 NEW: Collect raw data for pooled Sharpe calculation
        # Instead of computing noisy per-fold Sharpe from ~2 samples, collect all
        # non-overlapping samples across folds and compute ONE aggregate Sharpe
        pooled_strategy_returns: List[float] = []  # Non-overlapping strategy returns
        pooled_directions: List[float] = []  # Corresponding directions
        pooled_actuals: List[float] = []  # Corresponding actual returns
        
        skipped_due_to_sequences = 0
        skipped_due_to_short_val = 0
        fold_failures = 0  # Track training exceptions

        if sequence_model_type == "mamba":
            seq_len = int(mamba_params.get("mamba_seq_len", 128))
        else:
            seq_len = int(lstm_params.get("lstm_seq_len", 30))

        # Enforce a single, globally-valid seq_len across all folds.
        # This prevents per-fold clamping (which breaks comparability in Bayesian optimization).
        if sequence_model_type == "mamba":
            cap = self._compute_mamba_seq_len_cap_for_folds(walk_forward_folds)
            self._mamba_seq_len_cap_for_folds = cap
            if seq_len > cap:
                self.logger.warning(
                    "Rejecting trial: mamba_seq_len=%d exceeds fold-derived cap=%d",
                    seq_len,
                    cap,
                )
                return float("-inf"), {
                    "reject_reason": "mamba_seq_len_exceeds_fold_cap",
                    "mamba_seq_len": int(seq_len),
                    "mamba_seq_len_cap": int(cap),
                }

        def _fold_sharpe_from_returns(strategy_returns: List[float], horizon: int) -> float:
            """Compute annualized Sharpe from a list of per-period strategy returns.

            Uses sqrt(floor(252 / H)) as annualization where H is horizon.
            Returns 0.0 for degenerate or too-short samples.
            """
            if not strategy_returns or len(strategy_returns) < 2:
                return 0.0
            arr = np.asarray(strategy_returns, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2:
                return 0.0
            std = float(np.std(arr))
            if std <= 1e-9:
                return 0.0
            periods_per_year = max(1, 252 // max(1, int(horizon)))
            return float(np.mean(arr) / (std + 1e-9) * np.sqrt(periods_per_year))

        print(
            f"[TRACE] Starting fold loop: {len(walk_forward_folds)} folds, seq_len={seq_len}, time_so_far={time.time()-_start_eval:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        self.logger.info(f"Starting fold loop: {len(walk_forward_folds)} folds, seq_len={seq_len}, model={sequence_model_type}")

        if sequence_model_type == "lstm":
            # Build LSTM config once - include ALL hyperparameters for full model support
            cfg = {
                "sequence_model_type": "lstm",
                # Core LSTM parameters
                "lstm_seq_len": lstm_params["lstm_seq_len"],
                "lstm_hidden_dim": lstm_params["lstm_hidden_dim"],
                "lstm_layers": lstm_params["lstm_layers"],
                "lstm_dropout": lstm_params["lstm_dropout"],
                "lstm_learning_rate": lstm_params["lstm_learning_rate"],
                "lstm_batch_size": lstm_params["lstm_batch_size"],
                "lstm_epochs": 15,
                "lstm_early_stop_patience": 3,
                "lstm_use_amp": bool(lstm_params.get("lstm_use_amp", True)),
                "lstm_warm_start": False,
                # Optimizer parameters
                "lstm_optimizer": lstm_params.get("lstm_optimizer", "AdamW"),
                "lstm_weight_decay": lstm_params.get("weight_decay", 1e-4),
                "lstm_gradient_accumulation": lstm_params.get("lstm_gradient_accumulation", 1),
                "lstm_momentum": lstm_params.get("lstm_momentum", 0.9),
                "lstm_lr_scheduler": lstm_params.get("lstm_lr_scheduler", "cosine"),
                "lstm_loss_fn": lstm_params.get("lstm_loss_fn", "mse"),
                # Cell type and architecture
                "lstm_cell_type": lstm_params.get("lstm_cell_type", "lstm"),
                "lstm_bidirectional": lstm_params.get("lstm_bidirectional", False),
                # Activation functions
                "lstm_activation": lstm_params.get("lstm_activation", "gelu"),
                "lstm_output_activation": lstm_params.get("lstm_output_activation", "tanh"),
                # Dropout variants
                "lstm_input_dropout": lstm_params.get("lstm_input_dropout", 0.0),
                "lstm_output_dropout": lstm_params.get("lstm_output_dropout", 0.0),
                "lstm_recurrent_dropout": lstm_params.get("lstm_recurrent_dropout", 0.0),
                "lstm_weight_dropout": lstm_params.get("lstm_weight_dropout", 0.0),
                "lstm_time_dropout": lstm_params.get("lstm_time_dropout", 0.0),
                "lstm_zoneout": lstm_params.get("lstm_zoneout", 0.0),
                "lstm_sequence_noise_std": lstm_params.get("lstm_sequence_noise_std", 0.0),
                "lstm_recurrent_weight_decay": lstm_params.get("lstm_recurrent_weight_decay", 0.0),
                # Normalization and skip connections
                "lstm_layer_norm": lstm_params.get("lstm_layer_norm", False),
                "lstm_residual": lstm_params.get("lstm_residual", False),
                "lstm_skip_connect": lstm_params.get("lstm_skip_connect", False),
                # Output head configuration
                "lstm_fc_layers": lstm_params.get("lstm_fc_layers", 1),
                "lstm_fc_hidden": lstm_params.get("lstm_fc_hidden", 0),
                # Weight initialization
                "lstm_recurrent_kernel_init": lstm_params.get("lstm_recurrent_kernel_init", "orthogonal"),
                "lstm_hidden_state_init": lstm_params.get("lstm_hidden_state_init", "zeros"),
                # Training parameters
                "lstm_grad_clip": lstm_params.get("lstm_grad_clip", 1.0),
                "lstm_lr_multiplier": lstm_params.get("lstm_lr_multiplier", 1.0),
                "max_grad_norm": lstm_params.get("max_grad_norm", 1.0),
                # Attention parameters
                "lstm_attention_type": lstm_params.get("lstm_attention_type", "none"),
                "lstm_attn_hidden_dim": lstm_params.get("lstm_attn_hidden_dim", 64),
                "lstm_attn_dropout": lstm_params.get("lstm_attn_dropout", 0.1),
                "lstm_attn_heads": lstm_params.get("lstm_attn_heads", 1),
                "lstm_attn_score_fn": lstm_params.get("lstm_attn_score_fn", "general"),
                "lstm_attn_temperature": lstm_params.get("lstm_attn_temperature", 1.0),
                "lstm_attn_normalization": lstm_params.get("lstm_attn_normalization", "softmax"),
                "lstm_attn_merge": lstm_params.get("lstm_attn_merge", "concat"),
                "lstm_attn_positional_encoding": lstm_params.get("lstm_attn_positional_encoding", False),
                "lstm_attn_context_length": lstm_params.get("lstm_attn_context_length", 0),
                "lstm_attn_entropy_reg": lstm_params.get("lstm_attn_entropy_reg", 0.0),
                "lstm_attn_distance_reg": lstm_params.get("lstm_attn_distance_reg", 0.0),
                "lstm_attn_key_dim": lstm_params.get("lstm_attn_key_dim", 0),
                "lstm_attn_value_dim": lstm_params.get("lstm_attn_value_dim", 0),
                # CNN Frontend parameters
                "cnn_frontend_enabled": lstm_params.get("cnn_frontend_enabled", False),
                "cnn_blocks": lstm_params.get("cnn_blocks", 2),
                "cnn_filters": lstm_params.get("cnn_filters", 64),
                "cnn_kernel_size": lstm_params.get("cnn_kernel_size", 3),
                "cnn_dilation": lstm_params.get("cnn_dilation", 1),
                "cnn_stride": lstm_params.get("cnn_stride", 1),
                "cnn_pooling": lstm_params.get("cnn_pooling", None),
                "cnn_activation": lstm_params.get("cnn_activation", "gelu"),
                "cnn_dropout": lstm_params.get("cnn_dropout", 0.1),
                "cnn_batch_norm": lstm_params.get("cnn_batch_norm", True),
                "cnn_layer_norm": lstm_params.get("cnn_layer_norm", False),
                "cnn_residual": lstm_params.get("cnn_residual", True),
                # Training control parameters
                "max_epochs": lstm_params.get("max_epochs", 100),
                "early_stopping_patience": lstm_params.get("early_stopping_patience", 10),
                "plateau_patience": lstm_params.get("plateau_patience", 5),
                "warmup_steps": lstm_params.get("warmup_steps", 100),
                "input_noise_std": lstm_params.get("input_noise_std", 0.0),
                "label_smoothing": lstm_params.get("label_smoothing", 0.0),
                "amp_precision": lstm_params.get("amp_precision", "bf16"),
                # EMA parameters
                "ema_enabled": lstm_params.get("ema_enabled", True),
                "ema_decay": lstm_params.get("ema_decay", 0.9999),
                # LR Scheduler parameters
                "cosine_t_max": lstm_params.get("cosine_t_max", 50),
                "step_lr_step_size": lstm_params.get("step_lr_step_size", 10),
                "cyclical_base_lr": lstm_params.get("cyclical_base_lr", 1e-5),
                "cyclical_max_lr": lstm_params.get("cyclical_max_lr", 1e-3),
                "cyclical_step_size": lstm_params.get("cyclical_step_size", 10),
                # Loss function parameters
                "huber_delta": lstm_params.get("huber_delta", 1.0),
                "quantile_alpha": lstm_params.get("quantile_alpha", 0.5),
                # Advanced parameters
                "market_regime_model": lstm_params.get("market_regime_model", "hmm"),
                "volatility_regime_window": lstm_params.get("volatility_regime_window", 21),
                "train_seq_length": lstm_params.get("train_seq_length", 63),
                "target_seq_length": lstm_params.get("target_seq_length", 1),
                "mixout_prob": lstm_params.get("mixout_prob", 0.0),
                "stochastic_depth": lstm_params.get("stochastic_depth", 0.0),
                "rotary_embedding": lstm_params.get("rotary_embedding", False),
                "feedforward_dim": lstm_params.get("feedforward_dim", 256),
                "attn_layers": lstm_params.get("attn_layers", 1),
                "dense_activation": lstm_params.get("dense_activation", "gelu"),
                "batch_norm_head": lstm_params.get("batch_norm_head", False),
            }
        elif sequence_model_type == "mamba":
            cfg = {
                "sequence_model_type": "mamba",
                "mamba_seq_len": int(mamba_params.get("mamba_seq_len", 128)),
                "mamba_d_model": int(mamba_params.get("mamba_d_model", 128)),
                "mamba_n_layers": int(mamba_params.get("mamba_n_layers", 4)),
                "mamba_ssm_dim": int(mamba_params.get("mamba_ssm_dim", 96)),
                "mamba_expand_factor": float(mamba_params.get("mamba_expand_factor", 2.0)),
                "mamba_activation": mamba_params.get("mamba_activation", "silu"),
                "mamba_norm_type": mamba_params.get("mamba_norm_type", "rmsnorm"),
                "mamba_norm_strategy": mamba_params.get("mamba_norm_strategy", "pre"),
                "mamba_dropout": float(mamba_params.get("mamba_dropout", 0.1)),
                "mamba_resid_dropout": float(mamba_params.get("mamba_resid_dropout", 0.0)),
                "mamba_ssm_dropout": float(mamba_params.get("mamba_ssm_dropout", 0.0)),
                "mamba_gate_dropout": float(mamba_params.get("mamba_gate_dropout", 0.0)),
                "mamba_optimizer": mamba_params.get("mamba_optimizer", "adamw"),
                "mamba_learning_rate": float(mamba_params.get("mamba_learning_rate", 1e-3)),
                "mamba_weight_decay": float(mamba_params.get("mamba_weight_decay", 1e-4)),
                "mamba_grad_clip": float(mamba_params.get("mamba_grad_clip", 1.0)),
                "mamba_lr_scheduler": mamba_params.get("mamba_lr_scheduler", "cosine"),
                "mamba_warmup_steps": int(mamba_params.get("mamba_warmup_steps", 100)),
                "max_epochs": int(mamba_params.get("mamba_max_epochs", 10)),
                "mamba_batch_size": int(mamba_params.get("mamba_batch_size", 32)),
                "mamba_loss_fn": mamba_params.get("mamba_loss_fn", "smooth_l1"),
                "mamba_head_type": mamba_params.get("mamba_head_type", "linear"),
                "mamba_head_hidden_dim": int(mamba_params.get("mamba_head_hidden_dim", 128)),
                "mamba_head_num_layers": int(mamba_params.get("mamba_head_num_layers", 1)),
                "mamba_head_dropout": float(mamba_params.get("mamba_head_dropout", 0.0)),
            }
        else:
            raise ValueError(f"Unknown sequence_model_type: {sequence_model_type}")

        # Check if we should use fold-level Ray parallelism
        # NOTE: This only works from the main process. We cannot use trial workers
        # with fold parallelism because Optuna's n_jobs spawns separate processes
        # that each try to initialize their own Ray cluster.
        inside_actor = getattr(self, "_inside_ray_actor", False)
        use_ray_folds = (
            self.config.ray_fold_parallelism > 0
            and RAY_AVAILABLE
            and ray is not None
            and ray.is_initialized()
            and not inside_actor  # Cannot spawn tasks from inside an actor
        )
        fold_parallelism = self.config.ray_fold_parallelism if use_ray_folds else 1

        if use_ray_folds:
            gpu_per_fold = self.config.ray_fold_gpu_fraction if self.config.ray_fold_gpu_fraction else 0.167
            self.logger.info(f"Using Ray fold parallelism: {fold_parallelism} folds concurrent, gpu_per_fold={gpu_per_fold}")
        elif inside_actor:
            self.logger.info(f"Running inside Ray actor - using SEQUENTIAL fold execution")
        
        # Pre-filter folds to get valid indices (without building seq_data yet).
        # IMPORTANT: We do NOT clamp seq_len per-fold. seq_len is globally constrained
        # by fold geometry before trials run, so every fold uses the same seq_len.
        valid_fold_indices = []
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(walk_forward_folds):
            train_window_size = len(fold_train_idx)
            val_window_size = len(fold_val_idx)

            # Need at least seq_len + some buffer to create enough sequences.
            if train_window_size < seq_len + 20:
                skipped_due_to_sequences += 1
                continue

            # Validation also needs enough length to create sequences.
            if val_window_size < seq_len + 5:
                skipped_due_to_short_val += 1
                continue

            valid_fold_indices.append((fold_idx, fold_train_idx, fold_val_idx))
        
        self.logger.info(f"Pre-filtered {len(valid_fold_indices)}/{len(walk_forward_folds)} folds for seq_len={seq_len}")

        if use_ray_folds and len(valid_fold_indices) > 0:
            # ============================================
            # 🚀 BATCHED PARALLEL FOLD EXECUTION
            # ============================================
            # Key optimization: Each Ray task trains MULTIPLE folds
            # This reduces task overhead and keeps GPU warm
            
            # Calculate optimal batch size: each worker trains N folds sequentially
            # With 2 concurrent trials × 0.45 GPU each = 0.9 GPU total
            # We want ~2-4 batch tasks running in parallel
            folds_per_batch = max(5, len(valid_fold_indices) // 12)  # ~12 batches total
            n_parallel_batches = 2  # Run 2 batch tasks in parallel (matches concurrent trials)
            
            gpu_per_batch = min(0.45, gpu_per_fold * folds_per_batch)  # Cap at 0.45 per batch
            
            self.logger.info(
                f"Starting BATCHED fold execution: {len(valid_fold_indices)} folds, "
                f"{folds_per_batch} folds/batch, {n_parallel_batches} parallel batches, "
                f"gpu_per_batch={gpu_per_batch:.2f}"
            )
            
            # Put shared data in Ray object store ONCE (not per fold!)
            track_c_values_ref = ray.put(track_c.values.astype(np.float32))
            track_c_index_ref = ray.put(track_c.index.to_numpy())
            actual_returns_ref = ray.put(actual_returns.values.astype(np.float32))
            cfg_ref = ray.put(cfg)
            
            # Split folds into batches
            fold_batches = []
            for i in range(0, len(valid_fold_indices), folds_per_batch):
                batch = valid_fold_indices[i:i + folds_per_batch]
                fold_batches.append(batch)
            
            self.logger.info(f"Created {len(fold_batches)} fold batches")
            
            # Process batches with pipelining: launch next batch while current runs
            pending_batch_tasks = []
            batch_idx = 0
            
            # Launch initial parallel batches
            while batch_idx < len(fold_batches) and len(pending_batch_tasks) < n_parallel_batches:
                batch = fold_batches[batch_idx]
                
                task_options = {"num_gpus": gpu_per_batch, "num_cpus": 2}
                if PlacementGroupSchedulingStrategy is not None:
                    task_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(placement_group=None)
                
                task = _OptunaFoldBatchTask.options(**task_options).remote(
                    track_c_values_ref,
                    track_c_index_ref,
                    actual_returns_ref,
                    batch,
                    cfg_ref,
                    seq_len,
                )
                pending_batch_tasks.append((task, batch_idx))
                batch_idx += 1
            
            # Process results and launch new batches as slots free up
            while pending_batch_tasks:
                # Wait for ANY batch to complete (non-blocking pipeline)
                ready_tasks, pending_batch_tasks_remaining = [], []
                for task, bidx in pending_batch_tasks:
                    # Check if ready without blocking
                    ready, _ = ray.wait([task], timeout=0)
                    if ready:
                        ready_tasks.append((task, bidx))
                    else:
                        pending_batch_tasks_remaining.append((task, bidx))
                
                if not ready_tasks:
                    # None ready yet - wait for at least one
                    if pending_batch_tasks:
                        task, bidx = pending_batch_tasks[0]
                        ray.wait([task], timeout=None)
                        ready_tasks.append((task, bidx))
                        pending_batch_tasks_remaining = pending_batch_tasks[1:]
                
                pending_batch_tasks = pending_batch_tasks_remaining
                
                # Process completed batches
                for task, bidx in ready_tasks:
                    try:
                        batch_results = ray.get(task)
                        for res in batch_results:
                            try:
                                if res.get("error"):
                                    fold_failures += 1
                                    self.logger.debug(f"Fold {res['fold_idx']} error: {res['error']}")
                                    continue
                                result = res.get("result")
                                if result is None:
                                    fold_failures += 1
                                    continue
                                
                                predictions = result["preds"]
                                val_timestamps = result["timestamps"]
                                val_returns = actual_returns.reindex(pd.DatetimeIndex(val_timestamps))
                                
                                # 🔧 Extract raw metrics for pooled Sharpe calculation
                                raw_metrics = self._extract_fold_raw_metrics(
                                    predictions,
                                    val_returns.values,
                                    horizon,
                                    threshold_params=threshold_params,
                                )
                                
                                if raw_metrics.get("valid", False):
                                    # Collect non-overlapping samples for pooled calculation
                                    pooled_strategy_returns.extend(raw_metrics["strategy_returns"])
                                    pooled_directions.extend(raw_metrics["directions"])
                                    pooled_actuals.extend(raw_metrics["actuals"])
                                    fold_mean_preds.append(raw_metrics["mean_pred"])
                                    fold_sharpes.append(_fold_sharpe_from_returns(raw_metrics["strategy_returns"], horizon))
                                    fold_losses.append(float(result.get("val_loss", float("inf"))))
                                    # Keep per-fold score for intermediate reporting
                                    fold_score = self._compute_objective(
                                        predictions, val_returns.values, horizon, threshold_params
                                    )
                                    if np.isfinite(fold_score):
                                        fold_scores.append(float(fold_score))
                            except Exception as e:
                                fold_failures += 1
                                self.logger.debug(f"Fold result processing error: {e}")
                    except Exception as batch_err:
                        self.logger.warning(f"Batch {bidx} failed: {batch_err}")
                        fold_failures += len(fold_batches[bidx]) if bidx < len(fold_batches) else 1
                
                # Launch next batch if slots available
                while batch_idx < len(fold_batches) and len(pending_batch_tasks) < n_parallel_batches:
                    batch = fold_batches[batch_idx]
                    
                    task_options = {"num_gpus": gpu_per_batch, "num_cpus": 2}
                    if PlacementGroupSchedulingStrategy is not None:
                        task_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(placement_group=None)
                    
                    task = _OptunaFoldBatchTask.options(**task_options).remote(
                        track_c_values_ref,
                        track_c_index_ref,
                        actual_returns_ref,
                        batch,
                        cfg_ref,
                        seq_len,
                    )
                    pending_batch_tasks.append((task, batch_idx))
                    batch_idx += 1
                
                # Progress log and Hyperband pruning check
                if len(fold_scores) > 0 and len(fold_scores) % 10 == 0:
                    self.logger.info(f"Fold progress: {len(fold_scores)} scores collected")
                    # Report intermediate value for Hyperband pruning
                    if trial is not None and len(fold_scores) >= 5:
                        if sequence_model_type == "mamba" and len(fold_sharpes) >= 2:
                            cur = np.asarray(fold_sharpes, dtype=float)
                            cur = cur[np.isfinite(cur)]
                            mean_sharpe = float(np.mean(cur)) if cur.size else 0.0
                            robustness = float(np.std(cur)) if cur.size >= 2 else 0.0
                            intermediate_score = mean_sharpe - float(self.config.stability_weight) * robustness
                        else:
                            intermediate_score = float(np.mean(fold_scores))
                        trial.report(intermediate_score, step=len(fold_scores))
                        if trial.should_prune():
                            self.logger.info(f"Trial pruned at fold {len(fold_scores)} with score {intermediate_score:.4f}")
                            raise optuna.TrialPruned()
            
            # Cleanup shared refs
            del track_c_values_ref, track_c_index_ref, actual_returns_ref, cfg_ref
            gc.collect()
        
        else:
            # ============================================
            # 🚀 SEQUENTIAL FOLD TRAINING: Each fold trains its own model
            # ============================================
            # Process folds one-by-one for finer-grained ASHA pruning feedback
            # TPE gets faster learning via more frequent intermediate reports
            import torch as _torch
            import sys
            import time as _time
            import os as _os
            from contextlib import contextmanager as _contextmanager

            _nvtx_enabled = (
                _torch.cuda.is_available()
                and _os.environ.get("DCF_ENABLE_NVTX", "0") == "1"
                and hasattr(_torch.cuda, "nvtx")
            )

            @_contextmanager
            def _nvtx(msg: str):
                if not _nvtx_enabled:
                    yield
                    return
                try:
                    _torch.cuda.nvtx.range_push(msg)
                except Exception:
                    yield
                    return
                try:
                    yield
                finally:
                    try:
                        _torch.cuda.nvtx.range_pop()
                    except Exception:
                        pass
            
            print(f"[DEBUG] SEQUENTIAL FOLD TRAINING: torch.cuda.is_available()={_torch.cuda.is_available()}", file=sys.stderr, flush=True)
            if _torch.cuda.is_available():
                _device = _torch.device("cuda:0")
                print(f"[DEBUG] CUDA ENABLED: device={_device}, GPU={_torch.cuda.get_device_name(0)}", file=sys.stderr, flush=True)
                self.logger.info(f"Sequential fold training: {len(valid_fold_indices)} folds (CUDA on {_torch.cuda.get_device_name(0)})")
            else:
                _device = _torch.device("cpu")
                self.logger.warning(f"Sequential fold training: {len(valid_fold_indices)} folds (CPU only)")
            
            from src.stage_b.sequence_models import SequenceData as _SequenceData
            
            n_folds_total = len(valid_fold_indices)
            print(f"[TRACE] SEQUENTIAL TRAINING: {n_folds_total} folds, 1 model per fold", file=sys.stderr, flush=True)

            _trial_label = (
                _os.environ.get("TUNE_TRIAL_NAME")
                or _os.environ.get("TUNE_TRIAL_ID")
                or _os.environ.get("RAY_TUNE_TRIAL_NAME")
                or "trial"
            )
            with _nvtx(f"ray_trial:{_trial_label}"):
                fold_num = -1
                fold_time = 0.0
                for fold_num, (fold_idx, fold_train_idx, fold_val_idx) in enumerate(valid_fold_indices):
                    fold_t0 = _time.time()
                    
                    try:
                        with _nvtx(f"fold_{fold_idx}"):
                            # Debug: log input shapes for first fold
                            if fold_num == 0:
                                print(f"[DEBUG] Fold 0 input: track_c.iloc[fold_train_idx].shape={track_c.iloc[fold_train_idx].shape}, actual_returns.iloc[fold_train_idx].shape={actual_returns.iloc[fold_train_idx].shape}, seq_len={seq_len}", file=sys.stderr, flush=True)
                            
                            # ============================================
                            # STEP 1: Build sequences for this fold
                            # ============================================
                            with _nvtx("build_train_sequences"):
                                seq_result = build_sequence_data(
                                    track_c.iloc[fold_train_idx],
                                    actual_returns.iloc[fold_train_idx],
                                    seq_len,
                                )
                            
                            if seq_result is None:
                                skipped_due_to_sequences += 1
                                fold_failures += 1
                                raise RuntimeError("train_sequence_build_failed")
                            seq_data, _ = seq_result
                            if len(seq_data) < 20:
                                skipped_due_to_sequences += 1
                                fold_failures += 1
                                raise RuntimeError(f"insufficient_train_sequences:{len(seq_data)}")
                            
                            # Build validation sequences
                            with _nvtx("build_val_sequences"):
                                val_seq_result = build_sequence_data(
                                    track_c.iloc[fold_val_idx],
                                    actual_returns.iloc[fold_val_idx],
                                    seq_len,
                                )
                            if val_seq_result is None:
                                skipped_due_to_short_val += 1
                                del seq_data
                                fold_failures += 1
                                raise RuntimeError("val_sequence_build_failed")
                            val_seq_data, _ = val_seq_result
                            if len(val_seq_data) < 5:
                                skipped_due_to_short_val += 1
                                del seq_data
                                fold_failures += 1
                                raise RuntimeError(f"insufficient_val_sequences:{len(val_seq_data)}")
                            
                            val_timestamps = track_c.iloc[fold_val_idx].index[seq_len:]
                            
                            # ============================================
                            # STEP 2: Train model on this fold's training data
                            # ============================================
                            n_train = len(seq_data)
                            train_end = int(n_train * 0.9)
                            train_idx = np.arange(train_end)
                            holdout_idx = np.arange(train_end, n_train)
                            
                            with _nvtx("train"):
                                train_fn = train_mamba_fold if cfg.get("sequence_model_type") == "mamba" else train_lstm_fold
                                cfg_fold = dict(cfg)
                                cfg_fold["mamba_seq_len"] = seq_len if cfg.get("sequence_model_type") == "mamba" else cfg.get("mamba_seq_len")
                                result = train_fn(
                                    seq_data,  # SequenceData object
                                    train_idx,
                                    holdout_idx,
                                    cfg_fold,
                                    device=_device,
                                    return_model=True,
                                )
                            
                            # Get trained model for evaluation
                            trained_model = result.get("model")
                            if trained_model is None:
                                fold_failures += 1
                                del seq_data, val_seq_data
                                continue
                            
                            # Log VRAM for first fold
                            if fold_num == 0:
                                try:
                                    if _torch.cuda.is_available():
                                        alloc_mb = _torch.cuda.memory_allocated() / 1024**2
                                        reserved_mb = _torch.cuda.memory_reserved() / 1024**2
                                        self.logger.info(f"Fold 0 VRAM: allocated={alloc_mb:.0f}MB, reserved={reserved_mb:.0f}MB")
                                except Exception:
                                    pass
                            
                            # ============================================
                            # STEP 3: Evaluate model on validation set
                            # ============================================
                            with _nvtx("eval"):
                                trained_model.eval()
                                with _torch.no_grad():
                                    val_X = val_seq_data.sequences.astype(np.float32)
                                    val_X_tensor = _torch.from_numpy(val_X).to(_device)
                                    
                                    # Forward pass
                                    predictions = trained_model(val_X_tensor).cpu().numpy().flatten()
                                    
                                    # Align timestamps
                                    if len(predictions) > len(val_timestamps):
                                        predictions = predictions[:len(val_timestamps)]
                                    elif len(val_timestamps) > len(predictions):
                                        val_timestamps = val_timestamps[:len(predictions)]
                                    
                                    val_returns = actual_returns.reindex(pd.DatetimeIndex(val_timestamps))
                                    
                                    # Extract raw fold metrics for pooled aggregation
                                    raw_metrics = self._extract_fold_raw_metrics(
                                        predictions,
                                        val_returns.values,
                                        horizon,
                                        threshold_params=threshold_params,
                                    )
                                    
                                    if raw_metrics is not None and raw_metrics.get("valid", False):
                                        pooled_strategy_returns.extend(raw_metrics["strategy_returns"])
                                        pooled_directions.extend(raw_metrics["directions"])
                                        pooled_actuals.extend(raw_metrics["actuals"])

                                        fold_losses.append(float(result.get("val_loss", float("inf"))))
                                        fold_mean_preds.append(raw_metrics.get("mean_pred", 0.0))

                                        # Always count this fold (including no-trade folds)
                                        fold_sharpe = _fold_sharpe_from_returns(raw_metrics.get("strategy_returns", []), horizon)
                                        fold_sharpes.append(float(fold_sharpe))

                                        # Keep a per-fold objective for non-mamba intermediate reporting
                                        fold_score = self._compute_objective(
                                            predictions,
                                            val_returns.values,
                                            horizon,
                                            threshold_params,
                                        )
                                        if np.isfinite(fold_score):
                                            fold_scores.append(float(fold_score))
                                    else:
                                        fold_failures += 1
                                        raise RuntimeError("raw_metrics_invalid")
                                    
                                    del val_X, val_X_tensor, predictions
                            
                            # Cleanup fold resources
                            del trained_model, result, seq_data, val_seq_data
                        
                    except Exception as fold_err:
                        fold_failures += 1
                        import traceback
                        print(f"[ERROR] Fold {fold_idx} error: {fold_err}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                        continue
                    
                    finally:
                        # Always cleanup GPU memory after each fold
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                        gc.collect()
                    
                    fold_time = _time.time() - fold_t0

                    # Progress logging every 5 folds
                    if (fold_num + 1) % 5 == 0 or fold_num == 0:
                        print(
                            f"[TRACE] Fold {fold_num+1}/{n_folds_total}: sharpes={len(fold_sharpes)}, pooled_samples={len(pooled_strategy_returns)}, last_fold_s={fold_time:.1f}",
                            file=sys.stderr,
                            flush=True,
                        )

                    # ASHA/PRUNING REPORTING: report every fold so ASHA can prune mid-trial
                    if progress_callback is not None and len(fold_sharpes) >= 1:
                        intermediate_score = 0.0
                        if sequence_model_type == "mamba":
                            cur = np.asarray(fold_sharpes, dtype=float)
                            cur = cur[np.isfinite(cur)]
                            mean_sharpe = float(np.mean(cur)) if cur.size else 0.0
                            robustness = float(np.std(cur)) if cur.size >= 2 else 0.0
                            intermediate_score = mean_sharpe - float(self.config.stability_weight) * robustness
                        else:
                            cur_scores = np.asarray(fold_scores, dtype=float)
                            cur_scores = cur_scores[np.isfinite(cur_scores)]
                            intermediate_score = float(np.mean(cur_scores)) if cur_scores.size else float("-inf")

                        # Callback is responsible for reporting to Ray Tune.
                        # We intentionally do not early-stop here; Ray Tune/ASHA will terminate the trial externally.
                        progress_callback(fold_num + 1, float(intermediate_score))

                # Hyperband pruning check (native Optuna path): report every fold
                if trial is not None and len(fold_sharpes) >= 1:
                    cur = np.asarray(fold_sharpes, dtype=float)
                    cur = cur[np.isfinite(cur)]
                    mean_sharpe = float(np.mean(cur)) if cur.size else 0.0
                    robustness = float(np.std(cur)) if cur.size >= 2 else 0.0
                    intermediate_score = mean_sharpe - float(self.config.stability_weight) * robustness
                    trial.report(intermediate_score, step=fold_num + 1)
                    if trial.should_prune():
                        self.logger.info(
                            "Trial pruned at fold %d with intermediate_score %.4f",
                            fold_num + 1,
                            intermediate_score,
                        )
                        raise optuna.TrialPruned()
            
            # Final cleanup
            gc.collect()

        self.logger.info(f"Fold loop done: scores={len(fold_scores)}, seq_skips={skipped_due_to_sequences}, val_skips={skipped_due_to_short_val}, failures={fold_failures}")
        
        n_folds = len(walk_forward_folds)
        # Hedge-fund-grade comparability: every trial must be evaluated on the same fold set.
        # If any fold is skipped or fails, reject the trial.
        if (
            len(fold_sharpes) != n_folds
            or skipped_due_to_sequences > 0
            or skipped_due_to_short_val > 0
            or fold_failures > 0
        ):
            self.logger.info(
                "Trial rejected: folds=%d/%d (seq_len=%d, seq_skips=%d, val_skips=%d, failures=%d)",
                len(fold_sharpes),
                n_folds,
                seq_len,
                skipped_due_to_sequences,
                skipped_due_to_short_val,
                fold_failures,
            )
            return float("-inf"), {
                "n_successful_folds": len(fold_sharpes),
                "fold_failures": fold_failures,
                "skipped_seq": skipped_due_to_sequences,
                "skipped_val": skipped_due_to_short_val,
            }

        # ================================================================
        # ✅ MAMBA FLOW: fold-by-fold Sharpe aggregation with robustness penalty
        # One hyperparameter set per trial, evaluated across all folds.
        # Each fold: fresh model, train on fold-train, predict fold-valid, compute fold Sharpe.
        # Trial objective: mean_sharpe - penalty * std_sharpe
        # ================================================================
        if sequence_model_type == "mamba":
            sharpe_arr = np.asarray(fold_sharpes, dtype=float)
            sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]
            mean_sharpe = float(np.mean(sharpe_arr)) if sharpe_arr.size else float("-inf")
            robustness = float(np.std(sharpe_arr)) if sharpe_arr.size >= 2 else 0.0
            penalty = float(self.config.stability_weight)
            final_score = mean_sharpe - penalty * robustness

            # Backwards-compatible summary fields expected by downstream logging
            avg_stability = float(np.clip(1.0 - robustness, 0.0, 1.0))

            metadata = {
                "track_a_dims": track_a_dims,
                "track_b_dims": track_b_dims,
                "track_c_dims": track_c_dims,
                "avg_val_loss": float(np.mean(fold_losses)) if fold_losses else float("inf"),
                "n_successful_folds": int(sharpe_arr.size),
                "fold_sharpe_mean": mean_sharpe,
                "fold_sharpe_std": robustness,
                "robustness_penalty": penalty,
                "final_score": final_score,
                # Common keys used elsewhere in this module
                "avg_sharpe": mean_sharpe,
                "avg_score": final_score,
                "avg_stability": avg_stability,
                "avg_rwa": 0.5,
                "avg_coverage": 1.0,
                "threshold": float((threshold_params or {}).get("threshold", 0.0)),
                # New asymmetric regime params
                "bull_long_mult": float((threshold_params or {}).get("bull_long_mult", 0.8)),
                "bull_short_mult": float((threshold_params or {}).get("bull_short_mult", 1.0)),
                "bear_long_mult": float((threshold_params or {}).get("bear_long_mult", 1.5)),
                "bear_short_mult": float((threshold_params or {}).get("bear_short_mult", 0.8)),
                "crisis_long_mult": float((threshold_params or {}).get("crisis_long_mult", 2.5)),
                "crisis_short_mult": float((threshold_params or {}).get("crisis_short_mult", 2.0)),
                "conf_threshold": float((threshold_params or {}).get("conf_threshold", 0.5)),
                "vol_scaler": float((threshold_params or {}).get("vol_scaler", 0.5)),
                # Legacy fields (some downstream consumers still expect these)
                "bull_mult": float((threshold_params or {}).get("bull_mult", (threshold_params or {}).get("bull_long_mult", 0.8))),
                "bear_mult": float((threshold_params or {}).get("bear_mult", (threshold_params or {}).get("bear_long_mult", 1.5))),
                "crisis_mult": float((threshold_params or {}).get("crisis_mult", (threshold_params or {}).get("crisis_long_mult", 2.5))),
                "fold_failures": fold_failures,
                "skipped_seq": skipped_due_to_sequences,
                "skipped_val": skipped_due_to_short_val,
            }

            del track_c
            gc.collect()
            return final_score, metadata

        # ================================================================
        # POOLED SHARPE/RWA COMPUTATION
        # Instead of averaging noisy per-fold Sharpe estimates,
        # compute a single robust Sharpe from all pooled non-overlapping samples
        # ================================================================
        pooled_returns = np.array(pooled_strategy_returns)
        pooled_dirs = np.array(pooled_directions)
        n_pooled = len(pooled_returns)
        
        self.logger.info(f"Pooled samples: n={n_pooled} from {len(fold_scores)} folds")
        
        # Compute pooled Sharpe ratio from non-overlapping returns
        # Pooled returns are already non-overlapping (sampled [::horizon] in _extract_fold_raw_metrics)
        # So we compute mean and std directly from all pooled samples
        if n_pooled >= 3:
            mean_ret = float(np.mean(pooled_returns))
            std_ret = float(np.std(pooled_returns)) if n_pooled > 1 else 1e-9
            
            # Annualization factor: 252 trading days / horizon
            annual_factor = np.sqrt(252.0 / horizon)
            
            if std_ret > 1e-9:
                pooled_sharpe = (mean_ret / std_ret) * annual_factor
            else:
                pooled_sharpe = 0.0
            
            # Log for debugging (no cap - we want real values)
            self.logger.info(f"Pooled Sharpe: n={n_pooled}, mean={mean_ret:.6f}, std={std_ret:.6f}, sharpe={pooled_sharpe:.3f}")
            
            # Compute pooled RWA (directional accuracy weighted by return magnitude)
            correct_directions = pooled_dirs > 0
            rwa_numerator = float(np.sum(np.abs(pooled_returns) * correct_directions))
            rwa_denominator = float(np.sum(np.abs(pooled_returns)))
            pooled_rwa = rwa_numerator / rwa_denominator if rwa_denominator > 1e-9 else 0.5
        else:
            pooled_sharpe = 0.0
            pooled_rwa = 0.5
        
        # Compute stability from fold mean prediction variance
        # (Stable models should have consistent predictions across folds)
        if len(fold_mean_preds) > 1:
            pred_std = float(np.std(fold_mean_preds))
            pred_mean = float(np.abs(np.mean(fold_mean_preds))) + 1e-9
            cv = pred_std / pred_mean  # Coefficient of variation
            pooled_stability = max(0.0, 1.0 - cv)  # Lower CV = higher stability
        else:
            pooled_stability = 0.5
        
        # Combined objective: weighted sum of Sharpe, RWA, stability
        # Weights: 0.65 * Sharpe + 0.20 * RWA + 0.15 * stability
        avg_score = 0.65 * pooled_sharpe + 0.20 * pooled_rwa + 0.15 * pooled_stability
        
        avg_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        
        # Sign-stability penalty: penalize sign flips between consecutive fold predictions
        # This discourages models that flip between bullish/bearish across adjacent folds
        sign_stability_penalty = 0.0
        if len(fold_mean_preds) >= 2:
            sign_flips = 0
            for i in range(len(fold_mean_preds) - 1):
                sign_i = np.sign(fold_mean_preds[i]) if abs(fold_mean_preds[i]) > 1e-9 else 0
                sign_next = np.sign(fold_mean_preds[i + 1]) if abs(fold_mean_preds[i + 1]) > 1e-9 else 0
                if sign_i != sign_next and sign_i != 0 and sign_next != 0:
                    sign_flips += 1
            # Normalize by number of transitions and apply penalty weight
            sign_flip_rate = sign_flips / (len(fold_mean_preds) - 1)
            sign_stability_penalty = 0.15 * sign_flip_rate  # 0.15 weight for sign instability
        
        final_score = avg_score - sign_stability_penalty

        metadata = {
            "track_a_dims": track_a_dims,
            "track_b_dims": track_b_dims,
            "track_c_dims": track_c_dims,
            "avg_val_loss": avg_loss,
            "n_successful_folds": len(fold_scores),
            "n_pooled_samples": n_pooled,
            "avg_score": avg_score,
            "pooled_sharpe": pooled_sharpe,
            "pooled_rwa": pooled_rwa,
            "pooled_stability": pooled_stability,
            "sign_flip_rate": sign_flip_rate if len(fold_mean_preds) >= 2 else 0.0,
            "sign_stability_penalty": sign_stability_penalty,
            "threshold": float((threshold_params or {}).get("threshold", 0.0)),
            # New asymmetric regime params
            "bull_long_mult": float((threshold_params or {}).get("bull_long_mult", 0.8)),
            "bull_short_mult": float((threshold_params or {}).get("bull_short_mult", 1.0)),
            "bear_long_mult": float((threshold_params or {}).get("bear_long_mult", 1.5)),
            "bear_short_mult": float((threshold_params or {}).get("bear_short_mult", 0.8)),
            "crisis_long_mult": float((threshold_params or {}).get("crisis_long_mult", 2.5)),
            "crisis_short_mult": float((threshold_params or {}).get("crisis_short_mult", 2.0)),
            "conf_threshold": float((threshold_params or {}).get("conf_threshold", 0.5)),
            "vol_scaler": float((threshold_params or {}).get("vol_scaler", 0.5)),
            # Legacy fields
            "bull_mult": float((threshold_params or {}).get("bull_mult", (threshold_params or {}).get("bull_long_mult", 0.8))),
            "bear_mult": float((threshold_params or {}).get("bear_mult", (threshold_params or {}).get("bear_long_mult", 1.5))),
            "crisis_mult": float((threshold_params or {}).get("crisis_mult", (threshold_params or {}).get("crisis_long_mult", 2.5))),
            "fold_failures": fold_failures,
            "skipped_seq": skipped_due_to_sequences,
            "skipped_val": skipped_due_to_short_val,
            # Keep these for backwards compatibility
            "avg_sharpe": pooled_sharpe,
            "avg_stability": pooled_stability,
            "avg_rwa": pooled_rwa,
            "avg_coverage": 1.0 - (sign_stability_penalty / 0.15) if sign_stability_penalty > 0 else 1.0,
            "elapsed_time": _time.time() - _start_eval if '_time' in dir() else 0.0,
        }

        # Cleanup: free track_c and force garbage collection before returning
        del track_c
        gc.collect()
        
        return final_score, metadata
    
    def _run_fold_jobs_ray(
        self,
        fold_jobs: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute fold trainings in parallel via Ray tasks when available."""
        if not RAY_AVAILABLE or ray is None:
            return None

        desired_parallel = min(max(0, self.config.ray_fold_parallelism), len(fold_jobs))
        if desired_parallel <= 1:
            return None

        try:
            if not ray.is_initialized():
                init_kwargs: Dict[str, Any] = {}
                init_kwargs = _apply_ray_runtime_env_envvars(init_kwargs)
                ray.init(ignore_reinit_error=True, log_to_driver=False, **init_kwargs)
        except Exception as init_err:  # pragma: no cover - ray runtime
            self.logger.warning("Ray init failed for fold parallelism: %s", init_err)
            return None

        cluster = ray.cluster_resources()
        total_gpu = float(cluster.get("GPU", 0.0))
        total_cpu = float(cluster.get("CPU", 0.0))
        if total_gpu <= 0 or total_cpu <= 0:
            self.logger.debug("Fold parallelism skipped: insufficient cluster resources")
            return None

        gpu_fraction = self.config.ray_fold_gpu_fraction
        if not gpu_fraction or gpu_fraction <= 0:
            gpu_fraction = max(0.1, min(1.0, 1.0 / desired_parallel))

        max_gpu_slots = max(1, int(total_gpu / max(gpu_fraction, 1e-6)))
        usable_parallel = min(desired_parallel, max_gpu_slots, int(total_cpu))
        mem_note = ""

        gpu_stats = _detect_gpu_memory()
        reserve_gb = max(0.0, self.config.fold_gpu_reserve_gb)
        per_fold_mem_gb = max(0.1, self.config.fold_gpu_memory_gb)
        if gpu_stats and "min_free_gb" in gpu_stats:
            free_gb = gpu_stats.get("min_free_gb", 0.0)
            usable_free = max(0.0, free_gb - reserve_gb)
            max_mem_parallel = int(usable_free // per_fold_mem_gb)
            if max_mem_parallel <= 0:
                self.logger.debug(
                    "Fold parallelism skipped: free GPU %.2f GB < reserve %.2f GB + per-fold %.2f GB",
                    free_gb,
                    reserve_gb,
                    per_fold_mem_gb,
                )
                return None
            prev_parallel = usable_parallel
            usable_parallel = min(usable_parallel, max_mem_parallel)
            if usable_parallel < prev_parallel:
                mem_note = f", limited_by_mem={usable_parallel}/{prev_parallel}"
            mem_note += f", free_gb={free_gb:.2f}, reserve_gb={reserve_gb:.2f}"

        if usable_parallel <= 1:
            return None

        self.logger.info(
            "⚡ Optuna fold Ray parallelism: %d folds (%d-way, gpu_fraction=%.2f%s)",
            len(fold_jobs),
            usable_parallel,
            gpu_fraction,
            mem_note,
        )

        cfg_ref = ray.put(cfg)
        pending: List[Tuple[Any, Any, int]] = []
        results: List[Dict[str, Any]] = []

        def _resolve_pending() -> None:
            nonlocal pending, results
            if not pending:
                return
            refs = [ref for ref, _, _ in pending]
            seq_refs = [seq_ref for _, seq_ref, _ in pending]
            fold_ids = [fold_id for _, _, fold_id in pending]
            try:
                payloads = ray.get(refs)
            except Exception as batch_err:
                self.logger.debug("Fold batch %s Ray execution failed: %s", fold_ids, batch_err)
                payloads = []
            finally:
                for seq_ref in seq_refs:
                    del seq_ref

            for payload in payloads:
                if payload:
                    results.append(payload)
            pending = []

        # Calculate max memory per fold: (total - reserve) / parallelism
        max_mem_per_fold = max(0.5, (8.0 - reserve_gb) / usable_parallel)
        
        try:
            for job in fold_jobs:
                seq_ref = ray.put(job["seq_data"])
                try:
                    # Escape placement group if running inside Ray Tune trial
                    task_options = {"num_gpus": gpu_fraction}
                    if PlacementGroupSchedulingStrategy is not None:
                        task_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(placement_group=None)
                    
                    future = _OptunaFoldTrainTask.options(**task_options).remote(
                        seq_ref,
                        job["seq_train_idx"],
                        job["seq_val_idx"],
                        cfg_ref,
                        job["fold_idx"],
                        max_mem_per_fold,  # GPU memory limit per fold
                    )
                except Exception as submit_err:
                    self.logger.debug(
                        "Fold %d Ray launch failed: %s",
                        job["fold_idx"],
                        submit_err,
                    )
                    del seq_ref
                    continue

                pending.append((future, seq_ref, job["fold_idx"]))
                if len(pending) >= usable_parallel:
                    _resolve_pending()

            _resolve_pending()
        finally:
            del cfg_ref

        return results
    
    def _get_encoding_config(self, family: str) -> Tuple[str, int, int]:
        """Get dimension bounds for a family.
        
        UPDATED: No longer hardcodes method (PCA vs AE).
        Optuna now chooses dim_reduction_type per family per trial.
        
        Returns:
            Tuple of (placeholder_method, min_dim, max_dim)
            - placeholder_method: "optuna" (actual method suggested in _suggest_family_params)
            - min_dim: minimum latent dimension
            - max_dim: maximum latent dimension (capped by config)
        """
        size = self.family_sizes.get(family, 0)
        
        # Use config bounds for all families - Optuna chooses PCA vs AE
        # PCA range: [pca_components_min, pca_components_max]
        # AE range: [ae_latent_dim_min, ae_latent_dim_max]
        # We use the union of these ranges as the search space
        min_dim = min(self.config.pca_components_min, self.config.ae_latent_dim_min)
        max_dim = max(self.config.pca_components_max, self.config.ae_latent_dim_max)
        
        # Cap at family size (can't compress to more dims than input)
        max_dim = min(max_dim, size) if size > 0 else max_dim
        min_dim = min(min_dim, max_dim)
        
        # Return "optuna" as placeholder - actual method chosen in _suggest_family_params
        return "optuna", min_dim, max_dim
    
    def _normalize_family_weights(
        self,
        weights: Dict[str, float],
        method: str = "softmax",
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Normalize family weights for stability and soft-dropping.
        
        Args:
            weights: Raw family weights {family_name: weight}
            method: Normalization method:
                - "softmax": Temperature-scaled softmax (standard)
                - "sparsemax": Softmax with sparsity (true soft-drops)
                - "sum": Linear sum normalization
                - "none": No normalization (raw weights)
            temperature: Temperature scaling (lower = more sparse/aggressive)
            
        Returns:
            Normalized weights. Sum varies by method:
            - softmax/sum: sum = len(weights) for consistent scale
            - sparsemax: sum <= len(weights), truly low weights
            - none: sum = original sum
        """
        if not weights or method == "none":
            return weights
        
        families = list(weights.keys())
        raw_weights = np.array([weights[f] for f in families])
        
        if method == "softmax":
            # Temperature-scaled softmax normalization
            # Higher temperature = more uniform, lower = winner-take-all
            scaled = raw_weights / max(temperature, 0.01)
            # Numerical stability: subtract max before exp
            shifted = scaled - np.max(scaled)
            exp_weights = np.exp(shifted)
            normalized = exp_weights / (np.sum(exp_weights) + 1e-8)
            # Scale so sum = number of families (preserves average magnitude of 1.0)
            normalized = normalized * len(families)
            
        elif method == "sparsemax":
            # Sparse normalization for true soft-dropping
            # Low weights get pushed toward clip_min, high weights amplified
            # This creates actual sparsity in the weight distribution
            
            # Step 1: Apply power transformation (amplifies differences)
            power = 2.0 / max(temperature, 0.1)  # Lower temp = higher power = more sparse
            powered = np.power(raw_weights + 1e-8, power)
            
            # Step 2: Normalize to sum=len(families)
            normalized = powered / (np.sum(powered) + 1e-8) * len(families)
            
            # Step 3: Apply floor at clip_min (never true zero)
            clip_min = 0.01
            normalized = np.maximum(normalized, clip_min)
            
        elif method == "sum":
            # Simple sum normalization
            total = np.sum(raw_weights) + 1e-8
            normalized = raw_weights / total * len(families)
            
        else:
            normalized = raw_weights
        
        return {f: float(w) for f, w in zip(families, normalized)}
    
    def _suggest_family_params(self, trial: "Trial", family: str) -> Tuple[bool, int, str, float, Dict[str, Any]]:
        """Suggest Optuna parameters for a single family.
        
        Returns:
            Tuple of (include_family, latent_dim, encoder_method, weight, ae_params)
            - ae_params: Dict with AE hyperparameters (only used if method="ae")
            
        UPDATED: Optuna now chooses PCA vs AE per family per trial.
        No more hardcoded rules like ">30 cols = PCA".
        
        New hyperparameters suggested:
            - dim_reduction_type: "pca" or "ae"
            - For PCA: pca_components in [10, 60]
            - For AE: ae_latent_dim in [8, 64], ae_layers in {1,2,3},
                      ae_activation in {relu, leakyrelu, gelu},
                      ae_dropout in [0.0, 0.4], ae_lr in [1e-5, 1e-3]
        
        SOFT DROP DESIGN (no hard exclusions):
        ======================================
        - ALL families are always included (encoders fitted, shapes constant)
        - Weight determines contribution: low weight = soft drop
        - Weight clipped to >= clip_min (0.01) to prevent true zero
        - After softmax normalization, soft-dropped families have ~0.01-0.03 weight
        - This preserves pipeline stability while effectively "dropping" families
        
        Benefits:
        - Feature shapes constant across all folds
        - PCA/AE encoders always usable
        - No fold invalidation
        - Gradient-friendly optimization
        """
        def _suggest_fixed(name: str, value: Any) -> Any:
            # Use a single-choice categorical so the value is recorded in trial.params
            # (important for caching, serialization, and later extraction).
            return trial.suggest_categorical(name, [value])

        fixed = dict(getattr(self.config, "fixed_params", {}) or {})

        # Stage-B families are always included with minimum weight (passthrough)
        if family in MODEL_FAMILIES or family in HF_BLOCK_FAMILIES:
            weight = trial.suggest_float(
                f"weight_{family}",
                self.config.stage_b_family_min_weight,
                self.config.family_weight_max,
            )
            return True, 0, "passthrough", weight, {}

        # HF families (including Stage-A *_hf) must remain raw passthrough.
        if _is_raw_passthrough_family(family):
            raw_weight = trial.suggest_float(
                f"weight_{family}",
                self.config.family_weight_min,
                self.config.family_weight_max,
            )
            weight = max(raw_weight, self.config.family_weight_clip_min)
            return True, 0, "passthrough", weight, {}
        
        # Tier 1: Family weight (continuous, replaces boolean toggle)
        # Range [0.0, 1.0] - Optuna explores full space
        raw_weight = trial.suggest_float(
            f"weight_{family}",
            self.config.family_weight_min,
            self.config.family_weight_max,
        )
        
        # SOFT DROP: Clip to minimum floor (0.01) - never true zero
        # This ensures:
        #   - Family is always "included" (encoder fitted, columns created)
        #   - Very low weight = effectively dropped (contributes ~0)
        #   - Feature shapes remain constant across folds
        weight = max(raw_weight, self.config.family_weight_clip_min)
        
        # Get family size
        family_size = self.family_sizes.get(family, 0)
        
        # Handle zero-size families (no columns available)
        if family_size == 0:
            return False, 0, "none", 0.0, {}  # True exclusion only for missing data
        
        # =========================================================================
        # 3-PILLAR DIMENSIONALITY: Use pre-computed optimal dims if available
        # =========================================================================
        ae_params: Dict[str, Any] = {}

        # =========================================================================
        # WEIGHTS-ONLY MODE: freeze encoder/dim choices to fixed values
        # =========================================================================
        if bool(getattr(self.config, "tune_weights_only", False)):
            # Prefer explicit fixed params; otherwise fall back to 3-pillar dims when available.
            if self.config.use_three_pillar_dims and family in self.family_optimal_dims:
                dim_config = self.family_optimal_dims[family]
                method = str(dim_config.method)
                latent_dim = int(dim_config.k_final)
            else:
                method = str(fixed.get(f"{family}_dim_type", "pca") or "pca")
                if method == "ae":
                    latent_dim = int(fixed.get(f"{family}_ae_dim", 8) or 8)
                else:
                    latent_dim = int(fixed.get(f"{family}_pca_dim", 10) or 10)

            method = _suggest_fixed(f"{family}_dim_type", method)
            if method == "ae":
                latent_dim = int(_suggest_fixed(f"{family}_ae_dim", int(latent_dim)))
                ae_params = {
                    "ae_layers": int(_suggest_fixed(f"{family}_ae_layers", int(fixed.get(f"{family}_ae_layers", 2) or 2))),
                    "ae_activation": str(_suggest_fixed(f"{family}_ae_activation", str(fixed.get(f"{family}_ae_activation", "gelu") or "gelu"))),
                    "ae_dropout": float(_suggest_fixed(f"{family}_ae_dropout", float(fixed.get(f"{family}_ae_dropout", 0.1) or 0.1))),
                    "ae_lr": float(_suggest_fixed(f"{family}_ae_lr", float(fixed.get(f"{family}_ae_lr", 5e-4) or 5e-4))),
                }
            else:
                latent_dim = int(_suggest_fixed(f"{family}_pca_dim", int(latent_dim)))

            return True, int(latent_dim), str(method), float(weight), ae_params
        
        if self.config.use_three_pillar_dims and family in self.family_optimal_dims:
            # Use pre-computed optimal dimension from 3-pillar analysis
            dim_config = self.family_optimal_dims[family]
            
            # Method is pre-determined by 3-pillar analysis
            method = dim_config.method
            latent_dim = dim_config.k_final
            
            # For AE, use default architecture (no Optuna search needed)
            if method == "ae":
                ae_params = {
                    "ae_layers": 2,
                    "ae_activation": "gelu",
                    "ae_dropout": 0.1,
                    "ae_lr": 5e-4,
                }
            
            # Log for visibility
            # trial.set_user_attr(f"{family}_three_pillar", True)
        
        else:
            # =========================================================================
            # LEGACY: Optuna chooses dim_reduction_type per family (fallback)
            # =========================================================================
            method = trial.suggest_categorical(
                f"{family}_dim_type",
                list(self.config.dim_reduction_types),  # ["pca", "ae"]
            )
            
            # Tier 2: Dimensionality based on chosen method
            if method == "pca":
                # PCA: suggest components in [pca_components_min, pca_components_max]
                min_dim = self.config.pca_components_min
                max_dim = min(self.config.pca_components_max, family_size)
                min_dim = min(min_dim, max_dim)
                
                if min_dim >= max_dim:
                    latent_dim = min_dim
                else:
                    latent_dim = trial.suggest_int(f"{family}_pca_dim", min_dim, max_dim)
                    
            else:  # method == "ae"
                # AE: suggest latent_dim in [ae_latent_dim_min, ae_latent_dim_max]
                min_dim = self.config.ae_latent_dim_min
                max_dim = min(self.config.ae_latent_dim_max, family_size)
                min_dim = min(min_dim, max_dim)
                
                if min_dim >= max_dim:
                    latent_dim = min_dim
                else:
                    latent_dim = trial.suggest_int(f"{family}_ae_dim", min_dim, max_dim)
                
                # AE architecture hyperparameters
                ae_params = {
                    "ae_layers": trial.suggest_categorical(
                        f"{family}_ae_layers",
                        list(self.config.ae_layers_choices),
                    ),
                    "ae_activation": trial.suggest_categorical(
                        f"{family}_ae_activation",
                        list(self.config.ae_activation_choices),
                    ),
                    "ae_dropout": trial.suggest_float(
                        f"{family}_ae_dropout",
                        self.config.ae_dropout_min,
                        self.config.ae_dropout_max,
                    ),
                    "ae_lr": trial.suggest_float(
                        f"{family}_ae_lr",
                        self.config.ae_lr_min,
                        self.config.ae_lr_max,
                        log=True,
                    ),
                }
        
        # Always return include=True for soft-drop stability
        return True, latent_dim, method, weight, ae_params
    
    def _suggest_track_weights(self, trial: "Trial") -> Tuple[float, float]:
        """Suggest Track-B weight only.
        
        Track-A weight is HARD-CODED to 1.0 because per-family weights
        already control individual family contributions within Track A.
        Adding a global track_a_weight would be redundant double-weighting.
        
        Track-B weight IS tuned because block summaries have no per-family
        weighting (legacy). We still keep a global Track-B multiplier to
        control Track B importance relative to Track A.
        """
        w_a = 1.0  # HARD-CODED: family weights already control Track A
        if bool(getattr(self.config, "tune_weights_only", False)):
            fixed = dict(getattr(self.config, "fixed_params", {}) or {})
            w_b = trial.suggest_categorical("track_b_weight", [float(fixed.get("track_b_weight", 1.0))])
        else:
            w_b = trial.suggest_float(
                "track_b_weight",
                self.config.track_b_weight_min,
                self.config.track_b_weight_max,
            )
        return w_a, w_b

    def _suggest_track_b_family_weights(self, trial: "Trial") -> Dict[str, float]:
        """Suggest per-family weights for Track B.

        Track B families are treated as a separate category from Stage-A families.
        Weights are applied directly to raw (uncompressed) Track-B columns.

        Parameter names are namespaced to avoid collisions:
          - HF block family:  weight_b_<family>
          - Summary blocks:    weight_b_<summary_name>
        """
        weights: Dict[str, float] = {}

        # HF block families (raw passthrough columns)
        for family in HF_BLOCK_FAMILIES:
            raw = trial.suggest_float(
                f"weight_b_{family}",
                self.config.stage_b_family_min_weight,
                self.config.family_weight_max,
            )
            weights[family] = max(float(raw), float(self.config.family_weight_clip_min))

        # Track-B summary blocks (raw passthrough frames)
        for summary_name in TRACK_B_SUMMARY_BLOCKS:
            raw = trial.suggest_float(
                f"weight_b_{summary_name}",
                self.config.stage_b_family_min_weight,
                self.config.family_weight_max,
            )
            weights[summary_name] = max(float(raw), float(self.config.family_weight_clip_min))

        return weights
    
    def _suggest_lstm_params(self, trial: "Trial") -> Dict[str, Any]:
        """Suggest LSTM and training hyperparameters with HORIZON-BASED FORMULAS.
        
        HORIZON-BASED FORMULAS (H = config.horizon):
            - seq_len: H * [1.0, 3.0]
            - hidden_dim: [32 + 0.8*H, 64 + 2.0*H]
            - dropout_max: min(0.6, 0.05 + 0.004*H)
            - loss_fn: weighted by horizon (MSE/Huber for short, Quantile/NLL for long)
        
        Includes:
            - Core LSTM architecture (layers, hidden, dropout, seq_len)
            - Attention mechanism (type, heads, normalization, regularizers)
            - Optimizer settings (batch, lr, optimizer type, activation)
            - 4.3 Loss function choice (horizon-weighted)
            - 4.6 Feature smoothing
            - 4.7 Train window fraction
            - 4.8 Regularization & noise injection
            - 4.9 Learning rate scheduler
        """
        H = self.config.horizon
        
        # =====================================================================
        # HORIZON-BASED DYNAMIC BOUNDS
        # =====================================================================
        # seq_len: H * [1.0, 3.0]
        seq_len_min = max(10, int(H * self.config.seq_len_horizon_mult_min))
        seq_len_max = max(seq_len_min + 10, int(H * self.config.seq_len_horizon_mult_max))
        
        # hidden_dim: [32 + 0.8*H, 64 + 2.0*H]
        hidden_min = int(self.config.hidden_base_min + self.config.hidden_horizon_mult_min * H)
        hidden_max = int(self.config.hidden_base_max + self.config.hidden_horizon_mult_max * H)
        
        # dropout_max: min(0.6, 0.05 + 0.004*H)
        dropout_max = min(self.config.dropout_max_cap, 
                          self.config.dropout_base + self.config.dropout_horizon_mult * H)
        
        params = {}
        
        # =====================================================================
        # LSTM CORE PARAMS (Horizon-dependent)
        # =====================================================================
        # Sequence length (horizon-based)
        params["lstm_seq_len"] = trial.suggest_int(
            "lstm_seq_len", seq_len_min, seq_len_max,
        )
        
        # Hidden dimension (horizon-based)
        params["lstm_hidden_dim"] = trial.suggest_int(
            "lstm_hidden_dim", hidden_min, hidden_max,
        )
        
        # Dropout (horizon-based max)
        params["lstm_dropout"] = trial.suggest_float(
            "lstm_dropout", self.config.dropout_base, dropout_max,
        )
        
        # =====================================================================
        # LSTM OPTIMIZATION PARAMS (Global Optuna)
        # =====================================================================
        # Batch size
        params["lstm_batch_size"] = trial.suggest_categorical(
            "lstm_batch_size", list(self.config.lstm_batch_sizes),
        )
        
        # Learning rate
        params["lstm_learning_rate"] = trial.suggest_float(
            "lstm_learning_rate", self.config.lstm_lr_min, self.config.lstm_lr_max, log=True,
        )
        
        # Optimizer
        params["lstm_optimizer"] = trial.suggest_categorical(
            "lstm_optimizer", list(self.config.lstm_optimizers),
        )
        
        # LR Scheduler (NEW)
        params["lstm_lr_scheduler"] = trial.suggest_categorical(
            "lstm_lr_scheduler", ["cosine", "step", "plateau", "cyclical", "warmup_cosine", "none"],
        )
        
        # Loss Function (NEW)
        params["lstm_loss_fn"] = trial.suggest_categorical(
            "lstm_loss_fn", ["mse", "huber", "quantile"],
        )
        
        # HARD-CODED: use_amp always True (not Optuna-tuned)
        params["lstm_use_amp"] = True
        
        # =====================================================================
        # CNN FRONTEND (Global Optuna) - Dilated Residual CNN
        # Used at Jane Street, Jump, Two Sigma for local pattern extraction
        # =====================================================================
        params["cnn_frontend_enabled"] = trial.suggest_categorical(
            "cnn_frontend_enabled", list(self.config.cnn_frontend_enabled),
        )
        
        # Only suggest CNN params if frontend is enabled
        if params["cnn_frontend_enabled"]:
            params["cnn_blocks"] = trial.suggest_int(
                "cnn_blocks", self.config.cnn_blocks_min, self.config.cnn_blocks_max,
            )
            params["cnn_filters"] = trial.suggest_categorical(
                "cnn_filters", list(self.config.cnn_filters),
            )
            params["cnn_kernel_size"] = trial.suggest_categorical(
                "cnn_kernel_size", list(self.config.cnn_kernel_sizes),
            )
            params["cnn_dilation"] = trial.suggest_categorical(
                "cnn_dilation", list(self.config.cnn_dilations),
            )
            params["cnn_pooling"] = trial.suggest_categorical(
                "cnn_pooling", list(self.config.cnn_pooling),
            )
            params["cnn_stride"] = trial.suggest_categorical(
                "cnn_stride", list(self.config.cnn_strides),
            )
            params["cnn_activation"] = trial.suggest_categorical(
                "cnn_activation", list(self.config.cnn_activations),
            )
            params["cnn_dropout"] = trial.suggest_float(
                "cnn_dropout", self.config.cnn_dropout_min, self.config.cnn_dropout_max,
            )
            params["cnn_batch_norm"] = trial.suggest_categorical(
                "cnn_batch_norm", list(self.config.cnn_batch_norm_choices),
            )
            params["cnn_layer_norm"] = trial.suggest_categorical(
                "cnn_layer_norm", list(self.config.cnn_layer_norm_choices),
            )
            params["cnn_residual"] = trial.suggest_categorical(
                "cnn_residual", list(self.config.cnn_residual_choices),
            )
        else:
            # Default values when CNN is disabled
            params["cnn_blocks"] = 0
            params["cnn_filters"] = 64
            params["cnn_kernel_size"] = 3
            params["cnn_dilation"] = 1
            params["cnn_pooling"] = None
            params["cnn_stride"] = 1
            params["cnn_activation"] = "gelu"
            params["cnn_dropout"] = 0.1
            params["cnn_batch_norm"] = True
            params["cnn_layer_norm"] = False
            params["cnn_residual"] = True
        
        # =====================================================================
        # ADVANCED REGULARIZATION (Global Optuna)
        # =====================================================================
        # Label smoothing: prevents overconfident predictions
        params["label_smoothing"] = trial.suggest_float(
            "label_smoothing",
            self.config.label_smoothing_min,
            self.config.label_smoothing_max,
        )
        
        # Stochastic depth: randomly drop layers during training
        params["stochastic_depth"] = trial.suggest_float(
            "stochastic_depth",
            self.config.stochastic_depth_min,
            self.config.stochastic_depth_max,
        )
        
        # Mixout: interpolates between dropout and keeping original weights
        params["mixout_prob"] = trial.suggest_float(
            "mixout_prob",
            self.config.mixout_prob_min,
            self.config.mixout_prob_max,
        )
        
        # =====================================================================
        # ATTENTION ENHANCEMENTS (Global Optuna)
        # =====================================================================
        # Multi-layer attention: stack multiple attention layers
        params["attn_layers"] = trial.suggest_int(
            "attn_layers",
            self.config.attn_layers_min,
            self.config.attn_layers_max,
        )
        
        # Rotary positional embedding (RoPE): modern position encoding
        params["rotary_embedding"] = trial.suggest_categorical(
            "rotary_embedding",
            list(self.config.rotary_embedding_choices),
        )
        
        # Feedforward dimension in attention block (MLP expansion)
        params["feedforward_dim"] = trial.suggest_int(
            "feedforward_dim",
            self.config.feedforward_dim_min,
            self.config.feedforward_dim_max,
        )
        
        # =====================================================================
        # HEAD ARCHITECTURE (Global Optuna)
        # =====================================================================
        # Dense/FC activation choices
        params["dense_activation"] = trial.suggest_categorical(
            "dense_activation",
            list(self.config.dense_activations),
        )
        
        # Batch normalization in output head
        params["batch_norm_head"] = trial.suggest_categorical(
            "batch_norm_head",
            list(self.config.batch_norm_head_choices),
        )
        
        # =====================================================================
        # LSTM ARCHITECTURE (Global Optuna)
        # =====================================================================
        # Core LSTM architecture
        params["lstm_layers"] = trial.suggest_int(
            "lstm_layers", self.config.lstm_layers_min, self.config.lstm_layers_max,
        )
        params["lstm_activation"] = trial.suggest_categorical(
            "lstm_activation", list(self.config.lstm_activations),
        )
        
        # Bidirectional LSTM (now tunable instead of hard-coded)
        params["lstm_bidirectional"] = trial.suggest_categorical(
            "lstm_bidirectional",
            list(self.config.lstm_bidirectional_choices),
        )
        
        # Cell type (LSTM vs GRU)
        params["lstm_cell_type"] = trial.suggest_categorical(
            "lstm_cell_type", list(self.config.lstm_cell_types),
        )
        
        # Kernel initialization (critical for LSTM stability)
        params["lstm_recurrent_kernel_init"] = trial.suggest_categorical(
            "lstm_recurrent_kernel_init", list(self.config.lstm_recurrent_kernel_inits),
        )
        
        # Hidden state initialization
        params["lstm_hidden_state_init"] = trial.suggest_categorical(
            "lstm_hidden_state_init", list(self.config.lstm_hidden_state_inits),
        )
        
        # =====================================================================
        # LSTM REGULARIZATION (Global Optuna)
        # =====================================================================
        params["lstm_recurrent_dropout"] = trial.suggest_float(
            "lstm_recurrent_dropout", 
            self.config.lstm_recurrent_dropout_min, 
            self.config.lstm_recurrent_dropout_max,
        )
        params["lstm_input_dropout"] = trial.suggest_float(
            "lstm_input_dropout",
            self.config.lstm_input_dropout_min,
            self.config.lstm_input_dropout_max,
        )
        params["lstm_grad_clip"] = trial.suggest_float(
            "lstm_grad_clip",
            self.config.lstm_grad_clip_min,
            self.config.lstm_grad_clip_max,
        )
        # Time-wise dropout (dropout over time dimension)
        params["lstm_time_dropout"] = trial.suggest_float(
            "lstm_time_dropout",
            self.config.lstm_time_dropout_min,
            self.config.lstm_time_dropout_max,
        )
        # L2 recurrent weight regularization
        params["lstm_recurrent_weight_decay"] = trial.suggest_float(
            "lstm_recurrent_weight_decay",
            self.config.lstm_recurrent_weight_decay_min,
            self.config.lstm_recurrent_weight_decay_max,
        )
        # Weight dropout (AWD-LSTM style)
        params["lstm_weight_dropout"] = trial.suggest_float(
            "lstm_weight_dropout",
            self.config.lstm_weight_dropout_min,
            self.config.lstm_weight_dropout_max,
        )
        # Zoneout (RNN stabilizer)
        params["lstm_zoneout"] = trial.suggest_float(
            "lstm_zoneout",
            self.config.lstm_zoneout_min,
            self.config.lstm_zoneout_max,
        )
        # Sequence noise injection
        params["lstm_sequence_noise_std"] = trial.suggest_float(
            "lstm_sequence_noise_std",
            self.config.lstm_sequence_noise_min,
            self.config.lstm_sequence_noise_max,
        )
        
        # =====================================================================
        # LSTM ENHANCEMENTS (Global Optuna)
        # =====================================================================
        params["lstm_layer_norm"] = trial.suggest_categorical(
            "lstm_layer_norm", list(self.config.lstm_layer_norm_choices),
        )
        params["lstm_residual"] = trial.suggest_categorical(
            "lstm_residual", list(self.config.lstm_residual_choices),
        )
        # Skip connections between LSTM layers
        params["lstm_skip_connect"] = trial.suggest_categorical(
            "lstm_skip_connect", list(self.config.lstm_skip_connect_choices),
        )
        
        # LR multiplier (layer-wise scaling)
        params["lstm_lr_multiplier"] = trial.suggest_float(
            "lstm_lr_multiplier",
            self.config.lstm_lr_multiplier_min,
            self.config.lstm_lr_multiplier_max,
        )
        
        # =====================================================================
        # LSTM ATTENTION HYPERPARAMETERS (Global Optuna)
        # =====================================================================
        # Attention type selection
        params["lstm_attention_type"] = trial.suggest_categorical(
            "lstm_attention_type", list(self.config.lstm_attention_types),
        )
        
        # Only suggest attention params if attention is enabled
        if params["lstm_attention_type"] != "none":
            # Attention hidden dimension
            params["lstm_attn_hidden_dim"] = trial.suggest_int(
                "lstm_attn_hidden_dim",
                self.config.lstm_attn_hidden_dim_min,
                self.config.lstm_attn_hidden_dim_max,
            )
            # Attention dropout
            params["lstm_attn_dropout"] = trial.suggest_float(
                "lstm_attn_dropout",
                self.config.lstm_attn_dropout_min,
                self.config.lstm_attn_dropout_max,
            )
            # Attention normalization
            params["lstm_attn_normalization"] = trial.suggest_categorical(
                "lstm_attn_normalization", list(self.config.lstm_attn_normalization_types),
            )
            # Number of attention heads (lightweight)
            params["lstm_attn_heads"] = trial.suggest_int(
                "lstm_attn_heads",
                self.config.lstm_attn_heads_min,
                self.config.lstm_attn_heads_max,
            )
            # Attention scoring function
            params["lstm_attn_score_fn"] = trial.suggest_categorical(
                "lstm_attn_score_fn", list(self.config.lstm_attn_score_functions),
            )
            # Context vector merge type
            params["lstm_attn_merge"] = trial.suggest_categorical(
                "lstm_attn_merge", list(self.config.lstm_attn_merge_types),
            )
            # Positional encoding
            params["lstm_attn_positional_encoding"] = trial.suggest_categorical(
                "lstm_attn_positional_encoding", 
                list(self.config.lstm_attn_positional_encoding_choices),
            )
            # Attention temperature
            params["lstm_attn_temperature"] = trial.suggest_float(
                "lstm_attn_temperature",
                self.config.lstm_attn_temperature_min,
                self.config.lstm_attn_temperature_max,
            )
            # Attention regularizers
            params["lstm_attn_entropy_reg"] = trial.suggest_float(
                "lstm_attn_entropy_reg",
                self.config.lstm_attn_entropy_reg_min,
                self.config.lstm_attn_entropy_reg_max,
            )
            params["lstm_attn_distance_reg"] = trial.suggest_float(
                "lstm_attn_distance_reg",
                self.config.lstm_attn_distance_reg_min,
                self.config.lstm_attn_distance_reg_max,
            )
            # Attention context window (how many timesteps attention sees)
            seq_len = params.get("lstm_seq_len", 30)
            context_max = max(self.config.lstm_attn_context_length_min + 5,
                              int(seq_len * self.config.lstm_attn_context_length_ratio_max))
            params["lstm_attn_context_length"] = trial.suggest_int(
                "lstm_attn_context_length",
                self.config.lstm_attn_context_length_min,
                context_max,
            )
            # Key/Value dims for scaled_dot attention
            if params["lstm_attention_type"] == "scaled_dot":
                params["lstm_attn_key_dim"] = trial.suggest_int(
                    "lstm_attn_key_dim",
                    self.config.lstm_attn_key_dim_min,
                    self.config.lstm_attn_key_dim_max,
                )
                params["lstm_attn_value_dim"] = trial.suggest_int(
                    "lstm_attn_value_dim",
                    self.config.lstm_attn_value_dim_min,
                    self.config.lstm_attn_value_dim_max,
                )
            else:
                params["lstm_attn_key_dim"] = 64
                params["lstm_attn_value_dim"] = 64
        else:
            # Defaults when attention is disabled
            params["lstm_attn_hidden_dim"] = 64
            params["lstm_attn_dropout"] = 0.1
            params["lstm_attn_normalization"] = "softmax"
            params["lstm_attn_heads"] = 1
            params["lstm_attn_score_fn"] = "dot"
            params["lstm_attn_merge"] = "concat"
            params["lstm_attn_positional_encoding"] = False
            params["lstm_attn_temperature"] = 1.0
            params["lstm_attn_entropy_reg"] = 0.0
            params["lstm_attn_distance_reg"] = 0.0
            params["lstm_attn_context_length"] = 30
            params["lstm_attn_key_dim"] = 64
            params["lstm_attn_value_dim"] = 64
        
        # =====================================================================
        # LSTM OUTPUT LAYERS (Global Optuna)
        # =====================================================================
        params["lstm_fc_layers"] = trial.suggest_int(
            "lstm_fc_layers",
            self.config.lstm_fc_layers_min,
            self.config.lstm_fc_layers_max,
        )
        params["lstm_fc_hidden"] = trial.suggest_int(
            "lstm_fc_hidden",
            self.config.lstm_fc_hidden_min,
            self.config.lstm_fc_hidden_max,
        )
        params["lstm_output_dropout"] = trial.suggest_float(
            "lstm_output_dropout",
            self.config.lstm_output_dropout_min,
            self.config.lstm_output_dropout_max,
        )
        # Output activation (for normalized returns vs direction prob)
        params["lstm_output_activation"] = trial.suggest_categorical(
            "lstm_output_activation", list(self.config.lstm_output_activations),
        )
        
        # =====================================================================
        # LSTM OPTIMIZATION PARAMS (Additional)
        # =====================================================================
        # Gradient accumulation (stabilizes training with small batch)
        params["lstm_gradient_accumulation"] = trial.suggest_int(
            "lstm_gradient_accumulation",
            self.config.lstm_gradient_accumulation_min,
            self.config.lstm_gradient_accumulation_max,
        )
        
        # Momentum (only for SGD/RMSprop)
        optimizer = params.get("lstm_optimizer", "AdamW")
        if optimizer in ("SGD", "RMSprop"):
            params["lstm_momentum"] = trial.suggest_float(
                "lstm_momentum",
                self.config.lstm_momentum_min,
                self.config.lstm_momentum_max,
            )
        else:
            params["lstm_momentum"] = 0.9  # Default, not used
        
        # =====================================================================
        # 4.3 LOSS FUNCTION CHOICE (HORIZON-WEIGHTED)
        # =====================================================================
        # Horizon-based loss function sampling weights:
        #   - Short horizons (H < 30): prefer MSE/Huber
        #   - Long horizons (H >= 30): prefer Quantile/NLL
        # Weights: [mse, huber, quantile, nll_gauss]
        loss_choices = list(self.config.loss_functions)
        
        # Compute horizon-weighted probabilities
        mse_weight = max(0.05, 1.0 - H / 50)      # High for short horizon
        huber_weight = max(0.05, 1.0 - H / 50)    # High for short horizon
        quantile_weight = max(0.05, min(1.0, H / 50))  # High for long horizon
        nll_weight = max(0.05, min(1.0, H / 30))       # High for long horizon
        
        # Normalize weights
        weight_map = {
            "mse": mse_weight,
            "huber": huber_weight,
            "quantile": quantile_weight,
            "nll_gauss": nll_weight,
        }
        total_weight = sum(weight_map.get(lf, 0.1) for lf in loss_choices)
        
        # Use weighted categorical (Optuna doesn't support weighted directly, so we use random)
        import random
        weights = [weight_map.get(lf, 0.1) / total_weight for lf in loss_choices]
        # Optuna will still sample uniformly, but we can bias by using a different approach
        # For now, just use uniform categorical
        loss_fn = trial.suggest_categorical("loss_fn", loss_choices)
        params["loss_fn"] = loss_fn
        
        # Huber delta (only used if loss_fn == "huber")
        if loss_fn == "huber":
            params["huber_delta"] = trial.suggest_float(
                "huber_delta", self.config.huber_delta_min, self.config.huber_delta_max,
            )
        else:
            params["huber_delta"] = 1.0
        
        # Quantile alpha (only used if loss_fn == "quantile")
        if loss_fn == "quantile":
            params["quantile_alpha"] = trial.suggest_float(
                "quantile_alpha", self.config.quantile_alpha_min, self.config.quantile_alpha_max,
            )
        else:
            params["quantile_alpha"] = 0.5
        
        # =====================================================================
        # 4.8 REGULARIZATION & NOISE INJECTION
        # =====================================================================
        params["input_noise_std"] = trial.suggest_float(
            "input_noise_std", self.config.input_noise_std_min, self.config.input_noise_std_max,
        )
        params["weight_decay"] = trial.suggest_float(
            "weight_decay", self.config.weight_decay_min, self.config.weight_decay_max,
        )
        
        # =====================================================================
        # 4.9 LEARNING RATE SCHEDULER (Extended with step, warmup_cosine, cyclical, one_cycle)
        # =====================================================================
        lr_scheduler = trial.suggest_categorical(
            "lr_scheduler", list(self.config.lr_schedulers),
        )
        params["lr_scheduler"] = lr_scheduler
        
        # Initialize all scheduler params with defaults
        params["cosine_t_max"] = 50
        params["plateau_patience"] = 5
        params["step_lr_step_size"] = 30
        params["step_lr_gamma"] = 0.1
        params["cyclical_base_lr"] = 1e-5
        params["cyclical_max_lr"] = 1e-3
        params["cyclical_step_size"] = 500
        
        if lr_scheduler == "cosine":
            params["cosine_t_max"] = trial.suggest_int(
                "cosine_t_max", self.config.cosine_t_max_min, self.config.cosine_t_max_max,
            )
        elif lr_scheduler == "plateau":
            params["plateau_patience"] = trial.suggest_int(
                "plateau_patience", self.config.plateau_patience_min, self.config.plateau_patience_max,
            )
        elif lr_scheduler == "step":
            params["step_lr_step_size"] = trial.suggest_int(
                "step_lr_step_size",
                self.config.step_lr_step_size_min,
                self.config.step_lr_step_size_max,
            )
            params["step_lr_gamma"] = self.config.step_lr_gamma
        elif lr_scheduler == "warmup_cosine":
            # Uses warmup_steps + cosine_t_max
            params["cosine_t_max"] = trial.suggest_int(
                "cosine_t_max", self.config.cosine_t_max_min, self.config.cosine_t_max_max,
            )
        elif lr_scheduler == "cyclical":
            params["cyclical_base_lr"] = trial.suggest_float(
                "cyclical_base_lr",
                self.config.cyclical_base_lr_min,
                self.config.cyclical_base_lr_max,
                log=True,
            )
            params["cyclical_max_lr"] = trial.suggest_float(
                "cyclical_max_lr",
                self.config.cyclical_max_lr_min,
                self.config.cyclical_max_lr_max,
                log=True,
            )
            params["cyclical_step_size"] = trial.suggest_int(
                "cyclical_step_size",
                self.config.cyclical_step_size_min,
                self.config.cyclical_step_size_max,
            )
        elif lr_scheduler == "one_cycle":
            # One cycle uses max_lr from lstm_lr and pct_start
            params["one_cycle_pct_start"] = self.config.one_cycle_pct_start
        # else: "none" - no scheduler params needed
        
        # =====================================================================
        # 4.10 TRAINING CONTROL (Critical for preventing overtraining)
        # =====================================================================
        # Early stopping: THE main control - stops when validation stops improving
        params["early_stopping_patience"] = trial.suggest_int(
            "early_stopping_patience",
            self.config.early_stopping_patience_min,
            self.config.early_stopping_patience_max,
        )
        
        # Max epochs: upper bound CAP, not target (most folds converge in 20-80)
        params["max_epochs"] = trial.suggest_int(
            "max_epochs",
            self.config.max_epochs_min,
            self.config.max_epochs_max,
        )
        
        # Warmup steps: prevents exploding gradients early, helps AdamW adapt
        params["warmup_steps"] = trial.suggest_int(
            "warmup_steps",
            self.config.warmup_steps_min,
            self.config.warmup_steps_max,
        )
        
        # =====================================================================
        # TRAINING STABILITY PARAMETERS (Hedge Fund Best Practices)
        # =====================================================================
        # EMA decay for model weights (stabilizes returns forecasts)
        params["ema_enabled"] = trial.suggest_categorical(
            "ema_enabled", list(self.config.ema_enabled),
        )
        if params["ema_enabled"]:
            params["ema_decay"] = trial.suggest_float(
                "ema_decay",
                self.config.ema_decay_min,
                self.config.ema_decay_max,
            )
        else:
            params["ema_decay"] = 0.999  # Default, not used
        
        # AMP precision configuration
        # bf16 for forward/backward pass, fp32 for critical numerics (loss, LayerNorm, etc.)
        params["amp_precision"] = trial.suggest_categorical(
            "amp_precision", list(self.config.amp_precision_choices),
        )
        
        # Max gradient norm for clipping
        params["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm",
            self.config.max_grad_norm_min,
            self.config.max_grad_norm_max,
        )
        
        # Training sequence length (lookback window in days)
        params["train_seq_length"] = trial.suggest_int(
            "train_seq_length",
            self.config.train_seq_length_min,
            self.config.train_seq_length_max,
        )
        
        # Target sequence length (forecast horizon)
        params["target_seq_length"] = trial.suggest_int(
            "target_seq_length",
            self.config.target_seq_length_min,
            self.config.target_seq_length_max,
        )
        
        # =====================================================================
        # REGIME-AWARE PARAMETERS (Stage C Enhanced)
        # =====================================================================
        # Volatility regime detection window (days)
        params["volatility_regime_window"] = trial.suggest_int(
            "volatility_regime_window",
            self.config.volatility_regime_window_min,
            self.config.volatility_regime_window_max,
        )
        
        # Market regime detection model
        params["market_regime_model"] = trial.suggest_categorical(
            "market_regime_model", list(self.config.market_regime_models),
        )
        
        return params
    
    def _suggest_mamba_params(self, trial: "Trial") -> Dict[str, Any]:
        """Suggest Mamba (Structured State Space Model) hyperparameters.
        
        Mamba is an alternative to LSTM/Transformers with O(n) complexity.
        Particularly effective for long-range dependencies and regime shifts in financial data.
        
        Includes:
            - A. Core Architecture (d_model, n_layers, ssm_dim, expand_factor, seq_len, activation)
            - B. Normalization & Dropout (norm_type, norm_strategy, dropout variants)
            - C. Training Dynamics (optimizer, lr, weight_decay, grad_clip)
            - D. Training Schedule (lr_scheduler, warmup, epochs, batch_size)
            - E. Loss & Output (loss_fn, head_type, MLP head config)
            - F. Thresholds (handled separately via _suggest_threshold_params)
        """
        params = {}

        # WEIGHTS-ONLY MODE: freeze all Mamba params
        if bool(getattr(self.config, "tune_weights_only", False)):
            fixed = dict(getattr(self.config, "fixed_params", {}) or {})
            def _fix(name: str, default: Any) -> Any:
                return trial.suggest_categorical(name, [fixed.get(name, default)])

            params["mamba_d_model"] = _fix("mamba_d_model", self.config.mamba_d_model_choices[0])
            params["mamba_n_layers"] = _fix("mamba_n_layers", self.config.mamba_n_layers_choices[0])
            params["mamba_ssm_dim"] = _fix("mamba_ssm_dim", self.config.mamba_ssm_dim_choices[0])
            params["mamba_expand_factor"] = _fix("mamba_expand_factor", self.config.mamba_expand_factor_choices[0])
            # If user provides a seq_len that isn't in the effective choice set, coerce to nearest allowed.
            try:
                allowed = list(self._get_effective_mamba_seq_len_choices())
            except Exception:
                allowed = list(self.config.mamba_seq_len_choices)
            _seq = int(fixed.get("mamba_seq_len", allowed[0] if allowed else 128))
            if allowed and _seq not in allowed:
                _seq = min(allowed, key=lambda v: abs(int(v) - _seq))
            params["mamba_seq_len"] = trial.suggest_categorical("mamba_seq_len", [_seq])
            params["mamba_activation"] = _fix("mamba_activation", self.config.mamba_activation_choices[0])
            params["mamba_norm_type"] = _fix("mamba_norm_type", self.config.mamba_norm_type_choices[0])
            params["mamba_norm_strategy"] = _fix("mamba_norm_strategy", self.config.mamba_norm_strategy_choices[0])
            params["mamba_dropout"] = _fix("mamba_dropout", float(self.config.mamba_dropout_min))
            params["mamba_resid_dropout"] = _fix("mamba_resid_dropout", self.config.mamba_resid_dropout_choices[0])
            params["mamba_ssm_dropout"] = _fix("mamba_ssm_dropout", self.config.mamba_ssm_dropout_choices[0])
            params["mamba_gate_dropout"] = _fix("mamba_gate_dropout", self.config.mamba_gate_dropout_choices[0])
            params["mamba_optimizer"] = _fix("mamba_optimizer", self.config.mamba_optimizer_choices[0])
            params["mamba_learning_rate"] = _fix("mamba_learning_rate", float(self.config.mamba_lr_min))
            params["mamba_weight_decay"] = _fix("mamba_weight_decay", float(self.config.mamba_weight_decay_min))
            params["mamba_grad_clip"] = _fix("mamba_grad_clip", self.config.mamba_grad_clip_choices[0])
            params["mamba_lr_scheduler"] = _fix("mamba_lr_scheduler", self.config.mamba_lr_scheduler_choices[0])
            params["mamba_warmup_steps"] = _fix("mamba_warmup_steps", self.config.mamba_warmup_steps_choices[0])
            params["mamba_max_epochs"] = _fix("mamba_max_epochs", self.config.mamba_max_epochs_choices[0])
            params["mamba_batch_size"] = _fix("mamba_batch_size", self.config.mamba_batch_size_choices[0])
            params["mamba_loss_fn"] = _fix("mamba_loss_fn", self.config.mamba_loss_fn_choices[0])
            params["mamba_head_type"] = _fix("mamba_head_type", self.config.mamba_head_type_choices[0])
            params["mamba_head_hidden_dim"] = _fix("mamba_head_hidden_dim", 128)
            params["mamba_head_num_layers"] = _fix("mamba_head_num_layers", 1)
            params["mamba_head_dropout"] = _fix("mamba_head_dropout", 0.0)
            return params
        
        # =====================================================================
        # A. CORE MAMBA ARCHITECTURE
        # =====================================================================
        params["mamba_d_model"] = trial.suggest_categorical(
            "mamba_d_model", list(self.config.mamba_d_model_choices),
        )
        params["mamba_n_layers"] = trial.suggest_categorical(
            "mamba_n_layers", list(self.config.mamba_n_layers_choices),
        )
        params["mamba_ssm_dim"] = trial.suggest_categorical(
            "mamba_ssm_dim", list(self.config.mamba_ssm_dim_choices),
        )
        params["mamba_expand_factor"] = trial.suggest_categorical(
            "mamba_expand_factor", list(self.config.mamba_expand_factor_choices),
        )
        params["mamba_seq_len"] = trial.suggest_categorical(
            "mamba_seq_len", self._get_effective_mamba_seq_len_choices(),
        )
        params["mamba_activation"] = trial.suggest_categorical(
            "mamba_activation", list(self.config.mamba_activation_choices),
        )
        
        # =====================================================================
        # B. NORMALIZATION & DROPOUT
        # =====================================================================
        params["mamba_norm_type"] = trial.suggest_categorical(
            "mamba_norm_type", list(self.config.mamba_norm_type_choices),
        )
        params["mamba_norm_strategy"] = trial.suggest_categorical(
            "mamba_norm_strategy", list(self.config.mamba_norm_strategy_choices),
        )
        params["mamba_dropout"] = trial.suggest_float(
            "mamba_dropout",
            self.config.mamba_dropout_min,
            self.config.mamba_dropout_max,
        )
        params["mamba_resid_dropout"] = trial.suggest_categorical(
            "mamba_resid_dropout", list(self.config.mamba_resid_dropout_choices),
        )
        params["mamba_ssm_dropout"] = trial.suggest_categorical(
            "mamba_ssm_dropout", list(self.config.mamba_ssm_dropout_choices),
        )
        params["mamba_gate_dropout"] = trial.suggest_categorical(
            "mamba_gate_dropout", list(self.config.mamba_gate_dropout_choices),
        )
        
        # =====================================================================
        # C. TRAINING DYNAMICS
        # =====================================================================
        params["mamba_optimizer"] = trial.suggest_categorical(
            "mamba_optimizer", list(self.config.mamba_optimizer_choices),
        )
        params["mamba_learning_rate"] = trial.suggest_float(
            "mamba_learning_rate",
            self.config.mamba_lr_min,
            self.config.mamba_lr_max,
            log=True,
        )
        params["mamba_weight_decay"] = trial.suggest_float(
            "mamba_weight_decay",
            self.config.mamba_weight_decay_min,
            self.config.mamba_weight_decay_max,
            log=True,
        )
        params["mamba_grad_clip"] = trial.suggest_categorical(
            "mamba_grad_clip", list(self.config.mamba_grad_clip_choices),
        )
        
        # =====================================================================
        # D. TRAINING SCHEDULE
        # =====================================================================
        params["mamba_lr_scheduler"] = trial.suggest_categorical(
            "mamba_lr_scheduler", list(self.config.mamba_lr_scheduler_choices),
        )
        params["mamba_warmup_steps"] = trial.suggest_categorical(
            "mamba_warmup_steps", list(self.config.mamba_warmup_steps_choices),
        )
        params["mamba_max_epochs"] = trial.suggest_categorical(
            "mamba_max_epochs", list(self.config.mamba_max_epochs_choices),
        )
        params["mamba_batch_size"] = trial.suggest_categorical(
            "mamba_batch_size", list(self.config.mamba_batch_size_choices),
        )
        
        # =====================================================================
        # E. LOSS & OUTPUT STRUCTURE
        # =====================================================================
        params["mamba_loss_fn"] = trial.suggest_categorical(
            "mamba_loss_fn", list(self.config.mamba_loss_fn_choices),
        )
        params["mamba_head_type"] = trial.suggest_categorical(
            "mamba_head_type", list(self.config.mamba_head_type_choices),
        )
        
        # MLP head configuration (only used when head_type="mlp")
        if params["mamba_head_type"] == "mlp":
            params["mamba_head_hidden_dim"] = trial.suggest_int(
                "mamba_head_hidden_dim",
                self.config.mamba_head_hidden_dim_min,
                self.config.mamba_head_hidden_dim_max,
            )
            params["mamba_head_num_layers"] = trial.suggest_categorical(
                "mamba_head_num_layers", list(self.config.mamba_head_num_layers_choices),
            )
            params["mamba_head_dropout"] = trial.suggest_float(
                "mamba_head_dropout",
                self.config.mamba_head_dropout_min,
                self.config.mamba_head_dropout_max,
            )
        else:
            # Defaults for linear head
            params["mamba_head_hidden_dim"] = 128
            params["mamba_head_num_layers"] = 1
            params["mamba_head_dropout"] = 0.0
        
        # F. Thresholds (handled separately via _suggest_threshold_params)
        
        return params
    
    def _suggest_threshold_params(self, trial: "Trial") -> Dict[str, float]:
        """Suggest Stage-C threshold and regime adjustment parameters.
        
        STEP 7: UPGRADED Regime-Aware Threshold Adjustment
        
        Hyperparameters (learned):
            T_base, bull_long_mult, bull_short_mult,
            bear_long_mult, bear_short_mult,
            crisis_long_mult, crisis_short_mult,
            conf_threshold, vol_scaler
        
        Regime detection:
            regime = TrackC_regime.argmax()  # BULL, BEAR, CRISIS, FLAT
        
        Volatility scaling:
            vol_ratio = (vol20 / vol252)**vol_scaler
            pred_norm = pred / vol_ratio
        
        Thresholds:
            if BULL:
                T_long = T_base * bull_long_mult
                T_short = T_base * bull_short_mult
            elif BEAR:
                T_long = T_base * bear_long_mult
                T_short = T_base * bear_short_mult
            elif CRISIS:
                T_long = T_base * crisis_long_mult
                T_short = T_base * crisis_short_mult
            else:  # FLAT
                T_long = T_base
                T_short = T_base
        
        Confidence gating:
            if pred_conf < conf_threshold: signal = 0
        
        Direction output:
            if pred_norm > T_long:   LONG
            elif pred_norm < -T_short: SHORT
            else:                     NEUTRAL
        
        Returns:
            Dict with all threshold/regime hyperparameters
        """
        if bool(getattr(self.config, "tune_weights_only", False)):
            fixed = dict(getattr(self.config, "fixed_params", {}) or {})
            def _fix(name: str, default: Any) -> Any:
                return trial.suggest_categorical(name, [fixed.get(name, default)])
            return {
                "threshold": float(_fix("threshold", 0.10)),
                "bull_long_mult": float(_fix("bull_long_mult", 1.0)),
                "bull_short_mult": float(_fix("bull_short_mult", 1.0)),
                "bear_long_mult": float(_fix("bear_long_mult", 1.5)),
                "bear_short_mult": float(_fix("bear_short_mult", 0.5)),
                "crisis_long_mult": float(_fix("crisis_long_mult", 3.0)),
                "crisis_short_mult": float(_fix("crisis_short_mult", 3.0)),
                "conf_threshold": float(_fix("conf_threshold", 0.5)),
                "vol_scaler": float(_fix("vol_scaler", 1.0)),
            }

        # Base threshold
        threshold = trial.suggest_float(
            "threshold",
            self.config.threshold_min,
            self.config.threshold_max,
        )
        
        # Bull regime - separate long/short
        bull_long_mult = trial.suggest_float(
            "bull_long_mult",
            self.config.bull_long_mult_min,
            self.config.bull_long_mult_max,
        )
        bull_short_mult = trial.suggest_float(
            "bull_short_mult",
            self.config.bull_short_mult_min,
            self.config.bull_short_mult_max,
        )
        
        # Bear regime - separate long/short
        bear_long_mult = trial.suggest_float(
            "bear_long_mult",
            self.config.bear_long_mult_min,
            self.config.bear_long_mult_max,
        )
        bear_short_mult = trial.suggest_float(
            "bear_short_mult",
            self.config.bear_short_mult_min,
            self.config.bear_short_mult_max,
        )
        
        # Crisis regime - separate long/short
        crisis_long_mult = trial.suggest_float(
            "crisis_long_mult",
            self.config.crisis_long_mult_min,
            self.config.crisis_long_mult_max,
        )
        crisis_short_mult = trial.suggest_float(
            "crisis_short_mult",
            self.config.crisis_short_mult_min,
            self.config.crisis_short_mult_max,
        )
        
        # Confidence gating
        conf_threshold = trial.suggest_float(
            "conf_threshold",
            self.config.conf_threshold_min,
            self.config.conf_threshold_max,
        )
        
        # Volatility scaler
        vol_scaler = trial.suggest_float(
            "vol_scaler",
            self.config.vol_scaler_min,
            self.config.vol_scaler_max,
        )
        
        return {
            "threshold": threshold,
            "bull_long_mult": bull_long_mult,
            "bull_short_mult": bull_short_mult,
            "bear_long_mult": bear_long_mult,
            "bear_short_mult": bear_short_mult,
            "crisis_long_mult": crisis_long_mult,
            "crisis_short_mult": crisis_short_mult,
            "conf_threshold": conf_threshold,
            "vol_scaler": vol_scaler,
        }
    
    def _suggest_pipeline_params(self, trial: "Trial") -> Dict[str, Any]:
        """Suggest pipeline-level hyperparameters.
        
        These control feature preprocessing and data splitting, not the LSTM model.
        
        Hyperparameters:
            smoothing_type: Type of smoothing to apply to features
                - "none": No smoothing
                - "ema": Exponential moving average
                - "sma": Simple moving average
                - "gaussian": Gaussian smoothing
            smoothing_window: Window size for smoothing (if not "none")
            train_fraction: Fraction of data to use for training vs validation
        
        Returns:
            Dict with pipeline parameters
        """
        if bool(getattr(self.config, "tune_weights_only", False)):
            fixed = dict(getattr(self.config, "fixed_params", {}) or {})
            def _fix(name: str, default: Any) -> Any:
                return trial.suggest_categorical(name, [fixed.get(name, default)])
            return {
                "smoothing_type": str(_fix("smoothing_type", "none")),
                "smoothing_window": int(_fix("smoothing_window", 3)),
                "train_fraction": float(_fix("train_fraction", 0.8)),
            }

        # Feature smoothing
        smoothing_type = trial.suggest_categorical(
            "smoothing_type",
            self.config.smoothing_types,
        )
        
        # Smoothing window - only matters if smoothing_type != "none"
        smoothing_window = trial.suggest_int(
            "smoothing_window",
            self.config.smoothing_window_min,
            self.config.smoothing_window_max,
            step=3,  # Use step of 3: 3, 6, 9, 12, 15, 18, 21
        )
        
        # Train/validation split fraction
        train_fraction = trial.suggest_float(
            "train_fraction",
            self.config.train_fraction_min,
            self.config.train_fraction_max,
        )
        
        return {
            "smoothing_type": smoothing_type,
            "smoothing_window": smoothing_window,
            "train_fraction": train_fraction,
        }
    
    def _detect_regime(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Detect market regime from returns.
        
        Regime is an INPUT FEATURE (not optimization target).
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
        bull_long_mult: float = 0.8,
        bull_short_mult: float = 1.0,
        bear_long_mult: float = 1.5,
        bear_short_mult: float = 0.8,
        crisis_long_mult: float = 2.5,
        crisis_short_mult: float = 2.0,
        conf_threshold: float = 0.5,
        vol_scaler: float = 0.5,
        pred_confidence: Optional[np.ndarray] = None,
        vol20: Optional[np.ndarray] = None,
        vol252: Optional[np.ndarray] = None,
        # Legacy support
        bull_mult: Optional[float] = None,
        bear_mult: Optional[float] = None,
        crisis_mult: Optional[float] = None,
    ) -> np.ndarray:
        """Apply UPGRADED regime-aware thresholds to get directional signals.
        
        STEP 7 UPGRADED Final Direction Logic:
        
        1. Volatility scaling:
           vol_ratio = (vol20 / vol252)**vol_scaler
           pred_norm = pred / vol_ratio
        
        2. Regime-specific asymmetric thresholds:
           if BULL:   T_long = T * bull_long_mult,   T_short = T * bull_short_mult
           if BEAR:   T_long = T * bear_long_mult,   T_short = T * bear_short_mult
           if CRISIS: T_long = T * crisis_long_mult, T_short = T * crisis_short_mult
           else FLAT: T_long = T, T_short = T
        
        3. Confidence gating:
           if pred_conf < conf_threshold: signal = 0
        
        4. Direction:
           if pred_norm > T_long:   +1 (LONG)
           elif pred_norm < -T_short: -1 (SHORT)
           else:                     0 (NEUTRAL)
        
        Args:
            predictions: Raw model predictions
            regime: Regime labels (0=bull, 1=bear, 2=crisis, 3=flat)
            threshold: Base threshold T
            *_long_mult, *_short_mult: Asymmetric regime multipliers
            conf_threshold: Confidence gate
            vol_scaler: Volatility normalization power
            pred_confidence: Optional confidence scores
            vol20, vol252: Optional volatility arrays
            
        Returns:
            Directional signals: -1, 0, or +1
        """
        # Legacy support: convert old single multipliers to new format
        if bull_mult is not None:
            bull_long_mult = bull_mult
            bull_short_mult = bull_mult
        if bear_mult is not None:
            bear_long_mult = bear_mult
            bear_short_mult = bear_mult
        if crisis_mult is not None:
            crisis_long_mult = crisis_mult
            crisis_short_mult = crisis_mult
        
        # Step 1: Volatility scaling
        if vol20 is not None and vol252 is not None and vol_scaler > 0:
            # Avoid division by zero
            vol252_safe = np.maximum(vol252, 1e-8)
            vol_ratio = np.power(vol20 / vol252_safe, vol_scaler)
            vol_ratio = np.maximum(vol_ratio, 0.1)  # Floor to prevent explosion
            pred_norm = predictions / vol_ratio
        else:
            pred_norm = predictions
        
        # Step 2: Compute regime-specific asymmetric thresholds
        # Long thresholds
        T_long = np.where(
            regime == 0, threshold * bull_long_mult,
            np.where(
                regime == 1, threshold * bear_long_mult,
                np.where(
                    regime == 2, threshold * crisis_long_mult,
                    threshold  # FLAT (regime == 3)
                )
            )
        )
        
        # Short thresholds
        T_short = np.where(
            regime == 0, threshold * bull_short_mult,
            np.where(
                regime == 1, threshold * bear_short_mult,
                np.where(
                    regime == 2, threshold * crisis_short_mult,
                    threshold  # FLAT
                )
            )
        )
        
        # Step 3 & 4: Apply threshold logic with asymmetric thresholds
        signals = np.where(
            pred_norm > T_long, 1.0,
            np.where(pred_norm < -T_short, -1.0, 0.0)
        )
        
        # Step 3: Confidence gating (if provided)
        if pred_confidence is not None:
            signals = np.where(pred_confidence < conf_threshold, 0.0, signals)
        
        return signals
    
    def _build_track_a(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        family_params: Dict[str, Tuple[bool, int, str, float]],
        train_idx: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, FamilyEncoder]]:
        """Build Track A with cached encoders and family weights.
        
        SOFT DROP ARCHITECTURE:
        =======================
        - All included families get encoded (shapes constant)
        - Weights applied AFTER encoding (soft gating)
        - Low weight (~0.01) = soft drop (variance contribution ~0)
        - Normalization (softmax) redistributes weight to strong families
        
        This ensures:
        - Pipeline stability (no shape changes across trials)
        - Gradient-friendly optimization
        - Automatic feature selection via weight decay
        """
        import sys
        import time as _time
        track_a_frames: List[pd.DataFrame] = []
        fitted_encoders: Dict[str, FamilyEncoder] = {}
        
        _build_start = _time.time()
        _family_count = 0
        
        # First pass: collect raw weights for normalization (only families with data)
        raw_weights: Dict[str, float] = {}
        for family in STAGE_A_FAMILIES:
            params = family_params.get(family, (False, 0, "none", 0.0, {}))
            include, latent_dim, method, weight = params[0], params[1], params[2], params[3]
            # Include if: has columns, has dims, and weight > 0
            cols = self.family_columns.get(family, [])
            if include and latent_dim > 0 and weight > 0 and cols:
                raw_weights[family] = weight
        
        # Apply weight normalization if configured
        # Softmax with temperature controls exploration/exploitation
        normalized_weights = self._normalize_family_weights(
            raw_weights,
            method=self.config.weight_normalization,
            temperature=self.config.weight_temperature,
        )
        
        # Second pass: encode ALL included families (soft drop = encode but low weight)
        expected_encoded_cols: List[str] = []
        for family in STAGE_A_FAMILIES:
            params = family_params.get(family, (False, 0, "none", 0.0, {}))
            include, latent_dim, method, weight, ae_params = (
                params[0], params[1], params[2], params[3], params[4] if len(params) > 4 else {}
            )
            
            # Skip only if truly excluded (no data or zero dims)
            if not include or latent_dim == 0:
                continue
            
            # Get family columns
            cols = self.family_columns.get(family, [])
            if not cols:
                continue

            # Always reserve the encoded column names for this family.
            # Even if the symbol has missing inputs (all zeros) and sanitization drops
            # constant columns, Phase-2 requires stable shapes across symbols.
            expected_encoded_cols.extend([f"{family}_enc_{i}" for i in range(int(latent_dim))])
            
            _family_count += 1
            _t_family = _time.time()
            
            # Extract family data
            missing_cols = [c for c in cols if c not in panel.columns]
            if missing_cols:
                self.logger.warning(
                    "Track-A family '%s' missing %d/%d expected columns; filling with zeros. Examples: %s",
                    family,
                    len(missing_cols),
                    len(cols),
                    ", ".join(missing_cols[:10]),
                )
            family_frame = panel.reindex(columns=cols, fill_value=0.0)
            # Convert pd.NA to np.nan for pandas 2.x compatibility
            family_frame = family_frame.fillna(0.0)
            family_data = family_frame.values.astype(np.float32)
            
            # Handle NaN
            family_data = np.nan_to_num(family_data, nan=0.0)
            
            # Cache key: include ae_params hash for AE uniqueness
            ae_key = tuple(sorted(ae_params.items())) if ae_params else ()
            cache_key = (family, latent_dim, method, ae_key)
            
            # Check encoder cache - reuse if same config already fitted
            if cache_key in self._encoder_cache:
                encoder = self._encoder_cache[cache_key]
                self.logger.debug(f"Reusing cached encoder for {family} (dim={latent_dim}, method={method})")
            else:
                # Create and fit new encoder with Optuna-tuned AE params
                _t_fit = _time.time()
                encoder = FamilyEncoder(
                    family_name=family,
                    n_components=latent_dim,
                    method=method,
                    **ae_params,  # Pass ae_layers, ae_activation, ae_dropout, ae_lr if AE
                )
                train_data = family_data[train_idx]
                ae_info = f", ae={ae_params}" if ae_params else ""
                print(f"[TRACE:track_a] FIT {family} ({method}, {len(cols)} cols -> {latent_dim} dims, {train_data.shape[0]} rows{ae_info}) starting...", file=sys.stderr, flush=True)
                encoder.fit(train_data, epochs=self.config.ae_epochs)
                _fit_time = _time.time() - _t_fit
                print(f"[TRACE:track_a] FIT {family} done in {_fit_time:.1f}s", file=sys.stderr, flush=True)
                
                # Store in cache for future trials
                self._encoder_cache[cache_key] = encoder
                self.logger.debug(f"Fitted and cached encoder for {family} (dim={latent_dim}, method={method})")
            
            # Transform all data using (possibly cached) encoder
            encoded = encoder.transform(family_data)
            
            # Apply NORMALIZED family weight to encoded features
            # Use normalized weight if available, otherwise fall back to raw weight
            final_weight = normalized_weights.get(family, weight)
            encoded = encoded * final_weight
            
            # Create DataFrame with encoded columns
            encoded_cols = [f"{family}_enc_{i}" for i in range(encoded.shape[1])]
            encoded_df = pd.DataFrame(encoded, index=panel.index, columns=encoded_cols)
            
            track_a_frames.append(encoded_df)
            fitted_encoders[family] = encoder
        
        if not track_a_frames:
            return pd.DataFrame(index=panel.index), fitted_encoders

        track_a = pd.concat(track_a_frames, axis=1)
        track_a = self._sanitize_features(track_a)

        # Enforce stable Track-A schema across symbols:
        # If a symbol has an entire family missing, the encoded outputs can be constant
        # (all zeros) and `_sanitize_features` may drop them. Re-add them as zeros so
        # downstream Track-C union + post-std weighting can be consistent.
        if expected_encoded_cols:
            track_a = track_a.reindex(columns=expected_encoded_cols, fill_value=0.0)

        total_weight = sum(normalized_weights.values()) if normalized_weights else 0.0
        print(f"[TRACE:track_a] TOTAL: {_family_count} families, total_weight={total_weight:.2f} (norm={self.config.weight_normalization}), in {_time.time()-_build_start:.1f}s", file=sys.stderr, flush=True)
        return track_a, fitted_encoders
    
    def _build_track_b(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        block_summaries: Dict[str, pd.DataFrame],
        track_b_family_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Build Track B: HF blocks + Model-family summaries.
        
        Track B is always included. Each Track-B family is weightable by Optuna,
        but is treated as a *separate category* from Stage-A families:
        - HF blocks: raw passthrough columns
        - Summary blocks: raw passthrough frames
        
        No compression is applied to Track-B families.
        """
        track_b_frames: List[pd.DataFrame] = []

        weights_in = track_b_family_weights or {}

        # First pass: collect raw weights for normalization (only families with data)
        raw_weights: Dict[str, float] = {}
        for family in HF_BLOCK_FAMILIES:
            cols = self.family_columns.get(family, [])
            if cols:
                w = float(weights_in.get(family, 1.0))
                raw_weights[family] = max(w, float(self.config.family_weight_clip_min))

        for summary_name in TRACK_B_SUMMARY_BLOCKS:
            summary = block_summaries.get(summary_name)
            if summary is not None and not summary.empty:
                w = float(weights_in.get(summary_name, 1.0))
                raw_weights[summary_name] = max(w, float(self.config.family_weight_clip_min))

        # Normalize within Track B only (separate category from Stage A).
        normalized_weights = self._normalize_family_weights(
            raw_weights,
            method=self.config.weight_normalization,
            temperature=self.config.weight_temperature,
        )
        
        # HF block columns
        for family in HF_BLOCK_FAMILIES:
            cols = self.family_columns.get(family, [])
            if cols:
                w = float(normalized_weights.get(family, weights_in.get(family, 1.0)))
                track_b_frames.append(panel[cols] * w)
        
        # Model-family summaries from block_summaries
        for summary_name in TRACK_B_SUMMARY_BLOCKS:
            summary = block_summaries.get(summary_name)
            if summary is not None and not summary.empty:
                w = float(normalized_weights.get(summary_name, weights_in.get(summary_name, 1.0)))
                track_b_frames.append(summary * w)
        
        if not track_b_frames:
            return pd.DataFrame(index=panel.index)

        track_b = pd.concat(track_b_frames, axis=1)
        # Remove duplicate columns
        track_b = track_b.loc[:, ~track_b.columns.duplicated()]
        # Keep constant columns to preserve a stable Track-B schema across symbols.
        # (Constant columns are harmless after standardization but dropping them
        # can create cross-symbol schema mismatches.)
        track_b = self._sanitize_features(track_b, drop_constant_cols=False)
        return track_b
    
    def _build_track_c(
        self,
        track_a: pd.DataFrame,
        track_b: pd.DataFrame,
        weight_a: float,
        weight_b: float,
    ) -> pd.DataFrame:
        """Build Track C: Weighted concatenation of Track A and Track B."""
        # Apply weights
        track_a_weighted = track_a * weight_a if not track_a.empty else track_a
        track_b_weighted = track_b * weight_b if not track_b.empty else track_b
        
        # Concatenate
        if track_a_weighted.empty:
            return track_b_weighted
        if track_b_weighted.empty:
            return track_a_weighted
        
        track_c = pd.concat([track_a_weighted, track_b_weighted], axis=1)
        # Track-C must keep constant columns to maintain identical feature schemas
        # across symbols (Phase-2 pooled training assumption). Missing-family
        # encodings become all-zeros per symbol by design.
        return self._sanitize_features(track_c, drop_constant_cols=False)

    def _sanitize_features(self, frame: pd.DataFrame, *, drop_constant_cols: bool = True) -> pd.DataFrame:
        """Fill missing values and optionally drop constant columns for stability."""
        if frame is None or frame.empty:
            return frame
        sanitized = frame.copy()
        sanitized = sanitized.replace([np.inf, -np.inf], np.nan)
        sanitized = sanitized.ffill().bfill().fillna(0.0)
        sanitized = sanitized.loc[:, ~sanitized.columns.duplicated()]
        if not drop_constant_cols:
            return sanitized
        nunique = sanitized.nunique(dropna=False)
        keep_cols = nunique[nunique > 1].index
        if len(keep_cols) == 0:
            return sanitized
        if len(keep_cols) != len(sanitized.columns):
            self.logger.debug(
                "_sanitize_features dropped %d constant columns",
                len(sanitized.columns) - len(keep_cols),
            )
        return sanitized[keep_cols]
    
    def _compute_objective(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        horizon: int = 1,
        threshold_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute optimization objective with regime-aware thresholds.
        
        STEP 7: Stage-C Threshold & Regime Adjustment
        
        For each fold:
        1. Get raw predictions
        2. Detect regime from actual returns
        3. Apply threshold T_regime based on regime
        4. Compute directional signals
        5. Compute strategy returns
        6. Compute Sharpe, RWA, stability
        
        Objective: 0.65*sharpe + 0.20*rwa + 0.15*stability
        
        🔧 CRITICAL FIX: For horizons > 1, use non-overlapping samples for Sharpe
        to avoid autocorrelation-induced inflation.
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return float("-inf")
        
        # Use regime-aware thresholds if provided
        if threshold_params is not None:
            # Step 1-2: Detect regime from actual returns (regime is INPUT)
            regime = self._detect_regime(actuals)
            
            # Step 3-4: Apply regime-specific thresholds to get directional signals
            direction = self._apply_regime_thresholds(
                predictions,
                regime,
                float(threshold_params.get("threshold", 0.0)),
                bull_long_mult=float(threshold_params.get("bull_long_mult", 0.8)),
                bull_short_mult=float(threshold_params.get("bull_short_mult", 1.0)),
                bear_long_mult=float(threshold_params.get("bear_long_mult", 1.5)),
                bear_short_mult=float(threshold_params.get("bear_short_mult", 0.8)),
                crisis_long_mult=float(threshold_params.get("crisis_long_mult", 2.5)),
                crisis_short_mult=float(threshold_params.get("crisis_short_mult", 2.0)),
                conf_threshold=float(threshold_params.get("conf_threshold", 0.5)),
                vol_scaler=float(threshold_params.get("vol_scaler", 0.5)),
                # Legacy support
                bull_mult=threshold_params.get("bull_mult"),
                bear_mult=threshold_params.get("bear_mult"),
                crisis_mult=threshold_params.get("crisis_mult"),
            )
        else:
            # Fallback: simple sign-based direction (no threshold)
            direction = np.sign(predictions)
        
        # Step 5: Compute strategy returns
        strategy_returns = direction * actuals
        
        # Step 6: Compute metrics
        # 🔧 FIX: For horizons > 1, sample only non-overlapping returns to avoid
        # autocorrelation-induced Sharpe inflation (~8x for H63)
        non_neutral_mask = direction != 0
        if non_neutral_mask.any():
            active_returns = strategy_returns[non_neutral_mask]
            
            # 🔧 CRITICAL: Sample non-overlapping returns for Sharpe calculation
            # For H63, with ~109 samples per fold, slicing [::63] gives ~2 samples.
            # With so few samples, Sharpe estimates are very noisy - cap to reasonable range.
            MAX_SHARPE = 5.0  # Cap annualized Sharpe to prevent outliers from noisy estimates
            
            if horizon > 1 and len(active_returns) >= horizon:
                # Sample every H-th return to get independent observations
                non_overlapping_returns = active_returns[::horizon]
                if len(non_overlapping_returns) >= 2 and np.std(non_overlapping_returns) > 1e-9:
                    periods_per_year = max(1, 252 // horizon)
                    scale = np.sqrt(periods_per_year)
                    raw_sharpe = np.mean(non_overlapping_returns) / (np.std(non_overlapping_returns) + 1e-9) * scale
                    sharpe = np.clip(raw_sharpe, -MAX_SHARPE, MAX_SHARPE)
                else:
                    sharpe = 0.0
            else:
                # Daily horizon or insufficient samples: use all
                if np.std(active_returns) > 1e-9:
                    scale = np.sqrt(max(1, 252 // max(1, horizon)))
                    raw_sharpe = np.mean(active_returns) / (np.std(active_returns) + 1e-9) * scale
                    sharpe = np.clip(raw_sharpe, -MAX_SHARPE, MAX_SHARPE)
                else:
                    sharpe = 0.0
            
            # RWA: accuracy only on trades taken (can use all samples for this)
            correct = (direction[non_neutral_mask] == np.sign(actuals[non_neutral_mask])).astype(float)
            # 🔧 FIX: Also use non-overlapping for RWA to avoid inflated accuracy
            if horizon > 1 and len(correct) >= horizon:
                non_overlapping_correct = correct[::horizon]
                rwa = np.mean(non_overlapping_correct) if len(non_overlapping_correct) > 0 else 0.5
            else:
                rwa = np.mean(correct)
        else:
            # No trades taken = 0 performance
            sharpe = 0.0
            rwa = 0.5  # Neutral baseline
        
        # Stability (inverse of prediction variance relative to mean)
        if np.abs(np.mean(predictions)) > 1e-9:
            stability = 1.0 - np.clip(np.std(predictions) / (np.abs(np.mean(predictions)) + 1e-6), 0.0, 1.0)
        else:
            stability = 0.0
        
        # Coverage penalty: Enforce minimum coverage per fold (>= 25%)
        # Penalize thresholds that are too aggressive and miss opportunities
        coverage = non_neutral_mask.mean() if threshold_params is not None else 1.0
        min_coverage = 0.25  # Minimum 25% of samples should have non-neutral signals
        coverage_penalty = 0.0
        if coverage < min_coverage:
            # Progressive penalty: stronger as coverage drops below minimum
            coverage_penalty = 0.2 * (min_coverage - coverage)  # 0.2 weight for coverage
        
        # Combined objective
        objective = (
            self.config.sharpe_weight * sharpe +
            self.config.rwa_weight * rwa +
            self.config.stability_weight * stability -
            coverage_penalty
        )
        
        return float(objective)

    def _extract_fold_raw_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        horizon: int = 1,
        threshold_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Extract raw metrics from a fold for pooled aggregation.
        
        Instead of computing a single noisy per-fold Sharpe from ~2 samples,
        this extracts the raw non-overlapping strategy returns so they can be
        pooled across all folds for a statistically robust aggregate Sharpe.
        
        Returns:
            Dict with:
                - strategy_returns: Non-overlapping strategy returns for this fold
                - directions: Corresponding trade directions
                - actuals: Corresponding actual returns
                - predictions: Raw predictions (for stability calculation)
                - mean_pred: Mean prediction (for sign-stability)
                - val_loss: Validation loss if available
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return {"valid": False}

        preds = np.asarray(predictions, dtype=float)
        acts = np.asarray(actuals, dtype=float)
        finite_mask = np.isfinite(preds) & np.isfinite(acts)
        preds = preds[finite_mask]
        acts = acts[finite_mask]
        if preds.size == 0 or acts.size == 0:
            return {"valid": False}
        
        # Apply regime-aware thresholds if provided
        if threshold_params is not None:
            regime = self._detect_regime(acts)
            direction = self._apply_regime_thresholds(
                preds,
                regime,
                float(threshold_params.get("threshold", 0.0)),
                bull_long_mult=float(threshold_params.get("bull_long_mult", 0.8)),
                bull_short_mult=float(threshold_params.get("bull_short_mult", 1.0)),
                bear_long_mult=float(threshold_params.get("bear_long_mult", 1.5)),
                bear_short_mult=float(threshold_params.get("bear_short_mult", 0.8)),
                crisis_long_mult=float(threshold_params.get("crisis_long_mult", 2.5)),
                crisis_short_mult=float(threshold_params.get("crisis_short_mult", 2.0)),
                conf_threshold=float(threshold_params.get("conf_threshold", 0.5)),
                vol_scaler=float(threshold_params.get("vol_scaler", 0.5)),
                # Legacy support
                bull_mult=threshold_params.get("bull_mult"),
                bear_mult=threshold_params.get("bear_mult"),
                crisis_mult=threshold_params.get("crisis_mult"),
            )
        else:
            direction = np.sign(preds)
        
        # Compute strategy returns
        strategy_returns = direction * acts
        
        # Filter to non-neutral positions only
        non_neutral_mask = direction != 0
        if not non_neutral_mask.any():
            # No trades taken in this fold.
            # Treat as a valid fold with neutral performance so *every* fold contributes.
            # This keeps trial comparability across hyperparameters.
            return {
                "valid": True,
                "strategy_returns": [0.0],
                "directions": [0.0],
                "actuals": [0.0],
                "mean_pred": float(np.mean(preds)) if preds.size else 0.0,
                "n_samples": 1,
                "coverage": 0.0,
                "no_trades": True,
            }
        
        active_strategy_returns = strategy_returns[non_neutral_mask]
        active_directions = direction[non_neutral_mask]
        active_actuals = actuals[non_neutral_mask]
        
        # 🔧 CRITICAL: Sample non-overlapping returns for pooled Sharpe
        # For H63 with ~109 samples, [::63] gives ~2 truly independent samples
        if horizon > 1 and len(active_strategy_returns) >= horizon:
            non_overlap_returns = active_strategy_returns[::horizon].tolist()
            non_overlap_directions = active_directions[::horizon].tolist()
            non_overlap_actuals = active_actuals[::horizon].tolist()
        else:
            # Daily horizon: all samples are independent
            non_overlap_returns = active_strategy_returns.tolist()
            non_overlap_directions = active_directions.tolist()
            non_overlap_actuals = active_actuals.tolist()
        
        return {
            "valid": True,
            "strategy_returns": non_overlap_returns,
            "directions": non_overlap_directions,
            "actuals": non_overlap_actuals,
            "mean_pred": float(np.mean(preds)) if preds.size else 0.0,
            "n_samples": len(non_overlap_returns),
            "coverage": float(non_neutral_mask.mean()),
        }

    def _compute_strategy_returns_series(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamps: Sequence[Any],
        *,
        threshold_params: Optional[Dict[str, float]],
    ) -> "pd.Series":
        """Compute per-timestamp strategy returns (including zeros for neutral positions)."""
        preds = np.asarray(predictions, dtype=float)
        acts = np.asarray(actuals, dtype=float)
        n = int(min(len(preds), len(acts), len(timestamps)))
        if n <= 0:
            return pd.Series(dtype=float)

        preds = preds[:n]
        acts = acts[:n]
        ts = pd.to_datetime(np.asarray(list(timestamps))[:n])

        finite_mask = np.isfinite(preds) & np.isfinite(acts)
        if not finite_mask.any():
            return pd.Series(dtype=float)

        preds_f = preds.copy()
        acts_f = acts.copy()
        preds_f[~finite_mask] = 0.0
        acts_f[~finite_mask] = 0.0

        if threshold_params is not None:
            regime = self._detect_regime(acts_f)
            direction = self._apply_regime_thresholds(
                preds_f,
                regime,
                float(threshold_params.get("threshold", 0.0)),
                bull_long_mult=float(threshold_params.get("bull_long_mult", 0.8)),
                bull_short_mult=float(threshold_params.get("bull_short_mult", 1.0)),
                bear_long_mult=float(threshold_params.get("bear_long_mult", 1.5)),
                bear_short_mult=float(threshold_params.get("bear_short_mult", 0.8)),
                crisis_long_mult=float(threshold_params.get("crisis_long_mult", 2.5)),
                crisis_short_mult=float(threshold_params.get("crisis_short_mult", 2.0)),
                conf_threshold=float(threshold_params.get("conf_threshold", 0.5)),
                vol_scaler=float(threshold_params.get("vol_scaler", 0.5)),
                # Legacy support
                bull_mult=threshold_params.get("bull_mult"),
                bear_mult=threshold_params.get("bear_mult"),
                crisis_mult=threshold_params.get("crisis_mult"),
            )
        else:
            direction = np.sign(preds_f)

        strat = direction * acts_f
        return pd.Series(strat.astype(float), index=pd.DatetimeIndex(ts)).sort_index()

    @staticmethod
    def _compute_pooled_scaler_stats(
        feature_frames: Sequence[pd.DataFrame],
        target_series: Sequence[pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pooled (mean, std) for z-score normalization across symbols."""
        if not feature_frames:
            raise ValueError("No feature frames provided")

        # IMPORTANT: Match `build_sequence_data` semantics.
        # - Only numeric/bool feature columns
        # - Do NOT drop rows due to NaNs across wide feature matrices
        # - Replace non-finite feature values with 0.0
        # IMPORTANT: In multi-symbol pooled mode, schemas can drift (some symbols missing
        # a subset of engineered/encoded columns). We must never KeyError on missing
        # columns; missing cols are treated as deterministic zeros.
        base_numeric_cols = list(feature_frames[0].select_dtypes(include=["number", "bool"]).columns)
        cols = base_numeric_cols if base_numeric_cols else list(feature_frames[0].columns)
        count = 0
        sum_x = None
        sum_x2 = None

        for feats, tgt in zip(feature_frames, target_series):
            # Keep numeric/bool columns only; cast to float for stable sums.
            # Use reindex to avoid KeyError if a symbol/frame is missing some columns.
            feats_aligned = feats.reindex(columns=cols)
            feat_df = feats_aligned.select_dtypes(include=["number", "bool"]).astype(float)
            if feat_df.empty:
                continue
            feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            aligned = feat_df.join(tgt.rename("target"), how="inner")
            if aligned.empty:
                continue
            aligned = aligned.dropna(subset=["target"])
            if aligned.empty:
                continue

            values = aligned[feat_df.columns].to_numpy(dtype=np.float64)
            # Safety: ensure finite before accumulation.
            values = np.where(np.isfinite(values), values, 0.0)
            if values.size == 0:
                continue
            if sum_x is None:
                sum_x = np.zeros(values.shape[1], dtype=np.float64)
                sum_x2 = np.zeros(values.shape[1], dtype=np.float64)
            sum_x += np.sum(values, axis=0)
            sum_x2 += np.sum(values * values, axis=0)
            count += int(values.shape[0])

        if count <= 0 or sum_x is None or sum_x2 is None:
            # Safe fallback: no scaling
            mean = np.zeros(len(cols), dtype=np.float32)
            std = np.ones(len(cols), dtype=np.float32)
            return mean, std

        mean = (sum_x / float(count)).astype(np.float32)
        var = (sum_x2 / float(count)) - (mean.astype(np.float64) ** 2)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return mean, std


    @staticmethod
    def _concat_sequence_data(
        seq_datas: Sequence["SequenceData"],
    ) -> "SequenceData":
        if not seq_datas:
            raise ValueError("No SequenceData to concatenate")
        seq_len = int(seq_datas[0].sequences.shape[1])
        feat_dim = int(seq_datas[0].sequences.shape[2])
        for sd in seq_datas[1:]:
            if int(sd.sequences.shape[1]) != seq_len or int(sd.sequences.shape[2]) != feat_dim:
                raise ValueError("SequenceData shapes mismatch for concat")
        sequences = np.concatenate([sd.sequences for sd in seq_datas], axis=0)
        targets = np.concatenate([sd.targets for sd in seq_datas], axis=0)
        timestamps = np.concatenate([sd.timestamps for sd in seq_datas], axis=0)
        from src.stage_b.sequence_models import SequenceData as _SequenceData

        return _SequenceData(sequences=sequences, targets=targets, timestamps=timestamps)

    def _evaluate_trial_params_multi_symbol(
        self,
        *,
        panels_by_symbol: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
        labels_by_symbol: Dict[str, pd.DataFrame],
        block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
        walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]],
        family_params_by_symbol: Optional[
            Dict[str, Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]]]
        ] = None,
        weight_a: float,
        weight_b: float,
        sequence_model_type: str,
        lstm_params: Dict[str, Any],
        mamba_params: Dict[str, Any],
        threshold_params: Dict[str, float],
        pipeline_params: Dict[str, Any],
        horizon: int,
        track_b_family_weights: Optional[Dict[str, float]] = None,
        track_b_family_weights_by_symbol: Optional[Dict[str, Dict[str, float]]] = None,
        progress_callback: Optional[Callable[[int, float], bool]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a trial in global multi-symbol pooled mode."""
        import time
        import torch as _torch

        t0 = time.time()
        symbols = sorted(list(panels_by_symbol.keys()))
        if not symbols:
            return float("-inf"), {"reject_reason": "no_symbols"}
        if not walk_forward_folds_multi:
            return float("-inf"), {"reject_reason": "no_folds"}

        # Build per-symbol Track C with shared encoders.
        first_fold = walk_forward_folds_multi[0]
        pooled_train_frames: List[pd.DataFrame] = []
        for sym in symbols:
            tr = first_fold.get(sym)
            if tr is None:
                continue
            train_idx, _ = tr
            if len(train_idx) > 0:
                pooled_train_frames.append(panels_by_symbol[sym].iloc[train_idx])
        if not pooled_train_frames:
            return float("-inf"), {"reject_reason": "no_pooled_train_rows"}

        pooled_panel = pd.concat(pooled_train_frames, axis=0)
        pooled_idx = np.arange(len(pooled_panel), dtype=int)

        # Fit encoders once on pooled training rows.
        # In per-symbol weights mode, `family_params` is expected to represent a
        # pooled/union encoder-fit configuration (dims/methods) across symbols.
        try:
            _track_a_pooled, _ = self._build_track_a(pooled_panel, column_families, family_params, pooled_idx)
        except Exception as exc:
            return float("-inf"), {"reject_reason": "track_a_pooled_failed", "error": str(exc)}

        track_c_by_symbol: Dict[str, pd.DataFrame] = {}
        actual_returns_by_symbol: Dict[str, pd.Series] = {}
        for sym in symbols:
            panel = panels_by_symbol[sym]
            labels = labels_by_symbol[sym]
            actual_returns_by_symbol[sym] = labels["forward_return"].astype(float)
            tr = first_fold.get(sym)
            if tr is None:
                return float("-inf"), {"reject_reason": "missing_symbol_in_fold0", "symbol": sym}
            first_train_idx, _ = tr
            try:
                sym_family_params = (
                    (family_params_by_symbol or {}).get(sym)
                    or (family_params_by_symbol or {}).get(str(sym).upper())
                    or family_params
                )
                sym_track_b_weights = (
                    (track_b_family_weights_by_symbol or {}).get(sym)
                    or (track_b_family_weights_by_symbol or {}).get(str(sym).upper())
                    or track_b_family_weights
                )

                track_a_sym, _ = self._build_track_a(panel, column_families, sym_family_params, first_train_idx)
                track_b_sym = self._build_track_b(
                    panel,
                    column_families,
                    block_summaries_by_symbol.get(sym, {}),
                    track_b_family_weights=sym_track_b_weights,
                )
                track_c_sym = self._build_track_c(track_a_sym, track_b_sym, weight_a, weight_b)
            except Exception as exc:
                return float("-inf"), {"reject_reason": "track_build_failed", "symbol": sym, "error": str(exc)}

            if track_c_sym.empty:
                return float("-inf"), {"reject_reason": "track_c_empty", "symbol": sym}
            # Dimension cap: allow disabling by setting max_total_dims <= 0.
            if int(getattr(self.config, "max_total_dims", 0) or 0) > 0 and int(track_c_sym.shape[1]) > int(self.config.max_total_dims):
                return float("-inf"), {
                    "reject_reason": "track_c_too_large",
                    "symbol": sym,
                    "dims": int(track_c_sym.shape[1]),
                    "max_total_dims": int(self.config.max_total_dims),
                }
            track_c_by_symbol[sym] = track_c_sym

        # Ensure Track-C schema is identical across symbols for pooled scaling + training.
        # Missing columns are filled with deterministic zeros.
        if track_c_by_symbol:
            first_sym = symbols[0]
            ordered_cols: List[str] = list(track_c_by_symbol[first_sym].columns)
            seen = set(ordered_cols)
            for sym in symbols[1:]:
                df = track_c_by_symbol.get(sym)
                if df is None:
                    continue
                for c in df.columns:
                    if c not in seen:
                        ordered_cols.append(c)
                        seen.add(c)
            for sym in list(track_c_by_symbol.keys()):
                track_c_by_symbol[sym] = track_c_by_symbol[sym].reindex(columns=ordered_cols).fillna(0.0)

        # Enforce globally valid seq_len.
        if sequence_model_type == "mamba":
            seq_len = int(mamba_params.get("mamba_seq_len", 128))
            min_symbols_per_fold = int(getattr(self.config, "min_symbols_per_fold", 1))
            val_buffer = 5
            # A symbol/fold only "participates" in constraining seq_len if it can
            # actually be scored for at least a minimal viable seq_len.
            # NOTE: Using configured choices here (often 128+) can incorrectly mark
            # all folds as non-participating when validation windows are shorter,
            # collapsing the cap to 8.
            min_choice = 8
            cap = self._compute_mamba_seq_len_cap_for_multi_folds(
                walk_forward_folds_multi,
                min_symbols_per_fold=min_symbols_per_fold,
                val_buffer=val_buffer,
                min_val_len_for_participation=int(min_choice) + int(val_buffer),
            )
            self._mamba_seq_len_cap_for_folds = cap
            if int(seq_len) > int(cap):
                return float("-inf"), {
                    "reject_reason": "mamba_seq_len_exceeds_fold_cap",
                    "mamba_seq_len": int(seq_len),
                    "mamba_seq_len_cap": int(cap),
                }
        else:
            return float("-inf"), {"reject_reason": "unsupported_sequence_model_type", "type": sequence_model_type}

        # Training config (mamba only for now)
        cfg = {
            "sequence_model_type": "mamba",
            "mamba_seq_len": int(seq_len),
            "mamba_d_model": int(mamba_params.get("mamba_d_model", 128)),
            "mamba_n_layers": int(mamba_params.get("mamba_n_layers", 4)),
            "mamba_ssm_dim": int(mamba_params.get("mamba_ssm_dim", 96)),
            "mamba_expand_factor": float(mamba_params.get("mamba_expand_factor", 2.0)),
            "mamba_activation": mamba_params.get("mamba_activation", "silu"),
            "mamba_norm_type": mamba_params.get("mamba_norm_type", "rmsnorm"),
            "mamba_norm_strategy": mamba_params.get("mamba_norm_strategy", "pre"),
            "mamba_dropout": float(mamba_params.get("mamba_dropout", 0.1)),
            "mamba_resid_dropout": float(mamba_params.get("mamba_resid_dropout", 0.0)),
            "mamba_ssm_dropout": float(mamba_params.get("mamba_ssm_dropout", 0.0)),
            "mamba_gate_dropout": float(mamba_params.get("mamba_gate_dropout", 0.0)),
            "mamba_optimizer": mamba_params.get("mamba_optimizer", "adamw"),
            "mamba_learning_rate": float(mamba_params.get("mamba_learning_rate", 1e-3)),
            "mamba_weight_decay": float(mamba_params.get("mamba_weight_decay", 1e-4)),
            "mamba_grad_clip": float(mamba_params.get("mamba_grad_clip", 1.0)),
            "mamba_lr_scheduler": mamba_params.get("mamba_lr_scheduler", "cosine"),
            "mamba_warmup_steps": int(mamba_params.get("mamba_warmup_steps", 100)),
            "max_epochs": int(mamba_params.get("mamba_max_epochs", 10)),
            "mamba_batch_size": int(mamba_params.get("mamba_batch_size", 32)),
            "mamba_loss_fn": mamba_params.get("mamba_loss_fn", "smooth_l1"),
            "mamba_head_type": mamba_params.get("mamba_head_type", "linear"),
            "mamba_head_hidden_dim": int(mamba_params.get("mamba_head_hidden_dim", 128)),
            "mamba_head_num_layers": int(mamba_params.get("mamba_head_num_layers", 1)),
            "mamba_head_dropout": float(mamba_params.get("mamba_head_dropout", 0.0)),
        }

        device = _torch.device("cuda:0") if _torch.cuda.is_available() else _torch.device("cpu")

        pooled_non_overlap_returns: List[float] = []
        fold_sharpes: List[float] = []
        n_successful_folds = 0
        folds_skipped_undercoverage = 0
        symbols_used_counts: List[int] = []
        # Exposed to Ray Tune progress callback (trainable closure can merge it into tune.report)
        self._ray_progress_metrics = {}

        def _fold_sharpe_from_returns(strategy_returns: Sequence[float]) -> float:
            arr = np.asarray(list(strategy_returns), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2:
                return 0.0
            std = float(np.std(arr))
            if std <= 1e-9:
                return 0.0
            periods_per_year = max(1, 252 // max(1, int(horizon)))
            return float(np.mean(arr) / (std + 1e-9) * np.sqrt(periods_per_year))

        for fold_idx, fold in enumerate(walk_forward_folds_multi):
            # Runtime fold participation / geometry guards:
            # - A symbol is "usable" in this fold only if it can be scored
            #   (i.e., enough validation samples to build sequences)
            # - Skip folds with too few usable symbols
            min_symbols_per_fold = int(getattr(self.config, "min_symbols_per_fold", 1))
            train_buffer = 20
            val_buffer = 5

            # Do not demand more symbols than exist in the run.
            min_symbols_eff = max(1, min(int(min_symbols_per_fold), int(len(symbols))))

            usable_symbols: List[str] = []
            for sym in symbols:
                _train_idx, _val_idx = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
                if int(len(_val_idx)) >= int(seq_len) + int(val_buffer):
                    usable_symbols.append(sym)

            n_symbols_total = int(len(symbols))
            n_symbols_usable = int(len(usable_symbols))
            n_symbols_skipped = int(n_symbols_total - n_symbols_usable)
            self._ray_progress_metrics = {
                "symbols_total": n_symbols_total,
                "symbols_usable": n_symbols_usable,
                "symbols_skipped": n_symbols_skipped,
                "min_symbols_required": int(min_symbols_eff),
            }

            # Policy B: a fold is only scored if it has at least N scorable symbols.
            if int(len(usable_symbols)) < int(min_symbols_eff):
                folds_skipped_undercoverage += 1
                continue

            # Build pooled scaler stats from all symbols' training rows.
            train_frames: List[pd.DataFrame] = []
            train_targets: List[pd.Series] = []
            for sym in symbols:
                train_idx, _ = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
                if len(train_idx) == 0:
                    continue
                train_frames.append(track_c_by_symbol[sym].iloc[train_idx])
                train_targets.append(actual_returns_by_symbol[sym].iloc[train_idx])
            if not train_frames:
                continue

            scaler_stats = self._compute_pooled_scaler_stats(train_frames, train_targets)

            # Build per-symbol train SequenceData and pool by concatenation on sample axis.
            train_seq_datas = []
            for sym in symbols:
                train_idx, _ = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
                if len(train_idx) == 0:
                    continue
                seq_res = build_sequence_data(
                    track_c_by_symbol[sym].iloc[train_idx],
                    actual_returns_by_symbol[sym].iloc[train_idx],
                    int(seq_len),
                    scaler_stats=scaler_stats,
                )
                if seq_res is None:
                    train_seq_datas = []
                    break
                seq_data_sym, _ = seq_res
                if len(seq_data_sym) < 20:
                    train_seq_datas = []
                    break
                train_seq_datas.append(seq_data_sym)
            if not train_seq_datas:
                continue

            pooled_train_seq = self._concat_sequence_data(train_seq_datas)
            n_train = int(len(pooled_train_seq))
            train_end = int(n_train * 0.9)
            seq_train_idx = np.arange(train_end)
            seq_holdout_idx = np.arange(train_end, n_train)
            if int(len(seq_holdout_idx)) < 5:
                continue

            # Train model
            result = train_mamba_fold(
                pooled_train_seq,
                seq_train_idx,
                seq_holdout_idx,
                dict(cfg),
                device=device,
                return_model=True,
            )
            model = result.get("model")
            if model is None:
                continue

            model.eval()
            symbol_return_series: List[pd.Series] = []
            with _torch.no_grad():
                # Score only the participating symbols in this fold.
                for sym in usable_symbols:
                    _, val_idx = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
                    if len(val_idx) == 0:
                        continue

                    val_res = build_sequence_data(
                        track_c_by_symbol[sym].iloc[val_idx],
                        actual_returns_by_symbol[sym].iloc[val_idx],
                        int(seq_len),
                        scaler_stats=scaler_stats,
                    )
                    if val_res is None:
                        continue
                    val_seq_data, _ = val_res
                    if len(val_seq_data) < 5:
                        continue

                    val_timestamps = track_c_by_symbol[sym].iloc[val_idx].index[int(seq_len):]
                    val_X = val_seq_data.sequences.astype(np.float32)
                    val_X_tensor = _torch.from_numpy(val_X).to(device)
                    preds = model(val_X_tensor).detach().cpu().numpy().flatten()

                    # Align lengths
                    if len(preds) > len(val_timestamps):
                        preds = preds[: len(val_timestamps)]
                    elif len(val_timestamps) > len(preds):
                        val_timestamps = val_timestamps[: len(preds)]

                    val_returns = actual_returns_by_symbol[sym].reindex(pd.DatetimeIndex(val_timestamps))
                    strat_series = self._compute_strategy_returns_series(
                        preds,
                        val_returns.values,
                        val_timestamps,
                        threshold_params=threshold_params,
                    )
                    if not strat_series.empty:
                        symbol_return_series.append(strat_series)

                    del val_X_tensor

            # Equal-weight across symbols (missing predictions treated as 0).
            if symbol_return_series:
                df = pd.concat(symbol_return_series, axis=1).fillna(0.0)
                agg = df.mean(axis=1).sort_index()
            else:
                agg = pd.Series([0.0], index=pd.DatetimeIndex([pd.Timestamp.utcnow()]))

            arr = agg.to_numpy(dtype=float)
            if arr.size == 0:
                continue
            if int(horizon) > 1 and arr.size >= int(horizon):
                non_overlap = arr[:: int(horizon)]
            else:
                non_overlap = arr
            if non_overlap.size == 0:
                non_overlap = np.asarray([0.0], dtype=float)

            pooled_non_overlap_returns.extend([float(x) for x in non_overlap.tolist()])
            fold_sharpe = _fold_sharpe_from_returns(non_overlap.tolist())
            fold_sharpes.append(float(fold_sharpe))
            n_successful_folds += 1
            symbols_used_counts.append(int(len(usable_symbols)))

            # ASHA intermediate report
            if progress_callback is not None and n_successful_folds >= 1:
                running = float(np.mean(np.asarray(fold_sharpes, dtype=float))) if fold_sharpes else 0.0
                progress_callback(int(n_successful_folds), running)

            # Cleanup
            del model, result
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        # Final pooled Sharpe across all folds
        pooled_arr = np.asarray(pooled_non_overlap_returns, dtype=float)
        pooled_arr = pooled_arr[np.isfinite(pooled_arr)]
        if pooled_arr.size < 5:
            return float("-inf"), {"reject_reason": "insufficient_pooled_samples", "pooled_samples": int(pooled_arr.size)}

        pooled_mean = float(np.mean(pooled_arr))
        pooled_std = float(np.std(pooled_arr))
        if pooled_std > 1e-9:
            periods_per_year = max(1, 252 // max(1, int(horizon)))
            pooled_sharpe = float(pooled_mean / pooled_std * np.sqrt(periods_per_year))
        else:
            pooled_sharpe = 0.0

        elapsed = float(time.time() - t0)
        return pooled_sharpe, {
            "avg_score": pooled_sharpe,
            "n_successful_folds": int(n_successful_folds),
            "folds_total": int(len(walk_forward_folds_multi)),
            "folds_skipped_undercoverage": int(folds_skipped_undercoverage),
            "symbols_total": int(len(symbols)),
            "symbols_used_mean": float(np.mean(np.asarray(symbols_used_counts, dtype=float))) if symbols_used_counts else 0.0,
            "symbols_used_min": int(np.min(np.asarray(symbols_used_counts, dtype=int))) if symbols_used_counts else 0,
            "symbols_used_max": int(np.max(np.asarray(symbols_used_counts, dtype=int))) if symbols_used_counts else 0,
            "pooled_samples": int(pooled_arr.size),
            "elapsed": elapsed,
            "seq_len": int(seq_len),
            "mamba_seq_len": int(seq_len),
            "mamba_seq_len_cap": int(self._mamba_seq_len_cap_for_folds) if self._mamba_seq_len_cap_for_folds is not None else None,
        }

    def _compute_pooled_objective(
        self,
        pooled_strategy_returns: List[float],
        pooled_directions: List[float],
        pooled_actuals: List[float],
        fold_mean_preds: List[float],
        horizon: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute aggregate objective from pooled non-overlapping samples.
        
        This is much more statistically robust than averaging per-fold Sharpes:
        - 56 folds × ~2 samples each = ~112 independent samples
        - Gives meaningful Sharpe with reasonable confidence interval
        
        Returns:
            Tuple of (objective_score, metrics_dict)
        """
        if len(pooled_strategy_returns) < 5:
            return float("-inf"), {"valid": False, "reason": "insufficient_pooled_samples"}
        
        returns_arr = np.array(pooled_strategy_returns)
        directions_arr = np.array(pooled_directions)
        actuals_arr = np.array(pooled_actuals)
        
        # Pooled Sharpe: use all non-overlapping samples across all folds
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        if std_return > 1e-9:
            # Annualize: each sample represents one H-day period
            periods_per_year = max(1, 252 // horizon)
            pooled_sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        else:
            pooled_sharpe = 0.0
        
        # Pooled RWA: direction accuracy across all samples
        correct_directions = (directions_arr == np.sign(actuals_arr)).astype(float)
        pooled_rwa = np.mean(correct_directions)
        
        # Stability from prediction variance across folds
        if len(fold_mean_preds) >= 2 and np.abs(np.mean(fold_mean_preds)) > 1e-9:
            stability = 1.0 - np.clip(
                np.std(fold_mean_preds) / (np.abs(np.mean(fold_mean_preds)) + 1e-6),
                0.0, 1.0
            )
        else:
            stability = 0.5
        
        # Sign-stability penalty
        sign_flip_rate = 0.0
        if len(fold_mean_preds) >= 2:
            sign_flips = 0
            for i in range(len(fold_mean_preds) - 1):
                sign_i = np.sign(fold_mean_preds[i]) if abs(fold_mean_preds[i]) > 1e-9 else 0
                sign_next = np.sign(fold_mean_preds[i + 1]) if abs(fold_mean_preds[i + 1]) > 1e-9 else 0
                if sign_i != sign_next and sign_i != 0 and sign_next != 0:
                    sign_flips += 1
            sign_flip_rate = sign_flips / (len(fold_mean_preds) - 1)
        
        sign_stability_penalty = 0.15 * sign_flip_rate
        
        # Combined objective with pooled metrics
        objective = (
            self.config.sharpe_weight * pooled_sharpe +
            self.config.rwa_weight * pooled_rwa +
            self.config.stability_weight * stability -
            sign_stability_penalty
        )
        
        metrics = {
            "pooled_sharpe": float(pooled_sharpe),
            "pooled_rwa": float(pooled_rwa),
            "stability": float(stability),
            "sign_flip_rate": float(sign_flip_rate),
            "n_pooled_samples": len(pooled_strategy_returns),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
        }
        
        return float(objective), metrics
    
    def create_objective(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
        horizon: int = 1,
    ):
        """Create Optuna objective for walk-forward optimization."""
        n_folds = len(walk_forward_folds)
        self.logger.info(f"Creating walk-forward objective with {n_folds} folds")

        # Cache fold-derived caps for consistent hyperparameter suggestion.
        self.walk_forward_folds = walk_forward_folds
        self._mamba_seq_len_cap_for_folds = self._compute_mamba_seq_len_cap_for_folds(walk_forward_folds)

        if self.config.use_ray:
            self._setup_ray_workers(
                panel,
                column_families,
                labels,
                block_summaries,
                walk_forward_folds,
            )
        
        def objective(trial: "Trial") -> float:
            try:
                # ===========================================================
                # TIER 1 & 2: Family Selection, Weights, and Dimensionality
                # ===========================================================
                # family_params: (include, latent_dim, method, weight, ae_params)
                family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
                total_track_a_dims = 0
                family_weights: Dict[str, float] = {}
                
                for family in STAGE_A_FAMILIES:
                    include, dim, method, weight, ae_params = self._suggest_family_params(trial, family)
                    family_params[family] = (include, dim, method, weight, ae_params)
                    if include:
                        total_track_a_dims += dim
                        family_weights[family] = weight
                
                # Constraint: Track A dimensions
                if total_track_a_dims < self.config.min_track_a_dims:
                    return float("-inf")
                if total_track_a_dims > self.config.max_track_a_dims:
                    return float("-inf")
                
                # ===========================================================
                # TIER 3: Track Weights
                # ===========================================================
                weight_a, weight_b = self._suggest_track_weights(trial)

                # Track-B per-family weights (separate category from Stage-A families)
                track_b_family_weights = self._suggest_track_b_family_weights(trial)
                
                # ===========================================================
                # TIER 4: SEQUENCE MODEL HYPERPARAMETERS
                # ===========================================================
                # Model type is set in config, not chosen per trial
                # This allows separate optimization runs for LSTM vs Mamba
                sequence_model_type = self.config.sequence_model_type
                
                if sequence_model_type == "lstm":
                    lstm_params = self._suggest_lstm_params(trial)
                    mamba_params = {}  # Empty dict for consistency
                elif sequence_model_type == "mamba":
                    mamba_params = self._suggest_mamba_params(trial)
                    lstm_params = {}  # Empty dict for consistency
                else:
                    raise ValueError(f"Unknown sequence_model_type: {sequence_model_type}")
                
                # ===========================================================
                # STEP 7: Stage-C Threshold & Regime Adjustment
                # ===========================================================
                threshold_params = self._suggest_threshold_params(trial)
                
                # ===========================================================
                # TIER 5: Pipeline Hyperparameters (preprocessing & splitting)
                # ===========================================================
                pipeline_params = self._suggest_pipeline_params(trial)
                
                payload = {
                    "family_params": family_params,
                    "weight_a": weight_a,
                    "weight_b": weight_b,
                    "track_b_family_weights": track_b_family_weights,
                    "sequence_model_type": sequence_model_type,
                    "lstm_params": lstm_params,
                    "mamba_params": mamba_params,
                    "threshold_params": threshold_params,
                    "pipeline_params": pipeline_params,
                    "horizon": horizon,
                }

                if self._ray_enabled and self._ray_workers:
                    worker = self._next_ray_worker()
                    try:
                        final_score, metadata = ray.get(worker.run.remote(payload))
                    except Exception as ray_err:  # pragma: no cover - ray runtime
                        self.logger.warning(f"Ray trial execution failed: {ray_err}")
                        return float("-inf")
                else:
                    final_score, metadata = self._evaluate_trial_params(
                        panel,
                        column_families,
                        labels,
                        block_summaries,
                        walk_forward_folds,
                        family_params,
                        weight_a,
                        weight_b,
                        sequence_model_type,
                        lstm_params,
                        mamba_params,
                        threshold_params,
                        pipeline_params,
                        horizon,
                        track_b_family_weights=track_b_family_weights,
                        trial=trial,  # Pass trial for Hyperband pruning
                    )

                if not np.isfinite(final_score):
                    return float("-inf")

                for key, value in metadata.items():
                    trial.set_user_attr(key, value)

                avg_score = metadata.get("avg_score", final_score)
                score_std = metadata.get("score_std", 0.0)
                n_success = metadata.get("n_successful_folds", 0)
                track_a_dims = metadata.get("track_a_dims", 0)
                track_b_dims = metadata.get("track_b_dims", 0)
                track_c_dims = metadata.get("track_c_dims", 0)

                self.logger.info(
                    f"Trial {trial.number}: score={final_score:.4f} "
                    f"(avg={avg_score:.4f}, std={score_std:.4f}, "
                    f"folds={n_success}/{n_folds}, "
                    f"dims A/B/C={track_a_dims}/{track_b_dims}/{track_c_dims})"
                )
                
                # ============================================================
                # META-OPTIMIZER: Record trial for self-learning
                # ============================================================
                if self._meta_optimizer_enabled and self.meta_optimizer is not None:
                    # Extract individual scores from metadata
                    avg_sharpe = metadata.get("avg_sharpe", 0.0)
                    avg_stability = metadata.get("avg_stability", 0.0)
                    avg_rwa = metadata.get("avg_rwa", 0.0)
                    coverage = metadata.get("avg_coverage", 0.0)
                    elapsed = metadata.get("elapsed_time", 0.0)
                    
                    self.meta_optimizer.record_trial(
                        trial_id=trial.number,
                        params=trial.params,
                        sharpe=avg_sharpe,
                        stability=avg_stability,
                        coverage=coverage,
                        rwa=avg_rwa,
                        hitrate=metadata.get("avg_hitrate", 0.0),
                        final_score=final_score,
                        fold_count=n_success,
                        elapsed_time=elapsed,
                    )
                
                return final_score
                
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return float("-inf")
        
        return objective
    
    def optimize(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
        horizon: int = 1,
        stage_a_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizedParams:
        """Run Optuna optimization with FULL WALK-FORWARD.
        
        ONE TRIAL = ENTIRE WALK-FORWARD (~60 folds)
        
        This ensures consistent encoder caching across all folds within a trial.
        
        Args:
            panel: Full feature panel
            column_families: Column to family mapping
            labels: Labels DataFrame with 'forward_return'
            block_summaries: Stage-B block summaries
            walk_forward_folds: List of (train_idx, val_idx) for ALL WF windows
            horizon: Forecast horizon
            stage_a_weights: Optional Stage-A family weights
        
        Returns:
            OptimizedParams with best configuration
        """
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna not available")
            return OptimizedParams()
        
        # Check if we should use Ray Tune for trial-level parallelism
        if self.config.use_ray_tune and RAY_AVAILABLE and ray is not None:
            self.logger.info("Using Ray Tune for trial-level parallelism")
            return self.optimize_with_ray_tune(
                panel, column_families, labels, block_summaries,
                walk_forward_folds, horizon, stage_a_weights
            )
        
        n_folds = len(walk_forward_folds)
        self.logger.info(f"Walk-forward optimization: {n_folds} folds, {self.config.n_trials} trials")
        
        # Analyze family structure
        self._analyze_families(panel, column_families, stage_a_weights)
        
        # Clear encoder cache before optimization (fresh start)
        self.clear_encoder_cache()
        
        # Create objective with walk-forward folds
        objective = self.create_objective(
            panel, column_families, labels, block_summaries,
            walk_forward_folds, horizon
        )
        
        # Create study with Hyperband pruner for early stopping of bad trials
        # Hyperband dynamically allocates resources to promising trials
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10,       # Minimum 10 folds before pruning decision
            max_resource=60,       # Maximum ~60 folds per trial
            reduction_factor=3,    # Keep top 1/3 at each rung
        )
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            pruner=pruner,
        )
        self.logger.info("Using HyperbandPruner: min_resource=10, max_resource=60, reduction_factor=3")
        
        # Run optimization
        self.logger.info(
            "Starting Optuna optimization: %d trials, timeout=%ds",
            self.config.n_trials, self.config.timeout
        )
        
        try:
            # Determine number of parallel trials
            # IMPORTANT: When using Ray fold parallelism, we MUST use n_jobs=1.
            # Optuna's n_jobs spawns joblib threads/processes, each of which would try to
            # create its own Ray cluster or connect to the existing one incorrectly.
            # Parallelism comes from fold-level Ray tasks instead.
            if self.config.ray_fold_parallelism > 0 and RAY_AVAILABLE and ray is not None and ray.is_initialized():
                n_jobs = 1
                self.logger.info(f"Using n_jobs=1 (sequential trials) with {self.config.ray_fold_parallelism} fold parallelism via Ray")
            elif self._ray_enabled and self._ray_workers:
                # If we have Ray trial workers but no fold parallelism, we could use n_jobs
                # BUT this still has issues with joblib + Ray. Keep sequential for now.
                n_jobs = 1
                self.logger.info(f"Using n_jobs=1 with {len(self._ray_workers)} Ray trial workers")
            else:
                n_jobs = self.config.n_jobs

            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )
        finally:
            self._shutdown_ray_workers()
        
        # Extract best params
        if self.study.best_trial is None:
            self.logger.warning("No successful trials")
            return OptimizedParams()
        
        best = self.study.best_trial
        self.best_params = self._extract_params(best)
        
        # Log cache efficiency
        cache_stats = self.get_encoder_cache_stats()
        self.logger.info(
            "Encoder cache: %d encoders cached (%d families, methods: PCA=%d, AE=%d, Hybrid=%d)",
            cache_stats["total_cached"],
            cache_stats["families_cached"],
            cache_stats["by_method"]["pca"],
            cache_stats["by_method"]["ae"],
            cache_stats["by_method"]["hybrid"],
        )
        
        self.logger.info(
            "Optimization complete: best_score=%.4f, trial=%d",
            self.best_params.best_score, self.best_params.trial_number
        )
        
        # Log meta-optimizer summary if enabled and save state
        if self._meta_optimizer_enabled and self.meta_optimizer is not None:
            self._log_meta_summary()
            # 🔧 FIX: Save meta-optimizer state after optimization completes
            self.meta_optimizer.save()
            self.logger.info("💾 Saved meta-optimizer state for future runs")
        
        return self.best_params

    def optimize_multi_symbol(
        self,
        panels_by_symbol: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
        labels_by_symbol: Dict[str, pd.DataFrame],
        block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
        walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        horizon: int = 1,
        stage_a_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizedParams:
        """Run Optuna optimization in global multi-symbol mode.

        Each trial trains ONE shared sequence model per fold on pooled batches
        (concatenated on batch/sample axis only) across the provided symbols.
        """
        if not OPTUNA_AVAILABLE:
            self.logger.error("Optuna not available")
            return OptimizedParams()

        if not panels_by_symbol:
            self.logger.error("No panels provided for multi-symbol optimization")
            return OptimizedParams()

        if self.config.use_ray_tune and RAY_AVAILABLE and ray is not None:
            self.logger.info("Using Ray Tune for multi-symbol trial-level parallelism")
            return self.optimize_with_ray_tune_multi_symbol(
                panels_by_symbol=panels_by_symbol,
                column_families=column_families,
                labels_by_symbol=labels_by_symbol,
                block_summaries_by_symbol=block_summaries_by_symbol,
                walk_forward_folds_multi=walk_forward_folds_multi,
                horizon=horizon,
                stage_a_weights=stage_a_weights,
            )

        # Non-Ray path is intentionally not implemented yet.
        self.logger.error("Multi-symbol optimization currently requires Ray Tune (config.use_ray_tune=True)")
        return OptimizedParams()
    
    def _log_meta_summary(self):
        """Log a summary of meta-optimizer learnings."""
        if self.meta_optimizer is None:
            return
        
        summary = self.meta_optimizer.get_memory_summary()
        attr_summary = self.meta_optimizer.get_attribution_summary()
        inter_summary = self.meta_optimizer.get_interaction_summary()
        
        self.logger.info("=" * 60)
        self.logger.info("META-OPTIMIZER SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Trials analyzed: {summary.get('total_trials', 0)}")
        self.logger.info(f"Elite trials: {summary.get('elite_count', 0)}")
        
        baseline = summary.get('baseline_stats', {})
        self.logger.info(
            f"Baseline: Sharpe={baseline.get('sharpe_mean', 0):.4f}±{baseline.get('sharpe_std', 0):.4f}, "
            f"Stability={baseline.get('stability_mean', 0):.4f}±{baseline.get('stability_std', 0):.4f}"
        )
        
        # Top helpful params
        helpful = attr_summary.get('helpful', [])
        if helpful:
            self.logger.info("Top helpful parameters:")
            for p in helpful[:5]:
                self.logger.info(f"  {p['param']}: impact={p['impact']:+.3f}")
        
        # Top harmful params
        harmful = attr_summary.get('harmful', [])
        if harmful:
            self.logger.info("Top harmful parameters:")
            for p in harmful[:3]:
                self.logger.info(f"  {p['param']}: impact={p['impact']:+.3f}")
        
        # Top interactions
        interactions = inter_summary.get('top_interactions', [])
        if interactions:
            self.logger.info("Detected parameter interactions:")
            for inter in interactions[:3]:
                params = inter.get('params', ('?', '?'))
                self.logger.info(
                    f"  {params[0]} ↔ {params[1]}: "
                    f"strength={inter.get('strength', 0):.3f} ({inter.get('type', 'unknown')})"
                )
        
        self.logger.info("=" * 60)
    
    def _extract_params(self, trial: "Trial") -> OptimizedParams:
        """Extract OptimizedParams from a trial."""
        params = OptimizedParams()
        params.best_score = trial.value if trial.value is not None else 0.0
        params.trial_number = trial.number
        
        # Extract family params (weight-based)
        for family in STAGE_A_FAMILIES:
            weight_key = f"weight_{family}"
            dim_key = f"{family}_dim"
            
            # Get weight (replaces boolean toggle)
            weight = trial.params.get(weight_key, 0.0)
            params.family_weights[family] = weight
            
            # Derive inclusion from weight
            include = weight >= self.config.family_weight_clip_min
            params.included_families[family] = include
            
            if dim_key in trial.params:
                params.family_dimensions[family] = trial.params[dim_key]
            
            # Determine encoder type from trial params (Optuna-chosen)
            method_key = f"{family}_dim_type"
            method = trial.params.get(method_key, "pca")  # Default to PCA if not found
            params.family_encoders[family] = method
        
        # Track weights (track_a_weight hard-coded to 1.0 - family weights control Track A)
        params.track_a_weight = 1.0  # HARD-CODED
        params.track_b_weight = trial.params.get("track_b_weight", 1.0)

        # Track-B per-family weights (separate category)
        for family in HF_BLOCK_FAMILIES:
            params.track_b_family_weights[family] = float(trial.params.get(f"weight_b_{family}", 1.0))
        for summary_name in TRACK_B_SUMMARY_BLOCKS:
            params.track_b_family_weights[summary_name] = float(trial.params.get(f"weight_b_{summary_name}", 1.0))
        
        # Sequence model: LSTM only (TFT removed)
        params.sequence_model = "lstm"
        
        # LSTM params
        params.lstm_layers = trial.params.get("lstm_layers", 2)
        params.lstm_hidden_dim = trial.params.get("lstm_hidden_dim", 96)
        params.lstm_dropout = trial.params.get("lstm_dropout", 0.2)
        params.lstm_seq_len = trial.params.get("lstm_seq_len", 30)
        params.lstm_batch_size = trial.params.get("lstm_batch_size", 64)
        params.lstm_learning_rate = trial.params.get("lstm_learning_rate", 1e-3)
        params.lstm_optimizer = trial.params.get("lstm_optimizer", "AdamW")
        params.lstm_activation = trial.params.get("lstm_activation", "relu")
        params.lstm_use_amp = trial.params.get("lstm_use_amp", True)
        
        # LSTM Architecture (HARD-CODED: bidirectional always False)
        params.lstm_bidirectional = False  # HARD-CODED
        params.lstm_cell_type = trial.params.get("lstm_cell_type", "lstm")
        params.lstm_recurrent_kernel_init = trial.params.get("lstm_recurrent_kernel_init", "orthogonal")
        params.lstm_hidden_state_init = trial.params.get("lstm_hidden_state_init", "zeros")
        
        # LSTM Regularization
        params.lstm_recurrent_dropout = trial.params.get("lstm_recurrent_dropout", 0.0)
        params.lstm_input_dropout = trial.params.get("lstm_input_dropout", 0.0)
        params.lstm_grad_clip = trial.params.get("lstm_grad_clip", 1.0)
        params.lstm_time_dropout = trial.params.get("lstm_time_dropout", 0.0)
        params.lstm_recurrent_weight_decay = trial.params.get("lstm_recurrent_weight_decay", 0.0)
        params.lstm_weight_dropout = trial.params.get("lstm_weight_dropout", 0.0)
        params.lstm_zoneout = trial.params.get("lstm_zoneout", 0.0)
        params.lstm_sequence_noise_std = trial.params.get("lstm_sequence_noise_std", 0.0)
        
        # LSTM Enhancements
        params.lstm_layer_norm = trial.params.get("lstm_layer_norm", False)
        params.lstm_residual = trial.params.get("lstm_residual", False)
        params.lstm_skip_connect = trial.params.get("lstm_skip_connect", False)
        params.lstm_lr_multiplier = trial.params.get("lstm_lr_multiplier", 1.0)
        
        # LSTM Attention (comprehensive attention system)
        params.lstm_attention_type = trial.params.get("lstm_attention_type", "none")
        params.lstm_attn_hidden_dim = trial.params.get("lstm_attn_hidden_dim", 64)
        params.lstm_attn_dropout = trial.params.get("lstm_attn_dropout", 0.1)
        params.lstm_attn_heads = trial.params.get("lstm_attn_heads", 1)
        params.lstm_attn_normalization = trial.params.get("lstm_attn_normalization", "softmax")
        params.lstm_attn_score_fn = trial.params.get("lstm_attn_score_fn", "dot")
        params.lstm_attn_merge = trial.params.get("lstm_attn_merge", "concat")
        params.lstm_attn_positional_encoding = trial.params.get("lstm_attn_positional_encoding", False)
        params.lstm_attn_temperature = trial.params.get("lstm_attn_temperature", 1.0)
        params.lstm_attn_entropy_reg = trial.params.get("lstm_attn_entropy_reg", 0.0)
        params.lstm_attn_distance_reg = trial.params.get("lstm_attn_distance_reg", 0.0)
        params.lstm_attn_context_length = trial.params.get("lstm_attn_context_length", 30)
        params.lstm_attn_key_dim = trial.params.get("lstm_attn_key_dim", 64)
        params.lstm_attn_value_dim = trial.params.get("lstm_attn_value_dim", 64)
        
        # LSTM Output
        params.lstm_fc_layers = trial.params.get("lstm_fc_layers", 1)
        params.lstm_fc_hidden = trial.params.get("lstm_fc_hidden", 64)
        params.lstm_output_dropout = trial.params.get("lstm_output_dropout", 0.2)
        params.lstm_output_activation = trial.params.get("lstm_output_activation", "none")
        
        # LSTM Optimization
        params.lstm_gradient_accumulation = trial.params.get("lstm_gradient_accumulation", 1)
        params.lstm_momentum = trial.params.get("lstm_momentum", 0.9)
        
        # 4.3 Loss function
        params.loss_fn = trial.params.get("loss_fn", "mse")
        params.huber_delta = trial.params.get("huber_delta", 1.0)
        params.quantile_alpha = trial.params.get("quantile_alpha", 0.5)
        
        # 4.8 Regularization
        params.input_noise_std = trial.params.get("input_noise_std", 0.0)
        params.weight_decay = trial.params.get("weight_decay", 0.0)
        
        # 4.9 LR scheduler
        params.lr_scheduler = trial.params.get("lr_scheduler", "none")
        params.cosine_t_max = trial.params.get("cosine_t_max", 50)
        params.plateau_patience = trial.params.get("plateau_patience", 5)
        
        # 4.10 Training control
        params.early_stopping_patience = trial.params.get("early_stopping_patience", 10)
        params.max_epochs = trial.params.get("max_epochs", 100)
        params.warmup_steps = trial.params.get("warmup_steps", 100)
        
        # STEP 7: Threshold & Regime params
        params.threshold = trial.params.get("threshold", 0.10)
        params.bull_mult = trial.params.get("bull_mult", 0.8)
        params.bear_mult = trial.params.get("bear_mult", 1.5)
        params.crisis_mult = trial.params.get("crisis_mult", 3.0)
        
        # Pipeline-level hyperparameters (preprocessing & splitting)
        params.smoothing_type = trial.params.get("smoothing_type", "none")
        params.smoothing_window = trial.params.get("smoothing_window", 5)
        params.train_fraction = trial.params.get("train_fraction", 0.8)

        # Mamba params (Stage-B runtime)
        params.mamba_seq_len = int(trial.params.get("mamba_seq_len", params.mamba_seq_len))
        params.mamba_d_model = int(trial.params.get("mamba_d_model", params.mamba_d_model))
        params.mamba_n_layers = int(trial.params.get("mamba_n_layers", params.mamba_n_layers))
        params.mamba_ssm_dim = int(trial.params.get("mamba_ssm_dim", params.mamba_ssm_dim))
        params.mamba_expand_factor = float(trial.params.get("mamba_expand_factor", params.mamba_expand_factor))
        params.mamba_activation = str(trial.params.get("mamba_activation", params.mamba_activation))
        params.mamba_norm_type = str(trial.params.get("mamba_norm_type", params.mamba_norm_type))
        params.mamba_norm_strategy = str(trial.params.get("mamba_norm_strategy", params.mamba_norm_strategy))
        params.mamba_dropout = float(trial.params.get("mamba_dropout", params.mamba_dropout))
        params.mamba_resid_dropout = float(trial.params.get("mamba_resid_dropout", params.mamba_resid_dropout))
        params.mamba_ssm_dropout = float(trial.params.get("mamba_ssm_dropout", params.mamba_ssm_dropout))
        params.mamba_gate_dropout = float(trial.params.get("mamba_gate_dropout", params.mamba_gate_dropout))
        params.mamba_optimizer = str(trial.params.get("mamba_optimizer", params.mamba_optimizer))
        params.mamba_learning_rate = float(trial.params.get("mamba_learning_rate", params.mamba_learning_rate))
        params.mamba_weight_decay = float(trial.params.get("mamba_weight_decay", params.mamba_weight_decay))
        params.mamba_grad_clip = float(trial.params.get("mamba_grad_clip", params.mamba_grad_clip))
        params.mamba_lr_scheduler = str(trial.params.get("mamba_lr_scheduler", params.mamba_lr_scheduler))
        params.mamba_warmup_steps = int(trial.params.get("mamba_warmup_steps", params.mamba_warmup_steps))
        params.mamba_max_epochs = int(trial.params.get("mamba_max_epochs", params.mamba_max_epochs))
        params.mamba_batch_size = int(trial.params.get("mamba_batch_size", params.mamba_batch_size))
        params.mamba_loss_fn = str(trial.params.get("mamba_loss_fn", params.mamba_loss_fn))
        params.mamba_head_type = str(trial.params.get("mamba_head_type", params.mamba_head_type))
        params.mamba_head_hidden_dim = int(trial.params.get("mamba_head_hidden_dim", params.mamba_head_hidden_dim))
        params.mamba_head_num_layers = int(trial.params.get("mamba_head_num_layers", params.mamba_head_num_layers))
        params.mamba_head_dropout = float(trial.params.get("mamba_head_dropout", params.mamba_head_dropout))
        
        return params

    def _extract_params_from_dict(self, config: Dict[str, Any], score: float = 0.0) -> OptimizedParams:
        """Extract OptimizedParams from a Ray Tune config dict."""
        params = OptimizedParams()
        params.best_score = score
        params.trial_number = -1  # Ray Tune doesn't have trial numbers

        per_symbol_weights = bool(getattr(self.config, "global_multi_symbol", False)) and bool(
            getattr(self.config, "global_per_symbol_weights", False)
        )
        symbols: List[str] = []
        if per_symbol_weights:
            try:
                symbols = [str(s).upper() for s in (getattr(self.config, "global_symbols", ()) or ())]
            except Exception:
                symbols = []
            symbols = sorted([s for s in symbols if s])

        # Extract family params (weight-based)
        if per_symbol_weights and symbols:
            params.family_weights_by_symbol = {}
            params.track_b_family_weights_by_symbol = {}

            # Per-symbol Stage-A weights
            for sym in symbols:
                sym_weights: Dict[str, float] = {}
                for family in STAGE_A_FAMILIES:
                    sym_weights[family] = float(config.get(f"{sym}__weight_{family}", 0.0))
                params.family_weights_by_symbol[sym] = sym_weights

            # Derive union inclusion and provide a backward-compat aggregate weight (mean).
            for family in STAGE_A_FAMILIES:
                ws = [float(params.family_weights_by_symbol[sym].get(family, 0.0)) for sym in symbols]
                w_mean = float(np.mean(ws)) if ws else 0.0
                params.family_weights[family] = w_mean
                include_any = any(w >= self.config.family_weight_clip_min for w in ws)
                params.included_families[family] = bool(include_any)

                method_key = f"{family}_dim_type"
                method = config.get(method_key, "pca")
                params.family_encoders[family] = method

            # Per-symbol Track-B weights
            for sym in symbols:
                sym_tb: Dict[str, float] = {}
                for family in HF_BLOCK_FAMILIES:
                    sym_tb[family] = float(config.get(f"{sym}__weight_b_{family}", 1.0))
                for summary_name in TRACK_B_SUMMARY_BLOCKS:
                    sym_tb[summary_name] = float(config.get(f"{sym}__weight_b_{summary_name}", 1.0))
                params.track_b_family_weights_by_symbol[sym] = sym_tb

            # Backward-compat aggregate Track-B weights (mean)
            for family in HF_BLOCK_FAMILIES:
                ws = [float(params.track_b_family_weights_by_symbol[sym].get(family, 1.0)) for sym in symbols]
                params.track_b_family_weights[family] = float(np.mean(ws)) if ws else 1.0
            for summary_name in TRACK_B_SUMMARY_BLOCKS:
                ws = [float(params.track_b_family_weights_by_symbol[sym].get(summary_name, 1.0)) for sym in symbols]
                params.track_b_family_weights[summary_name] = float(np.mean(ws)) if ws else 1.0
        else:
            for family in STAGE_A_FAMILIES:
                weight_key = f"weight_{family}"
                dim_key = f"{family}_dim"

                weight = config.get(weight_key, 0.0)
                params.family_weights[family] = weight

                include = weight >= self.config.family_weight_clip_min
                params.included_families[family] = include

                if dim_key in config:
                    params.family_dimensions[family] = config[dim_key]

                method_key = f"{family}_dim_type"
                method = config.get(method_key, "pca")
                params.family_encoders[family] = method
        
        # Track weights (track_a_weight hard-coded to 1.0 - family weights control Track A)
        params.track_a_weight = 1.0  # HARD-CODED
        params.track_b_weight = config.get("track_b_weight", 1.0)

        # Track-B per-family weights (separate category)
        if not (per_symbol_weights and symbols):
            for family in HF_BLOCK_FAMILIES:
                params.track_b_family_weights[family] = float(config.get(f"weight_b_{family}", 1.0))
            for summary_name in TRACK_B_SUMMARY_BLOCKS:
                params.track_b_family_weights[summary_name] = float(config.get(f"weight_b_{summary_name}", 1.0))
        
        # Sequence model: LSTM only (TFT removed)
        params.sequence_model = "lstm"

        # Mamba params (present in Ray Tune configs and cached JSON in Mamba mode)
        params.mamba_seq_len = int(config.get("mamba_seq_len", params.mamba_seq_len))
        params.mamba_d_model = int(config.get("mamba_d_model", params.mamba_d_model))
        params.mamba_n_layers = int(config.get("mamba_n_layers", params.mamba_n_layers))
        params.mamba_ssm_dim = int(config.get("mamba_ssm_dim", params.mamba_ssm_dim))
        params.mamba_expand_factor = float(config.get("mamba_expand_factor", params.mamba_expand_factor))
        params.mamba_activation = str(config.get("mamba_activation", params.mamba_activation))
        params.mamba_norm_type = str(config.get("mamba_norm_type", params.mamba_norm_type))
        params.mamba_norm_strategy = str(config.get("mamba_norm_strategy", params.mamba_norm_strategy))
        params.mamba_dropout = float(config.get("mamba_dropout", params.mamba_dropout))
        params.mamba_resid_dropout = float(config.get("mamba_resid_dropout", params.mamba_resid_dropout))
        params.mamba_ssm_dropout = float(config.get("mamba_ssm_dropout", params.mamba_ssm_dropout))
        params.mamba_gate_dropout = float(config.get("mamba_gate_dropout", params.mamba_gate_dropout))
        params.mamba_optimizer = str(config.get("mamba_optimizer", params.mamba_optimizer))
        params.mamba_learning_rate = float(config.get("mamba_learning_rate", params.mamba_learning_rate))
        params.mamba_weight_decay = float(config.get("mamba_weight_decay", params.mamba_weight_decay))
        params.mamba_grad_clip = float(config.get("mamba_grad_clip", params.mamba_grad_clip))
        params.mamba_lr_scheduler = str(config.get("mamba_lr_scheduler", params.mamba_lr_scheduler))
        params.mamba_warmup_steps = int(config.get("mamba_warmup_steps", params.mamba_warmup_steps))
        params.mamba_max_epochs = int(config.get("mamba_max_epochs", params.mamba_max_epochs))
        params.mamba_batch_size = int(config.get("mamba_batch_size", params.mamba_batch_size))
        params.mamba_loss_fn = str(config.get("mamba_loss_fn", params.mamba_loss_fn))
        params.mamba_head_type = str(config.get("mamba_head_type", params.mamba_head_type))
        params.mamba_head_hidden_dim = int(config.get("mamba_head_hidden_dim", params.mamba_head_hidden_dim))
        params.mamba_head_num_layers = int(config.get("mamba_head_num_layers", params.mamba_head_num_layers))
        params.mamba_head_dropout = float(config.get("mamba_head_dropout", params.mamba_head_dropout))
        
        # LSTM params
        params.lstm_layers = config.get("lstm_layers", 2)
        params.lstm_hidden_dim = config.get("lstm_hidden_dim", 96)
        params.lstm_dropout = config.get("lstm_dropout", 0.2)
        params.lstm_seq_len = config.get("lstm_seq_len", 30)
        params.lstm_batch_size = config.get("lstm_batch_size", 64)
        params.lstm_learning_rate = config.get("lstm_learning_rate", 1e-3)
        params.lstm_optimizer = config.get("lstm_optimizer", "AdamW")
        params.lstm_activation = config.get("lstm_activation", "relu")
        params.lstm_use_amp = config.get("lstm_use_amp", True)
        
        # LSTM Architecture (HARD-CODED: bidirectional always False)
        params.lstm_bidirectional = False  # HARD-CODED
        params.lstm_cell_type = config.get("lstm_cell_type", "lstm")
        params.lstm_recurrent_kernel_init = config.get("lstm_recurrent_kernel_init", "orthogonal")
        params.lstm_hidden_state_init = config.get("lstm_hidden_state_init", "zeros")
        
        # LSTM Regularization
        params.lstm_recurrent_dropout = config.get("lstm_recurrent_dropout", 0.0)
        params.lstm_input_dropout = config.get("lstm_input_dropout", 0.0)
        params.lstm_grad_clip = config.get("lstm_grad_clip", 1.0)
        params.lstm_time_dropout = config.get("lstm_time_dropout", 0.0)
        params.lstm_recurrent_weight_decay = config.get("lstm_recurrent_weight_decay", 0.0)
        params.lstm_weight_dropout = config.get("lstm_weight_dropout", 0.0)
        params.lstm_zoneout = config.get("lstm_zoneout", 0.0)
        params.lstm_sequence_noise_std = config.get("lstm_sequence_noise_std", 0.0)
        
        # LSTM Enhancements
        params.lstm_layer_norm = config.get("lstm_layer_norm", False)
        params.lstm_residual = config.get("lstm_residual", False)
        params.lstm_skip_connect = config.get("lstm_skip_connect", False)
        params.lstm_lr_multiplier = config.get("lstm_lr_multiplier", 1.0)
        
        # LSTM Attention (comprehensive attention system)
        params.lstm_attention_type = config.get("lstm_attention_type", "none")
        params.lstm_attn_hidden_dim = config.get("lstm_attn_hidden_dim", 64)
        params.lstm_attn_dropout = config.get("lstm_attn_dropout", 0.1)
        params.lstm_attn_heads = config.get("lstm_attn_heads", 1)
        params.lstm_attn_normalization = config.get("lstm_attn_normalization", "softmax")
        params.lstm_attn_score_fn = config.get("lstm_attn_score_fn", "dot")
        params.lstm_attn_merge = config.get("lstm_attn_merge", "concat")
        params.lstm_attn_positional_encoding = config.get("lstm_attn_positional_encoding", False)
        params.lstm_attn_temperature = config.get("lstm_attn_temperature", 1.0)
        params.lstm_attn_entropy_reg = config.get("lstm_attn_entropy_reg", 0.0)
        params.lstm_attn_distance_reg = config.get("lstm_attn_distance_reg", 0.0)
        params.lstm_attn_context_length = config.get("lstm_attn_context_length", 30)
        params.lstm_attn_key_dim = config.get("lstm_attn_key_dim", 64)
        params.lstm_attn_value_dim = config.get("lstm_attn_value_dim", 64)
        
        # LSTM Output
        params.lstm_fc_layers = config.get("lstm_fc_layers", 1)
        params.lstm_fc_hidden = config.get("lstm_fc_hidden", 64)
        params.lstm_output_dropout = config.get("lstm_output_dropout", 0.2)
        params.lstm_output_activation = config.get("lstm_output_activation", "none")
        
        # LSTM Optimization
        params.lstm_gradient_accumulation = config.get("lstm_gradient_accumulation", 1)
        params.lstm_momentum = config.get("lstm_momentum", 0.9)
        
        # 4.3 Loss function
        params.loss_fn = config.get("loss_fn", "mse")
        params.huber_delta = config.get("huber_delta", 1.0)
        params.quantile_alpha = config.get("quantile_alpha", 0.5)
        
        # 4.8 Regularization
        params.input_noise_std = config.get("input_noise_std", 0.0)
        params.weight_decay = config.get("weight_decay", 0.0)
        
        # 4.9 LR scheduler
        params.lr_scheduler = config.get("lr_scheduler", "none")
        params.cosine_t_max = config.get("cosine_t_max", 50)
        params.plateau_patience = config.get("plateau_patience", 5)
        
        # 4.10 Training control
        params.early_stopping_patience = config.get("early_stopping_patience", 10)
        params.max_epochs = config.get("max_epochs", 100)
        params.warmup_steps = config.get("warmup_steps", 100)
        
        # STEP 7: Threshold & Regime params
        params.threshold = config.get("threshold", 0.10)
        params.bull_mult = config.get("bull_mult", 0.8)
        params.bear_mult = config.get("bear_mult", 1.5)
        params.crisis_mult = config.get("crisis_mult", 3.0)
        
        # Pipeline-level hyperparameters (preprocessing & splitting)
        params.smoothing_type = config.get("smoothing_type", "none")
        params.smoothing_window = config.get("smoothing_window", 5)
        params.train_fraction = config.get("train_fraction", 0.8)
        
        return params

    def _build_ray_tune_search_space(self) -> Dict[str, Any]:
        """Build Ray Tune search space matching the Optuna trial suggestion logic.
        
        Uses meta-optimizer adapted bounds when available to guide search.
        """
        if not RAY_AVAILABLE or tune is None:
            return {}
        
        search_space: Dict[str, Any] = {}

        def _fixed_choice(value: Any) -> Any:
            return tune.choice([value])

        fixed_params = dict(getattr(self.config, "fixed_params", {}) or {})
        weights_only = bool(getattr(self.config, "tune_weights_only", False))

        global_multi_symbol = bool(getattr(self.config, "global_multi_symbol", False))
        per_symbol_weights = bool(getattr(self.config, "global_per_symbol_weights", False)) and global_multi_symbol
        global_symbols: List[str] = []
        if per_symbol_weights:
            try:
                global_symbols = [str(s).upper() for s in (getattr(self.config, "global_symbols", ()) or ())]
            except Exception:
                global_symbols = []
            global_symbols = sorted([s for s in global_symbols if s])
        
        # Get adapted bounds from meta-optimizer if available
        adapted_bounds: Dict[str, Tuple[float, float]] = {}
        if self._meta_optimizer_enabled and self.meta_optimizer is not None:
            try:
                adapted_bounds = self.meta_optimizer.get_adapted_bounds()
                if adapted_bounds:
                    self.logger.info(f"🧠 Meta-optimizer providing {len(adapted_bounds)} adapted bounds")
            except Exception as e:
                self.logger.debug(f"Could not get adapted bounds: {e}")
        
        def get_bounds(param: str, default_min: float, default_max: float) -> Tuple[float, float]:
            """Get bounds for a parameter, using meta-optimizer if available."""
            if param in adapted_bounds:
                meta_min, meta_max = adapted_bounds[param]
                # Clamp to original bounds for safety
                return (max(meta_min, default_min), min(meta_max, default_max))
            return (default_min, default_max)
        
        # Family weights and dimension choices (weight-based, not boolean)
        # NEW: 3-pillar dims use pre-computed optimal dimensions per family
        for family in STAGE_A_FAMILIES:
            family_size = self.family_sizes.get(family, 0)
            
            if family_size == 0:
                # Zero-size family: weight = 0
                if per_symbol_weights and global_symbols:
                    for sym in global_symbols:
                        search_space[f"{sym}__weight_{family}"] = tune.choice([0.0])
                else:
                    search_space[f"weight_{family}"] = tune.choice([0.0])
                search_space[f"{family}_dim_type"] = tune.choice(["pca"])
                search_space[f"{family}_pca_dim"] = tune.choice([0])
            else:
                # Family weight (continuous, replaces boolean toggle)
                # Use meta-optimizer adapted bounds if available
                if per_symbol_weights and global_symbols:
                    for sym in global_symbols:
                        weight_param = f"{sym}__weight_{family}"
                        w_min, w_max = get_bounds(
                            weight_param,
                            self.config.family_weight_min,
                            self.config.family_weight_max,
                        )
                        search_space[weight_param] = tune.uniform(w_min, w_max)
                else:
                    weight_param = f"weight_{family}"
                    w_min, w_max = get_bounds(
                        weight_param,
                        self.config.family_weight_min,
                        self.config.family_weight_max
                    )
                    search_space[weight_param] = tune.uniform(w_min, w_max)

                # -----------------------------------------------------------------
                # WEIGHTS-ONLY MODE: freeze encoder/dim choices
                # -----------------------------------------------------------------
                if weights_only:
                    if self.config.use_three_pillar_dims and family in self.family_optimal_dims:
                        dim_config = self.family_optimal_dims[family]
                        method = str(dim_config.method)
                        k_final = int(dim_config.k_final)
                    else:
                        method = str(fixed_params.get(f"{family}_dim_type", "pca") or "pca")
                        k_final = int(
                            fixed_params.get(
                                f"{family}_ae_dim" if method == "ae" else f"{family}_pca_dim",
                                10,
                            )
                            or 10
                        )

                    search_space[f"{family}_dim_type"] = _fixed_choice(method)
                    if method == "ae":
                        search_space[f"{family}_ae_dim"] = _fixed_choice(int(k_final))
                        search_space[f"{family}_ae_layers"] = _fixed_choice(int(fixed_params.get(f"{family}_ae_layers", 2) or 2))
                        search_space[f"{family}_ae_activation"] = _fixed_choice(str(fixed_params.get(f"{family}_ae_activation", "gelu") or "gelu"))
                        search_space[f"{family}_ae_dropout"] = _fixed_choice(float(fixed_params.get(f"{family}_ae_dropout", 0.1) or 0.1))
                        search_space[f"{family}_ae_lr"] = _fixed_choice(float(fixed_params.get(f"{family}_ae_lr", 5e-4) or 5e-4))
                    else:
                        search_space[f"{family}_pca_dim"] = _fixed_choice(int(k_final))
                    continue

                # =========================================================================
                # 3-PILLAR: Use pre-computed optimal dims if available
                # =========================================================================
                if self.config.use_three_pillar_dims and family in self.family_optimal_dims:
                    dim_config = self.family_optimal_dims[family]

                    # Fixed method from 3-pillar analysis
                    search_space[f"{family}_dim_type"] = tune.choice([dim_config.method])

                    # Fixed dimension from 3-pillar analysis
                    search_space[f"{family}_pca_dim"] = tune.choice([dim_config.k_final])
                    search_space[f"{family}_ae_dim"] = tune.choice([dim_config.k_final])

                    # Fixed AE architecture (no search needed)
                    search_space[f"{family}_ae_layers"] = tune.choice([2])
                    search_space[f"{family}_ae_activation"] = tune.choice(["gelu"])
                    search_space[f"{family}_ae_dropout"] = tune.choice([0.1])
                    search_space[f"{family}_ae_lr"] = tune.choice([5e-4])

                else:
                    # =========================================================================
                    # LEGACY: Wide Optuna search over dim types and dims
                    # =========================================================================
                    # Dimension reduction type: PCA or AE
                    search_space[f"{family}_dim_type"] = tune.choice(list(self.config.dim_reduction_types))

                    # PCA dimension choices
                    pca_min = self.config.pca_components_min
                    pca_max = min(self.config.pca_components_max, family_size)
                    pca_min = min(pca_min, pca_max)
                    pca_choices = list(range(pca_min, pca_max + 1, 2))  # Step by 2 for efficiency
                    if not pca_choices:
                        pca_choices = [pca_min]
                    search_space[f"{family}_pca_dim"] = tune.choice(pca_choices)

                    # AE dimension choices
                    ae_min = self.config.ae_latent_dim_min
                    ae_max = min(self.config.ae_latent_dim_max, family_size)
                    ae_min = min(ae_min, ae_max)
                    ae_choices = list(range(ae_min, ae_max + 1, 4))  # Step by 4 for efficiency
                    if not ae_choices:
                        ae_choices = [ae_min]
                    search_space[f"{family}_ae_dim"] = tune.choice(ae_choices)

                    # AE architecture hyperparameters
                    search_space[f"{family}_ae_layers"] = tune.choice(list(self.config.ae_layers_choices))
                    search_space[f"{family}_ae_activation"] = tune.choice(list(self.config.ae_activation_choices))
                    search_space[f"{family}_ae_dropout"] = tune.uniform(
                        self.config.ae_dropout_min,
                        self.config.ae_dropout_max
                    )
                    search_space[f"{family}_ae_lr"] = tune.loguniform(
                        self.config.ae_lr_min,
                        self.config.ae_lr_max
                    )
        
        # Track weights - track_a_weight is HARD-CODED to 1.0 (family weights control Track A)
        # Track B:
        # - Global multiplier: track_b_weight
        # - Per-family weights: weight_b_* (HF blocks + summary blocks)
        if weights_only:
            search_space["track_b_weight"] = _fixed_choice(float(fixed_params.get("track_b_weight", 1.0)))
        else:
            tb_min, tb_max = get_bounds(
                "track_b_weight", 
                self.config.track_b_weight_min,
                self.config.track_b_weight_max
            )
            search_space["track_b_weight"] = tune.uniform(tb_min, tb_max)

        # Track-B per-family weights (separate category from Stage-A families)
        # HF block families
        if per_symbol_weights and global_symbols:
            for sym in global_symbols:
                for family in HF_BLOCK_FAMILIES:
                    search_space[f"{sym}__weight_b_{family}"] = tune.uniform(
                        float(self.config.stage_b_family_min_weight),
                        float(self.config.family_weight_max),
                    )
                for summary_name in TRACK_B_SUMMARY_BLOCKS:
                    search_space[f"{sym}__weight_b_{summary_name}"] = tune.uniform(
                        float(self.config.stage_b_family_min_weight),
                        float(self.config.family_weight_max),
                    )
        else:
            for family in HF_BLOCK_FAMILIES:
                search_space[f"weight_b_{family}"] = tune.uniform(
                    float(self.config.stage_b_family_min_weight),
                    float(self.config.family_weight_max),
                )
            for summary_name in TRACK_B_SUMMARY_BLOCKS:
                search_space[f"weight_b_{summary_name}"] = tune.uniform(
                    float(self.config.stage_b_family_min_weight),
                    float(self.config.family_weight_max),
                )

        sequence_model_type = self.config.sequence_model_type

        # WEIGHTS-ONLY MODE: freeze non-weight params (model/threshold/pipeline)
        if weights_only:
            if sequence_model_type == "mamba":
                # Coerce seq_len to allowed set when possible
                try:
                    allowed = list(self._get_effective_mamba_seq_len_choices())
                except Exception:
                    allowed = list(self.config.mamba_seq_len_choices)
                seq_len = int(fixed_params.get("mamba_seq_len", allowed[0] if allowed else 128))
                if allowed and seq_len not in allowed:
                    seq_len = min(allowed, key=lambda v: abs(int(v) - seq_len))

                search_space["mamba_d_model"] = _fixed_choice(int(fixed_params.get("mamba_d_model", self.config.mamba_d_model_choices[0])))
                search_space["mamba_n_layers"] = _fixed_choice(int(fixed_params.get("mamba_n_layers", self.config.mamba_n_layers_choices[0])))
                search_space["mamba_ssm_dim"] = _fixed_choice(int(fixed_params.get("mamba_ssm_dim", self.config.mamba_ssm_dim_choices[0])))
                search_space["mamba_expand_factor"] = _fixed_choice(float(fixed_params.get("mamba_expand_factor", self.config.mamba_expand_factor_choices[0])))
                search_space["mamba_seq_len"] = _fixed_choice(int(seq_len))
                search_space["mamba_activation"] = _fixed_choice(str(fixed_params.get("mamba_activation", self.config.mamba_activation_choices[0])))
                search_space["mamba_norm_type"] = _fixed_choice(str(fixed_params.get("mamba_norm_type", self.config.mamba_norm_type_choices[0])))
                search_space["mamba_norm_strategy"] = _fixed_choice(str(fixed_params.get("mamba_norm_strategy", self.config.mamba_norm_strategy_choices[0])))
                search_space["mamba_dropout"] = _fixed_choice(float(fixed_params.get("mamba_dropout", 0.1)))
                search_space["mamba_resid_dropout"] = _fixed_choice(float(fixed_params.get("mamba_resid_dropout", 0.0)))
                search_space["mamba_ssm_dropout"] = _fixed_choice(float(fixed_params.get("mamba_ssm_dropout", 0.0)))
                search_space["mamba_gate_dropout"] = _fixed_choice(float(fixed_params.get("mamba_gate_dropout", 0.0)))
                search_space["mamba_optimizer"] = _fixed_choice(str(fixed_params.get("mamba_optimizer", "adamw")))
                search_space["mamba_learning_rate"] = _fixed_choice(float(fixed_params.get("mamba_learning_rate", 1e-3)))
                search_space["mamba_weight_decay"] = _fixed_choice(float(fixed_params.get("mamba_weight_decay", 1e-4)))
                search_space["mamba_grad_clip"] = _fixed_choice(float(fixed_params.get("mamba_grad_clip", 1.0)))
                search_space["mamba_lr_scheduler"] = _fixed_choice(str(fixed_params.get("mamba_lr_scheduler", "cosine")))
                search_space["mamba_warmup_steps"] = _fixed_choice(int(fixed_params.get("mamba_warmup_steps", 100)))
                search_space["mamba_max_epochs"] = _fixed_choice(int(fixed_params.get("mamba_max_epochs", 10)))
                search_space["mamba_batch_size"] = _fixed_choice(int(fixed_params.get("mamba_batch_size", 32)))
                search_space["mamba_loss_fn"] = _fixed_choice(str(fixed_params.get("mamba_loss_fn", "smooth_l1")))
                search_space["mamba_head_type"] = _fixed_choice(str(fixed_params.get("mamba_head_type", "linear")))
                search_space["mamba_head_hidden_dim"] = _fixed_choice(int(fixed_params.get("mamba_head_hidden_dim", 128)))
                search_space["mamba_head_num_layers"] = _fixed_choice(int(fixed_params.get("mamba_head_num_layers", 1)))
                search_space["mamba_head_dropout"] = _fixed_choice(float(fixed_params.get("mamba_head_dropout", 0.0)))

            # Threshold/regime params
            search_space["threshold"] = _fixed_choice(float(fixed_params.get("threshold", 0.10)))
            search_space["bull_long_mult"] = _fixed_choice(float(fixed_params.get("bull_long_mult", 1.0)))
            search_space["bull_short_mult"] = _fixed_choice(float(fixed_params.get("bull_short_mult", 1.0)))
            search_space["bear_long_mult"] = _fixed_choice(float(fixed_params.get("bear_long_mult", 1.5)))
            search_space["bear_short_mult"] = _fixed_choice(float(fixed_params.get("bear_short_mult", 0.5)))
            search_space["crisis_long_mult"] = _fixed_choice(float(fixed_params.get("crisis_long_mult", 3.0)))
            search_space["crisis_short_mult"] = _fixed_choice(float(fixed_params.get("crisis_short_mult", 3.0)))
            search_space["conf_threshold"] = _fixed_choice(float(fixed_params.get("conf_threshold", 0.5)))
            search_space["vol_scaler"] = _fixed_choice(float(fixed_params.get("vol_scaler", 1.0)))

            # Pipeline params
            search_space["smoothing_type"] = _fixed_choice(str(fixed_params.get("smoothing_type", "none")))
            search_space["smoothing_window"] = _fixed_choice(int(fixed_params.get("smoothing_window", 3)))
            search_space["train_fraction"] = _fixed_choice(float(fixed_params.get("train_fraction", 0.8)))

            return search_space

        # If we're running a dedicated Mamba optimization job, skip ALL LSTM
        # hyperparameters to keep the search space clean and efficient.
        if sequence_model_type == "mamba":
            # A. Core architecture
            search_space["mamba_d_model"] = tune.choice(list(self.config.mamba_d_model_choices))
            search_space["mamba_n_layers"] = tune.choice(list(self.config.mamba_n_layers_choices))
            search_space["mamba_ssm_dim"] = tune.choice(list(self.config.mamba_ssm_dim_choices))
            search_space["mamba_expand_factor"] = tune.choice(list(self.config.mamba_expand_factor_choices))
            search_space["mamba_seq_len"] = tune.choice(self._get_effective_mamba_seq_len_choices())
            search_space["mamba_activation"] = tune.choice(list(self.config.mamba_activation_choices))

            # B. Normalization & Dropout
            search_space["mamba_norm_type"] = tune.choice(list(self.config.mamba_norm_type_choices))
            search_space["mamba_norm_strategy"] = tune.choice(list(self.config.mamba_norm_strategy_choices))
            search_space["mamba_dropout"] = tune.uniform(self.config.mamba_dropout_min, self.config.mamba_dropout_max)
            search_space["mamba_resid_dropout"] = tune.choice(list(self.config.mamba_resid_dropout_choices))
            search_space["mamba_ssm_dropout"] = tune.choice(list(self.config.mamba_ssm_dropout_choices))
            search_space["mamba_gate_dropout"] = tune.choice(list(self.config.mamba_gate_dropout_choices))

            # C. Training Dynamics
            search_space["mamba_optimizer"] = tune.choice(list(self.config.mamba_optimizer_choices))
            mamba_lr_min, mamba_lr_max = get_bounds(
                "mamba_learning_rate", self.config.mamba_lr_min, self.config.mamba_lr_max
            )
            search_space["mamba_learning_rate"] = tune.loguniform(mamba_lr_min, mamba_lr_max)
            search_space["mamba_weight_decay"] = tune.loguniform(
                self.config.mamba_weight_decay_min, self.config.mamba_weight_decay_max
            )
            search_space["mamba_grad_clip"] = tune.choice(list(self.config.mamba_grad_clip_choices))

            # D. Training Schedule
            search_space["mamba_lr_scheduler"] = tune.choice(list(self.config.mamba_lr_scheduler_choices))
            search_space["mamba_warmup_steps"] = tune.choice(list(self.config.mamba_warmup_steps_choices))
            search_space["mamba_max_epochs"] = tune.choice(list(self.config.mamba_max_epochs_choices))
            search_space["mamba_batch_size"] = tune.choice(list(self.config.mamba_batch_size_choices))

            # E. Loss & Output Structure
            search_space["mamba_loss_fn"] = tune.choice(list(self.config.mamba_loss_fn_choices))
            search_space["mamba_head_type"] = tune.choice(list(self.config.mamba_head_type_choices))
            mamba_head_hidden_choices = list(
                range(
                    self.config.mamba_head_hidden_dim_min,
                    self.config.mamba_head_hidden_dim_max + 1,
                    32,
                )
            )
            if not mamba_head_hidden_choices:
                mamba_head_hidden_choices = [self.config.mamba_head_hidden_dim_min]
            search_space["mamba_head_hidden_dim"] = tune.choice(mamba_head_hidden_choices)
            search_space["mamba_head_num_layers"] = tune.choice(list(self.config.mamba_head_num_layers_choices))
            search_space["mamba_head_dropout"] = tune.uniform(
                self.config.mamba_head_dropout_min, self.config.mamba_head_dropout_max
            )

            # Threshold and regime params (shared)
            thresh_min, thresh_max = get_bounds(
                "threshold",
                self.config.threshold_min,
                self.config.threshold_max,
            )
            search_space["threshold"] = tune.uniform(thresh_min, thresh_max)

            bl_min, bl_max = get_bounds(
                "bull_long_mult", self.config.bull_long_mult_min, self.config.bull_long_mult_max
            )
            search_space["bull_long_mult"] = tune.uniform(bl_min, bl_max)

            bs_min, bs_max = get_bounds(
                "bull_short_mult", self.config.bull_short_mult_min, self.config.bull_short_mult_max
            )
            search_space["bull_short_mult"] = tune.uniform(bs_min, bs_max)

            bel_min, bel_max = get_bounds(
                "bear_long_mult", self.config.bear_long_mult_min, self.config.bear_long_mult_max
            )
            search_space["bear_long_mult"] = tune.uniform(bel_min, bel_max)

            bes_min, bes_max = get_bounds(
                "bear_short_mult", self.config.bear_short_mult_min, self.config.bear_short_mult_max
            )
            search_space["bear_short_mult"] = tune.uniform(bes_min, bes_max)

            cl_min, cl_max = get_bounds(
                "crisis_long_mult", self.config.crisis_long_mult_min, self.config.crisis_long_mult_max
            )
            search_space["crisis_long_mult"] = tune.uniform(cl_min, cl_max)

            cs_min, cs_max = get_bounds(
                "crisis_short_mult", self.config.crisis_short_mult_min, self.config.crisis_short_mult_max
            )
            search_space["crisis_short_mult"] = tune.uniform(cs_min, cs_max)

            ct_min, ct_max = get_bounds(
                "conf_threshold", self.config.conf_threshold_min, self.config.conf_threshold_max
            )
            search_space["conf_threshold"] = tune.uniform(ct_min, ct_max)

            vs_min, vs_max = get_bounds("vol_scaler", self.config.vol_scaler_min, self.config.vol_scaler_max)
            search_space["vol_scaler"] = tune.uniform(vs_min, vs_max)

            # Pipeline-level hyperparameters
            search_space["smoothing_type"] = tune.choice(list(self.config.smoothing_types))
            smoothing_window_choices = list(
                range(self.config.smoothing_window_min, self.config.smoothing_window_max + 1, 3)
            )
            if not smoothing_window_choices:
                smoothing_window_choices = [self.config.smoothing_window_min]
            search_space["smoothing_window"] = tune.choice(smoothing_window_choices)
            search_space["train_fraction"] = tune.uniform(
                self.config.train_fraction_min, self.config.train_fraction_max
            )

            return search_space
        
        # =====================================================================
        # HORIZON-BASED DYNAMIC BOUNDS (same formulas as _suggest_lstm_params)
        # =====================================================================
        import math
        H = self.config.horizon
        
        # seq_len: H * [1.0, 3.0]
        seq_len_min = max(10, int(H * self.config.seq_len_horizon_mult_min))
        seq_len_max = max(seq_len_min + 10, int(H * self.config.seq_len_horizon_mult_max))
        
        # hidden_dim: [32 + 0.8*H, 64 + 2.0*H]
        hidden_min = int(self.config.hidden_base_min + self.config.hidden_horizon_mult_min * H)
        hidden_max = int(self.config.hidden_base_max + self.config.hidden_horizon_mult_max * H)
        
        # dropout_max: min(0.6, 0.05 + 0.004*H)
        dropout_max = min(self.config.dropout_max_cap, 
                          self.config.dropout_base + self.config.dropout_horizon_mult * H)
        
        # num_heads: [log2(H), log2(H) + 4] for attention-based architectures
        num_heads_base = max(2, int(math.log2(max(H, 4))))
        
        # LSTM hyperparameters - use tune.choice for integer ranges for reliability
        lstm_layer_choices = list(range(self.config.lstm_layers_min, self.config.lstm_layers_max + 1))
        search_space["lstm_layers"] = tune.choice(lstm_layer_choices)
        
        # Hidden dim: horizon-based, step by 8 for efficiency
        lstm_hidden_choices = list(range(hidden_min, hidden_max + 1, 8))
        if not lstm_hidden_choices:
            lstm_hidden_choices = [hidden_min]
        search_space["lstm_hidden_dim"] = tune.choice(lstm_hidden_choices)
        
        # Dropout: horizon-based max
        search_space["lstm_dropout"] = tune.uniform(self.config.dropout_base, dropout_max)
        
        # Seq len: horizon-based, step by 4 for efficiency
        lstm_seq_len_choices = list(range(seq_len_min, seq_len_max + 1, 4))
        if not lstm_seq_len_choices:
            lstm_seq_len_choices = [seq_len_min]
        search_space["lstm_seq_len"] = tune.choice(lstm_seq_len_choices)
        
        search_space["lstm_batch_size"] = tune.choice(list(self.config.lstm_batch_sizes))
        
        # LSTM learning rate - use meta-optimizer adapted bounds
        lr_min, lr_max = get_bounds(
            "lstm_learning_rate",
            self.config.lstm_lr_min,
            self.config.lstm_lr_max
        )
        search_space["lstm_learning_rate"] = tune.loguniform(lr_min, lr_max)
        
        search_space["lstm_optimizer"] = tune.choice(list(self.config.lstm_optimizers))
        search_space["lstm_activation"] = tune.choice(list(self.config.lstm_activations))
        # HARD-CODED: lstm_use_amp = True (not in search space)
        
        # =====================================================================
        # LSTM ARCHITECTURE (Global Optuna params)
        # =====================================================================
        search_space["lstm_cell_type"] = tune.choice(list(self.config.lstm_cell_types))
        search_space["lstm_recurrent_kernel_init"] = tune.choice(list(self.config.lstm_recurrent_kernel_inits))
        search_space["lstm_hidden_state_init"] = tune.choice(list(self.config.lstm_hidden_state_inits))
        
        # =====================================================================
        # LSTM REGULARIZATION (Global Optuna params)
        # =====================================================================
        search_space["lstm_recurrent_dropout"] = tune.uniform(
            self.config.lstm_recurrent_dropout_min, self.config.lstm_recurrent_dropout_max
        )
        search_space["lstm_input_dropout"] = tune.uniform(
            self.config.lstm_input_dropout_min, self.config.lstm_input_dropout_max
        )
        search_space["lstm_grad_clip"] = tune.uniform(
            self.config.lstm_grad_clip_min, self.config.lstm_grad_clip_max
        )
        # Time-wise dropout
        search_space["lstm_time_dropout"] = tune.uniform(
            self.config.lstm_time_dropout_min, self.config.lstm_time_dropout_max
        )
        # L2 recurrent weight regularization
        search_space["lstm_recurrent_weight_decay"] = tune.uniform(
            self.config.lstm_recurrent_weight_decay_min, self.config.lstm_recurrent_weight_decay_max
        )
        # NEW: Weight dropout (AWD-LSTM style)
        search_space["lstm_weight_dropout"] = tune.uniform(
            self.config.lstm_weight_dropout_min, self.config.lstm_weight_dropout_max
        )
        # NEW: Zoneout (RNN stabilizer)
        search_space["lstm_zoneout"] = tune.uniform(
            self.config.lstm_zoneout_min, self.config.lstm_zoneout_max
        )
        # NEW: Sequence noise injection
        search_space["lstm_sequence_noise_std"] = tune.uniform(
            self.config.lstm_sequence_noise_min, self.config.lstm_sequence_noise_max
        )
        
        # =====================================================================
        # LSTM ENHANCEMENTS (Global Optuna params)
        # =====================================================================
        search_space["lstm_layer_norm"] = tune.choice(list(self.config.lstm_layer_norm_choices))
        search_space["lstm_residual"] = tune.choice(list(self.config.lstm_residual_choices))
        search_space["lstm_skip_connect"] = tune.choice(list(self.config.lstm_skip_connect_choices))
        # HARD-CODED: lstm_bidirectional = False (not in search space)
        
        # NEW: LR multiplier (layer-wise scaling)
        search_space["lstm_lr_multiplier"] = tune.uniform(
            self.config.lstm_lr_multiplier_min, self.config.lstm_lr_multiplier_max
        )
        
        # =====================================================================
        # LSTM ATTENTION HYPERPARAMETERS (Comprehensive, Global Optuna)
        # =====================================================================
        # 5.1 Attention type selection
        search_space["lstm_attention_type"] = tune.choice(list(self.config.lstm_attention_types))
        
        # 5.2 Attention hidden dimension
        lstm_attn_hidden_choices = list(range(
            self.config.lstm_attn_hidden_dim_min, 
            self.config.lstm_attn_hidden_dim_max + 1, 32
        ))
        if not lstm_attn_hidden_choices:
            lstm_attn_hidden_choices = [self.config.lstm_attn_hidden_dim_min]
        search_space["lstm_attn_hidden_dim"] = tune.choice(lstm_attn_hidden_choices)
        
        # 5.3 Attention dropout
        search_space["lstm_attn_dropout"] = tune.uniform(
            self.config.lstm_attn_dropout_min, self.config.lstm_attn_dropout_max
        )
        
        # 5.4 Attention normalization
        search_space["lstm_attn_normalization"] = tune.choice(list(self.config.lstm_attn_normalization_types))
        
        # 5.5 Number of attention heads (lightweight)
        lstm_attn_heads_choices = list(range(
            self.config.lstm_attn_heads_min, self.config.lstm_attn_heads_max + 1
        ))
        search_space["lstm_attn_heads"] = tune.choice(lstm_attn_heads_choices)
        
        # 5.6 Attention scoring function
        search_space["lstm_attn_score_fn"] = tune.choice(list(self.config.lstm_attn_score_functions))
        
        # 5.7 Context vector merge type
        search_space["lstm_attn_merge"] = tune.choice(list(self.config.lstm_attn_merge_types))
        
        # 5.8 Positional encoding
        search_space["lstm_attn_positional_encoding"] = tune.choice(
            list(self.config.lstm_attn_positional_encoding_choices)
        )
        
        # 5.9 Attention temperature
        search_space["lstm_attn_temperature"] = tune.uniform(
            self.config.lstm_attn_temperature_min, self.config.lstm_attn_temperature_max
        )
        
        # 5.10 Attention regularizers
        search_space["lstm_attn_entropy_reg"] = tune.uniform(
            self.config.lstm_attn_entropy_reg_min, self.config.lstm_attn_entropy_reg_max
        )
        search_space["lstm_attn_distance_reg"] = tune.uniform(
            self.config.lstm_attn_distance_reg_min, self.config.lstm_attn_distance_reg_max
        )
        
        # 5.11 Attention context window (dynamic based on seq_len)
        # We use a fixed range here since Ray Tune doesn't support dynamic dependencies
        search_space["lstm_attn_context_length"] = tune.choice(
            list(range(self.config.lstm_attn_context_length_min, seq_len_max + 1, 5))
        )
        
        # 5.12 Key/Value projection dims for scaled_dot attention
        search_space["lstm_attn_key_dim"] = tune.choice(
            list(range(self.config.lstm_attn_key_dim_min, self.config.lstm_attn_key_dim_max + 1, 16))
        )
        search_space["lstm_attn_value_dim"] = tune.choice(
            list(range(self.config.lstm_attn_value_dim_min, self.config.lstm_attn_value_dim_max + 1, 16))
        )
        
        # =====================================================================
        # LSTM OUTPUT LAYERS (Global Optuna params)
        # =====================================================================
        lstm_fc_layers_choices = list(range(self.config.lstm_fc_layers_min, self.config.lstm_fc_layers_max + 1))
        search_space["lstm_fc_layers"] = tune.choice(lstm_fc_layers_choices)
        lstm_fc_hidden_choices = list(range(self.config.lstm_fc_hidden_min, self.config.lstm_fc_hidden_max + 1, 32))
        if not lstm_fc_hidden_choices:
            lstm_fc_hidden_choices = [self.config.lstm_fc_hidden_min]
        search_space["lstm_fc_hidden"] = tune.choice(lstm_fc_hidden_choices)
        search_space["lstm_output_dropout"] = tune.uniform(
            self.config.lstm_output_dropout_min, self.config.lstm_output_dropout_max
        )
        search_space["lstm_output_activation"] = tune.choice(list(self.config.lstm_output_activations))
        
        # Gradient accumulation
        search_space["lstm_gradient_accumulation"] = tune.choice(
            list(range(self.config.lstm_gradient_accumulation_min, self.config.lstm_gradient_accumulation_max + 1))
        )
        # Momentum (for SGD/RMSprop)
        search_space["lstm_momentum"] = tune.uniform(
            self.config.lstm_momentum_min, self.config.lstm_momentum_max
        )
        
        # =====================================================================
        # 4.3 LOSS FUNCTION CHOICE
        # =====================================================================
        search_space["loss_fn"] = tune.choice(list(self.config.loss_functions))
        search_space["huber_delta"] = tune.uniform(
            self.config.huber_delta_min,
            self.config.huber_delta_max
        )
        search_space["quantile_alpha"] = tune.uniform(
            self.config.quantile_alpha_min,
            self.config.quantile_alpha_max
        )
        
        # =====================================================================
        # 4.8 REGULARIZATION & NOISE INJECTION
        # =====================================================================
        search_space["input_noise_std"] = tune.uniform(
            self.config.input_noise_std_min,
            self.config.input_noise_std_max
        )
        search_space["weight_decay"] = tune.loguniform(
            max(self.config.weight_decay_min, 1e-8),  # Avoid log(0)
            max(self.config.weight_decay_max, 1e-7)
        )
        
        # =====================================================================
        # 4.9 LEARNING RATE SCHEDULER
        # =====================================================================
        search_space["lr_scheduler"] = tune.choice(list(self.config.lr_schedulers))
        search_space["cosine_t_max"] = tune.choice(
            list(range(self.config.cosine_t_max_min, self.config.cosine_t_max_max + 1, 10))
        )
        search_space["plateau_patience"] = tune.choice(
            list(range(self.config.plateau_patience_min, self.config.plateau_patience_max + 1, 2))
        )
        
        # =====================================================================
        # 4.10 TRAINING CONTROL (Critical for preventing overtraining)
        # =====================================================================
        search_space["early_stopping_patience"] = tune.choice(
            list(range(self.config.early_stopping_patience_min, self.config.early_stopping_patience_max + 1, 5))
        )
        search_space["max_epochs"] = tune.choice(
            list(range(self.config.max_epochs_min, self.config.max_epochs_max + 1, 50))
        )
        search_space["warmup_steps"] = tune.choice(
            list(range(self.config.warmup_steps_min, self.config.warmup_steps_max + 1, 100))
        )
        
        # =====================================================================
        # CNN FRONTEND - Dilated Residual CNN for local pattern extraction
        # Used at Jane Street, Jump, Two Sigma for capturing local patterns
        # =====================================================================
        search_space["cnn_frontend_enabled"] = tune.choice(list(self.config.cnn_frontend_enabled))
        search_space["cnn_blocks"] = tune.choice(
            list(range(self.config.cnn_blocks_min, self.config.cnn_blocks_max + 1))
        )
        search_space["cnn_filters"] = tune.choice(list(self.config.cnn_filters))
        search_space["cnn_kernel_size"] = tune.choice(list(self.config.cnn_kernel_sizes))
        search_space["cnn_dilation"] = tune.choice(list(self.config.cnn_dilations))
        search_space["cnn_pooling"] = tune.choice(list(self.config.cnn_pooling))
        search_space["cnn_stride"] = tune.choice(list(self.config.cnn_strides))
        search_space["cnn_activation"] = tune.choice(list(self.config.cnn_activations))
        search_space["cnn_dropout"] = tune.uniform(
            self.config.cnn_dropout_min, self.config.cnn_dropout_max
        )
        search_space["cnn_batch_norm"] = tune.choice(list(self.config.cnn_batch_norm_choices))
        search_space["cnn_layer_norm"] = tune.choice(list(self.config.cnn_layer_norm_choices))
        search_space["cnn_residual"] = tune.choice(list(self.config.cnn_residual_choices))
        
        if sequence_model_type == "mamba":
            # =====================================================================
            # MAMBA SEQUENCE MODEL (Structured State Space Model)
            # =====================================================================
            # Mamba is an alternative to LSTM with O(n) complexity for long sequences.
            # These params are only added when running a dedicated Mamba job.

            # A. Core Mamba Architecture
            search_space["mamba_d_model"] = tune.choice(list(self.config.mamba_d_model_choices))
            search_space["mamba_n_layers"] = tune.choice(list(self.config.mamba_n_layers_choices))
            search_space["mamba_ssm_dim"] = tune.choice(list(self.config.mamba_ssm_dim_choices))
            search_space["mamba_expand_factor"] = tune.choice(list(self.config.mamba_expand_factor_choices))
            search_space["mamba_seq_len"] = tune.choice(self._get_effective_mamba_seq_len_choices())
            search_space["mamba_activation"] = tune.choice(list(self.config.mamba_activation_choices))

            # B. Normalization & Dropout
            search_space["mamba_norm_type"] = tune.choice(list(self.config.mamba_norm_type_choices))
            search_space["mamba_norm_strategy"] = tune.choice(list(self.config.mamba_norm_strategy_choices))
            search_space["mamba_dropout"] = tune.uniform(
                self.config.mamba_dropout_min, self.config.mamba_dropout_max
            )
            search_space["mamba_resid_dropout"] = tune.choice(list(self.config.mamba_resid_dropout_choices))
            search_space["mamba_ssm_dropout"] = tune.choice(list(self.config.mamba_ssm_dropout_choices))
            search_space["mamba_gate_dropout"] = tune.choice(list(self.config.mamba_gate_dropout_choices))

            # C. Training Dynamics
            search_space["mamba_optimizer"] = tune.choice(list(self.config.mamba_optimizer_choices))
            mamba_lr_min, mamba_lr_max = get_bounds(
                "mamba_learning_rate", self.config.mamba_lr_min, self.config.mamba_lr_max
            )
            search_space["mamba_learning_rate"] = tune.loguniform(mamba_lr_min, mamba_lr_max)
            search_space["mamba_weight_decay"] = tune.loguniform(
                self.config.mamba_weight_decay_min, self.config.mamba_weight_decay_max
            )
            search_space["mamba_grad_clip"] = tune.choice(list(self.config.mamba_grad_clip_choices))

            # D. Training Schedule
            search_space["mamba_lr_scheduler"] = tune.choice(list(self.config.mamba_lr_scheduler_choices))
            search_space["mamba_warmup_steps"] = tune.choice(list(self.config.mamba_warmup_steps_choices))
            search_space["mamba_max_epochs"] = tune.choice(list(self.config.mamba_max_epochs_choices))
            search_space["mamba_batch_size"] = tune.choice(list(self.config.mamba_batch_size_choices))

            # E. Loss & Output Structure
            search_space["mamba_loss_fn"] = tune.choice(list(self.config.mamba_loss_fn_choices))
            search_space["mamba_head_type"] = tune.choice(list(self.config.mamba_head_type_choices))
            # MLP head config (used when head_type="mlp")
            mamba_head_hidden_choices = list(range(
                self.config.mamba_head_hidden_dim_min,
                self.config.mamba_head_hidden_dim_max + 1,
                32
            ))
            if not mamba_head_hidden_choices:
                mamba_head_hidden_choices = [self.config.mamba_head_hidden_dim_min]
            search_space["mamba_head_hidden_dim"] = tune.choice(mamba_head_hidden_choices)
            search_space["mamba_head_num_layers"] = tune.choice(list(self.config.mamba_head_num_layers_choices))
            search_space["mamba_head_dropout"] = tune.uniform(
                self.config.mamba_head_dropout_min, self.config.mamba_head_dropout_max
            )
        
        # F. Threshold & Signal Parameters (shared with LSTM - already defined below)
        
        # Threshold and regime params - use meta-optimizer adapted bounds
        thresh_min, thresh_max = get_bounds(
            "threshold",
            self.config.threshold_min,
            self.config.threshold_max
        )
        search_space["threshold"] = tune.uniform(thresh_min, thresh_max)
        
        bl_min, bl_max = get_bounds("bull_long_mult", self.config.bull_long_mult_min, self.config.bull_long_mult_max)
        search_space["bull_long_mult"] = tune.uniform(bl_min, bl_max)
        
        bs_min, bs_max = get_bounds("bull_short_mult", self.config.bull_short_mult_min, self.config.bull_short_mult_max)
        search_space["bull_short_mult"] = tune.uniform(bs_min, bs_max)
        
        bel_min, bel_max = get_bounds("bear_long_mult", self.config.bear_long_mult_min, self.config.bear_long_mult_max)
        search_space["bear_long_mult"] = tune.uniform(bel_min, bel_max)
        
        bes_min, bes_max = get_bounds("bear_short_mult", self.config.bear_short_mult_min, self.config.bear_short_mult_max)
        search_space["bear_short_mult"] = tune.uniform(bes_min, bes_max)
        
        cl_min, cl_max = get_bounds("crisis_long_mult", self.config.crisis_long_mult_min, self.config.crisis_long_mult_max)
        search_space["crisis_long_mult"] = tune.uniform(cl_min, cl_max)
        
        cs_min, cs_max = get_bounds("crisis_short_mult", self.config.crisis_short_mult_min, self.config.crisis_short_mult_max)
        search_space["crisis_short_mult"] = tune.uniform(cs_min, cs_max)
        
        ct_min, ct_max = get_bounds("conf_threshold", self.config.conf_threshold_min, self.config.conf_threshold_max)
        search_space["conf_threshold"] = tune.uniform(ct_min, ct_max)
        
        vs_min, vs_max = get_bounds("vol_scaler", self.config.vol_scaler_min, self.config.vol_scaler_max)
        search_space["vol_scaler"] = tune.uniform(vs_min, vs_max)
        
        # =====================================================================
        # PIPELINE-LEVEL HYPERPARAMETERS (preprocessing & data splitting)
        # =====================================================================
        search_space["smoothing_type"] = tune.choice(list(self.config.smoothing_types))
        smoothing_window_choices = list(range(
            self.config.smoothing_window_min,
            self.config.smoothing_window_max + 1,
            3  # Step by 3
        ))
        if not smoothing_window_choices:
            smoothing_window_choices = [self.config.smoothing_window_min]
        search_space["smoothing_window"] = tune.choice(smoothing_window_choices)
        search_space["train_fraction"] = tune.uniform(
            self.config.train_fraction_min, self.config.train_fraction_max
        )
        
        return search_space

    def optimize_with_ray_tune(
        self,
        panel: pd.DataFrame,
        column_families: Dict[str, str],
        labels: pd.DataFrame,
        block_summaries: Dict[str, pd.DataFrame],
        walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
        horizon: int = 1,
        stage_a_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizedParams:
        """Run optimization using Ray Tune for trial-level parallelism.
        
        This method uses Ray Tune to run multiple trials in parallel, with each
        trial running fold-level parallelism via Ray tasks. This enables true
        parallel trial execution without the joblib issues of Optuna's n_jobs.
        
        Architecture:
            Ray Tune (N concurrent trials)
              → Trial Worker 1: M fold tasks via Ray
              → Trial Worker 2: M fold tasks via Ray
              → ...
            = N×M concurrent fold tasks (limited by GPU resources)
        
        Args:
            panel: Full feature panel
            column_families: Column to family mapping
            labels: Labels DataFrame with 'forward_return'
            block_summaries: Stage-B block summaries
            walk_forward_folds: List of (train_idx, val_idx) for ALL WF windows
            horizon: Forecast horizon
            stage_a_weights: Optional Stage-A family weights
        
        Returns:
            OptimizedParams with best configuration
        """
        if not RAY_AVAILABLE or tune is None:
            self.logger.error("Ray Tune not available, falling back to standard optimize")
            return self.optimize(
                panel, column_families, labels, block_summaries,
                walk_forward_folds, horizon, stage_a_weights
            )
        
        n_folds = len(walk_forward_folds)
        self.logger.info(
            f"Ray Tune optimization: {n_folds} folds, {self.config.n_trials} trials, "
            f"{self.config.ray_tune_concurrent_trials} concurrent"
        )

        # Cache fold-derived caps so the Ray Tune search space is globally valid.
        self.walk_forward_folds = walk_forward_folds
        self._mamba_seq_len_cap_for_folds = self._compute_mamba_seq_len_cap_for_folds(walk_forward_folds)
        
        # Analyze family structure
        self._analyze_families(panel, column_families, stage_a_weights)
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            init_kwargs: Dict[str, Any] = {}
            init_kwargs = _apply_ray_runtime_env_envvars(init_kwargs)
            ray.init(ignore_reinit_error=True, log_to_driver=False, **init_kwargs)
        
        # Put shared data in Ray object store
        panel_ref = ray.put(panel)
        column_families_ref = ray.put(column_families)
        labels_ref = ray.put(labels)
        block_summaries_ref = ray.put(block_summaries)
        walk_forward_folds_ref = ray.put(walk_forward_folds)
        
        # Capture optimizer state for the trainable
        family_columns = self.family_columns.copy()
        family_sizes = self.family_sizes.copy()
        stage_a_weights_copy = (stage_a_weights or {}).copy()
        config_dict = {
            "ray_fold_parallelism": self.config.ray_fold_parallelism,
            "ray_fold_gpu_fraction": self.config.ray_fold_gpu_fraction,
            "fold_gpu_reserve_gb": self.config.fold_gpu_reserve_gb,
            "max_total_dims": self.config.max_total_dims,
            "sharpe_weight": self.config.sharpe_weight,
            "rwa_weight": self.config.rwa_weight,
            "stability_weight": self.config.stability_weight,
        }
        
        # Build search space
        search_space = self._build_ray_tune_search_space()
        
        # =========================================================================
        # OBJECT STORE APPROACH: True parallel trainables fetching from object store
        # Each trainable runs independently with its own GPU slice (80GB / 14 = ~5.7GB)
        # Data is fetched zero-copy from Ray object store
        # =========================================================================
        concurrent_trials = self.config.ray_tune_concurrent_trials
        # Ensure GPU allocation is consistent with requested concurrency (e.g. 4 trials -> 0.25 GPU each).
        # Users can still request smaller slices via ray_gpus_per_trial.
        gpu_per_trial = min(float(self.config.ray_gpus_per_trial), 1.0 / max(1, int(concurrent_trials)))

        # CPU allocation: auto-split across trials unless explicitly specified.
        try:
            import os as _os

            _total_cpus = 0
            try:
                _total_cpus = int(ray.cluster_resources().get("CPU", 0))
            except Exception:
                _total_cpus = 0
            if _total_cpus <= 0:
                _total_cpus = int(_os.cpu_count() or 1)

            _reserve_cpus = max(1, min(4, int(_total_cpus * 0.05)))
            if int(self.config.ray_cpus_per_trial) > 0:
                cpu_per_trial = float(int(self.config.ray_cpus_per_trial))
            else:
                _auto_cpu = max(1, int((_total_cpus - _reserve_cpus) // max(1, int(concurrent_trials))))
                # Heuristic cap: for single-GPU training, allocating almost all CPUs can hurt overall
                # system responsiveness and often doesn't improve throughput proportionally.
                # Keep this conservative; users can override by setting ray_cpus_per_trial > 0.
                _cap_cpu = max(4, min(16, _auto_cpu))
                cpu_per_trial = float(min(_auto_cpu, _cap_cpu))
        except Exception:
            cpu_per_trial = float(max(1, int(getattr(self.config, "ray_cpus_per_trial", 8) or 8)))

        est_vram_gb = gpu_per_trial * float(self.config.fold_gpu_memory_gb)
        
        self.logger.info(
            "🚀 Ray Tune resources: %d concurrent trials, cpu_per_trial=%.1f, gpu_per_trial=%.3f (~%.1f GB each)",
            concurrent_trials,
            cpu_per_trial,
            gpu_per_trial,
            est_vram_gb,
        )
        
        # Capture refs for closure (these are ObjectRefs, not data)
        _panel_ref = panel_ref
        _column_families_ref = column_families_ref
        _labels_ref = labels_ref
        _block_summaries_ref = block_summaries_ref
        _walk_forward_folds_ref = walk_forward_folds_ref
        _family_columns = family_columns
        _family_sizes = family_sizes
        _stage_a_weights = stage_a_weights_copy
        _config_dict = config_dict
        _horizon = horizon
        
        def make_trainable():
            """Factory to create trainable that fetches data from object store."""
            
            def trainable(config: Dict[str, Any]) -> Dict[str, Any]:
                """Ray Tune trainable with direct object store access.
                
                Each trainable:
                1. Fetches data from Ray object store (zero-copy on same node)
                2. Creates optimizer instance (imports cached by Ray worker reuse)
                3. Evaluates trial on its GPU slice
                4. Reports metrics to Ray Tune
                """
                import sys
                # Best-effort VRAM partitioning: when using fractional GPUs, cap the PyTorch
                # allocator to roughly the same fraction to reduce cross-trial OOM risk.
                try:
                    import torch as _torch

                    if _torch.cuda.is_available() and gpu_per_trial < 1.0:
                        _torch.cuda.set_per_process_memory_fraction(
                            float(max(0.05, min(0.95, gpu_per_trial))),
                            device=0,
                        )
                except Exception:
                    pass
                import time as _time
                import logging
                
                _t_start = _time.time()
                logger = logging.getLogger("stage_b.optuna.trainable")

                try:
                    _ray_trial_id = tune.get_trial_id() if tune is not None else None
                except Exception:
                    _ray_trial_id = None
                _optuna_trial_number = config.get("_optuna_trial_number")
                if _optuna_trial_number is not None:
                    logger.info(
                        "Ray trial started: ray_trial_id=%s optuna_trial_number=%s",
                        _ray_trial_id,
                        _optuna_trial_number,
                    )
                
                try:
                    # Fetch data from object store (zero-copy plasma fetch)
                    panel_data = ray.get(_panel_ref)
                    column_families_data = ray.get(_column_families_ref)
                    labels_data = ray.get(_labels_ref)
                    block_summaries_data = ray.get(_block_summaries_ref)
                    walk_forward_folds_data = ray.get(_walk_forward_folds_ref)
                    
                    fetch_time = _time.time() - _t_start
                    print(f"[TRACE:trainable] Data fetched from object store in {fetch_time:.2f}s", file=sys.stderr, flush=True)
                    
                    # Create optimizer instance (heavy imports cached by Ray worker reuse)
                    trial_config = OptunaConfig(
                        ray_fold_parallelism=0,  # Sequential folds within trial
                        ray_fold_gpu_fraction=0.0,
                        fold_gpu_reserve_gb=_config_dict["fold_gpu_reserve_gb"],
                        max_total_dims=_config_dict["max_total_dims"],
                        sharpe_weight=_config_dict["sharpe_weight"],
                        rwa_weight=_config_dict["rwa_weight"],
                        stability_weight=_config_dict["stability_weight"],
                    )
                    optimizer = StageBOptunaOptimizer(config=trial_config, logger=logger)
                    optimizer.family_columns = _family_columns
                    optimizer.family_sizes = _family_sizes
                    optimizer.stage_a_weights = _stage_a_weights
                    
                    # Build family_params from config
                    family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
                    for family in STAGE_A_FAMILIES:
                        weight_key = f"weight_{family}"
                        dim_type_key = f"{family}_dim_type"
                        weight = config.get(weight_key, 0.0)
                        include = weight >= optimizer.config.family_weight_clip_min
                        method = config.get(dim_type_key, "pca")
                        
                        if method == "ae":
                            dim_key = f"{family}_ae_dim"
                        else:
                            dim_key = f"{family}_pca_dim"
                        dim = config.get(dim_key, 4) if include else 0
                        
                        ae_params: Dict[str, Any] = {}
                        if method == "ae":
                            ae_params = {
                                "ae_layers": config.get(f"{family}_ae_layers", 2),
                                "ae_activation": config.get(f"{family}_ae_activation", "relu"),
                                "ae_dropout": config.get(f"{family}_ae_dropout", 0.0),
                                "ae_lr": config.get(f"{family}_ae_lr", 1e-3),
                            }
                        
                        family_params[family] = (include, dim, method, weight if include else 0.0, ae_params)

                    # Track-B per-family weights (separate category)
                    track_b_family_weights: Dict[str, float] = {}
                    for family in HF_BLOCK_FAMILIES:
                        track_b_family_weights[family] = float(config.get(f"weight_b_{family}", 1.0))
                    for summary_name in TRACK_B_SUMMARY_BLOCKS:
                        track_b_family_weights[summary_name] = float(config.get(f"weight_b_{summary_name}", 1.0))
                    
                    sequence_model_type = optimizer.config.sequence_model_type

                    if sequence_model_type == "lstm":
                        lstm_params = {
                            "lstm_layers": config.get("lstm_layers", 2),
                            "lstm_hidden_dim": config.get("lstm_hidden_dim", 96),
                            "lstm_dropout": config.get("lstm_dropout", 0.2),
                            "lstm_seq_len": config.get("lstm_seq_len", 30),
                            "lstm_batch_size": config.get("lstm_batch_size", 64),
                            "lstm_learning_rate": config.get("lstm_learning_rate", 1e-3),
                            "lstm_optimizer": config.get("lstm_optimizer", "AdamW"),
                            "lstm_activation": config.get("lstm_activation", "relu"),
                            "lstm_use_amp": config.get("lstm_use_amp", True),
                        }
                        mamba_params: Dict[str, Any] = {}
                    elif sequence_model_type == "mamba":
                        mamba_params = {
                            "mamba_d_model": config.get("mamba_d_model"),
                            "mamba_n_layers": config.get("mamba_n_layers"),
                            "mamba_ssm_dim": config.get("mamba_ssm_dim"),
                            "mamba_expand_factor": config.get("mamba_expand_factor"),
                            "mamba_seq_len": config.get("mamba_seq_len"),
                            "mamba_activation": config.get("mamba_activation", "silu"),
                            "mamba_norm_type": config.get("mamba_norm_type", "rmsnorm"),
                            "mamba_norm_strategy": config.get("mamba_norm_strategy", "pre"),
                            "mamba_dropout": config.get("mamba_dropout", 0.1),
                            "mamba_resid_dropout": config.get("mamba_resid_dropout", 0.0),
                            "mamba_ssm_dropout": config.get("mamba_ssm_dropout", 0.0),
                            "mamba_gate_dropout": config.get("mamba_gate_dropout", 0.0),
                            "mamba_optimizer": config.get("mamba_optimizer", "adamw"),
                            "mamba_learning_rate": config.get("mamba_learning_rate", 1e-3),
                            "mamba_weight_decay": config.get("mamba_weight_decay", 1e-4),
                            "mamba_grad_clip": config.get("mamba_grad_clip", 1.0),
                            "mamba_lr_scheduler": config.get("mamba_lr_scheduler", "cosine"),
                            "mamba_warmup_steps": config.get("mamba_warmup_steps", 100),
                            "mamba_max_epochs": config.get("mamba_max_epochs", 10),
                            "mamba_batch_size": config.get("mamba_batch_size", 32),
                            "mamba_loss_fn": config.get("mamba_loss_fn", "smooth_l1"),
                            "mamba_head_type": config.get("mamba_head_type", "linear"),
                            "mamba_head_hidden_dim": config.get("mamba_head_hidden_dim", 128),
                            "mamba_head_num_layers": config.get("mamba_head_num_layers", 1),
                            "mamba_head_dropout": config.get("mamba_head_dropout", 0.0),
                        }
                        lstm_params = {}
                    else:
                        raise ValueError(f"Unknown sequence_model_type: {sequence_model_type}")
                    
                    # Build other params
                    threshold_params = {
                        "threshold": config.get("threshold", 0.10),
                        "bull_long_mult": config.get("bull_long_mult", 1.0),
                        "bull_short_mult": config.get("bull_short_mult", 1.0),
                        "bear_long_mult": config.get("bear_long_mult", 1.5),
                        "bear_short_mult": config.get("bear_short_mult", 0.5),
                        "crisis_long_mult": config.get("crisis_long_mult", 3.0),
                        "crisis_short_mult": config.get("crisis_short_mult", 3.0),
                        "conf_threshold": config.get("conf_threshold", 0.5),
                        "vol_scaler": config.get("vol_scaler", 1.0),
                    }
                    pipeline_params = {
                        "smoothing_type": config.get("smoothing_type", "none"),
                        "smoothing_window": config.get("smoothing_window", 3),
                        "train_fraction": config.get("train_fraction", 0.8),
                    }
                    
                    # Evaluate trial (this runs on GPU slice)
                    def _ray_progress_callback(step: int, intermediate_score: float) -> bool:
                        # Report frequently so ASHA has intermediate signals to prune on.
                        # Return value is ignored by the caller (Ray Tune handles stopping externally).
                        tune.report(
                            {
                                "score": float(intermediate_score),
                                "avg_score": float(intermediate_score),
                                "n_successful_folds": int(step),
                                "folds_done": int(step),
                                "done": False,
                            }
                        )
                        return False

                    score, metadata = optimizer._evaluate_trial_params(
                        panel_data,
                        column_families_data,
                        labels_data,
                        block_summaries_data,
                        walk_forward_folds_data,
                        family_params,
                        config.get("track_a_weight", 1.0),
                        config.get("track_b_weight", 1.0),
                        sequence_model_type,
                        lstm_params,
                        mamba_params,
                        threshold_params,
                        pipeline_params,
                        _horizon,
                        track_b_family_weights=track_b_family_weights,
                        progress_callback=_ray_progress_callback,
                    )
                    
                    elapsed = _time.time() - _t_start
                    print(f"[TRACE:trainable] Trial completed: score={score:.4f} in {elapsed:.1f}s", file=sys.stderr, flush=True)
                    
                    final_metrics = {
                        "score": score,
                        "avg_score": metadata.get("avg_score", score),
                        "score_std": metadata.get("score_std", 0.0),
                        "n_successful_folds": metadata.get("n_successful_folds", 0),
                        "track_a_dims": metadata.get("track_a_dims", 0),
                        "track_b_dims": metadata.get("track_b_dims", 0),
                        "track_c_dims": metadata.get("track_c_dims", 0),
                        "elapsed": elapsed,
                        "done": True,
                    }
                    
                    tune.report(final_metrics)
                    return final_metrics
                    
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    print(f"[ERROR:trainable] {e}\n{tb_str}", file=sys.stderr, flush=True)
                    tune.report({"score": float("-inf"), "done": True})
                    return {"score": float("-inf")}
            
            return trainable
        
        # Resource allocation: each trainable gets its GPU slice
        resources_per_trial = {"cpu": float(cpu_per_trial), "gpu": float(gpu_per_trial)}
        
        # ASHA scheduler for early stopping of poor trials
        # grace_period: minimum iterations before pruning (late enough to not prune slow-learners)
        # max_t: maximum iterations per trial
        # reduction_factor: keep top 1/3 of trials at each rung
        n_folds = len(walk_forward_folds)
        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max",
            max_t=max(1, int(n_folds)),           # Maximum folds per trial
            grace_period=max(3, int(max(1, n_folds) // 10)),  # Don't prune too early
            reduction_factor=3, # Keep top 1/3 at each rung
        )
        
        self.logger.info(
            "🚀 Ray Tune: %d trials, %d concurrent, gpu_per_trial=%.3f (~%.1f GB each), ASHA enabled",
            self.config.n_trials,
            concurrent_trials,
            gpu_per_trial,
            est_vram_gb,
        )
        
        # Use OptunaSearch with TPE sampler for intelligent hyperparameter suggestions
        # This integrates Optuna's Bayesian optimization with Ray Tune's parallelism
        # PLUS meta-optimizer learned priors as initial evaluation points
        optuna_search = None
        if OptunaSearch is not None:
            # 🧠 PERSISTENT MEMORY: Create SQLite-backed Optuna study
            # This allows TPE to remember ALL previous trials across runs!
            # IMPORTANT: Optuna studies cannot change categorical choice sets over time.
            # If we reuse the same study_name after changing the search space (or even just
            # changing categorical ordering), Optuna will raise:
            #   ValueError: CategoricalDistribution does not support dynamic value space.
            # To keep persistence *and* avoid crashes, fingerprint the effective search space.
            import hashlib
            import json as _json

            try:
                _families_sig = sorted(list(family_sizes.keys()))
            except Exception:
                _families_sig = []

            _space_sig = {
                "symbol": str(getattr(self.config, "symbol", "") or ""),
                "horizon": int(horizon),
                "sequence_model_type": str(getattr(self.config, "sequence_model_type", "mamba")),
                "tune_weights_only": bool(getattr(self.config, "tune_weights_only", False)),
                "fixed_params": dict(getattr(self.config, "fixed_params", {}) or {}),
                "use_three_pillar_dims": bool(getattr(self.config, "use_three_pillar_dims", True)),
                "three_pillar_dim_min": int(getattr(self.config, "three_pillar_dim_min", 4)),
                "three_pillar_dim_max": int(getattr(self.config, "three_pillar_dim_max", 32)),
                "three_pillar_pca_variance": float(getattr(self.config, "three_pillar_pca_variance", 0.95)),
                "max_total_dims": int(getattr(self.config, "max_total_dims", 500)),
                "loss_functions": list(getattr(self.config, "loss_functions", ("mse", "huber", "quantile", "nll_gauss"))),
                "mamba_seq_len_choices": list(self._get_effective_mamba_seq_len_choices()),
                "mamba_d_model_choices": list(getattr(self.config, "mamba_d_model_choices", (96, 128, 160, 192, 224, 256, 288))),
                "mamba_n_layers_choices": list(getattr(self.config, "mamba_n_layers_choices", (3, 4, 5, 6, 8))),
                "mamba_ssm_dim_choices": list(getattr(self.config, "mamba_ssm_dim_choices", (64, 96, 128, 160))),
                "mamba_expand_factor_choices": list(getattr(self.config, "mamba_expand_factor_choices", (2.0, 2.5, 3.0, 4.0))),
                "mamba_loss_fn_choices": list(getattr(self.config, "mamba_loss_fn_choices", ("smooth_l1", "huber", "bce_logits", "mse"))),
                "mamba_head_type_choices": list(getattr(self.config, "mamba_head_type_choices", ("linear", "mlp"))),
                "families": _families_sig,
            }
            override_name = (
                getattr(self.config, "study_name_override", None)
                or os.environ.get("STAGE_B_OPTUNA_STUDY_NAME")
            )

            # Per-symbol mode safety: prevent accidental cross-symbol study reuse.
            # If an override name is provided, namespace it by symbol.
            _sym = str(getattr(self.config, "symbol", "") or "").upper()
            if override_name and _sym:
                override_name = f"{override_name}_{_sym}"
            if override_name:
                study_name = str(override_name)
            else:
                _space_fingerprint = hashlib.sha256(
                    _json.dumps(_space_sig, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:10]
                _sym2 = _space_sig.get("symbol") or ""
                if _sym2:
                    study_name = f"stage_b_{_sym2.upper()}_{_space_sig['sequence_model_type']}_h{horizon}_{_space_fingerprint}"
                else:
                    study_name = f"stage_b_{_space_sig['sequence_model_type']}_h{horizon}_{_space_fingerprint}"
            storage_path = Path("artifacts/optuna_studies")
            storage_path.mkdir(parents=True, exist_ok=True)
            # Per-symbol mode: keep one sqlite file per (model_type, horizon) and
            # create separate Optuna *studies* inside it per symbol/config.
            # This prevents cross-symbol mixing while avoiding a pile of DB files.
            storage_db = f"stage_b_{_space_sig['sequence_model_type']}_h{horizon}_per_symbol.db"
            storage_url = f"sqlite:///{storage_path / storage_db}"
            
            # Create RDBStorage object for OptunaSearch (required by Ray Tune API)
            optuna_storage = optuna.storages.RDBStorage(url=storage_url)
            
            # Create TPE sampler with our custom configuration
            tpe_sampler = optuna.samplers.TPESampler(
                seed=self.config.seed,
                n_startup_trials=20,
                multivariate=True,
                group=True,
                consider_prior=True,
                consider_magic_clip=True,
                consider_endpoints=True,
                n_ei_candidates=64,
                gamma=_tpe_custom_gamma,
            )
            
            # Pre-check if study exists to log properly
            try:
                existing_study = optuna.load_study(study_name=study_name, storage=optuna_storage)

                # Validate symbol binding to prevent cross mixing inside the shared DB.
                if _sym:
                    bound_sym = str(existing_study.user_attrs.get("symbol", "") or "").upper()
                    if bound_sym and bound_sym != _sym:
                        raise ValueError(
                            f"Optuna study '{study_name}' is bound to symbol '{bound_sym}', not '{_sym}'."
                        )
                    if not bound_sym:
                        existing_study.set_user_attr("symbol", _sym)

                n_previous = len(existing_study.trials)
                if n_previous > 0:
                    best_prev = existing_study.best_value if existing_study.best_trial else None
                    if best_prev is not None:
                        self.logger.info(
                            f"🧠 MEMORY LOADED: {n_previous} previous trials from {storage_url}, "
                            f"best score so far: {best_prev:.4f}"
                        )
                    else:
                        self.logger.info(f"🧠 MEMORY LOADED: {n_previous} previous trials")
                del existing_study
            except Exception:
                # Ensure the study exists and is tagged before Ray Tune starts sampling.
                try:
                    created = optuna.create_study(
                        study_name=study_name,
                        storage=optuna_storage,
                        direction="maximize",
                        load_if_exists=False,
                    )
                    if _sym:
                        created.set_user_attr("symbol", _sym)
                    del created
                except Exception:
                    # If creation fails due to a race or already-exists, OptunaSearch will handle it.
                    pass

                self.logger.info(f"🧠 New study '{study_name}' will be created in {storage_url}")
            
            # Get meta-optimizer suggestions as initial points for TPE
            points_to_evaluate = None
            if self._meta_optimizer_enabled and self.meta_optimizer is not None:
                suggestions = self.meta_optimizer.get_suggestions()
                if suggestions:
                    # Convert meta-optimizer suggestions to OptunaSearch format
                    # These are the elite parameter values that performed well
                    points_to_evaluate = [suggestions]  # Start with learned good config
                    self.logger.info(
                        "🧠 Meta-optimizer providing %d suggested priors to TPE",
                        len(suggestions)
                    )
            
            # Use study_name and storage object (RDBStorage instance)
            # Ray Tune's OptunaSearch requires a BaseStorage instance, not a URL string
            optuna_search = StageBOptunaSearch(
                metric="score",
                mode="max",
                points_to_evaluate=points_to_evaluate,  # Meta-optimizer learned priors
                sampler=tpe_sampler,
                study_name=study_name,
                storage=optuna_storage,  # Must be BaseStorage instance
                seed=self.config.seed,
            )
            if points_to_evaluate:
                self.logger.info("📊 Using OptunaSearch with TPE + meta-optimizer priors")
            else:
                self.logger.info("📊 Using OptunaSearch with TPE sampler (no meta-optimizer priors yet)")
        else:
            self.logger.warning("OptunaSearch not available, using random sampling")
        
        def _process_tune_results(_results) -> OptimizedParams:
            # Record ALL trial results to meta-optimizer for learning
            if self._meta_optimizer_enabled and self.meta_optimizer is not None:
                all_results = _results.get_dataframe()
                if all_results is not None and len(all_results) > 0:
                    recorded_count = 0
                    for idx, row in all_results.iterrows():
                        try:
                            score = row.get("score", 0.0)
                            if pd.isna(score) or score <= 0:
                                continue

                            config_cols = [c for c in all_results.columns if c.startswith("config/")]
                            params = {}
                            for col in config_cols:
                                param_name = col.replace("config/", "")
                                val = row[col]
                                if not pd.isna(val):
                                    params[param_name] = val

                            if not params:
                                continue

                            self.meta_optimizer.record_trial(
                                trial_id=idx,
                                params=params,
                                sharpe=score,  # Use score as proxy for sharpe
                                stability=0.5,
                                coverage=1.0,
                                hitrate=0.5,
                                final_score=score,
                                fold_count=self.n_folds,
                                elapsed_time=row.get("time_total_s", 0.0),
                            )
                            recorded_count += 1
                        except Exception as e:
                            self.logger.debug(f"Failed to record trial {idx}: {e}")

                    if recorded_count > 0:
                        self.logger.info(f"🧠 Meta-optimizer recorded {recorded_count} trials from Ray Tune")
                        self.meta_optimizer.save()

            best_result = _results.get_best_result(metric="score", mode="max")
            if best_result is None or best_result.config is None:
                self.logger.warning("Ray Tune failed to find best result")
                return OptimizedParams()

            best_score = best_result.metrics.get("score", float("-inf"))
            self.logger.info("🏆 Ray Tune best score: %.4f", best_score)
            self.best_params = self._extract_params_from_dict(best_result.config, best_score)
            return self.best_params

        def _run_tuner(_optuna_search):
            _tuner = tune.Tuner(
                tune.with_resources(make_trainable(), resources=resources_per_trial),
                tune_config=tune.TuneConfig(
                    num_samples=self.config.n_trials,
                    max_concurrent_trials=self.config.ray_tune_concurrent_trials,
                    scheduler=asha_scheduler,
                    search_alg=_optuna_search,
                ),
                param_space=search_space,
            )
            return _tuner.fit()

        try:
            results = _run_tuner(optuna_search)
            return _process_tune_results(results)

        except Exception as e:
            import traceback

            msg = str(e)
            is_dynamic_space = (
                "CategoricalDistribution does not support dynamic value space" in msg
                or "dynamic value space" in msg
            )
            if is_dynamic_space and optuna_search is not None:
                # This typically happens when reusing an existing Optuna study whose
                # categorical choices no longer match the current search space.
                if bool(getattr(self.config, "study_name_override", None)) or bool(
                    getattr(self.config, "disable_study_auto_reset", False)
                ) or bool(os.environ.get("STAGE_B_OPTUNA_DISABLE_AUTO_RESET")):
                    self.logger.error(
                        "Optuna study appears incompatible (dynamic value space), but auto-reset is disabled. "
                        "Refusing to create a new reset study. Fix by either: (a) keep the search space identical "
                        "to this study, or (b) intentionally choose a new study name / delete the DB."
                    )
                    raise
                try:
                    import time

                    reset_suffix = int(time.time())
                    reset_study_name = f"{study_name}_reset_{reset_suffix}"
                    self.logger.warning(
                        "Optuna study appears incompatible (dynamic value space). "
                        "Retrying once with fresh study: %s in %s",
                        reset_study_name,
                        storage_url,
                    )

                    reset_storage = optuna.storages.RDBStorage(url=storage_url)
                    reset_search = StageBOptunaSearch(
                        metric="score",
                        mode="max",
                        points_to_evaluate=points_to_evaluate,
                        sampler=tpe_sampler,
                        study_name=reset_study_name,
                        storage=reset_storage,
                        seed=self.config.seed,
                    )
                    results = _run_tuner(reset_search)
                    return _process_tune_results(results)
                except Exception as e2:
                    self.logger.error(f"Ray Tune retry after Optuna reset failed: {e2}")
                    self.logger.error(f"Full traceback (retry):\n{traceback.format_exc()}")
                    return OptimizedParams()

            self.logger.error(f"Ray Tune optimization failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return OptimizedParams()

    def optimize_with_ray_tune_multi_symbol(
        self,
        *,
        panels_by_symbol: Dict[str, pd.DataFrame],
        column_families: Dict[str, str],
        labels_by_symbol: Dict[str, pd.DataFrame],
        block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
        walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        horizon: int = 1,
        stage_a_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizedParams:
        """Ray Tune optimization in global multi-symbol mode."""
        if not RAY_AVAILABLE or tune is None or ray is None:
            self.logger.error("Ray Tune not available")
            return OptimizedParams()

        symbols = sorted(list(panels_by_symbol.keys()))
        n_folds = len(walk_forward_folds_multi)
        self.logger.info(
            "Ray Tune GLOBAL multi-symbol optimization: %d symbols, %d folds, %d trials (%d concurrent)",
            len(symbols),
            n_folds,
            self.config.n_trials,
            self.config.ray_tune_concurrent_trials,
        )

        # Cache fold-derived cap so seq_len choices are globally valid.
        # IMPORTANT: participation must mean "scorable for the seq_len search space",
        # otherwise partial-overlap symbols can collapse the global cap.
        _min_symbols_per_fold = int(getattr(self.config, "min_symbols_per_fold", 1))
        _val_buffer = 5
        _min_choice = 8
        self._mamba_seq_len_cap_for_folds = self._compute_mamba_seq_len_cap_for_multi_folds(
            walk_forward_folds_multi,
            min_symbols_per_fold=_min_symbols_per_fold,
            val_buffer=_val_buffer,
            min_val_len_for_participation=int(_min_choice) + int(_val_buffer),
        )

        # Analyze family structure using an anchor panel.
        anchor_panel = panels_by_symbol[symbols[0]]
        self._analyze_families(anchor_panel, column_families, stage_a_weights)

        if not ray.is_initialized():
            init_kwargs: Dict[str, Any] = {}
            init_kwargs = _apply_ray_runtime_env_envvars(init_kwargs)
            ray.init(ignore_reinit_error=True, log_to_driver=False, **init_kwargs)

        panels_ref = ray.put(panels_by_symbol)
        labels_ref = ray.put(labels_by_symbol)
        block_summaries_ref = ray.put(block_summaries_by_symbol)
        column_families_ref = ray.put(column_families)
        folds_ref = ray.put(walk_forward_folds_multi)

        family_columns = self.family_columns.copy()
        family_sizes = self.family_sizes.copy()
        stage_a_weights_copy = (stage_a_weights or {}).copy()
        config_dict = {
            "fold_gpu_reserve_gb": self.config.fold_gpu_reserve_gb,
            "max_total_dims": self.config.max_total_dims,
            "sharpe_weight": self.config.sharpe_weight,
            "rwa_weight": self.config.rwa_weight,
            "stability_weight": self.config.stability_weight,
            "sequence_model_type": getattr(self.config, "sequence_model_type", "mamba"),
            # Make sure global/per-symbol weight mode is correctly applied inside the Ray trainable.
            "global_multi_symbol": bool(getattr(self.config, "global_multi_symbol", True)),
            "global_per_symbol_weights": bool(getattr(self.config, "global_per_symbol_weights", False)),
            "family_weight_clip_min": float(getattr(self.config, "family_weight_clip_min", 0.01)),
        }

        search_space = self._build_ray_tune_search_space()

        concurrent_trials = int(self.config.ray_tune_concurrent_trials)
        gpu_per_trial = min(float(self.config.ray_gpus_per_trial), 1.0 / max(1, concurrent_trials))

        try:
            import os as _os

            _total_cpus = 0
            try:
                _total_cpus = int(ray.cluster_resources().get("CPU", 0))
            except Exception:
                _total_cpus = 0
            if _total_cpus <= 0:
                _total_cpus = int(_os.cpu_count() or 1)
            _reserve_cpus = max(1, min(4, int(_total_cpus * 0.05)))
            if int(self.config.ray_cpus_per_trial) > 0:
                cpu_per_trial = float(int(self.config.ray_cpus_per_trial))
            else:
                _auto_cpu = max(1, int((_total_cpus - _reserve_cpus) // max(1, concurrent_trials)))
                _cap_cpu = max(4, min(16, _auto_cpu))
                cpu_per_trial = float(min(_auto_cpu, _cap_cpu))
        except Exception:
            cpu_per_trial = float(max(1, int(getattr(self.config, "ray_cpus_per_trial", 8) or 8)))

        est_vram_gb = gpu_per_trial * float(self.config.fold_gpu_memory_gb)
        self.logger.info(
            "🚀 Ray Tune resources (GLOBAL): cpu_per_trial=%.1f, gpu_per_trial=%.3f (~%.1f GB)",
            cpu_per_trial,
            gpu_per_trial,
            est_vram_gb,
        )

        _panels_ref = panels_ref
        _labels_ref = labels_ref
        _block_summaries_ref = block_summaries_ref
        _column_families_ref = column_families_ref
        _folds_ref = folds_ref
        _family_columns = family_columns
        _family_sizes = family_sizes
        _stage_a_weights = stage_a_weights_copy
        _config_dict = config_dict
        _horizon = int(horizon)

        def make_trainable():
            def trainable(config: Dict[str, Any]) -> Dict[str, Any]:
                import time as _time
                import logging
                import sys

                _t_start = _time.time()
                logger = logging.getLogger("stage_b.optuna.trainable")

                try:
                    _ray_trial_id = tune.get_trial_id() if tune is not None else None
                except Exception:
                    _ray_trial_id = None
                _optuna_trial_number = config.get("_optuna_trial_number")
                if _optuna_trial_number is not None:
                    logger.info(
                        "Ray trial started: ray_trial_id=%s optuna_trial_number=%s",
                        _ray_trial_id,
                        _optuna_trial_number,
                    )

                try:
                    try:
                        import torch as _torch

                        if _torch.cuda.is_available() and gpu_per_trial < 1.0:
                            _torch.cuda.set_per_process_memory_fraction(
                                float(max(0.05, min(0.95, gpu_per_trial))),
                                device=0,
                            )
                    except Exception:
                        pass

                    panels_data = ray.get(_panels_ref)
                    labels_data = ray.get(_labels_ref)
                    block_summaries_data = ray.get(_block_summaries_ref)
                    column_families_data = ray.get(_column_families_ref)
                    folds_data = ray.get(_folds_ref)

                    # Lightweight optimizer instance inside trainable
                    trial_config = OptunaConfig(
                        ray_fold_parallelism=0,
                        ray_fold_gpu_fraction=0.0,
                        fold_gpu_reserve_gb=_config_dict["fold_gpu_reserve_gb"],
                        max_total_dims=_config_dict["max_total_dims"],
                        sharpe_weight=_config_dict["sharpe_weight"],
                        rwa_weight=_config_dict["rwa_weight"],
                        stability_weight=_config_dict["stability_weight"],
                        sequence_model_type=_config_dict.get("sequence_model_type", "mamba"),
                        global_multi_symbol=bool(_config_dict.get("global_multi_symbol", True)),
                        global_per_symbol_weights=bool(_config_dict.get("global_per_symbol_weights", False)),
                        family_weight_clip_min=float(_config_dict.get("family_weight_clip_min", 0.01)),
                        global_symbols=tuple(sorted(list(panels_data.keys()))),
                    )
                    optimizer = StageBOptunaOptimizer(config=trial_config, logger=logger)
                    optimizer.family_columns = _family_columns
                    optimizer.family_sizes = _family_sizes
                    optimizer.stage_a_weights = _stage_a_weights

                    per_symbol_weights = bool(getattr(optimizer.config, "global_per_symbol_weights", False))
                    symbols_for_weights = tuple(sorted(list(panels_data.keys())))

                    family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
                    family_params_by_symbol: Optional[
                        Dict[str, Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]]]
                    ] = None

                    track_b_family_weights: Dict[str, float] = {}
                    track_b_family_weights_by_symbol: Optional[Dict[str, Dict[str, float]]] = None

                    if per_symbol_weights:
                        family_params_by_symbol = {}
                        track_b_family_weights_by_symbol = {}

                        # Build per-symbol family params + Track-B weights from symbol-prefixed keys.
                        for sym in symbols_for_weights:
                            sym_u = str(sym).upper()
                            sym_family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
                            for family in STAGE_A_FAMILIES:
                                weight_key = f"{sym_u}__weight_{family}"
                                dim_type_key = f"{family}_dim_type"
                                weight = config.get(weight_key, 0.0)
                                include = weight >= optimizer.config.family_weight_clip_min
                                method = config.get(dim_type_key, "pca")

                                if method == "ae":
                                    dim_key = f"{family}_ae_dim"
                                else:
                                    dim_key = f"{family}_pca_dim"
                                dim = config.get(dim_key, 4) if include else 0

                                ae_params: Dict[str, Any] = {}
                                if method == "ae":
                                    ae_params = {
                                        "ae_layers": config.get(f"{family}_ae_layers", 2),
                                        "ae_activation": config.get(f"{family}_ae_activation", "relu"),
                                        "ae_dropout": config.get(f"{family}_ae_dropout", 0.0),
                                        "ae_lr": config.get(f"{family}_ae_lr", 1e-3),
                                    }
                                sym_family_params[family] = (include, dim, method, float(weight) if include else 0.0, ae_params)

                            family_params_by_symbol[sym_u] = sym_family_params

                            sym_track_b: Dict[str, float] = {}
                            for family in HF_BLOCK_FAMILIES:
                                sym_track_b[family] = float(config.get(f"{sym_u}__weight_b_{family}", 1.0))
                            for summary_name in TRACK_B_SUMMARY_BLOCKS:
                                sym_track_b[summary_name] = float(config.get(f"{sym_u}__weight_b_{summary_name}", 1.0))
                            track_b_family_weights_by_symbol[sym_u] = sym_track_b

                        # Build pooled/union family params for encoder fitting.
                        for family in STAGE_A_FAMILIES:
                            dim_type_key = f"{family}_dim_type"
                            method = config.get(dim_type_key, "pca")
                            if method == "ae":
                                dim_key = f"{family}_ae_dim"
                            else:
                                dim_key = f"{family}_pca_dim"

                            include_any = False
                            for sym_u, sym_fp in family_params_by_symbol.items():
                                fp = sym_fp.get(family)
                                if fp is not None and bool(fp[0]):
                                    include_any = True
                                    break

                            dim = config.get(dim_key, 4) if include_any else 0
                            ae_params: Dict[str, Any] = {}
                            if method == "ae":
                                ae_params = {
                                    "ae_layers": config.get(f"{family}_ae_layers", 2),
                                    "ae_activation": config.get(f"{family}_ae_activation", "relu"),
                                    "ae_dropout": config.get(f"{family}_ae_dropout", 0.0),
                                    "ae_lr": config.get(f"{family}_ae_lr", 1e-3),
                                }
                            # Weight does not affect encoder fitting (weights applied after encoding).
                            family_params[family] = (include_any, dim, method, 1.0 if include_any else 0.0, ae_params)
                    else:
                        # Global shared weights (legacy behavior)
                        for family in STAGE_A_FAMILIES:
                            weight_key = f"weight_{family}"
                            dim_type_key = f"{family}_dim_type"
                            weight = config.get(weight_key, 0.0)
                            include = weight >= optimizer.config.family_weight_clip_min
                            method = config.get(dim_type_key, "pca")

                            if method == "ae":
                                dim_key = f"{family}_ae_dim"
                            else:
                                dim_key = f"{family}_pca_dim"
                            dim = config.get(dim_key, 4) if include else 0

                            ae_params: Dict[str, Any] = {}
                            if method == "ae":
                                ae_params = {
                                    "ae_layers": config.get(f"{family}_ae_layers", 2),
                                    "ae_activation": config.get(f"{family}_ae_activation", "relu"),
                                    "ae_dropout": config.get(f"{family}_ae_dropout", 0.0),
                                    "ae_lr": config.get(f"{family}_ae_lr", 1e-3),
                                }
                            family_params[family] = (include, dim, method, float(weight) if include else 0.0, ae_params)

                        for family in HF_BLOCK_FAMILIES:
                            track_b_family_weights[family] = float(config.get(f"weight_b_{family}", 1.0))
                        for summary_name in TRACK_B_SUMMARY_BLOCKS:
                            track_b_family_weights[summary_name] = float(config.get(f"weight_b_{summary_name}", 1.0))

                    sequence_model_type = optimizer.config.sequence_model_type
                    if sequence_model_type == "mamba":
                        mamba_params = {
                            "mamba_d_model": config.get("mamba_d_model"),
                            "mamba_n_layers": config.get("mamba_n_layers"),
                            "mamba_ssm_dim": config.get("mamba_ssm_dim"),
                            "mamba_expand_factor": config.get("mamba_expand_factor"),
                            "mamba_seq_len": config.get("mamba_seq_len"),
                            "mamba_activation": config.get("mamba_activation", "silu"),
                            "mamba_norm_type": config.get("mamba_norm_type", "rmsnorm"),
                            "mamba_norm_strategy": config.get("mamba_norm_strategy", "pre"),
                            "mamba_dropout": config.get("mamba_dropout", 0.1),
                            "mamba_resid_dropout": config.get("mamba_resid_dropout", 0.0),
                            "mamba_ssm_dropout": config.get("mamba_ssm_dropout", 0.0),
                            "mamba_gate_dropout": config.get("mamba_gate_dropout", 0.0),
                            "mamba_optimizer": config.get("mamba_optimizer", "adamw"),
                            "mamba_learning_rate": config.get("mamba_learning_rate", 1e-3),
                            "mamba_weight_decay": config.get("mamba_weight_decay", 1e-4),
                            "mamba_grad_clip": config.get("mamba_grad_clip", 1.0),
                            "mamba_lr_scheduler": config.get("mamba_lr_scheduler", "cosine"),
                            "mamba_warmup_steps": config.get("mamba_warmup_steps", 100),
                            "mamba_max_epochs": config.get("mamba_max_epochs", 10),
                            "mamba_batch_size": config.get("mamba_batch_size", 32),
                            "mamba_loss_fn": config.get("mamba_loss_fn", "smooth_l1"),
                            "mamba_head_type": config.get("mamba_head_type", "linear"),
                            "mamba_head_hidden_dim": config.get("mamba_head_hidden_dim", 128),
                            "mamba_head_num_layers": config.get("mamba_head_num_layers", 1),
                            "mamba_head_dropout": config.get("mamba_head_dropout", 0.0),
                        }
                        lstm_params: Dict[str, Any] = {}
                    else:
                        # Keep parity with existing code paths; LSTM global mode can be added later.
                        raise ValueError("Global multi-symbol mode currently supports sequence_model_type='mamba' only")

                    threshold_params = {
                        "threshold": config.get("threshold", 0.10),
                        "bull_long_mult": config.get("bull_long_mult", 1.0),
                        "bull_short_mult": config.get("bull_short_mult", 1.0),
                        "bear_long_mult": config.get("bear_long_mult", 1.5),
                        "bear_short_mult": config.get("bear_short_mult", 0.5),
                        "crisis_long_mult": config.get("crisis_long_mult", 3.0),
                        "crisis_short_mult": config.get("crisis_short_mult", 3.0),
                        "conf_threshold": config.get("conf_threshold", 0.5),
                        "vol_scaler": config.get("vol_scaler", 1.0),
                    }
                    pipeline_params = {
                        "smoothing_type": config.get("smoothing_type", "none"),
                        "smoothing_window": config.get("smoothing_window", 3),
                        "train_fraction": config.get("train_fraction", 0.8),
                    }

                    def _ray_progress_callback(step: int, intermediate_score: float) -> bool:
                        metrics = {
                            "score": float(intermediate_score),
                            "avg_score": float(intermediate_score),
                            "n_successful_folds": int(step),
                            "folds_done": int(step),
                            "done": False,
                        }
                        extra = getattr(optimizer, "_ray_progress_metrics", None)
                        if isinstance(extra, dict) and extra:
                            metrics.update(extra)
                        tune.report(metrics)
                        return False

                    score, metadata = optimizer._evaluate_trial_params_multi_symbol(
                        panels_by_symbol=panels_data,
                        column_families=column_families_data,
                        labels_by_symbol=labels_data,
                        block_summaries_by_symbol=block_summaries_data,
                        walk_forward_folds_multi=folds_data,
                        family_params=family_params,
                        family_params_by_symbol=family_params_by_symbol,
                        weight_a=config.get("track_a_weight", 1.0),
                        weight_b=config.get("track_b_weight", 1.0),
                        sequence_model_type=sequence_model_type,
                        track_b_family_weights=track_b_family_weights,
                        track_b_family_weights_by_symbol=track_b_family_weights_by_symbol,
                        lstm_params=lstm_params,
                        mamba_params=mamba_params,
                        threshold_params=threshold_params,
                        pipeline_params=pipeline_params,
                        horizon=_horizon,
                        progress_callback=_ray_progress_callback,
                    )

                    elapsed = _time.time() - _t_start
                    print(f"[TRACE:trainable] GLOBAL Trial completed: score={score:.4f} in {elapsed:.1f}s", file=sys.stderr, flush=True)
                    final_metrics = {
                        "score": float(score),
                        "avg_score": float(metadata.get("avg_score", score)),
                        "n_successful_folds": int(metadata.get("n_successful_folds", 0)),
                        "elapsed": float(elapsed),
                        "done": True,
                    }
                    # Surface useful diagnostics in Ray Tune artifacts.
                    for _k in ("reject_reason", "pooled_samples", "seq_len", "mamba_seq_len_cap", "mamba_seq_len"):
                        if _k in metadata:
                            final_metrics[_k] = metadata.get(_k)
                    tune.report(final_metrics)
                    return final_metrics
                except Exception as e:
                    import traceback

                    tb_str = traceback.format_exc()
                    print(f"[ERROR:trainable] GLOBAL {e}\n{tb_str}", file=sys.stderr, flush=True)
                    tune.report({"score": float("-inf"), "done": True})
                    return {"score": float("-inf")}

            return trainable

        resources_per_trial = {"cpu": float(cpu_per_trial), "gpu": float(gpu_per_trial)}

        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max",
            max_t=max(1, int(n_folds)),
            grace_period=max(3, int(max(1, n_folds) // 10)),
            reduction_factor=3,
        )

        # OptunaSearch configuration is reused from the single-symbol path.
        optuna_search = None
        if OptunaSearch is not None:
            import hashlib
            import json as _json

            try:
                _families_sig = sorted(list(family_sizes.keys()))
            except Exception:
                _families_sig = []

            _space_sig = {
                "horizon": int(horizon),
                "sequence_model_type": str(getattr(self.config, "sequence_model_type", "mamba")),
                "tune_weights_only": bool(getattr(self.config, "tune_weights_only", False)),
                "fixed_params": dict(getattr(self.config, "fixed_params", {}) or {}),
                "global_symbols": tuple(sorted(symbols)),
                "global_per_symbol_weights": bool(getattr(self.config, "global_per_symbol_weights", False)),
                "max_total_dims": int(getattr(self.config, "max_total_dims", 500)),
                "mamba_seq_len_choices": list(self._get_effective_mamba_seq_len_choices()),
                "mamba_d_model_choices": list(getattr(self.config, "mamba_d_model_choices", (96, 128, 160, 192, 224, 256, 288))),
                "mamba_n_layers_choices": list(getattr(self.config, "mamba_n_layers_choices", (3, 4, 5, 6, 8))),
                "mamba_ssm_dim_choices": list(getattr(self.config, "mamba_ssm_dim_choices", (64, 96, 128, 160))),
                "mamba_expand_factor_choices": list(getattr(self.config, "mamba_expand_factor_choices", (2.0, 2.5, 3.0, 4.0))),
                "mamba_loss_fn_choices": list(getattr(self.config, "mamba_loss_fn_choices", ("smooth_l1", "huber", "bce_logits", "mse"))),
                "mamba_head_type_choices": list(getattr(self.config, "mamba_head_type_choices", ("linear", "mlp"))),
                "families": _families_sig,
            }
            _space_fingerprint = hashlib.sha256(
                _json.dumps(_space_sig, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:10]
            override_name = (
                getattr(self.config, "study_name_override", None)
                or os.environ.get("STAGE_B_OPTUNA_STUDY_NAME")
            )
            if override_name:
                study_name = str(override_name)
            else:
                study_name = f"stage_b_global_{_space_sig['sequence_model_type']}_h{horizon}_{_space_fingerprint}"
            storage_path = Path("artifacts/optuna_studies")
            storage_path.mkdir(parents=True, exist_ok=True)
            # Global multi-symbol mode: keep one sqlite file per (model_type, horizon)
            # and store separate Optuna *studies* inside it (by study_name).
            storage_db = f"stage_b_{_space_sig['sequence_model_type']}_h{horizon}_global.db"
            storage_url = f"sqlite:///{storage_path / storage_db}"

            optuna_storage = optuna.storages.RDBStorage(url=storage_url)

            # Guardrails: if a pinned/override study name is reused with different symbols/mode,
            # fail fast to prevent silent cross-mixing.
            try:
                existing_study = optuna.load_study(study_name=study_name, storage=optuna_storage)
                bound_symbols = tuple(existing_study.user_attrs.get("global_symbols", ()) or ())
                if bound_symbols and tuple(sorted(bound_symbols)) != tuple(sorted(symbols)):
                    raise ValueError(
                        f"Optuna study '{study_name}' is bound to global_symbols={bound_symbols}, not {tuple(sorted(symbols))}."
                    )
                if not bound_symbols:
                    existing_study.set_user_attr("global_symbols", tuple(sorted(symbols)))
                existing_study.set_user_attr(
                    "global_per_symbol_weights",
                    bool(getattr(self.config, "global_per_symbol_weights", False)),
                )
                del existing_study
            except Exception:
                # Best-effort create/tag; Ray Tune may race-create as well.
                try:
                    created = optuna.create_study(
                        study_name=study_name,
                        storage=optuna_storage,
                        direction="maximize",
                        load_if_exists=False,
                    )
                    created.set_user_attr("global_symbols", tuple(sorted(symbols)))
                    created.set_user_attr(
                        "global_per_symbol_weights",
                        bool(getattr(self.config, "global_per_symbol_weights", False)),
                    )
                    del created
                except Exception:
                    pass
            tpe_sampler = optuna.samplers.TPESampler(seed=self.config.seed)
            optuna_search = StageBOptunaSearch(
                metric="score",
                mode="max",
                sampler=tpe_sampler,
                study_name=study_name,
                storage=optuna_storage,
                seed=self.config.seed,
            )

        def _process_tune_results(_results) -> OptimizedParams:
            best_result = _results.get_best_result(metric="score", mode="max")
            if best_result is None or best_result.config is None:
                self.logger.warning("Ray Tune failed to find best result")
                return OptimizedParams()
            best_score = best_result.metrics.get("score", float("-inf"))
            self.logger.info("🏆 Ray Tune GLOBAL best score: %.4f", best_score)
            self.best_params = self._extract_params_from_dict(best_result.config, best_score)
            return self.best_params

        tuner = tune.Tuner(
            tune.with_resources(make_trainable(), resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                num_samples=self.config.n_trials,
                max_concurrent_trials=self.config.ray_tune_concurrent_trials,
                scheduler=asha_scheduler,
                search_alg=optuna_search,
            ),
            param_space=search_space,
        )

        try:
            results = tuner.fit()
            return _process_tune_results(results)
        except Exception as e:
            import traceback

            msg = str(e)
            is_dynamic_space = (
                "CategoricalDistribution does not support dynamic value space" in msg
                or "dynamic value space" in msg
            )

            # Common failure mode: we changed the categorical choices but OptunaSearch
            # is reusing an existing sqlite-backed study. Optuna doesn't allow changing
            # categorical value sets within an existing study.
            pinned_study = bool(getattr(self.config, "study_name_override", None)) or bool(
                os.environ.get("STAGE_B_OPTUNA_STUDY_NAME")
            )
            if is_dynamic_space and (not pinned_study) and OptunaSearch is not None and optuna_search is not None:
                try:
                    import hashlib
                    import time as _time

                    reset_suffix = hashlib.sha256(f"{_time.time()}".encode("utf-8")).hexdigest()[:8]
                    reset_study_name = f"{study_name}_reset_{reset_suffix}"
                    self.logger.warning(
                        "Optuna study appears incompatible (dynamic value space). "
                        "Retrying with fresh study '%s' in %s",
                        reset_study_name,
                        storage_url,
                    )

                    reset_storage = optuna.storages.RDBStorage(url=storage_url)
                    reset_sampler = optuna.samplers.TPESampler(seed=self.config.seed)
                    reset_search = StageBOptunaSearch(
                        metric="score",
                        mode="max",
                        sampler=reset_sampler,
                        study_name=reset_study_name,
                        storage=reset_storage,
                        seed=self.config.seed,
                    )

                    reset_tuner = tune.Tuner(
                        tune.with_resources(make_trainable(), resources=resources_per_trial),
                        tune_config=tune.TuneConfig(
                            num_samples=self.config.n_trials,
                            max_concurrent_trials=self.config.ray_tune_concurrent_trials,
                            scheduler=asha_scheduler,
                            search_alg=reset_search,
                        ),
                        param_space=search_space,
                    )
                    reset_results = reset_tuner.fit()
                    return _process_tune_results(reset_results)
                except Exception as e2:
                    self.logger.error(f"Ray Tune GLOBAL retry after Optuna reset failed: {e2}")
                    self.logger.error(f"Full traceback (retry):\n{traceback.format_exc()}")
                    return OptimizedParams()

            self.logger.error(f"Ray Tune GLOBAL multi-symbol optimization failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return OptimizedParams()
    
    def save_params(self, path: Path):
        """Save optimized parameters to file."""
        if self.best_params is None:
            self.logger.warning("No params to save")
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.best_params.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved optimized params to {path}")
    
    def load_params(self, path: Path) -> OptimizedParams:
        """Load optimized parameters from file."""
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Params file not found: {path}")
            return OptimizedParams()
        
        with open(path) as f:
            data = json.load(f)
        
        self.best_params = OptimizedParams.from_dict(data)
        return self.best_params


def generate_prediction_tapes_multi_symbol_mamba(
    *,
    panels_by_symbol: Dict[str, pd.DataFrame],
    column_families: Dict[str, str],
    labels_by_symbol: Dict[str, pd.DataFrame],
    block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
    walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    horizon: int,
    params: "OptimizedParams",
    max_total_dims: int = 500,
) -> Tuple[Dict[str, List[pd.DataFrame]], Dict[str, Any]]:
    """Generate per-symbol prediction tapes for a global pooled Mamba walk-forward.

    This is used after Optuna completes in GLOBAL multi-symbol mode. It:
      - builds Track-C per symbol using the best Optuna params
      - trains ONE shared Mamba per fold on pooled (concat batch-axis) sequences
      - predicts per symbol on the fold's out-of-sample indices
      - returns fold-level DataFrames compatible with Stage C's prediction tapes

    Important:
      - No per-fold seq_len clamping; we enforce the same global fold-derived cap.
      - Encoders are fit once per run on pooled fold-0 training rows (leakage-safe
        across symbols, and consistent with the Optuna global evaluator).
    """

    import torch as _torch

    symbols = sorted([str(s).upper() for s in panels_by_symbol.keys()])
    if not symbols:
        return {}, {"reject_reason": "no_symbols"}
    if not walk_forward_folds_multi:
        return {}, {"reject_reason": "no_folds"}

    # Build family param tuples in the format expected by _build_track_a.
    family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
    family_params_by_symbol: Optional[
        Dict[str, Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]]]
    ] = None
    track_b_family_weights_by_symbol: Optional[Dict[str, Dict[str, float]]] = None

    fw_by_symbol = getattr(params, "family_weights_by_symbol", None)
    tb_by_symbol = getattr(params, "track_b_family_weights_by_symbol", None)
    if isinstance(fw_by_symbol, dict) and fw_by_symbol:
        family_params_by_symbol = {}
        if isinstance(tb_by_symbol, dict) and tb_by_symbol:
            track_b_family_weights_by_symbol = tb_by_symbol

        # Per-symbol family params
        for sym in symbols:
            sym_u = str(sym).upper()
            sym_weights = (fw_by_symbol.get(sym_u) or fw_by_symbol.get(sym) or {})
            sym_fp: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
            for fam in STAGE_A_FAMILIES:
                weight = float(sym_weights.get(fam, 0.0))
                include = bool(weight > 0.0)
                dim = int((params.family_dimensions or {}).get(fam, 0))
                method = str((params.family_encoders or {}).get(fam, "pca") or "pca")
                sym_fp[fam] = (include, dim if include else 0, method, weight if include else 0.0, {})
            family_params_by_symbol[sym_u] = sym_fp

        # Pooled/union family params for encoder fitting
        for fam in STAGE_A_FAMILIES:
            include_any = False
            for sym in symbols:
                sym_u = str(sym).upper()
                fp = (family_params_by_symbol.get(sym_u) or {}).get(fam)
                if fp is not None and bool(fp[0]):
                    include_any = True
                    break
            dim = int((params.family_dimensions or {}).get(fam, 0)) if include_any else 0
            method = str((params.family_encoders or {}).get(fam, "pca") or "pca")
            family_params[fam] = (include_any, dim, method, 1.0 if include_any else 0.0, {})
    else:
        for fam in STAGE_A_FAMILIES:
            weight = float((params.family_weights or {}).get(fam, 0.0))
            include = bool((params.included_families or {}).get(fam, weight > 0.0))
            dim = int((params.family_dimensions or {}).get(fam, 0))
            method = str((params.family_encoders or {}).get(fam, "pca") or "pca")
            family_params[fam] = (include, dim, method, weight, {})

    weight_a = float(getattr(params, "track_a_weight", 1.0))
    weight_b = float(getattr(params, "track_b_weight", 1.0))

    threshold_params = {
        "threshold": float(getattr(params, "threshold", 0.10)),
        "bull_mult": float(getattr(params, "bull_mult", 0.8)),
        "bear_mult": float(getattr(params, "bear_mult", 1.5)),
        "crisis_mult": float(getattr(params, "crisis_mult", 3.0)),
    }

    mamba_params = {
        "mamba_seq_len": int(getattr(params, "mamba_seq_len", 128)),
        "mamba_d_model": int(getattr(params, "mamba_d_model", 128)),
        "mamba_n_layers": int(getattr(params, "mamba_n_layers", 4)),
        "mamba_ssm_dim": int(getattr(params, "mamba_ssm_dim", 96)),
        "mamba_expand_factor": float(getattr(params, "mamba_expand_factor", 2.0)),
        "mamba_activation": str(getattr(params, "mamba_activation", "silu")),
        "mamba_norm_type": str(getattr(params, "mamba_norm_type", "rmsnorm")),
        "mamba_norm_strategy": str(getattr(params, "mamba_norm_strategy", "pre")),
        "mamba_dropout": float(getattr(params, "mamba_dropout", 0.1)),
        "mamba_resid_dropout": float(getattr(params, "mamba_resid_dropout", 0.0)),
        "mamba_ssm_dropout": float(getattr(params, "mamba_ssm_dropout", 0.0)),
        "mamba_gate_dropout": float(getattr(params, "mamba_gate_dropout", 0.0)),
        "mamba_optimizer": str(getattr(params, "mamba_optimizer", "adamw")),
        "mamba_learning_rate": float(getattr(params, "mamba_learning_rate", 1e-3)),
        "mamba_weight_decay": float(getattr(params, "mamba_weight_decay", 1e-4)),
        "mamba_grad_clip": float(getattr(params, "mamba_grad_clip", 1.0)),
        "mamba_lr_scheduler": str(getattr(params, "mamba_lr_scheduler", "cosine")),
        "mamba_warmup_steps": int(getattr(params, "mamba_warmup_steps", 100)),
        "mamba_max_epochs": int(getattr(params, "mamba_max_epochs", 10)),
        "mamba_batch_size": int(getattr(params, "mamba_batch_size", 32)),
        "mamba_loss_fn": str(getattr(params, "mamba_loss_fn", "smooth_l1")),
        "mamba_head_type": str(getattr(params, "mamba_head_type", "linear")),
        "mamba_head_hidden_dim": int(getattr(params, "mamba_head_hidden_dim", 128)),
        "mamba_head_num_layers": int(getattr(params, "mamba_head_num_layers", 1)),
        "mamba_head_dropout": float(getattr(params, "mamba_head_dropout", 0.0)),
    }

    opt_cfg = OptunaConfig(max_total_dims=int(max_total_dims), horizon=int(horizon), sequence_model_type="mamba")
    optimizer = StageBOptunaOptimizer(config=opt_cfg)

    # Build per-symbol Track C with shared encoders, fitted on pooled fold-0 training rows.
    first_fold = walk_forward_folds_multi[0]
    pooled_train_frames: List[pd.DataFrame] = []
    for sym in symbols:
        tr = first_fold.get(sym)
        if tr is None:
            continue
        train_idx, _ = tr
        if len(train_idx) > 0:
            pooled_train_frames.append(panels_by_symbol[sym].iloc[train_idx])
    if not pooled_train_frames:
        return {}, {"reject_reason": "no_pooled_train_rows"}

    pooled_panel = pd.concat(pooled_train_frames, axis=0)
    pooled_idx = np.arange(len(pooled_panel), dtype=int)
    try:
        optimizer._build_track_a(pooled_panel, column_families, family_params, pooled_idx)
    except Exception as exc:
        return {}, {"reject_reason": "track_a_pooled_failed", "error": str(exc)}

    track_c_by_symbol: Dict[str, pd.DataFrame] = {}
    actual_returns_by_symbol: Dict[str, pd.Series] = {}
    for sym in symbols:
        panel = panels_by_symbol[sym]
        labels = labels_by_symbol[sym]
        actual_returns_by_symbol[sym] = labels["forward_return"].astype(float)
        tr = first_fold.get(sym)
        if tr is None:
            return {}, {"reject_reason": "missing_symbol_in_fold0", "symbol": sym}
        first_train_idx, _ = tr
        try:
            sym_u = str(sym).upper()
            sym_family_params = (
                (family_params_by_symbol or {}).get(sym_u)
                or (family_params_by_symbol or {}).get(sym)
                or family_params
            )
            sym_track_b_weights = (
                (track_b_family_weights_by_symbol or {}).get(sym_u)
                or (track_b_family_weights_by_symbol or {}).get(sym)
                or getattr(params, "track_b_family_weights", None)
                or {}
            )

            track_a_sym, _ = optimizer._build_track_a(panel, column_families, sym_family_params, first_train_idx)
            track_b_sym = optimizer._build_track_b(
                panel,
                column_families,
                block_summaries_by_symbol.get(sym, {}),
                track_b_family_weights=sym_track_b_weights,
            )
            track_c_sym = optimizer._build_track_c(track_a_sym, track_b_sym, weight_a, weight_b)
        except Exception as exc:
            return {}, {"reject_reason": "track_build_failed", "symbol": sym, "error": str(exc)}

        if track_c_sym is None or track_c_sym.empty:
            return {}, {"reject_reason": "track_c_empty", "symbol": sym}
        if int(track_c_sym.shape[1]) > int(max_total_dims):
            return {}, {
                "reject_reason": "track_c_too_large",
                "symbol": sym,
                "dims": int(track_c_sym.shape[1]),
                "max_total_dims": int(max_total_dims),
            }
        track_c_by_symbol[sym] = track_c_sym

    # Ensure Track-C schema is identical across symbols for pooled scaling + training.
    # Missing columns are filled with deterministic zeros.
    if track_c_by_symbol:
        first_sym = symbols[0]
        ordered_cols: List[str] = list(track_c_by_symbol[first_sym].columns)
        seen = set(ordered_cols)
        for sym in symbols[1:]:
            df = track_c_by_symbol.get(sym)
            if df is None:
                continue
            for c in df.columns:
                if c not in seen:
                    ordered_cols.append(c)
                    seen.add(c)
        for sym in list(track_c_by_symbol.keys()):
            track_c_by_symbol[sym] = track_c_by_symbol[sym].reindex(columns=ordered_cols).fillna(0.0)

    seq_len = int(mamba_params.get("mamba_seq_len", 128))
    cap = optimizer._compute_mamba_seq_len_cap_for_multi_folds(
        walk_forward_folds_multi,
        min_symbols_per_fold=int(getattr(optimizer.config, "min_symbols_per_fold", 1)),
        val_buffer=5,
        min_val_len_for_participation=int(8) + int(5),
    )
    optimizer._mamba_seq_len_cap_for_folds = cap
    if int(seq_len) > int(cap):
        return {}, {
            "reject_reason": "mamba_seq_len_exceeds_fold_cap",
            "mamba_seq_len": int(seq_len),
            "mamba_seq_len_cap": int(cap),
        }

    cfg = {
        "sequence_model_type": "mamba",
        "mamba_seq_len": int(seq_len),
        "mamba_d_model": int(mamba_params.get("mamba_d_model", 128)),
        "mamba_n_layers": int(mamba_params.get("mamba_n_layers", 4)),
        "mamba_ssm_dim": int(mamba_params.get("mamba_ssm_dim", 96)),
        "mamba_expand_factor": float(mamba_params.get("mamba_expand_factor", 2.0)),
        "mamba_activation": mamba_params.get("mamba_activation", "silu"),
        "mamba_norm_type": mamba_params.get("mamba_norm_type", "rmsnorm"),
        "mamba_norm_strategy": mamba_params.get("mamba_norm_strategy", "pre"),
        "mamba_dropout": float(mamba_params.get("mamba_dropout", 0.1)),
        "mamba_resid_dropout": float(mamba_params.get("mamba_resid_dropout", 0.0)),
        "mamba_ssm_dropout": float(mamba_params.get("mamba_ssm_dropout", 0.0)),
        "mamba_gate_dropout": float(mamba_params.get("mamba_gate_dropout", 0.0)),
        "mamba_optimizer": mamba_params.get("mamba_optimizer", "adamw"),
        "mamba_learning_rate": float(mamba_params.get("mamba_learning_rate", 1e-3)),
        "mamba_weight_decay": float(mamba_params.get("mamba_weight_decay", 1e-4)),
        "mamba_grad_clip": float(mamba_params.get("mamba_grad_clip", 1.0)),
        "mamba_lr_scheduler": mamba_params.get("mamba_lr_scheduler", "cosine"),
        "mamba_warmup_steps": int(mamba_params.get("mamba_warmup_steps", 100)),
        "max_epochs": int(mamba_params.get("mamba_max_epochs", 10)),
        "mamba_batch_size": int(mamba_params.get("mamba_batch_size", 32)),
        "mamba_loss_fn": mamba_params.get("mamba_loss_fn", "smooth_l1"),
        "mamba_head_type": mamba_params.get("mamba_head_type", "linear"),
        "mamba_head_hidden_dim": int(mamba_params.get("mamba_head_hidden_dim", 128)),
        "mamba_head_num_layers": int(mamba_params.get("mamba_head_num_layers", 1)),
        "mamba_head_dropout": float(mamba_params.get("mamba_head_dropout", 0.0)),
    }

    device = _torch.device("cuda:0") if _torch.cuda.is_available() else _torch.device("cpu")

    tape_folds_by_symbol: Dict[str, List[pd.DataFrame]] = {sym: [] for sym in symbols}
    pooled_non_overlap_returns: List[float] = []

    # Late import to avoid circulars.
    from src.stage_b.stage_b_export import collect_fold_predictions

    def _portfolio_sharpe(samples: Sequence[float]) -> float:
        arr = np.asarray(list(samples), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return 0.0
        std = float(np.std(arr))
        if std <= 1e-9:
            return 0.0
        periods_per_year = max(1, 252 // max(1, int(horizon)))
        return float(np.mean(arr) / (std + 1e-9) * np.sqrt(periods_per_year))

    for fold_id, fold in enumerate(walk_forward_folds_multi):
        # Participation-aware fold guards (match seq_len cap semantics):
        # - Only symbols with labeled out-of-sample rows constrain/participate
        # - Skip folds with too few participating symbols
        # - Skip folds if training geometry is insufficient for the chosen seq_len
        min_symbols_per_fold = int(getattr(optimizer.config, "min_symbols_per_fold", 1))
        train_buffer = 20
        val_buffer = 5

        min_symbols_eff = max(1, min(int(min_symbols_per_fold), int(len(symbols))))

        # Policy B: only score/generate fold outputs if we have minimum coverage.
        usable_symbols: List[str] = []
        for sym in symbols:
            _train_idx, _out_idx = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
            if int(len(_out_idx)) >= int(seq_len) + int(val_buffer):
                usable_symbols.append(sym)
        if int(len(usable_symbols)) < int(min_symbols_eff):
            continue

        train_len_eff = min(
            int(len(fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))[0]))
            for sym in usable_symbols
        )
        if int(train_len_eff) < int(seq_len) + int(train_buffer):
            continue

        # Pooled scaler stats across symbols' training rows.
        train_frames: List[pd.DataFrame] = []
        train_targets: List[pd.Series] = []
        for sym in symbols:
            train_idx, _ = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
            if len(train_idx) == 0:
                continue
            train_frames.append(track_c_by_symbol[sym].iloc[train_idx])
            train_targets.append(actual_returns_by_symbol[sym].iloc[train_idx])
        if not train_frames:
            continue

        scaler_stats = optimizer._compute_pooled_scaler_stats(train_frames, train_targets)

        # Build pooled train SequenceData
        train_seq_datas = []
        for sym in symbols:
            train_idx, _ = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
            if len(train_idx) == 0:
                continue
            seq_res = build_sequence_data(
                track_c_by_symbol[sym].iloc[train_idx],
                actual_returns_by_symbol[sym].iloc[train_idx],
                int(seq_len),
                scaler_stats=scaler_stats,
            )
            if seq_res is None:
                train_seq_datas = []
                break
            seq_data_sym, _ = seq_res
            if len(seq_data_sym) < 20:
                train_seq_datas = []
                break
            train_seq_datas.append(seq_data_sym)
        if not train_seq_datas:
            continue

        pooled_train_seq = optimizer._concat_sequence_data(train_seq_datas)
        n_train = int(len(pooled_train_seq))
        train_end = int(n_train * 0.9)
        seq_train_idx = np.arange(train_end)
        seq_holdout_idx = np.arange(train_end, n_train)
        if int(len(seq_holdout_idx)) < 5:
            continue

        result = train_mamba_fold(
            pooled_train_seq,
            seq_train_idx,
            seq_holdout_idx,
            dict(cfg),
            device=device,
            return_model=True,
        )
        model = result.get("model")
        if model is None:
            continue
        model.eval()

        # Predict per symbol on out-of-sample indices
        symbol_return_series: List[pd.Series] = []
        with _torch.no_grad():
            for sym in symbols:
                _, out_idx = fold.get(sym, (np.array([], dtype=int), np.array([], dtype=int)))
                if len(out_idx) == 0:
                    continue

                out_res = build_sequence_data(
                    track_c_by_symbol[sym].iloc[out_idx],
                    actual_returns_by_symbol[sym].iloc[out_idx],
                    int(seq_len),
                    scaler_stats=scaler_stats,
                )
                if out_res is None:
                    continue
                out_seq_data, _ = out_res
                if len(out_seq_data) < 5:
                    continue

                out_timestamps = track_c_by_symbol[sym].iloc[out_idx].index[int(seq_len) :]
                out_X = out_seq_data.sequences.astype(np.float32)
                out_X_tensor = _torch.from_numpy(out_X).to(device)
                preds = model(out_X_tensor).detach().cpu().numpy().flatten()

                # Align lengths
                if len(preds) > len(out_timestamps):
                    preds = preds[: len(out_timestamps)]
                elif len(out_timestamps) > len(preds):
                    out_timestamps = out_timestamps[: len(preds)]

                ts_index = pd.DatetimeIndex(out_timestamps)
                y_true = actual_returns_by_symbol[sym].reindex(ts_index).to_numpy(dtype=float)
                keep = np.isfinite(y_true)
                if keep.sum() == 0:
                    continue

                preds_keep = preds[keep]
                ts_keep = ts_index[keep]
                y_keep = y_true[keep]

                fold_df = collect_fold_predictions(
                    fold_id=int(fold_id),
                    timestamps=ts_keep,
                    y_true=y_keep,
                    y_pred=preds_keep,
                    metadata={
                        "global_multi_symbol": 1,
                        "n_symbols": int(len(symbols)),
                        "mamba_seq_len": int(seq_len),
                        "mamba_d_model": int(cfg["mamba_d_model"]),
                        "mamba_n_layers": int(cfg["mamba_n_layers"]),
                    },
                )
                tape_folds_by_symbol[sym].append(fold_df)

                # Portfolio return proxy for logging (sign(pred) * return)
                strat = np.sign(preds_keep) * y_keep
                if strat.size > 0:
                    symbol_return_series.append(pd.Series(strat.astype(float), index=ts_keep))

                del out_X_tensor

        # Equal-weight portfolio returns and pooled non-overlap samples
        if symbol_return_series:
            df = pd.concat(symbol_return_series, axis=1).fillna(0.0)
            agg = df.mean(axis=1).sort_index()
            arr = agg.to_numpy(dtype=float)
            if int(horizon) > 1 and arr.size >= int(horizon):
                non_overlap = arr[:: int(horizon)]
            else:
                non_overlap = arr
            pooled_non_overlap_returns.extend([float(x) for x in non_overlap.tolist()])

        del model, result
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    pooled_sharpe = _portfolio_sharpe(pooled_non_overlap_returns)
    meta = {
        "global_multi_symbol": True,
        "symbols": symbols,
        "n_symbols": int(len(symbols)),
        "folds": int(len(walk_forward_folds_multi)),
        "mamba_seq_len": int(seq_len),
        "mamba_seq_len_cap": int(cap),
        "portfolio_sharpe_sign_pred": float(pooled_sharpe),
        "threshold_params": dict(threshold_params),
    }
    return tape_folds_by_symbol, meta


if RAY_AVAILABLE:

    @ray.remote(num_cpus=1, num_gpus=0.167)  # Default; overridden by .options() at call time
    def _OptunaFoldTrainTask(
        seq_data,  # SequenceData object (passed directly, Ray handles serialization)
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        cfg: Dict[str, Any],  # Config dict (passed directly)
        fold_idx: int,
        max_memory_gb: float = 1.0,  # Max GPU memory per fold
    ) -> Dict[str, Any]:
        """Ray task that trains a single fold and returns predictions."""
        import os
        import torch
        logger = logging.getLogger("stage_b.optuna.fold")
        
        try:
            # Debug: Check CUDA visibility in this worker
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            if fold_idx == 0:
                logger.info(f"Fold task CUDA check: CUDA_VISIBLE_DEVICES={cuda_visible}, "
                           f"cuda_available={cuda_available}, device_count={device_count}")
            
            # Force device to cuda:0 if available
            if cuda_available and device_count > 0:
                device = torch.device("cuda:0")
                
                # Log GPU memory info (no limit - let PyTorch manage memory naturally)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                if fold_idx == 0:
                    logger.info(f"Fold task using device: {device}, total_gpu_mem={total_mem:.1f}GB")
            else:
                device = torch.device("cpu")
                if fold_idx == 0:
                    logger.warning(f"Fold task falling back to CPU!")
            
            # seq_data and cfg are passed directly (Ray handles serialization)
            train_fn = train_mamba_fold if cfg.get("sequence_model_type") == "mamba" else train_lstm_fold
            result = train_fn(seq_data, train_idx, val_idx, cfg, device=device)
            
            # 🔧 Release GPU memory after fold training completes
            if cuda_available:
                torch.cuda.empty_cache()
            
            return {"fold_idx": fold_idx, "result": result}
        except Exception as exc:  # pragma: no cover - Ray worker
            logger.error("Fold %d Ray worker failed: %s", fold_idx, exc)
            import traceback
            return {"fold_idx": fold_idx, "result": None, "error": str(exc), "traceback": traceback.format_exc()}

    @ray.remote(num_cpus=2, num_gpus=0.45)  # Batched task: trains multiple folds sequentially
    def _OptunaFoldBatchTask(
        track_c_values: np.ndarray,  # Raw feature values (n_samples, n_features)
        track_c_index: np.ndarray,  # DatetimeIndex as numpy array
        actual_returns_values: np.ndarray,  # Returns values
        fold_specs: List[Tuple[int, np.ndarray, np.ndarray]],  # List of (fold_idx, train_idx, val_idx)
        cfg: Dict[str, Any],
        seq_len: int,
    ) -> List[Dict[str, Any]]:
        """
        🚀 BATCHED Ray task: Trains MULTIPLE folds in a single worker.
        
        This dramatically reduces overhead by:
        1. Single worker spawn for N folds (vs N worker spawns)
        2. GPU stays warm across folds (no context switching)
        3. Data is in worker memory (no per-fold serialization)
        4. build_sequence_data runs inside worker (parallel CPU+GPU)
        """
        import os
        import torch
        import pandas as pd
        logger = logging.getLogger("stage_b.optuna.batch")
        
        results = []
        
        try:
            # Setup GPU once for all folds in this batch
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            if cuda_available and device_count > 0:
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
                logger.warning("Batch task falling back to CPU!")
            
            # Reconstruct DataFrames from numpy arrays
            track_c_df = pd.DataFrame(
                track_c_values,
                index=pd.DatetimeIndex(track_c_index),
            )
            actual_returns_series = pd.Series(
                actual_returns_values,
                index=pd.DatetimeIndex(track_c_index),
            )
            
            # Train each fold in this batch
            for fold_idx, fold_train_idx, fold_val_idx in fold_specs:
                try:
                    # =====================================================
                    # Fold-specific dataset build
                    # - Train sequences from fold_train
                    # - Validation sequences from fold_val (true walk-forward)
                    # =====================================================
                    train_seq_result = build_sequence_data(
                        track_c_df.iloc[fold_train_idx],
                        actual_returns_series.iloc[fold_train_idx],
                        seq_len,
                    )
                    if train_seq_result is None:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "train_seq_data too small"})
                        continue
                    train_seq_data, _ = train_seq_result
                    if len(train_seq_data) < 20:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "train_seq_data too small"})
                        continue

                    val_seq_result = build_sequence_data(
                        track_c_df.iloc[fold_val_idx],
                        actual_returns_series.iloc[fold_val_idx],
                        seq_len,
                    )
                    if val_seq_result is None:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "val_seq_data too small"})
                        del train_seq_data
                        continue
                    val_seq_data, _ = val_seq_result
                    if len(val_seq_data) < 5:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "val_seq_data too small"})
                        del train_seq_data, val_seq_data
                        continue

                    # =====================================================
                    # Fresh model per fold (no warm start)
                    # Train uses an internal holdout split inside fold_train.
                    # =====================================================
                    n_train_seq = len(train_seq_data)
                    train_end = int(n_train_seq * 0.9)
                    seq_train_idx = np.arange(train_end)
                    seq_holdout_idx = np.arange(train_end, n_train_seq)
                    if len(seq_holdout_idx) < 5:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "train_holdout too small"})
                        del train_seq_data, val_seq_data
                        continue

                    train_fn = train_mamba_fold if cfg.get("sequence_model_type") == "mamba" else train_lstm_fold
                    train_result = train_fn(
                        train_seq_data,
                        seq_train_idx,
                        seq_holdout_idx,
                        cfg,
                        device=device,
                    )

                    # =====================================================
                    # Predict fold validation window (true OOS)
                    # =====================================================
                    state_dict = train_result.get("state_dict")
                    if state_dict is None:
                        results.append({"fold_idx": fold_idx, "result": None, "error": "missing_state_dict"})
                        del train_seq_data, val_seq_data
                        continue

                    if cfg.get("sequence_model_type") == "mamba":
                        from src.stage_b.sequence_models import predict_mamba_on_data
                        pred_out = predict_mamba_on_data(val_seq_data, state_dict, cfg, device=device)
                    else:
                        from src.stage_b.sequence_models import predict_lstm_on_data
                        pred_out = predict_lstm_on_data(val_seq_data, state_dict, cfg, device=device)

                    results.append({
                        "fold_idx": fold_idx,
                        "result": {
                            "preds": pred_out["preds"],
                            "timestamps": pred_out["timestamps"],
                            "state_dict": state_dict,
                            "val_loss": float(train_result.get("val_loss", float("inf"))),
                        },
                    })

                    del train_seq_data, val_seq_data
                    
                except Exception as fold_err:
                    import traceback
                    results.append({
                        "fold_idx": fold_idx,
                        "result": None,
                        "error": str(fold_err),
                        "traceback": traceback.format_exc(),
                    })
            
            # Clear GPU cache once at end of batch
            if cuda_available:
                torch.cuda.empty_cache()
                
        except Exception as batch_err:
            import traceback
            logger.error(f"Batch task failed: {batch_err}")
            # Return error for all folds in batch
            for fold_idx, _, _ in fold_specs:
                results.append({
                    "fold_idx": fold_idx,
                    "result": None,
                    "error": str(batch_err),
                    "traceback": traceback.format_exc(),
                })
        
        return results

    @ray.remote
    class _OptunaTrialWorker:
        def __init__(
            self,
            shared_refs: Dict[str, "ray.ObjectRef"],
            config_dict: Dict[str, Any],
            family_columns: Dict[str, List[str]],
            family_sizes: Dict[str, int],
            stage_a_weights: Dict[str, float],
            log_level: int,
        ):
            self.logger = logging.getLogger("stage_b.optuna.worker")
            self.logger.setLevel(log_level)

            # Check PyTorch/CUDA availability in worker
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                device_count = torch.cuda.device_count() if cuda_available else 0
                self.logger.info(f"Worker PyTorch check: cuda={cuda_available}, devices={device_count}")
            except Exception as e:
                self.logger.info(f"Worker PyTorch import failed: {e}")

            self.logger.info(f"Worker init: received {len(family_columns)} families, sizes={len(family_sizes)}")

            config = OptunaConfig(**config_dict)
            self.optimizer = StageBOptunaOptimizer(config=config, logger=self.logger)
            self.optimizer.family_columns = family_columns
            self.optimizer.family_sizes = family_sizes
            self.optimizer.stage_a_weights = stage_a_weights
            
            # Flag to indicate we're inside a Ray actor - disables nested Ray task spawning
            self.optimizer._inside_ray_actor = True

            self.panel = ray.get(shared_refs["panel"])
            self.column_families = ray.get(shared_refs["column_families"])
            self.labels = ray.get(shared_refs["labels"])
            self.block_summaries = ray.get(shared_refs["block_summaries"])
            self.walk_forward_folds = ray.get(shared_refs["walk_forward_folds"])

        def run(self, payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
            try:
                score, metadata = self.optimizer._evaluate_trial_params(
                    self.panel,
                    self.column_families,
                    self.labels,
                    self.block_summaries,
                    self.walk_forward_folds,
                    payload["family_params"],
                    payload["weight_a"],
                    payload["weight_b"],
                    payload["sequence_model_type"],
                    payload["lstm_params"],
                    payload["mamba_params"],
                    payload["threshold_params"],
                    payload["pipeline_params"],
                    payload["horizon"],
                    track_b_family_weights=payload.get("track_b_family_weights") or {},
                )
                self.logger.info(f"Worker run completed: score={score}")
                return score, metadata
            except Exception as e:
                import traceback
                import sys
                err_msg = f"Worker _evaluate_trial_params failed: {e}\n{traceback.format_exc()}"
                self.logger.error(err_msg)
                print(err_msg, file=sys.stderr, flush=True)
                return float("-inf"), {"error": str(e)}

# =============================================================================
# Integration Functions
# =============================================================================

def run_optuna_optimization(
    panel: pd.DataFrame,
    column_families: Dict[str, str],
    labels: pd.DataFrame,
    block_summaries: Dict[str, pd.DataFrame],
    walk_forward_folds: List[Tuple[np.ndarray, np.ndarray]],
    horizon: int = 1,
    config: Optional[OptunaConfig] = None,
    stage_a_weights: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> OptimizedParams:
    """Run full Optuna optimization with WALK-FORWARD and return best parameters.
    
    ONE TRIAL = ENTIRE WALK-FORWARD (~60 folds)
    
    This is the main entry point for Optuna optimization.
    
    Args:
        panel: Full feature panel
        column_families: Column to family mapping
        labels: Labels DataFrame with 'forward_return'
        block_summaries: Stage-B block summaries
        walk_forward_folds: List of (train_idx, val_idx) for ALL WF windows
        horizon: Forecast horizon
        config: Optional OptunaConfig
        stage_a_weights: Optional Stage-A family weights
        output_path: Optional path to save best params
    
    Returns:
        OptimizedParams with best configuration
    """
    optimizer = StageBOptunaOptimizer(config=config)
    
    params = optimizer.optimize(
        panel=panel,
        column_families=column_families,
        labels=labels,
        block_summaries=block_summaries,
        walk_forward_folds=walk_forward_folds,
        horizon=horizon,
        stage_a_weights=stage_a_weights,
    )
    
    if output_path is not None:
        optimizer.save_params(output_path)
    
    return params


def run_optuna_optimization_multi_symbol(
    *,
    panels_by_symbol: Dict[str, pd.DataFrame],
    column_families: Dict[str, str],
    labels_by_symbol: Dict[str, pd.DataFrame],
    block_summaries_by_symbol: Dict[str, Dict[str, pd.DataFrame]],
    walk_forward_folds_multi: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    horizon: int = 1,
    config: Optional[OptunaConfig] = None,
    stage_a_weights: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> OptimizedParams:
    """Run Optuna optimization in global multi-symbol pooled mode."""
    optimizer = StageBOptunaOptimizer(config=config)
    params = optimizer.optimize_multi_symbol(
        panels_by_symbol=panels_by_symbol,
        column_families=column_families,
        labels_by_symbol=labels_by_symbol,
        block_summaries_by_symbol=block_summaries_by_symbol,
        walk_forward_folds_multi=walk_forward_folds_multi,
        horizon=horizon,
        stage_a_weights=stage_a_weights,
    )
    if output_path is not None:
        optimizer.save_params(output_path)
    return params


def apply_optimized_params(
    panel: pd.DataFrame,
    column_families: Dict[str, str],
    block_summaries: Dict[str, pd.DataFrame],
    params: OptimizedParams,
    train_idx: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, FamilyEncoder], Dict[str, Any]]:
    """Apply optimized parameters to build Track C features.
    
    Args:
        panel: Full feature panel
        column_families: Column to family mapping
        block_summaries: Stage-B block summaries
        params: OptimizedParams from optimization
        train_idx: Training indices for fitting encoders
    
    Returns:
        Tuple of (track_c_frame, fitted_encoders, track_metadata)
    """
    optimizer = StageBOptunaOptimizer()
    optimizer._analyze_families(panel, column_families)
    
    # Build family params from OptimizedParams (4-tuple with weight)
    family_params: Dict[str, Tuple[bool, int, str, float]] = {}
    for family in STAGE_A_FAMILIES:
        include = params.included_families.get(family, False)
        dim = params.family_dimensions.get(family, 0)
        method = params.family_encoders.get(family, "pca")
        # Use family_weights if available, otherwise default to 1.0 for backward compat
        weight = params.family_weights.get(family, 1.0 if include else 0.0)
        family_params[family] = (include, dim, method, weight)
    
    # Build tracks
    track_a, encoders = optimizer._build_track_a(
        panel, column_families, family_params, train_idx
    )
    track_b = optimizer._build_track_b(
        panel,
        column_families,
        block_summaries,
        track_b_family_weights=getattr(params, "track_b_family_weights", None) or {},
    )
    track_meta = {
        "track_a_cols": track_a.columns.tolist() if not track_a.empty else [],
        "track_b_cols": track_b.columns.tolist() if not track_b.empty else [],
        "track_a_weight": params.track_a_weight,
        "track_b_weight": params.track_b_weight,
    }
    # Return the unweighted Track C frame; callers can apply weights after any
    # downstream normalization/sanitization to guarantee the final tensors
    # reflect Optuna's decisions.
    track_c = optimizer._build_track_c(
        track_a,
        track_b,
        1.0,
        1.0,
    )
    
    return track_c, encoders, track_meta


__all__ = [
    "OptunaConfig",
    "OptimizedParams",
    "StageBOptunaOptimizer",
    "FamilyEncoder",
    "run_optuna_optimization",
    "apply_optimized_params",
]
