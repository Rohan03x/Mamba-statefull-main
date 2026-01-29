from __future__ import annotations

import json
import logging
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.dcf_lab.utils.trading_calendar import add_sessions
from src.stage_b.backtest import BacktestEngine
from src.stage_b.pipeline import GLOBAL_OPTUNA_SYMBOLS, StageBConfig, StageBPipeline
from src.stage_b.sequence_models import (
    MultiSymbolGPUMasterStore,
    SequenceData,
    WindowedSequenceData,
    build_master_arrays,
    build_sequence_data,
    train_mamba_fold,
    # Level 2: Cross-Section Mamba
    CrossSectionSequenceData,
    build_cross_section_data,
    CrossSectionMamba,
    train_cross_section_mamba,
    CrossSectionInferenceBuffer,
    compute_vram_requirements,
    # Multi-Horizon (Workstream 6)
    MultiHorizonSequenceData,
    MultiHorizonMambaRegressor,
    train_multi_horizon_mamba,
    build_multi_horizon_data,
)
from src.stage_b_stateful.maturity import maturity_cutoff
from src.stage_b_stateful.mamba_calibration import (
    MambaCalibrationTracker,
    MambaCalibrationSnapshot,
    create_mamba_calibration_tracker,
    MAMBA_CALIB_WINDOW,
)
from src.stage_b_stateful.governance_persistence import (
    GovernanceStateKey,
    make_state_key,
    save_mamba_calibration_state,
    load_mamba_calibration_state,
    save_online_learning_state,
    load_online_learning_state,
)
from src.portfolio.policy_controller import (
    PolicyAction,
    PolicyController,
    POLICY_STATE_DIM_V2,
    build_default_policy_actions_v2,
)
from src.portfolio.linear_alpha_combiner import (
    LinearCombinerState,
    LinearFeatureBuilder,
    RidgeModel,
    create_linear_combiner,
    ALL_FEATURE_NAMES,
)
from src.portfolio.linear_alpha_persistence import (
    save_linear_combiner_state,
    load_linear_combiner_state,
)
from src.features.symbol_graph_context import (
    build_sgc_for_multi_symbol,
    get_symbol_graph_context_columns,
)


logger = logging.getLogger(__name__)

# Disk cache directory for 3-pillar analysis results
_THREE_PILLAR_CACHE_DIR = Path("artifacts/three_pillar_cache")
_THREE_PILLAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _three_pillar_cache_key(
    symbols: Sequence[str],
    horizon: int,
    train_start: str,
    train_end: str,
    three_pillar_dim_max: int,
    three_pillar_pca_variance: float,
    stage_a_weights: Optional[Dict[str, float]],
) -> str:
    """Generate cache key for 3-pillar analysis results."""
    syms_str = ",".join(sorted(str(s).upper() for s in symbols))
    weights_str = json.dumps(stage_a_weights or {}, sort_keys=True)
    key_data = f"{syms_str}|{horizon}|{train_start}|{train_end}|{three_pillar_dim_max}|{three_pillar_pca_variance:.6f}|{weights_str}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _load_three_pillar_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load 3-pillar analysis results from disk cache."""
    cache_file = _THREE_PILLAR_CACHE_DIR / f"three_pillar_{cache_key}.pkl"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        logger.info(f"[3-pillar] ✅ DISK CACHE HIT - loaded from {cache_file.name}")
        return data
    except Exception as e:
        logger.warning(f"[3-pillar] Failed to load cache {cache_file.name}: {e}")
        return None


def _save_three_pillar_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Save 3-pillar analysis results to disk cache."""
    cache_file = _THREE_PILLAR_CACHE_DIR / f"three_pillar_{cache_key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[3-pillar] 💾 Saved to disk cache: {cache_file.name}")
    except Exception as e:
        logger.warning(f"[3-pillar] Failed to save cache {cache_file.name}: {e}")



def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if not bool(np.any(m)):
        return 0.0
    x = x[m]
    y = y[m]
    if x.size < 3:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _safe_spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        m = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(m)) < 5:
            return float("nan")
        ar = pd.Series(a[m]).rank(method="average").to_numpy(dtype=float)
        br = pd.Series(b[m]).rank(method="average").to_numpy(dtype=float)
        return float(_safe_corr(ar, br))
    except Exception:
        return float("nan")


def _xsec_entropy_from_scores(scores: np.ndarray, *, eps: float = 1e-12) -> float:
    """Normalized entropy in [0,1] using |scores| as a probability proxy."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    s = np.abs(s[np.isfinite(s)])
    if s.size < 3:
        return float("nan")
    z = float(np.sum(s))
    if not np.isfinite(z) or z <= 0:
        return float("nan")
    p = s / (z + float(eps))
    h = -float(np.sum(p * np.log(p + float(eps))))
    hmax = float(np.log(float(p.size) + float(eps)))
    if not np.isfinite(hmax) or hmax <= 0:
        return float("nan")
    return float(h / hmax)


def _compute_universe_rotation_diagnostics(
    *,
    equity_by_symbol: Mapping[str, pd.DataFrame],
    benchmark_returns: pd.Series,
    weight_threshold: float = 1e-4,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute universe rotation diagnostics from realized per-symbol weights/returns.

    Uses execution weights (lagged by 1 day) to mirror PnL realization.
    """

    if not equity_by_symbol:
        return pd.DataFrame(), {}

    syms = list(equity_by_symbol.keys())
    idx = None
    for s in syms:
        df = equity_by_symbol.get(s)
        if isinstance(df, pd.DataFrame) and not df.empty:
            idx = df.index
            break
    if idx is None:
        return pd.DataFrame(), {}

    w = pd.DataFrame(index=idx)
    r = pd.DataFrame(index=idx)
    for s in syms:
        df = equity_by_symbol.get(s)
        if df is None or df.empty:
            continue
        wcol = df["weight"] if "weight" in df.columns else pd.Series(0.0, index=df.index)
        rcol = df["return"] if "return" in df.columns else pd.Series(0.0, index=df.index)
        w[str(s).upper()] = pd.to_numeric(wcol, errors="coerce").fillna(0.0)
        r[str(s).upper()] = pd.to_numeric(rcol, errors="coerce").fillna(0.0)

    w = w.fillna(0.0)
    r = r.fillna(0.0)
    w_exec = w.shift(1).fillna(0.0)

    active = (w_exec.abs() > float(weight_threshold)).astype(int)
    active_count = active.sum(axis=1).astype(float)

    prev_active = active.shift(1).fillna(0).astype(int)
    entrants = ((active == 1) & (prev_active == 0)).sum(axis=1).astype(float)
    exits = ((active == 0) & (prev_active == 1)).sum(axis=1).astype(float)
    churn_rate = (entrants + exits).divide(active_count.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Weight-based turnover proxy (matches classic definition).
    turnover = 0.5 * (w_exec.diff().abs().sum(axis=1)).fillna(0.0)

    # Rank stability: Spearman correlation of weight ranks day-over-day.
    rank_stab = []
    w_exec_np = w_exec.to_numpy(dtype=float)
    for i in range(len(w_exec.index)):
        if i <= 0:
            rank_stab.append(float("nan"))
            continue
        rank_stab.append(_safe_spearman_rank_corr(w_exec_np[i - 1], w_exec_np[i]))
    rank_stability = pd.Series(rank_stab, index=w_exec.index, name="rank_stability")

    # Entrant vs incumbent performance (pnl and active pnl).
    rb = pd.to_numeric(benchmark_returns, errors="coerce").reindex(w_exec.index).fillna(0.0)
    pnl_sym = w_exec * r
    active_pnl_sym = w_exec * (r.sub(rb, axis=0))

    entrants_mask = (active == 1) & (prev_active == 0)
    incumbents_mask = (active == 1) & (prev_active == 1)
    entrants_pnl = pnl_sym.where(entrants_mask).sum(axis=1).fillna(0.0)
    incumbents_pnl = pnl_sym.where(incumbents_mask).sum(axis=1).fillna(0.0)
    entrants_active_pnl = active_pnl_sym.where(entrants_mask).sum(axis=1).fillna(0.0)
    incumbents_active_pnl = active_pnl_sym.where(incumbents_mask).sum(axis=1).fillna(0.0)

    out = pd.DataFrame(
        {
            "active_count": active_count,
            "entrants": entrants,
            "exits": exits,
            "churn_rate": churn_rate,
            "turnover_proxy": turnover,
            "rank_stability": rank_stability,
            "entrants_pnl": entrants_pnl,
            "incumbents_pnl": incumbents_pnl,
            "entrants_active_pnl": entrants_active_pnl,
            "incumbents_active_pnl": incumbents_active_pnl,
        },
        index=w_exec.index,
    )

    summary: Dict[str, float] = {}
    try:
        summary["avg_active_count"] = float(active_count.mean()) if len(active_count) else 0.0
        summary["avg_churn_rate"] = (
            float(pd.to_numeric(out["churn_rate"], errors="coerce").dropna().mean()) if len(out) else 0.0
        )
        summary["avg_turnover_proxy"] = float(turnover.mean()) if len(turnover) else 0.0
        rs = pd.to_numeric(out["rank_stability"], errors="coerce").dropna()
        summary["rank_stability_mean"] = float(rs.mean()) if not rs.empty else float("nan")
        summary["rank_stability_last"] = float(rs.iloc[-1]) if not rs.empty else float("nan")
        summary["entrants_active_pnl_sum"] = float(out["entrants_active_pnl"].sum())
        summary["incumbents_active_pnl_sum"] = float(out["incumbents_active_pnl"].sum())
    except Exception:
        pass

    return out, summary


def _house_policy_report(
    *,
    ts: pd.DataFrame,
    windows: Sequence[int],
    policy_max_drawdown: float,
    policy_min_ir: float,
    policy_beta_abs_max: float,
) -> Dict[str, Any]:
    """Evaluate a simple 'house policy' envelope from benchmark time-series."""

    active = pd.to_numeric(ts.get("active"), errors="coerce").dropna()
    dd = pd.to_numeric(ts.get("drawdown"), errors="coerce").dropna()
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    w0 = int(windows[1]) if len(windows) > 1 else int(windows[0])
    beta_col = f"beta_{w0}"
    ir_col = f"ir_{w0}"
    beta_last = float(pd.to_numeric(ts.get(beta_col), errors="coerce").dropna().iloc[-1]) if beta_col in ts.columns else float("nan")
    ir_last = float(pd.to_numeric(ts.get(ir_col), errors="coerce").dropna().iloc[-1]) if ir_col in ts.columns else float("nan")

    try:
        te = float(active.std() * np.sqrt(252)) if len(active) > 5 else float("nan")
        ir_full = float((active.mean() / (active.std() + 1e-12)) * np.sqrt(252)) if len(active) > 5 else float("nan")
    except Exception:
        te = float("nan")
        ir_full = float("nan")

    breaches: Dict[str, Any] = {}
    breaches["max_drawdown"] = {
        "value": float(max_dd) if np.isfinite(max_dd) else None,
        "limit": -abs(float(policy_max_drawdown)),
        "ok": bool(np.isfinite(max_dd) and max_dd >= -abs(float(policy_max_drawdown))),
    }
    breaches["beta_abs"] = {
        "value": float(beta_last) if np.isfinite(beta_last) else None,
        "limit": abs(float(policy_beta_abs_max)),
        "ok": bool(np.isfinite(beta_last) and abs(beta_last) <= abs(float(policy_beta_abs_max))),
    }
    breaches["ir_full"] = {
        "value": float(ir_full) if np.isfinite(ir_full) else None,
        "limit": float(policy_min_ir),
        "ok": bool(np.isfinite(ir_full) and ir_full >= float(policy_min_ir)),
    }
    breaches["ir_last"] = {
        "value": float(ir_last) if np.isfinite(ir_last) else None,
        "limit": float(policy_min_ir),
        "ok": bool(np.isfinite(ir_last) and ir_last >= float(policy_min_ir)),
    }

    ok_all = bool(all(bool(v.get("ok")) for v in breaches.values()))
    return {
        "policy": {
            "max_drawdown": float(policy_max_drawdown),
            "min_ir": float(policy_min_ir),
            "beta_abs_max": float(policy_beta_abs_max),
            "anchor_window": int(w0),
        },
        "stats": {
            "max_drawdown": float(max_dd) if np.isfinite(max_dd) else None,
            "beta_last": float(beta_last) if np.isfinite(beta_last) else None,
            "ir_full": float(ir_full) if np.isfinite(ir_full) else None,
            "tracking_error_full": float(te) if np.isfinite(te) else None,
        },
        "breaches": breaches,
        "ok": ok_all,
    }


def _sigma_health(mu: np.ndarray, sigma: np.ndarray, *, eps: float = 1e-3) -> Dict[str, float]:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    m = np.isfinite(mu) & np.isfinite(sigma)
    if not bool(np.any(m)):
        return {
            "sigma_median": float("nan"),
            "sigma_p90": float("nan"),
            "sigma_p99": float("nan"),
            "sigma_collapse_pct": float("nan"),
            "corr_abs_mu_sigma": float("nan"),
        }
    mu = mu[m]
    sigma = sigma[m]
    s_med = float(np.median(sigma)) if sigma.size else float("nan")
    s_p90 = float(np.percentile(sigma, 90)) if sigma.size else float("nan")
    s_p99 = float(np.percentile(sigma, 99)) if sigma.size else float("nan")
    collapse = float(np.mean(sigma <= float(eps))) if sigma.size else float("nan")
    corr = float(_safe_corr(np.abs(mu), sigma))
    return {
        "sigma_median": s_med,
        "sigma_p90": s_p90,
        "sigma_p99": s_p99,
        "sigma_collapse_pct": collapse,
        "corr_abs_mu_sigma": corr,
    }


def _effective_n_bets_from_weights(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).reshape(-1)
    gross = float(np.sum(np.abs(w)))
    if not np.isfinite(gross) or gross <= 1e-12:
        return 0.0
    denom = float(np.sum(w * w))
    if not np.isfinite(denom) or denom <= 1e-18:
        return 0.0
    return float((gross * gross) / denom)


# ---------------------------------------------------------------------------
# HF Block Confidence Synthesis (Phase-2 Intermediate Step)
# ---------------------------------------------------------------------------
# HF blocks sit BETWEEN Mamba inference and portfolio allocation.
# They annotate, NOT replace. They cannot flip Mamba direction.
#
# Architecture:
#   1. Load base family parquets
#   2. Run Mamba → z_mamba(t)
#   3. Load HF block parquets  ← HF synthesis starts here
#   4. Compute stateful confidence & stress
#   5. Produce final signal package
#   6. Hand off to portfolio engine
#
# HF blocks are model outputs (meta-opinions), NOT features.
# They do NOT participate in feature_roles.py.
# ---------------------------------------------------------------------------

@dataclass
class HFBlockContext:
    """HF block synthesis context for a single day.
    
    Contains aggregated confidence and optional stress from HF blocks.
    Used to modulate z_mamba → z_final without flipping direction.
    """
    # Aggregated confidence from all HF blocks
    confidence: np.ndarray  # shape (n_symbols,), in [0, 1]
    # Disagreement penalty (1.0 = full agreement, <1.0 = disagreement)
    agreement_factor: np.ndarray  # shape (n_symbols,), in [0, 1]
    # Final exposure scaler: alpha = confidence * agreement_factor
    alpha: np.ndarray  # shape (n_symbols,), in [alpha_min, 1.0]
    # HF uncertainty as stress (1 - confidence * agreement)
    hf_stress: np.ndarray  # shape (n_symbols,), in [0, 1]
    # Symbol list for alignment
    symbols: List[str]


def _load_hf_merged_panel(
    symbol: str,
    horizon: int,
    *,
    hf_blocks: Sequence[str] = ("tech_micro_hf", "forecast_hf"),
) -> Optional[pd.DataFrame]:
    """Load the HF merged parquet for a symbol/horizon.
    
    Returns DataFrame with columns: {block}_score, {block}_conf for each HF block.
    Returns None if not available.
    """
    from src.cache_paths import symbol_hf_merged_path, hf_block_path, hf_block_horizon_path
    
    merged_path = symbol_hf_merged_path(symbol, horizon)
    if merged_path.exists():
        try:
            return pd.read_parquet(merged_path)
        except Exception:
            pass
    
    # Fallback: load individual HF blocks and merge
    dfs = []
    for block in hf_blocks:
        # Try horizon-specific first (for horizon-bound blocks like forecast_hf)
        p = hf_block_horizon_path(symbol, horizon, block)
        if not p.exists():
            # Fall back to horizon-invariant
            p = hf_block_path(symbol, block)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                # Rename columns to have block prefix if not already
                cols_map = {}
                for c in df.columns:
                    if not str(c).startswith(block):
                        if c in ("score", "conf", "confidence", "score_raw"):
                            cols_map[c] = f"{block}_{c}"
                if cols_map:
                    df = df.rename(columns=cols_map)
                dfs.append(df)
            except Exception:
                continue
    
    if not dfs:
        return None
    
    # Merge on index
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="outer", rsuffix="_dup")
    return result


def _synthesize_hf_confidence(
    hf_panels: Dict[str, pd.DataFrame],
    day: pd.Timestamp,
    symbols: Sequence[str],
    *,
    hf_blocks: Sequence[str] = ("tech_micro_hf", "forecast_hf"),
    block_weights: Optional[Dict[str, float]] = None,
    disagreement_k: float = 2.0,
    alpha_min: float = 0.15,
) -> HFBlockContext:
    """Synthesize HF block confidence for a single day.
    
    HF blocks provide meta-opinions (score, confidence) that modulate exposure.
    They NEVER flip Mamba direction — only scale it down via alpha.
    
    Formal math:
        C_i(t) = Σ(w_b × c_i^b(t))  # weighted confidence
        Δs_i(t) = |s_i^a(t) - s_i^b(t)|  # disagreement (pairwise)
        P_i(t) = exp(-k × max_disagreement)  # penalty
        Ĉ_i(t) = C_i(t) × P_i(t)  # final confidence
        α_i(t) = clamp(Ĉ_i(t), α_min, 1.0)  # exposure scaler
        z_final(t) = α_i(t) × z_mamba(t)
    
    Properties:
        - sign(z_final) = sign(z_mamba) always
        - HF blocks cannot flip direction
        - Worst case: exposure shrinks toward zero
    """
    n = len(symbols)
    
    # Default weights if not provided
    if block_weights is None:
        n_blocks = len(hf_blocks)
        block_weights = {b: 1.0 / n_blocks for b in hf_blocks}
    
    # Normalize weights
    w_total = sum(block_weights.values())
    if w_total > 0:
        block_weights = {b: w / w_total for b, w in block_weights.items()}
    
    # Extract per-symbol scores and confidences
    scores_by_block: Dict[str, np.ndarray] = {}
    confs_by_block: Dict[str, np.ndarray] = {}
    
    for block in hf_blocks:
        scores = np.full(n, 0.5, dtype=float)  # neutral default
        confs = np.zeros(n, dtype=float)  # zero confidence if missing
        
        for j, sym in enumerate(symbols):
            sym_upper = str(sym).upper()
            panel = hf_panels.get(sym_upper)
            if panel is None or panel.empty:
                continue
            
            # Get row for this day
            if day in panel.index:
                row = panel.loc[day]
            else:
                # Try closest earlier date
                mask = panel.index <= day
                if mask.any():
                    row = panel.loc[panel.index[mask][-1]]
                else:
                    continue
            
            # Extract score and confidence
            score_col = f"{block}_score"
            conf_col = f"{block}_conf"
            if conf_col not in row.index:
                conf_col = f"{block}_confidence"
            
            if score_col in row.index:
                val = float(row[score_col])
                if np.isfinite(val):
                    scores[j] = np.clip(val, 0.0, 1.0)
            if conf_col in row.index:
                val = float(row[conf_col])
                if np.isfinite(val):
                    confs[j] = np.clip(val, 0.0, 1.0)
        
        scores_by_block[block] = scores
        confs_by_block[block] = confs
    
    # Weighted confidence aggregation
    # C_i(t) = Σ(w_b × c_i^b(t))
    confidence = np.zeros(n, dtype=float)
    for block in hf_blocks:
        w = block_weights.get(block, 0.0)
        confidence += w * confs_by_block.get(block, np.zeros(n))
    
    # Disagreement penalty (optional, recommended)
    # Δs_i(t) = max over pairs |s_i^a - s_i^b|
    max_disagreement = np.zeros(n, dtype=float)
    blocks_list = list(hf_blocks)
    for i in range(len(blocks_list)):
        for j in range(i + 1, len(blocks_list)):
            s_a = scores_by_block.get(blocks_list[i], np.full(n, 0.5))
            s_b = scores_by_block.get(blocks_list[j], np.full(n, 0.5))
            disagreement = np.abs(s_a - s_b)
            max_disagreement = np.maximum(max_disagreement, disagreement)
    
    # P_i(t) = exp(-k × Δs_i(t))
    agreement_factor = np.exp(-disagreement_k * max_disagreement)
    
    # Final confidence: Ĉ_i(t) = C_i(t) × P_i(t)
    final_confidence = confidence * agreement_factor
    
    # Exposure scaler: α_i(t) = clamp(Ĉ_i(t), α_min, 1.0)
    alpha = np.clip(final_confidence, alpha_min, 1.0)
    
    # HF stress = 1 - final_confidence (for stress aggregator)
    hf_stress = 1.0 - final_confidence
    
    return HFBlockContext(
        confidence=confidence,
        agreement_factor=agreement_factor,
        alpha=alpha,
        hf_stress=hf_stress,
        symbols=list(symbols),
    )


def _apply_hf_modulation(
    z_mamba: np.ndarray,
    hf_ctx: HFBlockContext,
    symbols: Sequence[str],
) -> np.ndarray:
    """Apply HF confidence modulation to Mamba z-scores.
    
    z_final(t) = α_i(t) × z_mamba(t)
    
    CRITICAL: This preserves sign(z_mamba) — HF blocks cannot flip direction.
    """
    z = np.asarray(z_mamba, dtype=float).copy()
    
    # Build symbol -> index map for HF context
    hf_idx_map = {str(s).upper(): j for j, s in enumerate(hf_ctx.symbols)}
    
    for j, sym in enumerate(symbols):
        sym_upper = str(sym).upper()
        hf_j = hf_idx_map.get(sym_upper)
        if hf_j is not None and hf_j < len(hf_ctx.alpha):
            z[j] *= hf_ctx.alpha[hf_j]
    
    return z


# ---------------------------------------------------------------------------
# Default Phase-2 date blocks (as requested)
# ---------------------------------------------------------------------------

# Feature-cache coverage across GLOBAL13 Track-C panels:
#   2005-07-02 .. 2025-06-20
DEFAULT_PHASE2_TRAIN_START = "2005-07-02"
DEFAULT_PHASE2_TRAIN_END = "2020-12-31"
DEFAULT_PHASE2_OOS_START = "2021-01-01"
DEFAULT_PHASE2_OOS_END = "2023-12-31"
DEFAULT_PHASE2_HOLDOUT_START = "2024-01-01"
DEFAULT_PHASE2_HOLDOUT_END = "2025-06-20"

# ---------------------------------------------------------------------------
# Multi-Horizon Training (Workstream 6)
# ---------------------------------------------------------------------------
# multi_horizon_mode: when True, trains on multiple horizons simultaneously
# multi_horizon_set: horizons used for training (short→long for representation learning)
# multi_horizon_primary: the horizon whose predictions are used for downstream
#                        (portfolio, policy controller, logging)
# All other horizons are training-only regularizers.
DEFAULT_MULTI_HORIZON_MODE = True
DEFAULT_MULTI_HORIZON_SET = [5, 21, 63, 126]
DEFAULT_MULTI_HORIZON_PRIMARY = 63


@dataclass(frozen=True)
class Phase2RefinementSpec:
    """Phase-2 (stateful) refinement search-space.

    Exact Phase-2 spec:
      - Load Stage-B best params JSON as `base_cfg`.
      - Freeze *everything* except the 5 keys below.
      - Explore only these 5 keys.
    """

    # Wider-but-safe defaults:
    # - seq_len: expand both ends because we observed best trials at the min (96) and also at the max (192)
    # - dropout/head_dropout: extend the upper bounds because top trials were near the previous highs
    # - grad_clip: add a slightly smaller clip option because top trials hit the minimum (0.5)
    seq_len_choices: Tuple[int, ...] = (64, 96, 128, 160, 192, 224, 256, 288)
    lr_log_low: float = 0.0003
    lr_log_high: float = 0.0025
    dropout_low: float = 0.0
    dropout_high: float = 0.25
    head_dropout_low: float = 0.0
    head_dropout_high: float = 0.40
    grad_clip_choices: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Phase2SearchSpec:
    """Controls how the stateful Optuna loop samples parameters."""

    mode: str = "refinement"  # {"refinement","full"}


@dataclass
class Phase2Result:
    objective: float
    per_symbol_metrics: Dict[str, Dict[str, float]]
    portfolio_metrics: Dict[str, float]
    preds_by_symbol: Dict[str, pd.Series]
    equity_by_symbol: Dict[str, pd.DataFrame]
    portfolio_equity: pd.DataFrame
    daily_traces: Optional[List["DailyTracePayload"]] = None  # Per-day trace instrumentation
    events: Optional[List[Dict[str, Any]]] = None  # Structured events for telemetry
    event_summary: Optional[str] = None  # Human-readable event summary
    z_explainer_logs: Optional[List[Dict[str, Any]]] = None  # Daily z-explainer logs with fidelity metrics


@dataclass(frozen=True)
class Phase2ObjectiveSpec:
    """Phase-2 objective parameters.

    Hedge-fund-grade objective with deployment awareness:
      score = Sharpe_OOS − 0.15 * MaxDrawdown_OOS − vol_underutilization_penalty
    
    Unused risk budget is a cost, not a safety feature.
    """

    max_drawdown_penalty: float = 0.15
    turnover_penalty: float = 0.0
    vol_underutilization_penalty: float = 2.0


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_stage_b_best_trial_params(best_trial_json: Path) -> Dict[str, Any]:
    """Load the frozen Stage-B best-trial params dict.

    Expected file is the exported bundle we wrote under artifacts/optuna_studies.
    """

    data = json.loads(Path(best_trial_json).read_text())

    # Preferred: resolve the true params from the Optuna DB referenced by the JSON.
    # The exported JSON `params` may contain placeholder-encoded values (e.g., 0.0
    # for categorical/text params) depending on how it was produced.
    db_path = data.get("db_path")
    best_trial = data.get("best_trial") or {}
    trial_number = best_trial.get("trial_number")
    if db_path and trial_number is not None:
        try:
            import optuna

            storage = f"sqlite:///{db_path}"
            summaries = optuna.study.get_all_study_summaries(storage=storage)
            for s in summaries:
                try:
                    study = optuna.load_study(study_name=s.study_name, storage=storage)
                    for t in study.trials:
                        if int(getattr(t, "number", -1)) == int(trial_number):
                            if t.params:
                                return dict(t.params)
                except Exception:
                    continue
        except Exception:
            # Fall back to JSON params below.
            pass

    # Fallback: use params embedded in JSON.
    params = data.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"best trial file missing params (and DB lookup failed): {best_trial_json}")
    return dict(params)


@dataclass
class Phase2PreparedData:
    label_id: str
    pipelines_by: Dict[str, StageBPipeline]
    labels_by: Dict[str, pd.DataFrame]
    trackc_aligned_by: Dict[str, pd.DataFrame]
    train_pos_by: Dict[str, np.ndarray]
    scaler_stats: Tuple[np.ndarray, np.ndarray]
    post_std_feature_weights_by_symbol: Dict[str, np.ndarray]
    features_std_full_by: Dict[str, np.ndarray]
    index_by: Dict[str, pd.DatetimeIndex]
    union_oos_index: pd.DatetimeIndex
    gpu_store_train: Optional[MultiSymbolGPUMasterStore]
    cpu_train_masters_by: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    cached_samples_by_seq_len: Dict[int, Tuple[List[Tuple[str, int]], np.ndarray]]


def _phase2_cfg_signature_for_prepared(cfg: Mapping[str, Any]) -> str:
    """Stable signature of params that affect prepared Track-C artifacts.

    Excludes pure training hyperparameters so Track-C can be reused across
    trials that only change training settings.
    
    NOTE: three_pillar_* params are FIXED governance inputs (not tuned), so they
    are excluded from cache key to ensure all trials share the same Track-C prep.
    """

    # NOTE: Phase-2 no longer tunes Track-A/Track-B weights or per-family params.
    # Prepared artifacts depend on:
    #   - how the panel is smoothed (FIXED to none, not tuned)
    #   - where Stage-A weights are sourced from (path/dir/filename)
    # Explicitly EXCLUDE:
    #   - three_pillar_dim_max, three_pillar_pca_variance (fixed, not tuned)
    #   - smoothing_window (irrelevant when smoothing_type='none')
    keep_keys: List[str] = [
        "smoothing_type",
        # Stage-A weights source
        "phase2_family_weights_path",
        "phase2_stage_a_weights_path",
        "stage_a_artifact_dir",
        "stage_a_weights_filename",
    ]

    out: Dict[str, Any] = {}
    for k in keep_keys:
        if k in cfg:
            out[k] = cfg.get(k)

    # Label choice affects CPU masters / training targets; it must be part of the prepared-cache signature.
    try:
        out["phase2_label_id"] = str(_phase2_label_id_from_cfg(cfg))
    except Exception:
        out["phase2_label_id"] = "base"

    payload = json.dumps(out, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _phase2_study_name_with_label(study_name: str, *, label_id: str) -> str:
    """Enforce one-label-per-study by suffixing the Optuna study name.

    If the caller already provided a suffix, validate it matches.
    """

    base = str(study_name)
    lid = str(label_id).lower().strip()
    if not lid:
        lid = "base"

    marker = "__label_"
    if marker in base:
        # Only treat the final marker occurrence as the label suffix.
        head, tail = base.rsplit(marker, 1)
        existing = tail.strip()
        if existing != lid:
            raise ValueError(f"study_name label mismatch: study_name={study_name} implies label={existing} but cfg label={lid}")
        return base

    return f"{base}{marker}{lid}"


def _apply_feature_smoothing(panel: pd.DataFrame, *, smoothing_type: str, smoothing_window: int) -> pd.DataFrame:
    st = str(smoothing_type or "none").lower().strip()
    w = int(smoothing_window) if smoothing_window is not None else 0
    if st == "none" or w <= 1:
        return panel

    df = panel.copy()
    num_cols = df.select_dtypes(include=["number", "bool"]).columns
    if len(num_cols) == 0:
        return df

    if st == "ema":
        df[num_cols] = df[num_cols].ewm(span=max(2, w), adjust=False, min_periods=1).mean()
        return df
    if st == "sma":
        df[num_cols] = df[num_cols].rolling(window=max(2, w), min_periods=1).mean()
        return df
    if st == "gaussian":
        # Requires SciPy in many pandas builds; fall back to SMA if unavailable.
        try:
            df[num_cols] = df[num_cols].rolling(window=max(2, w), win_type="gaussian", min_periods=1).mean(std=max(1.0, w / 6.0))
            return df
        except Exception:
            df[num_cols] = df[num_cols].rolling(window=max(2, w), min_periods=1).mean()
            return df

    return df


# ---------------------------------------------------------------------------
# Core: build Track-C (frozen Stage-B params)
# ---------------------------------------------------------------------------

def _to_datetime(x: str) -> pd.Timestamp:
    return pd.to_datetime(x, utc=False)


def _select_index_positions(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    mask = (index >= start) & (index <= end)
    # Depending on pandas version, comparisons may yield ndarray already.
    mask_arr = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    return np.where(mask_arr)[0].astype(int)


def _standardize_features_like_build_sequence_data(
    features: pd.DataFrame,
    scaler_stats: Tuple[np.ndarray, np.ndarray],
    post_standardization_feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    feat_mean, feat_std = scaler_stats
    feat_df = features.select_dtypes(include=["number", "bool"]).astype(float)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = feat_df.to_numpy(dtype=np.float32)
    if values.shape[1] != int(np.asarray(feat_mean).shape[0]):
        raise ValueError(
            f"Feature dim mismatch in standardization: values.shape={values.shape} mean.shape={np.asarray(feat_mean).shape}"
        )
    feat_std = np.where(np.asarray(feat_std) < 1e-8, 1.0, np.asarray(feat_std))
    values = (values - feat_mean) / feat_std

    if post_standardization_feature_weights is not None:
        w = np.asarray(post_standardization_feature_weights, dtype=np.float32)
        if w.ndim == 1:
            if w.shape[0] != values.shape[1]:
                raise ValueError(
                    f"post_standardization_feature_weights length {w.shape[0]} != n_features {values.shape[1]}"
                )
            values = values * w.reshape(1, -1)
        elif w.ndim == 2:
            if w.shape != values.shape:
                raise ValueError(
                    f"post_standardization_feature_weights shape {w.shape} != values shape {values.shape}"
                )
            values = values * w
        else:
            raise ValueError(f"post_standardization_feature_weights must be 1D or 2D; got ndim={w.ndim}")

    values = np.clip(values, -5.0, 5.0)
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    return values


def _read_stage_a_weights_payload(path: Path) -> Dict[str, float]:
    payload = _read_stage_a_weights_artifact(path)
    if not payload:
        return {}
    return _extract_stage_a_weights_from_payload(payload)


_STAGE_A_ARTIFACT_CACHE: Dict[str, Dict[str, Any]] = {}


def _read_stage_a_weights_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    key = str(path.resolve())
    cached = _STAGE_A_ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        payload = json.loads(path.read_text())
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    _STAGE_A_ARTIFACT_CACHE[key] = payload
    return payload


def _extract_stage_a_weights_from_payload(payload: Mapping[str, Any]) -> Dict[str, float]:
    weights = payload.get("family_normalized_scores") or payload.get("family_weights") or {}
    out: Dict[str, float] = {}
    if isinstance(weights, dict):
        for k, v in weights.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
    return out


def _phase2_static_stage_a_weights(*, payload: Mapping[str, Any], asof: pd.Timestamp) -> Dict[str, float]:
    """Return a single static Stage-A selector weights dict.

    Phase 2 does NOT use time-evolving weights. If the artifact contains a schedule,
    we select the most recent schedule entry with asof <= the provided timestamp.
    """

    w = _extract_stage_a_weights_from_payload(payload)
    if w:
        return w

    schedule = _extract_stage_a_schedule(payload)
    if schedule:
        return _stage_a_weights_for_timestamp(payload, pd.to_datetime(asof))
    return {}


def _extract_stage_a_schedule(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sched = payload.get("schedule")
    if not isinstance(sched, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in sched:
        if not isinstance(entry, dict):
            continue
        asof = entry.get("asof")
        weights = entry.get("weights") or entry.get("family_normalized_scores") or entry.get("family_weights")
        if asof is None or not isinstance(weights, dict):
            continue
        out.append({"asof": str(asof), "weights": dict(weights)})
    out.sort(key=lambda d: str(d.get("asof")))
    return out


def _resolve_phase2_stage_a_payload(cfg: Mapping[str, Any], *, symbol: str, horizon: int) -> Dict[str, Any]:
    """Resolve full Stage-A artifact payload (supports optional `schedule`)."""

    explicit = cfg.get("phase2_family_weights_path") or cfg.get("phase2_stage_a_weights_path")
    if explicit:
        return _read_stage_a_weights_artifact(Path(str(explicit)))

    base = Path(str(cfg.get("stage_a_artifact_dir", "artifacts/stage_a")))
    filename = str(cfg.get("stage_a_weights_filename", "family_weights_best.json"))

    sym = str(symbol).upper()
    candidates = [
        base / f"{sym}_h{int(horizon)}" / filename,
        base / sym / filename,
    ]
    for p in candidates:
        payload = _read_stage_a_weights_artifact(p)
        if payload:
            return payload
    return {}


def _stage_a_weights_for_timestamp(payload: Mapping[str, Any], ts: pd.Timestamp) -> Dict[str, float]:
    schedule = _extract_stage_a_schedule(payload)
    if not schedule:
        return _extract_stage_a_weights_from_payload(payload)

    t = pd.to_datetime(ts)
    chosen: Optional[Dict[str, Any]] = None
    for entry in schedule:
        try:
            asof = pd.to_datetime(entry.get("asof"))
        except Exception:
            continue
        if asof <= t:
            chosen = entry
        else:
            break
    if chosen is None:
        # If ts is before the first asof, fall back to the first schedule entry.
        chosen = schedule[0]
    weights = chosen.get("weights") or {}
    out: Dict[str, float] = {}
    if isinstance(weights, dict):
        for k, v in weights.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
    return out


def _build_time_varying_post_std_weight_matrix(
    *,
    index: pd.DatetimeIndex,
    track_c: pd.DataFrame,
    track_a: pd.DataFrame,
    panel: pd.DataFrame,
    block_summaries: Mapping[str, pd.DataFrame],
    optimizer: "StageBOptunaOptimizer",
    stage_a_payload: Mapping[str, Any],
    default_family_weights: Mapping[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """Build post-standardization feature weights.

    NOTE: Phase 2 uses STATIC Stage-A selector weights only.
    If a Stage-A artifact contains a schedule, we log and ignore it.
    """

    schedule = _extract_stage_a_schedule(stage_a_payload)
    if schedule:
        logger.info(
            "Phase2: detected Stage-A schedule (%d entries) but Phase2 uses STATIC selector weights; ignoring schedule",
            len(schedule),
        )

    w_vec, cols = _build_post_std_weights_vector(
        track_c=track_c,
        track_a=track_a,
        panel=panel,
        block_summaries=block_summaries,
        optimizer=optimizer,
        family_weights=default_family_weights,
    )
    return np.asarray(w_vec, dtype=np.float32), list(cols)


def _resolve_phase2_family_weights(cfg: Mapping[str, Any], *, symbol: str, horizon: int) -> Dict[str, float]:
    # Explicit override takes precedence.
    explicit = cfg.get("phase2_family_weights_path") or cfg.get("phase2_stage_a_weights_path")
    if explicit:
        return _read_stage_a_weights_payload(Path(str(explicit)))

    # Default Stage-A artifact locations.
    base = Path(str(cfg.get("stage_a_artifact_dir", "artifacts/stage_a")))
    filename = str(cfg.get("stage_a_weights_filename", "family_weights_best.json"))

    sym = str(symbol).upper()
    candidates = [
        base / f"{sym}_h{int(horizon)}" / filename,
        base / sym / filename,
    ]
    for p in candidates:
        weights = _read_stage_a_weights_payload(p)
        if weights:
            return weights
    return {}


def _build_post_std_weights_vector(
    *,
    track_c: pd.DataFrame,
    track_a: pd.DataFrame,
    panel: pd.DataFrame,
    block_summaries: Mapping[str, pd.DataFrame],
    optimizer: "StageBOptunaOptimizer",
    family_weights: Mapping[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """Return (weights_vector, feature_columns) aligned to build_sequence_data dtype filter.

    We apply weights AFTER standardization, so we need a per-column vector.
    Columns not mapped to a family keep weight 1.0.
    """

    from src.stage_b.optuna_optimizer import HF_BLOCK_FAMILIES, TRACK_B_SUMMARY_BLOCKS

    # Some Stage-A weight families refer to Track-B summary blocks by a different name.
    # Map summary block name -> Stage-A weight key.
    summary_weight_alias = {
        "quantile": "quantile_forecast",
        "online": "online_learning",
        "arima": "arima_forecast",
    }

    # Map columns -> family name.
    col_to_family: Dict[str, str] = {}

    # Track-A encoded columns: {family}_enc_{i}
    for c in list(track_a.columns):
        s = str(c)
        if "_enc_" in s:
            col_to_family[s] = s.split("_enc_")[0]

    # Track-B HF raw columns
    for fam in HF_BLOCK_FAMILIES:
        cols = getattr(optimizer, "family_columns", {}).get(str(fam), [])
        for c in cols or []:
            col_to_family[str(c)] = str(fam)

    # Track-B summary blocks (quantile/calibration/online/arima)
    for summary_name in TRACK_B_SUMMARY_BLOCKS:
        df = block_summaries.get(summary_name)
        if df is None or df.empty:
            continue
        fam_key = summary_weight_alias.get(str(summary_name), str(summary_name))
        for c in df.columns:
            col_to_family[str(c)] = fam_key

    feat_df = track_c.select_dtypes(include=["number", "bool"]).astype(float)
    cols = [str(c) for c in feat_df.columns]
    w = np.ones((len(cols),), dtype=np.float32)
    for j, c in enumerate(cols):
        fam = col_to_family.get(c)
        if fam is None:
            continue
        val = family_weights.get(fam)
        if val is None:
            continue
        w[j] = float(val)
    return w, cols


def _coerce_trackc_numeric_schema(track_c: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Force Track-C to a stable numeric schema.

    Phase-2 pooled training assumes all symbols share identical feature columns.
    Some symbols can surface dtype drift (e.g., object columns) or missing columns.
    We coerce any non-numeric columns to numeric (coerce->NaN) then fill NaNs to 0.
    """
    df = track_c.reindex(columns=list(cols))
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        df[c] = pd.to_numeric(s, errors="coerce")
    # Keep float for deterministic numpy conversion; missing values become 0.
    df = df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


class NonFiniteError(ValueError):
    pass


def _first_non_finite(arr: np.ndarray) -> Optional[Tuple[int, float]]:
    bad = ~np.isfinite(arr)
    if not np.any(bad):
        return None
    idx = int(np.argmax(bad))
    return idx, float(arr.reshape(-1)[idx])


def _assert_finite_np(
    *,
    name: str,
    symbol: str,
    values: np.ndarray,
    index: Optional[pd.DatetimeIndex] = None,
) -> None:
    bad = ~np.isfinite(values)
    if not np.any(bad):
        return

    # Try to map the first bad row to a timestamp.
    bad_pos = np.argwhere(bad)
    row = int(bad_pos[0][0]) if bad_pos.size else -1
    col = int(bad_pos[0][1]) if bad_pos.size and bad_pos.shape[1] > 1 else -1
    ts = None
    if index is not None and 0 <= row < len(index):
        ts = str(index[row])
    raise NonFiniteError(f"Non-finite detected: {name} symbol={symbol} row={row} col={col} ts={ts}")


def _assert_finite_series(*, name: str, symbol: str, s: pd.Series) -> None:
    if s is None or s.empty:
        return
    bad = ~np.isfinite(s.to_numpy(dtype=float))
    if not np.any(bad):
        return
    first = int(np.argmax(bad))
    ts = str(s.index[first])
    val = s.iloc[first]
    raise NonFiniteError(f"Non-finite detected: {name} symbol={symbol} ts={ts} value={val}")


def _phase2_apply_refinement_overrides(
    base_params: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a trial config dict obeying Phase-2 freeze rules."""

    cfg = dict(base_params)

    allowed = {
        "mamba_seq_len",
        "mamba_learning_rate",
        "mamba_dropout",
        "mamba_head_dropout",
        "mamba_grad_clip",
    }
    extra = [k for k in overrides.keys() if k not in allowed]
    if extra:
        raise ValueError(f"Phase-2 overrides contain non-allowed keys: {extra}")

    # Apply only the allowed refinement overrides (everything else remains exactly from Stage-B JSON).
    for k, v in overrides.items():
        cfg[k] = v

    return cfg


def _phase2_label_id_from_cfg(cfg: Mapping[str, Any]) -> str:
    v = cfg.get("phase2_label_id", "base")
    s = str(v).lower().strip()
    if s in {"base", "forward_return", "log_return", "logret"}:
        return "base"
    if s in {"voladj", "vol_adj", "vol_adjusted", "vol_adjusted_return"}:
        return "voladj"
    raise ValueError(f"unknown phase2_label_id: {v}")


def _phase2_label_column(label_id: str) -> str:
    lid = str(label_id).lower().strip()
    if lid == "base":
        return "forward_return"
    if lid == "voladj":
        return "forward_return_voladj"
    raise ValueError(f"unknown label_id: {label_id}")


# ---------------------------------------------------------------------------
# Phase-2 evaluation: fixed train block + stateful OOS loop
# ---------------------------------------------------------------------------

def _build_stage_b_pipeline(symbol: str, horizon: int, *, start: str, end: str) -> StageBPipeline:
    """Minimal StageBPipeline instance to access panel/label builders."""

    # Phase2 v2 policy: Dagster is the source of truth for Track-C panels.
    # StageBPipeline defaults to prep.enabled=True and will try to run
    # tools/prep_families.py unless it sees BOTH:
    #   (1) cache/features/<SYMBOL>_h<H>_merged.parquet + .meta.json AND
    #   (2) artifacts/prep_families/*_completeness.json with non-empty windows.
    # Dagster assets write the merged parquet but not the completeness manifest.
    # Therefore, Phase2 disables prep orchestration by default and expects
    # the merged parquet to already exist.
    import os
    from src.stage_b.pipeline import PrepFamiliesSettings

    allow_prep = os.environ.get("PHASE2_ALLOW_PREP_FAMILIES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    cfg = StageBConfig(
        symbol=symbol,
        horizons=[int(horizon)],
        horizon_clusters={"default": [int(horizon)]},
        run_lstm_for_clusters=["default"],
        min_samples_for_lstm=0,
        max_lstm_folds=1,
        lstm_trials_per_cluster=1,
        top_k_families_for_seq=0,
        performance_bar_for_lstm=0.0,
        start=str(start),
        end=str(end),
        prep=None if not allow_prep else PrepFamiliesSettings(enabled=True),
    )
    return StageBPipeline(cfg)


def _make_strategy_cfg_from_base(base_params: Mapping[str, Any]) -> Dict[str, Any]:
    # Keep Stage-B risk/score settings frozen; only inject regime thresholds if present.
    strategy_cfg: Dict[str, Any] = {
        "regime_threshold_enabled": True,
    }

    # Many Stage-B runs store these as Step-7 params.
    threshold = base_params.get("threshold", None)
    bull_mult = base_params.get("bull_mult", base_params.get("bull_long_mult", None))
    bear_mult = base_params.get("bear_mult", base_params.get("bear_long_mult", None))
    crisis_mult = base_params.get("crisis_mult", base_params.get("crisis_long_mult", None))

    bull_short_mult = base_params.get("bull_short_mult", None)
    bear_short_mult = base_params.get("bear_short_mult", None)
    crisis_short_mult = base_params.get("crisis_short_mult", None)
    conf_threshold = base_params.get("conf_threshold", None)
    vol_scaler = base_params.get("vol_scaler", None)

    if threshold is not None:
        params: Dict[str, Any] = {
            "threshold": float(threshold),
            # Keep legacy keys for older code paths / diagnostics.
            "bull_mult": float(bull_mult if bull_mult is not None else 0.8),
            "bear_mult": float(bear_mult if bear_mult is not None else 1.5),
            "crisis_mult": float(crisis_mult if crisis_mult is not None else 3.0),
        }
        # Newer Stage-B tuning knobs (no-ops unless the backtest uses them).
        if bull_short_mult is not None:
            params["bull_short_mult"] = float(bull_short_mult)
        if bear_short_mult is not None:
            params["bear_short_mult"] = float(bear_short_mult)
        if crisis_short_mult is not None:
            params["crisis_short_mult"] = float(crisis_short_mult)
        if conf_threshold is not None:
            params["conf_threshold"] = float(conf_threshold)
        if vol_scaler is not None:
            params["vol_scaler"] = float(vol_scaler)

        strategy_cfg["regime_threshold_params"] = params

    return strategy_cfg


def _stateful_predict_roll_window(
    *,
    model,
    device,
    features_std: np.ndarray,
    index: pd.DatetimeIndex,
    oos_pos: np.ndarray,
    burnin_end_pos: int,
    seq_len: int,
) -> pd.Series:
    """Continuous OOS inference with burn-in and no resets.

    Implementation detail:
    - This code treats the model's "state" as the rolling window of the last
      `seq_len` standardized feature rows.
    """

    import torch

    if burnin_end_pos < 0:
        raise ValueError("burnin_end_pos must be >= 0")

    # Burn-in window is the seq_len rows immediately before OOS start.
    burnin_slice = np.arange(max(0, burnin_end_pos - seq_len + 1), burnin_end_pos + 1)
    if len(burnin_slice) < seq_len:
        raise ValueError(f"insufficient burn-in rows: have {len(burnin_slice)}, need {seq_len}")

    buffer = features_std[burnin_slice].copy()  # (seq_len, n_features)

    preds: List[float] = []
    ts: List[pd.Timestamp] = []

    model.eval()
    with torch.no_grad():
        for pos in oos_pos:
            buffer[:-1] = buffer[1:]
            buffer[-1] = features_std[pos]

            # Phase-2 failure policy: explicitly fail on non-finite inputs.
            if not np.isfinite(buffer).all():
                ts = str(index[pos])
                raise NonFiniteError(f"Non-finite detected: state_buffer symbol=? ts={ts}")

            x = torch.from_numpy(buffer[None, :, :]).to(device)
            y_t = model(x)
            if not torch.isfinite(y_t).all():
                ts = str(index[pos])
                raise NonFiniteError(f"Non-finite detected: model_output symbol=? ts={ts}")
            y = y_t.detach().float().cpu().numpy().reshape(-1)[0]
            preds.append(float(y))
            ts.append(index[pos])

    return pd.Series(preds, index=pd.DatetimeIndex(ts))


def _stateful_predict_batched_across_symbols(
    *,
    model,
    device,
    features_std_by_symbol: Mapping[str, np.ndarray],
    index_by_symbol: Mapping[str, pd.DatetimeIndex],
    union_oos_index: pd.DatetimeIndex,
    burnin_end_ts: pd.Timestamp,
    seq_len: int,
) -> Dict[str, pd.Series]:
    """Stateful inference batched across symbols.

    Model state is the rolling buffer per symbol, stored as a tensor of shape:
      buffer: (B, seq_len, F)
    where B = number of symbols.

    For missing rows on a given day, feed zeros (neutral) so we don't trade.
    """

    import torch

    symbols = [str(s).upper() for s in features_std_by_symbol.keys()]
    if not symbols:
        raise ValueError("no symbols for batched inference")

    # Validate dimensions and feature dims are consistent.
    feat_dims = {features_std_by_symbol[s].shape[1] for s in symbols}
    if len(feat_dims) != 1:
        raise ValueError(f"feature_dim mismatch across symbols: {feat_dims}")
    feature_dim = int(next(iter(feat_dims)))

    # Build initial burn-in buffers for each symbol using the last seq_len rows <= burnin_end_ts.
    init_buffers: List[np.ndarray] = []
    for sym in symbols:
        idx = index_by_symbol[sym]
        feats = features_std_by_symbol[sym]
        # Positions up to burnin_end_ts.
        pos = np.where((idx <= burnin_end_ts).to_numpy() if hasattr(idx <= burnin_end_ts, "to_numpy") else np.asarray(idx <= burnin_end_ts))[0]
        if len(pos) < seq_len:
            raise ValueError(f"insufficient burn-in rows for {sym}: have {len(pos)}, need {seq_len}")
        burnin_slice = pos[-seq_len:]
        buf = feats[burnin_slice]
        if buf.shape != (seq_len, feature_dim):
            raise ValueError(f"burn-in buffer shape mismatch for {sym}: {buf.shape}")
        init_buffers.append(buf)

    buffer = torch.from_numpy(np.stack(init_buffers, axis=0)).to(device)  # (B, L, F)

    # Precompute per-symbol date->row maps for fast gather.
    maps: Dict[str, Dict[pd.Timestamp, int]] = {}
    for sym in symbols:
        idx = index_by_symbol[sym]
        maps[sym] = {pd.Timestamp(t): int(i) for i, t in enumerate(idx)}

    preds_out = {sym: np.full(len(union_oos_index), 0.0, dtype=np.float32) for sym in symbols}
    model.eval()
    with torch.no_grad():
        for t_i, ts in enumerate(union_oos_index):
            # Build x_t for all symbols.
            x_np = np.zeros((len(symbols), feature_dim), dtype=np.float32)
            has_row = np.zeros(len(symbols), dtype=bool)
            for s_i, sym in enumerate(symbols):
                row = maps[sym].get(pd.Timestamp(ts))
                if row is None:
                    continue
                x_np[s_i] = features_std_by_symbol[sym][row]
                has_row[s_i] = True

            if has_row.any() and (not np.isfinite(x_np[has_row]).all()):
                raise NonFiniteError(f"Non-finite detected: x_t_batched ts={ts}")

            x_t = torch.from_numpy(x_np).to(device)

            # Update rolling buffers only for symbols that have data for this timestamp.
            # For missing rows, keep the buffer unchanged and force prediction to 0
            # so we don't trade on synthetic/empty inputs.
            if has_row.any():
                mask = torch.from_numpy(has_row).to(device)
                buf_m = buffer[mask]
                buf_m[:, :-1, :] = buf_m[:, 1:, :]
                buf_m[:, -1, :] = x_t[mask]
                buffer[mask] = buf_m

            y_t = model(buffer)
            if not torch.isfinite(y_t).all():
                raise NonFiniteError(f"Non-finite detected: model_output_batched ts={ts}")
            y = y_t.detach().float().cpu().numpy().reshape(-1)
            for s_i, sym in enumerate(symbols):
                if not has_row[s_i]:
                    preds_out[sym][t_i] = 0.0
                else:
                    preds_out[sym][t_i] = float(y[s_i])

    return {sym: pd.Series(preds_out[sym], index=union_oos_index) for sym in symbols}


def _stateful_predict_roll_window_mu_sigma(
    *,
    model,
    device,
    features_std: np.ndarray,
    index: pd.DatetimeIndex,
    oos_pos: np.ndarray,
    burnin_end_pos: int,
    seq_len: int,
    sigma_floor: float = 1e-6,
    sigma_cap: float = 10.0,
) -> Tuple[pd.Series, pd.Series]:
    """Continuous OOS distributional inference with burn-in and no resets.

    Returns both mu (mean prediction) and sigma (uncertainty estimate).

    Implementation detail:
    - This code treats the model's "state" as the rolling window of the last
      `seq_len` standardized feature rows.
    - For models with Gaussian head, returns true sigma from the model.
    - For scalar head models, returns a proxy sigma.
    """

    import torch

    if burnin_end_pos < 0:
        raise ValueError("burnin_end_pos must be >= 0")

    # Burn-in window is the seq_len rows immediately before OOS start.
    burnin_slice = np.arange(max(0, burnin_end_pos - seq_len + 1), burnin_end_pos + 1)
    if len(burnin_slice) < seq_len:
        raise ValueError(f"insufficient burn-in rows: have {len(burnin_slice)}, need {seq_len}")

    buffer = features_std[burnin_slice].copy()  # (seq_len, n_features)

    mu_list: List[float] = []
    sigma_list: List[float] = []
    ts_list: List[pd.Timestamp] = []

    # Check if model has predict_distribution method
    has_predict_dist = hasattr(model, 'predict_distribution') and callable(getattr(model, 'predict_distribution'))
    has_uncertainty_head = getattr(model, 'has_uncertainty_head', False)

    model.eval()
    with torch.no_grad():
        for pos in oos_pos:
            buffer[:-1] = buffer[1:]
            buffer[-1] = features_std[pos]

            # Phase-2 failure policy: explicitly fail on non-finite inputs.
            if not np.isfinite(buffer).all():
                ts_str = str(index[pos])
                raise NonFiniteError(f"Non-finite detected: state_buffer symbol=? ts={ts_str}")

            x = torch.from_numpy(buffer[None, :, :]).to(device)

            if has_predict_dist:
                # Use predict_distribution for proper mu/sigma extraction
                mu_t, sigma_t = model.predict_distribution(x, sigma_floor=sigma_floor, sigma_cap=sigma_cap)
                if not (torch.isfinite(mu_t).all() and torch.isfinite(sigma_t).all()):
                    ts_str = str(index[pos])
                    raise NonFiniteError(f"Non-finite detected: predict_distribution output ts={ts_str}")
                mu = float(mu_t.detach().float().cpu().numpy().reshape(-1)[0])
                sigma = float(sigma_t.detach().float().cpu().numpy().reshape(-1)[0])
            elif has_uncertainty_head:
                # Model returns (mu, log_var) - extract manually
                y_t = model(x)
                if not torch.isfinite(y_t).all():
                    ts_str = str(index[pos])
                    raise NonFiniteError(f"Non-finite detected: model_output ts={ts_str}")
                if y_t.ndim == 2 and y_t.shape[1] == 2:
                    mu = float(y_t[0, 0].detach().float().cpu().numpy())
                    log_var = float(y_t[0, 1].detach().float().cpu().numpy())
                    var = float(np.clip(np.exp(log_var), sigma_floor**2, sigma_cap**2))
                    sigma = float(np.sqrt(var + sigma_floor))
                else:
                    mu = float(y_t.detach().float().cpu().numpy().reshape(-1)[0])
                    sigma = 0.02  # Default proxy
            else:
                # Scalar head: return mu and proxy sigma
                y_t = model(x)
                if not torch.isfinite(y_t).all():
                    ts_str = str(index[pos])
                    raise NonFiniteError(f"Non-finite detected: model_output ts={ts_str}")
                mu = float(y_t.detach().float().cpu().numpy().reshape(-1)[0])
                sigma = 0.02  # Default proxy ~2% daily vol

            mu_list.append(mu)
            sigma_list.append(sigma)
            ts_list.append(index[pos])

    ts_index = pd.DatetimeIndex(ts_list)
    return pd.Series(mu_list, index=ts_index), pd.Series(sigma_list, index=ts_index)


def _stateful_predict_multi_horizon_primary(
    *,
    model: "MultiHorizonMambaRegressor",
    device: "torch.device",
    primary_horizon: int,
    features_std_by_symbol: Mapping[str, np.ndarray],
    index_by_symbol: Mapping[str, pd.DatetimeIndex],
    union_oos_index: pd.DatetimeIndex,
    burnin_end_ts: pd.Timestamp,
    seq_len: int,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Stateful inference for multi-horizon model using only the primary horizon.
    
    The multi-horizon model is trained on multiple horizons simultaneously, but
    at inference time we only use the primary horizon (e.g., 63 days) for
    downstream portfolio construction and policy controller.
    
    This keeps the interface identical to single-horizon inference, ensuring
    zero changes to portfolio logic.
    
    Args:
        model: MultiHorizonMambaRegressor trained on multiple horizons
        device: PyTorch device
        primary_horizon: The horizon (in days) to use for predictions (e.g., 63)
        features_std_by_symbol: Standardized features per symbol
        index_by_symbol: DatetimeIndex per symbol
        union_oos_index: Union OOS index across symbols
        burnin_end_ts: End of burn-in period
        seq_len: Sequence length
        
    Returns:
        Tuple of (preds_by_symbol, sigma_by_symbol) - same format as single-horizon
    """
    import torch
    
    symbols = [str(s).upper() for s in features_std_by_symbol.keys()]
    if not symbols:
        raise ValueError("no symbols for multi-horizon inference")
    
    # Validate dimensions
    feat_dims = {features_std_by_symbol[s].shape[1] for s in symbols}
    if len(feat_dims) != 1:
        raise ValueError(f"feature_dim mismatch across symbols: {feat_dims}")
    feature_dim = int(next(iter(feat_dims)))
    
    # Build initial burn-in buffers
    init_buffers: List[np.ndarray] = []
    for sym in symbols:
        idx = index_by_symbol[sym]
        feats = features_std_by_symbol[sym]
        pos = np.where((idx <= burnin_end_ts).to_numpy() if hasattr(idx <= burnin_end_ts, "to_numpy") else np.asarray(idx <= burnin_end_ts))[0]
        if len(pos) < seq_len:
            raise ValueError(f"insufficient burn-in rows for {sym}: have {len(pos)}, need {seq_len}")
        burnin_slice = pos[-seq_len:]
        buf = feats[burnin_slice]
        init_buffers.append(buf)
    
    buffer = torch.from_numpy(np.stack(init_buffers, axis=0)).to(device)  # (B, L, F)
    
    # Precompute per-symbol date->row maps
    maps: Dict[str, Dict[pd.Timestamp, int]] = {}
    for sym in symbols:
        idx = index_by_symbol[sym]
        maps[sym] = {pd.Timestamp(t): int(i) for i, t in enumerate(idx)}
    
    # Output arrays
    mu_out = {sym: np.full(len(union_oos_index), 0.0, dtype=np.float32) for sym in symbols}
    sigma_out = {sym: np.full(len(union_oos_index), 0.02, dtype=np.float32) for sym in symbols}
    
    model.eval()
    with torch.no_grad():
        for t_i, ts in enumerate(union_oos_index):
            # Build x_t for all symbols
            x_np = np.zeros((len(symbols), feature_dim), dtype=np.float32)
            has_row = np.zeros(len(symbols), dtype=bool)
            for s_i, sym in enumerate(symbols):
                row = maps[sym].get(pd.Timestamp(ts))
                if row is None:
                    continue
                x_np[s_i] = features_std_by_symbol[sym][row]
                has_row[s_i] = True
            
            x_t = torch.from_numpy(x_np).to(device)
            
            # Update rolling buffers for symbols with data
            if has_row.any():
                mask = torch.from_numpy(has_row).to(device)
                buf_m = buffer[mask]
                buf_m[:, :-1, :] = buf_m[:, 1:, :]
                buf_m[:, -1, :] = x_t[mask]
                buffer[mask] = buf_m
            
            # Query model for primary horizon only
            # model.query(x, horizon) returns (mu, sigma) for that horizon
            mu_t, sigma_t = model.query(buffer, primary_horizon)
            
            mu_arr = mu_t.detach().float().cpu().numpy().reshape(-1)
            sigma_arr = sigma_t.detach().float().cpu().numpy().reshape(-1)
            
            # Store predictions
            for s_i, sym in enumerate(symbols):
                if has_row[s_i]:
                    mu_out[sym][t_i] = float(mu_arr[s_i])
                    sigma_out[sym][t_i] = float(np.clip(sigma_arr[s_i], 1e-6, 10.0))
    
    # Convert to Series
    preds_by_symbol = {
        sym: pd.Series(mu_out[sym], index=union_oos_index)
        for sym in symbols
    }
    sigma_by_symbol = {
        sym: pd.Series(sigma_out[sym], index=union_oos_index)
        for sym in symbols
    }
    
    return preds_by_symbol, sigma_by_symbol


def _stateful_predict_batched_mu_sigma(
    *,
    model,
    device,
    features_std_by_symbol: Mapping[str, np.ndarray],
    index_by_symbol: Mapping[str, pd.DatetimeIndex],
    union_oos_index: pd.DatetimeIndex,
    burnin_end_ts: pd.Timestamp,
    seq_len: int,
    sigma_floor: float = 1e-6,
    sigma_cap: float = 10.0,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Stateful distributional inference batched across symbols.

    Returns both mu (mean predictions) and sigma (uncertainty estimates) per symbol.

    Model state is the rolling buffer per symbol, stored as a tensor of shape:
      buffer: (B, seq_len, F)
    where B = number of symbols.

    For missing rows on a given day, feed zeros (neutral) so we don't trade.
    """

    import torch

    symbols = [str(s).upper() for s in features_std_by_symbol.keys()]
    if not symbols:
        raise ValueError("no symbols for batched inference")

    # Validate dimensions and feature dims are consistent.
    feat_dims = {features_std_by_symbol[s].shape[1] for s in symbols}
    if len(feat_dims) != 1:
        raise ValueError(f"feature_dim mismatch across symbols: {feat_dims}")
    feature_dim = int(next(iter(feat_dims)))

    # Build initial burn-in buffers for each symbol using the last seq_len rows <= burnin_end_ts.
    init_buffers: List[np.ndarray] = []
    for sym in symbols:
        idx = index_by_symbol[sym]
        feats = features_std_by_symbol[sym]
        # Positions up to burnin_end_ts.
        pos = np.where((idx <= burnin_end_ts).to_numpy() if hasattr(idx <= burnin_end_ts, "to_numpy") else np.asarray(idx <= burnin_end_ts))[0]
        if len(pos) < seq_len:
            raise ValueError(f"insufficient burn-in rows for {sym}: have {len(pos)}, need {seq_len}")
        burnin_slice = pos[-seq_len:]
        buf = feats[burnin_slice]
        if buf.shape != (seq_len, feature_dim):
            raise ValueError(f"burn-in buffer shape mismatch for {sym}: {buf.shape}")
        init_buffers.append(buf)

    buffer = torch.from_numpy(np.stack(init_buffers, axis=0)).to(device)  # (B, L, F)

    # Precompute per-symbol date->row maps for fast gather.
    maps: Dict[str, Dict[pd.Timestamp, int]] = {}
    for sym in symbols:
        idx = index_by_symbol[sym]
        maps[sym] = {pd.Timestamp(t): int(i) for i, t in enumerate(idx)}

    # Output arrays
    mu_out = {sym: np.full(len(union_oos_index), 0.0, dtype=np.float32) for sym in symbols}
    sigma_out = {sym: np.full(len(union_oos_index), 0.02, dtype=np.float32) for sym in symbols}

    # Check if model has predict_distribution method
    has_predict_dist = hasattr(model, 'predict_distribution') and callable(getattr(model, 'predict_distribution'))
    has_uncertainty_head = getattr(model, 'has_uncertainty_head', False)

    model.eval()
    with torch.no_grad():
        for t_i, ts in enumerate(union_oos_index):
            # Build x_t for all symbols.
            x_np = np.zeros((len(symbols), feature_dim), dtype=np.float32)
            has_row = np.zeros(len(symbols), dtype=bool)
            for s_i, sym in enumerate(symbols):
                row = maps[sym].get(pd.Timestamp(ts))
                if row is None:
                    continue
                x_np[s_i] = features_std_by_symbol[sym][row]
                has_row[s_i] = True

            if has_row.any() and (not np.isfinite(x_np[has_row]).all()):
                raise NonFiniteError(f"Non-finite detected: x_t_batched ts={ts}")

            x_t = torch.from_numpy(x_np).to(device)

            # Update rolling buffers only for symbols that have data for this timestamp.
            if has_row.any():
                mask = torch.from_numpy(has_row).to(device)
                buf_m = buffer[mask]
                buf_m[:, :-1, :] = buf_m[:, 1:, :]
                buf_m[:, -1, :] = x_t[mask]
                buffer[mask] = buf_m

            if has_predict_dist:
                # Use predict_distribution for proper mu/sigma extraction
                mu_t, sigma_t = model.predict_distribution(buffer, sigma_floor=sigma_floor, sigma_cap=sigma_cap)
                if not (torch.isfinite(mu_t).all() and torch.isfinite(sigma_t).all()):
                    raise NonFiniteError(f"Non-finite detected: predict_distribution output ts={ts}")
                mu_arr = mu_t.detach().float().cpu().numpy().reshape(-1)
                sigma_arr = sigma_t.detach().float().cpu().numpy().reshape(-1)
            elif has_uncertainty_head:
                # Model returns (mu, log_var) - extract manually
                y_t = model(buffer)
                if not torch.isfinite(y_t).all():
                    raise NonFiniteError(f"Non-finite detected: model_output_batched ts={ts}")
                if y_t.ndim == 2 and y_t.shape[1] == 2:
                    mu_arr = y_t[:, 0].detach().float().cpu().numpy().reshape(-1)
                    log_var_arr = y_t[:, 1].detach().float().cpu().numpy().reshape(-1)
                    var_arr = np.clip(np.exp(log_var_arr), sigma_floor**2, sigma_cap**2)
                    sigma_arr = np.sqrt(var_arr + sigma_floor)
                else:
                    mu_arr = y_t.detach().float().cpu().numpy().reshape(-1)
                    sigma_arr = np.full_like(mu_arr, 0.02)
            else:
                # Scalar head: return mu and proxy sigma
                y_t = model(buffer)
                if not torch.isfinite(y_t).all():
                    raise NonFiniteError(f"Non-finite detected: model_output_batched ts={ts}")
                mu_arr = y_t.detach().float().cpu().numpy().reshape(-1)
                sigma_arr = np.full_like(mu_arr, 0.02)

            for s_i, sym in enumerate(symbols):
                if not has_row[s_i]:
                    mu_out[sym][t_i] = 0.0
                    sigma_out[sym][t_i] = 0.02  # Default proxy
                else:
                    mu_out[sym][t_i] = float(mu_arr[s_i])
                    sigma_out[sym][t_i] = float(sigma_arr[s_i])

    return (
        {sym: pd.Series(mu_out[sym], index=union_oos_index) for sym in symbols},
        {sym: pd.Series(sigma_out[sym], index=union_oos_index) for sym in symbols},
    )


def _evaluate_phase2_stateful_symbol_once(
    *,
    best_trial_json: Path,
    symbol: str,
    horizon: int,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    overrides: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    """Train+evaluate a single symbol once using the Phase-2 stateful OOS loop."""

    import torch

    base_params = load_stage_b_best_trial_params(best_trial_json)
    cfg = _phase2_apply_refinement_overrides(base_params, overrides)

    # StageBPipeline requires config.start/end to fetch price history for labels.
    pipeline = _build_stage_b_pipeline(
        symbol=str(symbol).upper(),
        horizon=int(horizon),
        start=str(train_start),
        end=str(oos_end),
    )

    panel = pipeline._build_panel(int(horizon))
    labels = pipeline._construct_labels(int(horizon), panel.index)

    label_col = _phase2_label_column(_phase2_label_id_from_cfg(cfg))

    valid_mask = labels[label_col].notna()
    panel = panel.loc[valid_mask]
    labels = labels.loc[valid_mask]

    index = pd.DatetimeIndex(panel.index)
    t_train_start = _to_datetime(train_start)
    t_train_end = _to_datetime(train_end)
    t_oos_start = _to_datetime(oos_start)
    t_oos_end = _to_datetime(oos_end)

    train_pos = _select_index_positions(index, t_train_start, t_train_end)
    oos_pos = _select_index_positions(index, t_oos_start, t_oos_end)
    if len(train_pos) < 100:
        raise ValueError(f"train block too small: {len(train_pos)} rows")
    if len(oos_pos) < 10:
        raise ValueError(f"oos block too small: {len(oos_pos)} rows")

    # Prevent leakage in Track-B summaries (train-only), forward-filled into OOS.
    pipeline._current_train_idx = train_pos
    block_summaries = pipeline._compute_block_summaries(panel)

    # Build Track-C using the Stage-B optimizer code-path (this respects frozen weights/dims).
    from src.stage_b.optuna_optimizer import OptunaConfig, StageBOptunaOptimizer

    opt_cfg = OptunaConfig(
        # Phase2: do not hard-cap total Track-C dimensions.
        max_total_dims=0,
        horizon=int(horizon),
        sequence_model_type="mamba",
        use_three_pillar_dims=True,
        # Allow 3-pillar to pick enough dims to hit the variance target on large families.
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
    )
    # Phase-2 does NOT tune/apply family weights inside Track construction.
    # We apply family weights after standardization (see build_sequence_data hook).
    try:
        setattr(opt_cfg, "weight_normalization", "none")
    except Exception:
        pass
    optimizer = StageBOptunaOptimizer(config=opt_cfg)

    column_families = pipeline._infer_column_families(panel.columns)

    # Fit Track-A encoders on train rows only.
    pooled_panel = panel.iloc[train_pos]
    pooled_idx = np.arange(len(pooled_panel), dtype=int)

    # Analyze families + run 3-pillar dim selection (PCA ~95% variance).
    # Phase 2 full-mode must incorporate Stage-A selector family weights.
    # We use a single static snapshot as-of Phase2 train_end to avoid leakage.
    stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(symbol).upper(), horizon=int(horizon))
    stage_a_weights_static = _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end))
    
    # Try loading 3-pillar results from disk cache
    cache_key = _three_pillar_cache_key(
        symbols=[symbol],
        horizon=horizon,
        train_start=train_start,
        train_end=train_end,
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
        stage_a_weights=stage_a_weights_static,
    )
    cached_three_pillar = _load_three_pillar_cache(cache_key)
    
    if cached_three_pillar is not None:
        # Restore optimizer state from cache
        optimizer.family_sizes = cached_three_pillar.get("family_sizes", {})
        optimizer.family_optimal_dims = cached_three_pillar.get("family_optimal_dims", {})
        logger.info(f"[3-pillar] Restored {len(optimizer.family_optimal_dims)} family dims from cache")
    else:
        # Run 3-pillar analysis (expensive)
        logger.info(f"[3-pillar] 🔧 CACHE MISS - running 3-pillar analysis")
        optimizer._analyze_families(pooled_panel, column_families, stage_a_weights=stage_a_weights_static or None)
        
        # Save to disk cache
        _save_three_pillar_cache(cache_key, {
            "family_sizes": getattr(optimizer, "family_sizes", {}),
            "family_optimal_dims": getattr(optimizer, "family_optimal_dims", {}),
        })

    # Build family params from 3-pillar dims only (NO per-family tuning, NO per-family weights).
    from src.stage_b.optuna_optimizer import STAGE_A_FAMILIES, _is_raw_passthrough_family

    family_params: Dict[str, Tuple[bool, int, str, float, Dict[str, Any]]] = {}
    for fam in STAGE_A_FAMILIES:
        fam = str(fam)
        cols = [c for c in panel.columns if column_families.get(c) == fam]
        family_size = int(len(cols))
        include = bool(family_size > 0)

        if not include:
            family_params[fam] = (False, 0, "none", 0.0, {})
            continue

        # Enforce raw passthrough policy (no compression). Use full family size.
        if _is_raw_passthrough_family(fam):
            family_params[fam] = (True, int(family_size), "passthrough", 1.0, {})
            continue

        dim_cfg = getattr(optimizer, "family_optimal_dims", {}).get(fam)
        if dim_cfg is None:
            # Fallback: minimal PCA dims.
            k_final = int(min(int(getattr(opt_cfg, "three_pillar_dim_min", 4)), int(family_size)))
            method = "pca"
        else:
            k_final = int(min(int(getattr(dim_cfg, "k_final", 0)), int(family_size)))
            method = str(getattr(dim_cfg, "method", "pca"))
        k_final = int(max(k_final, 1))
        family_params[fam] = (True, int(k_final), str(method), 1.0, {})

    optimizer._build_track_a(pooled_panel, column_families, family_params, pooled_idx)

    track_a, _ = optimizer._build_track_a(panel, column_families, family_params, train_pos)
    track_b = optimizer._build_track_b(
        panel,
        column_families,
        block_summaries,
        track_b_family_weights=None,
    )
    track_c = optimizer._build_track_c(track_a, track_b, 1.0, 1.0)
    if track_c is None or track_c.empty:
        raise ValueError("Track C is empty")

    # HYGIENE GUARD: Block governance columns from model inputs (suffix-based)
    # These columns are for data quality tracking, NOT predictive features.
    gov_suffixes = (
        "_has_data", "_activity", "_days_since_update", 
        "_source_asof_ts", "_confidence", "_conf"
    )
    bad_cols = [c for c in track_c.columns if str(c).endswith(gov_suffixes)]
    if bad_cols:
        raise ValueError(
            f"GOVERNANCE LEAKAGE DETECTED: {len(bad_cols)} columns with governance suffixes found in model inputs.\n"
            f"First 50: {bad_cols[:50]}\n"
            f"These columns must be excluded before Stage-B. Check feature_roles.py classification."
        )

    seq_len = int(cfg.get("mamba_seq_len", 128))

    # Pooled scaler stats (single symbol here, but keep same API).
    scaler_stats = optimizer._compute_pooled_scaler_stats([track_c.iloc[train_pos]], [labels[label_col].iloc[train_pos]])

    # Load Stage-A family weights (fixed governance input to Phase 2).
    # If the Stage-A artifact contains a schedule, Phase 2 uses a single static
    # snapshot as-of the Phase 2 train_end to avoid leakage.
    stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(symbol).upper(), horizon=int(horizon))
    family_weights = _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end)) or _resolve_phase2_family_weights(
        cfg, symbol=str(symbol).upper(), horizon=int(horizon)
    )
    post_w_all, _ = _build_time_varying_post_std_weight_matrix(
        index=index,
        track_c=track_c,
        track_a=track_a,
        panel=panel,
        block_summaries=block_summaries,
        optimizer=optimizer,
        stage_a_payload=stage_a_payload,
        default_family_weights=family_weights,
    )
    post_w_train = post_w_all[train_pos, :] if np.asarray(post_w_all).ndim == 2 else post_w_all

    train_seq_res = build_sequence_data(
        track_c.iloc[train_pos],
        labels[label_col].iloc[train_pos],
        seq_len,
        scaler_stats=scaler_stats,
        post_standardization_feature_weights=post_w_train,
    )
    if train_seq_res is None:
        raise ValueError("train sequence data too small")
    train_seq_data, _ = train_seq_res

    n_train = int(len(train_seq_data))
    split = int(n_train * 0.9)
    train_idx = np.arange(split)
    holdout_idx = np.arange(split, n_train)
    if len(holdout_idx) < 5:
        raise ValueError("train holdout too small")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_result = train_mamba_fold(train_seq_data, train_idx, holdout_idx, cfg, device=device, return_model=True)
    model = train_result.get("model")
    if model is None:
        raise ValueError("training failed: no model")

    # Stateful OOS loop.
    # Standardize full Track-C and apply post-standardization weights (may be time-varying).
    features_std = _standardize_features_like_build_sequence_data(
        track_c,
        scaler_stats,
        post_standardization_feature_weights=post_w_all,
    )
    _assert_finite_np(name="features_std", symbol=str(symbol).upper(), values=features_std, index=index)

    # Burn-in end pos is the row immediately before OOS start.
    burnin_end_pos = int(oos_pos[0]) - 1
    preds = _stateful_predict_roll_window(
        model=model,
        device=device,
        features_std=features_std,
        index=index,
        oos_pos=oos_pos,
        burnin_end_pos=burnin_end_pos,
        seq_len=seq_len,
    )
    _assert_finite_series(name="preds", symbol=str(symbol).upper(), s=preds)

    # Try to get model sigma using the new mu_sigma inference
    model_sigma: Optional[pd.Series] = None
    head_type = str(cfg.get("mamba_head_type", "linear")).lower()
    if head_type in ("gaussian", "uncertainty", "gaussian_nll"):
        try:
            _, model_sigma = _stateful_predict_roll_window_mu_sigma(
                model=model,
                device=device,
                features_std=features_std,
                index=index,
                oos_pos=oos_pos,
                burnin_end_pos=burnin_end_pos,
                seq_len=seq_len,
                sigma_floor=float(cfg.get("phase2_sigma_floor_base", 1e-6)),
                sigma_cap=float(cfg.get("phase2_sigma_cap", 10.0)),
            )
        except Exception as e:
            logger.debug(f"Failed to get model sigma, using proxy: {e}")
            model_sigma = None

    # Build preds_df and evaluate with Stage-B backtest engine (frozen decision logic).
    price_data = pipeline._get_price_data_for_horizon(int(horizon))
    if price_data is None or price_data.empty:
        raise ValueError("missing price data for backtest")

    strategy_cfg = _make_strategy_cfg_from_base(cfg)
    fee = float(strategy_cfg.get("fee_bp", 0.0)) / 10000.0
    slippage = float(strategy_cfg.get("slippage_bp", 0.0)) / 10000.0
    engine = BacktestEngine(price_data, fee=fee, slippage_bp=slippage)

    actual_returns = labels["forward_return"].reindex(preds.index).astype(float)
    preds_df = pd.DataFrame(index=preds.index)
    preds_df["actual_return"] = actual_returns
    preds_df["mu_hat"] = preds.astype(float)

    # Derive probability proxy and sigma
    # Use true model sigma if available, otherwise fall back to realized volatility proxy
    sigma_proxy = actual_returns.abs().rolling(window=max(5, int(horizon))).std().fillna(0.02).clip(lower=1e-4)
    
    if model_sigma is not None:
        # Use true model sigma (calibrated)
        sigma_hat = model_sigma.reindex(preds.index).clip(lower=1e-6, upper=10.0)
        # Fill any missing/non-finite with proxy
        sigma_hat = sigma_hat.where(sigma_hat.notna() & np.isfinite(sigma_hat), sigma_proxy)
        preds_df["sigma_hat"] = sigma_hat
        preds_df["sigma_source"] = "model"
    else:
        # Use proxy sigma
        preds_df["sigma_hat"] = sigma_proxy
        preds_df["sigma_source"] = "proxy"
    
    denom = preds_df["sigma_hat"].replace(0.0, np.nan).fillna(0.02)
    logits = (preds / denom).clip(-8, 8)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds_df["p_up"] = probs.clip(0.0, 1.0)
    # Confidence proxy in [0,1]: closer to 0.5 => low confidence.
    preds_df["rho"] = (preds_df["p_up"] - 0.5).abs().mul(2.0).clip(0.0, 1.0)
    preds_df["drift_flag"] = 0.0

    # Failure policy: if any non-finite shows up in the backtest inputs, fail explicitly.
    for col in ["mu_hat", "p_up", "sigma_hat", "actual_return"]:
        if col in preds_df:
            _assert_finite_series(name=f"preds_df[{col}]", symbol=str(symbol).upper(), s=preds_df[col])

    equity, metrics = engine.run(preds_df, int(horizon), strategy_cfg, symbol=str(symbol).upper())
    if "net_return" in equity:
        _assert_finite_series(name="equity.net_return", symbol=str(symbol).upper(), s=equity["net_return"].astype(float))
    if "equity_curve" in equity:
        _assert_finite_series(name="equity.equity_curve", symbol=str(symbol).upper(), s=equity["equity_curve"].astype(float))

    return equity, {k: float(v) for k, v in metrics.items()}, preds


def _build_trackc_multi_symbol(
    *,
    symbols: Sequence[str],
    horizon: int,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    cfg: Mapping[str, Any],
) -> Tuple[
    Dict[str, StageBPipeline],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, Tuple[np.ndarray, List[str]]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """Build panel/labels/block_summaries/track_c per symbol using pooled Track-A fitting.

        Returns:
            pipelines_by_symbol, panel_by_symbol, labels_by_symbol, trackc_by_symbol,
            post_std_weights_by_symbol (vec, cols), train_pos_by_symbol,
            oos_pos_by_symbol (on each symbol's *valid label* index)
    """

    from src.stage_b.optuna_optimizer import OptunaConfig, StageBOptunaOptimizer, STAGE_A_FAMILIES

    syms = [str(s).upper() for s in symbols]
    if not syms:
        raise ValueError("symbols must be non-empty")

    pipelines: Dict[str, StageBPipeline] = {}
    panels: Dict[str, pd.DataFrame] = {}
    labels_by: Dict[str, pd.DataFrame] = {}
    train_pos_by: Dict[str, np.ndarray] = {}
    oos_pos_by: Dict[str, np.ndarray] = {}

    label_col = _phase2_label_column(_phase2_label_id_from_cfg(cfg))

    # Optional authoritative universe registry gate (preferred over delisting_meta).
    # This prevents zombie symbols from entering panel build at all, and also caps
    # panel history to each symbol's stop_date.
    _universe_stop_by_symbol: Dict[str, pd.Timestamp] = {}
    try:
        reg_path_raw = cfg.get("phase2_universe_registry_path")
        reg_enabled = bool(cfg.get("phase2_universe_registry_enabled", True)) or bool(reg_path_raw)
        reg_strict = bool(cfg.get("phase2_universe_registry_strict", False))
        if reg_enabled:
            from pathlib import Path

            from src.stage_b.universe_registry import load_universe_registry

            reg_path = Path(str(reg_path_raw)) if reg_path_raw else (Path("data") / "cache" / "universe" / "universe_registry.parquet")
            if reg_strict and not reg_path.exists():
                raise ValueError(
                    "phase2_universe_registry_strict enabled but no universe_registry parquet exists; "
                    "run the registry refresh job or set phase2_universe_registry_path"
                )
            reg_df = load_universe_registry(reg_path)
            if reg_df is not None and not reg_df.empty and "symbol" in reg_df.columns and "stop_date" in reg_df.columns:
                tmp = reg_df[["symbol", "stop_date"]].copy()
                tmp["symbol"] = tmp["symbol"].astype(str).str.upper()
                tmp["stop_date"] = pd.to_datetime(tmp["stop_date"], errors="coerce").dt.tz_localize(None)
                _universe_stop_by_symbol = {
                    str(r["symbol"]).upper(): pd.Timestamp(r["stop_date"]).normalize()
                    for _, r in tmp.dropna(subset=["symbol", "stop_date"]).iterrows()
                }

            # Pre-filter symbols before *any* panel build.
            t_train_start = _to_datetime(train_start).normalize()
            kept: list[str] = []
            dropped: list[str] = []
            for s in syms:
                stop_dt = _universe_stop_by_symbol.get(str(s).upper())
                if stop_dt is None or pd.isna(stop_dt):
                    if reg_strict:
                        dropped.append(str(s).upper())
                        continue
                    kept.append(str(s).upper())
                    continue
                if pd.Timestamp(stop_dt).normalize() < pd.Timestamp(t_train_start).normalize():
                    dropped.append(str(s).upper())
                    continue
                kept.append(str(s).upper())
            if dropped:
                logger.warning(
                    "[phase2.universe_registry] dropped %d symbols before panel build (stop_date < train_start or missing): %s",
                    len(dropped),
                    ",".join(dropped[:25]) + ("..." if len(dropped) > 25 else ""),
                )
            syms = kept
            if not syms:
                raise ValueError("All symbols were filtered out by universe_registry before panel build")
    except Exception as e:
        logger.warning("[phase2.universe_registry] failed to load/apply universe registry: %s", e)
        _universe_stop_by_symbol = {}

    # Optional survivorship control (delisting metadata). This is intentionally
    # *not* a model feature family; it only caps history to avoid post-delisting leakage.
    delisted_date_by_symbol: Dict[str, pd.Timestamp] = {}
    try:
        delisting_enabled = bool(cfg.get("phase2_delisting_meta_enabled", False))
        delisting_strict = bool(cfg.get("phase2_delisting_meta_strict", False))
        if delisting_enabled:
            from src.stage_b.delisting_meta import (
                get_delisted_date_map,
                load_delisting_registry,
                refresh_delisting_registry,
            )

            exchange_code = str(cfg.get("phase2_delisting_meta_exchange", "US") or "US").upper()
            cache_path = cfg.get("phase2_delisting_meta_cache_path")
            refresh = bool(cfg.get("phase2_delisting_meta_refresh", False))

            # Strict mode means: do not silently run without a registry source.
            # An empty registry is acceptable; missing registry source is not.
            if delisting_strict and not refresh:
                from pathlib import Path

                implied = Path(str(cache_path)) if cache_path else (Path("data") / "cache" / "eodhd" / f"delisted_companies_{exchange_code}.parquet")
                if not implied.exists():
                    raise ValueError(
                        "phase2_delisting_meta_strict enabled but no local delisting registry cache exists; "
                        "run tools/build_delisting_meta.py or set phase2_delisting_meta_refresh=true"
                    )
            if refresh:
                from src.data_sources.eodhd_provider import EODHDProvider

                provider = EODHDProvider()
                # Prefer bounded per-symbol refresh (fundamentals-based) for Phase-2.
                try:
                    from src.stage_b.delisting_meta import refresh_delisting_registry_for_symbols

                    reg = refresh_delisting_registry_for_symbols(
                        provider,
                        symbols=syms,
                        exchange_code=exchange_code,
                        cache_path=str(cache_path) if cache_path else None,
                        timeout_seconds=int(cfg.get("phase2_delisting_meta_timeout_seconds", 60) or 60),
                    )
                except Exception:
                    # Fallback to the historical delisted-companies endpoint (may be plan-gated).
                    reg = refresh_delisting_registry(
                        provider,
                        exchange_code=exchange_code,
                        start_date=str(cfg.get("phase2_delisting_meta_start_date") or train_start),
                        end_date=str(cfg.get("phase2_delisting_meta_end_date") or oos_end),
                        cache_path=str(cache_path) if cache_path else None,
                        timeout_seconds=int(cfg.get("phase2_delisting_meta_timeout_seconds", 120) or 120),
                        chunk_days=int(cfg.get("phase2_delisting_meta_chunk_days", 7) or 7),
                    )
            else:
                reg = load_delisting_registry(
                    exchange_code=exchange_code,
                    cache_path=str(cache_path) if cache_path else None,
                )
            delisted_date_by_symbol = get_delisted_date_map(reg)
    except Exception as e:
        logger.warning("[phase2.delisting_meta] failed to load delisting registry: %s", e)
        delisted_date_by_symbol = {}

    # Build per-symbol panels + labels and align to non-NaN labels.
    for sym in syms:
        pipe = _build_stage_b_pipeline(symbol=sym, horizon=int(horizon), start=str(train_start), end=str(oos_end))
        panel = pipe._build_panel(int(horizon))

        # Preferred cap: authoritative universe registry stop_date (covers delisting + stale OHLCV).
        stop_dt = _universe_stop_by_symbol.get(str(sym).upper())
        if stop_dt is not None and not pd.isna(stop_dt):
            stop_dt = pd.Timestamp(stop_dt).normalize()
            panel = panel.loc[pd.DatetimeIndex(panel.index).normalize() <= stop_dt]
            if panel.empty:
                raise ValueError(f"{sym} panel is empty after applying universe registry stop_date cap at {stop_dt.date()}")

        delist_dt = delisted_date_by_symbol.get(str(sym).upper())
        if delist_dt is not None and not pd.isna(delist_dt):
            delist_dt = pd.Timestamp(delist_dt).normalize()
            # Back-compat cap (only if registry not present).
            if stop_dt is None or pd.isna(stop_dt):
                panel = panel.loc[pd.DatetimeIndex(panel.index).normalize() <= delist_dt]
            if panel.empty:
                raise ValueError(f"{sym} panel is empty after applying delisting cap at {delist_dt.date()}")

        labels = pipe._construct_labels(int(horizon), panel.index)
        valid_mask = labels[label_col].notna()
        panel = panel.loc[valid_mask]
        labels = labels.loc[valid_mask]

        idx = pd.DatetimeIndex(panel.index)
        t_train_start = _to_datetime(train_start)
        t_train_end = _to_datetime(train_end)
        t_oos_start = _to_datetime(oos_start)

        train_pos = _select_index_positions(idx, t_train_start, t_train_end)

        oos_end_ts = _to_datetime(oos_end)
        if stop_dt is not None and not pd.isna(stop_dt):
            oos_end_ts = min(pd.Timestamp(oos_end_ts).normalize(), pd.Timestamp(stop_dt).normalize())
        elif delist_dt is not None and not pd.isna(delist_dt):
            oos_end_ts = min(pd.Timestamp(oos_end_ts).normalize(), pd.Timestamp(delist_dt).normalize())
        oos_pos = _select_index_positions(idx, t_oos_start, oos_end_ts)
        if delist_dt is not None and not pd.isna(delist_dt) and len(oos_pos) == 0:
            raise ValueError(
                f"{sym} has no OOS rows after delisting cap (delisted={pd.Timestamp(delist_dt).date()}, oos_start={t_oos_start.date()})"
            )

        pipelines[sym] = pipe
        panels[sym] = panel
        labels_by[sym] = labels
        train_pos_by[sym] = train_pos
        oos_pos_by[sym] = oos_pos

    # Fit Track-A encoders on pooled train rows across all symbols.
    first_pipe = pipelines[syms[0]]
    column_families = first_pipe._infer_column_families(panels[syms[0]].columns)

    pooled_panel = pd.concat([panels[sym].iloc[train_pos_by[sym]] for sym in syms], axis=0)
    pooled_idx = np.arange(len(pooled_panel), dtype=int)

    opt_cfg = OptunaConfig(
        max_total_dims=0,
        horizon=int(horizon),
        sequence_model_type="mamba",
        use_three_pillar_dims=True,
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
    )
    try:
        setattr(opt_cfg, "weight_normalization", "none")
    except Exception:
        pass

    optimizer = StageBOptunaOptimizer(config=opt_cfg)

    from src.stage_b.optuna_optimizer import _is_raw_passthrough_family

    family_sizes: Dict[str, int] = {}
    for _col, fam in column_families.items():
        if fam:
            family_sizes[str(fam)] = int(family_sizes.get(str(fam), 0) + 1)

    stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(syms[0]).upper(), horizon=int(horizon))
    stage_a_weights_static = _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end))

    cache_key = _three_pillar_cache_key(
        symbols=syms,
        horizon=horizon,
        train_start=train_start,
        train_end=train_end,
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
        stage_a_weights=stage_a_weights_static,
    )
    cached_three_pillar = _load_three_pillar_cache(cache_key)

    if cached_three_pillar is not None:
        optimizer.family_sizes = cached_three_pillar.get("family_sizes", {})
        optimizer.family_optimal_dims = cached_three_pillar.get("family_optimal_dims", {})
        logger.info(
            f"[3-pillar pooled] Restored {len(optimizer.family_optimal_dims)} family dims from disk cache"
        )
    else:
        logger.info(f"[3-pillar pooled] 🔧 DISK CACHE MISS - running pooled 3-pillar for {len(syms)} symbols")
        optimizer._analyze_families(pooled_panel, column_families, stage_a_weights=stage_a_weights_static or None)
        _save_three_pillar_cache(
            cache_key,
            {
                "family_sizes": getattr(optimizer, "family_sizes", {}),
                "family_optimal_dims": getattr(optimizer, "family_optimal_dims", {}),
            },
        )

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

    optimizer._build_track_a(pooled_panel, column_families, family_params, pooled_idx)

    trackc_by: Dict[str, pd.DataFrame] = {}
    post_std_weights_by: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for sym in syms:
        pipe = pipelines[sym]
        panel = panels[sym]
        train_pos = train_pos_by[sym]

        pipe._current_train_idx = train_pos
        block_summaries = pipe._compute_block_summaries(panel)

        track_a, _ = optimizer._build_track_a(panel, column_families, family_params, train_pos)
        track_b = optimizer._build_track_b(
            panel,
            column_families,
            block_summaries,
            track_b_family_weights=None,
        )
        track_c = optimizer._build_track_c(track_a, track_b, 1.0, 1.0)
        if track_c is None or track_c.empty:
            raise ValueError(f"Track C is empty for {sym}")
        trackc_by[sym] = track_c

        stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(sym).upper(), horizon=int(horizon))
        family_weights = (
            _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end))
            or _resolve_phase2_family_weights(cfg, symbol=str(sym).upper(), horizon=int(horizon))
        )
        post_w, post_cols = _build_time_varying_post_std_weight_matrix(
            index=pd.DatetimeIndex(track_c.index),
            track_c=track_c,
            track_a=track_a,
            panel=panel,
            block_summaries=block_summaries,
            optimizer=optimizer,
            stage_a_payload=stage_a_payload,
            default_family_weights=family_weights,
        )
        post_std_weights_by[sym] = (np.asarray(post_w, dtype=np.float32), list(post_cols))

    # ─────────────────────────────────────────────────────────────────────────────
    # Symbol Graph Context (cross-sectional embeddings + neighbor features)
    # Runs AFTER Track-C is built, using the ActiveUniverse(t) = syms + anchor ETFs
    # ─────────────────────────────────────────────────────────────────────────────
    sgc_enabled = bool(cfg.get("sgc_enabled", True))
    if sgc_enabled and len(syms) >= 2:
        try:
            # Extract price series (close) from panels for correlation adjacency
            price_series_by_symbol: Dict[str, pd.Series] = {}
            for sym in syms:
                pnl = panels.get(sym)
                if pnl is not None and "ohlcv_close" in pnl.columns:
                    price_series_by_symbol[sym] = pnl["ohlcv_close"].copy()
                elif pnl is not None and "close" in pnl.columns:
                    price_series_by_symbol[sym] = pnl["close"].copy()

            # Extract optional sector/group map for sector adjacency blending
            group_map_by_symbol: Optional[Dict[str, str]] = None
            group_map_path = cfg.get("phase2_symbol_group_map_path")
            if group_map_path:
                try:
                    gmap_df = pd.read_csv(str(group_map_path))
                    if "symbol" in gmap_df.columns and "group" in gmap_df.columns:
                        group_map_by_symbol = dict(zip(
                            gmap_df["symbol"].astype(str).str.upper(),
                            gmap_df["group"].astype(str),
                        ))
                except Exception as gmap_err:
                    logger.debug(f"[SGC] Failed to load group map: {gmap_err}")

            # Build the SGC panel for all active symbols using multi-symbol builder
            sgc_panel = build_sgc_for_multi_symbol(
                active_symbols=syms,
                price_series_by_symbol=price_series_by_symbol,
                group_map_by_symbol=group_map_by_symbol,
                train_end_date=train_end,
                corr_window=int(cfg.get("sgc_corr_window", 63)),
                embed_dim=int(cfg.get("sgc_embed_dim", 8)),
                top_k=int(cfg.get("sgc_top_k", 10)),
                sector_blend_weight=float(cfg.get("sgc_sector_blend_weight", 0.3)),
                temperature_base=float(cfg.get("sgc_temperature_base", 1.0)),
                regime_aware=bool(cfg.get("sgc_regime_aware", True)),
                anchor_etfs=list(cfg.get("sgc_anchor_etfs", ["SPY", "QQQ", "IWM", "XLF", "XLE", "XLV", "XLK", "XLI", "XLU", "XLP", "XLY", "TLT", "GLD", "HYG", "VIX", "EEM", "EFA", "DIA"])),
                cache_dir=str(cfg.get("sgc_cache_dir", "cache/shared/symbol_graph")),
            )

            # Merge SGC features into each symbol's Track-C
            sgc_cols = get_symbol_graph_context_columns()
            if sgc_panel is not None and not sgc_panel.empty:
                for sym in syms:
                    tc = trackc_by.get(sym)
                    if tc is None or tc.empty:
                        continue
                    # Filter SGC panel to this symbol
                    if "symbol" in sgc_panel.columns:
                        sym_sgc = sgc_panel[sgc_panel["symbol"].astype(str).str.upper() == str(sym).upper()].copy()
                        sym_sgc = sym_sgc.set_index("date") if "date" in sym_sgc.columns else sym_sgc
                    else:
                        sym_sgc = sgc_panel  # Assume already per-symbol

                    # Align to Track-C index (forward-fill for weekly adjacency)
                    sym_sgc_aligned = sym_sgc.reindex(tc.index, method="ffill")
                    for col in sgc_cols:
                        if col in sym_sgc_aligned.columns:
                            tc[col] = sym_sgc_aligned[col].values
                    trackc_by[sym] = tc

                n_sgc_cols = len([c for c in sgc_cols if c in trackc_by[syms[0]].columns])
                logger.info(f"[SGC] ✓ Merged {n_sgc_cols} symbol_graph_context features into Track-C for {len(syms)} symbols")
        except Exception as sgc_err:
            logger.warning(f"[SGC] Failed to compute symbol_graph_context: {sgc_err}", exc_info=True)

    return pipelines, panels, labels_by, trackc_by, post_std_weights_by, train_pos_by, oos_pos_by


def _annualized_sharpe(returns: pd.Series, trading_days: int = 252) -> float:
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std())
    if std <= 0 or not np.isfinite(std):
        return 0.0
    return float(r.mean() / (std + 1e-9) * np.sqrt(trading_days))


def _max_drawdown_from_equity(equity_curve: pd.Series) -> float:
    eq = equity_curve.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    dd = 1.0 - (eq / running_max.replace(0.0, np.nan)).fillna(1.0)
    return float(dd.max()) if len(dd) else 0.0


def _normalize_portfolio_weights(
    symbols: Sequence[str],
    portfolio_weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    syms = [str(s).upper() for s in symbols]
    if not portfolio_weights:
        w = 1.0 / max(1, len(syms))
        return {s: w for s in syms}

    weights = {str(k).upper(): float(v) for k, v in portfolio_weights.items() if str(k).upper() in syms}
    if not weights:
        w = 1.0 / max(1, len(syms))
        return {s: w for s in syms}

    total = float(sum(abs(v) for v in weights.values()))
    if total <= 0:
        w = 1.0 / max(1, len(syms))
        return {s: w for s in syms}
    return {k: float(v) / total for k, v in weights.items()}


def _pick_price_col(df: pd.DataFrame) -> str:
    for c in ["adj_close", "Adj Close", "close", "Close", "price", "Price"]:
        if c in df.columns:
            return str(c)
    # Fallback: first numeric column if available.
    try:
        num_cols = list(df.select_dtypes(include=["number", "bool"]).columns)
        if num_cols:
            return str(num_cols[0])
    except Exception:
        pass
    return "close"


def _load_symbol_group_map(path: Optional[str], *, symbols: Sequence[str]) -> Optional[Dict[str, str]]:
    """Load a symbol->group mapping for runtime exposure caps.

    Supported formats:
    - JSON dict: {"AAPL": "tech", ...}
    - CSV with columns: symbol, group
    """

    if not path:
        return None
    try:
        p = Path(str(path))
        if not p.exists():
            return None

        out: Dict[str, str]
        if p.suffix.lower() in {".json"}:
            text = p.read_text()
            data = json.loads(text)
            if isinstance(data, dict):
                out = {str(k).upper(): str(v) for k, v in data.items()}
            else:
                return None
        elif p.suffix.lower() in {".csv"}:
            df = pd.read_csv(p)
            if "symbol" not in df.columns or "group" not in df.columns:
                return None
            out = {str(r["symbol"]).upper(): str(r["group"]) for _, r in df.iterrows()}
        else:
            logger.warning("Unsupported group map format: %s", p)
            return None

        syms = {str(s).upper() for s in symbols}
        return {s: g for s, g in out.items() if s in syms}
    except Exception:
        logger.exception("Failed to load phase2 group map")
        return None


def _apply_group_exposure_caps(
    w: np.ndarray,
    *,
    symbols: Sequence[str],
    group_by_symbol: Optional[Mapping[str, str]],
    max_group_gross: float,
    max_group_net: float,
) -> np.ndarray:
    """Apply simple per-group gross/net caps by scaling weights within each group."""

    if not group_by_symbol:
        return w

    w = np.asarray(w, dtype=float)
    max_gross = float(max_group_gross)
    max_net = float(max_group_net)
    if (not np.isfinite(max_gross) or max_gross <= 0) and (not np.isfinite(max_net) or max_net <= 0):
        return w

    syms = [str(s).upper() for s in symbols]
    # Build group -> indices
    idxs_by_group: Dict[str, List[int]] = {}
    for j, s in enumerate(syms):
        g = group_by_symbol.get(s) if group_by_symbol is not None else None
        if not g:
            continue
        idxs_by_group.setdefault(str(g), []).append(int(j))

    if not idxs_by_group:
        return w

    w_out = w.copy()
    for g, idxs in idxs_by_group.items():
        sub = w_out[idxs]
        gross = float(np.sum(np.abs(sub)))
        if np.isfinite(max_gross) and max_gross > 0 and gross > max_gross and gross > 0:
            w_out[idxs] = sub * (max_gross / gross)
            sub = w_out[idxs]
        net = float(np.sum(sub))
        if np.isfinite(max_net) and max_net > 0 and abs(net) > max_net and abs(net) > 0:
            w_out[idxs] = sub * (max_net / abs(net))

    return w_out


def _apply_basic_constraints(
    w: np.ndarray,
    *,
    max_name: float,
    max_gross: float,
    max_net: float,
) -> np.ndarray:
    """Apply basic portfolio constraints (name cap, gross cap, net cap).

    This is a best-effort projector used inside Phase2 v2. It preserves the
    relative structure of weights via clipping + global scaling.
    """

    w_out = np.asarray(w, dtype=float).copy()
    w_out = np.where(np.isfinite(w_out), w_out, 0.0)

    mn = float(max_name)
    if np.isfinite(mn) and mn > 0:
        w_out = np.clip(w_out, -mn, mn)

    mg = float(max_gross)
    gross = float(np.sum(np.abs(w_out)))
    if np.isfinite(mg) and mg > 0 and np.isfinite(gross) and gross > mg and gross > 0:
        w_out *= mg / gross

    mn2 = float(max_net)
    net = float(np.sum(w_out))
    if np.isfinite(mn2) and mn2 > 0 and np.isfinite(net) and abs(net) > mn2 and abs(net) > 0:
        w_out *= mn2 / abs(net)

    return w_out


def _project_turnover_budget(prev_w: np.ndarray, w: np.ndarray, *, turnover_cap: float) -> np.ndarray:
    """Project weights onto an L1 turnover budget relative to `prev_w`.

    Turnover is defined as: sum_i |w_i - prev_w_i|.
    If the requested turnover exceeds `turnover_cap`, we shrink the update step.
    """

    cap = float(turnover_cap)
    if not np.isfinite(cap) or cap <= 0:
        return np.asarray(w, dtype=float)

    p = np.asarray(prev_w, dtype=float).reshape(-1)
    x = np.asarray(w, dtype=float).reshape(-1)
    if p.shape != x.shape:
        return x

    delta = x - p
    t = float(np.sum(np.abs(delta)))
    if not np.isfinite(t) or t <= cap or t <= 0:
        return x
    scale = cap / t
    return p + delta * scale


def _ewma_cov_update(cov: np.ndarray, r_vec: np.ndarray, lam: float) -> np.ndarray:
    """EWMA covariance update: cov <- lam*cov + (1-lam) * (r r^T)."""

    c = np.asarray(cov, dtype=float)
    r = np.asarray(r_vec, dtype=float).reshape(-1)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] != r.shape[0]:
        return c

    l = float(lam)
    if not np.isfinite(l):
        l = 0.94
    l = float(np.clip(l, 0.0, 0.9999))

    rr = np.outer(np.where(np.isfinite(r), r, 0.0), np.where(np.isfinite(r), r, 0.0))
    c0 = np.where(np.isfinite(c), c, 0.0)
    out = l * c0 + (1.0 - l) * rr
    return 0.5 * (out + out.T)


def _estimate_beta_vector(
    returns_window: np.ndarray,
    *,
    market: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate per-asset beta to a market proxy using a simple covariance ratio.

    returns_window: shape (T, N)
    market: shape (T,), optional. If None, uses equal-weight market proxy.
    """

    x = np.asarray(returns_window, dtype=float)
    if x.ndim != 2 or x.shape[0] < 5:
        return np.zeros((x.shape[1],), dtype=float)
    if market is None:
        m = np.nanmean(x, axis=1)
    else:
        m = np.asarray(market, dtype=float).reshape(-1)
        if m.shape[0] != x.shape[0]:
            m = np.nanmean(x, axis=1)

    m = np.where(np.isfinite(m), m, 0.0)
    xm = np.where(np.isfinite(x), x, 0.0)

    m0 = m - float(np.mean(m))
    var_m = float(np.mean(m0 * m0))
    if not np.isfinite(var_m) or var_m <= eps:
        return np.zeros((xm.shape[1],), dtype=float)

    betas = []
    for j in range(xm.shape[1]):
        r = xm[:, j]
        r0 = r - float(np.mean(r))
        cov = float(np.mean(r0 * m0))
        betas.append(cov / (var_m + eps))
    return np.asarray(betas, dtype=float)


def _apply_beta_neutralization(w: np.ndarray, beta: np.ndarray, *, max_abs_beta_exposure: float) -> np.ndarray:
    """Apply HARD beta neutralization with optional exposure cap.
    
    This is NOT a "soft guardrail" - it ALWAYS neutralizes beta exposure first,
    then optionally allows a capped beta exposure if max_abs_beta_exposure > 0.
    
    Args:
        w: Portfolio weights [n_assets]
        beta: Beta vector [n_assets]
        max_abs_beta_exposure: Maximum absolute beta exposure (0 = full neutralization)
    
    Returns:
        Adjusted weights with beta neutralized/capped
    """
    w = np.asarray(w, dtype=float)
    b = np.asarray(beta, dtype=float).reshape(-1)
    if w.shape[0] != b.shape[0]:
        return w

    denom = float(b @ b)
    if not np.isfinite(denom) or denom <= 1e-12:
        return w
    exposure = float(b @ w)

    # STEP 1: Always neutralize first (HARD neutralization, not soft)
    w0 = w - b * (exposure / denom)

    # STEP 2: If cap specified, allow up to max_abs_beta_exposure
    cap = float(max_abs_beta_exposure)
    if not np.isfinite(cap) or cap <= 0:
        # No cap specified → full neutralization
        return w0
    
    exp0 = float(b @ w0)
    if abs(exp0) <= cap:
        # Already within cap (numeric precision)
        return w0
    
    # If numeric drift remains, scale the component.
    # (This should rarely trigger.)
    return w0 - b * ((exp0 - np.sign(exp0) * cap) / denom)


def _rolling_realized_vol(net_returns: np.ndarray, *, window: int, trading_days: int = 252) -> float:
    w = int(window)
    if w <= 2:
        return 0.0
    r = np.asarray(net_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    tail = r[-w:] if r.size >= w else r
    std = float(np.std(tail, ddof=1)) if tail.size >= 2 else 0.0
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(std * np.sqrt(float(trading_days)))


def _policy_reward_from_window(
    *,
    net_returns: np.ndarray,
    turnover: np.ndarray,
    equity: np.ndarray,
    start: int,
    end: int,
    lambda_turnover: float,
    lambda_drawdown: float,
) -> float:
    if end < start:
        return 0.0
    s = int(max(0, start))
    e = int(min(len(net_returns) - 1, end))
    if e < s:
        return 0.0
    r = np.asarray(net_returns[s : e + 1], dtype=float)
    t = np.asarray(turnover[s : e + 1], dtype=float)
    eq = np.asarray(equity[s : e + 1], dtype=float)
    pnl = float(np.sum(np.where(np.isfinite(r), r, 0.0)))
    turn_mean = float(np.mean(np.where(np.isfinite(t), t, 0.0))) if t.size else 0.0
    dd = float(_max_drawdown_from_equity(pd.Series(eq))) if eq.size else 0.0
    return float(pnl - float(lambda_turnover) * turn_mean - float(lambda_drawdown) * dd)


# Import v2 state utilities from policy_controller
from src.portfolio.policy_controller import (
    POLICY_STATE_DIM_V2,
    compute_vol_of_vol,
    compute_correlation_hhi,
    compute_expected_shortfall,
    compute_drawdown_velocity,
    compute_mu_sigma_dispersion,
    normalize_state_feature,
    compute_policy_reward,  # v2 reward function
)


def _policy_state_vector(
    *,
    i: int,
    horizon: int,
    returns_df: pd.DataFrame,
    net_ret: np.ndarray,
    turnover: np.ndarray,
    costs: np.ndarray,
    equity: np.ndarray,
    mu_mat: np.ndarray,
    sigma_mat: np.ndarray,  # NEW: predicted sigma
    fwd_ret_mat: np.ndarray,
    market_regime: pd.Series,
    state_window: int = 20,
    sharpe_window: int = 63,
    calib_window: int = 63,
    # Portfolio family features from RoleAwareDayContext
    cboe_panic_premium: float = 0.0,
    cboe_term_slope: float = 0.0,
    cboe_vol_risk_premium_z: float = 0.0,
    calibration_overall_score: float = 1.0,
    online_trust_score: float = 1.0,
    # NEW v2: Additional context for expanded state
    weights: Optional[np.ndarray] = None,
    sector_map: Optional[Dict[int, str]] = None,
    adv_arr: Optional[np.ndarray] = None,  # Average daily volume
) -> np.ndarray:
    """Compute 25-dimensional policy state vector (v2).
    
    Dimensions 0-9: Portfolio performance metrics (historical) - PRESERVED
    Dimensions 10-14: Portfolio family features (CBOE, calibration, trust) - PRESERVED
    Dimensions 15-24: NEW v2 features for enhanced policy control:
      - Tail risk (Expected Shortfall)
      - Volatility-of-volatility
      - Liquidity stress
      - Model reliability metrics
      - Regime transition indicators
      - Drawdown momentum
      - Sector concentration
    
    This gives the contextual bandit comprehensive information about:
    - How the portfolio has been performing (adapt based on recent results)
    - Current market stress/quality conditions (adapt based on environment)
    - Model reliability and regime transitions (adapt proactively)
    """
    # Use only information up to t-1.
    end = int(max(0, i - 1))
    if end <= 0:
        return np.zeros((POLICY_STATE_DIM_V2,), dtype=float)

    w = int(max(5, state_window))
    start = int(max(0, end - w + 1))
    r_hist = np.asarray(net_ret[start : end + 1], dtype=float)
    t_hist = np.asarray(turnover[start : end + 1], dtype=float)
    c_hist = np.asarray(costs[start : end + 1], dtype=float)
    eq_hist = np.asarray(equity[start : end + 1], dtype=float)

    realized_vol = float(np.std(r_hist) * np.sqrt(252.0)) if r_hist.size >= 2 else 0.0
    dd = float(_max_drawdown_from_equity(pd.Series(eq_hist))) if end >= start else 0.0

    sw = int(max(10, sharpe_window))
    s_start = int(max(0, end - sw + 1))
    r_sh = np.asarray(net_ret[s_start : end + 1], dtype=float)
    sharpe = float(_annualized_sharpe(pd.Series(r_sh))) if r_sh.size >= 5 else 0.0

    turn_mean = float(np.mean(np.where(np.isfinite(t_hist), t_hist, 0.0))) if t_hist.size else 0.0
    cost_mean = float(np.mean(np.where(np.isfinite(c_hist), c_hist, 0.0))) if c_hist.size else 0.0

    # Correlation concentration and dispersion from cross-asset returns.
    corr_hhi = 0.0
    avg_corr = 0.0
    disp = 0.0
    top_eigen_share = 0.0  # FIX Gap #7: Use top eigenvalue share instead of duplicating corr_hhi
    try:
        r_win = returns_df.iloc[start : end + 1].to_numpy(dtype=float)
        if r_win.shape[0] >= 5:
            r_win = np.where(np.isfinite(r_win), r_win, 0.0)
            corr = np.corrcoef(r_win.T)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            # Use HHI of eigenvalues for concentration (v2)
            corr_hhi = compute_correlation_hhi(corr)
            if corr.shape[0] > 1:
                avg_corr = float((np.sum(corr) - np.trace(corr)) / (corr.shape[0] * (corr.shape[0] - 1)))
            disp = float(np.mean(np.std(r_win, axis=1)))
            # FIX Gap #7: Compute top eigenvalue share as distinct metric
            try:
                eigs = np.linalg.eigvalsh(corr)
                eigs = np.sort(eigs)[::-1]  # Descending
                eigs = np.maximum(eigs, 0.0)  # Numerical stability
                total = float(np.sum(eigs))
                if total > 0:
                    top_eigen_share = float(eigs[0] / total)  # First PC variance share
            except Exception:
                top_eigen_share = 0.0
    except Exception:
        pass

    # Calibration proxy: sign accuracy on matured forward returns.
    calib_end = int(max(0, end - int(horizon)))
    calib_start = int(max(0, calib_end - int(max(5, calib_window)) + 1))
    acc = 0.0
    if calib_end >= calib_start:
        mu_hist = np.asarray(mu_mat[calib_start : calib_end + 1], dtype=float)
        fr_hist = np.asarray(fwd_ret_mat[calib_start : calib_end + 1], dtype=float)
        m = np.isfinite(mu_hist) & np.isfinite(fr_hist)
        if np.any(m):
            mu_s = np.sign(mu_hist[m])
            fr_s = np.sign(fr_hist[m])
            acc = float(np.mean(mu_s == fr_s)) if mu_s.size else 0.0

    try:
        regime_val = float(market_regime.iloc[end]) if end < len(market_regime) else 0.0
    except Exception:
        regime_val = 0.0

    # =========================================================================
    # v2 NEW FEATURES (dims 15-24)
    # =========================================================================
    
    # 15: Tail risk - Expected Shortfall at 5%
    tail_risk_es = compute_expected_shortfall(r_hist, alpha=0.05)
    
    # 16: Volatility-of-volatility (market stress indicator)
    vol_of_vol = compute_vol_of_vol(r_hist, vol_window=5, vov_window=min(20, len(r_hist)))
    
    # 17: Liquidity stress (inverse of ADV, normalized)
    # FIX Gap #6: adv_arr is per-asset (1D), not per-time (2D)
    liquidity_stress = 0.0
    if adv_arr is not None:
        try:
            adv = np.asarray(adv_arr, dtype=float)
            adv = adv[np.isfinite(adv)]
            if adv.size > 0:
                # Lower ADV = higher stress
                # Use min ADV relative to mean as stress indicator
                mean_adv = float(np.mean(adv))
                min_adv = float(np.min(adv))
                if mean_adv > 0:
                    # Stress = 1 - (min/mean), higher when one asset has low ADV
                    liquidity_stress = float(1.0 - min_adv / mean_adv)
                    liquidity_stress = float(np.clip(liquidity_stress, 0.0, 1.0))
        except Exception:
            liquidity_stress = 0.0
    
    # 18: Model reliability - mean sigma_reliability (from predictions)
    mu_reliability_mean = 0.0
    if sigma_mat is not None and end < len(sigma_mat):
        sigma_day = np.asarray(sigma_mat[end], dtype=float)
        sigma_day = sigma_day[np.isfinite(sigma_day)]
        if sigma_day.size > 0:
            # Lower sigma = higher reliability
            mean_sigma = float(np.mean(sigma_day))
            mu_reliability_mean = 1.0 / (1.0 + mean_sigma) if mean_sigma > 0 else 1.0
    
    # 19: |μ|/σ dispersion (model confidence spread)
    mu_sigma_dispersion = 0.0
    if mu_mat is not None and sigma_mat is not None and end < len(mu_mat):
        mu_day = np.asarray(mu_mat[end], dtype=float)
        sigma_day = np.asarray(sigma_mat[end], dtype=float)
        mu_sigma_dispersion = compute_mu_sigma_dispersion(mu_day, sigma_day)
    
    # 20: Sector imbalance (max sector deviation from equal weight)
    sector_imbalance = 0.0
    if weights is not None and sector_map is not None:
        from src.portfolio.policy_controller import compute_sector_imbalance
        sector_imbalance = compute_sector_imbalance(weights, sector_map)
    
    # 21: Recent hit rate (last 5 sessions sign accuracy)
    recent_hit_rate = 0.0
    recent_end = int(max(0, end - int(horizon)))
    recent_start = int(max(0, recent_end - 5))
    if recent_end >= recent_start and recent_end < len(mu_mat):
        mu_recent = np.asarray(mu_mat[recent_start : recent_end + 1], dtype=float)
        fr_recent = np.asarray(fwd_ret_mat[recent_start : recent_end + 1], dtype=float)
        m = np.isfinite(mu_recent) & np.isfinite(fr_recent)
        if np.any(m):
            recent_hit_rate = float(np.mean(np.sign(mu_recent[m]) == np.sign(fr_recent[m])))
    
    # 22: Drawdown velocity (momentum of drawdown)
    dd_velocity = compute_drawdown_velocity(eq_hist, window=5)
    
    # 23: Regime transition probability (change frequency)
    regime_trans_prob = 0.0
    try:
        if end >= 5 and len(market_regime) > end:
            regime_hist = market_regime.iloc[max(0, end - 20) : end + 1].to_numpy()
            regime_hist = regime_hist[np.isfinite(regime_hist)]
            if len(regime_hist) >= 2:
                changes = np.sum(np.diff(regime_hist) != 0)
                regime_trans_prob = float(changes) / (len(regime_hist) - 1)
    except Exception:
        pass
    
    # 24: Top eigenvalue share (distinct from corr_hhi in dim 6)
    # Measures first principal component's variance share

    # =========================================================================
    # Build 25-dim state vector
    # =========================================================================
    vec = np.asarray(
        [
            # Dims 0-9: Portfolio performance metrics (PRESERVED from v1)
            float(np.clip(realized_vol, 0.0, 1.0)),
            float(np.clip(dd, 0.0, 1.0)),
            float(np.clip(sharpe / 3.0, -1.0, 1.0)),
            float(np.clip(turn_mean, 0.0, 1.0)),
            float(np.clip(cost_mean, 0.0, 1.0)),
            float(np.clip(avg_corr, -1.0, 1.0)),
            float(np.clip(corr_hhi, 0.0, 1.0)),  # v2: use HHI instead of top_share
            float(np.clip(disp, 0.0, 1.0)),
            float(np.clip(acc, 0.0, 1.0)),
            float(np.clip(regime_val / 2.0, 0.0, 1.0)),
            # Dims 10-14: Portfolio family features (PRESERVED from v1)
            float(np.clip(cboe_panic_premium / 3.0, -1.0, 1.0)),
            float(np.clip(cboe_term_slope / 3.0, -1.0, 1.0)),
            float(np.clip(cboe_vol_risk_premium_z / 3.0, -1.0, 1.0)),
            float(np.clip(calibration_overall_score, 0.0, 1.0)),
            float(np.clip(online_trust_score, 0.0, 1.0)),
            # Dims 15-24: NEW v2 features
            float(np.clip(tail_risk_es * 10, 0.0, 1.0)),  # Scale ES
            float(np.clip(vol_of_vol * 100, 0.0, 1.0)),   # Scale vol-of-vol
            float(np.clip(liquidity_stress, 0.0, 1.0)),
            float(np.clip(mu_reliability_mean, 0.0, 1.0)),
            float(np.clip(mu_sigma_dispersion, 0.0, 1.0)),
            float(np.clip(sector_imbalance * 5, 0.0, 1.0)),  # Scale sector imbalance
            float(np.clip(recent_hit_rate, 0.0, 1.0)),
            float(np.clip(dd_velocity * 10 + 0.5, 0.0, 1.0)),  # Center and scale
            float(np.clip(regime_trans_prob, 0.0, 1.0)),
            float(np.clip(top_eigen_share, 0.0, 1.0)),  # FIX Gap #7: Top eigenvalue share (distinct from corr_hhi)
        ],
        dtype=float,
    )
    return vec


def _build_default_policy_actions(
    *,
    base_threshold: float,
    regime_mult_bull: float,
    regime_mult_bear: float,
    regime_mult_crisis: float,
    target_vol: float,
    turnover_cap: float,
    max_gross: float,
    max_net: float,
    max_name: float,
    weight_smoothing_alpha: float,
    vol_scaler: float,
) -> List[PolicyAction]:
    base = PolicyAction(
        name="base",
        base_threshold=float(base_threshold),
        regime_mult_bull=float(regime_mult_bull),
        regime_mult_bear=float(regime_mult_bear),
        regime_mult_crisis=float(regime_mult_crisis),
        target_vol=float(target_vol),
        turnover_cap=float(turnover_cap),
        max_gross=float(max_gross),
        max_net=float(max_net),
        max_name=float(max_name),
        weight_smoothing_alpha=float(weight_smoothing_alpha),
        vol_scaler=float(vol_scaler),
    )
    # Conservative / aggressive variants.
    conservative = PolicyAction(
        name="conservative",
        base_threshold=float(base_threshold) * 1.2,
        regime_mult_bull=float(regime_mult_bull) * 1.1,
        regime_mult_bear=float(regime_mult_bear) * 1.2,
        regime_mult_crisis=float(regime_mult_crisis) * 1.2,
        target_vol=float(target_vol) * 0.8,
        turnover_cap=float(turnover_cap) * 0.8,
        max_gross=float(max_gross) * 0.8,
        max_net=float(max_net) * 0.8,
        max_name=float(max_name) * 0.8,
        weight_smoothing_alpha=float(min(0.9, max(weight_smoothing_alpha, 0.1))),
        vol_scaler=float(vol_scaler) * 0.9,
        quantile_blend_weight=0.1,  # Slightly trust quantile in conservative mode
    )
    aggressive = PolicyAction(
        name="aggressive",
        base_threshold=float(base_threshold) * 0.85,
        regime_mult_bull=float(regime_mult_bull) * 0.9,
        regime_mult_bear=float(regime_mult_bear) * 0.9,
        regime_mult_crisis=float(regime_mult_crisis) * 0.9,
        target_vol=float(target_vol) * 1.15,
        turnover_cap=float(turnover_cap) * 1.2,
        max_gross=float(max_gross) * 1.15,
        max_net=float(max_net) * 1.15,
        max_name=float(max_name) * 1.1,
        weight_smoothing_alpha=float(max(0.0, weight_smoothing_alpha * 0.7)),
        vol_scaler=float(vol_scaler) * 1.05,
        quantile_blend_weight=0.0,  # Pure Mamba in aggressive mode
    )
    low_turn = PolicyAction(
        name="low_turnover",
        base_threshold=float(base_threshold) * 1.05,
        regime_mult_bull=float(regime_mult_bull),
        regime_mult_bear=float(regime_mult_bear),
        regime_mult_crisis=float(regime_mult_crisis),
        target_vol=float(target_vol) * 0.95,
        turnover_cap=float(turnover_cap) * 0.5 if float(turnover_cap) > 0 else 0.0,
        max_gross=float(max_gross),
        max_net=float(max_net),
        max_name=float(max_name),
        weight_smoothing_alpha=float(min(0.9, max(weight_smoothing_alpha, 0.2))),
        vol_scaler=float(vol_scaler),
        quantile_blend_weight=0.15,  # Blend in quantile for stability
    )
    # Quantile-focused variant: trust quantile forecast more
    quantile_trust = PolicyAction(
        name="quantile_trust",
        base_threshold=float(base_threshold),
        regime_mult_bull=float(regime_mult_bull),
        regime_mult_bear=float(regime_mult_bear),
        regime_mult_crisis=float(regime_mult_crisis),
        target_vol=float(target_vol),
        turnover_cap=float(turnover_cap),
        max_gross=float(max_gross),
        max_net=float(max_net),
        max_name=float(max_name),
        weight_smoothing_alpha=float(weight_smoothing_alpha),
        vol_scaler=float(vol_scaler),
        quantile_blend_weight=0.3,  # Significant quantile blending
    )
    return [base, conservative, aggressive, low_turn, quantile_trust]


# ---------------------------------------------------------------------------
# Structured Event Bus (Workstream: Telemetry)
# ---------------------------------------------------------------------------

class EventSeverity:
    """Event severity levels for structured telemetry."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCode:
    """Machine-readable event codes for dashboard aggregation."""
    # Cooldown/Gating
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    UNIVERSE_CHANGED = "UNIVERSE_CHANGED"
    REGISTRY_GATE = "REGISTRY_GATE"
    
    # Calibration
    MAMBA_CALIB_UPDATED = "MAMBA_CALIB_UPDATED"
    
    # Signal Processing
    SIGMA_CLIPPED = "SIGMA_CLIPPED"
    HYGIENE_VETO = "HYGIENE_VETO"
    RISK_SCALE_APPLIED = "RISK_SCALE_APPLIED"
    REGIME_MULT_APPLIED = "REGIME_MULT_APPLIED"
    SPLIT_STRESS_APPLIED = "SPLIT_STRESS_APPLIED"
    EVENT_RISK_APPLIED = "EVENT_RISK_APPLIED"
    LINEAR_BLEND_APPLIED = "LINEAR_BLEND_APPLIED"
    QUANTILE_BLEND_APPLIED = "QUANTILE_BLEND_APPLIED"
    
    # Learning Governance
    LEARNING_FROZEN = "LEARNING_FROZEN"
    LEARNING_GOV_DECISION = "LEARNING_GOV_DECISION"
    LEARNING_GOV_LOCKED = "LEARNING_GOV_LOCKED"
    LEARNING_GOV_UNLOCKED = "LEARNING_GOV_UNLOCKED"
    
    # Kill Switches
    NAN_INF_KILLSWITCH = "NAN_INF_KILLSWITCH"
    DD_THROTTLE = "DD_THROTTLE"
    DD_KILL = "DD_KILL"
    VOL_THROTTLE = "VOL_THROTTLE"
    VOL_KILL = "VOL_KILL"
    TURNOVER_KILL = "TURNOVER_KILL"
    
    # Risk Latch (Unified state machine)
    RISK_LATCH_THROTTLE = "RISK_LATCH_THROTTLE"
    RISK_LATCH_FLATTEN = "RISK_LATCH_FLATTEN"
    RISK_LATCH_SAFE_FALLBACK = "RISK_LATCH_SAFE_FALLBACK"
    RISK_LATCH_EMERGENCY = "RISK_LATCH_EMERGENCY"
    
    # Optimizer
    OPTIMIZER_FALLBACK = "OPTIMIZER_FALLBACK"
    OPTIMIZER_INFEASIBLE = "OPTIMIZER_INFEASIBLE"
    OPTIMIZER_SUCCESS = "OPTIMIZER_SUCCESS"
    
    # Policy
    POLICY_ACTION = "POLICY_ACTION"
    POLICY_REWARD = "POLICY_REWARD"
    
    # Execution
    PARTIAL_FILL = "PARTIAL_FILL"
    BORROW_CONSTRAINT = "BORROW_CONSTRAINT"
    HALTED_ASSET = "HALTED_ASSET"


@dataclass
class StructuredEvent:
    """A single structured event for telemetry.
    
    Machine-readable event that can be aggregated into dashboards.
    """
    date: str
    day_idx: int
    severity: str
    code: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "day_idx": self.day_idx,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Collects structured events during Phase-2 execution.
    
    Provides:
    - Event emission with severity/code/message/payload
    - Event filtering by code or severity
    - Summary generation for dashboards
    - JSON export for downstream analysis
    """
    
    def __init__(self):
        self.events: List[StructuredEvent] = []
        self._enabled = True
    
    def emit(
        self,
        date: str,
        day_idx: int,
        severity: str,
        code: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured event."""
        if not self._enabled:
            return
        event = StructuredEvent(
            date=str(date),
            day_idx=int(day_idx),
            severity=str(severity),
            code=str(code),
            message=str(message),
            payload=dict(payload or {}),
        )
        self.events.append(event)
    
    def filter_by_code(self, code: str) -> List[StructuredEvent]:
        """Get all events with a specific code."""
        return [e for e in self.events if e.code == code]
    
    def filter_by_severity(self, severity: str) -> List[StructuredEvent]:
        """Get all events at or above a severity level."""
        levels = [EventSeverity.DEBUG, EventSeverity.INFO, EventSeverity.WARNING, 
                  EventSeverity.ERROR, EventSeverity.CRITICAL]
        try:
            min_idx = levels.index(severity)
        except ValueError:
            min_idx = 0
        return [e for e in self.events if levels.index(e.severity) >= min_idx]
    
    def count_by_code(self) -> Dict[str, int]:
        """Count events by code for summary."""
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e.code] = counts.get(e.code, 0) + 1
        return counts
    
    def generate_summary(self) -> str:
        """Generate a human-readable summary for dashboards."""
        counts = self.count_by_code()
        if not counts:
            return "No events recorded."
        
        lines = ["=== Event Summary ==="]
        
        # Group by category
        kill_events = ["DD_THROTTLE", "DD_KILL", "VOL_THROTTLE", "VOL_KILL", 
                       "TURNOVER_KILL", "NAN_INF_KILLSWITCH"]
        signal_events = ["HYGIENE_VETO", "SIGMA_CLIPPED", "RISK_SCALE_APPLIED",
                        "REGIME_MULT_APPLIED", "SPLIT_STRESS_APPLIED"]
        optimizer_events = ["OPTIMIZER_FALLBACK", "OPTIMIZER_INFEASIBLE", "OPTIMIZER_SUCCESS"]
        
        # Kill switches
        kill_counts = {k: v for k, v in counts.items() if k in kill_events}
        if kill_counts:
            lines.append("\n🚨 Kill Switches / Throttles:")
            for code, count in sorted(kill_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {code}: {count} events")
        
        # Signal processing
        sig_counts = {k: v for k, v in counts.items() if k in signal_events}
        if sig_counts:
            lines.append("\n⚡ Signal Processing:")
            for code, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {code}: {count} events")
        
        # Optimizer
        opt_counts = {k: v for k, v in counts.items() if k in optimizer_events}
        if opt_counts:
            lines.append("\n🔧 Optimizer:")
            for code, count in sorted(opt_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {code}: {count} events")
        
        # Other
        other_counts = {k: v for k, v in counts.items() 
                       if k not in kill_events + signal_events + optimizer_events}
        if other_counts:
            lines.append("\n📊 Other:")
            for code, count in sorted(other_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {code}: {count} events")
        
        return "\n".join(lines)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Export all events as list of dicts."""
        return [e.to_dict() for e in self.events]
    
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()


# ---------------------------------------------------------------------------
# Per-Day Trace Instrumentation (Workstream: Debuggability)
# ---------------------------------------------------------------------------

@dataclass
class DailyTracePayload:
    """Complete per-day trace capturing the full signal chain.
    
    Captures: mu/sigma → sigma_exec → z_raw → z_overlays → z_blend → z_thr 
              → w_target → w_safety → exec_w → pnl_gross → costs → pnl_net → reward
    
    This trace makes debugging 10x faster by capturing boundary artifacts.
    """
    # ─── Identifiers ───
    date: str
    day_idx: int
    horizon: int
    
    # ─── Masks (summary) ───
    n_eligible: int = 0
    n_universe: int = 0
    n_total: int = 0
    dropped_syms: List[str] = field(default_factory=list)  # Top dropped from prev day
    added_syms: List[str] = field(default_factory=list)    # Top added from prev day
    
    # ─── Realized Returns ───
    r_vec_mean: float = 0.0
    r_vec_std: float = 0.0
    r_vec_min: float = 0.0
    r_vec_max: float = 0.0
    
    # ─── Model Outputs ───
    mu_vec_mean: float = 0.0
    mu_vec_std: float = 0.0
    mu_vec_min: float = 0.0
    mu_vec_max: float = 0.0
    mu_top_k: List[Tuple[str, float]] = field(default_factory=list)  # Top 5 by |mu|
    
    sigma_raw_mean: float = 0.0
    sigma_raw_std: float = 0.0
    sigma_raw_min: float = 0.0
    sigma_raw_max: float = 0.0
    
    sigma_exec_mean: float = 0.0
    sigma_exec_std: float = 0.0
    sigma_exec_min: float = 0.0
    sigma_exec_max: float = 0.0
    sigma_exec_n_clipped: int = 0
    
    # ─── Signal Chain ───
    z_raw_mean: float = 0.0
    z_raw_std: float = 0.0
    z_raw_min: float = 0.0
    z_raw_max: float = 0.0
    
    z_after_role_overlays_mean: float = 0.0
    z_after_role_overlays_std: float = 0.0
    
    z_after_quantile_blend_mean: float = 0.0
    z_after_quantile_blend_std: float = 0.0
    quantile_blend_weight: float = 0.0
    
    z_thr_mean: float = 0.0
    z_thr_std: float = 0.0
    z_thr_n_zeroed: int = 0
    
    # ─── Policy State ───
    policy_state_vec: Optional[List[float]] = None  # 25-dim state vector
    policy_action_idx: Optional[int] = None
    policy_knobs: Dict[str, float] = field(default_factory=dict)  # All action knobs
    
    # ─── Optimizer ───
    optimizer_path: str = "unknown"  # "robust" or "mv_fallback"
    optimizer_status: str = "unknown"
    optimizer_iterations: int = 0
    constraints_used: Dict[str, float] = field(default_factory=dict)
    
    w_target_gross: float = 0.0
    w_target_net: float = 0.0
    w_target_n_nonzero: int = 0
    w_target_max_abs: float = 0.0
    
    # ─── Safety Overlays ───
    w_after_safety_gross: float = 0.0
    w_after_safety_net: float = 0.0
    dd_current: float = 0.0
    rv_current: float = 0.0
    turnover_intent: float = 0.0
    dd_throttle_triggered: bool = False
    vol_throttle_triggered: bool = False
    kill_triggered: bool = False
    flattened: bool = False
    
    # ─── Execution & Accounting ───
    exec_w_gross: float = 0.0
    exec_w_net: float = 0.0
    prev_exec_w_gross: float = 0.0
    prev_exec_w_net: float = 0.0
    turnover_exec: float = 0.0
    
    pnl_gross: float = 0.0
    cost_spread: float = 0.0
    cost_impact: float = 0.0
    cost_borrow: float = 0.0
    cost_total: float = 0.0
    pnl_net: float = 0.0
    
    equity: float = 1.0
    drawdown: float = 0.0
    equity_peak: float = 1.0
    
    # ─── Reward (delayed) ───
    reward: Optional[float] = None
    reward_window_start: Optional[int] = None
    reward_window_end: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, (np.floating, np.integer)):
                val = float(val) if isinstance(val, np.floating) else int(val)
            d[f.name] = val
        return d


def _summarize_vector(v: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Compute summary stats for a vector."""
    if v is None or len(v) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    v_clean = np.asarray(v, dtype=float)
    v_clean = np.where(np.isfinite(v_clean), v_clean, 0.0)
    return {
        "mean": float(np.mean(v_clean)),
        "std": float(np.std(v_clean) + eps),
        "min": float(np.min(v_clean)),
        "max": float(np.max(v_clean)),
    }


def _top_k_by_abs(v: np.ndarray, syms: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """Get top k symbols by absolute value."""
    if v is None or len(v) == 0:
        return []
    v_clean = np.asarray(v, dtype=float)
    v_clean = np.where(np.isfinite(v_clean), v_clean, 0.0)
    idx = np.argsort(-np.abs(v_clean))[:k]
    return [(str(syms[i]), float(v_clean[i])) for i in idx if i < len(syms)]


# ---------------------------------------------------------------------------
# Execution Realism Helpers (Workstream: Backtest Realism)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """Per-day execution context for realistic backtest simulation.
    
    Tracks:
    - ADV per asset (for liquidity participation constraints)
    - Spread per asset (for asset-aware slippage)
    - Borrow status per asset (HTB tiers, availability)
    - Partial fill residuals (carried trades from liquidity constraints)
    - Corporate action flags (dividends, halts)
    """
    n_assets: int
    capital_usd: float = 1_000_000.0
    
    # Per-asset ADV in USD (rolling 20d)
    adv_usd: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Per-asset spread in bps (estimated from bid-ask or price impact)
    spread_bps: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Per-asset borrow fee in annual bps (0 = GC, 50-100 = easy, 100-500 = hard, >500 = HTB)
    borrow_fee_bps: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Per-asset borrow availability (0 = unavailable, 1 = available)
    borrow_available: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Partial fill residuals (unfilled trade from previous day)
    partial_fill_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Corporate action flags
    halted: np.ndarray = field(default_factory=lambda: np.array([]))
    ex_dividend: np.ndarray = field(default_factory=lambda: np.array([]))
    dividend_yield: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        n = int(self.n_assets)
        if self.adv_usd.size == 0:
            self.adv_usd = np.full(n, 10_000_000.0)  # Default $10M ADV
        if self.spread_bps.size == 0:
            self.spread_bps = np.full(n, 5.0)  # Default 5 bps spread
        if self.borrow_fee_bps.size == 0:
            self.borrow_fee_bps = np.full(n, 50.0)  # Default 50 bps GC rate
        if self.borrow_available.size == 0:
            self.borrow_available = np.ones(n, dtype=float)
        if self.partial_fill_residual.size == 0:
            self.partial_fill_residual = np.zeros(n, dtype=float)
        if self.halted.size == 0:
            self.halted = np.zeros(n, dtype=float)
        if self.ex_dividend.size == 0:
            self.ex_dividend = np.zeros(n, dtype=float)
        if self.dividend_yield.size == 0:
            self.dividend_yield = np.zeros(n, dtype=float)


def _compute_asset_aware_slippage(
    delta_w: np.ndarray,
    exec_ctx: ExecutionContext,
    *,
    impact_exponent: float = 1.0,
    impact_scale: float = 0.1,
) -> np.ndarray:
    """Compute per-asset slippage based on spread + market impact.
    
    Model:
        slippage[i] = 0.5 * spread_bps[i] * |Δw[i]| 
                    + impact_scale * (trade_$/ADV_$)^impact_exponent
    
    Args:
        delta_w: Weight changes per asset
        exec_ctx: Execution context with ADV, spreads
        impact_exponent: Market impact exponent (typically 1.0-1.7)
        impact_scale: Market impact coefficient
        
    Returns:
        Per-asset slippage in return space (to be summed for total cost)
    """
    n = len(delta_w)
    slippage = np.zeros(n, dtype=float)
    
    capital = float(exec_ctx.capital_usd)
    if capital <= 0:
        return slippage
    
    for i in range(n):
        dw = abs(float(delta_w[i]))
        if dw < 1e-12:
            continue
        
        # Spread component: half-spread * notional (in bps)
        spread_bps = float(exec_ctx.spread_bps[i]) if i < len(exec_ctx.spread_bps) else 5.0
        spread_cost = 0.5 * (spread_bps / 10000.0) * dw
        
        # Impact component: k * (trade_$/ADV_$)^gamma
        adv_usd = float(exec_ctx.adv_usd[i]) if i < len(exec_ctx.adv_usd) else 10_000_000.0
        trade_usd = dw * capital
        if adv_usd > 0:
            participation = trade_usd / adv_usd
            impact_cost = impact_scale * (participation ** impact_exponent) * dw
        else:
            impact_cost = 0.0
        
        slippage[i] = spread_cost + impact_cost
    
    return slippage


def _apply_liquidity_participation_constraint(
    target_w: np.ndarray,
    prev_w: np.ndarray,
    exec_ctx: ExecutionContext,
    *,
    max_participation_rate: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply liquidity participation constraint with partial fills.
    
    Caps daily traded notional per name:
        trade_$ <= participation_rate * ADV_$
    
    If violated: partial fill and carry residual to next day.
    
    Args:
        target_w: Target weights for today
        prev_w: Previous weights (starting point)
        exec_ctx: Execution context with ADV
        max_participation_rate: Max fraction of ADV to trade (default 10%)
        
    Returns:
        Tuple of:
        - executed_w: Actually executed weights (may differ from target)
        - residual: Unfilled portion to carry forward
    """
    n = len(target_w)
    capital = float(exec_ctx.capital_usd)
    
    if capital <= 0 or max_participation_rate <= 0:
        return target_w.copy(), np.zeros(n, dtype=float)
    
    # FIX Gap #4: Do NOT add residual_in to delta_w
    # Rationale: prev_w already reflects the partially executed position from yesterday.
    # The difference (target_w - prev_w) already captures what still needs to be done.
    # Adding residual_in would DOUBLE-COUNT the unfilled portion.
    #
    # Example: Day 1: target=10%, prev=0%, execute 5% → executed=5%, residual=5%
    #          Day 2: target=10%, prev=5% (from day 1 execution)
    #                 delta should be 5% (10% - 5%), NOT 10% (10% - 5% + 5%)
    #
    # The residual tracking is kept for diagnostics/monitoring but not used in computation.
    delta_w = target_w - prev_w
    
    executed_delta = np.zeros(n, dtype=float)
    residual_out = np.zeros(n, dtype=float)
    
    for i in range(n):
        dw = float(delta_w[i])
        if abs(dw) < 1e-12:
            continue
        
        adv_usd = float(exec_ctx.adv_usd[i]) if i < len(exec_ctx.adv_usd) else 10_000_000.0
        max_trade_usd = max_participation_rate * adv_usd
        max_trade_w = max_trade_usd / capital if capital > 0 else 1.0
        
        if abs(dw) <= max_trade_w:
            # Full fill
            executed_delta[i] = dw
        else:
            # Partial fill: cap at max participation
            sign = 1.0 if dw > 0 else -1.0
            executed_delta[i] = sign * max_trade_w
            residual_out[i] = dw - executed_delta[i]
    
    executed_w = prev_w + executed_delta
    return executed_w, residual_out


def _apply_borrow_constraints(
    w: np.ndarray,
    exec_ctx: ExecutionContext,
    *,
    htb_threshold_bps: float = 500.0,
    htb_max_short_weight: float = 0.05,
) -> np.ndarray:
    """Apply borrow constraints for short positions.
    
    - If borrow unavailable: zero short weight
    - If HTB (high borrow fee): cap short weight
    
    Args:
        w: Target weights
        exec_ctx: Execution context with borrow info
        htb_threshold_bps: Borrow fee above which name is HTB
        htb_max_short_weight: Max short weight for HTB names
        
    Returns:
        Adjusted weights with borrow constraints applied
    """
    w_out = w.copy()
    n = len(w)
    
    for i in range(n):
        if w_out[i] >= 0:
            continue  # Long position, no borrow needed
        
        # Check availability
        avail = float(exec_ctx.borrow_available[i]) if i < len(exec_ctx.borrow_available) else 1.0
        if avail < 0.5:
            w_out[i] = 0.0  # Borrow unavailable
            continue
        
        # Check HTB status
        borrow_fee = float(exec_ctx.borrow_fee_bps[i]) if i < len(exec_ctx.borrow_fee_bps) else 50.0
        if borrow_fee >= htb_threshold_bps:
            # HTB: cap short weight
            w_out[i] = max(w_out[i], -htb_max_short_weight)
    
    return w_out


def _compute_per_asset_borrow_cost(
    w: np.ndarray,
    exec_ctx: ExecutionContext,
) -> float:
    """Compute total daily borrow cost across all short positions.
    
    Each short pays: (short_notional * borrow_fee_bps / 10000) / 252
    
    Args:
        w: Current weights
        exec_ctx: Execution context with per-asset borrow fees
        
    Returns:
        Total daily borrow cost as fraction of capital
    """
    capital = float(exec_ctx.capital_usd)
    if capital <= 0:
        return 0.0
    
    total_cost = 0.0
    for i in range(len(w)):
        if w[i] >= 0:
            continue  # Long, no borrow cost
        
        short_w = abs(float(w[i]))
        borrow_bps = float(exec_ctx.borrow_fee_bps[i]) if i < len(exec_ctx.borrow_fee_bps) else 50.0
        daily_rate = (borrow_bps / 10000.0) / 252.0
        total_cost += short_w * daily_rate
    
    return total_cost


def _apply_corporate_action_constraints(
    w: np.ndarray,
    exec_ctx: ExecutionContext,
    *,
    reduce_on_ex_div: bool = True,
    ex_div_reduction_mult: float = 0.5,
) -> np.ndarray:
    """Apply corporate action constraints.
    
    - Halted: zero weight (cannot trade)
    - Ex-dividend: optionally reduce weight (tax drag on shorts)
    
    Args:
        w: Target weights
        exec_ctx: Execution context with corp action flags
        reduce_on_ex_div: Whether to reduce short positions on ex-div
        ex_div_reduction_mult: Multiplier for shorts on ex-div day
        
    Returns:
        Adjusted weights
    """
    w_out = w.copy()
    n = len(w)
    
    for i in range(n):
        # Halted: force flat
        halted = float(exec_ctx.halted[i]) if i < len(exec_ctx.halted) else 0.0
        if halted > 0.5:
            w_out[i] = 0.0
            continue
        
        # Ex-dividend: reduce shorts (dividend payment obligation)
        if reduce_on_ex_div and w_out[i] < 0:
            ex_div = float(exec_ctx.ex_dividend[i]) if i < len(exec_ctx.ex_dividend) else 0.0
            if ex_div > 0.5:
                w_out[i] = w_out[i] * ex_div_reduction_mult
    
    return w_out


def _compute_open_to_open_returns(price_data: pd.DataFrame) -> pd.Series:
    """Compute open-to-open daily returns from a price DataFrame.
    
    For realistic execution: weights decided at close, executed at next open.
    Returns are measured open-to-open to match execution timing.
    """
    if price_data is None or price_data.empty:
        return pd.Series(dtype=float)
    
    df = price_data.copy()
    try:
        df = df.sort_index()
    except Exception:
        pass
    
    # Try to find open price column
    open_col = None
    for col in ("open", "Open", "OPEN", "adjusted_open", "adj_open"):
        if col in df.columns:
            open_col = col
            break
    
    if open_col is None:
        # Fall back to close-to-close
        return _compute_close_to_close_returns(price_data)
    
    px = pd.to_numeric(df[open_col], errors="coerce").astype(float)
    r = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def _compute_close_to_close_returns(price_data: pd.DataFrame) -> pd.Series:
    """Compute close-to-close daily returns from a price DataFrame.

    Uses `_pick_price_col` to select a reasonable close/price column, then computes
    simple returns. Missing/invalid values are treated as 0 return after the pct_change.
    """

    if price_data is None or price_data.empty:
        return pd.Series(dtype=float)

    df = price_data.copy()
    try:
        df = df.sort_index()
    except Exception:
        pass

    col = _pick_price_col(df)
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    px = pd.to_numeric(df[col], errors="coerce").astype(float)
    r = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def _shrink_cov_to_diag(cov: np.ndarray, alpha: float) -> np.ndarray:
    """Simple covariance shrinkage toward the diagonal.

    `alpha=0` keeps the original covariance; `alpha=1` returns diagonal-only.
    """

    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.size == 0:
        return c

    a = float(alpha)
    if not np.isfinite(a):
        a = 0.0
    a = float(np.clip(a, 0.0, 1.0))

    # Replace invalid entries with 0 to avoid NaN poisoning.
    c = np.where(np.isfinite(c), c, 0.0)
    d = np.diag(np.diag(c))
    out = (1.0 - a) * c + a * d
    # Enforce symmetry (best-effort).
    return 0.5 * (out + out.T)


def _compute_adv_usd(price_data: pd.DataFrame, *, window: int) -> pd.Series:
    """Compute rolling ADV in USD if price_data provides enough columns.

    Supported inputs:
    - `dollar_volume` column
    - OR (`volume` and a price column)
    - OR `microstructure_micro_turnover` (computed as `volume * close` in Track-C)
    """

    if price_data is None or price_data.empty:
        return pd.Series(dtype=float)
    df = price_data.copy()
    if "dollar_volume" in df.columns:
        dv = pd.to_numeric(df["dollar_volume"], errors="coerce").astype(float)
        return dv.rolling(window=max(2, int(window)), min_periods=1).mean().replace([np.inf, -np.inf], np.nan)
    # Track-C stores a derived USD turnover proxy under microstructure.*
    for col in ("microstructure_micro_turnover", "micro_turnover"):
        if col in df.columns:
            dv = pd.to_numeric(df[col], errors="coerce").astype(float)
            dv = dv.replace([np.inf, -np.inf], np.nan)
            return dv.rolling(window=max(2, int(window)), min_periods=1).mean()
    if "volume" in df.columns:
        col = _pick_price_col(df)
        px = pd.to_numeric(df[col], errors="coerce").astype(float)
        vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        dv = (px * vol).replace([np.inf, -np.inf], np.nan)
        return dv.rolling(window=max(2, int(window)), min_periods=1).mean()
    return pd.Series(dtype=float)


def _apply_vol_target(
    w: np.ndarray,
    cov: np.ndarray,
    *,
    target_vol_annual: float,
    trading_days: int = 252,
    max_scale: float = 5.0,
) -> np.ndarray:
    target = float(target_vol_annual)
    if not np.isfinite(target) or target <= 0:
        return w
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    var = float(w.T @ cov @ w)
    if not np.isfinite(var) or var <= 0:
        return w
    vol_annual = float(np.sqrt(var) * np.sqrt(float(trading_days)))
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return w
    scale = float(np.clip(target / vol_annual, 0.0, float(max_scale)))
    return w * scale


def _regime_threshold_for_label(
    reg: int,
    base: float,
    bull_mult: float,
    bear_mult: float,
    crisis_mult: float,
) -> float:
    if int(reg) == 2:
        return float(base) * float(crisis_mult)
    if int(reg) == 1:
        return float(base) * float(bear_mult)
    return float(base) * float(bull_mult)


def _predict_mu_sigma_for_day(
    *,
    model: "torch.nn.Module",
    device: "torch.device",
    symbols: Sequence[str],
    features_std_by_symbol: Mapping[str, np.ndarray],
    index_by_symbol: Mapping[str, pd.DatetimeIndex],
    day: pd.Timestamp,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    syms = [str(s).upper() for s in symbols]
    d0 = None
    xs: List[np.ndarray] = []
    for sym in syms:
        feats = np.asarray(features_std_by_symbol[sym], dtype=float)
        idx = pd.DatetimeIndex(index_by_symbol[sym])
        if d0 is None:
            d0 = int(feats.shape[1]) if feats.ndim == 2 else 0
        pos = int(idx.get_indexer([pd.Timestamp(day)])[0])
        if pos < 0:
            x = np.zeros((int(seq_len), int(d0)), dtype=np.float32)
        else:
            start = int(pos) - int(seq_len) + 1
            if start < 0:
                pad = np.zeros((abs(start), int(d0)), dtype=np.float32)
                x = feats[0 : int(pos) + 1]
                x = np.asarray(x, dtype=np.float32)
                x = np.concatenate([pad, x], axis=0)
            else:
                x = feats[int(start) : int(pos) + 1]
                x = np.asarray(x, dtype=np.float32)
            if x.shape[0] != int(seq_len):
                # Defensive: pad/truncate.
                if x.shape[0] < int(seq_len):
                    pad = np.zeros((int(seq_len) - x.shape[0], int(d0)), dtype=np.float32)
                    x = np.concatenate([pad, x], axis=0)
                else:
                    x = x[-int(seq_len) :]
        xs.append(x)

    xb = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    model.eval()
    with torch.no_grad():
        out = model(xb)
        if out.ndim == 2 and out.shape[1] >= 2:
            mu = out[:, 0]
            log_var = out[:, 1]
            var = F.softplus(log_var) + 1e-6
            sigma = torch.sqrt(var)
        else:
            mu = out.reshape(-1)
            sigma = torch.full_like(mu, 0.02)
    mu_np = mu.detach().float().cpu().numpy()
    sigma_np = sigma.detach().float().cpu().numpy()
    sigma_np = np.clip(sigma_np, 1e-4, 10.0)
    return mu_np, sigma_np


def evaluate_phase2_stateful_once(
    *,
    best_trial_json: Optional[Path] = None,
    trial_cfg: Optional[Mapping[str, Any]] = None,
    symbols: Sequence[str],
    horizon: int,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    overrides: Optional[Mapping[str, Any]] = None,
    runtime_overrides: Optional[Mapping[str, Any]] = None,
    portfolio_weights: Optional[Mapping[str, float]] = None,
    aggregation_rule: str = "equal_weight",
    objective_spec: Phase2ObjectiveSpec = Phase2ObjectiveSpec(),
    trial: Optional[Any] = None,
    prune_oos_days: Optional[int] = None,
    prune_update_sessions: int = 21,
    prune_warmup_folds: int = 12,
    safety_prune_after_folds: int = 12,
    safety_prune_sharpe_floor: float = 0.0,
    safety_prune_maxdd_ceiling: float = 0.12,
    deterministic_all: bool = False,
) -> Phase2Result:
    """Phase-2 stateful evaluation with explicit portfolio aggregation.

    Aggregation is explicit via `aggregation_rule` / `portfolio_weights` so Phase-2
    cannot silently change deployment behavior.
    """

    import os
    import random

    import torch
    import time

    def _cuda_mem_snapshot(label: str) -> Optional[Dict[str, float]]:
        if device.type != "cuda":
            return None
        try:
            alloc = float(torch.cuda.memory_allocated(device))
            reserved = float(torch.cuda.memory_reserved(device))
            peak_alloc = float(torch.cuda.max_memory_allocated(device))
            peak_reserved = float(torch.cuda.max_memory_reserved(device))
        except Exception:
            return None
        snap = {
            "alloc_mb": alloc / (1024.0 * 1024.0),
            "reserved_mb": reserved / (1024.0 * 1024.0),
            "peak_alloc_mb": peak_alloc / (1024.0 * 1024.0),
            "peak_reserved_mb": peak_reserved / (1024.0 * 1024.0),
        }
        logger.info(
            "[phase2.cuda_mem] %s alloc=%.0fMiB reserved=%.0fMiB peak_alloc=%.0fMiB peak_reserved=%.0fMiB",
            str(label),
            snap["alloc_mb"],
            snap["reserved_mb"],
            snap["peak_alloc_mb"],
            snap["peak_reserved_mb"],
        )
        return snap

    syms = [str(s).upper() for s in symbols]
    if not syms:
        raise ValueError("symbols must be non-empty")

    rule = str(aggregation_rule).lower().strip()
    if rule not in {"equal_weight", "explicit_weights"}:
        raise ValueError(f"unsupported aggregation_rule: {aggregation_rule}")

    weights = _normalize_portfolio_weights(syms, portfolio_weights if rule == "explicit_weights" else None)

    if trial_cfg is not None:
        cfg = dict(trial_cfg)
    else:
        if best_trial_json is None:
            raise ValueError("best_trial_json is required when trial_cfg is not provided")
        if overrides is None:
            raise ValueError("overrides is required when trial_cfg is not provided")
        base_params = load_stage_b_best_trial_params(best_trial_json)
        cfg = _phase2_apply_refinement_overrides(base_params, overrides)

    if runtime_overrides:
        # Runtime overrides are restricted to Phase2 engine knobs only.
        extra = [k for k in runtime_overrides.keys() if not str(k).startswith("phase2_")]
        if extra:
            raise ValueError(f"runtime_overrides contains non-phase2 keys: {extra}")
        for k, v in dict(runtime_overrides).items():
            cfg[str(k)] = v
    seq_len = int(cfg.get("mamba_seq_len", 128))

    if trial_cfg is not None:
        prepared = _prepare_phase2_for_cfg(
            symbols=syms,
            horizon=int(horizon),
            train_start=train_start,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            cfg=cfg,
        )
    else:
        prepared = _prepare_phase2_once(
            best_trial_json=Path(best_trial_json),
            symbols=syms,
            horizon=int(horizon),
            train_start=train_start,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
        )

    # Audit: record the effective Track-C / model input feature dimension (post Track-A/B/C build).
    if trial is not None:
        try:
            per = {str(s): int(prepared.trackc_aligned_by[s].shape[1]) for s in syms if s in prepared.trackc_aligned_by}
            trial.set_user_attr("phase2_trackc_feature_dim_by_symbol", dict(per))
            dims = {int(v) for v in per.values()}
            if len(dims) == 1:
                d = int(next(iter(dims)))
                trial.set_user_attr("phase2_trackc_feature_dim", int(d))
                logger.info("[phase2.dims] trial=%s trackc_feature_dim=%s", getattr(trial, "number", None), int(d))
            else:
                logger.warning(
                    "[phase2.dims] trial=%s feature_dim mismatch across symbols: %s",
                    getattr(trial, "number", None),
                    per,
                )
        except Exception:
            pass

    # Audit: confirm Mamba sees feature-family weights via post-std scaling.
    # We record stats once per trial to make it unambiguous whether weights were applied.
    if trial is not None:
        try:
            per_sym: Dict[str, Dict[str, float]] = {}
            for sym in syms:
                w = prepared.post_std_feature_weights_by_symbol.get(sym)
                if w is None:
                    continue
                arr = np.asarray(w, dtype=np.float64).reshape(-1)
                if arr.size == 0:
                    continue
                non1 = int(np.sum(np.abs(arr - 1.0) > 1e-6))
                per_sym[str(sym)] = {
                    "n_features": float(arr.size),
                    "n_non1": float(non1),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            trial.set_user_attr("phase2_post_std_feature_weight_stats", per_sym)
        except Exception:
            pass

    # Build (or reuse) windowed samples for this seq_len.
    samples, ts = prepared.cached_samples_by_seq_len.get(seq_len, (None, None))  # type: ignore[assignment]
    if samples is None or ts is None:
        samples, ts = _build_window_samples_for_seq_len(
            symbols=syms,
            seq_len=seq_len,
            prepared=prepared,
        )
        prepared.cached_samples_by_seq_len[seq_len] = (samples, ts)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Determinism controls.
    # - deterministic_rerun: only for replay diagnostics
    # - deterministic_all: force determinism for ALL trials (requested)
    deterministic_rerun = bool(trial is not None and trial.user_attrs.get("rerun_kind") == "top10_replay")
    deterministic_enabled = bool(deterministic_all or deterministic_rerun)
    if trial is not None:
        try:
            trial.set_user_attr("phase2_deterministic", bool(deterministic_enabled))
        except Exception:
            pass

    seed_base = int(os.environ.get("STAGE_B_PHASE2_SEED", "1337"))
    seed = seed_base
    if deterministic_enabled and trial is not None:
        try:
            if deterministic_rerun:
                seed = seed_base + int(trial.user_attrs.get("rerun_of_trial") or trial.number or 0)
            else:
                seed = seed_base + int(trial.number or 0)
        except Exception:
            seed = seed_base

    if deterministic_enabled:
        try:
            # Best-effort: for full determinism this env var is ideally set before
            # importing torch, but setting it here still helps many kernels.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            pass
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        if trial is not None:
            try:
                trial.set_user_attr("phase2_train_seed", int(seed))
            except Exception:
                pass

    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    _cuda_mem_snapshot("trial_start")

    # ------------------------------------------------------------------
    # Walk-forward engine (v2): maturity-gated online updates + covariance sizing.
    # ------------------------------------------------------------------
    engine = str(cfg.get("phase2_engine", "v2")).lower().strip()
    if engine in {"v2", "walkforward", "walkforward_v2"}:
        # Require GPU master store for efficient maturity-gated replay. If CUDA is
        # not available, fall back to CPU sequences (slower but functional).
        update_sessions = int(cfg.get("phase2_update_sessions", prune_update_sessions))
        replay_days = int(cfg.get("phase2_replay_days", 252))
        update_epochs = int(cfg.get("phase2_update_epochs", 2))

        run_mode = str(cfg.get("phase2_run_mode", "refine")).lower().strip()  # {"quick","refine","full"}

        # Diagnostics (primarily for full-mode tuning). Defaults chosen to be safe:
        # - per-update pre/post eval is off by default because it adds compute
        # - fold summaries and sigma health are cheap and on by default
        diag_enabled = bool(cfg.get("phase2_enable_diagnostics", True))
        diag_pre_post = bool(cfg.get("phase2_diag_pre_post_update_eval", False))
        diag_sigma_eps = float(cfg.get("phase2_diag_sigma_eps", 1e-3))
        diag_write_json = bool(cfg.get("phase2_write_diagnostics", False))
        diag_out_dir = Path(str(cfg.get("phase2_diag_out_dir", "artifacts/stage_b_stateful/diagnostics")))

        # Full-mode defaults: insight > speed.
        if run_mode == "full":
            diag_enabled = True
            diag_pre_post = True
            diag_write_json = True

        # Canonical objective scalar (what TPE/Hyperband actually optimize).
        # NOTE: Sharpe already uses net returns (costs applied); turnover penalty is an
        # additional regularizer to reduce pathological churn.
        lambda_turn = float(
            cfg.get(
                "phase2_obj_lambda_turn",
                float(objective_spec.turnover_penalty)
                if float(objective_spec.turnover_penalty) > 0
                else (0.2 if run_mode == "full" else 0.0),
            )
        )
        lambda_flat = float(cfg.get("phase2_obj_lambda_flat", 0.05 if run_mode == "full" else 0.0))

        # Deterministic prune guards (cheap, powerful).
        prune_guards_enabled = bool(cfg.get("phase2_prune_guards", run_mode == "full"))
        prune_sigma_collapse_pct = float(cfg.get("phase2_prune_sigma_collapse_pct", 0.05))
        prune_sigma_collapse_folds = int(cfg.get("phase2_prune_sigma_collapse_folds", 3))
        prune_eff_n_bets_min = float(cfg.get("phase2_prune_eff_n_bets_min", 3.0))
        prune_eff_n_bets_folds = int(cfg.get("phase2_prune_eff_n_bets_folds", 3))
        prune_update_harm_rate = float(cfg.get("phase2_prune_update_harm_rate", 0.60))
        prune_update_harm_after_folds = int(cfg.get("phase2_prune_update_harm_after_folds", 12))
        prune_update_harm_min_updates = int(cfg.get("phase2_prune_update_harm_min_updates", 10))

        sigma_collapse_bad_folds = 0
        eff_n_bets_bad_folds = 0

        cov_lam = float(cfg.get("phase2_cov_ewma_lambda", 0.94))
        shrink_alpha = float(cfg.get("phase2_shrinkage_alpha", 0.10))

        target_vol = float(cfg.get("phase2_target_vol", 0.15))
        max_gross = float(cfg.get("phase2_max_gross", 1.0))
        max_name = float(cfg.get("phase2_max_name", 0.20))
        max_net = float(cfg.get("phase2_max_net", 0.20))

        k_spread = float(cfg.get("phase2_k_spread", 0.0001))
        k_impact = float(cfg.get("phase2_k_impact", 0.0))

        z_clip = float(cfg.get("phase2_z_clip", 8.0))

        # Optional HF-grade overlays (deterministic runtime controls).
        safety_enabled = bool(cfg.get("phase2_safety_overlays", True))
        turnover_cap = float(cfg.get("phase2_turnover_cap", 0.0))
        weight_smoothing_alpha = float(cfg.get("phase2_weight_smoothing_alpha", 0.0))

        dd_throttle_1 = float(cfg.get("phase2_dd_throttle_1", 0.05))
        dd_gross_mult_1 = float(cfg.get("phase2_dd_gross_mult_1", 0.7))
        dd_throttle_2 = float(cfg.get("phase2_dd_throttle_2", 0.10))
        dd_gross_mult_2 = float(cfg.get("phase2_dd_gross_mult_2", 0.4))
        dd_kill = float(cfg.get("phase2_dd_kill", 0.25))

        vol_window = int(cfg.get("phase2_realized_vol_window", 20))
        vol_throttle_mult = float(cfg.get("phase2_vol_throttle_mult", 2.0))
        vol_kill_mult = float(cfg.get("phase2_vol_kill_mult", 3.0))

        kill_on_turnover_gt = float(cfg.get("phase2_kill_on_turnover_gt", 0.0))
        flat_cooldown_sessions = int(cfg.get("phase2_flat_cooldown_sessions", 0))

        # Exposure controls: group caps + beta neutralization.
        group_map_path = cfg.get("phase2_group_map_path")
        group_by_symbol = _load_symbol_group_map(str(group_map_path) if group_map_path else None, symbols=syms)
        max_group_gross = float(cfg.get("phase2_group_max_gross", 0.0))
        max_group_net = float(cfg.get("phase2_group_max_net", 0.0))

        # Beta neutralization: Controlled by phase2_beta_max_abs_exposure cap
        # - If cap = 0: Full beta neutralization (market neutral)
        # - If cap > 0: Neutralize first, then allow up to ±cap beta exposure
        # - If cap not set or negative: No beta constraint
        # NOTE: phase2_beta_neutral flag is DEPRECATED (use beta_cap threshold instead)
        beta_neutral = bool(cfg.get("phase2_beta_neutral", False))  # DEPRECATED
        beta_cap = float(cfg.get("phase2_beta_max_abs_exposure", 0.0))
        beta_lookback = int(cfg.get("phase2_beta_lookback_days", 252))

        # Liquidity controls (optional): requires setting a capital base and having volume/dollar_volume data.
        capital_usd = cfg.get("phase2_capital_usd")
        capital_usd_f = float(capital_usd) if capital_usd is not None else 0.0
        adv_window = int(cfg.get("phase2_adv_window", 20))
        max_adv_frac_name = float(cfg.get("phase2_max_adv_frac_name", 0.0))
        max_turnover_adv_frac = float(cfg.get("phase2_max_turnover_adv_frac", 0.0))
        borrow_fee_bps_annual = float(cfg.get("phase2_borrow_fee_bps_annual", 0.0))

        # ──────────────────────────────────────────────────────────────────────
        # EXECUTION REALISM CONTROLS (Workstream: Backtest Realism)
        # ──────────────────────────────────────────────────────────────────────
        # Execution price timing: 'close_to_close' (default), 'open_to_open', 'vwap'
        exec_price_mode = str(cfg.get("phase2_exec_price_mode", "close_to_close"))
        
        # Liquidity participation constraint: max fraction of ADV per name per day
        # If trade exceeds this, partial fill with residual carried forward
        exec_participation_rate = float(cfg.get("phase2_exec_participation_rate", 0.10))
        exec_enable_partial_fills = bool(cfg.get("phase2_exec_enable_partial_fills", True))
        
        # Asset-aware slippage: per-asset spread + market impact model
        # slippage = 0.5 * spread_bps * |Δw| + impact_scale * (trade_$/ADV_$)^impact_exp
        exec_asset_aware_slippage = bool(cfg.get("phase2_exec_asset_aware_slippage", True))
        exec_impact_exponent = float(cfg.get("phase2_exec_impact_exponent", 1.0))
        exec_impact_scale = float(cfg.get("phase2_exec_impact_scale", 0.1))
        exec_default_spread_bps = float(cfg.get("phase2_exec_default_spread_bps", 5.0))
        
        # Per-name borrow fees (vs flat borrow_fee_bps_annual)
        # HTB threshold: names with borrow fee >= this are capped
        exec_per_name_borrow = bool(cfg.get("phase2_exec_per_name_borrow", True))
        exec_htb_threshold_bps = float(cfg.get("phase2_exec_htb_threshold_bps", 500.0))
        exec_htb_max_short_weight = float(cfg.get("phase2_exec_htb_max_short_weight", 0.05))
        
        # Corporate actions handling
        exec_halted_to_flat = bool(cfg.get("phase2_exec_halted_to_flat", True))
        exec_reduce_short_on_ex_div = bool(cfg.get("phase2_exec_reduce_short_on_ex_div", False))
        exec_ex_div_reduction_mult = float(cfg.get("phase2_exec_ex_div_reduction_mult", 0.5))

        # Workstream-7: Robust Portfolio Optimizer config
        use_robust_optimizer = bool(cfg.get("phase2_use_robust_optimizer", False))
        robust_lambda_var = float(cfg.get("phase2_robust_lambda_var", 1.0))
        robust_lambda_turnover = float(cfg.get("phase2_robust_lambda_turnover", 0.0))
        robust_lambda_tail = float(cfg.get("phase2_robust_lambda_tail", 0.0))
        robust_use_predicted_sigma = bool(cfg.get("phase2_robust_use_predicted_sigma", True))
        robust_predicted_sigma_blend = float(cfg.get("phase2_robust_predicted_sigma_blend", 0.7))
        robust_cov_method = str(cfg.get("phase2_robust_covariance_method", "ewma_shrink"))
        robust_cvar_constraint = bool(cfg.get("phase2_robust_cvar_constraint", False))
        robust_cvar_alpha = float(cfg.get("phase2_robust_cvar_alpha", 0.05))
        robust_cvar_limit = cfg.get("phase2_robust_cvar_limit")
        robust_cvar_limit_f = float(robust_cvar_limit) if robust_cvar_limit is not None else None
        robust_uncertainty_caps = bool(cfg.get("phase2_robust_uncertainty_caps", True))
        robust_mu_sigma_cap = float(cfg.get("phase2_robust_mu_sigma_cap", 3.0))
        robust_reliability_min = float(cfg.get("phase2_robust_reliability_min", 0.3))
        robust_n_factors = int(cfg.get("phase2_robust_n_factors", 5))
        robust_use_graph_shrinkage = bool(cfg.get("phase2_robust_use_graph_shrinkage", False))

        # ──────────────────────────────────────────────────────────────────────
        # LINEAR ALPHA COMBINER: Ridge-regularized alpha signal blending
        # Replaces static quantile blending with learned z_lin = f(z_mamba, quantile_z, ...)
        # ──────────────────────────────────────────────────────────────────────
        linear_model_enabled = bool(cfg.get("phase2_linear_model_enabled", False))
        linear_update_interval = int(cfg.get("phase2_linear_update_interval", 21))
        linear_ridge_lambda = float(cfg.get("phase2_linear_ridge_lambda", 10.0))
        linear_window = int(cfg.get("phase2_linear_window", 126))
        linear_max_window = int(cfg.get("phase2_linear_max_window", 252))
        linear_min_samples = int(cfg.get("phase2_linear_min_samples", 63))
        linear_blend_max = float(cfg.get("phase2_linear_blend_max", 0.5))

        # Initialize robust optimizer if enabled
        robust_optimizer = None
        if use_robust_optimizer:
            try:
                from src.portfolio.robust_optimizer import (
                    RobustPortfolioOptimizer,
                    PredictionBundle,
                    CovarianceModel,
                    PortfolioConstraints,
                    build_sector_adjacency,
                )
                robust_optimizer = RobustPortfolioOptimizer(
                    lambda_var=robust_lambda_var,
                    lambda_turnover=robust_lambda_turnover,
                    lambda_tail=robust_lambda_tail,
                    use_predicted_sigma=robust_use_predicted_sigma,
                    predicted_sigma_blend=robust_predicted_sigma_blend,
                    covariance_method=robust_cov_method,
                    cvar_constraint=robust_cvar_constraint,
                    uncertainty_caps=robust_uncertainty_caps,
                )
                logger.info("[phase2.robust_optimizer] Initialized with lambda_var=%.2f, uncertainty_caps=%s", 
                           robust_lambda_var, robust_uncertainty_caps)
            except Exception as e:
                logger.warning("[phase2.robust_optimizer] Failed to initialize: %s; falling back to standard", e)
                robust_optimizer = None
                use_robust_optimizer = False

        # Initialize linear alpha combiner if enabled
        linear_state: Optional[LinearCombinerState] = None
        if linear_model_enabled:
            try:
                # Try to load existing state
                saved_state = load_linear_combiner_state(
                    horizon=int(horizon),
                    symbols=syms,
                )
                if saved_state is not None:
                    linear_state = LinearCombinerState.from_dict(saved_state)
                    logger.info(
                        "[phase2.linear] Loaded saved state: n_updates=%d, is_ready=%s",
                        linear_state._n_updates, linear_state.is_ready()
                    )
                else:
                    # Create fresh state
                    linear_state = create_linear_combiner(
                        horizon=int(horizon),
                        update_interval=linear_update_interval,
                        ridge_lambda=linear_ridge_lambda,
                        window=linear_window,
                        max_window=linear_max_window,
                        min_samples=linear_min_samples,
                        symbols=syms,
                    )
                    logger.info(
                        "[phase2.linear] Created new combiner: lambda=%.1f, window=%d, min_samples=%d",
                        linear_ridge_lambda, linear_window, linear_min_samples
                    )
            except Exception as e:
                logger.warning("[phase2.linear] Failed to initialize: %s; disabled", e)
                linear_state = None
                linear_model_enabled = False

        # Regime-aware thresholding knobs (re-using Stage-B convention).
        thr_base = float(cfg.get("regime_threshold_base", cfg.get("threshold", 0.10)))
        thr_bull = float(cfg.get("regime_threshold_bull_mult", 0.8))
        thr_bear = float(cfg.get("regime_threshold_bear_mult", 1.5))
        thr_crisis = float(cfg.get("regime_threshold_crisis_mult", 3.0))

        # Optional role-aware overlays (portfolio parquet). Fail open if missing.
        role_aware_enabled = bool(cfg.get("phase2_role_aware_enabled", True))
        role_ctx = None
        if role_aware_enabled:
            try:
                from src.portfolio.role_aware_context import RoleAwareContext

                # Default to cache/merged which contains per-symbol subdirectories
                # (e.g., cache/merged/AAPL/AAPL_h63_merged_portfolio.parquet)
                role_parquet_dir = Path(str(cfg.get("phase2_role_aware_parquet_dir", "cache/merged")))
                role_registry_path = cfg.get("phase2_role_aware_registry_path")
                role_ctx = RoleAwareContext(
                    symbols=syms,
                    horizon=int(horizon),
                    parquet_dir=role_parquet_dir,
                    strict=bool(cfg.get("phase2_role_aware_strict", False)),
                    registry_path=Path(str(role_registry_path)) if role_registry_path else None,
                    data_source=str(cfg.get("phase2_role_aware_data_source", "portfolio")),
                )
            except Exception as e:
                logger.warning("[phase2.role_aware] failed to initialize: %s", e)
                role_ctx = None

        # Build per-symbol daily returns aligned to union OOS index.
        union_oos_index = pd.DatetimeIndex(prepared.union_oos_index).sort_values()
        returns_cols: List[pd.Series] = []
        regimes_by_sym: Dict[str, pd.Series] = {}
        adv_usd_by_sym: Dict[str, pd.Series] = {}
        from src.stage_b.backtest import detect_regime

        for sym in syms:
            pipe = prepared.pipelines_by[sym]
            price_data = pipe._get_price_data_for_horizon(int(horizon))
            r = _compute_close_to_close_returns(price_data)
            r = r.reindex(union_oos_index).fillna(0.0).rename(sym)
            returns_cols.append(r)
            try:
                regimes_by_sym[sym] = detect_regime(r)
            except Exception:
                regimes_by_sym[sym] = pd.Series(0, index=r.index, dtype=int)

            # Optional ADV series (USD). Only used if capital_usd and ADV constraints are enabled.
            # We compute it here to re-use the same price_data we already loaded.
            if capital_usd_f > 0 and (
                (np.isfinite(max_adv_frac_name) and max_adv_frac_name > 0)
                or (np.isfinite(max_turnover_adv_frac) and max_turnover_adv_frac > 0)
            ):
                try:
                    adv = _compute_adv_usd(price_data, window=int(adv_window)).reindex(union_oos_index).ffill()
                except Exception:
                    adv = pd.Series(dtype=float)
                adv_usd_by_sym[str(sym).upper()] = adv

        returns_df = pd.concat(returns_cols, axis=1).sort_index().fillna(0.0)
        _assert_finite_np(
            name="phase2.v2.returns_df",
            symbol="PORTFOLIO",
            values=returns_df.to_numpy(dtype=float),
            index=pd.DatetimeIndex(returns_df.index),
        )

        # Forward returns (label) matrix for calibration features.
        label_col = _phase2_label_column(_phase2_label_id_from_cfg(cfg))
        fwd_ret_mat = np.full((len(union_oos_index), len(syms)), np.nan, dtype=float)
        for j, sym in enumerate(syms):
            try:
                lbl = prepared.labels_by[sym][label_col].reindex(union_oos_index)
                fwd_ret_mat[:, j] = pd.to_numeric(lbl, errors="coerce").to_numpy(dtype=float)
            except Exception:
                pass

        # Market regime proxy for policy state.
        try:
            market_returns = returns_df.mean(axis=1)
            market_regime = detect_regime(market_returns)
        except Exception:
            market_regime = pd.Series(0, index=returns_df.index, dtype=int)

        # Optional policy controller (contextual bandit over knobs).
        policy_enabled = bool(cfg.get("phase2_policy_controller_enabled", True))
        policy = None
        policy_action_history: List[Optional[int]] = [None for _ in range(len(union_oos_index))]
        policy_state_history: List[Optional[np.ndarray]] = [None for _ in range(len(union_oos_index))]
        policy_reward_window = int(cfg.get("phase2_policy_reward_window", 10))
        policy_reward_lambda_turn = float(cfg.get("phase2_policy_reward_lambda_turn", 0.2))
        policy_reward_lambda_dd = float(cfg.get("phase2_policy_reward_lambda_dd", 0.5))
        # v2 exploration annealing
        policy_exploration_half_life = int(cfg.get("phase2_policy_exploration_half_life", 63))
        policy_exploration_floor = float(cfg.get("phase2_policy_exploration_floor", 0.15))
        
        if policy_enabled:
            try:
                # Use v2 action menu (7 actions with expanded knobs)
                policy_actions = build_default_policy_actions_v2(
                    base_threshold=thr_base,
                    regime_mult_bull=thr_bull,
                    regime_mult_bear=thr_bear,
                    regime_mult_crisis=thr_crisis,
                    target_vol=target_vol,
                    turnover_cap=turnover_cap,
                    max_gross=max_gross,
                    max_net=max_net,
                    max_name=max_name,
                )
                policy = PolicyController(
                    actions=policy_actions,
                    feature_dim=POLICY_STATE_DIM_V2,  # v2: 25 dims
                    method=str(cfg.get("phase2_policy_controller_method", "lin_ts")),
                    seed=int(cfg.get("phase2_policy_controller_seed", 1337)),
                    ts_prior_var=float(cfg.get("phase2_policy_ts_prior_var", 1.0)),
                    ts_noise_var=float(cfg.get("phase2_policy_ts_noise_var", 1.0)),
                    ewa_eta=float(cfg.get("phase2_policy_ewa_eta", 0.25)),
                    ewa_temperature=float(cfg.get("phase2_policy_ewa_temperature", 1.0)),
                    warmup_steps=int(cfg.get("phase2_policy_warmup_days", 20)),
                    # v2: Exploration annealing
                    exploration_half_life=policy_exploration_half_life,
                    exploration_floor=policy_exploration_floor,
                )
            except Exception as e:
                logger.warning("[phase2.policy] failed to initialize: %s", e)
                policy = None

        # Online update schedule (every U sessions, starting at the first OOS day).
        step = max(1, int(update_sessions))
        update_positions = set(range(0, len(union_oos_index), step))

        # Pruning granularity: fold = one update boundary (U sessions). We report
        # matured-prefix portfolio metrics at each fold end.
        fold_step_sessions = max(1, int(prune_update_sessions))
        fold_end_positions: List[int] = []
        if trial is not None and fold_step_sessions > 0:
            fold_end_positions = list(range(fold_step_sessions, len(union_oos_index) + 1, fold_step_sessions))
            if not fold_end_positions or fold_end_positions[-1] != len(union_oos_index):
                fold_end_positions.append(len(union_oos_index))
        fold_end_set = set(int(x) for x in fold_end_positions)
        allow_prune = bool(trial is not None and getattr(trial, "user_attrs", {}).get("rerun_kind") != "top10_replay")

        # Initialize covariance (diagonal) and state.
        n_assets = len(syms)
        cov = np.eye(n_assets, dtype=float) * 1e-4
        prev_w = np.zeros(n_assets, dtype=float)

        # Execution realism: optional additional trade-delay beyond the existing 1-session delay.
        # Default is 1 which matches the current convention:
        # - PnL uses prev weights
        # - today we compute next weights
        trade_delay_sessions = int(cfg.get("phase2_trade_delay_sessions", 1))
        trade_delay_sessions = int(max(1, trade_delay_sessions))
        exec_w = prev_w.copy()
        if trade_delay_sessions > 1:
            from collections import deque

            _queue = deque([np.zeros(n_assets, dtype=float) for _ in range(trade_delay_sessions - 1)])

        # Safety state.
        equity_peak = 1.0
        flat_until_idx = -1

        # Track time series outputs.
        mu_mat = np.zeros((len(union_oos_index), n_assets), dtype=float)
        sigma_mat = np.zeros((len(union_oos_index), n_assets), dtype=float)
        w_mat = np.zeros((len(union_oos_index), n_assets), dtype=float)
        
        # DIAGNOSTIC FIX: Track BOTH intent and executed turnover separately
        # - turnover_intent: from optimizer target (w - prev_w), used for policy enforcement
        # - turnover_exec: from execution queue (exec_w - prev_exec_w), used for cost computation
        # This fixes the blind spot where costs were attributed to intent but PnL came from exec
        turnover_intent = np.zeros(len(union_oos_index), dtype=float)
        turnover_exec = np.zeros(len(union_oos_index), dtype=float)
        turnover = np.zeros(len(union_oos_index), dtype=float)  # Legacy: points to turnover_exec for backwards compat
        
        costs = np.zeros(len(union_oos_index), dtype=float)
        net_ret = np.zeros(len(union_oos_index), dtype=float)
        
        # Track prev_exec_w for executed turnover calculation
        prev_exec_w = np.zeros(n_assets, dtype=float)

        # ─────────────────────────────────────────────────────────────────────
        # Mamba-based calibration tracker (AUTHORITATIVE for learning gates).
        # Tracks Mamba (μ, σ²) predictions vs realized returns.
        # Replaces quantile-based calibration as the primary learning authority.
        #
        # GOVERNANCE PERSISTENCE:
        # - State key = (model_id, horizon, symbol_hash)
        # - Persists to artifacts/governance/mamba_calibration/
        # - If horizon changes → key mismatch → clean initialization
        # ─────────────────────────────────────────────────────────────────────
        
        # Build governance state key (includes horizon for leakage prevention)
        _gov_model_id = "phase2_stateful"
        if best_trial_json is not None:
            _gov_model_id = f"phase2_{Path(best_trial_json).stem}"
        elif trial_cfg is not None:
            # Hash the trial config for uniqueness
            import hashlib as _hlib
            _cfg_hash = _hlib.sha256(str(sorted(trial_cfg.items())).encode()).hexdigest()[:8]
            _gov_model_id = f"phase2_cfg_{_cfg_hash}"
        
        # For multi-symbol portfolio, use hash of symbol set as "symbol" in key
        _sym_hash = hashlib.sha256(",".join(sorted(syms)).encode()).hexdigest()[:8]
        governance_state_key = make_state_key(
            model_id=_gov_model_id,
            horizon=int(horizon),
            symbol=f"portfolio_{_sym_hash}",
        )
        
        # Try to load existing tracker state
        _saved_calib_state = load_mamba_calibration_state(governance_state_key)
        if _saved_calib_state is not None:
            try:
                mamba_calib_tracker = MambaCalibrationTracker.deserialize(_saved_calib_state)
                logger.info("[phase2.governance] ✅ Loaded Mamba calibration state: %s", governance_state_key)
            except Exception as e:
                logger.warning("[phase2.governance] Failed to deserialize calibration state: %s", e)
                mamba_calib_tracker = create_mamba_calibration_tracker(
                    horizon=int(horizon),
                    window=MAMBA_CALIB_WINDOW,
                )
        else:
            mamba_calib_tracker = create_mamba_calibration_tracker(
                horizon=int(horizon),
                window=MAMBA_CALIB_WINDOW,
            )
            logger.info("[phase2.governance] New Mamba calibration tracker for: %s", governance_state_key)
        
        mamba_calib_snapshot: Optional[MambaCalibrationSnapshot] = None
        
        # ─────────────────────────────────────────────────────────────────────
        # Learning Governor: Calibration-driven runtime risk control.
        # Three-zone policy:
        #   GREEN (cal >= 0.70): Allow slow updates
        #   YELLOW (0.55 <= cal < 0.70): Allow normal updates
        #   RED (cal < 0.55): Freeze learning
        #
        # Also handles:
        #   - Drift-triggered micro-updates
        #   - Human-in-the-loop auto-revert gate
        # ─────────────────────────────────────────────────────────────────────
        learning_governor = None
        learning_governor_enabled = bool(cfg.get("phase2_learning_governor_enabled", True))
        if learning_governor_enabled:
            try:
                from src.stage_b_stateful.learning_governor import (
                    LearningGovernor,
                    create_learning_governor,
                    LearningDecision,
                )
                
                gov_checkpoint_dir = Path(cfg.get(
                    "phase2_learning_governor_checkpoint_dir",
                    "artifacts/governance/learning_governor"
                ))
                gov_log_path = Path(cfg.get(
                    "phase2_learning_governor_log_path",
                    "artifacts/logs/learning_governor_decisions.jsonl"
                ))
                
                learning_governor = create_learning_governor(
                    checkpoint_dir=gov_checkpoint_dir,
                    decision_log_path=gov_log_path,
                )
                logger.info("[phase2.governance] ✅ Learning Governor initialized")
            except Exception as e:
                logger.warning("[phase2.governance] Failed to initialize Learning Governor: %s", e)
                learning_governor = None
        equity = np.ones(len(union_oos_index), dtype=float)

        # Daily diagnostics (cheap to compute).
        active_frac = np.zeros(len(union_oos_index), dtype=float)
        flat_flag = np.zeros(len(union_oos_index), dtype=float)
        gross_exposure = np.zeros(len(union_oos_index), dtype=float)
        net_exposure = np.zeros(len(union_oos_index), dtype=float)
        eff_n_bets = np.zeros(len(union_oos_index), dtype=float)
        sigma_med_daily = np.zeros(len(union_oos_index), dtype=float)
        sigma_p90_daily = np.zeros(len(union_oos_index), dtype=float)
        sigma_collapse_daily = np.zeros(len(union_oos_index), dtype=float)
        corr_abs_mu_sigma_daily = np.zeros(len(union_oos_index), dtype=float)

        # HF-grade overlay telemetry (risk-committee style).
        overlay_dd = np.zeros(len(union_oos_index), dtype=float)
        overlay_rv = np.zeros(len(union_oos_index), dtype=float)
        overlay_dd_gross_scale = np.ones(len(union_oos_index), dtype=float)
        overlay_vol_scale = np.ones(len(union_oos_index), dtype=float)
        overlay_flattened = np.zeros(len(union_oos_index), dtype=float)
        overlay_beta_neutralized = np.zeros(len(union_oos_index), dtype=float)

        # Execution sigma discipline (sizing-only):
        # sigma_exec(t) = clamp( EMA(span=21) of sigma_train(t), rolling_p10(252), rolling_p90(252) )
        # NOTE: sigma_train remains used for training/loss/diagnostics.
        from collections import deque

        _sigma_exec_span = 21
        _sigma_exec_alpha = 2.0 / (float(_sigma_exec_span) + 1.0)
        _sigma_exec_roll = 252
        _sigma_exec_eps = 1e-4
        sigma_ema = np.ones(n_assets, dtype=float) * 0.02
        sigma_ema_hist = [deque(maxlen=int(_sigma_exec_roll)) for _ in range(n_assets)]

        # ─────────────────────────────────────────────────────────────────────
        # EXECUTION REALISM: Initialize ExecutionContext for daily updates
        # ─────────────────────────────────────────────────────────────────────
        exec_ctx = ExecutionContext(
            n_assets=n_assets,
            capital_usd=capital_usd_f if capital_usd_f > 0 else 1_000_000.0,
        )
        # Default spread and borrow fees (can be updated per-day from data)
        exec_ctx.spread_bps = np.full(n_assets, exec_default_spread_bps, dtype=float)
        exec_ctx.borrow_fee_bps = np.full(n_assets, borrow_fee_bps_annual, dtype=float)
        
        # Compute open-to-open returns if needed
        open_returns_df: Optional[pd.DataFrame] = None
        if exec_price_mode == "open_to_open":
            allow_exec_fallback = bool(cfg.get("phase2_allow_exec_price_fallback", False))
            open_returns_cols: Dict[str, pd.Series] = {}
            for sym in syms:
                if sym not in prepared.pipelines_by:
                    msg = f"[phase2.exec_realism] Missing pipeline for {sym}; cannot build open-to-open returns"
                    if allow_exec_fallback:
                        logger.warning("%s", msg)
                        open_returns_cols = {}
                        break
                    raise ValueError(msg)
                pipe = prepared.pipelines_by[sym]
                try:
                    price_data = pipe._get_price_data_for_horizon(
                        sym, int(horizon), union_oos_index[0], union_oos_index[-1]
                    )
                    open_returns_cols[sym] = _compute_open_to_open_returns(price_data)
                except Exception as e:
                    msg = f"[phase2.exec_realism] Failed open-to-open for {sym}: {e}"
                    if allow_exec_fallback:
                        logger.warning("%s", msg)
                        open_returns_cols = {}
                        break
                    raise

            if open_returns_cols:
                open_returns_df = pd.concat(open_returns_cols, axis=1).reindex(union_oos_index).fillna(0.0)
                logger.info("[phase2.exec_realism] Using open-to-open returns for execution pricing")
            else:
                open_returns_df = None
                if allow_exec_fallback:
                    logger.warning("[phase2.exec_realism] open_to_open failed; falling back to close_to_close")
                    exec_price_mode = "close_to_close"

        # ─────────────────────────────────────────────────────────────────────
        # DAILY TRACE INSTRUMENTATION: Initialize trace collection
        # ─────────────────────────────────────────────────────────────────────
        trace_enabled = bool(cfg.get("phase2_trace_enabled", True))
        trace_persist = bool(cfg.get("phase2_trace_persist", True))
        trace_persist_path = cfg.get("phase2_trace_persist_path")
        daily_traces: List[DailyTracePayload] = []
        # Note: prev_exec_w is now initialized earlier in the main tracking section
        _prev_eligible_set: Optional[Set[str]] = None  # For dropped/added tracking

        # ─────────────────────────────────────────────────────────────────────
        # STRUCTURED EVENT BUS: Initialize telemetry collection
        # ─────────────────────────────────────────────────────────────────────
        events_enabled = bool(cfg.get("phase2_events_enabled", True))
        events_persist = bool(cfg.get("phase2_events_persist", True))
        event_bus = EventBus()
        event_bus._enabled = events_enabled

        # ─────────────────────────────────────────────────────────────────────
        # GAP #8: PER-ASSET CALIBRATION TRACKING
        # Track per-symbol (mu, sigma, realized) for post-hoc calibration stats.
        # 
        # USAGE: Reporting/diagnostics ONLY. Not used in risk latch or model structure.
        # Output: Included in per_symbol_metrics for dashboard/analysis.
        # Future: Could be used for per-asset risk scaling or single-stock anomaly detection.
        # ─────────────────────────────────────────────────────────────────────
        per_symbol_calib_enabled = bool(cfg.get("phase2_per_symbol_calib", True))
        # Accumulators: [n_samples, n_assets] - grow dynamically
        _per_sym_mu_acc: List[np.ndarray] = []
        _per_sym_sigma_acc: List[np.ndarray] = []
        _per_sym_realized_acc: List[np.ndarray] = []

        # ─────────────────────────────────────────────────────────────────────
        # Z-EXPLAINER: Track z-signal chain for interpretability
        # Explains z_final (post-threshold) from interpretable features.
        # Uses Rolling Ridge surrogate with weekly refitting and fidelity gating.
        # ─────────────────────────────────────────────────────────────────────
        z_explainer_enabled = bool(cfg.get("phase2_z_explainer", True))
        z_explainer_buffer: Optional["ZExplainerBuffer"] = None
        z_surrogate: Optional["RollingSurrogate"] = None
        z_validation_companion: Optional["ValidationCompanion"] = None  # Step 7
        z_explainer_logs: List[Dict[str, Any]] = []  # Daily explanation logs
        
        if z_explainer_enabled:
            try:
                from src.stage_b_stateful.z_explainer import (
                    ZExplainerBuffer,
                    RollingSurrogate,
                    ValidationCompanion,
                )
                z_explainer_buffer = ZExplainerBuffer()
                z_surrogate = RollingSurrogate(
                    window=int(cfg.get("phase2_z_explainer_window", 126)),
                    max_window=int(cfg.get("phase2_z_explainer_max_window", 252)),
                    refit_interval=int(cfg.get("phase2_z_explainer_refit_interval", 5)),
                    regularization=float(cfg.get("phase2_z_explainer_regularization", 10.0)),
                    min_samples_for_fit=int(cfg.get("phase2_z_explainer_min_samples", 50)),
                    top_n_symbols=int(cfg.get("phase2_z_explainer_top_symbols", 10)),
                    top_k_features=int(cfg.get("phase2_z_explainer_top_features", 5)),
                )
                
                # Step 7: Validation companion (sanity check z_final vs forward returns)
                validation_enabled = bool(cfg.get("phase2_z_explainer_validation", True))
                if validation_enabled:
                    z_validation_companion = ValidationCompanion(
                        window=int(cfg.get("phase2_z_explainer_window", 126)),
                        max_window=int(cfg.get("phase2_z_explainer_max_window", 252)),
                        refit_interval=int(cfg.get("phase2_z_explainer_validation_refit", 10)),
                        regularization=float(cfg.get("phase2_z_explainer_regularization", 10.0)),
                        min_samples=int(cfg.get("phase2_z_explainer_validation_min_samples", 100)),
                    )
                
                logger.info(
                    "[phase2.z_explainer] Enabled: window=%d-%d, refit every %d days, λ=%.2f, validation=%s",
                    z_surrogate.window, z_surrogate.max_window,
                    z_surrogate.refit_interval, z_surrogate.regularization,
                    "ON" if z_validation_companion else "OFF"
                )
            except ImportError as e:
                logger.warning("[phase2.z_explainer] z_explainer module not available: %s", e)
                z_explainer_enabled = False

        # ─────────────────────────────────────────────────────────────────────
        # RISK LATCH: Unified risk-control state machine
        # Consolidates all risk mechanisms (policy, kill-switches, overlays)
        # into a single state machine with clear precedence.
        # ─────────────────────────────────────────────────────────────────────
        risk_latch_enabled = bool(cfg.get("phase2_risk_latch_enabled", True))
        risk_latch: Optional["RiskLatch"] = None
        risk_events_ledger = None  # type: ignore
        
        if risk_latch_enabled:
            try:
                from src.stage_b_stateful.risk_latch import (
                    RiskLatch,
                    RiskLatchMode,
                    create_risk_latch,
                )
                risk_latch = create_risk_latch(cfg, n_assets)
                logger.info(
                    "[phase2.risk_latch] Enabled: DD_kill=%.1f%%, vol_kill=%.1f%%, throttle_min=%.1f%%",
                    risk_latch.thresholds.max_drawdown_kill * 100,
                    risk_latch.thresholds.max_vol_kill * 100,
                    risk_latch.thresholds.throttle_min_scale * 100,
                )
                
                # ─────────────────────────────────────────────────────────────
                # RISK EVENTS LEDGER: Daily diagnostic output
                # Writes JSONL file with all risk events for post-run analysis.
                # ─────────────────────────────────────────────────────────────
                try:
                    from src.stage_b_stateful.risk_events_ledger import (
                        RiskEventsLedger,
                        EventType,
                        ActionTaken,
                        create_ledger,
                    )
                    risk_events_ledger = create_ledger(
                        symbol=symbol_to_run,
                        horizon=horizon,
                        base_dir=str(cfg.get(
                            "phase2_risk_events_dir",
                            "cache/debugging/risk_events",
                        )),
                        run_id=str(cfg.get("phase2_run_id", None)),
                    )
                    logger.info(
                        "[phase2.risk_events_ledger] Enabled: path=%s",
                        risk_events_ledger.path,
                    )
                except ImportError as e:
                    logger.warning(
                        "[phase2.risk_events_ledger] risk_events_ledger not available: %s",
                        e,
                    )
                    risk_events_ledger = None
                    
            except ImportError as e:
                logger.warning("[phase2.risk_latch] risk_latch module not available: %s", e)
                risk_latch_enabled = False
        
        # ─────────────────────────────────────────────────────────────────────
        # INTRADAY MONITOR: Read live monitor snapshots (for live trading)
        # The intraday monitor runs as a separate process and writes a JSON
        # snapshot. We read it at the top of each day to check for alerts.
        # ─────────────────────────────────────────────────────────────────────
        intraday_monitor_enabled = bool(cfg.get("phase2_intraday_monitor_enabled", False))
        intraday_snapshot_path = str(cfg.get(
            "phase2_intraday_snapshot_path",
            "artifacts/intraday_monitor/latest_snapshot.json",
        ))
        intraday_snapshot_max_age = float(cfg.get("phase2_intraday_snapshot_max_age", 300.0))
        
        if intraday_monitor_enabled:
            logger.info(
                "[phase2.intraday_monitor] Enabled: snapshot_path=%s, max_age=%.0fs",
                intraday_snapshot_path,
                intraday_snapshot_max_age,
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # STRESS OVERLAY: On-the-fly scenario stress tests
        # Applied after optimizer, before final safety overlays.
        # ─────────────────────────────────────────────────────────────────────
        stress_overlay_enabled = bool(cfg.get("phase2_stress_enabled", True))
        stress_overlay = None
        
        if stress_overlay_enabled:
            try:
                from src.stage_b_stateful.stress_overlay import (
                    StressTestOverlay,
                    create_stress_overlay,
                )
                stress_overlay = create_stress_overlay(cfg, n_assets)
                logger.info(
                    "[phase2.stress_overlay] Enabled: k_sigma=%.0f, loss_throttle=%.1f%%, loss_flatten=%.1f%%",
                    stress_overlay.config.shock_k_sigma,
                    stress_overlay.config.shock_loss_throttle_pct * 100,
                    stress_overlay.config.shock_loss_flatten_pct * 100,
                )
            except ImportError as e:
                logger.warning("[phase2.stress_overlay] stress_overlay module not available: %s", e)
                stress_overlay_enabled = False

        # Per-update and per-fold structured logs.
        update_logs: List[Dict[str, Any]] = []
        fold_logs: List[Dict[str, Any]] = []

        def _eval_warm_state_on_val(
            seq_data: SequenceData,
            val_idx: np.ndarray,
            state_dict: Dict[str, "torch.Tensor"],
            cfg_eval: Mapping[str, Any],
        ) -> Dict[str, float]:
            """Evaluate warm_state on val subset for pre/post update diagnostics."""
            import torch
            import torch.nn.functional as F
            from src.stage_b.sequence_models import MambaLikeRegressor

            batch_size = int(cfg_eval.get("mamba_batch_size", 32))
            d_model = int(cfg_eval.get("mamba_d_model", 128))
            n_layers = int(cfg_eval.get("mamba_n_layers", 4))
            ssm_dim = int(cfg_eval.get("mamba_ssm_dim", 96))
            expand_factor = float(cfg_eval.get("mamba_expand_factor", 2.0))
            activation = str(cfg_eval.get("mamba_activation", "silu"))
            norm_type = str(cfg_eval.get("mamba_norm_type", "rmsnorm"))
            norm_strategy = str(cfg_eval.get("mamba_norm_strategy", "pre"))
            dropout = float(cfg_eval.get("mamba_dropout", 0.1))
            resid_dropout = float(cfg_eval.get("mamba_resid_dropout", 0.0))
            ssm_dropout = float(cfg_eval.get("mamba_ssm_dropout", 0.0))
            gate_dropout = float(cfg_eval.get("mamba_gate_dropout", 0.0))

            head_type = str(cfg_eval.get("mamba_head_type", "gaussian"))
            head_hidden_dim = int(cfg_eval.get("mamba_head_hidden_dim", 128))
            head_num_layers = int(cfg_eval.get("mamba_head_num_layers", 1))
            head_dropout = float(cfg_eval.get("mamba_head_dropout", 0.0))

            conv_kernel = int(min(max(3, ssm_dim // 8 * 2 + 1), 31))

            model = MambaLikeRegressor(
                input_dim=seq_data.feature_dim,
                d_model=d_model,
                n_layers=n_layers,
                expand_factor=expand_factor,
                conv_kernel=conv_kernel,
                activation=activation,
                norm_type=norm_type,
                norm_strategy=norm_strategy,
                dropout=dropout,
                resid_dropout=resid_dropout,
                ssm_dropout=ssm_dropout,
                gate_dropout=gate_dropout,
                head_type=head_type,
                head_hidden_dim=head_hidden_dim,
                head_num_layers=head_num_layers,
                head_dropout=head_dropout,
            ).to(device)

            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            loader = seq_data.make_loader(val_idx, batch_size=batch_size, shuffle=False, device=device)
            model.eval()
            mu_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            total = torch.tensor(0.0, device=device)
            n_batches = 0
            with torch.no_grad():
                for xb, yb in loader:
                    if xb.device != device:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                    out = model(xb)
                    if out.ndim != 2 or int(out.shape[1]) < 2:
                        # Fall back to mse-like proxy if head is not gaussian (shouldn't happen in v2).
                        mu = out.reshape(-1)
                        loss = torch.mean((yb.reshape(-1) - mu) ** 2)
                    else:
                        mu = out[:, 0]
                        log_var = out[:, 1]
                        var = F.softplus(log_var) + 1e-6
                        loss = 0.5 * (torch.log(var) + (yb.reshape(-1) - mu) ** 2 / var)
                        loss = loss.mean()
                    total = total + loss.detach()
                    n_batches += 1
                    mu_list.append(mu.detach())
                    y_list.append(yb.detach().reshape(-1))

            if n_batches <= 0:
                return {"val_loss": float("nan"), "val_corr": float("nan")}
            mu_all = torch.cat(mu_list, dim=0).float().cpu().numpy()
            y_all = torch.cat(y_list, dim=0).float().cpu().numpy()
            out_loss = float((total / max(1, n_batches)).item())
            out_corr = float(_safe_corr(mu_all, y_all))
            return {"val_loss": out_loss, "val_corr": out_corr}

        # Train/update model with maturity-gated replay.
        model = None
        warm_state = None

        n_oos = len(union_oos_index)
        n_updates_total = len(update_positions)
        updates_done = 0
        _last_progress_log = 0
        logger.info(
            "[phase2.v2] walkforward starting: %d OOS sessions, %d updates (every %d sessions), seq_len=%d, %d symbols",
            n_oos, n_updates_total, step, seq_len, n_assets,
        )

        # ------------------------------------------------------------------
        # Optional runtime universe schedule (piecewise-constant, monthly-ish)
        #
        # Spec:
        # - Training remains pooled across all trial symbols.
        # - Trading weights are gated each day:
        #     w_final[i] = w_raw[i] if i in current_universe else 0
        #   then proceed through the usual constraints/overlays.
        # - current_universe is updated every U=21 sessions (not daily).
        #
        # This uses Track-C (prepared.trackc_aligned_by) and the pre-Phase2
        # rule-based event_score implementation.
        # ------------------------------------------------------------------
        universe_mode = str(cfg.get("phase2_runtime_universe_mode", "full")).lower().strip()
        universe_enabled = universe_mode in {"core_satellite", "full"}

        # Optional authoritative universe registry gating (applies even if runtime universe is disabled).
        _registry_enabled = bool(cfg.get("phase2_universe_registry_enabled", True)) or bool(cfg.get("phase2_universe_registry_path"))
        _registry_strict = bool(cfg.get("phase2_universe_registry_strict", False))
        _registry_df = None
        _eligible_mask: Optional[np.ndarray] = None
        _eligible_prev: Optional[set[str]] = None
        if _registry_enabled:
            try:
                from src.stage_b.universe_registry import load_universe_registry

                reg_path_raw = cfg.get("phase2_universe_registry_path")
                reg_path = Path(str(reg_path_raw)) if reg_path_raw else (Path("data") / "cache" / "universe" / "universe_registry.parquet")
                if _registry_strict and not reg_path.exists():
                    raise ValueError(
                        "phase2_universe_registry_strict enabled but no universe_registry parquet exists; "
                        "run the registry refresh job or set phase2_universe_registry_path"
                    )
                _registry_df = load_universe_registry(reg_path)
            except Exception as e:
                logger.warning("[phase2.universe_registry] failed to load registry: %s", e)
                _registry_df = None

        def _eligible_symbols(asof_ts: pd.Timestamp) -> Optional[set[str]]:
            if _registry_df is None:
                return None
            try:
                from src.stage_b.universe_registry import eligible_symbols_for_date

                return eligible_symbols_for_date(_registry_df, asof_date=asof_ts, symbols=syms)
            except Exception:
                return None

        # Universe selection config/state (best-effort). Kept local to the v2 engine.
        current_universe: Optional[set[str]] = None
        _universe_mask: Optional[np.ndarray] = None
        _universe_state_path: Optional[Path] = None
        _universe_state: dict[str, Any] = {}
        _universe_sessions: Optional[pd.DatetimeIndex] = None
        _universe_cfg = None

        # "full" mode: trade the full eligible universe daily (no core/satellite rotation).
        # "core_satellite" mode: monthly-ish rotation using event score.
        if universe_enabled and universe_mode == "core_satellite":
            from src.stage_b.universe_selector import (
                DEFAULT_CANDIDATE_UNIVERSE,
                DEFAULT_CORE_UNIVERSE,
                UniverseSelectorConfig,
                compute_event_score,
            )

            def _norm_syms(symbols: Any) -> list[str]:
                if symbols is None:
                    return []
                if isinstance(symbols, str):
                    parts = [s.strip().upper() for s in symbols.split(",") if s.strip()]
                elif isinstance(symbols, (list, tuple, set)):
                    parts = [str(s).strip().upper() for s in symbols if str(s).strip()]
                else:
                    parts = [str(symbols).strip().upper()] if str(symbols).strip() else []
                out: list[str] = []
                seen: set[str] = set()
                for s in parts:
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                return out

            def _load_universe_state(path: Path) -> dict[str, Any]:
                if not path.exists():
                    return {}
                try:
                    with open(path, "r") as handle:
                        data = json.load(handle)
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}

            def _save_universe_state(path: Path, state: dict[str, Any]) -> None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    with open(tmp, "w") as handle:
                        json.dump(state, handle, indent=2, sort_keys=True)
                    tmp.replace(path)
                except Exception:
                    return

            def _count_sessions_between(sessions: Optional[pd.DatetimeIndex], start: pd.Timestamp, end: pd.Timestamp) -> Optional[int]:
                if sessions is None or len(sessions) == 0:
                    return None
                s = pd.Timestamp(start).normalize()
                e = pd.Timestamp(end).normalize()
                if e < s:
                    return 0
                mask = (sessions >= s) & (sessions <= e)
                return int(mask.sum())

            # Resolve CORE/candidates from cfg or defaults.
            core_syms = _norm_syms(cfg.get("phase2_runtime_universe_core_symbols")) or list(DEFAULT_CORE_UNIVERSE)
            cand_syms = _norm_syms(cfg.get("phase2_runtime_universe_candidate_symbols")) or list(DEFAULT_CANDIDATE_UNIVERSE)
            # Optional survivorship filter for runtime universe gating.
            _delisted_date_by_symbol: Dict[str, pd.Timestamp] = {}
            _delisting_strict_runtime = bool(cfg.get("phase2_delisting_meta_strict", False))
            try:
                if bool(cfg.get("phase2_delisting_meta_enabled", False)):
                    from src.stage_b.delisting_meta import get_delisted_date_map, load_delisting_registry

                    exchange_code = str(cfg.get("phase2_delisting_meta_exchange", "US") or "US").upper()
                    cache_path = cfg.get("phase2_delisting_meta_cache_path")
                    if _delisting_strict_runtime:
                        implied = Path(str(cache_path)) if cache_path else (Path("data") / "cache" / "eodhd" / f"delisted_companies_{exchange_code}.parquet")
                        if not implied.exists():
                            raise ValueError(
                                "phase2_delisting_meta_strict enabled but no local delisting registry cache exists for runtime universe gating; "
                                "run tools/build_delisting_meta.py or set phase2_delisting_meta_cache_path"
                            )
                    reg = load_delisting_registry(
                        exchange_code=exchange_code,
                        cache_path=str(cache_path) if cache_path else None,
                    )
                    _delisted_date_by_symbol = get_delisted_date_map(reg)
            except Exception:
                _delisted_date_by_symbol = {}

            # Always constrain candidates to the actual trial symbols we have Track-C for.
            available = {str(s).upper() for s in syms}
            core_syms = [s for s in core_syms if s in available]
            cand_syms = [s for s in cand_syms if s in available]
            if not cand_syms:
                cand_syms = list(sorted(available))

            # Build sessions calendar from union_oos_index.
            try:
                _universe_sessions = pd.DatetimeIndex(sorted({pd.Timestamp(x).normalize() for x in union_oos_index}))
            except Exception:
                _universe_sessions = None

            _universe_cfg = UniverseSelectorConfig(
                core_symbols=core_syms,
                candidate_symbols=cand_syms,
                lookback_sessions=int(cfg.get("phase2_runtime_universe_lookback_sessions", 63)),
                min_history_sessions=int(cfg.get("phase2_runtime_universe_min_history_sessions", 63)),
                rebalance_every_sessions=int(cfg.get("phase2_runtime_universe_rebalance_every_sessions", 21)),
                min_stay_sessions=int(cfg.get("phase2_runtime_universe_min_stay_sessions", 21)),
                max_stay_sessions=int(cfg.get("phase2_runtime_universe_max_stay_sessions", 63)),
                decay_fraction=float(cfg.get("phase2_runtime_universe_decay_fraction", 0.5)),
                remove_abs_threshold=float(cfg.get("phase2_runtime_universe_remove_abs_threshold", 0.25)),
                min_adv_usd=(
                    float(cfg.get("phase2_runtime_universe_min_adv_usd"))
                    if cfg.get("phase2_runtime_universe_min_adv_usd") is not None
                    else None
                ),
                peer_snapshot_path=(
                    str(cfg.get("phase2_runtime_universe_peer_snapshot_path")).strip()
                    if cfg.get("phase2_runtime_universe_peer_snapshot_path") is not None
                    else None
                )
                or None,
                peer_require_has_data=bool(cfg.get("phase2_runtime_universe_peer_require_has_data", False)),
                peer_min_sector_peers=int(cfg.get("phase2_runtime_universe_peer_min_sector_peers", 3)),
                peer_min_industry_peers=int(cfg.get("phase2_runtime_universe_peer_min_industry_peers", 3)),
            )

            state_path_raw = cfg.get("phase2_runtime_universe_state_path")
            if state_path_raw:
                try:
                    _universe_state_path = Path(str(state_path_raw))
                except Exception:
                    _universe_state_path = None

            if _universe_state_path is not None:
                _universe_state = _load_universe_state(_universe_state_path)
            else:
                _universe_state = {}

            # Seed current_universe from state (if any).
            sat_prev_obj = _universe_state.get("satellites")
            sat_prev = sat_prev_obj if isinstance(sat_prev_obj, dict) else {}
            prev_sat_syms = [str(k).upper() for k in sat_prev.keys() if isinstance(k, str) and str(k).strip()]
            current_universe = set(_norm_syms([*core_syms, *prev_sat_syms]))
            if not current_universe:
                current_universe = set(core_syms)

            def _rebalance_universe(asof_ts: pd.Timestamp) -> set[str]:
                assert _universe_cfg is not None

                core_pre = _norm_syms(_universe_cfg.core_symbols)
                core = list(core_pre)
                candidates = _norm_syms(_universe_cfg.candidate_symbols)

                # Apply authoritative registry eligibility first.
                elig = _eligible_symbols(asof_ts)
                if elig is not None:
                    core = [s for s in core if s in elig]
                    candidates = [s for s in candidates if s in elig]

                if _delisted_date_by_symbol:
                    core = [s for s in core if _delisted_date_by_symbol.get(s) is None or _delisted_date_by_symbol[s] >= asof_ts]
                    candidates = [
                        s
                        for s in candidates
                        if _delisted_date_by_symbol.get(s) is None or _delisted_date_by_symbol[s] >= asof_ts
                    ]

                # Enforce CORE size bounds (governance).
                # Apply the governance check to the configured CORE size, not the registry-filtered result.
                if len(core_pre) < 15 or len(core_pre) > 25:
                    # If misconfigured, disable the gate rather than hard-failing the run.
                    return set(str(s).upper() for s in syms)

                # Clamp satellite min/max to keep total universe in [30,40].
                sat_min = max(int(_universe_cfg.satellite_min), max(0, 30 - len(core)))
                sat_max = min(int(_universe_cfg.satellite_max), max(0, 40 - len(core)))
                if sat_max < sat_min:
                    sat_max = sat_min

                # Cadence gate (monthly-ish).
                last_rebalance_raw = _universe_state.get("last_rebalance")
                last_rebalance = pd.to_datetime(str(last_rebalance_raw), errors="coerce") if last_rebalance_raw else pd.NaT
                last_rebalance = pd.Timestamp(last_rebalance).normalize() if pd.notna(last_rebalance) else None
                if last_rebalance is not None:
                    n_since = _count_sessions_between(_universe_sessions, last_rebalance, asof_ts)
                    if n_since is not None and n_since < int(_universe_cfg.rebalance_every_sessions):
                        sat_state_prev_obj = _universe_state.get("satellites")
                        sat_state_prev = sat_state_prev_obj if isinstance(sat_state_prev_obj, dict) else {}
                        prev_sat = [str(sym).upper() for sym in sat_state_prev.keys() if isinstance(sym, str)]
                        return set(_norm_syms([*core, *prev_sat]))

                sat_state_obj = _universe_state.get("satellites")
                sat_state = sat_state_obj if isinstance(sat_state_obj, dict) else {}
                active_satellites = _norm_syms(list(sat_state.keys()))

                # Score all candidates as-of.
                scores: dict[str, float] = {}

                def _estimate_adv_usd(panel: pd.DataFrame, *, window: int = 20) -> Optional[float]:
                    if panel.empty:
                        return None
                    close_col = None
                    for c in ("close", "price_close", "adj_close"):
                        if c in panel.columns:
                            close_col = c
                            break
                    vol_col = None
                    for c in ("volume", "price_volume"):
                        if c in panel.columns:
                            vol_col = c
                            break
                    if close_col is None or vol_col is None:
                        return None
                    close = pd.to_numeric(panel[close_col], errors="coerce").ffill()
                    vol = pd.to_numeric(panel[vol_col], errors="coerce").ffill()
                    dollar_vol = (close * vol).replace([np.inf, -np.inf], np.nan)
                    adv = dollar_vol.rolling(window=window, min_periods=max(5, window // 4)).mean()
                    try:
                        return float(adv.dropna().iloc[-1]) if not adv.empty else None
                    except Exception:
                        return None

                def score_symbol(sym: str) -> Optional[float]:
                    panel = prepared.trackc_aligned_by.get(sym)
                    if panel is None or getattr(panel, "empty", True):
                        return None
                    try:
                        df = panel
                        df = df.loc[pd.to_datetime(df.index, errors="coerce") <= pd.Timestamp(asof_ts)]
                    except Exception:
                        return None
                    if df.empty:
                        return None
                    tail = df.tail(int(_universe_cfg.lookback_sessions))
                    if len(tail) < int(_universe_cfg.min_history_sessions):
                        return None
                    if _universe_cfg.min_adv_usd is not None:
                        adv_usd = _estimate_adv_usd(tail)
                        if adv_usd is None or adv_usd < float(_universe_cfg.min_adv_usd):
                            return None
                    es = compute_event_score(tail)
                    if es.empty:
                        return None
                    try:
                        return float(pd.to_numeric(es, errors="coerce").fillna(0.0).mean())
                    except Exception:
                        return None

                for sym in candidates:
                    if sym in core:
                        continue
                    sc = score_symbol(sym)
                    if sc is None:
                        continue
                    scores[sym] = float(sc)

                # Update peaks for active satellites.
                for sym in active_satellites:
                    cur = scores.get(sym)
                    if cur is None:
                        continue
                    entry = sat_state.get(sym, {}) if isinstance(sat_state.get(sym), dict) else {}
                    peak = float(entry.get("peak", 0.0) or 0.0)
                    if cur > peak:
                        entry["peak"] = float(cur)
                        sat_state[sym] = entry

                # Removal pass: only remove after min-stay and only on decay.
                kept: list[str] = []
                removed: list[str] = []

                for sym in active_satellites:
                    entry = sat_state.get(sym, {}) if isinstance(sat_state.get(sym), dict) else {}
                    added_raw = entry.get("added")
                    added_ts_parsed = pd.to_datetime(str(added_raw), errors="coerce") if added_raw else pd.NaT
                    added_ts = pd.Timestamp(added_ts_parsed).normalize() if pd.notna(added_ts_parsed) else None

                    cur = float(scores.get(sym, 0.0) or 0.0)
                    peak = float(entry.get("peak", cur) or cur)
                    decay_gate = max(float(_universe_cfg.remove_abs_threshold), float(peak) * float(_universe_cfg.decay_fraction))

                    age_sessions = None
                    if added_ts is not None:
                        age_sessions = _count_sessions_between(_universe_sessions, added_ts, asof_ts)

                    if age_sessions is not None and age_sessions < int(_universe_cfg.min_stay_sessions):
                        kept.append(sym)
                        continue

                    if cur <= decay_gate:
                        removed.append(sym)
                    else:
                        kept.append(sym)

                for sym in removed:
                    sat_state.pop(sym, None)

                # Add pass: top up to sat_min..sat_max.
                kept_set = set(kept)
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                additions: list[str] = []
                for sym, _ in ranked:
                    if sym in kept_set or sym in core:
                        continue
                    kept_set.add(sym)
                    additions.append(sym)
                    if len(kept_set) >= sat_max:
                        break

                if len(kept_set) < sat_min:
                    for sym, _ in ranked:
                        if sym in kept_set or sym in core:
                            continue
                        kept_set.add(sym)
                        additions.append(sym)
                        if len(kept_set) >= sat_min:
                            break

                # Update state entries for new additions.
                for sym in additions:
                    cur = float(scores.get(sym, 0.0) or 0.0)
                    sat_state[sym] = {"added": asof_ts.strftime("%Y-%m-%d"), "peak": float(cur)}

                # Persist state.
                _universe_state["last_rebalance"] = asof_ts.strftime("%Y-%m-%d")
                _universe_state["satellites"] = sat_state
                if _universe_state_path is not None:
                    _save_universe_state(_universe_state_path, _universe_state)

                satellites = _norm_syms([*kept, *additions])
                return set(_norm_syms([*core, *satellites]))

            logger.info(
                "[phase2.v2] runtime universe gate enabled: core=%d candidates=%d rebalance_every=%d",
                len(_norm_syms(_universe_cfg.core_symbols)),
                len(_norm_syms(_universe_cfg.candidate_symbols)),
                int(_universe_cfg.rebalance_every_sessions),
            )
        elif universe_enabled and universe_mode == "full":
            logger.info("[phase2.v2] runtime universe gate enabled: mode=full (eligible universe)")

        # Track previous eligible count for collapse detection
        _prev_eligible_count: int = n_assets
        # Track previous z-scores for sign flip detection
        _prev_z_for_flip: Optional[np.ndarray] = None

        for i, day in enumerate(union_oos_index):
            # ─────────────────────────────────────────────────────────────────
            # RISK LATCH: Begin session (decrement latches, check releases)
            # ─────────────────────────────────────────────────────────────────
            if risk_latch_enabled and risk_latch is not None:
                risk_latch.begin_session(i)
            
            # ─────────────────────────────────────────────────────────────────
            # INTRADAY MONITOR: Check for live alerts (top of each day)
            # If the intraday monitor has flagged EMERGENCY, we do not argue:
            # we set w_target=0 and skip the rest of the day.
            # KEY RULE: Intraday monitor never touches Mamba. It only sets a
            # risk latch that overrides allocations.
            # ─────────────────────────────────────────────────────────────────
            intraday_emergency = False
            if intraday_monitor_enabled:
                try:
                    from src.stage_b_stateful.intraday_monitor import (
                        read_intraday_snapshot,
                        apply_intraday_snapshot_to_latch,
                    )
                    intraday_snapshot = read_intraday_snapshot(
                        path=intraday_snapshot_path,
                        max_age_seconds=intraday_snapshot_max_age,
                    )
                    
                    if intraday_snapshot is not None:
                        if intraday_snapshot.should_emergency_stop():
                            intraday_emergency = True
                            logger.warning(
                                "[phase2.intraday_monitor] EMERGENCY from live monitor: %s",
                                intraday_snapshot.trigger_reason,
                            )
                            
                            # Apply to risk latch
                            if risk_latch is not None:
                                apply_intraday_snapshot_to_latch(intraday_snapshot, risk_latch)
                            
                            # EMIT: RISK_LATCH_EMERGENCY
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.CRITICAL,
                                code=EventCode.RISK_LATCH_EMERGENCY,
                                message=f"Intraday monitor EMERGENCY: {intraday_snapshot.trigger_reason}",
                                payload=intraday_snapshot.to_dict(),
                            )
                            
                            # Force flatten for this day
                            w = np.zeros(n_assets, dtype=float)
                            w_mat[i, :] = w
                            
                            # Still need to track PnL on current positions
                            r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                            pnl = float(exec_w @ r_vec)
                            
                            if trade_delay_sessions > 1:
                                _queue.append(w.copy())
                                w_next = np.asarray(_queue.popleft(), dtype=float)
                            else:
                                w_next = w.copy()
                            
                            tval_intent = float(np.sum(np.abs(w - prev_w)))
                            tval_exec = float(np.sum(np.abs(w_next - exec_w)))
                            turnover_intent[i] = tval_intent
                            turnover_exec[i] = tval_exec
                            turnover[i] = tval_exec
                            
                            cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
                            costs[i] = cost
                            pnl_net = float(pnl - cost)
                            net_ret[i] = pnl_net
                            equity[i] = float((equity[i - 1] if i > 0 else 1.0) * (1.0 + pnl_net))
                            equity_peak = float(max(equity_peak, equity[i]))
                            cov = _ewma_cov_update(cov, r_vec, cov_lam)
                            
                            prev_exec_w = exec_w.copy()
                            exec_w = w_next
                            prev_w = w
                            continue
                        
                        elif intraday_snapshot.should_throttle():
                            # Apply throttle to risk latch
                            if risk_latch is not None:
                                apply_intraday_snapshot_to_latch(intraday_snapshot, risk_latch)
                            logger.info(
                                "[phase2.intraday_monitor] Throttle from live monitor: %s (scale=%.2f)",
                                intraday_snapshot.trigger_reason,
                                intraday_snapshot.recommended_scale,
                            )
                
                except Exception as e:
                    logger.debug("[phase2.intraday_monitor] Error reading snapshot: %s", e)
            
            # Compute registry eligibility mask for this day (used for training/inference/weights gating).
            if _registry_df is not None:
                try:
                    asof_ts_reg = pd.Timestamp(day).normalize()
                    elig_now = _eligible_symbols(asof_ts_reg)
                    if elig_now is None:
                        _eligible_mask = None
                    else:
                        _eligible_mask = np.asarray([str(sym).upper() in elig_now for sym in syms], dtype=bool)
                        # Log eligibility changes (useful for debugging zombie drops).
                        if _eligible_prev is not None:
                            dropped = sorted(list(_eligible_prev - elig_now))
                            if dropped:
                                logger.warning(
                                    "[phase2.universe_registry] %s dropped %d ineligible symbols: %s",
                                    str(asof_ts_reg)[:10],
                                    len(dropped),
                                    ",".join(dropped[:25]) + ("..." if len(dropped) > 25 else ""),
                                )
                        _eligible_prev = set(elig_now)
                except Exception:
                    _eligible_mask = None

            # Update runtime universe (piecewise-constant) on the current day.
            # We do this early so any subsequent weight computation is gated.
            if universe_enabled and universe_mode == "core_satellite" and _universe_cfg is not None:
                try:
                    asof_ts = pd.Timestamp(day).normalize()
                    current_universe = _rebalance_universe(asof_ts)
                    # Constrain to the symbols actually in this run.
                    current_universe = set(str(s).upper() for s in current_universe) & {str(s).upper() for s in syms}
                    # Enforce registry eligibility again even if state carries over.
                    elig_day = _eligible_symbols(asof_ts)
                    if elig_day is not None:
                        current_universe = set(sym for sym in current_universe if sym in elig_day)
                    _universe_mask = np.asarray([str(sym).upper() in current_universe for sym in syms], dtype=bool)
                except Exception:
                    # Fail open: keep behavior unchanged.
                    current_universe = None
                    _universe_mask = None
            elif universe_enabled and universe_mode == "full":
                # Full-universe gating is updated daily. If registry is present, it defines eligibility.
                try:
                    asof_ts = pd.Timestamp(day).normalize()
                    elig_day = _eligible_symbols(asof_ts)
                    if elig_day is None:
                        current_universe = set(str(s).upper() for s in syms)
                    else:
                        current_universe = set(str(s).upper() for s in syms) & set(str(s).upper() for s in elig_day)
                    _universe_mask = np.asarray([str(sym).upper() in current_universe for sym in syms], dtype=bool)
                except Exception:
                    current_universe = None
                    _universe_mask = None

            # ─────────────────────────────────────────────────────────────────
            # POINT 1: INPUT HEALTH CHECKS (before inference)
            # Check role_ctx availability, hygiene_ok %, eligible mask collapse
            # ─────────────────────────────────────────────────────────────────
            input_health_fallback = False
            if risk_latch_enabled and risk_latch is not None:
                try:
                    # 1. Role context availability and staleness
                    role_ctx_ok = role_ctx is not None
                    role_ctx_stale_days = 0
                    
                    # 2. Hygiene percentage from role_ctx
                    hygiene_ok_pct_for_check = 1.0
                    if role_ctx is not None:
                        try:
                            _day_ctx_health = role_ctx.get_for_day(pd.Timestamp(day))
                            _hygiene_arr = np.asarray(_day_ctx_health.hygiene_ok, dtype=bool)
                            if len(_hygiene_arr) > 0:
                                hygiene_ok_pct_for_check = float(np.mean(_hygiene_arr))
                            # Check staleness (compare day_ctx date to current day)
                            if hasattr(_day_ctx_health, 'date') and _day_ctx_health.date is not None:
                                ctx_date = pd.Timestamp(_day_ctx_health.date).normalize()
                                day_date = pd.Timestamp(day).normalize()
                                role_ctx_stale_days = int((day_date - ctx_date).days)
                        except Exception:
                            pass
                    
                    # 3. Eligible mask collapse detection
                    eligible_count_today = n_assets  # Default to all eligible
                    if _eligible_mask is not None:
                        eligible_count_today = int(np.sum(_eligible_mask))
                    
                    # Run the input health check
                    input_health_mode = risk_latch.check_input_health(
                        hygiene_ok_pct=hygiene_ok_pct_for_check,
                        role_ctx_available=role_ctx_ok,
                        role_ctx_days_stale=role_ctx_stale_days,
                        eligible_count_today=eligible_count_today,
                        eligible_count_yesterday=_prev_eligible_count,
                    )
                    
                    # Update prev count for next iteration
                    _prev_eligible_count = eligible_count_today
                    
                    # If triggered SAFE_FALLBACK or worse, resolve and handle
                    if input_health_mode >= RiskLatchMode.SAFE_FALLBACK:
                        input_health_state = risk_latch.resolve()
                        
                        # Check if we should flatten OR hold fallback
                        if input_health_state.should_flatten() or input_health_state.should_hold_fallback():
                            input_health_fallback = True
                            
                            # Determine action taken for logging
                            if input_health_state.should_hold_fallback():
                                action_taken = "SAFE_FALLBACK_HOLD"
                                exposure_scale_log = 1.0 if input_health_state.last_good_weights is not None else 0.0
                            else:
                                action_taken = "FLATTEN"
                                exposure_scale_log = 0.0
                            
                            # ─────────────────────────────────────────────────────
                            # LEDGER: Log input health fail event
                            # ─────────────────────────────────────────────────────
                            if risk_events_ledger is not None:
                                try:
                                    risk_events_ledger.add_event(
                                        date=str(pd.Timestamp(day))[:10],
                                        day_idx=i,
                                        event_type="INPUT_HEALTH_FAIL",
                                        threshold="hygiene_ok >= 30% AND role_ctx fresh AND eligible stable",
                                        threshold_crossed=True,
                                        action_taken=action_taken,
                                        exposure_scale=exposure_scale_log,
                                        trigger_reason=(
                                            input_health_state.primary_reason.name
                                            if input_health_state.primary_reason
                                            else "unknown"
                                        ),
                                        details={
                                            "hygiene_ok_pct": hygiene_ok_pct_for_check,
                                            "role_ctx_ok": role_ctx_ok,
                                            "role_ctx_stale_days": role_ctx_stale_days,
                                            "eligible_count_today": eligible_count_today,
                                            "eligible_count_yesterday": _prev_eligible_count,
                                        },
                                    )
                                except Exception as le:
                                    logger.debug("[phase2.ledger] Failed to log event: %s", le)
                            
                            # EMIT: RISK_LATCH event
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.RISK_LATCH_SAFE_FALLBACK,
                                message=f"Input health {input_health_state.mode.name}: {input_health_state.primary_reason.name if input_health_state.primary_reason else 'unknown'}",
                                payload=input_health_state.to_dict(),
                            )
                            
                            # Apply safe fallback or flatten
                            w = input_health_state.get_effective_weights(
                                np.zeros(n_assets, dtype=float),
                                n_assets,
                            )
                            w_mat[i, :] = w
                            
                            # Handle PnL and execution queue
                            r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                            pnl = float(exec_w @ r_vec)
                            
                            if trade_delay_sessions > 1:
                                _queue.append(w.copy())
                                w_next = np.asarray(_queue.popleft(), dtype=float)
                            else:
                                w_next = w.copy()
                            
                            tval_intent = float(np.sum(np.abs(w - prev_w)))
                            tval_exec = float(np.sum(np.abs(w_next - exec_w)))
                            turnover_intent[i] = tval_intent
                            turnover_exec[i] = tval_exec
                            turnover[i] = tval_exec
                            
                            cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
                            costs[i] = cost
                            pnl_net = float(pnl - cost)
                            net_ret[i] = pnl_net
                            equity[i] = float((equity[i - 1] if i > 0 else 1.0) * (1.0 + pnl_net))
                            equity_peak = float(max(equity_peak, equity[i]))
                            cov = _ewma_cov_update(cov, r_vec, cov_lam)
                            
                            prev_exec_w = exec_w.copy()
                            exec_w = w_next
                            prev_w = w
                            
                            if risk_latch is not None:
                                risk_latch.end_session(None)
                            continue
                except Exception as e:
                    logger.debug("[phase2.risk_latch] Input health check error: %s", e)

            # If we are in a forced-flat safety cooldown, keep portfolio flat.
            if safety_enabled and int(flat_until_idx) >= 0 and i <= int(flat_until_idx):
                # EMIT: COOLDOWN_ACTIVE
                event_bus.emit(
                    date=str(pd.Timestamp(day))[:10],
                    day_idx=i,
                    severity=EventSeverity.WARNING,
                    code=EventCode.COOLDOWN_ACTIVE,
                    message=f"Cooldown active until day {flat_until_idx}",
                    payload={"flat_until_idx": int(flat_until_idx), "days_remaining": int(flat_until_idx - i)},
                )
                
                # FIX Gap #5: Cooldown means FLAT (zeros), not "hold prev_w"
                # Intent is to flatten, so target = zeros
                w_target_cooldown = np.zeros(n_assets, dtype=float)
                w_mat[i, :] = w_target_cooldown
                
                # Realized PnL accrues on current EXECUTED weights
                r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                pnl = float(exec_w @ r_vec)
                
                # FIX Gap #2: ALWAYS advance the execution queue, even during cooldown
                # Otherwise exec_w becomes stale and we get mis-timed exposures
                if trade_delay_sessions > 1:
                    _queue.append(w_target_cooldown.copy())
                    w_next_hold = np.asarray(_queue.popleft(), dtype=float)
                else:
                    w_next_hold = w_target_cooldown.copy()
                
                # Compute turnover for the actual execution that will happen
                tval_intent = float(np.sum(np.abs(w_target_cooldown - prev_w)))
                tval_exec = float(np.sum(np.abs(w_next_hold - exec_w)))
                turnover_intent[i] = tval_intent
                turnover_exec[i] = tval_exec
                turnover[i] = tval_exec
                
                # Costs on executed turnover (liquidation cost if we're unwinding)
                cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
                costs[i] = cost
                pnl_net = float(pnl - cost)
                net_ret[i] = pnl_net
                equity[i] = float((equity[i - 1] if i > 0 else 1.0) * (1.0 + pnl_net))
                equity_peak = float(max(equity_peak, equity[i]))
                cov = _ewma_cov_update(cov, r_vec, cov_lam)
                
                # Update state for next iteration
                prev_exec_w = exec_w.copy()
                exec_w = w_next_hold
                prev_w = w_target_cooldown
                continue

            if i in update_positions:
                # Only use matured labels up to cutoff = day - H.
                cutoff = maturity_cutoff(pd.Timestamp(day), H=int(horizon))
                try:
                    start = add_sessions(cutoff, -int(replay_days)) if int(replay_days) > 0 else pd.Timestamp.min
                except Exception:
                    start = pd.Timestamp(cutoff) - pd.Timedelta(days=int(replay_days))

                # Select samples with label timestamps in [start, cutoff].
                if prepared.gpu_store_train is not None and len(ts) > 0:
                    ts_sym = np.asarray([x[0] for x in ts], dtype=object)
                    ts_day = pd.to_datetime(np.asarray([x[1] for x in ts], dtype=object))
                    mask = (ts_day >= pd.Timestamp(start)) & (ts_day <= pd.Timestamp(cutoff))
                    # Keep only symbols in this trial (defensive).
                    mask = mask & np.isin(ts_sym, np.asarray(syms, dtype=object))
                    # Registry gate: do not train updates on ineligible symbols.
                    if _registry_df is not None:
                        try:
                            elig_cutoff = _eligible_symbols(pd.Timestamp(cutoff).normalize())
                            if elig_cutoff is not None:
                                mask = mask & np.isin(ts_sym, np.asarray(sorted(list(elig_cutoff)), dtype=object))
                        except Exception:
                            pass
                    idxs = np.where(mask)[0]
                    if len(idxs) >= 64:
                        sub_samples = [samples[int(j)] for j in idxs]
                        sub_ts = ts[idxs]
                        pooled_seq = WindowedSequenceData(
                            store=prepared.gpu_store_train,
                            seq_len=seq_len,
                            samples=sub_samples,
                            timestamps=sub_ts,
                        )
                    else:
                        pooled_seq = None
                else:
                    pooled_seq = None

                if pooled_seq is None:
                    # CPU fallback: build from full masters and then filter by cutoff.
                    pooled_seq = _build_pooled_sequence_cpu(
                        symbols=syms,
                        seq_len=seq_len,
                        prepared=prepared,
                    )

                    try:
                        ts_day_cpu = pd.to_datetime(np.asarray([x[1] for x in pooled_seq.timestamps], dtype=object))
                        mask_cpu = (ts_day_cpu >= pd.Timestamp(start)) & (ts_day_cpu <= pd.Timestamp(cutoff))
                        if _registry_df is not None:
                            try:
                                ts_sym_cpu = np.asarray([x[0] for x in pooled_seq.timestamps], dtype=object)
                                elig_cutoff = _eligible_symbols(pd.Timestamp(cutoff).normalize())
                                if elig_cutoff is not None:
                                    mask_cpu = mask_cpu & np.isin(ts_sym_cpu, np.asarray(sorted(list(elig_cutoff)), dtype=object))
                            except Exception:
                                pass
                        idxs_cpu = np.where(mask_cpu)[0]
                        if len(idxs_cpu) >= 32:
                            pooled_seq = SequenceData(
                                sequences=np.asarray(pooled_seq.sequences)[idxs_cpu],
                                targets=np.asarray(pooled_seq.targets)[idxs_cpu],
                                timestamps=np.asarray(pooled_seq.timestamps, dtype=object)[idxs_cpu],
                            )
                    except Exception:
                        pass

                n_train = int(len(pooled_seq))
                if n_train < 64:
                    # Not enough history; skip update.
                    pass
                else:
                    split = int(n_train * float(np.clip(float(cfg.get("train_fraction", 0.9)), 0.5, 0.95)))
                    train_idx = np.arange(max(1, split))
                    val_idx = np.arange(max(1, split), n_train)
                    if len(val_idx) < 5:
                        val_idx = np.arange(max(0, n_train - 5), n_train)
                        train_idx = np.arange(0, max(1, n_train - len(val_idx)))

                    cfg_upd = dict(cfg)
                    # Spec: uncertainty head (mu,sigma) + Gaussian NLL.
                    cfg_upd["mamba_head_type"] = "gaussian"
                    cfg_upd["mamba_loss_fn"] = "gaussian_nll"
                    cfg_upd["mamba_max_epochs"] = int(update_epochs)
                    cfg_upd["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 2))
                    cfg_upd["max_epochs"] = int(update_epochs)

                    # ─────────────────────────────────────────────────────────────
                    # Build calibration context for calibration-gated learning.
                    # AUTHORITATIVE: Use Mamba-based calibration (μ, σ²) tracking.
                    # Quantile-based calibration is now SECONDARY (20-30% weight).
                    # ─────────────────────────────────────────────────────────────
                    calibration_context: Optional[Dict[str, Any]] = None
                    learning_gov_decision = None
                    
                    try:
                        # Primary: Mamba calibration from tracker.
                        if mamba_calib_snapshot is not None and mamba_calib_snapshot.is_reliable:
                            calib_overall = float(mamba_calib_snapshot.calibration_overall)
                        else:
                            # Fallback: quantile-based during warm-up.
                            calib_overall = 1.0
                            if role_ctx is not None:
                                try:
                                    upd_day_ctx = role_ctx.get_for_day(pd.Timestamp(cutoff))
                                    calib_overall = float(upd_day_ctx.calibration_overall_score)
                                except Exception:
                                    pass
                        
                        # ─────────────────────────────────────────────────────────
                        # Learning Governor: Three-zone policy with drift detection
                        # This gates learning based on calibration:
                        #   GREEN (cal >= 0.70): slow updates
                        #   YELLOW (0.55 <= cal < 0.70): normal updates
                        #   RED (cal < 0.55): freeze learning
                        # ─────────────────────────────────────────────────────────
                        if learning_governor is not None:
                            try:
                                from src.stage_b_stateful.learning_governor import LearningDecision
                                
                                learning_gov_decision = learning_governor.get_decision(
                                    date=str(pd.Timestamp(day))[:10],
                                    day_idx=i,
                                    calibration_score=calib_overall,
                                )
                                
                                # Apply learning rate multiplier from governor
                                lr_mult_from_gov = float(learning_gov_decision.learning_rate_mult)
                                
                                # If frozen or locked, skip update entirely
                                if learning_gov_decision.decision in (LearningDecision.FREEZE_LEARNING, LearningDecision.REQUIRE_UNLOCK):
                                    logger.info(
                                        "[phase2.learning_gov] Skipping update: %s (zone=%s, cal=%.3f)",
                                        learning_gov_decision.reason,
                                        learning_gov_decision.zone.value,
                                        calib_overall,
                                    )
                                    # Emit event
                                    event_bus.emit(
                                        date=str(pd.Timestamp(day))[:10],
                                        day_idx=i,
                                        severity=EventSeverity.WARNING,
                                        code=EventCode.LEARNING_FROZEN,
                                        message=f"Learning frozen: {learning_gov_decision.reason}",
                                        payload={
                                            "zone": learning_gov_decision.zone.value,
                                            "calibration_score": calib_overall,
                                            "decision": learning_gov_decision.decision.value,
                                            "is_locked": learning_gov_decision.is_locked,
                                        },
                                    )
                                    # Skip the update entirely
                                    continue
                            except Exception as e:
                                logger.warning("[phase2.learning_gov] Decision error: %s", e)
                                lr_mult_from_gov = 1.0
                        else:
                            lr_mult_from_gov = 1.0
                        
                        # Online trust from role context (still valid).
                        online_trust = 1.0
                        if role_ctx is not None:
                            try:
                                upd_day_ctx = role_ctx.get_for_day(pd.Timestamp(cutoff))
                                online_trust = float(upd_day_ctx.online_trust_score)
                            except Exception:
                                pass
                        
                        calibration_context = {
                            "calibration_overall_score": calib_overall,
                            "online_trust_score": online_trust,
                            # Thresholds from cfg (with defaults)
                            "calib_freeze_threshold": float(cfg.get("calib_freeze_threshold", 0.3)),
                            "trust_freeze_threshold": float(cfg.get("trust_freeze_threshold", 0.3)),
                            "calib_decay_threshold": float(cfg.get("calib_decay_threshold", 0.6)),
                            "overconfidence_penalty_lambda": float(cfg.get("overconfidence_penalty_lambda", 0.1)),
                            # Learning governor multiplier
                            "lr_mult_from_governor": lr_mult_from_gov,
                            # Mamba calibration details for diagnostics.
                            "mamba_calib_reliable": bool(mamba_calib_snapshot.is_reliable) if mamba_calib_snapshot else False,
                            "mamba_mu_score": float(mamba_calib_snapshot.mu_score) if mamba_calib_snapshot else 0.5,
                            "mamba_sigma_score": float(mamba_calib_snapshot.sigma_score) if mamba_calib_snapshot else 0.5,
                            "mamba_nll_score": float(mamba_calib_snapshot.nll_score) if mamba_calib_snapshot else 0.5,
                            "mamba_n_matured": int(mamba_calib_snapshot.n_matured) if mamba_calib_snapshot else 0,
                        }
                    except Exception as e:
                        logger.debug("[phase2.calib_gate] failed to build calibration context for %s: %s", cutoff, e)
                        calibration_context = None

                    # Apply learning governor's learning rate multiplier to cfg_upd
                    if lr_mult_from_gov != 1.0:
                        base_lr = float(cfg_upd.get("learning_rate", cfg_upd.get("lr", 1e-4)))
                        cfg_upd["learning_rate"] = base_lr * lr_mult_from_gov
                        cfg_upd["lr"] = base_lr * lr_mult_from_gov
                        logger.debug(
                            "[phase2.learning_gov] Applied lr_mult=%.3f: base_lr=%.2e -> effective_lr=%.2e",
                            lr_mult_from_gov, base_lr, cfg_upd["learning_rate"],
                        )

                    # Optional: baseline evaluation before update.
                    pre_eval: Optional[Dict[str, float]] = None
                    if diag_enabled and diag_pre_post and warm_state is not None:
                        try:
                            t0 = time.time()
                            pre_eval = _eval_warm_state_on_val(pooled_seq, val_idx, warm_state, cfg_upd)
                            pre_eval["elapsed_s"] = float(max(0.0, time.time() - t0))
                        except Exception:
                            pre_eval = None

                    upd_t0 = time.time()

                    train_result = train_mamba_fold(
                        pooled_seq,
                        train_idx,
                        val_idx,
                        cfg_upd,
                        device=device,
                        warm_state=warm_state,
                        return_model=True,
                        calibration_context=calibration_context,
                    )

                    upd_elapsed = float(max(0.0, time.time() - upd_t0))
                    model = train_result.get("model")
                    if model is not None:
                        try:
                            warm_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        except Exception:
                            warm_state = None

                    # Post-update eval from train_result.
                    post_val_loss = float(train_result.get("val_loss", float("nan")))
                    post_corr = float("nan")
                    try:
                        mu_val = np.asarray(train_result.get("preds"), dtype=float).reshape(-1)
                        y_val = np.asarray(pooled_seq.targets, dtype=float).reshape(-1)[np.asarray(val_idx, dtype=int)]
                        post_corr = float(_safe_corr(mu_val, y_val))
                    except Exception:
                        post_corr = float("nan")

                    # ─────────────────────────────────────────────────────────
                    # Report update outcome to LearningGovernor for auto-revert
                    # ─────────────────────────────────────────────────────────
                    if learning_governor is not None and learning_gov_decision is not None:
                        try:
                            # Get post-update calibration (use the updated model's calibration)
                            post_update_calib = calib_overall  # Start with pre-update
                            if mamba_calib_tracker is not None:
                                try:
                                    # Try to get fresh calibration after update
                                    # Note: Full calibration update happens later, so we estimate
                                    # based on val_corr improvement as proxy
                                    delta_corr = post_corr - (float(pre_eval.get("val_corr", 0.5)) if pre_eval else 0.5)
                                    # Scale calibration estimate by correlation improvement
                                    post_update_calib = min(1.0, max(0.0, calib_overall + 0.1 * delta_corr))
                                except Exception:
                                    post_update_calib = calib_overall
                            
                            # Check lock state before update
                            was_locked = learning_gov_decision.is_locked
                            
                            # Report outcome - this can trigger auto-lock if calibration dropped
                            update_accepted = learning_governor.report_update_outcome(
                                date=str(pd.Timestamp(day))[:10],
                                pre_calib=float(calib_overall),
                                post_calib=float(post_update_calib),
                                checkpoint_path=None,  # TODO: implement checkpoint saving
                            )
                            
                            # Check if lock state changed
                            # Get fresh decision to check current lock state
                            from src.stage_b_stateful.learning_governor import LearningDecision
                            current_decision = learning_governor.get_decision(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                calibration_score=float(post_update_calib),
                            )
                            is_now_locked = current_decision.is_locked
                            
                            # Emit LOCKED event if state transitioned to locked
                            if not was_locked and is_now_locked:
                                event_bus.emit(
                                    date=str(pd.Timestamp(day))[:10],
                                    day_idx=i,
                                    severity=EventSeverity.WARNING,
                                    code=EventCode.LEARNING_GOV_LOCKED,
                                    message=f"Learning governor auto-locked: calibration dropped {post_update_calib - calib_overall:.3f}",
                                    payload={
                                        "pre_calibration": float(calib_overall),
                                        "post_calibration": float(post_update_calib),
                                        "delta": float(post_update_calib - calib_overall),
                                        "lock_reason": str(getattr(learning_governor._state, "lock_reason", "unknown")),
                                    },
                                )
                            
                            # Emit event for governor decision
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.INFO,
                                code=EventCode.LEARNING_GOV_DECISION,
                                message=f"Learning update: zone={learning_gov_decision.zone.value}, lr_mult={learning_gov_decision.learning_rate_mult:.3f}",
                                payload={
                                    "zone": learning_gov_decision.zone.value,
                                    "decision": learning_gov_decision.decision.value,
                                    "lr_mult": float(learning_gov_decision.learning_rate_mult),
                                    "cadence_mult": float(learning_gov_decision.cadence_mult),
                                    "pre_calibration": float(calib_overall),
                                    "post_calibration": float(post_update_calib),
                                    "drift_detected": learning_gov_decision.drift_detected,
                                    "is_locked": is_now_locked,
                                    "update_accepted": bool(update_accepted),
                                },
                            )
                        except Exception as e:
                            logger.warning("[phase2.learning_gov] Outcome reporting error: %s", e)

                    if diag_enabled:
                        rec: Dict[str, Any] = {
                            "kind": "update",
                            "day": str(pd.Timestamp(day))[:10],
                            "cutoff": str(pd.Timestamp(cutoff))[:10],
                            "start": str(pd.Timestamp(start))[:10],
                            "replay_days": int(replay_days),
                            "n_train": int(n_train),
                            "n_val": int(len(val_idx)),
                            "update_epochs": int(update_epochs),
                            "elapsed_s": float(upd_elapsed),
                            "val_loss_post": float(post_val_loss),
                            "val_corr_post": float(post_corr),
                        }
                        # Add calibration-gated learning decisions to diagnostics
                        learning_decisions = train_result.get("learning_decisions", {})
                        if learning_decisions:
                            rec.update({
                                "calib_learning_allowed": bool(learning_decisions.get("learning_allowed", True)),
                                "calib_learning_frozen_reason": str(learning_decisions.get("learning_frozen_reason") or ""),
                                "calib_score": float(learning_decisions.get("calib_score", 1.0)),
                                "trust_score": float(learning_decisions.get("trust_score", 1.0)),
                                "calib_lr_scale": float(learning_decisions.get("lr_scale", 1.0)),
                                "calib_effective_lr": float(learning_decisions.get("effective_lr", 0.0)),
                                "calib_gradients_applied": int(learning_decisions.get("gradients_applied", 0)),
                                "calib_gradients_skipped": int(learning_decisions.get("gradients_skipped", 0)),
                            })
                        # Add Mamba calibration tracker details (AUTHORITATIVE source)
                        if mamba_calib_snapshot is not None:
                            rec.update({
                                "mamba_calib_overall": float(mamba_calib_snapshot.calibration_overall),
                                "mamba_calib_mamba_only": float(mamba_calib_snapshot.mamba_overall),
                                "mamba_mu_score": float(mamba_calib_snapshot.mu_score),
                                "mamba_mu_correlation": float(mamba_calib_snapshot.mu_correlation),
                                "mamba_mu_directional_accuracy": float(mamba_calib_snapshot.mu_directional_accuracy),
                                "mamba_sigma_score": float(mamba_calib_snapshot.sigma_score),
                                "mamba_sigma_calibration": float(mamba_calib_snapshot.sigma_calibration),
                                "mamba_sigma_coverage_1std": float(mamba_calib_snapshot.sigma_coverage_1std),
                                "mamba_sigma_coverage_2std": float(mamba_calib_snapshot.sigma_coverage_2std),
                                "mamba_nll_score": float(mamba_calib_snapshot.nll_score),
                                "mamba_nll_mean": float(mamba_calib_snapshot.nll_mean),
                                "mamba_nll_std": float(mamba_calib_snapshot.nll_std),
                                "mamba_quantile_secondary": float(mamba_calib_snapshot.quantile_score),
                                "mamba_n_matured": int(mamba_calib_snapshot.n_matured),
                                "mamba_is_reliable": bool(mamba_calib_snapshot.is_reliable),
                            })
                        if pre_eval is not None:
                            rec.update(
                                {
                                    "val_loss_pre": float(pre_eval.get("val_loss", float("nan"))),
                                    "val_corr_pre": float(pre_eval.get("val_corr", float("nan"))),
                                    "pre_eval_elapsed_s": float(pre_eval.get("elapsed_s", float("nan"))),
                                }
                            )
                            try:
                                rec["delta_val_loss"] = float(rec["val_loss_post"] - rec["val_loss_pre"])  # lower is better
                                rec["delta_val_corr"] = float(rec["val_corr_post"] - rec["val_corr_pre"])  # higher is better
                            except Exception:
                                pass
                        # Add LearningGovernor decision to diagnostics
                        if learning_gov_decision is not None:
                            try:
                                rec.update({
                                    "gov_zone": str(learning_gov_decision.zone.value),
                                    "gov_decision": str(learning_gov_decision.decision.value),
                                    "gov_lr_mult": float(learning_gov_decision.learning_rate_mult),
                                    "gov_cadence_mult": float(learning_gov_decision.cadence_mult),
                                    "gov_is_drift_triggered": bool(learning_gov_decision.is_drift_triggered),
                                    "gov_is_locked": bool(learning_gov_decision.is_locked),
                                    "gov_reason": str(learning_gov_decision.reason),
                                })
                            except Exception:
                                pass
                        update_logs.append(rec)
                        # Keep log single-line and scan-friendly.
                        logger.info(
                            "[phase2.v2.diag] update day=%s n_train=%d val_loss=%.5f corr=%.3f elapsed=%.1fs",
                            str(pd.Timestamp(day))[:10],
                            int(n_train),
                            float(post_val_loss),
                            float(post_corr) if np.isfinite(post_corr) else float("nan"),
                            float(upd_elapsed),
                        )

                    updates_done += 1
                    # Periodic progress logging (every 5 updates or every 100 sessions).
                    if updates_done % 5 == 0 or (i - _last_progress_log) >= 100:
                        eq_now = float(equity[max(0, i - 1)] if i > 0 else 1.0)
                        logger.info(
                            "[phase2.v2] progress: session %d/%d (%.1f%%), updates=%d/%d, equity=%.4f, day=%s",
                            i + 1, n_oos, 100.0 * (i + 1) / n_oos, updates_done, n_updates_total, eq_now, str(day)[:10],
                        )
                        _last_progress_log = i

            if model is None:
                # If we somehow never trained, keep portfolio flat.
                w_mat[i, :] = prev_w
                equity[i] = equity[i - 1] if i > 0 else 1.0
                continue

            try:
                mu_vec, sigma_vec = _predict_mu_sigma_for_day(
                    model=model,
                    device=device,
                    symbols=syms,
                    features_std_by_symbol=prepared.features_std_full_by,
                    index_by_symbol=prepared.index_by,
                    day=pd.Timestamp(day),
                    seq_len=seq_len,
                )
            except Exception:
                if safety_enabled:
                    logger.exception("Phase2 v2 predict failed; forcing flat (%s)", str(day))
                    mu_vec = np.zeros(n_assets, dtype=float)
                    sigma_vec = np.ones(n_assets, dtype=float) * 0.02
                else:
                    raise
            # Registry gate: do not infer/act on ineligible symbols.
            if _eligible_mask is not None and _eligible_mask.size == mu_vec.size:
                try:
                    mu_vec = np.where(_eligible_mask, mu_vec, 0.0)
                    sigma_vec = np.where(_eligible_mask, sigma_vec, 1.0)
                except Exception:
                    pass

            mu_mat[i, :] = mu_vec
            sigma_mat[i, :] = sigma_vec

            # ─────────────────────────────────────────────────────────────────
            # Mamba calibration tracking:
            # 1. Add today's prediction to tracker.
            # 2. If matured predictions exist (i >= horizon), add realized returns.
            # 3. Get updated calibration score for learning gates.
            # ─────────────────────────────────────────────────────────────────
            try:
                mamba_calib_tracker.add_prediction(i, mu_vec, sigma_vec)
                
                # Add realized returns for matured predictions.
                matured_idx = i - horizon
                if matured_idx >= 0:
                    # fwd_ret_mat[matured_idx, :] contains realized H-day returns
                    # for predictions made at matured_idx.
                    realized_ret = fwd_ret_mat[matured_idx, :]
                    if np.any(np.isfinite(realized_ret)):
                        mamba_calib_tracker.add_realized(matured_idx, realized_ret)
                
                # ─────────────────────────────────────────────────────────────
                # LINEAR ALPHA COMBINER: Update with matured predictions
                # Uses same maturity gating as Mamba calibration.
                # Freeze updates when calibration < 0.55 (RED zone).
                # ─────────────────────────────────────────────────────────────
                if linear_state is not None and matured_idx >= 0:
                    try:
                        # Get calibration score for freeze decision
                        # (use quantile score as fallback during warmup)
                        _calib_for_freeze = 1.0
                        if mamba_calib_snapshot is not None and mamba_calib_snapshot.is_reliable:
                            _calib_for_freeze = float(mamba_calib_snapshot.calibration_overall)
                        elif role_ctx is not None:
                            try:
                                _day_ctx_freeze = role_ctx.get_for_day(pd.Timestamp(day))
                                _calib_for_freeze = float(_day_ctx_freeze.calibration_overall_score)
                            except Exception:
                                pass
                        
                        refit_report = linear_state.update_if_matured(
                            i, fwd_ret_mat, 
                            freeze=(_calib_for_freeze < 0.55)
                        )
                        
                        # Log refit if it occurred
                        if refit_report is not None and refit_report.get("status") == "fitted":
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.INFO,
                                code=EventCode.MAMBA_CALIB_UPDATED,  # Reuse event code
                                message=f"Linear combiner refit: R²={refit_report.get('r_squared', 0):.3f}",
                                payload={
                                    "linear_refit": True,
                                    "r_squared": float(refit_report.get("r_squared", 0.0)),
                                    "n_samples": int(refit_report.get("n_samples", 0)),
                                    "drift": float(refit_report.get("drift", 0.0)),
                                    "z_mamba_coef": float(refit_report.get("z_mamba_coef", 0.0)),
                                    "quantile_z_coef": float(refit_report.get("quantile_z_coef", 0.0)),
                                },
                            )
                    except Exception as e:
                        logger.debug("[phase2.linear] Maturity update failed: %s", e)
                
                # ─────────────────────────────────────────────────────────────
                # GAP #8: Accumulate per-symbol (mu, sigma, realized) for post-hoc stats
                # ─────────────────────────────────────────────────────────────
                if per_symbol_calib_enabled and matured_idx >= 0:
                    # Store prediction made at matured_idx paired with realized_ret
                    # We need mu/sigma from matured_idx, but those are in mu_mat/sigma_mat
                    if matured_idx < len(mu_mat) and matured_idx < len(sigma_mat):
                        _per_sym_mu_acc.append(mu_mat[matured_idx, :].copy())
                        _per_sym_sigma_acc.append(sigma_mat[matured_idx, :].copy())
                        _per_sym_realized_acc.append(fwd_ret_mat[matured_idx, :].copy())
                
                # Get current Mamba-based calibration (AUTHORITATIVE).
                # Optionally blend with quantile-based score as secondary diagnostic.
                quantile_score_for_blend: Optional[float] = None
                if role_ctx is not None:
                    try:
                        _day_ctx_calib = role_ctx.get_for_day(pd.Timestamp(day))
                        quantile_score_for_blend = float(_day_ctx_calib.calibration_overall_score)
                    except Exception:
                        pass
                
                mamba_calib_snapshot = mamba_calib_tracker.get_calibration(
                    quantile_score=quantile_score_for_blend
                )
                
                # EMIT: MAMBA_CALIB_UPDATED
                if mamba_calib_snapshot is not None:
                    event_bus.emit(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        severity=EventSeverity.INFO,
                        code=EventCode.MAMBA_CALIB_UPDATED,
                        message=f"Mamba calibration updated: score={mamba_calib_snapshot.calibration_overall:.3f}",
                        payload={
                            "calibration_overall": float(mamba_calib_snapshot.calibration_overall),
                            "is_reliable": bool(mamba_calib_snapshot.is_reliable),
                            "quantile_score_blend": float(quantile_score_for_blend) if quantile_score_for_blend else None,
                            "n_matured": int(getattr(mamba_calib_snapshot, "n_matured", 0)),
                        },
                    )
            except Exception as e:
                logger.debug("[phase2.mamba_calib] tracking error: %s", e)

            # Compute sigma_exec from sigma_train (sizing-only).
            sigma_vec = np.asarray(sigma_vec, dtype=float)
            sigma_vec = np.clip(sigma_vec, _sigma_exec_eps, 10.0)
            if i == 0:
                sigma_ema = sigma_vec.copy()
            else:
                sigma_ema = _sigma_exec_alpha * sigma_vec + (1.0 - _sigma_exec_alpha) * sigma_ema

            sigma_exec = np.zeros(n_assets, dtype=float)
            for j in range(n_assets):
                sigma_ema_hist[j].append(float(sigma_ema[j]))
                hist = np.asarray(list(sigma_ema_hist[j]), dtype=float)
                if hist.size >= 5:
                    lo = float(np.percentile(hist, 10.0))
                    hi = float(np.percentile(hist, 90.0))
                else:
                    lo = float(np.min(hist)) if hist.size else float(_sigma_exec_eps)
                    hi = float(np.max(hist)) if hist.size else float(10.0)
                if not np.isfinite(lo):
                    lo = float(_sigma_exec_eps)
                if not np.isfinite(hi):
                    hi = float(10.0)
                if hi < lo:
                    hi, lo = lo, hi
                sigma_exec[j] = float(np.clip(float(sigma_ema[j]), lo, hi))
            sigma_exec = np.clip(sigma_exec, _sigma_exec_eps, 10.0)
            
            # EMIT: SIGMA_CLIPPED (if any assets were clipped)
            n_clipped = int(np.sum((sigma_ema != sigma_exec)))
            if n_clipped > 0:
                # Find worst offenders (largest clip magnitude)
                clip_diff = np.abs(sigma_ema - sigma_exec)
                worst_idx = np.argsort(-clip_diff)[:5]
                worst_syms = [(str(syms[j]), float(clip_diff[j])) for j in worst_idx if clip_diff[j] > 1e-9]
                event_bus.emit(
                    date=str(pd.Timestamp(day))[:10],
                    day_idx=i,
                    severity=EventSeverity.INFO,
                    code=EventCode.SIGMA_CLIPPED,
                    message=f"Sigma clipped for {n_clipped} assets",
                    payload={
                        "n_clipped": n_clipped,
                        "worst_offenders": worst_syms,
                        "sigma_exec_mean": float(np.mean(sigma_exec)),
                        "sigma_exec_std": float(np.std(sigma_exec)),
                    },
                )

            # Optional split-stress execution conservatism.
            # Mode:
            # - shrink_z (default): apply after z is computed
            # - inflate_sigma: inflate sigma_exec before z is computed
            # - both: apply both
            day_ctx_for_overlays = None
            split_mode = str(os.environ.get("PORTFOLIO_SPLIT_STRESS_MODE", "shrink_z")).strip().lower()
            use_split_inflate = "inflate" in split_mode
            use_split_shrink = "shrink" in split_mode or split_mode == "both"
            split_penalty_k = float(os.environ.get("PORTFOLIO_SPLIT_STRESS_K", "0.3"))
            if role_ctx is not None and use_split_inflate:
                try:
                    day_ctx_for_overlays = role_ctx.get_for_day(pd.Timestamp(day))
                    idx_map = {str(s).upper(): int(j) for j, s in enumerate(day_ctx_for_overlays.symbols)}
                    split_stress = np.asarray(
                        [
                            day_ctx_for_overlays.split_stress[idx_map.get(str(s).upper(), -1)]
                            if idx_map.get(str(s).upper(), -1) >= 0
                            else 0.0
                            for s in syms
                        ],
                        dtype=float,
                    )
                    sigma_exec = sigma_exec * (1.0 + split_penalty_k * split_stress)
                    sigma_exec = np.clip(sigma_exec, _sigma_exec_eps, 10.0)
                except Exception:
                    day_ctx_for_overlays = None

            z = np.divide(mu_vec, sigma_exec + 1e-9)
            z = np.clip(z, -z_clip, z_clip)

            # Apply role-aware overlays (hygiene/risk/regime multipliers).
            if role_ctx is not None:
                try:
                    day_ctx = day_ctx_for_overlays or role_ctx.get_for_day(pd.Timestamp(day))
                    idx_map = {str(s).upper(): int(j) for j, s in enumerate(day_ctx.symbols)}
                    risk_scale = np.asarray([day_ctx.risk_scale[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 1.0 for s in syms], dtype=float)
                    regime_mult = np.asarray([day_ctx.regime_multiplier[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 1.0 for s in syms], dtype=float)
                    hygiene_ok = np.asarray([day_ctx.hygiene_ok[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else True for s in syms], dtype=bool)
                    # Split stress penalty: shrink z when split recently occurred
                    # z *= (1 - k * split_stress) where split_stress = max(flag, post_5d, post_20d, recency) * |log_ratio|
                    split_stress = np.asarray([day_ctx.split_stress[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 0.0 for s in syms], dtype=float)
                    split_discount = np.clip(1.0 - split_penalty_k * split_stress, 0.1, 1.0) if use_split_shrink else 1.0
                    
                    # EMIT: HYGIENE_VETO
                    n_hygiene_veto = int(np.sum(~hygiene_ok))
                    if n_hygiene_veto > 0:
                        veto_syms = [str(syms[j]) for j in range(n_assets) if not hygiene_ok[j]][:10]
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.WARNING,
                            code=EventCode.HYGIENE_VETO,
                            message=f"Hygiene veto for {n_hygiene_veto} assets",
                            payload={"n_vetoed": n_hygiene_veto, "vetoed_syms": veto_syms},
                        )
                    
                    # EMIT: RISK_SCALE_APPLIED (if non-trivial)
                    n_risk_scaled = int(np.sum(np.abs(risk_scale - 1.0) > 0.01))
                    if n_risk_scaled > 0:
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.INFO,
                            code=EventCode.RISK_SCALE_APPLIED,
                            message=f"Risk scale applied to {n_risk_scaled} assets",
                            payload={
                                "n_scaled": n_risk_scaled,
                                "risk_scale_mean": float(np.mean(risk_scale)),
                                "risk_scale_min": float(np.min(risk_scale)),
                            },
                        )
                    
                    # EMIT: REGIME_MULT_APPLIED (if non-trivial)
                    n_regime_mult = int(np.sum(np.abs(regime_mult - 1.0) > 0.01))
                    if n_regime_mult > 0:
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.INFO,
                            code=EventCode.REGIME_MULT_APPLIED,
                            message=f"Regime multiplier applied to {n_regime_mult} assets",
                            payload={
                                "n_scaled": n_regime_mult,
                                "regime_mult_mean": float(np.mean(regime_mult)),
                            },
                        )
                    
                    # EMIT: SPLIT_STRESS_APPLIED (if non-trivial)
                    n_split_stress = int(np.sum(split_stress > 0.01))
                    if n_split_stress > 0 and use_split_shrink:
                        stressed_syms = [(str(syms[j]), float(split_stress[j])) for j in range(n_assets) if split_stress[j] > 0.01][:5]
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.WARNING,
                            code=EventCode.SPLIT_STRESS_APPLIED,
                            message=f"Split stress applied to {n_split_stress} assets",
                            payload={
                                "mode": split_mode,
                                "n_stressed": n_split_stress,
                                "stressed_syms": stressed_syms,
                                "max_stress": float(np.max(split_stress)),
                            },
                        )
                    
                    # ─────────────────────────────────────────────────────────
                    # EVENT RISK OVERLAY: Explicit, controllable event management
                    # - earnings_next_1d: scale down z for names with earnings within 1 day
                    # - earnings_next_3d: lighter scale for 3-day window (optional)
                    # - macro_next_1d: scale down entire book for major macro events
                    # ─────────────────────────────────────────────────────────
                    event_risk_k_earnings = float(os.environ.get("PORTFOLIO_EVENT_RISK_K_EARNINGS", "0.5"))
                    event_risk_k_macro = float(os.environ.get("PORTFOLIO_EVENT_RISK_K_MACRO", "0.3"))
                    event_risk_enabled = str(os.environ.get("PORTFOLIO_EVENT_RISK_ENABLED", "1")).strip() in {"1", "true", "yes"}
                    
                    event_discount = np.ones(n_assets, dtype=float)
                    if event_risk_enabled:
                        # Extract event risk features from day_ctx
                        earnings_1d = np.asarray(
                            [day_ctx.event_risk_earnings_next_1d[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 0.0 for s in syms],
                            dtype=float
                        )
                        earnings_3d = np.asarray(
                            [day_ctx.event_risk_earnings_next_3d[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 0.0 for s in syms],
                            dtype=float
                        )
                        macro_1d = float(day_ctx.event_risk_macro_next_1d)
                        event_score = float(np.nanmean(day_ctx.event_risk_score)) if hasattr(day_ctx, "event_risk_score") else 0.0
                        
                        # Apply earnings overlay: z *= (1 - k_earn) for names with earnings_next_1d == 1
                        # Softer discount for 3-day window (25% of full penalty)
                        earnings_discount = 1.0 - event_risk_k_earnings * earnings_1d - 0.25 * event_risk_k_earnings * (earnings_3d - earnings_1d)
                        earnings_discount = np.clip(earnings_discount, 0.1, 1.0)
                        
                        # Apply macro overlay: global book scale-down
                        macro_discount = 1.0 - event_risk_k_macro * macro_1d
                        macro_discount = float(np.clip(macro_discount, 0.3, 1.0))

                        # Apply composite event risk score (0..1) as an additional mild discount
                        event_score_discount = float(np.clip(1.0 - 0.3 * event_score, 0.7, 1.0))
                        
                        event_discount = earnings_discount * macro_discount * event_score_discount
                        
                        # EMIT: EVENT_RISK_APPLIED
                        n_earnings_risk = int(np.sum(earnings_1d > 0.5))
                        if n_earnings_risk > 0 or macro_1d > 0.5 or event_score > 0.1:
                            earnings_syms = [str(syms[j]) for j in range(n_assets) if earnings_1d[j] > 0.5][:5]
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.EVENT_RISK_APPLIED,
                                message=f"Event risk applied: {n_earnings_risk} earnings, macro={macro_1d:.0f}, score={event_score:.2f}",
                                payload={
                                    "n_earnings_risk": n_earnings_risk,
                                    "earnings_syms": earnings_syms,
                                    "macro_1d": float(macro_1d),
                                    "event_score": float(event_score),
                                    "k_earnings": float(event_risk_k_earnings),
                                    "k_macro": float(event_risk_k_macro),
                                },
                            )
                    
                    z = np.where(hygiene_ok, z * risk_scale * regime_mult * split_discount * event_discount, 0.0)
                    # Store quantile_z and policy state features for later use
                    day_quantile_z = np.asarray([day_ctx.quantile_z[idx_map.get(str(s).upper(), -1)] if idx_map.get(str(s).upper(), -1) >= 0 else 0.0 for s in syms], dtype=float)
                    day_cboe_panic = float(day_ctx.cboe_panic_premium)
                    day_cboe_slope = float(day_ctx.cboe_term_slope)
                    day_cboe_vrp = float(day_ctx.cboe_vol_risk_premium_z)
                    # ─────────────────────────────────────────────────────────
                    # AUTHORITATIVE calibration: Mamba-based (μ, σ²) tracking.
                    # Quantile-based score is now SECONDARY (20-30% weight).
                    # ─────────────────────────────────────────────────────────
                    day_quantile_calib_score = float(day_ctx.calibration_overall_score)  # Secondary
                    if mamba_calib_snapshot is not None and mamba_calib_snapshot.is_reliable:
                        # Use Mamba calibration as primary (already blended with quantile).
                        day_calib_score = float(mamba_calib_snapshot.calibration_overall)
                    else:
                        # Fallback to quantile if Mamba not yet reliable (early warm-up).
                        day_calib_score = day_quantile_calib_score
                    day_online_trust = float(day_ctx.online_trust_score)
                except Exception:
                    day_quantile_z = np.zeros(n_assets, dtype=float)
                    day_cboe_panic = 0.0
                    day_cboe_slope = 0.0
                    day_cboe_vrp = 0.0
                    day_calib_score = 1.0
                    day_online_trust = 1.0
            else:
                day_quantile_z = np.zeros(n_assets, dtype=float)
                day_cboe_panic = 0.0
                day_cboe_slope = 0.0
                day_cboe_vrp = 0.0
                # Use Mamba calibration if available.
                if mamba_calib_snapshot is not None and mamba_calib_snapshot.is_reliable:
                    day_calib_score = float(mamba_calib_snapshot.calibration_overall)
                else:
                    day_calib_score = 1.0
                day_online_trust = 1.0

            # Policy controller (contextual bandit) selects knobs for this day.
            thr_base_day = float(thr_base)
            thr_bull_day = float(thr_bull)
            thr_bear_day = float(thr_bear)
            thr_crisis_day = float(thr_crisis)
            target_vol_day = float(target_vol)
            turnover_cap_day = float(turnover_cap)
            max_gross_day = float(max_gross)
            max_net_day = float(max_net)
            max_name_day = float(max_name)
            weight_smoothing_alpha_day = float(weight_smoothing_alpha)
            vol_scaler_day = float(cfg.get("phase2_policy_vol_scaler", 1.0))
            quantile_blend_weight_day = 0.0  # Default: pure Mamba
            confidence_floor_day = 0.0
            max_group_gross_day = float(max_group_gross)
            max_group_net_day = float(max_group_net)

            if policy is not None:
                try:
                    # Build sector map for v2 state vector
                    sector_map_for_policy: Optional[Dict[int, str]] = None
                    if sector_groups is not None:
                        sector_map_for_policy = {
                            j: str(sector_groups.get(str(s).upper(), "unknown"))
                            for j, s in enumerate(syms)
                        }
                    
                    # FIX Gap #3: Compute ADV vector for liquidity stress calculation
                    adv_vec_for_policy: Optional[np.ndarray] = None
                    if adv_usd_by_sym:
                        try:
                            adv_vec_for_policy = np.zeros(len(syms), dtype=float)
                            for j, sym in enumerate(syms):
                                adv_s = adv_usd_by_sym.get(str(sym).upper())
                                if adv_s is not None and not adv_s.empty:
                                    try:
                                        adv_val = float(pd.to_numeric(adv_s.reindex([pd.Timestamp(day)]).iloc[0], errors="coerce"))
                                        if np.isfinite(adv_val):
                                            adv_vec_for_policy[j] = adv_val
                                        else:
                                            adv_vec_for_policy[j] = 0.0
                                    except Exception:
                                        adv_vec_for_policy[j] = 0.0
                                else:
                                    adv_vec_for_policy[j] = 0.0
                            # Only use if we have valid data
                            if not np.all(adv_vec_for_policy == 0.0):
                                pass  # Keep the computed vector
                            else:
                                adv_vec_for_policy = None
                        except Exception:
                            adv_vec_for_policy = None
                    
                    state_vec = _policy_state_vector(
                        i=i,
                        horizon=int(horizon),
                        returns_df=returns_df,
                        net_ret=net_ret,
                        turnover=turnover,
                        costs=costs,
                        equity=equity,
                        mu_mat=mu_mat,
                        sigma_mat=sigma_mat,  # v2: pass sigma predictions
                        fwd_ret_mat=fwd_ret_mat,
                        market_regime=market_regime,
                        state_window=int(cfg.get("phase2_policy_state_window", 20)),
                        sharpe_window=int(cfg.get("phase2_policy_sharpe_window", 63)),
                        calib_window=int(cfg.get("phase2_policy_calibration_window", 63)),
                        # Portfolio family features from RoleAwareDayContext
                        cboe_panic_premium=day_cboe_panic,
                        cboe_term_slope=day_cboe_slope,
                        cboe_vol_risk_premium_z=day_cboe_vrp,
                        calibration_overall_score=day_calib_score,
                        online_trust_score=day_online_trust,
                        # v2: Additional context for expanded state
                        weights=prev_w,
                        sector_map=sector_map_for_policy,
                        adv_arr=adv_vec_for_policy,  # FIX Gap #3: Pass computed ADV vector
                    )
                    action_idx, action = policy.select_action(state_vec)
                    policy_action_history[i] = int(action_idx)
                    policy_state_history[i] = np.asarray(state_vec, dtype=float)
                    thr_base_day = float(action.base_threshold)
                    thr_bull_day = float(action.regime_mult_bull)
                    thr_bear_day = float(action.regime_mult_bear)
                    thr_crisis_day = float(action.regime_mult_crisis)
                    target_vol_day = float(action.target_vol)
                    turnover_cap_day = float(action.turnover_cap)
                    max_gross_day = float(action.max_gross)
                    max_net_day = float(action.max_net)
                    max_name_day = float(action.max_name)
                    weight_smoothing_alpha_day = float(action.weight_smoothing_alpha)
                    vol_scaler_day = float(action.vol_scaler)
                    quantile_blend_weight_day = float(action.quantile_blend_weight)
                    confidence_floor_day = float(action.confidence_floor)
                    # Risk-off leverage schedule (policy-controlled)
                    if np.isfinite(action.max_leverage_schedule) and action.max_leverage_schedule >= 0:
                        max_gross_day = float(max_gross_day) * float(action.max_leverage_schedule)
                    # Sector cap strength: scale group caps if provided
                    if np.isfinite(action.sector_cap_strength) and action.sector_cap_strength >= 0:
                        max_group_gross_day = float(max_group_gross_day) * float(action.sector_cap_strength)
                        max_group_net_day = float(max_group_net_day) * float(action.sector_cap_strength)
                    
                    # EMIT: POLICY_ACTION
                    event_bus.emit(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        severity=EventSeverity.INFO,
                        code=EventCode.POLICY_ACTION,
                        message=f"Policy selected action {action_idx}",
                        payload={
                            "action_idx": int(action_idx),
                            "base_threshold": float(thr_base_day),
                            "target_vol": float(target_vol_day),
                            "max_gross": float(max_gross_day),
                            "max_net": float(max_net_day),
                            "turnover_cap": float(turnover_cap_day),
                            "quantile_blend_weight": float(quantile_blend_weight_day),
                            "confidence_floor": float(confidence_floor_day),
                            "sector_cap_strength": float(action.sector_cap_strength),
                            "max_leverage_schedule": float(action.max_leverage_schedule),
                        },
                    )
                except Exception:
                    pass

            # ─────────────────────────────────────────────────────────────────
            # LINEAR ALPHA COMBINER or QUANTILE BLENDING
            # When linear_model_enabled AND model is ready, use learned z_lin blending.
            # Otherwise (disabled, warmup, or failure), fall back to static quantile blend.
            # These are MUTUALLY EXCLUSIVE to prevent double-blending.
            # ─────────────────────────────────────────────────────────────────
            linear_blend_applied = False
            
            if linear_state is not None:
                # Build feature matrix for linear combiner
                try:
                    # Compute corr_hhi for linear builder
                    # FIX Gap C: Convert cov → corr properly and compute HHI
                    corr_hhi_for_linear = 0.0
                    try:
                        diag_vals = np.sqrt(np.diag(cov))
                        diag_inv = np.where(diag_vals > 1e-8, 1.0 / diag_vals, 0.0)
                        corr_mat_linear = (cov * diag_inv[:, None]) * diag_inv[None, :]
                        from src.portfolio.policy_controller import compute_correlation_hhi
                        corr_hhi_for_linear = float(compute_correlation_hhi(corr_mat_linear))
                    except Exception:
                        pass
                    
                    X_day = linear_state.feature_builder.build_day(
                        i=i,
                        syms=syms,
                        z_mamba=z,
                        sigma_exec=sigma_exec,
                        day_ctx=day_ctx if role_ctx is not None else None,
                        day_calib_score=day_calib_score,
                        day_online_trust=day_online_trust,
                        day_cboe_panic=day_cboe_panic,
                        day_cboe_slope=day_cboe_slope,
                        day_cboe_vrp=day_cboe_vrp,
                        equity=equity,
                        turnover=turnover,
                        costs=costs,
                        returns_df=returns_df,
                        corr_hhi=corr_hhi_for_linear,
                    )
                    
                    # Observe this day's features for later maturity processing
                    linear_state.observe_day(i, X_day, sigma_exec)
                    
                    # Compute quality-adaptive blend weight
                    # Start with policy knob, then apply quality gates
                    base_weight = float(np.clip(quantile_blend_weight_day, 0.0, linear_blend_max))
                    
                    if linear_state.is_ready() and base_weight > 0:
                        z_lin = linear_state.predict_z(X_day)
                        
                        # Apply adaptive quality gating
                        w_L, adaptive_diag = linear_state.compute_adaptive_blend_weight(
                            base_weight=base_weight,
                            z_lin=z_lin,
                            z_mamba=z,
                            r_squared_threshold=float(cfg.get("linear_r2_threshold", 0.05)),
                            drift_threshold=float(cfg.get("linear_drift_threshold", 0.15)),
                            corr_threshold=float(cfg.get("linear_corr_threshold", 0.3)),
                            sign_disagree_threshold=float(cfg.get("linear_sign_disagree_threshold", 0.4)),
                            min_stable_updates=int(cfg.get("linear_min_stable_updates", 3)),
                        )
                        
                        if w_L > 1e-6:
                            z_before_blend = z.copy()
                            z = (1.0 - w_L) * z + w_L * z_lin
                            linear_blend_applied = True
                            
                            # HARD hygiene veto must remain absolute after blend
                            if 'hygiene_ok' in dir() and isinstance(hygiene_ok, np.ndarray):
                                z = np.where(hygiene_ok, z, 0.0)
                            
                            # EMIT: LINEAR_BLEND_APPLIED with adaptive diagnostics
                            try:
                                diag = linear_state.get_daily_diagnostics(z_lin, z_before_blend)
                                event_bus.emit(
                                    date=str(pd.Timestamp(day))[:10],
                                    day_idx=i,
                                    severity=EventSeverity.INFO,
                                    code=EventCode.LINEAR_BLEND_APPLIED,
                                    message=f"Linear blend: base={base_weight:.2f} adaptive={w_L:.2f}",
                                    payload={
                                        "base_weight": float(base_weight),
                                        "adaptive_weight": float(w_L),
                                        "blend_type": "linear_adaptive",
                                        "corr_z_lin_z_mamba": float(diag.get("corr_z_lin_z_mamba", 0.0)),
                                        "sign_disagreement": float(diag.get("sign_disagreement_frac", 0.0)),
                                        "n_updates": int(diag.get("n_updates", 0)),
                                        "quality_penalties": adaptive_diag.get("penalties_applied", []),
                                        "r_squared": float(adaptive_diag.get("current_r_squared", 0.0)),
                                        "drift": float(adaptive_diag.get("current_drift", 0.0)),
                                    },
                                )
                            except Exception:
                                pass
                    # else: linear not ready or w_L=0, fall through to quantile blending
                except Exception as e:
                    logger.debug("[phase2.linear] Feature build failed: %s; falling back to quantile", e)
            
            # Static quantile blending: ONLY if linear blend was NOT applied
            # This ensures mutually exclusive blending (no double application)
            if not linear_blend_applied:
                # Original quantile blending path
                # z = (1 - w_q) * z_mamba + w_q * z_quantile
                if quantile_blend_weight_day > 0 and np.any(np.abs(day_quantile_z) > 1e-9):
                    w_q = float(np.clip(quantile_blend_weight_day, 0.0, 1.0))
                    z = (1.0 - w_q) * z + w_q * day_quantile_z
                    
                    # EMIT: QUANTILE_BLEND_APPLIED
                    event_bus.emit(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        severity=EventSeverity.INFO,
                        code=EventCode.QUANTILE_BLEND_APPLIED,
                        message=f"Quantile blend applied with weight {w_q:.2f}",
                        payload={"weight": float(w_q), "blend_type": "quantile"},
                    )

            if np.isfinite(vol_scaler_day) and vol_scaler_day > 0:
                z = z * float(vol_scaler_day)

            # Policy confidence floor: scale z if calibration is below threshold
            if np.isfinite(confidence_floor_day) and confidence_floor_day > 0:
                if day_calib_score < confidence_floor_day:
                    conf_scale = float(np.clip(day_calib_score / confidence_floor_day, 0.0, 1.0))
                    z = z * conf_scale

            if safety_enabled and (not np.all(np.isfinite(z))):
                # Data/model integrity issue: go flat.
                logger.error("Phase2 v2 non-finite z detected; forcing flat (%s)", str(day))
                
                # EMIT: NAN_INF_KILLSWITCH
                n_nan = int(np.sum(np.isnan(z)))
                n_inf = int(np.sum(np.isinf(z)))
                event_bus.emit(
                    date=str(pd.Timestamp(day))[:10],
                    day_idx=i,
                    severity=EventSeverity.ERROR,
                    code=EventCode.NAN_INF_KILLSWITCH,
                    message=f"NaN/Inf detected in z: {n_nan} NaN, {n_inf} Inf",
                    payload={"n_nan": n_nan, "n_inf": n_inf, "z_mean": float(np.nanmean(z))},
                )
                
                w_target_flat = np.zeros(n_assets, dtype=float)
                w_mat[i, :] = w_target_flat
                r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                pnl = float(exec_w @ r_vec)
                
                # FIX Gap #2: ALWAYS advance the execution queue
                if trade_delay_sessions > 1:
                    _queue.append(w_target_flat.copy())
                    w_next_hold = np.asarray(_queue.popleft(), dtype=float)
                else:
                    w_next_hold = w_target_flat.copy()
                
                # Track both intent and exec turnover separately
                tval_intent = float(np.sum(np.abs(w_target_flat - prev_w)))
                tval_exec = float(np.sum(np.abs(w_next_hold - exec_w)))
                turnover_intent[i] = tval_intent
                turnover_exec[i] = tval_exec
                turnover[i] = tval_exec
                
                # Costs computed on EXECUTED turnover
                cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
                costs[i] = cost
                pnl_net = float(pnl - cost)
                net_ret[i] = pnl_net
                equity[i] = float((equity[i - 1] if i > 0 else 1.0) * (1.0 + pnl_net))
                equity_peak = float(max(equity_peak, equity[i]))
                cov = _ewma_cov_update(cov, r_vec, cov_lam)
                
                # Update state for next iteration
                prev_exec_w = exec_w.copy()
                exec_w = w_next_hold
                prev_w = w_target_flat
                if flat_cooldown_sessions > 0:
                    flat_until_idx = int(min(len(union_oos_index) - 1, i + int(flat_cooldown_sessions)))
                continue

            # Regime-aware thresholding (zero out weak signals).
            z_thr = np.zeros_like(z)
            for j, sym in enumerate(syms):
                reg = int(regimes_by_sym[sym].reindex([pd.Timestamp(day)]).fillna(0).iloc[0])
                thr = _regime_threshold_for_label(reg, thr_base_day, thr_bull_day, thr_bear_day, thr_crisis_day)
                z_thr[j] = z[j] if abs(z[j]) >= float(thr) else 0.0

            if diag_enabled:
                try:
                    active_frac[i] = float(np.mean(np.abs(z_thr) > 0.0))
                except Exception:
                    active_frac[i] = 0.0

            # ─────────────────────────────────────────────────────────────────
            # Z-EXPLAINER: Collect data for interpretability
            # ─────────────────────────────────────────────────────────────────
            if z_explainer_enabled and z_explainer_buffer is not None:
                try:
                    from src.stage_b_stateful.z_explainer import create_explainer_day
                    
                    # Compute portfolio state for global features
                    eq_prev = float(equity[i - 1]) if i > 0 else 1.0
                    peak_val = float(max(equity_peak, eq_prev))
                    dd_for_explainer = float(1.0 - (eq_prev / peak_val)) if peak_val > 0 else 0.0
                    
                    # Realized vol from recent net returns
                    rv_lookback = min(20, i)
                    rv_for_explainer = float(np.std(net_ret[max(0, i - rv_lookback):i]) * np.sqrt(252)) if rv_lookback > 1 else 0.0
                    
                    # Correlation HHI from covariance matrix
                    # FIX Gap C: Convert cov → corr properly (not np.corrcoef which computes corr of rows)
                    # corr = D^-1/2 * cov * D^-1/2 where D = diag(cov)
                    try:
                        diag_vals = np.sqrt(np.diag(cov))
                        diag_inv = np.where(diag_vals > 1e-8, 1.0 / diag_vals, 0.0)
                        corr_mat = (cov * diag_inv[:, None]) * diag_inv[None, :]
                        # Use eigenvalue-based HHI (proper correlation concentration metric)
                        from src.portfolio.policy_controller import compute_correlation_hhi
                        corr_hhi_val = float(compute_correlation_hhi(corr_mat))
                    except Exception:
                        corr_hhi_val = 0.0
                    
                    # Previous turnover and cost
                    turn_prev = float(turnover[i - 1]) if i > 0 else 0.0
                    cost_prev_val = float(costs[i - 1]) if i > 0 else 0.0
                    
                    # Get overlay values (may have been set earlier in the loop)
                    _risk_scale_arr = risk_scale if 'risk_scale' in dir() and isinstance(risk_scale, np.ndarray) else np.ones(n_assets)
                    _regime_mult_arr = regime_mult if 'regime_mult' in dir() and isinstance(regime_mult, np.ndarray) else np.ones(n_assets)
                    _hygiene_ok_arr = hygiene_ok if 'hygiene_ok' in dir() and isinstance(hygiene_ok, np.ndarray) else np.ones(n_assets, dtype=bool)
                    _split_stress_arr = split_stress if 'split_stress' in dir() and isinstance(split_stress, np.ndarray) else np.zeros(n_assets)
                    _split_discount_arr = split_discount if 'split_discount' in dir() and isinstance(split_discount, np.ndarray) else np.ones(n_assets)
                    _quantile_z_arr = day_quantile_z if 'day_quantile_z' in dir() and isinstance(day_quantile_z, np.ndarray) else np.zeros(n_assets)
                    
                    # z_pre_overlay = mu / sigma_exec (before overlays)
                    z_pre_overlay = np.divide(mu_vec, sigma_exec + 1e-9)
                    z_pre_overlay = np.clip(z_pre_overlay, -z_clip, z_clip)
                    
                    # sigma was clipped (approximation: check if near clip bounds)
                    sigma_was_clipped = ((sigma_exec <= _sigma_exec_eps * 1.1) | (sigma_exec >= 9.9)).astype(float)
                    
                    # z_post_overlay (after overlays, before blend)
                    z_post_overlay = np.where(_hygiene_ok_arr, z_pre_overlay * _risk_scale_arr * _regime_mult_arr * _split_discount_arr, 0.0)
                    
                    # z_post_blend (after quantile blend)
                    w_q_val = float(quantile_blend_weight_day) if 'quantile_blend_weight_day' in dir() else 0.0
                    if w_q_val > 0 and np.any(np.abs(_quantile_z_arr) > 1e-9):
                        z_post_blend = (1.0 - w_q_val) * z_post_overlay + w_q_val * _quantile_z_arr
                    else:
                        z_post_blend = z_post_overlay.copy()
                    
                    # z_post_scale (after vol_scaler)
                    vol_scaler_val = float(vol_scaler_day) if 'vol_scaler_day' in dir() else 1.0
                    z_post_scale = z_post_blend * vol_scaler_val
                    
                    explainer_day = create_explainer_day(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        symbols=list(syms),
                        mu_raw=mu_vec,
                        sigma_exec=sigma_exec,
                        sigma_was_clipped=sigma_was_clipped,
                        z_pre_overlay=z_pre_overlay,
                        risk_scale=_risk_scale_arr,
                        regime_multiplier=_regime_mult_arr,
                        hygiene_ok=_hygiene_ok_arr.astype(float),
                        split_stress=_split_stress_arr,
                        split_discount=_split_discount_arr if isinstance(_split_discount_arr, np.ndarray) else np.full(n_assets, _split_discount_arr),
                        quantile_z=_quantile_z_arr,
                        quantile_blend_weight=w_q_val,
                        z_post_overlay=z_post_overlay,
                        z_post_blend=z_post_blend,
                        z_post_scale=z_post_scale,
                        z_final=z_thr,  # TARGET: post-threshold z
                        day_calib_score=day_calib_score if 'day_calib_score' in dir() else 1.0,
                        online_trust_score=day_online_trust if 'day_online_trust' in dir() else 1.0,
                        cboe_panic=day_cboe_panic if 'day_cboe_panic' in dir() else 0.0,
                        cboe_slope=day_cboe_slope if 'day_cboe_slope' in dir() else 0.0,
                        cboe_vrp=day_cboe_vrp if 'day_cboe_vrp' in dir() else 0.0,
                        vol_scaler=vol_scaler_val,
                        portfolio_drawdown=dd_for_explainer,
                        portfolio_realized_vol=rv_for_explainer,
                        corr_hhi=corr_hhi_val,
                        turnover_prev=turn_prev,
                        cost_prev=cost_prev_val,
                        equity_level=eq_prev,
                        market_regime=int(market_regime),
                    )
                    z_explainer_buffer.add(explainer_day)
                    
                    # ─────────────────────────────────────────────────────────
                    # STEP 7: VALIDATION COMPANION (sanity check z_final vs fwd returns)
                    # ─────────────────────────────────────────────────────────
                    if z_validation_companion is not None:
                        try:
                            # Use realized returns from H days ago
                            matured_idx = i - horizon
                            if matured_idx >= 0 and matured_idx < len(fwd_ret_mat):
                                realized_ret = fwd_ret_mat[matured_idx, :]
                                
                                # We need the z_final from matured_idx (the prediction day)
                                # stored in z_mat if it's available
                                if matured_idx < len(z_mat):
                                    z_final_matured = z_mat[matured_idx, :]
                                    
                                    # Build features for matured predictions
                                    # We use today's global context but matured z_final
                                    # This is approximate but captures regime/stress context
                                    for j in range(n_assets):
                                        if np.isfinite(realized_ret[j]) and np.isfinite(z_final_matured[j]):
                                            # Build feature vector matching explainer_day structure
                                            per_sym_vec = np.array([
                                                float(z_mat[matured_idx, j]) if matured_idx < len(z_mat) else 0.0,  # z_pre_overlay approx
                                                float(mu_mat[matured_idx, j]) if matured_idx < len(mu_mat) else 0.0,
                                                float(sigma_mat[matured_idx, j]) if matured_idx < len(sigma_mat) else 0.0,
                                                0.0,  # sigma_was_clipped
                                                1.0,  # risk_scale
                                                1.0,  # regime_multiplier
                                                1.0,  # hygiene_ok
                                                0.0,  # split_stress
                                                1.0,  # split_discount
                                                0.0,  # quantile_z
                                                0.0,  # quantile_blend_weight
                                            ], dtype=np.float32)
                                            
                                            global_vec = np.array([
                                                day_calib_score if 'day_calib_score' in dir() else 1.0,
                                                day_online_trust if 'day_online_trust' in dir() else 1.0,
                                                day_cboe_panic if 'day_cboe_panic' in dir() else 0.0,
                                                day_cboe_slope if 'day_cboe_slope' in dir() else 0.0,
                                                day_cboe_vrp if 'day_cboe_vrp' in dir() else 0.0,
                                                vol_scaler_val,
                                                dd_for_explainer,
                                                rv_for_explainer,
                                                corr_hhi_val,
                                                turn_prev,
                                                cost_prev_val,
                                                eq_prev,
                                                float(market_regime),
                                            ], dtype=np.float32)
                                            
                                            features = np.concatenate([per_sym_vec, global_vec])
                                            z_validation_companion.add_sample(
                                                features=features,
                                                z_final=float(z_final_matured[j]),
                                                fwd_return=float(realized_ret[j]),
                                                date=str(pd.Timestamp(day))[:10],
                                                symbol=syms[j],
                                            )
                            
                            # Refit if it's time
                            if z_validation_companion.should_refit(i):
                                z_validation_companion.refit(day_idx=i)
                        except Exception as e:
                            logger.debug("[phase2.z_validation] update error: %s", e)
                    
                    # ─────────────────────────────────────────────────────────
                    # ROLLING SURROGATE: Add to buffer, refit, explain
                    # ─────────────────────────────────────────────────────────
                    if z_surrogate is not None:
                        # Add day to rolling buffer
                        z_surrogate.add_day(explainer_day)
                        
                        # Refit if it's time (every K days)
                        if z_surrogate.should_refit(i):
                            z_surrogate.refit(day_idx=i)
                        
                        # Generate explanation (purely observational)
                        if z_surrogate.is_fitted:
                            # Determine top traded symbols from current weights
                            traded_syms = None
                            if 'prev_w' in dir() and prev_w is not None:
                                top_weight_idx = np.argsort(np.abs(prev_w))[::-1][:z_surrogate.top_n_symbols]
                                traded_syms = [syms[j] for j in top_weight_idx if np.abs(prev_w[j]) > 1e-6]
                            
                            daily_explanation = z_surrogate.explain_day(
                                day=explainer_day,
                                traded_symbols=traded_syms,
                            )
                            
                            # Log explanation (minimum daily output)
                            explanation_log = {
                                "date": daily_explanation.date,
                                "day_idx": daily_explanation.day_idx,
                                "fidelity": {
                                    "correlation": float(daily_explanation.fidelity.correlation),
                                    "directional_agreement": float(daily_explanation.fidelity.directional_agreement),
                                    "r2_score": float(daily_explanation.fidelity.r2_score),
                                    "is_reliable": bool(daily_explanation.fidelity.is_reliable),
                                },
                                "explanation_suppressed": daily_explanation.explanation_suppressed,
                                "suppression_reason": daily_explanation.suppression_reason,
                            }
                            
                            if not daily_explanation.explanation_suppressed:
                                # Add global drivers
                                explanation_log["global_drivers"] = [
                                    {"feature": f, "mean_abs_contrib": round(v, 5)}
                                    for f, v in daily_explanation.global_drivers[:10]
                                ]
                                
                                # Add top symbol contributions
                                explanation_log["top_symbols"] = []
                                for sc in daily_explanation.symbol_contributions[:5]:
                                    sym_log = {
                                        "symbol": sc.symbol,
                                        "z_final": round(sc.z_final, 4),
                                        "z_predicted": round(sc.z_predicted, 4),
                                        "top_positive": [
                                            {"feature": f, "value": round(v, 4), "beta": round(b, 4), "contrib": round(c, 4)}
                                            for f, v, b, c in sc.top_positive[:3]
                                        ],
                                        "top_negative": [
                                            {"feature": f, "value": round(v, 4), "beta": round(b, 4), "contrib": round(c, 4)}
                                            for f, v, b, c in sc.top_negative[:3]
                                        ],
                                    }
                                    explanation_log["top_symbols"].append(sym_log)
                                
                                # ─────────────────────────────────────────────────
                                # STEP 6: Generate human-readable reason codes
                                # ─────────────────────────────────────────────────
                                try:
                                    from src.stage_b_stateful.z_explainer import (
                                        generate_portfolio_reason_code,
                                        generate_symbol_reason_codes,
                                    )
                                    
                                    # Compute portfolio exposures
                                    net_exp = float(np.sum(prev_w)) if 'prev_w' in dir() and prev_w is not None else 0.0
                                    gross_exp = float(np.sum(np.abs(prev_w))) if 'prev_w' in dir() and prev_w is not None else 0.0
                                    
                                    # Portfolio reason code
                                    portfolio_rc = generate_portfolio_reason_code(
                                        explanation=daily_explanation,
                                        net_exposure=net_exp,
                                        gross_exposure=gross_exp,
                                    )
                                    
                                    # Symbol reason codes
                                    symbol_rcs = generate_symbol_reason_codes(
                                        explanation=daily_explanation,
                                        max_symbols=5,
                                    )
                                    
                                    # Add reason codes to log
                                    explanation_log["reason_codes"] = {
                                        "portfolio": {
                                            "direction": portfolio_rc.direction,
                                            "drivers": portfolio_rc.drivers[:3],
                                            "narrative": portfolio_rc.to_narrative(),
                                            "confidence": portfolio_rc.confidence,
                                        },
                                        "symbols": [
                                            {
                                                "symbol": rc.symbol,
                                                "direction": rc.direction,
                                                "drivers": rc.drivers[:3],
                                                "narrative": rc.to_narrative(),
                                                "confidence": rc.confidence,
                                            }
                                            for rc in symbol_rcs[:5]
                                        ],
                                    }
                                except Exception as e:
                                    logger.debug("[phase2.z_explainer] reason_codes error: %s", e)
                                
                                # ─────────────────────────────────────────────────
                                # STEP 7: Add validation companion metrics to log
                                # ─────────────────────────────────────────────────
                                if z_validation_companion is not None and z_validation_companion.is_fitted:
                                    try:
                                        vf = z_validation_companion.last_fidelity
                                        explanation_log["validation_companion"] = {
                                            "accuracy": round(vf.accuracy, 4),
                                            "auc_roc": round(vf.auc_roc, 4),
                                            "precision_long": round(vf.precision_long, 4),
                                            "precision_short": round(vf.precision_short, 4),
                                            "is_reliable": vf.is_reliable,
                                            "z_final_coefficient": round(z_validation_companion.get_z_final_coefficient(), 4),
                                            "z_final_is_predictive": abs(z_validation_companion.get_z_final_coefficient()) > 0.1,
                                        }
                                    except Exception as e:
                                        logger.debug("[phase2.z_explainer] validation_companion log error: %s", e)
                            
                            z_explainer_logs.append(explanation_log)
                            
                            # Log to console (DEBUG level to avoid spam)
                            if not daily_explanation.explanation_suppressed:
                                logger.debug(
                                    "[phase2.z_explainer] Day %d: corr=%.3f dir_agree=%.1f%% | Top driver: %s",
                                    i, daily_explanation.fidelity.correlation,
                                    daily_explanation.fidelity.directional_agreement * 100,
                                    daily_explanation.global_drivers[0][0] if daily_explanation.global_drivers else "N/A"
                                )
                            else:
                                logger.debug(
                                    "[phase2.z_explainer] Day %d: SUPPRESSED - %s",
                                    i, daily_explanation.suppression_reason
                                )
                except Exception as e:
                    logger.debug("[phase2.z_explainer] collection error: %s", e)

            # ─────────────────────────────────────────────────────────────────
            # POINT 2: OUTPUT SANITY CHECKS (after z finalized, before optimizer)
            # Check sigma collapse, z saturation, mu constant, sign flip rate
            # ─────────────────────────────────────────────────────────────────
            output_sanity_fallback = False
            if risk_latch_enabled and risk_latch is not None:
                try:
                    output_sanity_mode = risk_latch.check_output_sanity(
                        mu_vec=mu_vec,
                        sigma_vec=sigma_vec,
                        z_vec=z_thr,
                        z_prev=_prev_z_for_flip,
                        z_clip=z_clip,
                    )
                    
                    # Update prev z for next iteration
                    _prev_z_for_flip = z_thr.copy()
                    
                    # If triggered, resolve and handle
                    if output_sanity_mode >= RiskLatchMode.SAFE_FALLBACK:
                        output_state = risk_latch.resolve()
                        
                        # Check if we should flatten OR hold fallback
                        if output_state.should_flatten() or output_state.should_hold_fallback():
                            output_sanity_fallback = True
                            
                            # Determine action taken for logging
                            if output_state.should_hold_fallback():
                                action_taken = "SAFE_FALLBACK_HOLD"
                                exposure_scale_log = 1.0 if output_state.last_good_weights is not None else 0.0
                            else:
                                action_taken = "FLATTEN"
                                exposure_scale_log = 0.0
                            
                            # ─────────────────────────────────────────────────────
                            # LEDGER: Log output sanity fail event
                            # ─────────────────────────────────────────────────────
                            if risk_events_ledger is not None:
                                try:
                                    risk_events_ledger.add_event(
                                        date=str(pd.Timestamp(day))[:10],
                                        day_idx=i,
                                        event_type="OUTPUT_SANITY_FAIL",
                                        threshold="sigma stable AND z < clip AND mu varying",
                                        threshold_crossed=True,
                                        action_taken=action_taken,
                                        exposure_scale=exposure_scale_log,
                                        trigger_reason=(
                                            output_state.primary_reason.name
                                            if output_state.primary_reason
                                            else "unknown"
                                        ),
                                        details={
                                            "sigma_min": float(np.min(sigma_vec)) if sigma_vec is not None else None,
                                            "sigma_max": float(np.max(sigma_vec)) if sigma_vec is not None else None,
                                            "z_abs_max": float(np.max(np.abs(z_thr))) if z_thr is not None else None,
                                            "mu_std": float(np.std(mu_vec)) if mu_vec is not None else None,
                                        },
                                    )
                                except Exception as le:
                                    logger.debug("[phase2.ledger] Failed to log event: %s", le)
                            
                            # EMIT: RISK_LATCH event
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.RISK_LATCH_SAFE_FALLBACK,
                                message=f"Output sanity {output_state.mode.name}: {output_state.primary_reason.name if output_state.primary_reason else 'unknown'}",
                                payload=output_state.to_dict(),
                            )
                            
                            # Apply safe fallback or flatten
                            w = output_state.get_effective_weights(
                                np.zeros(n_assets, dtype=float),
                                n_assets,
                            )
                            w_mat[i, :] = w
                            
                            # Handle PnL and execution queue
                            r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                            pnl = float(exec_w @ r_vec)
                            
                            if trade_delay_sessions > 1:
                                _queue.append(w.copy())
                                w_next = np.asarray(_queue.popleft(), dtype=float)
                            else:
                                w_next = w.copy()
                            
                            tval_intent = float(np.sum(np.abs(w - prev_w)))
                            tval_exec = float(np.sum(np.abs(w_next - exec_w)))
                            turnover_intent[i] = tval_intent
                            turnover_exec[i] = tval_exec
                            turnover[i] = tval_exec
                            
                            cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
                            costs[i] = cost
                            pnl_net = float(pnl - cost)
                            net_ret[i] = pnl_net
                            equity[i] = float((equity[i - 1] if i > 0 else 1.0) * (1.0 + pnl_net))
                            equity_peak = float(max(equity_peak, equity[i]))
                            cov = _ewma_cov_update(cov, r_vec, cov_lam)
                            
                            prev_exec_w = exec_w.copy()
                            exec_w = w_next
                            prev_w = w
                            
                            if risk_latch is not None:
                                risk_latch.end_session(None)
                            continue
                        
                        elif output_sanity_mode == RiskLatchMode.THROTTLE:
                            # Apply throttle scale to z
                            output_state = risk_latch.resolve()
                            throttle_scale = float(output_state.exposure_scale)
                            z_thr = z_thr * throttle_scale
                            
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.INFO,
                                code=EventCode.RISK_LATCH_THROTTLE,
                                message=f"Output sanity THROTTLE: scale={throttle_scale:.2f}",
                                payload={"scale": throttle_scale, "reason": output_state.primary_reason.name if output_state.primary_reason else "unknown"},
                            )
                except Exception as e:
                    logger.debug("[phase2.risk_latch] Output sanity check error: %s", e)

            # ─────────────────────────────────────────────────────────────────
            # Workstream-7: Robust Portfolio Optimizer path
            # ─────────────────────────────────────────────────────────────────
            if use_robust_optimizer and robust_optimizer is not None:
                try:
                    from src.portfolio.robust_optimizer import (
                        PredictionBundle,
                        CovarianceModel,
                        PortfolioConstraints,
                        build_sector_adjacency,
                    )
                    
                    # Get sigma reliability from Mamba calibration tracker
                    sigma_reliability = np.ones(n_assets, dtype=float)
                    if mamba_calib_snapshot is not None:
                        # Use overall calibration score as reliability for all assets
                        # (Future: per-symbol reliability from multi-task calibration)
                        calib_score = float(mamba_calib_snapshot.calibration_overall)
                        calib_score = float(np.clip(calib_score, 0.0, 1.0))
                        sigma_reliability = np.full(n_assets, calib_score, dtype=float)
                    
                    # FIX Gap #4: Robust optimizer expects mu as RETURN predictions, not z-scores
                    # Use thresholded mu (apply same thresholding to preserve signal sparsity)
                    # mu_vec is the raw return prediction from Mamba, z_thr is dimensionless
                    mu_for_robust = np.zeros_like(mu_vec)
                    for j in range(n_assets):
                        # Apply same threshold logic but on mu scale
                        if abs(z_thr[j]) > 0:  # Signal passed threshold
                            mu_for_robust[j] = mu_vec[j]  # Use actual return prediction
                        # else: leave as 0 (filtered out)
                    
                    # Build prediction bundle from current day's Mamba output
                    pred_bundle = PredictionBundle(
                        mu=mu_for_robust,  # FIX: Use thresholded mu (return scale), not z_thr
                        sigma=sigma_vec,  # Model-predicted sigma (from Mamba)
                        sigma_reliability=sigma_reliability,
                        p_up=None,
                        rho=None,
                    )
                    
                    # Build covariance model
                    # Use sector map for graph shrinkage if available
                    adjacency = None
                    sector_ids = None
                    if robust_use_graph_shrinkage and group_by_symbol:
                        try:
                            # Convert group_by_symbol to sector_ids array
                            sector_labels = [group_by_symbol.get(s, 0) for s in syms]
                            sector_ids = np.array(sector_labels, dtype=int)
                            adjacency = build_sector_adjacency(sector_ids)
                        except Exception:
                            adjacency = None
                    
                    cov_model = CovarianceModel(
                        cov=cov.copy(),
                        sector_ids=sector_ids,
                        adjacency=adjacency,
                    )
                    
                    # Build constraints from config
                    # Compute current drawdown for throttling
                    eq_prev_for_dd = float(equity[i - 1]) if i > 0 else 1.0
                    peak_for_dd = float(max(equity_peak, eq_prev_for_dd))
                    current_dd = float(1.0 - (eq_prev_for_dd / peak_for_dd)) if peak_for_dd > 0 else 0.0
                    
                    portfolio_constraints = PortfolioConstraints(
                        max_gross=float(max_gross_day),
                        max_net=float(max_net_day),
                        max_name=float(max_name_day),
                        min_name=-float(max_name_day),
                        target_vol_annual=float(target_vol_day),
                        max_vol_annual=float(target_vol_day) * 2.0,
                        cvar_alpha=float(robust_cvar_alpha),
                        cvar_limit=robust_cvar_limit_f,
                        turnover_cap=float(turnover_cap_day) if turnover_cap_day > 0 else None,
                        turnover_penalty=float(robust_lambda_turnover),
                        beta_neutral=bool(beta_neutral),
                        max_beta_exposure=float(beta_cap) if beta_cap > 0 else 0.1,
                        max_sector_exposure=float(max_group_gross) if max_group_gross > 0 else 0.3,
                        current_drawdown=float(current_dd),
                        drawdown_throttle=float(dd_throttle_1) if dd_throttle_1 > 0 else 0.10,
                        drawdown_gross_mult=float(dd_gross_mult_1) if dd_gross_mult_1 > 0 else 0.5,
                        mu_sigma_cap=float(robust_mu_sigma_cap),
                        reliability_min=float(robust_reliability_min),
                    )
                    
                    # Get historical returns for CVaR estimation
                    hist_returns = None
                    if robust_cvar_constraint and i >= 60:
                        hist_returns = returns_df.iloc[max(0, i - 252):i].to_numpy(dtype=float)
                    
                    # Run optimizer
                    opt_result = robust_optimizer.optimize(
                        predictions=pred_bundle,
                        cov_model=cov_model,
                        constraints=portfolio_constraints,
                        prev_weights=prev_w,
                        historical_returns=hist_returns,
                    )
                    
                    w = opt_result.weights
                    # Ensure cov_shrunk is defined for downstream overlays/vol-targeting
                    cov_shrunk = _shrink_cov_to_diag(cov, shrink_alpha)
                    
                    # EMIT: OPTIMIZER_SUCCESS
                    event_bus.emit(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        severity=EventSeverity.INFO,
                        code=EventCode.OPTIMIZER_SUCCESS,
                        message="Robust optimizer succeeded",
                        payload={
                            "optimizer_path": "robust",
                            "status": str(getattr(opt_result, "status", "unknown")),
                            "iterations": int(getattr(opt_result, "iterations", 0)),
                        },
                    )
                    
                    # Skip the standard path since optimizer already applied vol target and constraints
                    # Continue directly to overlays
                    
                except Exception as e:
                    logger.warning("[phase2.robust_optimizer] Optimization failed: %s; falling back to standard", e)
                    # EMIT: OPTIMIZER_FALLBACK
                    event_bus.emit(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=i,
                        severity=EventSeverity.WARNING,
                        code=EventCode.OPTIMIZER_FALLBACK,
                        message=f"Robust optimizer failed, falling back to MV: {e}",
                        payload={"error": str(e)},
                    )
                    # Fall back to standard path
                    cov_shrunk = _shrink_cov_to_diag(cov, shrink_alpha)
                    cov_shrunk = cov_shrunk + np.eye(n_assets, dtype=float) * 1e-8
                    try:
                        inv = np.linalg.pinv(cov_shrunk)
                    except Exception:
                        inv = np.eye(n_assets, dtype=float)
                    w_raw = inv @ z_thr
                    if safety_enabled and (not np.all(np.isfinite(w_raw))):
                        logger.error("Phase2 v2 non-finite w_raw detected; forcing flat (%s)", str(day))
                        w_raw = np.zeros(n_assets, dtype=float)
                    w = _apply_basic_constraints(w_raw, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)
                    w = _apply_vol_target(w, cov_shrunk, target_vol_annual=target_vol_day)
                    w = _apply_basic_constraints(w, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)
            else:
                # Standard covariance-aware sizing (original path)
                cov_shrunk = _shrink_cov_to_diag(cov, shrink_alpha)
                cov_shrunk = cov_shrunk + np.eye(n_assets, dtype=float) * 1e-8
                try:
                    inv = np.linalg.pinv(cov_shrunk)
                except Exception:
                    inv = np.eye(n_assets, dtype=float)
                w_raw = inv @ z_thr

                if safety_enabled and (not np.all(np.isfinite(w_raw))):
                    logger.error("Phase2 v2 non-finite w_raw detected; forcing flat (%s)", str(day))
                    w_raw = np.zeros(n_assets, dtype=float)

                w = _apply_basic_constraints(w_raw, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)
                w = _apply_vol_target(w, cov_shrunk, target_vol_annual=target_vol_day)
                w = _apply_basic_constraints(w, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)

            # ----------------------------
            # STRESS OVERLAY: On-the-fly scenario stress tests
            # Applied after optimizer, before final safety overlays.
            # ----------------------------
            stress_triggered = False
            stress_result_log: Optional[Dict[str, Any]] = None
            
            if stress_overlay_enabled and stress_overlay is not None:
                try:
                    # Compute corr_hhi for stress overlay (if not already computed)
                    corr_hhi_stress = 0.0
                    try:
                        diag_vals = np.sqrt(np.diag(cov))
                        diag_inv = np.where(diag_vals > 1e-8, 1.0 / diag_vals, 0.0)
                        corr_mat_stress = (cov * diag_inv[:, None]) * diag_inv[None, :]
                        corr_triu_stress = corr_mat_stress[np.triu_indices_from(corr_mat_stress, k=1)]
                        corr_hhi_stress = float(np.sum(corr_triu_stress ** 2)) if len(corr_triu_stress) > 0 else 0.0
                    except Exception:
                        corr_hhi_stress = 0.0
                    
                    # Get previous/current realized vol for "vol rising" detection
                    # rv_prev is tracked by stress_overlay._prev_vol internally
                    # We pass current rv which was computed earlier
                    rv_for_stress = _rolling_realized_vol(net_ret[:i], window=int(vol_window)) if i > 0 else 0.0
                    rv_prev_stress = _rolling_realized_vol(net_ret[:max(0, i - 1)], window=int(vol_window)) if i > 1 else 0.0
                    
                    # Build sector weights if group_by_symbol available
                    sector_weights_stress: Optional[Dict[str, float]] = None
                    if group_by_symbol is not None:
                        sector_weights_stress = {}
                        for j, sym in enumerate(syms):
                            sector = group_by_symbol.get(sym, "unknown")
                            sector_weights_stress[sector] = sector_weights_stress.get(sector, 0.0) + abs(float(w[j]))
                    
                    # Apply stress overlay
                    w, stress_result = stress_overlay.apply(
                        w=w,
                        cov=cov_shrunk if 'cov_shrunk' in dir() and cov_shrunk is not None else cov,
                        vol_prev=rv_prev_stress,
                        vol_curr=rv_for_stress,
                        corr_hhi=corr_hhi_stress,
                        sector_weights=sector_weights_stress,
                        symbols=syms,
                        group_by_symbol=group_by_symbol,
                    )
                    
                    stress_triggered = stress_result.triggered
                    stress_result_log = stress_result.to_dict()
                    
                    if stress_triggered:
                        # Emit structured event
                        if emit_events:
                            emit_event({
                                "event": "phase2.stress_overlay",
                                "day_idx": int(i),
                                "date": str(day) if day is not None else None,
                                "stress_result": stress_result_log,
                            })
                        logger.info(
                            "[phase2.stress_overlay] Day %d: scale=%.2f | %s",
                            i, stress_result.final_scale, " | ".join(stress_result.messages)
                        )
                except Exception as e:
                    logger.warning("[phase2.stress_overlay] Error applying stress overlay: %s", e)

            # ----------------------------
            # Deterministic HF-grade overlays
            # ----------------------------
            if safety_enabled:
                dd_scale = 1.0
                vol_scale = 1.0
                flattened = 0.0
                beta_neut_flag = 0.0

                # Rolling drawdown and realized-vol throttles (based on portfolio history up to t-1).
                eq_prev = float(equity[i - 1]) if i > 0 else 1.0
                equity_peak = float(max(equity_peak, eq_prev))
                dd = float(1.0 - (eq_prev / equity_peak)) if equity_peak > 0 else 0.0
                overlay_dd[i] = float(dd)
                
                # Realized vol (computed early for risk latch)
                rv = _rolling_realized_vol(net_ret[:i], window=int(vol_window)) if i > 0 else 0.0
                overlay_rv[i] = float(rv)
                
                # ─────────────────────────────────────────────────────────────
                # RISK LATCH: Check all conditions (replaces individual kill-switches)
                # ─────────────────────────────────────────────────────────────
                risk_latch_override = False
                if risk_latch_enabled and risk_latch is not None:
                    # Check NaN/Inf in predictions
                    risk_latch.check_nan_inf([mu_vec, sigma_vec, z_thr, w], ["mu", "sigma", "z_thr", "w"])
                    
                    # Check drawdown conditions
                    risk_latch.check_drawdown(dd)
                    
                    # Check volatility conditions
                    risk_latch.check_volatility(rv)
                    
                    # Check single-day crash
                    if i > 0:
                        daily_ret = float(net_ret[i - 1])
                        risk_latch.check_daily_return(daily_ret)
                    
                    # Check model health (calibration)
                    calib_score = 1.0
                    if mamba_calib_snapshot is not None:
                        calib_score = float(mamba_calib_snapshot.calibration_overall)
                    risk_latch.check_model_health(calib_score, z_thr)
                    
                    # Check CBOE panic (if available)
                    if 'day_cboe_panic' in dir() and np.isfinite(day_cboe_panic):
                        vix_approx = 20.0 * (1.0 + float(day_cboe_panic))  # Rough VIX estimate
                        risk_latch.check_cboe_panic(vix_approx, float(day_cboe_panic))
                    
                    # Check correlation HHI
                    try:
                        diag_vals = np.sqrt(np.diag(cov))
                        diag_inv = np.where(diag_vals > 1e-8, 1.0 / diag_vals, 0.0)
                        corr_mat = (cov * diag_inv[:, None]) * diag_inv[None, :]
                        corr_triu = corr_mat[np.triu_indices_from(corr_mat, k=1)]
                        corr_hhi_check = float(np.sum(corr_triu ** 2)) if len(corr_triu) > 0 else 0.0
                        risk_latch.check_correlation_hhi(corr_hhi_check)
                    except Exception:
                        pass
                    
                    # Resolve to final state
                    latch_state = risk_latch.resolve()
                    
                    # Apply risk latch override
                    if latch_state.mode >= RiskLatchMode.FLATTEN:
                        # FLATTEN or higher: force weights to zero
                        w = np.zeros(n_assets, dtype=float)
                        flattened = 1.0
                        risk_latch_override = True
                        
                        # EMIT: RISK_LATCH_FLATTEN
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.CRITICAL,
                            code=EventCode.RISK_LATCH_FLATTEN,
                            message=f"Risk latch: {latch_state.mode.name} due to {latch_state.primary_reason.name if latch_state.primary_reason else 'unknown'}",
                            payload=latch_state.to_dict(),
                        )
                    
                    elif latch_state.mode == RiskLatchMode.THROTTLE:
                        # THROTTLE: scale exposure
                        w = w * latch_state.exposure_scale
                        dd_scale = min(dd_scale, latch_state.exposure_scale)
                        
                        # EMIT: RISK_LATCH_THROTTLE
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.WARNING,
                            code=EventCode.RISK_LATCH_THROTTLE,
                            message=f"Risk latch throttle: scale={latch_state.exposure_scale:.2f}",
                            payload=latch_state.to_dict(),
                        )

                # Legacy kill-switches (only if risk_latch didn't override)
                if not risk_latch_override:
                    # Hard kill-switch on deep drawdown.
                    if np.isfinite(dd_kill) and dd_kill > 0 and dd >= dd_kill:
                        logger.warning("Phase2 v2 DD kill-switch triggered (dd=%.4f >= %.4f); flattening", dd, dd_kill)
                        # EMIT: DD_KILL
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.CRITICAL,
                            code=EventCode.DD_KILL,
                            message=f"DD kill-switch triggered: dd={dd:.4f} >= {dd_kill:.4f}",
                            payload={"dd": float(dd), "dd_kill": float(dd_kill), "equity_peak": float(equity_peak)},
                        )
                        w = np.zeros(n_assets, dtype=float)
                        flattened = 1.0
                    else:
                        # Drawdown throttles (reduce gross).
                        gross_scale = 1.0
                        if np.isfinite(dd_throttle_2) and dd_throttle_2 > 0 and dd >= dd_throttle_2:
                            gross_scale = float(np.clip(dd_gross_mult_2, 0.0, 1.0))
                            # EMIT: DD_THROTTLE
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.DD_THROTTLE,
                                message=f"DD throttle tier-2: dd={dd:.4f}, scale={gross_scale:.2f}",
                                payload={"dd": float(dd), "tier": 2, "gross_scale": float(gross_scale)},
                            )
                        elif np.isfinite(dd_throttle_1) and dd_throttle_1 > 0 and dd >= dd_throttle_1:
                            gross_scale = float(np.clip(dd_gross_mult_1, 0.0, 1.0))
                            # EMIT: DD_THROTTLE
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.DD_THROTTLE,
                                message=f"DD throttle tier-1: dd={dd:.4f}, scale={gross_scale:.2f}",
                                payload={"dd": float(dd), "tier": 1, "gross_scale": float(gross_scale)},
                            )
                        if gross_scale < 1.0:
                            w = w * gross_scale
                        dd_scale = float(gross_scale)
                    
                    # Realized vol throttle/kill (annualized) - rv already computed above.
                    if np.isfinite(rv) and rv > 0 and np.isfinite(target_vol) and target_vol > 0:
                        if np.isfinite(vol_kill_mult) and vol_kill_mult > 0 and rv >= float(vol_kill_mult) * float(target_vol_day):
                            logger.warning(
                                "Phase2 v2 vol kill-switch triggered (rv=%.3f >= %.3f); flattening",
                                rv,
                                float(vol_kill_mult) * float(target_vol_day),
                            )
                            # EMIT: VOL_KILL
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.CRITICAL,
                                code=EventCode.VOL_KILL,
                                message=f"Vol kill-switch triggered: rv={rv:.3f} >= {float(vol_kill_mult) * float(target_vol_day):.3f}",
                                payload={"rv": float(rv), "target_vol": float(target_vol_day), "vol_kill_mult": float(vol_kill_mult)},
                            )
                            w = np.zeros(n_assets, dtype=float)
                            flattened = 1.0
                        elif np.isfinite(vol_throttle_mult) and vol_throttle_mult > 0 and rv >= float(vol_throttle_mult) * float(target_vol_day):
                            # Scale down to bring realized vol back toward the throttle limit.
                            scale = float((float(vol_throttle_mult) * float(target_vol_day)) / (rv + 1e-12))
                            # EMIT: VOL_THROTTLE
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.VOL_THROTTLE,
                                message=f"Vol throttle applied: rv={rv:.3f}, scale={scale:.2f}",
                                payload={"rv": float(rv), "target_vol": float(target_vol_day), "scale": float(scale)},
                            )
                            vol_scale = float(np.clip(scale, 0.0, 1.0))
                            w = w * float(vol_scale)
                
                overlay_dd_gross_scale[i] = float(dd_scale)
                overlay_vol_scale[i] = float(vol_scale)

                # Optional beta neutralization / beta exposure cap.
                # Apply if: (1) deprecated beta_neutral flag is True, OR
                #           (2) beta_cap is set (>= 0 means cap constraint active)
                # Note: beta_cap = 0 → full neutralization, beta_cap > 0 → capped exposure
                if beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0):
                    look = int(max(5, min(int(beta_lookback), i)))
                    if look >= 5:
                        win = returns_df.iloc[max(0, i - look) : i].to_numpy(dtype=float)
                        beta_vec = _estimate_beta_vector(win)
                        w = _apply_beta_neutralization(w, beta_vec, max_abs_beta_exposure=float(beta_cap))
                        beta_neut_flag = 1.0
                overlay_beta_neutralized[i] = float(beta_neut_flag)

                # Optional group exposure caps (sector/industry/cluster map).
                w = _apply_group_exposure_caps(
                    w,
                    symbols=syms,
                    group_by_symbol=group_by_symbol,
                    max_group_gross=max_group_gross_day,
                    max_group_net=max_group_net_day,
                )

                # Optional exponential weight smoothing.
                if np.isfinite(weight_smoothing_alpha_day) and 0.0 < weight_smoothing_alpha_day < 1.0:
                    a = float(weight_smoothing_alpha_day)
                    w = a * prev_w + (1.0 - a) * w

                # Turnover budgeting (L1 delta-w cap).
                w = _project_turnover_budget(prev_w, w, turnover_cap=float(turnover_cap_day))

                # Liquidity constraints in USD (requires capital + ADV series).
                if capital_usd_f > 0 and (
                    (np.isfinite(max_adv_frac_name) and max_adv_frac_name > 0)
                    or (np.isfinite(max_turnover_adv_frac) and max_turnover_adv_frac > 0)
                ):
                    adv_vec = np.zeros(n_assets, dtype=float)
                    ok_adv = True
                    for j, sym in enumerate(syms):
                        try:
                            adv_s = adv_usd_by_sym.get(str(sym).upper())
                            if adv_s is None or adv_s.empty:
                                ok_adv = False
                                break
                            adv_vec[j] = float(pd.to_numeric(adv_s.reindex([pd.Timestamp(day)]).iloc[0], errors="coerce"))
                        except Exception:
                            ok_adv = False
                            break

                    if ok_adv and np.all(np.isfinite(adv_vec)) and np.max(adv_vec) > 0:
                        # Per-name max position by ADV participation.
                        if np.isfinite(max_adv_frac_name) and max_adv_frac_name > 0:
                            max_w_by_adv = (float(max_adv_frac_name) * adv_vec) / float(capital_usd_f)
                            max_w_by_adv = np.clip(max_w_by_adv, 0.0, 10.0)
                            w = np.clip(w, -max_w_by_adv, max_w_by_adv)

                        # Max daily turnover dollars by ADV participation.
                        if np.isfinite(max_turnover_adv_frac) and max_turnover_adv_frac > 0:
                            dw = w - prev_w
                            traded_usd = float(np.sum(np.abs(dw)) * float(capital_usd_f))
                            cap_usd = float(max_turnover_adv_frac) * float(np.sum(adv_vec))
                            if traded_usd > 0 and cap_usd > 0 and traded_usd > cap_usd:
                                w = prev_w + dw * (cap_usd / traded_usd)
                    else:
                        # If ADV is not available, keep behavior unchanged.
                        pass

                overlay_flattened[i] = float(flattened)

            # Hard universe gate (trading only): zero out inactive names before
            # final constraints/vol-target, then proceed as usual.
            if universe_enabled and _universe_mask is not None and _universe_mask.size == w.size:
                w = np.where(_universe_mask, w, 0.0)

            # Registry gate (authoritative): drop ineligible names, then re-apply constraints.
            if _eligible_mask is not None and _eligible_mask.size == w.size:
                if bool(np.any(~_eligible_mask)) and float(np.sum(np.abs(w))) > 0:
                    dropped = [str(s).upper() for j, s in enumerate(syms) if (not bool(_eligible_mask[j])) and abs(float(w[j])) > 1e-12]
                    if dropped:
                        logger.warning(
                            "[phase2.universe_registry] %s zeroed %d ineligible weights: %s",
                            str(pd.Timestamp(day))[:10],
                            len(dropped),
                            ",".join(dropped[:25]) + ("..." if len(dropped) > 25 else ""),
                        )
                        # EMIT: REGISTRY_GATE
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.WARNING,
                            code=EventCode.REGISTRY_GATE,
                            message=f"Registry gate zeroed {len(dropped)} ineligible weights",
                            payload={"n_dropped": len(dropped), "dropped_syms": dropped[:10]},
                        )
                w = np.where(_eligible_mask, w, 0.0)

            # Re-apply core constraints after overlays.
            w = _apply_basic_constraints(w, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)
            w = _apply_vol_target(w, cov_shrunk, target_vol_annual=target_vol_day)
            w = _apply_basic_constraints(w, max_name=max_name_day, max_gross=max_gross_day, max_net=max_net_day)

            w_mat[i, :] = w

            if diag_enabled:
                gross_exposure[i] = float(np.sum(np.abs(w)))
                net_exposure[i] = float(np.sum(w))
                eff_n_bets[i] = float(_effective_n_bets_from_weights(w))
                flat_flag[i] = 1.0 if gross_exposure[i] <= 1e-12 else 0.0
                sh = _sigma_health(mu_vec, sigma_vec, eps=float(diag_sigma_eps))
                sigma_med_daily[i] = float(sh.get("sigma_median", float("nan")))
                sigma_p90_daily[i] = float(sh.get("sigma_p90", float("nan")))
                sigma_collapse_daily[i] = float(sh.get("sigma_collapse_pct", float("nan")))
                corr_abs_mu_sigma_daily[i] = float(sh.get("corr_abs_mu_sigma", float("nan")))

            # Realized PnL for day uses previous weights (weights set at close for next day).
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Choose return source based on exec_price_mode
            # ─────────────────────────────────────────────────────────────────────
            if exec_price_mode == "open_to_open" and open_returns_df is not None:
                try:
                    r_vec = open_returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
                except Exception:
                    r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
            else:
                r_vec = returns_df.loc[pd.Timestamp(day)].to_numpy(dtype=float)
            
            # ─────────────────────────────────────────────────────────────────────
            # FAST CRASH TRIGGER B: Gap/Overnight Shock (pre-trade)
            # Detect dangerous gaps at open that may invalidate yesterday's positions.
            # ─────────────────────────────────────────────────────────────────────
            gap_shock_triggered = False
            if risk_latch_enabled and risk_latch is not None and i > 0:
                try:
                    # Get per-asset rolling vol for gap detection
                    rolling_vol_gap = None
                    if i >= 20:
                        # Compute rolling 20-day vol (annualized)
                        ret_window = returns_df.iloc[max(0, i - 20):i].to_numpy(dtype=float)
                        rolling_vol_gap = np.std(ret_window, axis=0) * np.sqrt(252.0)
                    
                    gap_shock_triggered = risk_latch.check_gap_shock(
                        r_vec=r_vec,
                        rolling_vol=rolling_vol_gap,
                        prev_weights=prev_exec_w,  # Yesterday's executed weights
                        symbols=syms,
                    )
                    
                    if gap_shock_triggered:
                        # Resolve immediately to get action
                        gap_state = risk_latch.resolve()
                        if gap_state.mode >= RiskLatchMode.FLATTEN:
                            # Force flatten for today
                            w = np.zeros(n_assets, dtype=float)
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.CRITICAL,
                                code=EventCode.RISK_LATCH_FLATTEN,
                                message=f"GAP SHOCK → FLATTEN: {gap_state.all_triggers[-1].message if gap_state.all_triggers else 'Gap detected'}",
                                payload=gap_state.to_dict(),
                            )
                        elif gap_state.mode == RiskLatchMode.THROTTLE:
                            # Scale down weights
                            w = w * gap_state.exposure_scale
                            event_bus.emit(
                                date=str(pd.Timestamp(day))[:10],
                                day_idx=i,
                                severity=EventSeverity.WARNING,
                                code=EventCode.RISK_LATCH_THROTTLE,
                                message=f"GAP SHOCK → THROTTLE: scale={gap_state.exposure_scale:.2f}",
                                payload=gap_state.to_dict(),
                            )
                except Exception as e:
                    logger.debug("[phase2.risk_latch.gap_shock] Error: %s", e)
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Update ExecutionContext with daily ADV data
            # ─────────────────────────────────────────────────────────────────────
            if capital_usd_f > 0:
                exec_ctx.capital_usd = float(capital_usd_f)
                try:
                    adv_vec_exec = np.full(n_assets, 10_000_000.0, dtype=float)
                    for j, sym in enumerate(syms):
                        adv_s = adv_usd_by_sym.get(str(sym).upper())
                        if adv_s is not None and not adv_s.empty:
                            try:
                                adv_val = float(pd.to_numeric(adv_s.reindex([pd.Timestamp(day)]).iloc[0], errors="coerce"))
                                if np.isfinite(adv_val) and adv_val > 0:
                                    adv_vec_exec[j] = adv_val
                            except Exception:
                                pass
                    exec_ctx.adv_usd = adv_vec_exec
                except Exception:
                    pass
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Apply borrow constraints if enabled
            # ─────────────────────────────────────────────────────────────────────
            if exec_per_name_borrow:
                w = _apply_borrow_constraints(
                    w,
                    exec_ctx,
                    htb_threshold_bps=exec_htb_threshold_bps,
                    htb_max_short_weight=exec_htb_max_short_weight,
                )
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Apply corporate action constraints if enabled
            # ─────────────────────────────────────────────────────────────────────
            if exec_halted_to_flat or exec_reduce_short_on_ex_div:
                w = _apply_corporate_action_constraints(
                    w,
                    exec_ctx,
                    reduce_on_ex_div=exec_reduce_short_on_ex_div,
                    ex_div_reduction_mult=exec_ex_div_reduction_mult,
                )
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Apply liquidity participation constraint
            # ─────────────────────────────────────────────────────────────────────
            if exec_enable_partial_fills and capital_usd_f > 0:
                w, residual_out = _apply_liquidity_participation_constraint(
                    w,
                    prev_w,
                    exec_ctx,
                    max_participation_rate=exec_participation_rate,
                )
                # Carry residual to next day
                exec_ctx.partial_fill_residual = residual_out
            
            pnl = float(exec_w @ r_vec)
            
            # ─────────────────────────────────────────────────────────────────────
            # DIAGNOSTIC FIX: Track BOTH intent and executed turnover separately
            # - turnover_intent: optimizer target diff (for policy enforcement)
            # - turnover_exec: execution queue diff (for cost computation)
            # This fixes the blind spot where costs were attributed to intent but PnL came from exec
            # ─────────────────────────────────────────────────────────────────────
            tval_intent = float(np.sum(np.abs(w - prev_w)))
            tval_exec = float(np.sum(np.abs(exec_w - prev_exec_w)))
            turnover_intent[i] = tval_intent
            turnover_exec[i] = tval_exec
            turnover[i] = tval_exec  # Legacy: points to exec turnover for backwards compat

            # Turnover kill-switch operates on INTENT (policy control of optimizer output)
            if safety_enabled and np.isfinite(kill_on_turnover_gt) and kill_on_turnover_gt > 0 and tval_intent >= kill_on_turnover_gt:
                logger.warning("Phase2 v2 turnover kill-switch triggered (intent=%.3f >= %.3f); flattening", tval_intent, kill_on_turnover_gt)
                # EMIT: TURNOVER_KILL
                event_bus.emit(
                    date=str(pd.Timestamp(day))[:10],
                    day_idx=i,
                    severity=EventSeverity.CRITICAL,
                    code=EventCode.TURNOVER_KILL,
                    message=f"Turnover kill-switch triggered: intent={tval_intent:.3f} >= {kill_on_turnover_gt:.3f}",
                    payload={
                        "turnover_intent": float(tval_intent),
                        "turnover_exec": float(tval_exec),
                        "kill_threshold": float(kill_on_turnover_gt),
                        "policy_action_idx": int(policy_action_history[i]) if policy_action_history[i] is not None else None,
                    },
                )
                w = np.zeros(n_assets, dtype=float)
                w_mat[i, :] = w
                try:
                    overlay_flattened[i] = 1.0
                except Exception:
                    pass
                # Re-compute intent turnover after flattening
                tval_intent = float(np.sum(np.abs(w - prev_w)))
                turnover_intent[i] = tval_intent
                if flat_cooldown_sessions > 0:
                    flat_until_idx = int(min(len(union_oos_index) - 1, i + int(flat_cooldown_sessions)))
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Compute cost with asset-aware slippage or legacy
            # CRITICAL FIX: Costs are computed on EXECUTED turnover, not intent
            # ─────────────────────────────────────────────────────────────────────
            if exec_asset_aware_slippage and capital_usd_f > 0:
                # Per-asset slippage model using EXECUTED weight changes
                delta_w_exec = exec_w - prev_exec_w
                slippage_per_asset = _compute_asset_aware_slippage(
                    delta_w_exec,
                    exec_ctx,
                    impact_exponent=exec_impact_exponent,
                    impact_scale=exec_impact_scale,
                )
                cost = float(np.sum(slippage_per_asset))
            else:
                # Legacy aggregate model using EXECUTED turnover
                cost = float(k_spread * tval_exec + k_impact * (tval_exec ** 1.5))
            
            # ─────────────────────────────────────────────────────────────────────
            # EXECUTION REALISM: Per-name borrow cost or legacy flat rate
            # ─────────────────────────────────────────────────────────────────────
            if exec_per_name_borrow:
                # Per-name borrow fees
                borrow_cost = _compute_per_asset_borrow_cost(exec_w, exec_ctx)
                cost += borrow_cost
            elif safety_enabled and np.isfinite(borrow_fee_bps_annual) and borrow_fee_bps_annual > 0:
                # Legacy flat rate
                short_notional = float(np.sum(np.maximum(-exec_w, 0.0)))
                cost += float((borrow_fee_bps_annual / 1e4) / 252.0) * short_notional
            
            costs[i] = cost
            pnl_net = float(pnl - cost)
            net_ret[i] = pnl_net
            if i == 0:
                equity[i] = 1.0 * (1.0 + pnl_net)
            else:
                equity[i] = float(equity[i - 1] * (1.0 + pnl_net))

            # ─────────────────────────────────────────────────────────────────────
            # FAST CRASH TRIGGER A: 1-Day Loss Kill → EMERGENCY_STOP
            # Called RIGHT AFTER computing pnl_net, BEFORE policy update so policy
            # can learn from the event. Sets cooldown for N sessions.
            # ─────────────────────────────────────────────────────────────────────
            if risk_latch_enabled and risk_latch is not None and i > 0:
                try:
                    equity_prev = float(equity[i - 1])
                    equity_today = float(equity[i])
                    
                    one_day_kill = risk_latch.check_one_day_loss_kill(
                        pnl_net_today=pnl_net,
                        equity_today=equity_today,
                        equity_prev=equity_prev,
                    )
                    
                    if one_day_kill:
                        # EMERGENCY_STOP triggered - set cooldown
                        emergency_state = risk_latch.resolve()
                        flat_until_idx = int(min(
                            len(union_oos_index) - 1,
                            i + risk_latch.thresholds.emergency_cooldown_sessions
                        ))
                        
                        # ─────────────────────────────────────────────────────────
                        # LEDGER: Log 1-day loss kill event
                        # ─────────────────────────────────────────────────────────
                        if risk_events_ledger is not None:
                            try:
                                risk_events_ledger.add_event(
                                    date=str(pd.Timestamp(day))[:10],
                                    day_idx=i,
                                    event_type="ONE_DAY_LOSS_KILL",
                                    threshold=f"Loss > {risk_latch.thresholds.one_day_loss_kill * 100:.1f}%",
                                    threshold_value=float(risk_latch.thresholds.one_day_loss_kill),
                                    observed_value=abs(pnl_net),
                                    threshold_crossed=True,
                                    action_taken="EMERGENCY_STOP",
                                    exposure_scale=0.0,
                                    emergency_remaining=int(risk_latch.thresholds.emergency_cooldown_sessions),
                                    trigger_reason="ONE_DAY_LOSS_KILL",
                                    details={
                                        "pnl_net": float(pnl_net),
                                        "equity_prev": float(equity_prev),
                                        "equity_today": float(equity_today),
                                        "flat_until_idx": int(flat_until_idx),
                                    },
                                )
                            except Exception as le:
                                logger.debug("[phase2.ledger] Failed to log event: %s", le)
                        
                        # EMIT: RISK_LATCH_EMERGENCY
                        event_bus.emit(
                            date=str(pd.Timestamp(day))[:10],
                            day_idx=i,
                            severity=EventSeverity.CRITICAL,
                            code=EventCode.RISK_LATCH_EMERGENCY,
                            message=f"1-DAY LOSS KILL: {pnl_net:.2%} loss → EMERGENCY_STOP, flat until day {flat_until_idx}",
                            payload={
                                "pnl_net": float(pnl_net),
                                "equity_prev": float(equity_prev),
                                "equity_today": float(equity_today),
                                "flat_until_idx": int(flat_until_idx),
                                "cooldown_sessions": int(risk_latch.thresholds.emergency_cooldown_sessions),
                                **emergency_state.to_dict(),
                            },
                        )
                        
                        logger.warning(
                            "[phase2.risk_latch] 1-DAY LOSS KILL: pnl=%.2f%%, equity %.4f→%.4f, flat until day %d",
                            pnl_net * 100, equity_prev, equity_today, flat_until_idx,
                        )
                except Exception as e:
                    logger.debug("[phase2.risk_latch.1day_kill] Error: %s", e)

            # Policy update (delayed reward over a rolling window).
            # FIX Gap #3: Action at time t affects holdings at t+(delay-1), so we need to
            # shift attribution by (delay-1) when delay > 1. Otherwise the bandit learns noise.
            if policy is not None and policy_reward_window > 1:
                try:
                    window = int(max(2, policy_reward_window))
                    # Adjust start_idx for execution delay: action at t affects holdings at t+(delay-1)
                    # So reward attribution should be shifted back by (delay-1)
                    delay_shift = int(max(0, trade_delay_sessions - 1))
                    start_idx = int(i - window + 1 - delay_shift)
                    
                    if start_idx >= 0:
                        action_idx = policy_action_history[start_idx]
                        state_vec = policy_state_history[start_idx]
                        if action_idx is not None and state_vec is not None:
                            # Reward window also shifted to align with when action's effects materialize
                            reward_start = int(start_idx + delay_shift)
                            reward_end = int(min(i, len(net_ret) - 1))
                            
                            if reward_end > reward_start:
                                reward = _policy_reward_from_window(
                                    net_returns=net_ret,
                                    turnover=turnover,
                                    equity=equity,
                                    start=reward_start,
                                    end=reward_end,
                                    lambda_turnover=policy_reward_lambda_turn,
                                    lambda_drawdown=policy_reward_lambda_dd,
                                )
                                policy.update(action_index=int(action_idx), x=state_vec, reward=float(reward))
                                
                                # Update trace with delayed reward
                                if trace_enabled and start_idx < len(daily_traces):
                                    try:
                                        daily_traces[start_idx].reward = float(reward)
                                        daily_traces[start_idx].reward_window_start = int(reward_start)
                                        daily_traces[start_idx].reward_window_end = int(reward_end)
                                    except Exception:
                                        pass
                except Exception:
                    pass

            # Roll covariance forward using today's realized returns.
            cov = _ewma_cov_update(cov, r_vec, cov_lam)

            # Update the execution-delay queue.
            if trade_delay_sessions > 1:
                _queue.append(w.copy())
                exec_w = np.asarray(_queue.popleft(), dtype=float)
            else:
                exec_w = w

            # ─────────────────────────────────────────────────────────────────────
            # DAILY TRACE INSTRUMENTATION: Capture full signal chain
            # ─────────────────────────────────────────────────────────────────────
            if trace_enabled:
                try:
                    # Compute current drawdown
                    eq_for_dd = float(equity[i])
                    peak_for_trace = float(equity_peak)
                    dd_for_trace = float((peak_for_trace - eq_for_dd) / peak_for_trace) if peak_for_trace > 0 else 0.0
                    
                    # Compute realized volatility (rolling 20d)
                    rv_window = 20
                    rv_start = max(0, i - rv_window + 1)
                    rv_for_trace = float(np.std(net_ret[rv_start:i+1]) * np.sqrt(252.0)) if i >= rv_start else 0.0
                    
                    # Compute split costs (based on EXECUTED turnover for accuracy)
                    cost_spread_trace = 0.0
                    cost_impact_trace = 0.0
                    cost_borrow_trace = 0.0
                    if exec_asset_aware_slippage and capital_usd_f > 0:
                        delta_w_exec_trace = exec_w - prev_exec_w
                        slippage_trace = _compute_asset_aware_slippage(
                            delta_w_exec_trace, exec_ctx,
                            impact_exponent=exec_impact_exponent,
                            impact_scale=exec_impact_scale,
                        )
                        # Approximate split: spread = 50% of slippage, impact = 50%
                        cost_spread_trace = float(np.sum(slippage_trace)) * 0.5
                        cost_impact_trace = float(np.sum(slippage_trace)) * 0.5
                    else:
                        cost_spread_trace = float(k_spread * tval_exec)
                        cost_impact_trace = float(k_impact * (tval_exec ** 1.5))
                    
                    if exec_per_name_borrow:
                        cost_borrow_trace = _compute_per_asset_borrow_cost(exec_w, exec_ctx)
                    elif safety_enabled and np.isfinite(borrow_fee_bps_annual) and borrow_fee_bps_annual > 0:
                        short_not_trace = float(np.sum(np.maximum(-exec_w, 0.0)))
                        cost_borrow_trace = float((borrow_fee_bps_annual / 1e4) / 252.0) * short_not_trace
                    
                    # Eligible/universe mask summary
                    n_elig_trace = int(np.sum(_eligible_mask)) if _eligible_mask is not None else n_assets
                    n_univ_trace = int(np.sum(_universe_mask)) if _universe_mask is not None else n_assets
                    
                    # Dropped/added symbols tracking
                    dropped_trace: List[str] = []
                    added_trace: List[str] = []
                    if _eligible_mask is not None:
                        curr_elig_set = set(s for j, s in enumerate(syms) if _eligible_mask[j])
                        if _prev_eligible_set is not None:
                            dropped_trace = sorted(list(_prev_eligible_set - curr_elig_set))[:5]
                            added_trace = sorted(list(curr_elig_set - _prev_eligible_set))[:5]
                        _prev_eligible_set = curr_elig_set
                    
                    # Get z vectors at various stages (need to preserve them earlier)
                    z_raw_trace = np.divide(mu_vec, sigma_exec + 1e-9)
                    z_raw_trace = np.clip(z_raw_trace, -z_clip, z_clip)
                    
                    # Policy knobs
                    policy_knobs_trace = {
                        "base_threshold": float(thr_base_day),
                        "regime_mult_bull": float(thr_bull_day),
                        "regime_mult_bear": float(thr_bear_day),
                        "regime_mult_crisis": float(thr_crisis_day),
                        "target_vol": float(target_vol_day),
                        "turnover_cap": float(turnover_cap_day),
                        "max_gross": float(max_gross_day),
                        "max_net": float(max_net_day),
                        "max_name": float(max_name_day),
                        "weight_smoothing_alpha": float(weight_smoothing_alpha_day),
                        "vol_scaler": float(vol_scaler_day),
                        "quantile_blend_weight": float(quantile_blend_weight_day),
                    }
                    
                    # Constraints used
                    constraints_trace = {
                        "max_gross": float(max_gross_day),
                        "max_net": float(max_net_day),
                        "max_name": float(max_name_day),
                        "target_vol": float(target_vol_day),
                        "turnover_cap": float(turnover_cap_day),
                        "beta_neutral": float(beta_neutral),
                        "group_max_gross": float(max_group_gross),
                        "group_max_net": float(max_group_net),
                    }
                    
                    # Optimizer path
                    optimizer_path_trace = "robust" if (use_robust_optimizer and robust_optimizer is not None) else "mv_fallback"
                    
                    # Determine if throttles triggered
                    dd_throttle_flag = (dd_for_trace >= dd_throttle_1) if np.isfinite(dd_throttle_1) else False
                    vol_throttle_flag = (rv_for_trace >= vol_throttle_mult * target_vol_day) if np.isfinite(vol_throttle_mult) else False
                    kill_flag = (dd_for_trace >= dd_kill) if np.isfinite(dd_kill) else False
                    flattened_flag = bool(overlay_flattened[i] > 0.5) if i < len(overlay_flattened) else False
                    
                    # Build trace payload
                    trace = DailyTracePayload(
                        date=str(pd.Timestamp(day))[:10],
                        day_idx=int(i),
                        horizon=int(horizon),
                        
                        # Masks
                        n_eligible=n_elig_trace,
                        n_universe=n_univ_trace,
                        n_total=n_assets,
                        dropped_syms=dropped_trace,
                        added_syms=added_trace,
                        
                        # Returns
                        **{f"r_vec_{k}": v for k, v in _summarize_vector(r_vec).items()},
                        
                        # Model outputs
                        **{f"mu_vec_{k}": v for k, v in _summarize_vector(mu_vec).items()},
                        mu_top_k=_top_k_by_abs(mu_vec, syms, k=5),
                        **{f"sigma_raw_{k}": v for k, v in _summarize_vector(sigma_vec).items()},
                        **{f"sigma_exec_{k}": v for k, v in _summarize_vector(sigma_exec).items()},
                        sigma_exec_n_clipped=int(np.sum((sigma_ema < sigma_exec) | (sigma_ema > sigma_exec))),
                        
                        # Signal chain
                        **{f"z_raw_{k}": v for k, v in _summarize_vector(z_raw_trace).items()},
                        z_after_role_overlays_mean=float(np.mean(z)) if z is not None else 0.0,
                        z_after_role_overlays_std=float(np.std(z)) if z is not None else 0.0,
                        z_after_quantile_blend_mean=float(np.mean(z)) if z is not None else 0.0,
                        z_after_quantile_blend_std=float(np.std(z)) if z is not None else 0.0,
                        quantile_blend_weight=float(quantile_blend_weight_day),
                        z_thr_mean=float(np.mean(z_thr)) if z_thr is not None else 0.0,
                        z_thr_std=float(np.std(z_thr)) if z_thr is not None else 0.0,
                        z_thr_n_zeroed=int(np.sum(np.abs(z_thr) < 1e-12)) if z_thr is not None else 0,
                        
                        # Policy
                        policy_state_vec=policy_state_history[i].tolist() if policy_state_history[i] is not None else None,
                        policy_action_idx=policy_action_history[i],
                        policy_knobs=policy_knobs_trace,
                        
                        # Optimizer
                        optimizer_path=optimizer_path_trace,
                        optimizer_status="success",
                        optimizer_iterations=0,
                        constraints_used=constraints_trace,
                        w_target_gross=float(np.sum(np.abs(w))),
                        w_target_net=float(np.sum(w)),
                        w_target_n_nonzero=int(np.sum(np.abs(w) > 1e-12)),
                        w_target_max_abs=float(np.max(np.abs(w))) if len(w) else 0.0,
                        
                        # Safety
                        w_after_safety_gross=float(np.sum(np.abs(w))),
                        w_after_safety_net=float(np.sum(w)),
                        dd_current=dd_for_trace,
                        rv_current=rv_for_trace,
                        turnover_intent=float(tval_intent),
                        dd_throttle_triggered=dd_throttle_flag,
                        vol_throttle_triggered=vol_throttle_flag,
                        kill_triggered=kill_flag,
                        flattened=flattened_flag,
                        
                        # Execution
                        exec_w_gross=float(np.sum(np.abs(exec_w))),
                        exec_w_net=float(np.sum(exec_w)),
                        prev_exec_w_gross=float(np.sum(np.abs(prev_exec_w))),
                        prev_exec_w_net=float(np.sum(prev_exec_w)),
                        turnover_exec=float(tval_exec),  # Use pre-computed exec turnover
                        
                        pnl_gross=float(pnl),
                        cost_spread=cost_spread_trace,
                        cost_impact=cost_impact_trace,
                        cost_borrow=cost_borrow_trace,
                        cost_total=float(cost),
                        pnl_net=float(pnl_net),
                        
                        equity=float(equity[i]),
                        drawdown=dd_for_trace,
                        equity_peak=float(equity_peak),
                        
                        # Reward (will be filled in later if available)
                        reward=None,
                        reward_window_start=None,
                        reward_window_end=None,
                    )
                    daily_traces.append(trace)
                    
                except Exception as e:
                    logger.debug("[phase2.trace] Failed to capture trace for day %s: %s", day, e)

            # Update state for next iteration
            prev_exec_w = exec_w.copy()
            prev_w = w
            
            # ─────────────────────────────────────────────────────────────
            # RISK LATCH: End session - store final weights for recovery
            # ─────────────────────────────────────────────────────────────
            if risk_latch_enabled and risk_latch is not None:
                try:
                    # Only store good weights if not flattened
                    good_w = w.copy() if flattened < 0.5 else None
                    risk_latch.end_session(good_weights=good_w)
                except Exception as e:
                    logger.debug("[phase2.risk_latch] end_session error: %s", e)

            # Fold-level reporting/pruning (maturity-gated).
            # Only evaluate metrics on a matured prefix to avoid lookahead bias.
            if trial is not None and (i + 1) in fold_end_set:
                fold_num = int((i + 1 + fold_step_sessions - 1) // fold_step_sessions)
                try:
                    end_ts = pd.Timestamp(day)
                    mature_end_ts = maturity_cutoff(end_ts, H=int(horizon))
                    # Include sessions up to mature_end_ts (drops last ~H sessions).
                    mature_mask = union_oos_index <= pd.Timestamp(mature_end_ts)
                    if bool(np.any(mature_mask)):
                        r_pref = pd.Series(net_ret[mature_mask], index=union_oos_index[mature_mask], name="net_return").astype(float)
                        t_pref = pd.Series(turnover[mature_mask], index=union_oos_index[mature_mask], name="turnover").astype(float)
                        eq_pref = (1.0 + r_pref).cumprod().rename("equity_curve")
                        sharpe_pref = float(_annualized_sharpe(r_pref))
                        maxdd_pref = float(_max_drawdown_from_equity(eq_pref))
                        turnover_pref = float(t_pref.mean() if len(t_pref) else 0.0)
                        # Canonical fold score (the ONLY scalar TPE/Hyperband optimizes).
                        # - Sharpe uses net returns (explicit costs already applied)
                        # - turnover is an additional regularizer for churn
                        # - flat_rate penalizes "do nothing" solutions
                        try:
                            w_pref_days = np.asarray(w_mat[mature_mask, :], dtype=float)
                            gross_pref = np.sum(np.abs(w_pref_days), axis=1) if w_pref_days.size else np.asarray([], dtype=float)
                            flat_rate_pref_obj = float(np.mean(gross_pref <= 1e-12)) if gross_pref.size else 0.0
                        except Exception:
                            flat_rate_pref_obj = 0.0

                        fold_score = float(
                            sharpe_pref
                            - float(objective_spec.max_drawdown_penalty) * maxdd_pref
                            - float(lambda_turn) * turnover_pref
                            - float(lambda_flat) * float(flat_rate_pref_obj)
                        )

                        if diag_enabled:
                            try:
                                sigma_pref = np.asarray(sigma_mat[mature_mask, :], dtype=float).reshape(-1)
                                mu_pref = np.asarray(mu_mat[mature_mask, :], dtype=float).reshape(-1)
                                sig_diag = _sigma_health(mu_pref, sigma_pref, eps=float(diag_sigma_eps))
                            except Exception:
                                sig_diag = {}
                            try:
                                w_pref = np.asarray(w_mat[mature_mask, :], dtype=float)
                                eff_pref = float(np.nanmean([_effective_n_bets_from_weights(row) for row in w_pref])) if w_pref.size else 0.0
                            except Exception:
                                eff_pref = 0.0
                            try:
                                realized_vol_pref = float(np.std(r_pref) * np.sqrt(252.0)) if len(r_pref) else 0.0
                            except Exception:
                                realized_vol_pref = 0.0
                            try:
                                mean_active_pref = float(np.mean(active_frac[mature_mask])) if bool(np.any(mature_mask)) else 0.0
                                flat_rate_pref = float(np.mean(flat_flag[mature_mask])) if bool(np.any(mature_mask)) else 0.0
                                gross_mean_pref = float(np.mean(gross_exposure[mature_mask])) if bool(np.any(mature_mask)) else 0.0
                            except Exception:
                                mean_active_pref = 0.0
                                flat_rate_pref = 0.0
                                gross_mean_pref = 0.0

                            fold_logs.append(
                                {
                                    "kind": "fold",
                                    "fold": int(fold_num),
                                    "end_day": str(pd.Timestamp(day))[:10],
                                    "mature_end": str(pd.Timestamp(mature_end_ts))[:10],
                                    "score": float(fold_score),
                                    "sharpe": float(sharpe_pref),
                                    "max_drawdown": float(maxdd_pref),
                                    "turnover": float(turnover_pref),
                                    "realized_vol": float(realized_vol_pref),
                                    "gross_mean": float(gross_mean_pref),
                                    "flat_rate": float(flat_rate_pref),
                                    "active_frac_mean": float(mean_active_pref),
                                    "eff_n_bets_mean": float(eff_pref),
                                    "lambda_turn": float(lambda_turn),
                                    "lambda_flat": float(lambda_flat),
                                    **{f"sigma_{k}": float(v) for k, v in sig_diag.items() if isinstance(v, (int, float)) and np.isfinite(v)},
                                }
                            )
                            logger.info(
                                "[phase2.v2.diag] fold=%d score=%.4f sharpe=%.3f maxdd=%.3f vol=%.3f active=%.2f flat=%.2f",
                                int(fold_num),
                                float(fold_score),
                                float(sharpe_pref),
                                float(maxdd_pref),
                                float(realized_vol_pref),
                                float(mean_active_pref),
                                float(flat_rate_pref),
                            )

                        # Deterministic safety prune first (hard rule).
                        if allow_prune and int(fold_num) >= int(max(0, safety_prune_after_folds)):
                            if (sharpe_pref < float(safety_prune_sharpe_floor)) and (maxdd_pref > float(safety_prune_maxdd_ceiling)):
                                import optuna

                                raise optuna.TrialPruned(
                                    f"safety_prune: fold={fold_num} sharpe={sharpe_pref:.3f} maxdd={maxdd_pref:.3f}"
                                )

                        # Deterministic prune guards (before Hyperband pruning).
                        if allow_prune and prune_guards_enabled:
                            # Sigma collapse for N consecutive folds.
                            try:
                                sigma_pref_guard = np.asarray(sigma_mat[mature_mask, :], dtype=float).reshape(-1)
                                mu_pref_guard = np.asarray(mu_mat[mature_mask, :], dtype=float).reshape(-1)
                                sig_diag_guard = _sigma_health(mu_pref_guard, sigma_pref_guard, eps=float(diag_sigma_eps))
                                collapse_pct = float(sig_diag_guard.get("sigma_collapse_pct", 0.0))
                            except Exception:
                                collapse_pct = 0.0
                            if np.isfinite(collapse_pct) and collapse_pct > float(prune_sigma_collapse_pct):
                                sigma_collapse_bad_folds += 1
                            else:
                                sigma_collapse_bad_folds = 0

                            # Degenerate portfolio: too few effective bets for N consecutive folds.
                            try:
                                w_pref_guard = np.asarray(w_mat[mature_mask, :], dtype=float)
                                eff_pref_guard = float(np.nanmean([_effective_n_bets_from_weights(row) for row in w_pref_guard])) if w_pref_guard.size else 0.0
                            except Exception:
                                eff_pref_guard = 0.0
                            if np.isfinite(eff_pref_guard) and eff_pref_guard < float(prune_eff_n_bets_min):
                                eff_n_bets_bad_folds += 1
                            else:
                                eff_n_bets_bad_folds = 0

                            # Update harm rate guard (requires pre/post).
                            harm_rate_now = float("nan")
                            if diag_pre_post and int(fold_num) >= int(prune_update_harm_after_folds):
                                deltas = [
                                    r.get("delta_val_loss")
                                    for r in update_logs
                                    if isinstance(r, dict) and r.get("delta_val_loss") is not None
                                ]
                                deltas_f = [float(x) for x in deltas if isinstance(x, (int, float)) and np.isfinite(x)]
                                if len(deltas_f) >= int(prune_update_harm_min_updates):
                                    harm_rate_now = float(np.mean(np.asarray(deltas_f) > 0.0))

                            # Report canonical fold score first so Optuna sees the datapoint.
                            trial.report(float(fold_score), step=int(fold_num))

                            if sigma_collapse_bad_folds >= int(max(1, prune_sigma_collapse_folds)):
                                import optuna

                                raise optuna.TrialPruned(
                                    f"sigma_collapse_guard: fold={fold_num} collapse={collapse_pct:.3%} "
                                    f"bad_folds={sigma_collapse_bad_folds}/{prune_sigma_collapse_folds}"
                                )
                            if eff_n_bets_bad_folds >= int(max(1, prune_eff_n_bets_folds)):
                                import optuna

                                raise optuna.TrialPruned(
                                    f"degenerate_portfolio_guard: fold={fold_num} eff_n_bets={eff_pref_guard:.2f} "
                                    f"bad_folds={eff_n_bets_bad_folds}/{prune_eff_n_bets_folds}"
                                )
                            if (
                                diag_pre_post
                                and np.isfinite(harm_rate_now)
                                and int(fold_num) >= int(prune_update_harm_after_folds)
                                and harm_rate_now > float(prune_update_harm_rate)
                            ):
                                import optuna

                                raise optuna.TrialPruned(
                                    f"update_harm_guard: fold={fold_num} harm_rate={harm_rate_now:.1%} > {prune_update_harm_rate:.1%}"
                                )
                        else:
                            # Canonical scalar for Hyperband/TPE.
                            trial.report(float(fold_score), step=int(fold_num))

                        # Hyperband/Optuna pruning second.
                        if allow_prune and int(fold_num) >= int(max(0, prune_warmup_folds)):
                            if trial.should_prune():
                                import optuna

                                raise optuna.TrialPruned(f"hyperband_prune: fold={fold_num}")
                except Exception as e:
                    # Never let reporting crashes break a trial, but DO
                    # propagate Optuna pruning signals.
                    try:
                        import optuna

                        if isinstance(e, optuna.TrialPruned):
                            raise
                    except Exception:
                        pass
                    pass

        # Build outputs.
        preds_by_symbol = {
            sym: pd.Series(mu_mat[:, j], index=union_oos_index, name="mu") for j, sym in enumerate(syms)
        }
        equity_curve = pd.Series(equity, index=union_oos_index, name="equity_curve")
        port_net = pd.Series(net_ret, index=union_oos_index, name="net_return")
        port_turn = pd.Series(turnover, index=union_oos_index, name="turnover")
        port_cost = pd.Series(costs, index=union_oos_index, name="cost")

        # Extra telemetry series (non-alpha).
        port_active_frac = pd.Series(active_frac, index=union_oos_index, name="active_frac")
        port_flat_flag = pd.Series(flat_flag, index=union_oos_index, name="flat_flag")
        port_gross_exposure = pd.Series(gross_exposure, index=union_oos_index, name="gross_exposure")
        port_net_exposure = pd.Series(net_exposure, index=union_oos_index, name="net_exposure")
        port_eff_n_bets = pd.Series(eff_n_bets, index=union_oos_index, name="eff_n_bets")
        port_sigma_med = pd.Series(sigma_med_daily, index=union_oos_index, name="sigma_median")
        port_sigma_p90 = pd.Series(sigma_p90_daily, index=union_oos_index, name="sigma_p90")
        port_sigma_collapse = pd.Series(sigma_collapse_daily, index=union_oos_index, name="sigma_collapse_pct")
        port_corr_abs_mu_sigma = pd.Series(corr_abs_mu_sigma_daily, index=union_oos_index, name="corr_abs_mu_sigma")

        # Cross-sectional prediction dispersion (entropy proxy).
        try:
            mu_entropy_vals = [
                _xsec_entropy_from_scores(mu_mat[i, :]) for i in range(len(union_oos_index))
            ]
            port_mu_entropy = pd.Series(mu_entropy_vals, index=union_oos_index, name="mu_entropy")
        except Exception:
            port_mu_entropy = pd.Series(index=union_oos_index, data=np.nan, name="mu_entropy")

        # Sigma inflation frequency: fraction of names with sigma above (median * mult).
        try:
            infl_mult = float(cfg.get("phase2_sigma_inflation_mult", 2.0))
            infl_mult = float(max(1.0, infl_mult))
            infl_vals: List[float] = []
            for i in range(len(union_oos_index)):
                s = np.asarray(sigma_mat[i, :], dtype=float)
                s = s[np.isfinite(s)]
                if s.size < 3:
                    infl_vals.append(float("nan"))
                    continue
                med = float(np.median(s))
                if not np.isfinite(med) or med <= 0:
                    infl_vals.append(float("nan"))
                    continue
                infl_vals.append(float(np.mean(s > med * infl_mult)))
            port_sigma_infl = pd.Series(infl_vals, index=union_oos_index, name="sigma_inflation_frac")
        except Exception:
            port_sigma_infl = pd.Series(index=union_oos_index, data=np.nan, name="sigma_inflation_frac")

        # Regime transition frequency across symbols.
        try:
            reg_mat = np.zeros((len(union_oos_index), len(syms)), dtype=float)
            for j, sym in enumerate(syms):
                s = regimes_by_sym.get(sym)
                if s is None:
                    reg_mat[:, j] = 0.0
                else:
                    reg_mat[:, j] = (
                        pd.to_numeric(s.reindex(union_oos_index), errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
            trans = np.full(len(union_oos_index), np.nan, dtype=float)
            for i in range(1, len(union_oos_index)):
                prev = reg_mat[i - 1, :]
                cur = reg_mat[i, :]
                m = np.isfinite(prev) & np.isfinite(cur)
                trans[i] = float(np.mean(cur[m] != prev[m])) if int(np.sum(m)) > 0 else float("nan")
            port_regime_transition = pd.Series(trans, index=union_oos_index, name="regime_transition_frac")
        except Exception:
            port_regime_transition = pd.Series(index=union_oos_index, data=np.nan, name="regime_transition_frac")

        try:
            flat_rate_full = float(
                np.mean(np.sum(np.abs(w_mat), axis=1) <= 1e-12)
                if w_mat.size and len(w_mat) == len(union_oos_index)
                else 0.0
            )
        except Exception:
            flat_rate_full = 0.0

        # Compute realized vol and vol underutilization penalty (hedge fund discipline)
        realized_vol = float(port_net.std() * np.sqrt(252)) if len(port_net) > 1 else 0.0
        target_vol = float(cfg.get("phase2_target_vol", 0.12))
        vol_underutil = max(0.0, target_vol - realized_vol) if target_vol > 0 else 0.0

        score = float(
            _annualized_sharpe(port_net)
            - float(objective_spec.max_drawdown_penalty) * _max_drawdown_from_equity(equity_curve)
            - float(lambda_turn) * float(port_turn.mean() if len(port_turn) else 0.0)
            - float(lambda_flat) * float(flat_rate_full)
            - float(objective_spec.vol_underutilization_penalty) * vol_underutil
        )

        portfolio_metrics = {
            "sharpe": float(_annualized_sharpe(port_net)),
            "max_drawdown": float(_max_drawdown_from_equity(equity_curve)),
            "turnover": float(port_turn.mean() if len(port_turn) else 0.0),
            "turnover_intent": float(np.mean(turnover_intent) if len(turnover_intent) else 0.0),
            "turnover_exec": float(np.mean(turnover_exec) if len(turnover_exec) else 0.0),
            "turnover_drift": float(np.mean(np.abs(turnover_intent - turnover_exec)) if len(turnover_intent) else 0.0),
            "flat_rate": float(flat_rate_full),
            "realized_vol": float(realized_vol),
            "target_vol": float(target_vol),
            "vol_underutil": float(vol_underutil),
            "score": float(score),
        }
        portfolio_equity = pd.DataFrame(
            {
                "net_return": port_net,
                "turnover": port_turn,
                "turnover_intent": pd.Series(turnover_intent, index=union_oos_index, name="turnover_intent"),
                "turnover_exec": pd.Series(turnover_exec, index=union_oos_index, name="turnover_exec"),
                "cost": port_cost,
                "active_frac": port_active_frac,
                "flat_flag": port_flat_flag,
                "gross_exposure": port_gross_exposure,
                "net_exposure": port_net_exposure,
                "eff_n_bets": port_eff_n_bets,
                "sigma_median": port_sigma_med,
                "sigma_p90": port_sigma_p90,
                "sigma_collapse_pct": port_sigma_collapse,
                "corr_abs_mu_sigma": port_corr_abs_mu_sigma,
                "mu_entropy": port_mu_entropy,
                "sigma_inflation_frac": port_sigma_infl,
                "regime_transition_frac": port_regime_transition,
                "overlay_dd": pd.Series(overlay_dd, index=union_oos_index, name="overlay_dd"),
                "overlay_rv": pd.Series(overlay_rv, index=union_oos_index, name="overlay_rv"),
                "overlay_dd_gross_scale": pd.Series(overlay_dd_gross_scale, index=union_oos_index, name="overlay_dd_gross_scale"),
                "overlay_vol_scale": pd.Series(overlay_vol_scale, index=union_oos_index, name="overlay_vol_scale"),
                "overlay_beta_neutralized": pd.Series(overlay_beta_neutralized, index=union_oos_index, name="overlay_beta_neutralized"),
                "overlay_flattened": pd.Series(overlay_flattened, index=union_oos_index, name="overlay_flattened"),
                "equity_curve": equity_curve,
            }
        )

        # Minimal per-symbol equity breakdown (weight + realized return + contribution).
        equity_by_symbol: Dict[str, pd.DataFrame] = {}
        per_symbol_metrics: Dict[str, Dict[str, float]] = {}
        w_df = pd.DataFrame(w_mat, index=union_oos_index, columns=syms)
        
        # ─────────────────────────────────────────────────────────────────────
        # GAP #8: Compute per-symbol calibration stats from accumulated data
        # ─────────────────────────────────────────────────────────────────────
        per_sym_calib: Dict[str, Dict[str, float]] = {}
        if per_symbol_calib_enabled and len(_per_sym_mu_acc) >= 10:
            try:
                # Stack accumulators: [n_matured_days, n_assets]
                mu_stack = np.vstack(_per_sym_mu_acc)
                sigma_stack = np.vstack(_per_sym_sigma_acc)
                realized_stack = np.vstack(_per_sym_realized_acc)
                
                for j, sym in enumerate(syms):
                    mu_j = mu_stack[:, j]
                    sigma_j = sigma_stack[:, j]
                    real_j = realized_stack[:, j]
                    
                    # Mask valid (non-NaN) observations
                    valid = np.isfinite(mu_j) & np.isfinite(sigma_j) & np.isfinite(real_j)
                    n_valid = int(np.sum(valid))
                    
                    if n_valid < 5:
                        per_sym_calib[sym] = {
                            "calib_n_obs": n_valid,
                            "calib_dir_acc": float("nan"),
                            "calib_corr": float("nan"),
                            "calib_coverage_1std": float("nan"),
                            "calib_sigma_ratio": float("nan"),
                        }
                        continue
                    
                    mu_v = mu_j[valid]
                    sigma_v = sigma_j[valid]
                    real_v = real_j[valid]
                    
                    # Directional accuracy: P(sign(μ) == sign(realized))
                    dir_acc = float(np.mean((mu_v > 0) == (real_v > 0)))
                    
                    # Correlation: Corr(μ, realized)
                    try:
                        corr = float(np.corrcoef(mu_v, real_v)[0, 1])
                        if not np.isfinite(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                    
                    # Sigma coverage: fraction within ±1σ
                    z_scores = np.abs(real_v - mu_v) / (sigma_v + 1e-12)
                    cov_1std = float(np.mean(z_scores <= 1.0))
                    
                    # Sigma calibration ratio: mean(σ²) / mean((r - μ)²)
                    mean_sigma_sq = float(np.mean(np.square(sigma_v)))
                    mean_sq_err = float(np.mean(np.square(real_v - mu_v)))
                    sigma_ratio = mean_sigma_sq / (mean_sq_err + 1e-12) if mean_sq_err > 1e-12 else 1.0
                    
                    per_sym_calib[sym] = {
                        "calib_n_obs": n_valid,
                        "calib_dir_acc": dir_acc,
                        "calib_corr": corr,
                        "calib_coverage_1std": cov_1std,
                        "calib_sigma_ratio": sigma_ratio,
                    }
            except Exception as e:
                logger.debug("[phase2.per_sym_calib] computation error: %s", e)
        
        for j, sym in enumerate(syms):
            r = returns_df[sym].reindex(union_oos_index).fillna(0.0)
            w_sym = w_df[sym].fillna(0.0)
            contrib = (w_sym.shift(1).fillna(0.0) * r).rename("contrib")
            eq = pd.DataFrame(
                {
                    "weight": w_sym,
                    "return": r,
                    "contrib": contrib,
                }
            )
            equity_by_symbol[sym] = eq
            
            # Base metrics
            base_metrics = {
                "avg_abs_weight": float(w_sym.abs().mean() if len(w_sym) else 0.0),
                "avg_return": float(r.mean() if len(r) else 0.0),
            }
            
            # Merge per-symbol calibration stats (GAP #8)
            if sym in per_sym_calib:
                base_metrics.update(per_sym_calib[sym])
            
            per_symbol_metrics[sym] = base_metrics

        if trial is not None:
            try:
                trial.set_user_attr("phase2_oos_sharpe", float(portfolio_metrics.get("sharpe", np.nan)))
                trial.set_user_attr("phase2_oos_max_drawdown", float(portfolio_metrics.get("max_drawdown", np.nan)))
                trial.set_user_attr("phase2_oos_turnover", float(portfolio_metrics.get("turnover", np.nan)))
                trial.set_user_attr("phase2_oos_realized_vol", float(portfolio_metrics.get("realized_vol", np.nan)))
                trial.set_user_attr("phase2_oos_target_vol", float(portfolio_metrics.get("target_vol", np.nan)))
                trial.set_user_attr("phase2_oos_vol_underutil", float(portfolio_metrics.get("vol_underutil", np.nan)))
                trial.set_user_attr("phase2_oos_flat_rate", float(portfolio_metrics.get("flat_rate", np.nan)))
                trial.set_user_attr("phase2_oos_score", float(portfolio_metrics.get("score", score)))
                trial.set_user_attr("phase2_engine", str(engine))
                trial.set_user_attr(
                    "phase2_fold_count",
                    int(
                        sum(
                            1
                            for r in (fold_logs or [])
                            if isinstance(r, dict) and str(r.get("kind")) == "fold"
                        )
                    ),
                )
                trial.set_user_attr("phase2_run_mode", str(run_mode))
                trial.set_user_attr("phase2_obj_lambda_turn", float(lambda_turn))
                trial.set_user_attr("phase2_obj_lambda_flat", float(lambda_flat))

                # High-level diagnostics summary for Optuna filtering.
                if diag_enabled:
                    try:
                        pref_mask = np.isfinite(sigma_med_daily) & (sigma_med_daily > 0)
                        sigma_med = float(np.median(sigma_med_daily[pref_mask])) if bool(np.any(pref_mask)) else float("nan")
                    except Exception:
                        sigma_med = float("nan")
                    try:
                        collapse = float(np.nanmean(sigma_collapse_daily)) if len(sigma_collapse_daily) else float("nan")
                    except Exception:
                        collapse = float("nan")
                    try:
                        corr_ms = float(np.nanmean(corr_abs_mu_sigma_daily)) if len(corr_abs_mu_sigma_daily) else float("nan")
                    except Exception:
                        corr_ms = float("nan")
                    # Update harm rate based on val_loss delta when pre/post eval enabled.
                    harm_rate = float("nan")
                    if diag_pre_post and update_logs:
                        deltas = [r.get("delta_val_loss") for r in update_logs if isinstance(r, dict) and r.get("delta_val_loss") is not None]
                        deltas_f = [float(x) for x in deltas if isinstance(x, (int, float)) and np.isfinite(x)]
                        if deltas_f:
                            harm_rate = float(np.mean(np.asarray(deltas_f) > 0.0))

                    trial.set_user_attr("phase2_diag_sigma_median", float(sigma_med))
                    trial.set_user_attr("phase2_diag_sigma_collapse_pct", float(collapse))
                    trial.set_user_attr("phase2_diag_corr_abs_mu_sigma", float(corr_ms))
                    if diag_pre_post:
                        trial.set_user_attr("phase2_diag_update_harm_rate", float(harm_rate))

                    # Canonical names for post-hoc analysis.
                    try:
                        trial.set_user_attr("sigma_median_mean", float(np.nanmean(sigma_med_daily)))
                    except Exception:
                        pass
                    try:
                        trial.set_user_attr("sigma_collapse_rate", float(np.nanmean(sigma_collapse_daily)))
                    except Exception:
                        pass
                    try:
                        trial.set_user_attr("eff_n_bets_mean", float(np.nanmean(eff_n_bets)))
                    except Exception:
                        pass
                    try:
                        trial.set_user_attr("flat_rate_mean", float(np.nanmean(flat_flag)))
                    except Exception:
                        pass
                    try:
                        trial.set_user_attr("active_frac_mean", float(np.nanmean(active_frac)))
                    except Exception:
                        pass
                    if diag_pre_post:
                        try:
                            trial.set_user_attr("update_harm_rate", float(harm_rate))
                        except Exception:
                            pass
            except Exception:
                pass

        # Optional JSON diagnostics artifact (kept separate from Optuna attrs).
        if diag_enabled and diag_write_json:
            try:
                diag_out_dir.mkdir(parents=True, exist_ok=True)
                trial_tag = None
                if trial is not None:
                    try:
                        trial_tag = f"trial{int(getattr(trial, 'number', -1))}"
                    except Exception:
                        trial_tag = None
                name = f"phase2_v2_diag_h{int(horizon)}_{trial_tag or 'single'}_{_phase2_cfg_signature_for_prepared(cfg)}.json"
                payload = {
                    "engine": str(engine),
                    "symbols": list(syms),
                    "horizon": int(horizon),
                    "seq_len": int(seq_len),
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "oos_start": str(oos_start),
                    "oos_end": str(oos_end),
                    "update_sessions": int(update_sessions),
                    "replay_days": int(replay_days),
                    "update_epochs": int(update_epochs),
                    "diagnostics": {
                        "pre_post_eval": bool(diag_pre_post),
                        "sigma_eps": float(diag_sigma_eps),
                    },
                    "updates": update_logs,
                    "folds": fold_logs,
                }
                (diag_out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
            except Exception:
                # Diagnostics must never break evaluation.
                pass

        logger.info(
            "[phase2.v2] walkforward complete: %d sessions, %d updates, final_equity=%.4f, sharpe=%.3f, max_dd=%.3f, realized_vol=%.3f, target_vol=%.3f, vol_underutil=%.4f, score=%.4f",
            n_oos,
            updates_done,
            float(equity[-1]) if len(equity) > 0 else np.nan,
            float(portfolio_metrics.get("sharpe", np.nan)),
            float(portfolio_metrics.get("max_drawdown", np.nan)),
            float(portfolio_metrics.get("realized_vol", np.nan)),
            float(portfolio_metrics.get("target_vol", np.nan)),
            float(portfolio_metrics.get("vol_underutil", np.nan)),
            float(score),
        )
        _cuda_mem_snapshot("after_v2_walkforward")

        # ------------------------------------------------------------
        # Persist first-class artifacts (hedge-fund polish)
        # ------------------------------------------------------------
        try:
            import os

            from src.analytics.benchmarking import BenchmarkFramework, compute_benchmark_timeseries, summarize_benchmark
            from src.analytics.eodhd_benchmark_data import fetch_eodhd_adjusted_close, prices_to_returns

            out_base = Path(str(os.getenv("PHASE2_PORTFOLIO_BACKTEST_DIR", "artifacts/backtests")))
            outdir = out_base / "PORTFOLIO" / f"h{int(horizon)}"
            outdir.mkdir(parents=True, exist_ok=True)

            eq_out = portfolio_equity.sort_index()
            eq_out.to_parquet(outdir / "bt_equity.parquet")

            # Benchmark config (defaults: SPY.US, adjusted close).
            bench_ticker = str(os.getenv("STAGEB_BENCHMARK_TICKER", "SPY")).strip().upper() or "SPY"
            bench_suffix = str(os.getenv("STAGEB_BENCHMARK_EXCHANGE_SUFFIX", ".US")).strip() or ".US"
            use_adj = str(os.getenv("STAGEB_BENCHMARK_USE_ADJUSTED", "1")).strip() not in {"0", "false", "False"}
            windows_raw = str(os.getenv("STAGEB_BENCHMARK_WINDOWS", "20,63,126"))
            windows = tuple(int(x.strip()) for x in windows_raw.split(",") if x.strip())
            rf_annual = float(os.getenv("STAGEB_BENCHMARK_RF_ANNUAL", "0.0"))
            cvar_alpha = float(os.getenv("STAGEB_BENCHMARK_CVAR_ALPHA", "0.05"))
            use_excess_alpha = str(os.getenv("STAGEB_BENCHMARK_USE_EXCESS_ALPHA", "0")).strip() in {"1", "true", "True"}

            rp = pd.to_numeric(eq_out.get("net_return"), errors="coerce").dropna().sort_index()
            if not rp.empty and isinstance(rp.index, pd.DatetimeIndex):
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
                    risk_free=rf_annual,
                    trading_days=252,
                    cvar_alpha=cvar_alpha,
                    target_vol_annual=None,
                    use_excess_alpha=use_excess_alpha,
                )

                # Join bt_equity telemetry (non-alpha diagnostics) onto benchmark timeseries.
                try:
                    extra_cols = [c for c in eq_out.columns if c not in {"net_return", "equity_curve"}]
                    if extra_cols:
                        extras = eq_out[extra_cols].copy()
                        extras.index = pd.to_datetime(extras.index, errors="coerce").tz_localize(None).normalize()
                        ts = ts.join(extras, how="left")
                except Exception:
                    pass

                ts.to_parquet(outdir / "bt_benchmark_timeseries.parquet")

                summary = summarize_benchmark(ts=ts, windows=windows)
                (outdir / "bt_benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                fw = BenchmarkFramework(primary=bench_ticker, secondary=(), trading_days=252)
                (outdir / "bt_benchmark_framework.json").write_text(json.dumps(fw.__dict__, indent=2, sort_keys=True) + "\n")

                # Explicit alpha vs beta decomposition artifacts.
                try:
                    cols: List[str] = ["rp", "rb", "active", "drawdown"]
                    for w in windows:
                        cols.extend(
                            [
                                f"beta_{int(w)}",
                                f"alpha_{int(w)}",
                                f"alpha_roll_{int(w)}",
                                f"alpha_cum_{int(w)}",
                                f"up_capture_{int(w)}",
                                f"down_capture_{int(w)}",
                            ]
                        )
                    cols = [c for c in cols if c in ts.columns]
                    if cols:
                        ts[cols].to_parquet(outdir / "bt_alpha_beta_timeseries.parquet")

                    # Whole-period capture ratios (committee-style, non-rolling).
                    capture_payload: Dict[str, Any] = {}
                    try:
                        rp0 = pd.to_numeric(ts.get("rp"), errors="coerce").dropna()
                        rb0 = pd.to_numeric(ts.get("rb"), errors="coerce").dropna()
                        x = pd.concat([rp0, rb0], axis=1).dropna()
                        if not x.empty:
                            pr = x.iloc[:, 0]
                            br = x.iloc[:, 1]
                            up = br > 0
                            dn = br < 0
                            if int(up.sum()) >= 3:
                                pr_c = float(np.prod(1.0 + pr[up]) - 1.0)
                                br_c = float(np.prod(1.0 + br[up]) - 1.0)
                                capture_payload["up_capture_full"] = float(pr_c / br_c) if abs(br_c) > 1e-12 else None
                            if int(dn.sum()) >= 3:
                                pr_c = float(np.prod(1.0 + pr[dn]) - 1.0)
                                br_c = float(np.prod(1.0 + br[dn]) - 1.0)
                                capture_payload["down_capture_full"] = float(pr_c / br_c) if abs(br_c) > 1e-12 else None
                    except Exception:
                        pass
                    (outdir / "bt_alpha_beta_summary.json").write_text(json.dumps({"summary": summary, **capture_payload}, indent=2, sort_keys=True) + "\n")
                except Exception:
                    pass

                # Universe rotation diagnostics (core/satellite gating + churn).
                try:
                    wt = float(cfg.get("phase2_rotation_weight_threshold", 1e-4))
                    rot_daily, rot_summary = _compute_universe_rotation_diagnostics(
                        equity_by_symbol=equity_by_symbol,
                        benchmark_returns=rb,
                        weight_threshold=wt,
                    )
                    if isinstance(rot_daily, pd.DataFrame) and not rot_daily.empty:
                        rot_daily.to_parquet(outdir / "bt_universe_rotation_daily.parquet")
                    (outdir / "bt_universe_rotation_summary.json").write_text(
                        json.dumps(rot_summary, indent=2, sort_keys=True) + "\n"
                    )
                except Exception:
                    pass

                # Model health & drift metrics (non-alpha).
                try:
                    health_cols = [
                        "active_frac",
                        "flat_flag",
                        "gross_exposure",
                        "net_exposure",
                        "eff_n_bets",
                        "sigma_median",
                        "sigma_p90",
                        "sigma_collapse_pct",
                        "corr_abs_mu_sigma",
                        "mu_entropy",
                        "sigma_inflation_frac",
                        "regime_transition_frac",
                        "overlay_dd",
                        "overlay_rv",
                        "overlay_dd_gross_scale",
                        "overlay_vol_scale",
                        "overlay_beta_neutralized",
                        "overlay_flattened",
                    ]
                    keep = [c for c in health_cols if c in eq_out.columns]
                    if keep:
                        eq_out[keep].to_parquet(outdir / "bt_model_health_daily.parquet")
                    mh = {}
                    try:
                        if "overlay_flattened" in eq_out.columns:
                            mh["pct_days_flattened"] = float(np.mean(pd.to_numeric(eq_out["overlay_flattened"], errors="coerce").fillna(0.0) > 0))
                        if "sigma_collapse_pct" in eq_out.columns:
                            mh["sigma_collapse_mean"] = float(pd.to_numeric(eq_out["sigma_collapse_pct"], errors="coerce").dropna().mean())
                        if "mu_entropy" in eq_out.columns:
                            mh["mu_entropy_mean"] = float(pd.to_numeric(eq_out["mu_entropy"], errors="coerce").dropna().mean())
                    except Exception:
                        pass
                    (outdir / "bt_model_health_summary.json").write_text(json.dumps(mh, indent=2, sort_keys=True) + "\n")
                except Exception:
                    pass

                # Formal house policy layer (explicit pass/fail envelope).
                try:
                    pol = _house_policy_report(
                        ts=ts,
                        windows=windows,
                        policy_max_drawdown=float(cfg.get("phase2_policy_max_drawdown", 0.20)),
                        policy_min_ir=float(cfg.get("phase2_policy_min_ir", 0.0)),
                        policy_beta_abs_max=float(cfg.get("phase2_policy_beta_abs_max", 0.30)),
                    )
                    (outdir / "bt_house_policy.json").write_text(json.dumps(pol, indent=2, sort_keys=True) + "\n")
                except Exception:
                    pass

        except Exception:
            # Artifact writes must never break evaluation.
            pass

        # ─────────────────────────────────────────────────────────────────────
        # Persist Mamba calibration governance state (with horizon-aware key).
        # ─────────────────────────────────────────────────────────────────────
        try:
            if mamba_calib_tracker is not None:
                calib_state = mamba_calib_tracker.serialize()
                save_mamba_calibration_state(governance_state_key, calib_state)
                logger.info("[phase2.governance] 💾 Saved Mamba calibration state: %s", governance_state_key)
        except Exception as e:
            logger.warning("[phase2.governance] Failed to persist calibration state: %s", e)

        # ─────────────────────────────────────────────────────────────────────
        # DAILY TRACE INSTRUMENTATION: Persist traces if enabled
        # ─────────────────────────────────────────────────────────────────────
        if trace_enabled and trace_persist and daily_traces:
            try:
                import gzip
                if trace_persist_path:
                    trace_path = Path(trace_persist_path)
                else:
                    trace_dir = Path("cache/debugging")
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    sym_hash = hashlib.sha256(",".join(sorted(syms)).encode()).hexdigest()[:8]
                    trace_path = trace_dir / f"daily_traces_h{horizon}_{sym_hash}_{ts_str}.json.gz"
                
                trace_dicts = [t.to_dict() for t in daily_traces]
                with gzip.open(trace_path, "wt", encoding="utf-8") as f:
                    json.dump(trace_dicts, f, indent=2, default=str)
                logger.info("[phase2.trace] 💾 Saved %d daily traces to: %s", len(daily_traces), trace_path)
            except Exception as e:
                logger.warning("[phase2.trace] Failed to persist traces: %s", e)

        # ─────────────────────────────────────────────────────────────────────
        # STRUCTURED EVENTS: Persist events and generate summary
        # ─────────────────────────────────────────────────────────────────────
        event_summary_str = None
        if events_enabled:
            try:
                event_summary_str = event_bus.generate_summary()
                logger.info("[phase2.events]\n%s", event_summary_str)
                
                if events_persist and event_bus.events:
                    event_dir = Path("cache/debugging")
                    event_dir.mkdir(parents=True, exist_ok=True)
                    ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    sym_hash = hashlib.sha256(",".join(sorted(syms)).encode()).hexdigest()[:8]
                    event_path = event_dir / f"events_h{horizon}_{sym_hash}_{ts_str}.json.gz"
                    
                    event_dicts = event_bus.to_list()
                    with gzip.open(event_path, "wt", encoding="utf-8") as f:
                        json.dump({
                            "events": event_dicts,
                            "summary": event_summary_str,
                            "counts_by_code": event_bus.count_by_code(),
                        }, f, indent=2, default=str)
                    logger.info("[phase2.events] 💾 Saved %d events to: %s", len(event_bus.events), event_path)
            except Exception as e:
                logger.warning("[phase2.events] Failed to process events: %s", e)

        # ─────────────────────────────────────────────────────────────────────
        # Z-EXPLAINER: Fit interpretable model on accumulated data
        # ─────────────────────────────────────────────────────────────────────
        z_explainer_report: Optional[Dict[str, Any]] = None
        if z_explainer_enabled and z_explainer_buffer is not None and len(z_explainer_buffer) > 20:
            try:
                from src.stage_b_stateful.z_explainer import (
                    LinearZExplainer,
                    generate_explanation_report,
                )
                
                # Build feature matrix
                X, y, index = z_explainer_buffer.build_feature_matrix()
                
                if len(y) >= 50:
                    # Fit linear explainer
                    explainer = LinearZExplainer(regularization=0.1)
                    explainer.fit(X, y, standardize=True)
                    
                    # Generate report
                    z_explainer_report = generate_explanation_report(
                        explainer=explainer,
                        buffer=z_explainer_buffer,
                        top_k_features=15,
                        top_k_samples=25,
                    )
                    
                    # Log top features
                    top_features = z_explainer_report.get("feature_importance", [])[:5]
                    if top_features:
                        logger.info(
                            "[phase2.z_explainer] R²=%.3f | Top features: %s",
                            explainer.r2_score,
                            ", ".join(f"{f['feature']}={f['coefficient']:.3f}" for f in top_features)
                        )
                    
                    # Save report to diagnostics
                    if diagnostics_enabled:
                        try:
                            import json
                            from pathlib import Path
                            output_dir = Path("cache/debugging")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            report_path = output_dir / f"z_explainer_report_{run_id if 'run_id' in dir() else 'latest'}.json"
                            with open(report_path, "w") as f:
                                json.dump(z_explainer_report, f, indent=2, default=str)
                            logger.info("[phase2.z_explainer] 📊 Saved report to: %s", report_path)
                        except Exception as e:
                            logger.debug("[phase2.z_explainer] Failed to save report: %s", e)
                else:
                    logger.info("[phase2.z_explainer] Insufficient samples for fitting: %d", len(y))
            except Exception as e:
                logger.warning("[phase2.z_explainer] Failed to fit explainer: %s", e)

        # ─────────────────────────────────────────────────────────────────────
        # DIAGNOSTICS ARTIFACTS: Write structured outputs for dashboard
        # ─────────────────────────────────────────────────────────────────────
        diagnostics_enabled = bool(cfg.get("phase2_diagnostics_enabled", True))
        if diagnostics_enabled:
            try:
                from src.stage_b_stateful.diagnostics_writer import (
                    generate_run_id,
                    write_all_diagnostics,
                )
                
                # Generate unique run ID
                trial_num = trial.number if trial is not None else None
                run_id = generate_run_id(syms, int(horizon), trial_num)
                
                # Write all artifacts
                bundle = write_all_diagnostics(
                    run_id=run_id,
                    traces=[t.to_dict() for t in daily_traces] if daily_traces else None,
                    events=event_bus.to_list() if events_enabled else None,
                    event_counts=event_bus.count_by_code() if events_enabled else None,
                    portfolio_metrics=portfolio_metrics,
                    symbols=syms,
                    horizon=int(horizon),
                    config=dict(cfg) if cfg else None,
                    z_explainer_logs=z_explainer_logs if z_explainer_logs else None,
                )
                logger.info("[phase2.diagnostics] 📊 Wrote diagnostics bundle: run_id=%s", run_id)
                
                # Store run_id in trial for reference
                if trial is not None:
                    try:
                        trial.set_user_attr("phase2_diagnostics_run_id", run_id)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("[phase2.diagnostics] Failed to write diagnostics: %s", e)

        # ─────────────────────────────────────────────────────────────────────
        # LINEAR ALPHA COMBINER: Save state for persistence
        # ─────────────────────────────────────────────────────────────────────
        if linear_state is not None and linear_state.is_ready():
            try:
                save_linear_combiner_state(
                    linear_state.to_dict(),
                    horizon=int(horizon),
                    symbols=syms,
                )
                logger.info(
                    "[phase2.linear] Saved state: n_updates=%d, R²=%.3f",
                    linear_state._n_updates, linear_state.model.r_squared_
                )
            except Exception as e:
                logger.warning("[phase2.linear] Failed to save state: %s", e)

        if device.type == "cuda":
            try:
                del model
            except Exception:
                pass
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # ─────────────────────────────────────────────────────────────────────
        # RISK EVENTS LEDGER: Flush and generate run summary
        # ─────────────────────────────────────────────────────────────────────
        if risk_events_ledger is not None:
            try:
                # Add final session end marker
                risk_events_ledger.add_event(
                    date=str(pd.Timestamp(union_oos_index[-1]))[:10] if len(union_oos_index) > 0 else "unknown",
                    day_idx=len(union_oos_index) - 1,
                    event_type="RUN_END",
                    action_taken="NONE",
                    details={
                        "total_days": len(union_oos_index),
                        "final_score": float(score),
                        "final_sharpe": float(portfolio_metrics.get("sharpe", 0.0)),
                        "final_max_dd": float(portfolio_metrics.get("max_drawdown", 0.0)),
                    },
                )
                
                # Flush all remaining events
                flushed = risk_events_ledger.flush()
                
                # Get summary
                ledger_summary = risk_events_ledger.get_summary()
                logger.info(
                    "[phase2.risk_events_ledger] 📋 Flushed %d events to %s | Total: %d events, Types: %d, Actions: %d",
                    flushed,
                    risk_events_ledger.path,
                    ledger_summary.get("total_events", 0),
                    len(ledger_summary.get("events_by_type", {})),
                    len(ledger_summary.get("events_by_action", {})),
                )
                
                # Store summary in portfolio_metrics for downstream access
                portfolio_metrics["risk_events_summary"] = ledger_summary
                portfolio_metrics["risk_events_path"] = str(risk_events_ledger.path)
                
            except Exception as e:
                logger.warning("[phase2.risk_events_ledger] Failed to flush/summarize: %s", e)

        return Phase2Result(
            objective=float(score),
            per_symbol_metrics=per_symbol_metrics,
            portfolio_metrics=portfolio_metrics,
            preds_by_symbol=preds_by_symbol,
            equity_by_symbol=equity_by_symbol,
            portfolio_equity=portfolio_equity,
            daily_traces=daily_traces if trace_enabled else None,
            events=event_bus.to_list() if events_enabled else None,
            event_summary=event_summary_str,
            z_explainer_logs=z_explainer_logs if z_explainer_logs else None,
        )

    # ------------------------------------------------------------------
    # Legacy fallback path (train once + per-symbol backtests)
    # ------------------------------------------------------------------
    if bool(cfg.get("phase2_require_v2", False)):
        raise ValueError("phase2_require_v2=True but phase2_engine is not v2")

    # ─────────────────────────────────────────────────────────────────────
    # Multi-Horizon Training Branch (Workstream 6)
    # ─────────────────────────────────────────────────────────────────────
    multi_horizon_mode = bool(cfg.get("multi_horizon_mode", DEFAULT_MULTI_HORIZON_MODE))
    multi_horizon_set = list(cfg.get("multi_horizon_set", DEFAULT_MULTI_HORIZON_SET))
    multi_horizon_primary = int(cfg.get("multi_horizon_primary", DEFAULT_MULTI_HORIZON_PRIMARY))
    
    if multi_horizon_mode:
        logger.info(
            f"[MH] Multi-horizon training ENABLED: horizons={multi_horizon_set}, primary={multi_horizon_primary}"
        )
        
        # Build multi-horizon sequence data
        mh_seq_data = _build_pooled_multi_horizon_sequence_data(
            symbols=syms,
            horizons=multi_horizon_set,
            seq_len=seq_len,
            prepared=prepared,
            pipelines_by=prepared.pipelines_by,
            panels_by=prepared.trackc_aligned_by,
            train_pos_by=prepared.train_pos_by,
            label_type=str(getattr(prepared, "label_id", "base")),
        )
        
        # Train/val split
        n_train = mh_seq_data.n_samples
        train_fraction = float(np.clip(float(cfg.get("train_fraction", 0.9)), 0.5, 0.95))
        split = int(n_train * train_fraction)
        train_idx = np.arange(split)
        holdout_idx = np.arange(split, n_train)
        if len(holdout_idx) < 5:
            raise ValueError("train holdout too small for multi-horizon")
        
        # Map single-horizon cfg keys to multi-horizon keys
        cfg_train = dict(cfg)
        cfg_train["mh_d_model"] = cfg_train.get("mamba_d_model", 128)
        cfg_train["mh_n_layers"] = cfg_train.get("mamba_n_layers", 4)
        cfg_train["mh_dropout"] = cfg_train.get("mamba_dropout", 0.1)
        cfg_train["mh_batch_size"] = cfg_train.get("mamba_batch_size", 32)
        cfg_train["mh_max_epochs"] = cfg_train.get("mamba_max_epochs", 10)
        cfg_train["mh_learning_rate"] = cfg_train.get("mamba_learning_rate", 1e-4)
        cfg_train["mh_weight_decay"] = cfg_train.get("mamba_weight_decay", 1e-4)
        cfg_train["mh_grad_clip"] = cfg_train.get("mamba_grad_clip", 1.0)
        cfg_train["mh_head_hidden_dim"] = cfg_train.get("mamba_head_hidden_dim", 64)
        cfg_train["mh_head_num_layers"] = cfg_train.get("mamba_head_num_layers", 2)
        
        if deterministic_enabled:
            cfg_train["torch_deterministic"] = True
            cfg_train["torch_disable_tf32"] = True
            cfg_train["torch_seed"] = int(seed)
        
        train_result = train_multi_horizon_mamba(
            mh_seq_data, train_idx, holdout_idx, cfg_train, device=device, return_model=True
        )
        mh_model = train_result.get("model")
        if mh_model is None:
            raise ValueError("multi-horizon training failed: no model")
        
        logger.info(f"[MH] Training complete. Using primary horizon={multi_horizon_primary} for inference.")
        
        _cuda_mem_snapshot("after_mh_train")
        
        # Standardize full Track-C per symbol using pooled scaler stats.
        features_std_by = dict(prepared.features_std_full_by)
        index_by = dict(prepared.index_by)
        union_oos_index = prepared.union_oos_index
        burnin_end_ts = pd.to_datetime(oos_start) - pd.Timedelta(days=1)
        
        # Multi-horizon inference: use only primary horizon for downstream
        preds_by_symbol, sigma_by_symbol = _stateful_predict_multi_horizon_primary(
            model=mh_model,
            device=device,
            primary_horizon=multi_horizon_primary,
            features_std_by_symbol=features_std_by,
            index_by_symbol=index_by,
            union_oos_index=union_oos_index,
            burnin_end_ts=pd.Timestamp(burnin_end_ts),
            seq_len=seq_len,
        )
        
        _cuda_mem_snapshot("after_mh_stateful_predict")
        
        # Model reference for downstream (use the multi-horizon model)
        model = mh_model
        
    else:
        # ─────────────────────────────────────────────────────────────────
        # Single-Horizon Training (existing behavior unchanged)
        # ─────────────────────────────────────────────────────────────────
        if prepared.gpu_store_train is not None and device.type == "cuda":
            pooled_seq = WindowedSequenceData(store=prepared.gpu_store_train, seq_len=seq_len, samples=samples, timestamps=ts)
        else:
            # CPU fallback (keeps legacy behavior)
            pooled_seq = _build_pooled_sequence_cpu(
                symbols=syms,
                seq_len=seq_len,
                prepared=prepared,
            )

        n_train = int(len(pooled_seq))
        try:
            train_fraction = float(cfg.get("train_fraction", 0.9))
        except Exception:
            train_fraction = 0.9
        train_fraction = float(np.clip(train_fraction, 0.5, 0.95))
        split = int(n_train * train_fraction)
        train_idx = np.arange(split)
        holdout_idx = np.arange(split, n_train)
        if len(holdout_idx) < 5:
            raise ValueError("train holdout too small")

        cfg_train = dict(cfg)
        if deterministic_enabled:
            cfg_train["torch_deterministic"] = True
            cfg_train["torch_disable_tf32"] = True
            cfg_train["torch_seed"] = int(seed)

        train_result = train_mamba_fold(pooled_seq, train_idx, holdout_idx, cfg_train, device=device, return_model=True)
        model = train_result.get("model")
        if model is None:
            raise ValueError("training failed: no model")

        _cuda_mem_snapshot("after_train")

        # Standardize full Track-C per symbol using pooled scaler stats.
        features_std_by = dict(prepared.features_std_full_by)
        index_by = dict(prepared.index_by)

        # Build union OOS index (fixed, shared across symbols) and batched stateful inference.
        union_oos_index = prepared.union_oos_index

        burnin_end_ts = pd.to_datetime(oos_start) - pd.Timedelta(days=1)
        
        # Use distributional inference for true mu + sigma when model supports it
        head_type = str(cfg.get("mamba_head_type", "linear")).lower()
        sigma_by_symbol: Optional[Dict[str, pd.Series]] = None
        
        if head_type in ("gaussian", "uncertainty", "gaussian_nll"):
            try:
                preds_by_symbol, sigma_by_symbol = _stateful_predict_batched_mu_sigma(
                    model=model,
                    device=device,
                    features_std_by_symbol=features_std_by,
                    index_by_symbol=index_by,
                    union_oos_index=union_oos_index,
                    burnin_end_ts=pd.Timestamp(burnin_end_ts),
                    seq_len=seq_len,
                    sigma_floor=float(cfg.get("phase2_sigma_floor_base", 1e-6)),
                    sigma_cap=float(cfg.get("phase2_sigma_cap", 10.0)),
                )
            except Exception as e:
                logger.warning(f"Distributional inference failed, falling back to scalar: {e}")
                preds_by_symbol = _stateful_predict_batched_across_symbols(
                    model=model,
                    device=device,
                    features_std_by_symbol=features_std_by,
                    index_by_symbol=index_by,
                    union_oos_index=union_oos_index,
                    burnin_end_ts=pd.Timestamp(burnin_end_ts),
                    seq_len=seq_len,
                )
        else:
            preds_by_symbol = _stateful_predict_batched_across_symbols(
                model=model,
                device=device,
                features_std_by_symbol=features_std_by,
                index_by_symbol=index_by,
                union_oos_index=union_oos_index,
                burnin_end_ts=pd.Timestamp(burnin_end_ts),
                seq_len=seq_len,
            )

        _cuda_mem_snapshot("after_stateful_predict")

    def _backtest_all(
        preds_by_symbol_local: Mapping[str, pd.Series],
        sigma_by_symbol_local: Optional[Dict[str, pd.Series]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
        per_symbol_metrics_local: Dict[str, Dict[str, float]] = {}
        equity_by_symbol_local: Dict[str, pd.DataFrame] = {}

        # Backtest per symbol (same logic as Stage-B), using neutral predictions on missing-feature days.
        for sym in syms:
            pipe = prepared.pipelines_by[sym]
            labels = prepared.labels_by[sym]
            preds = preds_by_symbol_local[sym]

            price_data = pipe._get_price_data_for_horizon(int(horizon))
            if price_data is None or price_data.empty:
                raise ValueError(f"missing price data for backtest: {sym}")

            strategy_cfg = _make_strategy_cfg_from_base(cfg)
            fee = float(strategy_cfg.get("fee_bp", 0.0)) / 10000.0
            slippage = float(strategy_cfg.get("slippage_bp", 0.0)) / 10000.0
            engine = BacktestEngine(price_data, fee=fee, slippage_bp=slippage)

            actual_returns = labels["forward_return"].reindex(preds.index).astype(float)
            preds_df = pd.DataFrame(index=preds.index)
            preds_df["actual_return"] = actual_returns
            preds_df["mu_hat"] = preds.astype(float)

            # Compute sigma_proxy as fallback for missing/invalid model sigma
            sigma_proxy = actual_returns.abs().rolling(window=max(5, int(horizon))).std().fillna(0.02).clip(lower=1e-4)
            
            # Use true model sigma if available from distributional inference
            if sigma_by_symbol_local is not None and sym in sigma_by_symbol_local:
                model_sigma = sigma_by_symbol_local[sym].reindex(preds.index)
                # Use model sigma where valid, proxy as fallback
                sigma_hat = model_sigma.where(model_sigma.notna() & np.isfinite(model_sigma), sigma_proxy)
                preds_df["sigma_source"] = np.where(
                    model_sigma.notna() & np.isfinite(model_sigma), 
                    "model", 
                    "proxy"
                )
            else:
                sigma_hat = sigma_proxy
                preds_df["sigma_source"] = "proxy"
            
            denom = sigma_hat.replace(0.0, np.nan).fillna(0.02)
            logits = (preds / denom).clip(-8, 8)
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds_df["p_up"] = probs.clip(0.0, 1.0)
            preds_df["sigma_hat"] = sigma_hat
            preds_df["rho"] = (preds_df["p_up"] - 0.5).abs().mul(2.0).clip(0.0, 1.0)
            preds_df["drift_flag"] = 0.0

            for col in ["mu_hat", "p_up", "sigma_hat"]:
                _assert_finite_series(name=f"preds_df[{col}]", symbol=sym, s=preds_df[col])

            equity, metrics = engine.run(preds_df, int(horizon), strategy_cfg, symbol=sym)
            equity_by_symbol_local[sym] = equity
            per_symbol_metrics_local[sym] = {k: float(v) for k, v in metrics.items()}

        return equity_by_symbol_local, per_symbol_metrics_local

    def _aggregate_portfolio(
        equity_by_symbol_local: Mapping[str, pd.DataFrame],
    ) -> Tuple[float, Dict[str, float], pd.DataFrame]:
        # Aggregate portfolio daily net returns.
        net_cols = []
        tw_cols = []
        for sym, eq in equity_by_symbol_local.items():
            if "net_return" not in eq.columns or "target_weight" not in eq.columns:
                raise ValueError(f"missing net_return/target_weight in equity for {sym}")
            net_cols.append(eq["net_return"].rename(sym))
            tw_cols.append(eq["target_weight"].rename(sym))

        net_df = pd.concat(net_cols, axis=1).sort_index().fillna(0.0)
        tw_df = pd.concat(tw_cols, axis=1).sort_index().fillna(0.0)
        _assert_finite_np(name="portfolio.net_df", symbol="PORTFOLIO", values=net_df.to_numpy(dtype=float), index=pd.DatetimeIndex(net_df.index))

        aligned_weights = {sym: weights.get(sym, 0.0) for sym in net_df.columns}
        w_vec = np.array([aligned_weights[c] for c in net_df.columns], dtype=float)
        portfolio_net = pd.Series(net_df.to_numpy(dtype=float) @ w_vec, index=net_df.index, name="net_return")
        _assert_finite_series(name="portfolio.net_return", symbol="PORTFOLIO", s=portfolio_net)

        # Portfolio turnover: sum_i alloc_i * |Δ target_weight_i|.
        tw_delta = tw_df.diff().abs().fillna(tw_df.abs())
        port_turnover_daily = pd.Series(tw_delta.to_numpy(dtype=float) @ w_vec, index=tw_df.index, name="turnover")
        _assert_finite_series(name="portfolio.turnover", symbol="PORTFOLIO", s=port_turnover_daily)

        portfolio_equity_curve = (1.0 + portfolio_net).cumprod().rename("equity_curve")
        max_dd = _max_drawdown_from_equity(portfolio_equity_curve)
        sharpe = _annualized_sharpe(portfolio_net)
        turnover_mean = float(port_turnover_daily.mean()) if len(port_turnover_daily) else 0.0
        
        # Compute realized vol and vol underutilization penalty (hedge fund discipline)
        logger.info(f"[DEBUG] portfolio_net len={len(portfolio_net)}, std={portfolio_net.std()}, sample={portfolio_net.head(3).tolist()}")
        realized_vol = float(portfolio_net.std() * np.sqrt(252)) if len(portfolio_net) > 1 else 0.0
        target_vol = float(cfg.get("phase2_target_vol", 0.12))
        logger.info(f"[DEBUG] realized_vol={realized_vol}, target_vol={target_vol}, cfg.phase2_target_vol={cfg.get('phase2_target_vol')}")
        vol_underutil = max(0.0, target_vol - realized_vol) if target_vol > 0 else 0.0

        score_local = float(
            sharpe 
            - objective_spec.max_drawdown_penalty * max_dd 
            - objective_spec.turnover_penalty * turnover_mean
            - objective_spec.vol_underutilization_penalty * vol_underutil
        )
        portfolio_metrics_local = {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "turnover": float(turnover_mean),
            "realized_vol": float(realized_vol),
            "target_vol": float(target_vol),
            "vol_underutil": float(vol_underutil),
            "score": float(score_local),
        }

        portfolio_equity_local = pd.DataFrame(
            {
                "net_return": portfolio_net,
                "turnover": port_turnover_daily,
                "equity_curve": portfolio_equity_curve,
            }
        )

        return score_local, portfolio_metrics_local, portfolio_equity_local

    equity_by_symbol, per_symbol_metrics = _backtest_all(preds_by_symbol, sigma_by_symbol)
    
    # Re-aggregate portfolio with vol underutilization penalty
    score, portfolio_metrics, portfolio_equity = _aggregate_portfolio(equity_by_symbol)

    # ------------------------------------------------------------------
    # Fold-based intermediate reporting for pruning.
    # We compute the full backtest once, then score prefixes to avoid re-running
    # the engine for every fold.
    # ------------------------------------------------------------------
    def _build_portfolio_series(
        equity_by_symbol_local: Mapping[str, pd.DataFrame],
    ) -> Tuple[pd.Series, pd.Series]:
        net_cols = []
        tw_cols = []
        for sym, eq in equity_by_symbol_local.items():
            if "net_return" not in eq.columns or "target_weight" not in eq.columns:
                raise ValueError(f"missing net_return/target_weight in equity for {sym}")
            net_cols.append(eq["net_return"].rename(sym))
            tw_cols.append(eq["target_weight"].rename(sym))

        net_df = pd.concat(net_cols, axis=1).sort_index().fillna(0.0)
        tw_df = pd.concat(tw_cols, axis=1).sort_index().fillna(0.0)
        _assert_finite_np(
            name="portfolio.net_df",
            symbol="PORTFOLIO",
            values=net_df.to_numpy(dtype=float),
            index=pd.DatetimeIndex(net_df.index),
        )

        aligned_weights = {sym: weights.get(sym, 0.0) for sym in net_df.columns}
        w_vec = np.array([aligned_weights[c] for c in net_df.columns], dtype=float)
        portfolio_net = pd.Series(net_df.to_numpy(dtype=float) @ w_vec, index=net_df.index, name="net_return")
        _assert_finite_series(name="portfolio.net_return", symbol="PORTFOLIO", s=portfolio_net)

        tw_delta = tw_df.diff().abs().fillna(tw_df.abs())
        port_turnover_daily = pd.Series(tw_delta.to_numpy(dtype=float) @ w_vec, index=tw_df.index, name="turnover")
        _assert_finite_series(name="portfolio.turnover", symbol="PORTFOLIO", s=port_turnover_daily)

        return portfolio_net, port_turnover_daily

    portfolio_net_full, portfolio_turnover_full = _build_portfolio_series(equity_by_symbol)

    def _score_prefix(prefix_end: pd.Timestamp) -> Tuple[float, Dict[str, float], pd.DataFrame]:
        end_ts = pd.Timestamp(prefix_end)
        mask = portfolio_net_full.index <= end_ts
        r = portfolio_net_full.loc[mask]
        t = portfolio_turnover_full.loc[mask]
        portfolio_equity_curve = (1.0 + r).cumprod().rename("equity_curve")
        max_dd = _max_drawdown_from_equity(portfolio_equity_curve)
        sharpe = _annualized_sharpe(r)
        turnover_mean = float(t.mean()) if len(t) else 0.0

        score_local = float(
            sharpe
            - objective_spec.max_drawdown_penalty * max_dd
            - objective_spec.turnover_penalty * turnover_mean
        )
        metrics_local = {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "turnover": float(turnover_mean),
            "score": float(score_local),
        }
        equity_local = pd.DataFrame({"net_return": r, "turnover": t, "equity_curve": portfolio_equity_curve})
        return score_local, metrics_local, equity_local

    # Fold schedule: report score every `prune_update_sessions` sessions.
    # Uses the union OOS session index (already aligned with panel trading days).
    oos_idx = pd.DatetimeIndex(union_oos_index)
    oos_idx = pd.DatetimeIndex(pd.to_datetime(oos_idx)).sort_values()

    # Label maturity rule (critical to avoid leakage in intermediate reporting):
    # The label for day t (forward H-session return) becomes known at t + H.
    # Therefore, at boundary T we can only score predictions with t <= T - H.
    horizon_sessions = max(1, int(horizon))

    warmup_folds = max(0, int(prune_warmup_folds))
    safety_after = max(0, int(safety_prune_after_folds))
    safety_sharpe_floor = float(safety_prune_sharpe_floor)
    safety_maxdd_ceiling = float(safety_prune_maxdd_ceiling)
    step_sessions_raw = int(prune_update_sessions)
    step_sessions = int(step_sessions_raw)

    if trial is not None:
        try:
            trial.set_user_attr("phase2_prune_update_sessions", int(step_sessions))
            trial.set_user_attr("phase2_prune_warmup_folds", int(warmup_folds))
            trial.set_user_attr("phase2_safety_prune_after_folds", int(safety_after))
            trial.set_user_attr("phase2_safety_prune_sharpe_floor", float(safety_sharpe_floor))
            trial.set_user_attr("phase2_safety_prune_maxdd_ceiling", float(safety_maxdd_ceiling))
        except Exception:
            pass

    # Legacy pruning mode (OOS-prefix days) - only if fold reporting is disabled.
    # NOTE: fold reporting uses step>=1; avoid conflicting step numbers.
    if trial is not None and prune_oos_days is not None and step_sessions_raw <= 0:
        n = int(prune_oos_days)
        if n >= 10 and len(oos_idx) > n:
            end_ts = pd.Timestamp(oos_idx[n - 1])
            short_score, _short_portfolio_metrics, _short_portfolio_equity = _score_prefix(end_ts)
            try:
                trial.report(float(short_score), step=1)
                trial.set_user_attr("phase2_prune_oos_days", int(n))
                trial.set_user_attr("phase2_prune_oos_score", float(short_score))
            except Exception:
                pass
            if trial.user_attrs.get("rerun_kind") != "top10_replay":
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

    # Primary: fold-based intermediate reporting.
    fold_end_positions: List[int] = []
    if step_sessions_raw > 0:
        fold_end_positions = list(range(step_sessions, len(oos_idx) + 1, step_sessions))
        if not fold_end_positions or fold_end_positions[-1] != len(oos_idx):
            fold_end_positions.append(len(oos_idx))

    # Never prune explicit reruns (we want full evaluations for diagnostics).
    allow_prune = bool(trial is not None and trial.user_attrs.get("rerun_kind") != "top10_replay")

    fold_scores: List[float] = []
    for fold_num, end_pos in enumerate(fold_end_positions, start=1):
        end_ts = pd.Timestamp(oos_idx[int(end_pos) - 1])

        # Score only matured labels at this boundary.
        try:
            mature_end_ts = add_sessions(end_ts, -int(horizon_sessions))
        except Exception:
            # Fallback if session arithmetic fails (should be rare); use calendar days.
            mature_end_ts = pd.Timestamp(end_ts) - pd.Timedelta(days=int(horizon_sessions))

        fold_score, fold_metrics, _fold_equity = _score_prefix(mature_end_ts)
        fold_scores.append(float(fold_score))
        if trial is not None:
            try:
                trial.report(float(fold_score), step=int(fold_num))
            except Exception:
                pass

            # Optional safety prune rule (independent of Hyperband):
            # after N folds, prune if Sharpe is too low AND MaxDD is too high.
            if allow_prune and safety_after > 0 and fold_num >= safety_after:
                try:
                    sharpe = float(fold_metrics.get("sharpe", 0.0))
                    max_dd = float(fold_metrics.get("max_drawdown", 0.0))
                except Exception:
                    sharpe = 0.0
                    max_dd = 0.0
                if np.isfinite(sharpe) and np.isfinite(max_dd):
                    if sharpe < safety_sharpe_floor and max_dd > safety_maxdd_ceiling:
                        try:
                            trial.set_user_attr("phase2_safety_pruned_at_fold", int(fold_num))
                            trial.set_user_attr("phase2_safety_pruned_sharpe", float(sharpe))
                            trial.set_user_attr("phase2_safety_pruned_maxdd", float(max_dd))
                        except Exception:
                            pass
                        import optuna

                        raise optuna.TrialPruned()

            if allow_prune and fold_num >= warmup_folds:
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

    # Final score uses the full OOS span.
    score, portfolio_metrics, portfolio_equity = _score_prefix(pd.Timestamp(oos_idx[-1]))

    # Persist final OOS portfolio metric decomposition for post-hoc diagnosis.
    # (Older trials may not have these attrs; new trials will.)
    if trial is not None:
        try:
            trial.set_user_attr("phase2_oos_sharpe", float(portfolio_metrics.get("sharpe", np.nan)))
            trial.set_user_attr("phase2_oos_max_drawdown", float(portfolio_metrics.get("max_drawdown", np.nan)))
            trial.set_user_attr("phase2_oos_turnover", float(portfolio_metrics.get("turnover", np.nan)))
            trial.set_user_attr("phase2_oos_score", float(portfolio_metrics.get("score", score)))
        except Exception:
            pass

    _cuda_mem_snapshot("after_backtest")

    # Final report already emitted as the last fold step.
    if trial is not None:
        try:
            trial.set_user_attr("phase2_fold_count", int(len(fold_end_positions)))
            trial.set_user_attr("phase2_fold_scores_tail", [float(x) for x in fold_scores[-5:]])
        except Exception:
            pass

    # Record memory telemetry on the trial (helps diagnose rare long-running trials / VRAM pressure).
    if trial is not None and device.type == "cuda":
        try:
            snap_end = _cuda_mem_snapshot("trial_end_before_cleanup") or {}
            trial.set_user_attr("cuda_mem_end", snap_end)
        except Exception:
            pass

    # Best-effort end-of-trial cleanup (does not clear the prepared GPU master store,
    # but can release transient allocations held by the CUDA caching allocator).
    if device.type == "cuda":
        try:
            del model
        except Exception:
            pass
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if trial is not None:
            try:
                snap_post = _cuda_mem_snapshot("trial_end_after_empty_cache") or {}
                trial.set_user_attr("cuda_mem_post_empty_cache", snap_post)
            except Exception:
                pass

    return Phase2Result(
        objective=score,
        per_symbol_metrics=per_symbol_metrics,
        portfolio_metrics=portfolio_metrics,
        preds_by_symbol=preds_by_symbol,
        equity_by_symbol=equity_by_symbol,
        portfolio_equity=portfolio_equity,
    )


_PHASE2_PREPARED_CACHE: Dict[Tuple[str, Tuple[str, ...], int, str, str, str, str], Phase2PreparedData] = {}
_PHASE2_PREPARED_CACHE_BY_CFG: Dict[Tuple[Tuple[str, ...], int, str, str, str, str, str], Phase2PreparedData] = {}


def _prepare_phase2_once(
    *,
    best_trial_json: Path,
    symbols: Sequence[str],
    horizon: int,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
) -> Phase2PreparedData:
    """Prepare expensive Phase-2 artifacts once and reuse across trials."""

    key = (
        str(best_trial_json),
        tuple([str(s).upper() for s in symbols]),
        int(horizon),
        str(train_start),
        str(train_end),
        str(oos_start),
        str(oos_end),
    )
    cached = _PHASE2_PREPARED_CACHE.get(key)
    if cached is not None:
        return cached

    base_params = load_stage_b_best_trial_params(best_trial_json)
    label_id = _phase2_label_id_from_cfg(base_params)
    label_col = _phase2_label_column(label_id)

    # Build Track-C per symbol once (mamba refinement keys do not affect Track-C construction).
    pipelines_by, _panels_by, labels_by, trackc_by, post_std_weights_by, train_pos_by, _ = _build_trackc_multi_symbol(
        symbols=[str(s).upper() for s in symbols],
        horizon=int(horizon),
        train_start=train_start,
        train_end=train_end,
        oos_start=oos_start,
        oos_end=oos_end,
        cfg=dict(base_params),
    )

    # Align Track-C schemas across symbols (ordered union).
    syms = [str(s).upper() for s in symbols]
    ordered_cols: List[str] = list(trackc_by[syms[0]].columns)
    seen_cols = set(ordered_cols)
    for sym in syms[1:]:
        df = trackc_by.get(sym)
        if df is None:
            continue
        for c in df.columns:
            if c not in seen_cols:
                ordered_cols.append(c)
                seen_cols.add(c)

    trackc_aligned_by: Dict[str, pd.DataFrame] = {sym: _coerce_trackc_numeric_schema(trackc_by[sym], ordered_cols) for sym in syms}

    # Align per-symbol post-std weights (1D vector or 2D matrix) to the aligned Track-C numeric schema.
    post_std_feature_weights_by_symbol: Dict[str, np.ndarray] = {}
    for sym in syms:
        vec_cols = post_std_weights_by.get(sym)
        if vec_cols is None:
            # No Stage-A weights found; default to 1.0 for all numeric cols.
            feat_cols = list(trackc_aligned_by[sym].select_dtypes(include=["number", "bool"]).astype(float).columns)
            post_std_feature_weights_by_symbol[sym] = np.ones((len(feat_cols),), dtype=np.float32)
            continue

        w_arr, cols = vec_cols
        w_arr = np.asarray(w_arr, dtype=np.float32)
        feat_cols = [str(c) for c in trackc_aligned_by[sym].select_dtypes(include=["number", "bool"]).astype(float).columns]
        if w_arr.ndim == 1:
            col_to_w = {str(c): float(w_arr[i]) for i, c in enumerate(list(cols))}
            aligned_w = np.ones((len(feat_cols),), dtype=np.float32)
            for j, c in enumerate(feat_cols):
                if c in col_to_w:
                    aligned_w[j] = float(col_to_w[c])
            post_std_feature_weights_by_symbol[sym] = aligned_w
        elif w_arr.ndim == 2:
            col_to_idx = {str(c): int(i) for i, c in enumerate(list(cols))}
            aligned_w = np.ones((w_arr.shape[0], len(feat_cols)), dtype=np.float32)
            for j, c in enumerate(feat_cols):
                i = col_to_idx.get(c)
                if i is not None:
                    aligned_w[:, j] = w_arr[:, i]
            post_std_feature_weights_by_symbol[sym] = aligned_w
        else:
            raise ValueError(f"post-std weight array must be 1D or 2D; got shape={w_arr.shape}")

    # Compute pooled scaler stats on pooled train Track-C.
    from src.stage_b.optuna_optimizer import OptunaConfig, StageBOptunaOptimizer

    opt_cfg = OptunaConfig(
        # Phase2: do not hard-cap total Track-C dimensions.
        max_total_dims=0,
        horizon=int(horizon),
        sequence_model_type="mamba",
        use_three_pillar_dims=True,
        # Allow 3-pillar to pick enough dims to hit the variance target on large families.
        three_pillar_dim_max=int(base_params.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(base_params.get("three_pillar_pca_variance", 0.95) or 0.95),
    )
    optimizer = StageBOptunaOptimizer(config=opt_cfg)

    scaler_stats = optimizer._compute_pooled_scaler_stats(
        [trackc_aligned_by[s].iloc[train_pos_by[s]] for s in syms],
        [labels_by[s][label_col].iloc[train_pos_by[s]] for s in syms],
    )

    # Full standardized Track-C per symbol (used for stateful inference + backtest).
    features_std_full_by: Dict[str, np.ndarray] = {}
    index_by: Dict[str, pd.DatetimeIndex] = {}
    for sym in syms:
        idx = pd.DatetimeIndex(trackc_aligned_by[sym].index)
        index_by[sym] = idx
        feats = _standardize_features_like_build_sequence_data(
            trackc_aligned_by[sym],
            scaler_stats,
            post_standardization_feature_weights=post_std_feature_weights_by_symbol.get(sym),
        )
        _assert_finite_np(name="features_std", symbol=sym, values=feats, index=idx)
        features_std_full_by[sym] = feats

    union_oos_index = pd.DatetimeIndex(sorted({
        ts for sym in syms for ts in index_by[sym] if pd.to_datetime(oos_start) <= ts <= pd.to_datetime(oos_end)
    }))
    if len(union_oos_index) < 10:
        raise ValueError("oos block too small")

    # Build full master arrays per symbol (CPU) and optionally move to GPU once.
    # NOTE: These masters cover the full label-valid timeline (train + any future
    # rows with non-NaN forward_return). We will enforce maturity gating when
    # selecting samples for training/online updates.
    X_by: Dict[str, np.ndarray] = {}
    y_by: Dict[str, np.ndarray] = {}
    ts_by: Dict[str, np.ndarray] = {}
    cpu_train_masters_by: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym in syms:
        train_pos = train_pos_by[sym]
        res = build_master_arrays(
            trackc_aligned_by[sym],
            labels_by[sym][label_col],
            scaler_stats=scaler_stats,
            post_standardization_feature_weights=post_std_feature_weights_by_symbol.get(sym),
        )
        if res is None:
            raise ValueError(f"train master arrays too small for {sym}")
        X, y, ts, _ = res
        X_by[sym] = X
        y_by[sym] = y
        ts_by[sym] = ts
        cpu_train_masters_by[sym] = (X, y, ts)

    gpu_store_train: Optional[MultiSymbolGPUMasterStore] = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_store_train = MultiSymbolGPUMasterStore(
                X_by_symbol=X_by,
                y_by_symbol=y_by,
                ts_by_symbol=ts_by,
                device=torch.device("cuda:0"),
                x_dtype=torch.float32,
            )
    except Exception:
        gpu_store_train = None

    prepared = Phase2PreparedData(
        label_id=str(label_id),
        pipelines_by={k: v for k, v in pipelines_by.items()},
        labels_by={k: v for k, v in labels_by.items()},
        trackc_aligned_by=trackc_aligned_by,
        train_pos_by=train_pos_by,
        scaler_stats=scaler_stats,
        post_std_feature_weights_by_symbol=post_std_feature_weights_by_symbol,
        features_std_full_by=features_std_full_by,
        index_by=index_by,
        union_oos_index=union_oos_index,
        gpu_store_train=gpu_store_train,
        cpu_train_masters_by=cpu_train_masters_by,
        cached_samples_by_seq_len={},
    )

    _PHASE2_PREPARED_CACHE[key] = prepared
    return prepared


def _prepare_phase2_for_cfg(
    *,
    symbols: Sequence[str],
    horizon: int,
    train_start: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    cfg: Mapping[str, Any],
) -> Phase2PreparedData:
    """Prepare Phase-2 artifacts for a specific full trial config."""

    import os

    label_id = _phase2_label_id_from_cfg(cfg)
    label_col = _phase2_label_column(label_id)

    # Phase2 v2 policy: never silently fall back to live panel generation.
    # Dagster/feature-cache is the point of contact; if the cached artifact is
    # missing, we fail fast with instructions.
    allow_live_panel_fallback = os.environ.get("PHASE2_ALLOW_LIVE_PANEL_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    syms = tuple([str(s).upper() for s in symbols])
    sig = _phase2_cfg_signature_for_prepared(cfg)
    key = (
        syms,
        int(horizon),
        str(train_start),
        str(train_end),
        str(oos_start),
        str(oos_end),
        str(sig),
    )
    cached = _PHASE2_PREPARED_CACHE_BY_CFG.get(key)
    if cached is not None:
        logger.info("[phase2.prepare] ✅ CACHE HIT - reusing prepared data (skipping 3-pillar analysis)")
        return cached

    logger.info("[phase2.prepare] 🔧 CACHE MISS - preparing data with pooled 3-pillar analysis for %d symbols", len(syms))
    from src.stage_b.optuna_optimizer import OptunaConfig, StageBOptunaOptimizer, STAGE_A_FAMILIES

    pipelines: Dict[str, StageBPipeline] = {}
    panels: Dict[str, pd.DataFrame] = {}
    labels_by: Dict[str, pd.DataFrame] = {}
    train_pos_by: Dict[str, np.ndarray] = {}

    for sym in syms:
        pipe = _build_stage_b_pipeline(symbol=sym, horizon=int(horizon), start=str(train_start), end=str(oos_end))
        panel = pipe._build_panel(int(horizon))

        # Detect whether we loaded from an on-disk cached artifact. If not, StageBPipeline
        # fell back to live build_panel(), which can trigger expensive/fragile feature
        # generation (e.g. transcript embeddings) and breaks the "Dagster is source of truth" contract.
        try:
            panel_source = getattr(pipe, "_panel_sources", {}).get(int(horizon))
        except Exception:
            panel_source = None
        if (panel_source is None) and (not allow_live_panel_fallback):
            raise FileNotFoundError(
                f"Phase2 v2 requires cached feature panels (Dagster point-of-contact) but no cached panel was used for "
                f"{sym} H{int(horizon)}. Expected one of: "
                f"cache/features/{sym.upper()}_h{int(horizon)}_merged.parquet, "
                f"cache/features/{sym.upper()}_h{int(horizon)}_features.parquet + _index.parquet, "
                f"or cache/features/{sym.upper()}_h{int(horizon)}__features.parquet + __index.parquet. "
                f"Provenance: cache/features/{sym.upper()}_h{int(horizon)}_merged.meta.json. "
                f"Run Dagster prep-families for this symbol/horizon or set PHASE2_ALLOW_LIVE_PANEL_FALLBACK=1 to override."
            )
        labels = pipe._construct_labels(int(horizon), panel.index)
        valid_mask = labels[label_col].notna()
        panel = panel.loc[valid_mask]
        labels = labels.loc[valid_mask]

        panel = _apply_feature_smoothing(
            panel,
            smoothing_type=str(cfg.get("smoothing_type", "none")),
            smoothing_window=int(cfg.get("smoothing_window", 3)),
        )

        idx = pd.DatetimeIndex(panel.index)
        train_pos = _select_index_positions(idx, _to_datetime(train_start), _to_datetime(train_end))

        pipe._current_train_idx = train_pos

        pipelines[sym] = pipe
        panels[sym] = panel
        labels_by[sym] = labels
        train_pos_by[sym] = train_pos

    first_pipe = pipelines[syms[0]]
    column_families = first_pipe._infer_column_families(panels[syms[0]].columns)

    pooled_panel = pd.concat([panels[sym].iloc[train_pos_by[sym]] for sym in syms], axis=0)
    pooled_idx = np.arange(len(pooled_panel), dtype=int)

    opt_cfg = OptunaConfig(
        # Phase2: do not hard-cap total Track-C dimensions.
        max_total_dims=0,
        horizon=int(horizon),
        sequence_model_type="mamba",
        use_three_pillar_dims=True,
        # Allow 3-pillar to pick enough dims to hit the variance target on large families.
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
    )
    try:
        setattr(opt_cfg, "weight_normalization", "none")
    except Exception:
        pass
    optimizer = StageBOptunaOptimizer(config=opt_cfg)

    from src.stage_b.optuna_optimizer import _is_raw_passthrough_family

    # Analyze families + run 3-pillar dim selection (PCA ~95% variance) on POOLED data.
    # Try loading from disk cache first
    cache_key = _three_pillar_cache_key(
        symbols=syms,
        horizon=horizon,
        train_start=train_start,
        train_end=train_end,
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
        stage_a_weights=None,  # Will fetch below
    )
    
    stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(syms[0]).upper(), horizon=int(horizon))
    stage_a_weights_static = _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end))
    
    # Update cache key with actual weights
    cache_key = _three_pillar_cache_key(
        symbols=syms,
        horizon=horizon,
        train_start=train_start,
        train_end=train_end,
        three_pillar_dim_max=int(cfg.get("three_pillar_dim_max", 512) or 512),
        three_pillar_pca_variance=float(cfg.get("three_pillar_pca_variance", 0.95) or 0.95),
        stage_a_weights=stage_a_weights_static,
    )
    
    cached_three_pillar = _load_three_pillar_cache(cache_key)
    
    if cached_three_pillar is not None:
        optimizer.family_sizes = cached_three_pillar.get("family_sizes", {})
        optimizer.family_optimal_dims = cached_three_pillar.get("family_optimal_dims", {})
        logger.info("[phase2.3pillar] ✅ Restored from disk cache (%d families)", len(optimizer.family_optimal_dims))
    else:
        logger.info("[phase2.3pillar] 🔧 DISK CACHE MISS - running pooled 3-pillar on %d symbols (%d rows)...", len(syms), len(pooled_panel))
        import time
        _t0_3pillar = time.time()
        optimizer._analyze_families(pooled_panel, column_families, stage_a_weights=stage_a_weights_static or None)
        _t1_3pillar = time.time()
        logger.info("[phase2.3pillar] ✅ Pooled 3-pillar complete in %.1fs", _t1_3pillar - _t0_3pillar)
        
        # Save to disk cache
        _save_three_pillar_cache(cache_key, {
            "family_sizes": getattr(optimizer, "family_sizes", {}),
            "family_optimal_dims": getattr(optimizer, "family_optimal_dims", {}),
        })

    # Build family params from 3-pillar dims only (NO per-family tuning, NO per-family weights).
    family_sizes: Dict[str, int] = {}
    for col, fam in column_families.items():
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

    optimizer._build_track_a(pooled_panel, column_families, family_params, pooled_idx)

    trackc_by: Dict[str, pd.DataFrame] = {}
    # Store (weight_vec, cols) on each symbol's raw Track-C; we'll align to the final schema after union.
    post_std_weights_by: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for sym in syms:
        pipe = pipelines[sym]
        panel = panels[sym]
        train_pos = train_pos_by[sym]

        pipe._current_train_idx = train_pos
        block_summaries = pipe._compute_block_summaries(panel)

        track_a, _ = optimizer._build_track_a(panel, column_families, family_params, train_pos)
        track_b = optimizer._build_track_b(
            panel,
            column_families,
            block_summaries,
            track_b_family_weights=None,
        )
        track_c = optimizer._build_track_c(track_a, track_b, 1.0, 1.0)
        if track_c is None or track_c.empty:
            raise ValueError(f"Track C is empty for {sym}")
        trackc_by[sym] = track_c

        stage_a_payload = _resolve_phase2_stage_a_payload(cfg, symbol=str(sym).upper(), horizon=int(horizon))
        family_weights = _phase2_static_stage_a_weights(payload=stage_a_payload, asof=_to_datetime(train_end)) or _resolve_phase2_family_weights(
            cfg, symbol=str(sym).upper(), horizon=int(horizon)
        )
        post_w, post_cols = _build_time_varying_post_std_weight_matrix(
            index=pd.DatetimeIndex(track_c.index),
            track_c=track_c,
            track_a=track_a,
            panel=panel,
            block_summaries=block_summaries,
            optimizer=optimizer,
            stage_a_payload=stage_a_payload,
            default_family_weights=family_weights,
        )
        post_std_weights_by[sym] = (np.asarray(post_w, dtype=np.float32), list(post_cols))

    ordered_cols: List[str] = list(trackc_by[syms[0]].columns)
    seen_cols = set(ordered_cols)
    for sym in syms[1:]:
        df = trackc_by.get(sym)
        if df is None:
            continue
        for c in df.columns:
            if c not in seen_cols:
                ordered_cols.append(c)
                seen_cols.add(c)

    trackc_aligned_by: Dict[str, pd.DataFrame] = {sym: _coerce_trackc_numeric_schema(trackc_by[sym], ordered_cols) for sym in syms}

    # Align per-symbol post-std weights (1D vector or 2D matrix) to the aligned Track-C numeric schema.
    post_std_feature_weights_by_symbol: Dict[str, np.ndarray] = {}
    for sym in syms:
        vec_cols = post_std_weights_by.get(sym)
        if vec_cols is None:
            feat_cols = list(trackc_aligned_by[sym].select_dtypes(include=["number", "bool"]).astype(float).columns)
            post_std_feature_weights_by_symbol[sym] = np.ones((len(feat_cols),), dtype=np.float32)
            continue
        w_arr, cols = vec_cols
        w_arr = np.asarray(w_arr, dtype=np.float32)
        feat_cols = [str(c) for c in trackc_aligned_by[sym].select_dtypes(include=["number", "bool"]).astype(float).columns]
        if w_arr.ndim == 1:
            col_to_w = {str(c): float(w_arr[i]) for i, c in enumerate(list(cols))}
            aligned_w = np.ones((len(feat_cols),), dtype=np.float32)
            for j, c in enumerate(feat_cols):
                if c in col_to_w:
                    aligned_w[j] = float(col_to_w[c])
            post_std_feature_weights_by_symbol[sym] = aligned_w
        elif w_arr.ndim == 2:
            col_to_idx = {str(c): int(i) for i, c in enumerate(list(cols))}
            aligned_w = np.ones((w_arr.shape[0], len(feat_cols)), dtype=np.float32)
            for j, c in enumerate(feat_cols):
                i = col_to_idx.get(c)
                if i is not None:
                    aligned_w[:, j] = w_arr[:, i]
            post_std_feature_weights_by_symbol[sym] = aligned_w
        else:
            raise ValueError(f"post-std weight array must be 1D or 2D; got shape={w_arr.shape}")

    scaler_stats = optimizer._compute_pooled_scaler_stats(
        [trackc_aligned_by[s].iloc[train_pos_by[s]] for s in syms],
        [labels_by[s][_phase2_label_column(_phase2_label_id_from_cfg(cfg))].iloc[train_pos_by[s]] for s in syms],
    )

    features_std_full_by: Dict[str, np.ndarray] = {}
    index_by: Dict[str, pd.DatetimeIndex] = {}
    for sym in syms:
        idx = pd.DatetimeIndex(trackc_aligned_by[sym].index)
        index_by[sym] = idx
        feats = _standardize_features_like_build_sequence_data(
            trackc_aligned_by[sym],
            scaler_stats,
            post_standardization_feature_weights=post_std_feature_weights_by_symbol.get(sym),
        )
        _assert_finite_np(name="features_std", symbol=sym, values=feats, index=idx)
        features_std_full_by[sym] = feats

    union_oos_index = pd.DatetimeIndex(sorted({
        ts for sym in syms for ts in index_by[sym] if pd.to_datetime(oos_start) <= ts <= pd.to_datetime(oos_end)
    }))
    if len(union_oos_index) < 10:
        raise ValueError("oos block too small")

    # Build full master arrays per symbol so we can sample matured history during
    # walk-forward online updates. We will enforce maturity gating (H) when
    # selecting samples for training.
    X_by: Dict[str, np.ndarray] = {}
    y_by: Dict[str, np.ndarray] = {}
    ts_by: Dict[str, np.ndarray] = {}
    cpu_train_masters_by: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym in syms:
        res = build_master_arrays(
            trackc_aligned_by[sym],
            labels_by[sym][_phase2_label_column(_phase2_label_id_from_cfg(cfg))],
            scaler_stats=scaler_stats,
            post_standardization_feature_weights=post_std_feature_weights_by_symbol.get(sym),
        )
        if res is None:
            raise ValueError(f"train master arrays too small for {sym}")
        X, y, ts, _ = res
        X_by[sym] = X
        y_by[sym] = y
        ts_by[sym] = ts
        cpu_train_masters_by[sym] = (X, y, ts)

    gpu_store_train: Optional[MultiSymbolGPUMasterStore] = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_store_train = MultiSymbolGPUMasterStore(
                X_by_symbol=X_by,
                y_by_symbol=y_by,
                ts_by_symbol=ts_by,
                device=torch.device("cuda:0"),
                x_dtype=torch.float32,
            )
    except Exception:
        gpu_store_train = None

    prepared = Phase2PreparedData(
        label_id=str(_phase2_label_id_from_cfg(cfg)),
        pipelines_by={k: v for k, v in pipelines.items()},
        labels_by={k: v for k, v in labels_by.items()},
        trackc_aligned_by=trackc_aligned_by,
        train_pos_by=train_pos_by,
        scaler_stats=scaler_stats,
        post_std_feature_weights_by_symbol=post_std_feature_weights_by_symbol,
        features_std_full_by=features_std_full_by,
        index_by=index_by,
        union_oos_index=union_oos_index,
        gpu_store_train=gpu_store_train,
        cpu_train_masters_by=cpu_train_masters_by,
        cached_samples_by_seq_len={},
    )

    _PHASE2_PREPARED_CACHE_BY_CFG[key] = prepared
    return prepared


def _build_window_samples_for_seq_len(
    *,
    symbols: Sequence[str],
    seq_len: int,
    prepared: Phase2PreparedData,
) -> Tuple[List[Tuple[str, int]], np.ndarray]:
    syms = [str(s).upper() for s in symbols]
    samples: List[Tuple[str, int]] = []
    ts_list: List[object] = []

    if prepared.gpu_store_train is None:
        # CPU fallback will rebuild sequences anyway; timestamps come from that path.
        return samples, np.asarray(ts_list, dtype=object)

    store = prepared.gpu_store_train
    for sym in syms:
        n = store.n_samples(sym, int(seq_len))
        ts_aligned = store.get_ts_aligned_next_step(sym, int(seq_len))
        for start in range(n):
            samples.append((sym, int(start)))
            ts_list.append((sym, ts_aligned[int(start)]))

    return samples, np.asarray(ts_list, dtype=object)


# =============================================================================
# CROSS-SECTION MAMBA TRAINING (Level 2)
# =============================================================================

def _build_cross_section_data_from_prepared(
    *,
    symbols: Sequence[str],
    seq_len: int,
    prepared: "Phase2PreparedData",
    adjacency_by_date: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
    min_symbols_per_sample: int = 10,
) -> Optional[CrossSectionSequenceData]:
    """Build CrossSectionSequenceData from Phase2PreparedData.
    
    Converts per-symbol aligned Track-C into [B, T, N, F] format for cross-section training.
    """
    syms = [str(s).upper() for s in symbols]
    label_col = _phase2_label_column(str(getattr(prepared, "label_id", "base")))
    
    # Collect per-symbol data
    X_by_symbol: Dict[str, np.ndarray] = {}
    y_by_symbol: Dict[str, np.ndarray] = {}
    ts_by_symbol: Dict[str, np.ndarray] = {}
    
    for sym in syms:
        if sym not in prepared.trackc_aligned_by:
            continue
        trackc = prepared.trackc_aligned_by[sym]
        labels = prepared.labels_by.get(sym)
        if labels is None or label_col not in labels.columns:
            continue
        
        # Get feature matrix
        X = trackc.values.astype(np.float32)
        y = labels[label_col].values.astype(np.float32)
        ts = trackc.index.values
        
        if len(X) >= seq_len + 1:
            X_by_symbol[sym] = X
            y_by_symbol[sym] = y
            ts_by_symbol[sym] = ts
    
    if len(X_by_symbol) < min_symbols_per_sample:
        logger.warning(f"[CS-Mamba] Only {len(X_by_symbol)} symbols have enough data (need {min_symbols_per_sample})")
        return None
    
    return build_cross_section_data(
        X_by_symbol=X_by_symbol,
        y_by_symbol=y_by_symbol,
        ts_by_symbol=ts_by_symbol,
        seq_len=seq_len,
        universe_symbols=list(X_by_symbol.keys()),
        min_symbols_per_sample=min_symbols_per_sample,
        adjacency_by_date=adjacency_by_date,
        scaler_stats=prepared.scaler_stats,
    )


def _train_cross_section_mamba_phase2(
    *,
    symbols: Sequence[str],
    prepared: "Phase2PreparedData",
    cfg: Mapping[str, Any],
    device: "torch.device",
    adjacency_by_date: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
) -> Optional[Dict[str, Any]]:
    """Train CrossSectionMamba model in Phase2 context.
    
    Returns dict with 'model', 'symbol_order', 'metrics' or None if data insufficient.
    """
    seq_len = int(cfg.get("mamba_seq_len", 128))
    min_symbols = int(cfg.get("cs_min_symbols_per_sample", 10))
    
    cs_data = _build_cross_section_data_from_prepared(
        symbols=symbols,
        seq_len=seq_len,
        prepared=prepared,
        adjacency_by_date=adjacency_by_date,
        min_symbols_per_sample=min_symbols,
    )
    
    if cs_data is None:
        logger.warning("[CS-Mamba] Insufficient data for cross-section training")
        return None
    
    logger.info(f"[CS-Mamba] Built cross-section data: {len(cs_data)} samples, "
                f"N={cs_data.universe_size}, T={cs_data.seq_len}, F={cs_data.feature_dim}")
    
    # Estimate VRAM
    vram_est = compute_vram_requirements(
        B=int(cfg.get("cs_batch_size", 16)),
        T=cs_data.seq_len,
        N=cs_data.universe_size,
        F=cs_data.feature_dim,
        d_model=int(cfg.get("cs_d_model", 128)),
        n_layers=int(cfg.get("cs_n_temporal_layers", 4)),
    )
    logger.info(f"[CS-Mamba] VRAM estimate: {vram_est['total_with_reserve_gb']:.2f} GB")
    
    # Split data
    n_samples = len(cs_data)
    try:
        train_fraction = float(cfg.get("train_fraction", 0.9))
    except Exception:
        train_fraction = 0.9
    train_fraction = float(np.clip(train_fraction, 0.5, 0.95))
    
    split = int(n_samples * train_fraction)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n_samples)
    
    if len(val_idx) < 5:
        logger.warning("[CS-Mamba] Validation set too small")
        return None
    
    # Train
    try:
        result = train_cross_section_mamba(
            cs_data=cs_data,
            train_idx=train_idx,
            val_idx=val_idx,
            cfg=cfg,
            device=device,
            return_model=True,
        )
        result["cs_data"] = cs_data
        return result
    except Exception as e:
        logger.error(f"[CS-Mamba] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _predict_cross_section_phase2(
    *,
    model: "CrossSectionMamba",
    symbols: List[str],
    prepared: "Phase2PreparedData",
    union_oos_index: pd.DatetimeIndex,
    seq_len: int,
    device: "torch.device",
    adjacency_by_date: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Run cross-section inference to get per-symbol predictions.
    
    Returns:
        preds_by_symbol: Dict[symbol] -> pd.Series of mu predictions
        sigma_by_symbol: Dict[symbol] -> pd.Series of sigma predictions
    """
    import torch
    
    preds_by_symbol: Dict[str, pd.Series] = {}
    sigma_by_symbol: Dict[str, pd.Series] = {}
    
    # Build inference buffer
    F = prepared.trackc_aligned_by[symbols[0]].shape[1] if symbols else 0
    buffer = CrossSectionInferenceBuffer(
        symbols=symbols,
        seq_len=seq_len,
        feature_dim=F,
        device=device,
    )
    
    # Build features dict for warm-up
    features_by = {s: prepared.features_std_full_by.get(s) for s in symbols if s in prepared.features_std_full_by}
    
    # Initialize results
    for sym in symbols:
        preds_by_symbol[sym] = pd.Series(index=union_oos_index, dtype=float)
        sigma_by_symbol[sym] = pd.Series(index=union_oos_index, dtype=float)
    
    # Get all available dates across symbols
    all_dates: set = set()
    for sym, feats in features_by.items():
        if feats is not None:
            idx = prepared.index_by.get(sym)
            if idx is not None:
                all_dates.update(pd.DatetimeIndex(idx).normalize())
    
    sorted_dates = sorted(all_dates)
    oos_dates = set(union_oos_index.normalize())
    
    # Roll through time
    model.eval()
    for date in sorted_dates:
        # Update buffer for each symbol
        for sym in symbols:
            idx = prepared.index_by.get(sym)
            feats = features_by.get(sym)
            if idx is None or feats is None:
                continue
            
            # Find row for this date
            idx_norm = pd.DatetimeIndex(idx).normalize()
            match = idx_norm == date
            if match.any():
                row_idx = np.where(match)[0][0]
                feat_vec = torch.from_numpy(feats[row_idx]).float().to(device)
                buffer.update(sym, feat_vec)
        
        # Predict if this is an OOS date
        if date in oos_dates:
            adj = None
            if adjacency_by_date is not None and date in adjacency_by_date:
                adj = torch.from_numpy(adjacency_by_date[date]).float().to(device)
            
            predictions = buffer.predict(model, adjacency=adj)
            
            for sym, (mu, sigma) in predictions.items():
                if sym in preds_by_symbol:
                    # Find matching OOS timestamp
                    oos_match = union_oos_index.normalize() == date
                    if oos_match.any():
                        oos_ts = union_oos_index[oos_match][0]
                        preds_by_symbol[sym].loc[oos_ts] = mu
                        sigma_by_symbol[sym].loc[oos_ts] = sigma
    
    return preds_by_symbol, sigma_by_symbol


# ---------------------------------------------------------------------------
# Multi-Horizon Training Helpers (Workstream 6)
# ---------------------------------------------------------------------------

def _build_multi_horizon_labels_for_symbol(
    pipe: "StageBPipeline",
    panel: pd.DataFrame,
    horizons: List[int],
    label_type: str = "base",
) -> Dict[int, pd.Series]:
    """Build forward-return labels for multiple horizons using existing pipeline.
    
    Args:
        pipe: StageBPipeline for the symbol
        panel: Track-C panel with DatetimeIndex
        horizons: List of horizon days (e.g. [5, 21, 63, 126])
        label_type: 'base' for forward_return, 'voladj' for forward_return_voladj
        
    Returns:
        Dict mapping horizon -> pd.Series of forward returns aligned to panel index
    """
    label_col = _phase2_label_column(label_type)
    labels_by_horizon = {}
    
    for h in horizons:
        try:
            labels_df = pipe._construct_labels(int(h), panel.index)
            labels_by_horizon[h] = labels_df[label_col].reindex(panel.index)
        except Exception as e:
            logger.warning(f"[MH] Failed to build labels for horizon {h}: {e}")
            labels_by_horizon[h] = pd.Series(np.nan, index=panel.index)
    
    return labels_by_horizon


def _build_pooled_multi_horizon_sequence_data(
    *,
    symbols: Sequence[str],
    horizons: List[int],
    seq_len: int,
    prepared: "Phase2PreparedData",
    pipelines_by: Dict[str, "StageBPipeline"],
    panels_by: Dict[str, pd.DataFrame],
    train_pos_by: Dict[str, np.ndarray],
    label_type: str = "base",
) -> MultiHorizonSequenceData:
    """Build pooled MultiHorizonSequenceData from Phase-2 prepared artifacts.
    
    Combines all symbols into a single dataset with per-horizon targets.
    
    Args:
        symbols: List of symbols
        horizons: List of horizon days [5, 21, 63, 126]
        seq_len: Sequence length
        prepared: Phase2PreparedData with scaler_stats, trackc_aligned_by, etc.
        pipelines_by: Dict[symbol] -> StageBPipeline for label construction
        panels_by: Dict[symbol] -> Track-C panel
        train_pos_by: Dict[symbol] -> training position indices
        label_type: 'base' or 'voladj'
        
    Returns:
        MultiHorizonSequenceData with pooled sequences and per-horizon targets
    """
    syms = [str(s).upper() for s in symbols]
    
    # Collect per-symbol multi-horizon data
    all_sequences = []
    all_targets = {h: [] for h in horizons}
    all_timestamps = []
    
    for sym in syms:
        # Build multi-horizon labels
        mh_labels = _build_multi_horizon_labels_for_symbol(
            pipe=pipelines_by[sym],
            panel=panels_by[sym],
            horizons=horizons,
            label_type=label_type,
        )
        
        # Get Track-C and scaler stats
        track_c = prepared.trackc_aligned_by[sym]
        train_pos = train_pos_by[sym]
        post_w = prepared.post_std_feature_weights_by_symbol.get(sym)
        
        # Standardize features
        features_std = _standardize_features_like_build_sequence_data(
            track_c,
            prepared.scaler_stats,
            post_standardization_feature_weights=post_w,
        )
        
        # Build sequences from training positions only
        train_features = features_std[train_pos]
        train_index = track_c.index[train_pos]
        
        # Extract per-horizon targets at training positions
        train_targets_by_h = {}
        for h in horizons:
            h_labels = mh_labels[h].reindex(train_index)
            train_targets_by_h[h] = h_labels.values.astype(np.float32)
        
        # Build sliding-window sequences
        T_train = len(train_features)
        n_samples = T_train - seq_len + 1
        if n_samples <= 0:
            logger.warning(f"[MH] Symbol {sym} has insufficient training data for seq_len={seq_len}")
            continue
            
        for i in range(n_samples):
            # Sequence: [i, i+seq_len)
            seq = train_features[i:i + seq_len]
            all_sequences.append(seq)
            
            # Target at end of sequence (t = i + seq_len - 1)
            t = i + seq_len - 1
            for h in horizons:
                target_val = train_targets_by_h[h][t] if t < len(train_targets_by_h[h]) else np.nan
                all_targets[h].append(target_val)
            
            # Timestamp at end of sequence
            ts_val = train_index[t] if t < len(train_index) else train_index[-1]
            all_timestamps.append((sym, ts_val))
    
    if not all_sequences:
        raise ValueError("[MH] No valid sequences built across symbols")
    
    # Stack into arrays
    sequences = np.stack(all_sequences, axis=0).astype(np.float32)
    targets = {h: np.array(all_targets[h], dtype=np.float32) for h in horizons}
    timestamps = np.array(all_timestamps, dtype=object)
    
    logger.info(
        f"[MH] Built pooled MultiHorizonSequenceData: "
        f"{sequences.shape[0]} samples, {len(horizons)} horizons, {len(syms)} symbols"
    )
    
    return MultiHorizonSequenceData(
        sequences=sequences,
        targets=targets,
        timestamps=timestamps,
        symbol="POOLED",
        horizons=horizons,
    )


def _build_pooled_sequence_cpu(
    *,
    symbols: Sequence[str],
    seq_len: int,
    prepared: Phase2PreparedData,
) -> SequenceData:
    # Legacy CPU path: build 3D windows per symbol and concatenate.
    syms = [str(s).upper() for s in symbols]
    seq_blocks: List[np.ndarray] = []
    tgt_blocks: List[np.ndarray] = []
    ts_blocks: List[np.ndarray] = []

    label_col = _phase2_label_column(str(getattr(prepared, "label_id", "base")))
    for sym in syms:
        X, y, ts = prepared.cpu_train_masters_by[sym] if prepared.cpu_train_masters_by is not None else (None, None, None)
        if X is None or y is None or ts is None:
            raise ValueError("missing cpu train masters")
        # Reconstruct DataFrame/Series semantics minimally isn't worth it; use existing build_sequence_data.
        # Fall back to the original implementation using pandas.
        seq_res = build_sequence_data(
            prepared.trackc_aligned_by[sym],
            prepared.labels_by[sym][label_col],
            int(seq_len),
            scaler_stats=prepared.scaler_stats,
            post_standardization_feature_weights=prepared.post_std_feature_weights_by_symbol.get(sym),
        )
        if seq_res is None:
            raise ValueError(f"train sequence data too small for {sym}")
        seq_data, _ = seq_res
        seq_blocks.append(seq_data.sequences)
        tgt_blocks.append(seq_data.targets)
        ts_blocks.append(np.asarray([(sym, t) for t in seq_data.timestamps], dtype=object))

    return SequenceData(
        sequences=np.concatenate(seq_blocks, axis=0),
        targets=np.concatenate(tgt_blocks, axis=0),
        timestamps=np.concatenate(ts_blocks, axis=0),
    )


# ---------------------------------------------------------------------------
# Optuna driver (Phase-2 search)
# ---------------------------------------------------------------------------

def run_phase2_stateful_optuna(
    *,
    best_trial_json: Optional[Path],
    symbol: str,
    horizon: int,
    train_start: str = DEFAULT_PHASE2_TRAIN_START,
    train_end: str = DEFAULT_PHASE2_TRAIN_END,
    oos_start: str = DEFAULT_PHASE2_OOS_START,
    oos_end: str = DEFAULT_PHASE2_OOS_END,
    holdout_start: Optional[str] = DEFAULT_PHASE2_HOLDOUT_START,
    holdout_end: Optional[str] = DEFAULT_PHASE2_HOLDOUT_END,
    n_trials: int,
    refinement: Phase2RefinementSpec = Phase2RefinementSpec(),
    search_spec: Phase2SearchSpec = Phase2SearchSpec(),
    objective_spec: Phase2ObjectiveSpec = Phase2ObjectiveSpec(),
    symbols: Optional[Sequence[str]] = None,
    portfolio_weights: Optional[Mapping[str, float]] = None,
    aggregation_rule: str = "equal_weight",
    study_db_path: Optional[Path] = None,
    study_name: str = "stage_b_stateful_phase2",
    export_winner_path: Optional[Path] = None,
    no_prune: bool = False,
    deterministic_all: bool = False,
    prune_update_sessions: int = 21,
    prune_warmup_folds: int = 12,
    safety_prune_after_folds: int = 12,
    safety_prune_sharpe_floor: float = 0.0,
    safety_prune_maxdd_ceiling: float = 0.12,
    tpe_top_fraction: float = 0.15,
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Optuna over Phase-2 using stateful evaluation."""

    import optuna

    mode = str(getattr(search_spec, "mode", "refinement") or "refinement").lower().strip()
    if mode not in {"refinement", "full"}:
        raise ValueError(f"unknown Phase-2 search mode: {mode}")

    # Full-mode: tune all parameter groups jointly (model + thresholds + portfolio + overlays)
    # on every trial. The previous trial-number-based 4-phase gating was removed
    # because it prevents interactions from being discovered early.

    # Phase-2 refinement is intentionally small to avoid OOS overfit.
    if mode == "refinement" and int(n_trials) > 40:
        logger.warning("Phase-2 refinement requested n_trials=%s (>40). This increases risk of OOS overfit.", int(n_trials))

    # Fixed Tune-OOS period sanity check.
    t_train_end = pd.to_datetime(train_end)
    t_oos_start = pd.to_datetime(oos_start)
    t_oos_end = pd.to_datetime(oos_end)
    if t_oos_start <= t_train_end:
        raise ValueError(f"Phase-2 requires non-overlapping blocks: oos_start({oos_start}) must be after train_end({train_end})")
    if t_oos_end <= t_oos_start:
        raise ValueError(f"Phase-2 requires a valid OOS range: oos_end({oos_end}) must be after oos_start({oos_start})")

    # Optional holdout sanity checks.
    if (holdout_start is None) != (holdout_end is None):
        raise ValueError("holdout_start and holdout_end must be provided together")
    if holdout_start is not None and holdout_end is not None:
        t_holdout_start = pd.to_datetime(holdout_start)
        t_holdout_end = pd.to_datetime(holdout_end)
        if t_holdout_start <= t_oos_end:
            raise ValueError(
                f"Holdout should start after Tune-OOS end: holdout_start({holdout_start}) <= oos_end({oos_end})"
            )
        if t_holdout_end <= t_holdout_start:
            raise ValueError(f"Invalid holdout range: holdout_end({holdout_end}) must be after holdout_start({holdout_start})")

    # One label variant per Optuna study (label is NOT sampled).
    rt0 = dict(runtime_overrides or {})
    label_id = _phase2_label_id_from_cfg(rt0)
    study_name = _phase2_study_name_with_label(study_name, label_id=label_id)

    if study_db_path is None:
        study_db_path = Path("artifacts/optuna_studies") / f"{study_name}.db"
    study_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Default to the same fixed GLOBAL13 universe used by Stage-B pooled training.
    # Users can override explicitly by passing `symbols=[...]`.
    eval_symbols = list(symbols) if symbols is not None else list(GLOBAL_OPTUNA_SYMBOLS)

    # Default winner export path (deterministic) if not provided.
    if export_winner_path is None:
        export_winner_path = Path("artifacts") / "optuna" / f"GLOBAL13_h{int(horizon)}_phase2_stateful_winner.json"
    export_winner_path.parent.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{study_db_path}"

    # Optional pruning.
    # We report one intermediate portfolio score per OOS fold at steps 1..N.
    # Hyperband needs max_resource ~= max step.
    def _estimate_oos_folds() -> int:
        step = max(1, int(prune_update_sessions))
        start_ts = pd.Timestamp(oos_start)
        end_ts = pd.Timestamp(oos_end)
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_localize(None)
        if getattr(end_ts, "tzinfo", None) is not None:
            end_ts = end_ts.tz_localize(None)
        start_ts = start_ts.normalize()
        end_ts = end_ts.normalize()
        try:
            import exchange_calendars as xcals  # type: ignore

            cal = xcals.get_calendar("XNYS")
            s = cal.date_to_session(start_ts, direction="next")
            e = cal.date_to_session(end_ts, direction="previous")
            n_sessions = int(len(cal.sessions_in_range(s, e)))
        except Exception:
            n_sessions = int(len(pd.bdate_range(start_ts, end_ts)))

        if n_sessions <= 0:
            return 1
        return int((n_sessions + step - 1) // step)

    max_resource = int(_estimate_oos_folds())
    pruner = optuna.pruners.NopPruner() if bool(no_prune) else optuna.pruners.HyperbandPruner(
        min_resource=max(1, int(prune_warmup_folds)),
        max_resource=max(1, max_resource),
        reduction_factor=3,
    )

    # Sampler: configure TPE for expensive GPU trials.
    # - Reduce random-warmup waste.
    # - Use multivariate + group to learn interactions.
    # - Use a stricter good-set fraction because pruning biases completed trials.
    top_frac = float(np.clip(float(tpe_top_fraction), 0.05, 0.5))
    seed = int(getattr(search_spec, "seed", 1337) or 1337)
    n_startup = 5
    try:
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(n_startup),
            gamma=lambda n: max(1, int(np.ceil(top_frac * float(n)))),
            multivariate=True,
            group=True,
        )
    except TypeError:
        # Compatibility with older Optuna versions (no group/multivariate args).
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(n_startup),
            gamma=lambda n: max(1, int(np.ceil(top_frac * float(n)))),
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    try:
        study.set_user_attr("phase2_label_id", str(label_id))
    except Exception:
        pass

    # IMPORTANT: Optuna categorical distributions cannot change once a study has history.
    # If this study already has trials, we must keep the original categorical space for
    # `mamba_seq_len`, otherwise Optuna raises:
    #   ValueError: CategoricalDistribution does not support dynamic value space.
    existing_seq_len_choices: Optional[Tuple[int, ...]] = None
    for t in study.trials:
        dist = getattr(t, "distributions", {}).get("mamba_seq_len") if hasattr(t, "distributions") else None
        if dist is not None and hasattr(dist, "choices"):
            try:
                existing_seq_len_choices = tuple(int(x) for x in dist.choices)
                break
            except Exception:
                continue

    if existing_seq_len_choices is not None:
        # IMPORTANT: once a study has history, we must keep categorical choices unchanged.
        seq_len_choices_for_study: Tuple[int, ...] = tuple(existing_seq_len_choices)
    else:
        # New study: expose the full stable range directly via `mamba_seq_len`.
        # User request: cover roughly horizon..315 with +32 steps (and include 315).
        # We keep the same choices for both full and refinement modes to avoid drift.
        base = int(horizon)
        choices = [int(base + 32 * k) for k in range(0, 32) if int(base + 32 * k) <= 315]
        if 315 not in set(choices):
            choices.append(315)
        seq_len_choices_for_study = tuple(sorted(set(int(x) for x in choices)))

    # Pin categorical spaces for the uncertainty configuration.
    # Existing studies may already have sampled different head/loss choices; we must
    # preserve those spaces to avoid Optuna dynamic value-space errors.
    existing_loss_choices: Optional[Tuple[str, ...]] = None
    existing_head_choices: Optional[Tuple[str, ...]] = None
    for t in study.trials:
        dists = getattr(t, "distributions", {}) if hasattr(t, "distributions") else {}
        dist_loss = dists.get("mamba_loss_fn")
        if existing_loss_choices is None and dist_loss is not None and hasattr(dist_loss, "choices"):
            try:
                existing_loss_choices = tuple(str(x) for x in dist_loss.choices)
            except Exception:
                pass
        dist_head = dists.get("mamba_head_type")
        if existing_head_choices is None and dist_head is not None and hasattr(dist_head, "choices"):
            try:
                existing_head_choices = tuple(str(x) for x in dist_head.choices)
            except Exception:
                pass
        if existing_loss_choices is not None and existing_head_choices is not None:
            break

    loss_choices_for_study: Tuple[str, ...] = (
        tuple(existing_loss_choices) if existing_loss_choices is not None else ("gaussian_nll",)
    )
    head_choices_for_study: Tuple[str, ...] = (
        tuple(existing_head_choices) if existing_head_choices is not None else ("gaussian",)
    )

    # HF-grade due diligence: the Phase2 v2 engine forces uncertainty training.
    # If an existing study has broader categorical spaces here, Optuna will still
    # sample those values, but evaluation will override them.
    if existing_loss_choices is not None and any(str(x) != "gaussian_nll" for x in existing_loss_choices):
        logger.warning(
            "Phase2 study '%s' has non-gaussian loss choices (%s). v2 evaluation will override to gaussian_nll; "
            "consider starting a fresh v2-only study name to avoid search noise.",
            study_name,
            ",".join(list(existing_loss_choices)[:10]),
        )
    if existing_head_choices is not None and any(str(x) != "gaussian" for x in existing_head_choices):
        logger.warning(
            "Phase2 study '%s' has non-gaussian head choices (%s). v2 evaluation will override to gaussian; "
            "consider starting a fresh v2-only study name to avoid search noise.",
            study_name,
            ",".join(list(existing_head_choices)[:10]),
        )

    # For full mode, precompute a stable family-size map (does not depend on trial params).
    _full_family_sizes: Optional[Dict[str, int]] = None
    _full_three_pillar_dims: Optional[Dict[str, Dict[str, Any]]] = None
    if mode == "full":
        try:
            probe_sym = str(eval_symbols[0]).upper()
            pipe = _build_stage_b_pipeline(symbol=probe_sym, horizon=int(horizon), start=str(train_start), end=str(oos_end))
            panel = pipe._build_panel(int(horizon))
            column_families = pipe._infer_column_families(panel.columns)
            fam_sizes: Dict[str, int] = {}
            for _col, fam in column_families.items():
                fam_sizes[str(fam)] = int(fam_sizes.get(str(fam), 0) + 1)
            _full_family_sizes = fam_sizes

            # Stage-B parity: if 3-pillar dims are enabled, precompute them ONCE
            # (same behavior as Stage-B runner: fixed dim_type/dims/AE hyperparams).
            # OPTIMIZATION: Skip per-symbol probe 3-pillar analysis here - it will be computed
            # later on pooled multi-symbol data in _build_trackc_multi_symbol for better accuracy.
            try:
                from src.stage_b.optuna_optimizer import OptunaConfig, StageBOptunaOptimizer

                _probe_opt_cfg = OptunaConfig(
                    # Phase2: no hard Track-C total-dim cap; rely on 3-pillar variance target.
                    max_total_dims=0,
                    horizon=int(horizon),
                    sequence_model_type="mamba",
                    use_three_pillar_dims=True,
                    three_pillar_dim_max=512,
                    three_pillar_pca_variance=0.95,
                )
                _probe_optimizer = StageBOptunaOptimizer(config=_probe_opt_cfg)
                # NOTE: `_cfg_for_stage_a` is constructed later during trial assembly.
                # For the full-mode probe, we only need Stage-A weights resolution inputs,
                # which live in `runtime_overrides` (e.g., phase2_family_weights_path).
                _probe_stage_a_payload = _resolve_phase2_stage_a_payload(
                    dict(runtime_overrides or {}),
                    symbol=str(probe_sym).upper(),
                    horizon=int(horizon),
                )
                _probe_stage_a_weights = _phase2_static_stage_a_weights(payload=_probe_stage_a_payload, asof=pd.to_datetime(train_end))
                # SKIP: _probe_optimizer._analyze_families(panel, column_families, stage_a_weights=_probe_stage_a_weights or None)
                # 3-pillar analysis will be performed on pooled data in _build_trackc_multi_symbol
                _full_three_pillar_dims = {
                    str(fam): {"method": str(cfg_.method), "k_final": int(cfg_.k_final)}
                    for fam, cfg_ in dict(getattr(_probe_optimizer, "family_optimal_dims", {}) or {}).items()
                }

                # Persist the (fixed) 3-pillar dims used for full-mode Track-C construction.
                try:
                    if _full_three_pillar_dims:
                        study.set_user_attr("phase2_three_pillar_dims", dict(_full_three_pillar_dims))
                        study.set_user_attr(
                            "phase2_three_pillar_total_dim",
                            int(sum(int(v.get("k_final", 0) or 0) for v in _full_three_pillar_dims.values())),
                        )
                except Exception:
                    pass
            except Exception:
                _full_three_pillar_dims = None
        except Exception:
            _full_family_sizes = None

    def _audit_phase2_trial_params(
        *,
        mode_local: str,
        trial: "optuna.Trial",
        cfg_local: Mapping[str, Any],
        helper_used: Optional[Sequence[str]] = None,
    ) -> None:
        """Fail-fast audit: every sampled Optuna param must be consumed.

        This guards against silent regressions where we sample params but they never
        affect Track-C construction, training, or backtest.
        """

        sampled = set((getattr(trial, "params", {}) or {}).keys())
        helper_used = set(str(x) for x in (helper_used or ()))

        # Known alias groups: allowed to coexist ONLY if values agree.
        alias_groups = [
            ("bull_mult", "bull_long_mult"),
            ("bear_mult", "bear_long_mult"),
            ("crisis_mult", "crisis_long_mult"),
            ("mamba_max_epochs", "max_epochs"),
        ]
        alias_conflicts: List[str] = []
        for a, b in alias_groups:
            if a in cfg_local and b in cfg_local:
                try:
                    va = cfg_local.get(a)
                    vb = cfg_local.get(b)
                    if va is not None and vb is not None and float(va) != float(vb):
                        alias_conflicts.append(f"{a}!={b}")
                except Exception:
                    # If we can't safely compare, don't hard fail.
                    pass

        # Define the set/patterns of params that are expected to be consumed.
        # Note: family params are patterns to cover STAGE_A_FAMILIES.
        direct_used = {
            # Pipeline/prep
            "max_total_dims",
            "track_a_weight",
            "track_b_weight",
            "smoothing_type",
            "smoothing_window",
            "train_fraction",
            # Phase-2 engine
            "phase2_engine",
            "phase2_update_sessions",
            "phase2_replay_days",
            "phase2_update_epochs",
            "phase2_cov_ewma_lambda",
            "phase2_shrinkage_alpha",
            "phase2_target_vol",
            "phase2_max_gross",
            "phase2_max_name",
            "phase2_max_net",
            "phase2_k_spread",
            "phase2_k_impact",
            "phase2_z_clip",
            # Strategy thresholds
            "threshold",
            "bull_long_mult",
            "bull_short_mult",
            "bear_long_mult",
            "bear_short_mult",
            "crisis_long_mult",
            "crisis_short_mult",
            "conf_threshold",
            "vol_scaler",
            # v2 regime-aware thresholding
            "regime_threshold_base",
            "regime_threshold_bull_mult",
            "regime_threshold_bear_mult",
            "regime_threshold_crisis_mult",
            # Optional overlay enable flags (conditional search-space blocks)
            "phase2_enable_turnover_overlay",
            "phase2_enable_turnover_kill_switch",
            "phase2_enable_weight_smoothing",
            "phase2_enable_group_caps",
            "phase2_enable_beta_neutral",
            "phase2_enable_liquidity_constraints",
            # Optional overlay knobs (only sampled if enabled)
            "phase2_turnover_cap",
            "phase2_kill_on_turnover_gt",
            "phase2_flat_cooldown_sessions",
            "phase2_weight_smoothing_alpha",
            "phase2_group_max_gross",
            "phase2_group_max_net",
            "phase2_beta_neutral",
            "phase2_beta_max_abs_exposure",
            "phase2_beta_lookback_days",
            "phase2_adv_window",
            "phase2_max_adv_frac_name",
            "phase2_max_turnover_adv_frac",
            "phase2_borrow_fee_bps_annual",
            # Execution realism (Workstream: Backtest Realism)
            "phase2_exec_price_mode",
            "phase2_exec_participation_rate",
            "phase2_exec_enable_partial_fills",
            "phase2_exec_asset_aware_slippage",
            "phase2_exec_impact_exponent",
            "phase2_exec_impact_scale",
            "phase2_exec_default_spread_bps",
            "phase2_exec_per_name_borrow",
            "phase2_exec_htb_threshold_bps",
            "phase2_exec_htb_max_short_weight",
            "phase2_exec_halted_to_flat",
            "phase2_exec_reduce_short_on_ex_div",
            "phase2_exec_ex_div_reduction_mult",
            # Daily trace instrumentation
            "phase2_trace_enabled",
            "phase2_trace_persist",
            "phase2_trace_persist_path",
            # Structured event bus
            "phase2_events_enabled",
            "phase2_events_persist",
            # Diagnostics artifacts
            "phase2_diagnostics_enabled",
            # Per-symbol calibration (GAP #8)
            "phase2_per_symbol_calib",
            # Z-Explainer for interpretability
            "phase2_z_explainer",
            "phase2_z_explainer_window",
            "phase2_z_explainer_max_window",
            "phase2_z_explainer_refit_interval",
            "phase2_z_explainer_regularization",
            "phase2_z_explainer_min_samples",
            "phase2_z_explainer_top_symbols",
            "phase2_z_explainer_top_features",
            # Z-Explainer validation companion (Step 7)
            "phase2_z_explainer_validation",
            "phase2_z_explainer_validation_refit",
            "phase2_z_explainer_validation_min_samples",
            # Risk Latch (Unified state machine)
            "phase2_risk_latch_enabled",
            "phase2_risk_latch_max_dd_kill",
            "phase2_risk_latch_max_vol_kill",
            "phase2_risk_latch_throttle_min_scale",
            "phase2_risk_latch_dd_throttle_start",
            "phase2_risk_latch_vol_throttle_mult",
            "phase2_risk_latch_panic_vix",
            "phase2_risk_latch_min_calib",
            "phase2_risk_latch_latch_sessions",
            "phase2_risk_latch_corr_hhi_kill",
            "phase2_risk_latch_single_day_crash",
            # Fast crash triggers (1-day loss kill, gap shock)
            "phase2_kill_1day_loss_pct",
            "phase2_emergency_cooldown_sessions",
            "phase2_gap_shock_mult",
            "phase2_gap_shock_min_pct",
            "phase2_gap_shock_n_symbols",
            "phase2_gap_shock_portfolio_pct",
            "phase2_gap_shock_mode",
            "phase2_gap_shock_throttle_scale",
            # Stress Overlay (scenario stress tests)
            "phase2_stress_enabled",
            "phase2_stress_k_sigma",
            "phase2_stress_loss_throttle",
            "phase2_stress_loss_flatten",
            "phase2_stress_shock_scale",
            "phase2_stress_corr_hhi_throttle",
            "phase2_stress_corr_hhi_flatten",
            "phase2_stress_corr_scale",
            "phase2_stress_sector_max",
            "phase2_stress_sector_scale",
            "phase2_stress_min_scale",
            # Intraday Monitor
            "phase2_intraday_monitor_enabled",
            "phase2_intraday_snapshot_path",
            "phase2_intraday_snapshot_max_age",
            # Mamba training
            "mamba_seq_len",
            "mamba_d_model",
            "mamba_n_layers",
            "mamba_ssm_dim",
            "mamba_expand_factor",
            "mamba_activation",
            "mamba_norm_type",
            "mamba_norm_strategy",
            "mamba_dropout",
            "mamba_resid_dropout",
            "mamba_ssm_dropout",
            "mamba_gate_dropout",
            "mamba_optimizer",
            "mamba_learning_rate",
            "mamba_weight_decay",
            "mamba_grad_clip",
            "mamba_lr_scheduler",
            "mamba_warmup_steps",
            "mamba_max_epochs",
            "mamba_batch_size",
            "mamba_loss_fn",
            "mamba_head_type",
            "mamba_head_hidden_dim",
            "mamba_head_num_layers",
            "mamba_head_dropout",
            "phase2_regime_feature_index",
            "phase2_regime_loss_alpha",
            "phase2_sigma_floor_base",
            "phase2_sigma_floor_stress",
            "phase2_sigma_floor_strength",
            "phase2_anchor_l2",
            "phase2_sigma_cs_strength",
            # Workstream-7 Robust Portfolio Optimizer
            "phase2_use_robust_optimizer",
            "phase2_robust_lambda_var",
            "phase2_robust_lambda_turnover",
            "phase2_robust_lambda_tail",
            "phase2_robust_use_predicted_sigma",
            "phase2_robust_predicted_sigma_blend",
            "phase2_robust_covariance_method",
            "phase2_robust_cvar_constraint",
            "phase2_robust_cvar_alpha",
            "phase2_robust_cvar_limit",
            "phase2_robust_uncertainty_caps",
            "phase2_robust_mu_sigma_cap",
            "phase2_robust_reliability_min",
            "phase2_robust_n_factors",
            "phase2_robust_use_graph_shrinkage",
        }

        def _matches_family_pattern(k: str) -> bool:
            if k.startswith("weight_b_"):
                return True
            if k.startswith("weight_"):
                return True
            # Stage-A family dim keys
            if k.endswith("_dim") or k.endswith("_dim_type"):
                return True
            if k.endswith("_pca_dim") or k.endswith("_ae_dim"):
                return True
            if k.endswith("_ae_layers") or k.endswith("_ae_activation") or k.endswith("_ae_dropout") or k.endswith("_ae_lr"):
                return True
            return False

        # In refinement mode, only a subset is expected.
        if mode_local == "refinement":
            allowed = {
                "mamba_seq_len",
                "mamba_learning_rate",
                "mamba_dropout",
                "mamba_head_dropout",
                "mamba_grad_clip",
            }
            unexpected = sorted([k for k in sampled if k not in allowed])
            unused = sorted([k for k in sampled if k not in allowed and k not in helper_used])
        else:
            unexpected = sorted([k for k in sampled if (k not in direct_used and not _matches_family_pattern(k))])
            unused = sorted([k for k in unexpected if k not in helper_used])

        try:
            trial.set_user_attr("phase2_param_audit_mode", str(mode_local))
            trial.set_user_attr("phase2_param_audit_sampled", int(len(sampled)))
            trial.set_user_attr("phase2_param_audit_unexpected", list(unexpected[:200]))
            trial.set_user_attr("phase2_param_audit_alias_conflicts", list(alias_conflicts))
        except Exception:
            pass

        if alias_conflicts:
            raise ValueError(f"Phase-2 param alias conflicts: {alias_conflicts}")
        if unused:
            raise ValueError(f"Phase-2 sampled params not consumed: {unused[:50]}")

    def objective(trial: "optuna.Trial") -> float:
        import optuna

        # Metadata: full-mode uses joint tuning (all groups every trial).
        if mode == "full":
            try:
                trial.set_user_attr("phase2_tuning_enabled", True)
                trial.set_user_attr("phase2_tuning_strategy", "joint")
                trial.set_user_attr("phase2_tuning_phases", 1)
                trial.set_user_attr("phase2_tuning_phase", 1)
                trial.set_user_attr("phase2_tuning_phase_name", "joint")
            except Exception:
                pass

        try:
            if mode == "refinement":
                if best_trial_json is None:
                    raise ValueError("best_trial_json is required for refinement mode")

                helper_used: List[str] = []
                effective_seq_len = int(trial.suggest_categorical("mamba_seq_len", list(seq_len_choices_for_study)))

                overrides = {
                    "mamba_seq_len": int(effective_seq_len),
                    "mamba_learning_rate": trial.suggest_float(
                        "mamba_learning_rate", refinement.lr_log_low, refinement.lr_log_high, log=True
                    ),
                    "mamba_dropout": trial.suggest_float("mamba_dropout", refinement.dropout_low, refinement.dropout_high),
                    "mamba_head_dropout": trial.suggest_float(
                        "mamba_head_dropout", refinement.head_dropout_low, refinement.head_dropout_high
                    ),
                    "mamba_grad_clip": trial.suggest_categorical("mamba_grad_clip", list(refinement.grad_clip_choices)),
                }

                # Audit that we didn't accidentally introduce unused sampled params in refinement mode.
                _audit_phase2_trial_params(mode_local="refinement", trial=trial, cfg_local=overrides, helper_used=helper_used)

                res = evaluate_phase2_stateful_once(
                    best_trial_json=best_trial_json,
                    symbols=eval_symbols,
                    horizon=horizon,
                    train_start=train_start,
                    train_end=train_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    overrides=overrides,
                    portfolio_weights=portfolio_weights,
                    aggregation_rule=aggregation_rule,
                    objective_spec=objective_spec,
                    trial=trial,
                    prune_oos_days=None,
                    prune_update_sessions=int(prune_update_sessions),
                    prune_warmup_folds=int(prune_warmup_folds),
                    safety_prune_after_folds=0 if bool(no_prune) else int(safety_prune_after_folds),
                    safety_prune_sharpe_floor=float(safety_prune_sharpe_floor),
                    safety_prune_maxdd_ceiling=float(safety_prune_maxdd_ceiling),
                    deterministic_all=bool(deterministic_all),
                )
                return float(res.objective)

            # mode == "full"
            from src.stage_b.optuna_optimizer import (
                HF_BLOCK_FAMILIES,
                TRACK_B_SUMMARY_BLOCKS,
                OptunaConfig,
                StageBOptunaOptimizer,
                STAGE_A_FAMILIES,
            )

            opt_cfg = OptunaConfig(
                # Phase2: no hard Track-C total-dim cap; rely on 3-pillar variance target.
                max_total_dims=0,
                horizon=int(horizon),
                sequence_model_type="mamba",
                use_three_pillar_dims=True,
                three_pillar_dim_max=512,
                three_pillar_pca_variance=0.95,
                # FIX: Disable smoothing tuning to enable Track-C cache reuse across trials
                smoothing_types=("none",),
            )
            # Pin seq_len categorical to the study space to avoid dynamic value-space errors.
            try:
                setattr(opt_cfg, "mamba_seq_len_choices", tuple(int(x) for x in seq_len_choices_for_study))
            except Exception:
                pass

            optimizer = StageBOptunaOptimizer(config=opt_cfg)
            if _full_family_sizes is not None:
                try:
                    optimizer.family_sizes = dict(_full_family_sizes)
                except Exception:
                    pass

            # Joint tuning: always sample pipeline params.
            pipeline_params = optimizer._suggest_pipeline_params(trial)
            # Governance: Track weights + all per-family weights are owned by Stage-A selector.
            # Phase2 full-mode Optuna does NOT tune them.
            w_a = 1.0
            w_b = 1.0

            # Load Stage-A governance weights once (resolver uses first eval symbol).
            try:
                # NOTE: In full-mode Optuna, we don't have a single "base cfg" object in scope.
                # Stage-A weights are governed by runtime_overrides (explicit path) and/or
                # Stage-A artifact discovery keys; both can be provided via runtime_overrides.
                _cfg_for_stage_a = dict(runtime_overrides or {})

                _stage_a_payload = _resolve_phase2_stage_a_payload(
                    _cfg_for_stage_a,
                    symbol=str(eval_symbols[0]).upper(),
                    horizon=int(horizon),
                )

                _top_level_w = _extract_stage_a_weights_from_payload(_stage_a_payload) or {}
                _has_schedule = bool(_extract_stage_a_schedule(_stage_a_payload))
                _stage_a_weights_all = (
                    _top_level_w
                    or _phase2_static_stage_a_weights(payload=_stage_a_payload, asof=pd.to_datetime(train_end))
                    or {}
                )

                try:
                    source = "top_level" if _top_level_w else ("schedule_asof" if _has_schedule else "none")
                    trial.set_user_attr("phase2_stage_a_weights_source", str(source))
                    trial.set_user_attr("phase2_stage_a_weights_n", int(len(_stage_a_weights_all)))
                    trial.set_user_attr("phase2_stage_a_weights_has_schedule", bool(_has_schedule))
                    trial.set_user_attr("phase2_stage_a_weights_asof", str(pd.to_datetime(train_end)))
                    # If an explicit path is provided, record it for auditability.
                    _p = (_cfg_for_stage_a.get("phase2_family_weights_path") or _cfg_for_stage_a.get("phase2_stage_a_weights_path"))
                    if _p:
                        trial.set_user_attr("phase2_stage_a_weights_path", str(_p))
                except Exception:
                    pass

                if _has_schedule and _top_level_w:
                    logger.info(
                        "Phase2: Stage-A artifact contains schedule but top-level weights exist; using STATIC top-level weights (n=%d) and ignoring schedule",
                        int(len(_top_level_w)),
                    )
            except Exception as e:
                logger.exception("Phase2: failed to resolve Stage-A selector weights (falling back to 1.0)")
                try:
                    trial.set_user_attr("phase2_stage_a_weights_error", f"{type(e).__name__}: {e}")
                except Exception:
                    pass
                _stage_a_weights_all = {}
            # Sample Mamba params in a stable way, matching Stage-B's Ray Tune search space.
            mamba_seq_choices = tuple(int(x) for x in getattr(opt_cfg, "mamba_seq_len_choices", ()) or ())
            if not mamba_seq_choices:
                # Fallback to optimizer logic if not explicitly pinned.
                try:
                    mamba_seq_choices = tuple(int(x) for x in optimizer._get_effective_mamba_seq_len_choices())
                except Exception:
                    mamba_seq_choices = tuple(int(x) for x in seq_len_choices_for_study)

            mamba_head_hidden_choices = list(
                range(
                    int(getattr(opt_cfg, "mamba_head_hidden_dim_min", 64)),
                    int(getattr(opt_cfg, "mamba_head_hidden_dim_max", 256)) + 1,
                    32,
                )
            )
            if not mamba_head_hidden_choices:
                mamba_head_hidden_choices = [int(getattr(opt_cfg, "mamba_head_hidden_dim_min", 64))]

            # ------------------------------
            # Mamba: conditional blocks (optimizer / scheduler)
            # ------------------------------
            sampled_opt = str(
                trial.suggest_categorical(
                    "mamba_optimizer", list(getattr(opt_cfg, "mamba_optimizer_choices", ("adamw",)))
                )
            )
            sampled_sched = str(
                trial.suggest_categorical(
                    "mamba_lr_scheduler", list(getattr(opt_cfg, "mamba_lr_scheduler_choices", ("none",)))
                )
            )

            if sampled_opt.lower() == "lion":
                # Lion uses weight decay in our implementation; keep the search tight.
                sampled_wd = trial.suggest_float("mamba_weight_decay", 1e-8, 1e-4, log=True)
            else:
                sampled_wd = trial.suggest_float(
                    "mamba_weight_decay",
                    float(getattr(opt_cfg, "mamba_weight_decay_min", 1e-6)),
                    float(getattr(opt_cfg, "mamba_weight_decay_max", 1e-2)),
                    log=True,
                )

            if sampled_sched.lower() == "linear_warmup_cosine":
                sampled_warmup = int(
                    trial.suggest_categorical(
                        "mamba_warmup_steps", list(getattr(opt_cfg, "mamba_warmup_steps_choices", (0,)))
                    )
                )
            else:
                sampled_warmup = 0

            # In v2-only studies, head/loss are forced to gaussian.
            # For NEW studies, do not sample head/loss to keep search space clean.
            suggest_head_loss = bool(existing_loss_choices is not None or existing_head_choices is not None)

            mamba_params: Dict[str, Any] = {
                "mamba_d_model": trial.suggest_categorical(
                    "mamba_d_model", list(getattr(opt_cfg, "mamba_d_model_choices", (128,)))
                ),
                "mamba_n_layers": trial.suggest_categorical(
                    "mamba_n_layers", list(getattr(opt_cfg, "mamba_n_layers_choices", (4,)))
                ),
                "mamba_ssm_dim": trial.suggest_categorical(
                    "mamba_ssm_dim", list(getattr(opt_cfg, "mamba_ssm_dim_choices", (96,)))
                ),
                "mamba_expand_factor": trial.suggest_categorical(
                    "mamba_expand_factor", list(getattr(opt_cfg, "mamba_expand_factor_choices", (2.0,)))
                ),
                "mamba_seq_len": 315,  # FIXED: force max sequence length for better long-term memory
                "mamba_activation": trial.suggest_categorical(
                    "mamba_activation", list(getattr(opt_cfg, "mamba_activation_choices", ("silu",)))
                ),
                "mamba_norm_type": trial.suggest_categorical(
                    "mamba_norm_type", list(getattr(opt_cfg, "mamba_norm_type_choices", ("rmsnorm",)))
                ),
                "mamba_norm_strategy": trial.suggest_categorical(
                    "mamba_norm_strategy", list(getattr(opt_cfg, "mamba_norm_strategy_choices", ("pre",)))
                ),
                "mamba_dropout": trial.suggest_float(
                    "mamba_dropout",
                    float(getattr(opt_cfg, "mamba_dropout_min", 0.0)),
                    float(getattr(opt_cfg, "mamba_dropout_max", 0.5)),
                ),
                "mamba_resid_dropout": trial.suggest_categorical(
                    "mamba_resid_dropout", list(getattr(opt_cfg, "mamba_resid_dropout_choices", (0.0,)))
                ),
                "mamba_ssm_dropout": trial.suggest_categorical(
                    "mamba_ssm_dropout", list(getattr(opt_cfg, "mamba_ssm_dropout_choices", (0.0,)))
                ),
                "mamba_gate_dropout": trial.suggest_categorical(
                    "mamba_gate_dropout", list(getattr(opt_cfg, "mamba_gate_dropout_choices", (0.0,)))
                ),
                "mamba_optimizer": str(sampled_opt),
                "mamba_learning_rate": trial.suggest_float(
                    "mamba_learning_rate",
                    float(getattr(opt_cfg, "mamba_lr_min", 1e-5)),
                    float(getattr(opt_cfg, "mamba_lr_max", 1e-3)),
                    log=True,
                ),
                "mamba_weight_decay": float(sampled_wd),
                "mamba_grad_clip": trial.suggest_categorical(
                    "mamba_grad_clip", list(getattr(opt_cfg, "mamba_grad_clip_choices", (1.0,)))
                ),
                "mamba_lr_scheduler": str(sampled_sched),
                "mamba_warmup_steps": int(sampled_warmup),
                "mamba_max_epochs": trial.suggest_categorical(
                    "mamba_max_epochs", list(getattr(opt_cfg, "mamba_max_epochs_choices", (10,)))
                ),
                "mamba_batch_size": trial.suggest_categorical(
                    "mamba_batch_size", list(getattr(opt_cfg, "mamba_batch_size_choices", (32,)))
                ),
                "mamba_loss_fn": (
                    trial.suggest_categorical("mamba_loss_fn", list(loss_choices_for_study))
                    if suggest_head_loss
                    else "gaussian_nll"
                ),
                "mamba_head_type": (
                    trial.suggest_categorical("mamba_head_type", list(head_choices_for_study))
                    if suggest_head_loss
                    else "gaussian"
                ),
                "mamba_head_hidden_dim": trial.suggest_categorical(
                    "mamba_head_hidden_dim", mamba_head_hidden_choices
                ),
                "mamba_head_num_layers": trial.suggest_categorical(
                    "mamba_head_num_layers", list(getattr(opt_cfg, "mamba_head_num_layers_choices", (1, 2, 3)))
                ),
                "mamba_head_dropout": trial.suggest_float(
                    "mamba_head_dropout",
                    float(getattr(opt_cfg, "mamba_head_dropout_min", 0.0)),
                    float(getattr(opt_cfg, "mamba_head_dropout_max", 0.3)),
                ),
            }

            # Compute envelope guardrail (cheap proxy): prune pathological configs early.
            d_model_v = float(mamba_params.get("mamba_d_model", 128))
            n_layers_v = float(mamba_params.get("mamba_n_layers", 4))
            seq_len_v = float(mamba_params.get("mamba_seq_len", 128))
            batch_v = float(mamba_params.get("mamba_batch_size", 32))
            epochs_v = float(mamba_params.get("mamba_max_epochs", 10))
            cost_proxy = (d_model_v / 128.0) * (n_layers_v / 4.0) * (seq_len_v / 128.0) * (batch_v / 32.0) * (epochs_v / 10.0)
            try:
                trial.set_user_attr("phase2_cost_proxy", float(cost_proxy))
            except Exception:
                pass
            if float(cost_proxy) > 8.0:
                raise optuna.exceptions.TrialPruned(f"compute envelope: cost_proxy={cost_proxy:.2f} > 8.0")

            cfg_local: Dict[str, Any] = {}
            cfg_local.update(pipeline_params)
            cfg_local["track_a_weight"] = float(w_a)
            cfg_local["track_b_weight"] = float(w_b)
            # Phase2: remove hard cap on total Track-C dims.
            cfg_local["max_total_dims"] = 0
            
            # ─────────────────────────────────────────────────────────────────────
            # Phase 2.4: Label Type Optuna Parameter
            # Search between raw forward_return vs volatility-adjusted return
            # ─────────────────────────────────────────────────────────────────────
            label_type_searchable = bool(getattr(opt_cfg, "label_type_searchable", False))
            if label_type_searchable:
                sampled_label_id = trial.suggest_categorical(
                    "phase2_label_id", 
                    ["base", "voladj"]  # base=forward_return, voladj=forward_return_voladj
                )
                cfg_local["phase2_label_id"] = str(sampled_label_id)
            
            # ─────────────────────────────────────────────────────────────────────
            # Phase 2.2: Info-Weighted Loss
            # Weight samples by microstructure-derived importance
            # ─────────────────────────────────────────────────────────────────────
            info_weighted_loss_searchable = bool(getattr(opt_cfg, "info_weighted_loss_searchable", False))
            if info_weighted_loss_searchable:
                use_info_weighted = trial.suggest_categorical(
                    "mamba_use_info_weighted_loss",
                    [False, True]
                )
                cfg_local["mamba_use_info_weighted_loss"] = bool(use_info_weighted)
            
            # ─────────────────────────────────────────────────────────────────────
            # Phase 2.3: dt_days Feature
            # Add time-delta as input feature for irregular sampling awareness
            # ─────────────────────────────────────────────────────────────────────
            dt_days_feature_searchable = bool(getattr(opt_cfg, "dt_days_feature_searchable", False))
            if dt_days_feature_searchable:
                include_dt_days = trial.suggest_categorical(
                    "include_dt_days_feature",
                    [False, True]
                )
                cfg_local["include_dt_days_feature"] = bool(include_dt_days)

            # ─────────────────────────────────────────────────────────────────────
            # Phase 2.6: Multi-Horizon Training (Workstream 6)
            # Train on multiple horizons simultaneously for shared representation
            # learning and implicit regime smoothing. At inference, only the
            # primary horizon is used for downstream portfolio/policy.
            # ─────────────────────────────────────────────────────────────────────
            multi_horizon_searchable = bool(getattr(opt_cfg, "multi_horizon_searchable", False))
            if multi_horizon_searchable:
                use_multi_horizon = trial.suggest_categorical(
                    "multi_horizon_mode",
                    [False, True]
                )
                cfg_local["multi_horizon_mode"] = bool(use_multi_horizon)
            else:
                # Use default from cfg or global default
                cfg_local["multi_horizon_mode"] = bool(cfg_local.get(
                    "multi_horizon_mode", 
                    DEFAULT_MULTI_HORIZON_MODE
                ))
            
            # Multi-horizon set and primary are FIXED (not Optuna-tuned initially)
            # to avoid combinatorial explosion.
            cfg_local["multi_horizon_set"] = list(cfg_local.get(
                "multi_horizon_set", DEFAULT_MULTI_HORIZON_SET
            ))
            cfg_local["multi_horizon_primary"] = int(cfg_local.get(
                "multi_horizon_primary", DEFAULT_MULTI_HORIZON_PRIMARY
            ))

            # Persist the 3-pillar configuration used for Track-C dimensionality selection.
            # (Not sampled; governance-level knobs.)
            try:
                cfg_local["three_pillar_dim_min"] = int(getattr(opt_cfg, "three_pillar_dim_min", 4))
                cfg_local["three_pillar_dim_max"] = int(getattr(opt_cfg, "three_pillar_dim_max", 32))
                cfg_local["three_pillar_pca_variance"] = float(getattr(opt_cfg, "three_pillar_pca_variance", 0.95))
            except Exception:
                pass

            # Persist Stage-A weights artifact path in cfg (usually passed via runtime_overrides).
            try:
                rt = dict(runtime_overrides or {})
                if rt.get("phase2_stage_a_weights_path"):
                    cfg_local["phase2_stage_a_weights_path"] = rt.get("phase2_stage_a_weights_path")
                if rt.get("phase2_family_weights_path"):
                    cfg_local["phase2_family_weights_path"] = rt.get("phase2_family_weights_path")
            except Exception:
                pass

            clip_min = float(getattr(opt_cfg, "family_weight_clip_min", 0.01))
            # Track-B family weights (fixed, governed by Stage-A selector if provided).
            try:
                for fam in HF_BLOCK_FAMILIES:
                    v = float(_stage_a_weights_all.get(str(fam), 1.0))
                    cfg_local[f"weight_b_{fam}"] = float(max(v, clip_min))
                for summary_name in TRACK_B_SUMMARY_BLOCKS:
                    v = float(_stage_a_weights_all.get(str(summary_name), 1.0))
                    cfg_local[f"weight_b_{summary_name}"] = float(max(v, clip_min))
            except Exception:
                pass

            # IMPORTANT: For Phase-2 v2 grouped TPE, we avoid sampling params that are
            # irrelevant given other choices. We still populate a complete cfg so the
            # downstream pipeline can build Track-C deterministically.
            for fam in STAGE_A_FAMILIES:
                fam = str(fam)
                family_size = int(getattr(optimizer, "family_sizes", {}).get(fam, 0) or 0)

                # Per-family weights are fixed governance inputs from Stage-A selector.
                w_val = float(_stage_a_weights_all.get(fam, 1.0))
                cfg_local[f"weight_{fam}"] = float(max(w_val, clip_min))

                if family_size <= 0:
                    # Keep stable keys even for missing families.
                    cfg_local[f"{fam}_dim_type"] = "pca"
                    cfg_local[f"{fam}_pca_dim"] = 0
                    cfg_local[f"{fam}_ae_dim"] = 0
                    cfg_local[f"{fam}_ae_layers"] = int((getattr(opt_cfg, "ae_layers_choices", (2,)) or (2,))[0])
                    cfg_local[f"{fam}_ae_activation"] = str((getattr(opt_cfg, "ae_activation_choices", ("gelu",)) or ("gelu",))[0])
                    cfg_local[f"{fam}_ae_dropout"] = float(getattr(opt_cfg, "ae_dropout_min", 0.0))
                    cfg_local[f"{fam}_ae_lr"] = float(getattr(opt_cfg, "ae_lr_min", 1e-5))
                    cfg_local[f"{fam}_dim"] = 0
                    continue

                # =========================================================================
                # 3-PILLAR: fixed dims ONLY (no Optuna sampling)
                # =========================================================================
                method = "pca"
                k_final = int(min(int(getattr(opt_cfg, "three_pillar_dim_min", 4)), int(family_size)))
                try:
                    if _full_three_pillar_dims and fam in _full_three_pillar_dims:
                        method = str(_full_three_pillar_dims[fam].get("method", method))
                        k_final = int(_full_three_pillar_dims[fam].get("k_final", k_final))
                except Exception:
                    pass

                k_min = int(getattr(opt_cfg, "three_pillar_dim_min", 4))
                k_max = int(getattr(opt_cfg, "three_pillar_dim_max", 32))
                k_final = int(min(max(int(k_final), int(k_min)), min(int(k_max), int(family_size))))

                cfg_local[f"{fam}_dim_type"] = str(method)
                cfg_local[f"{fam}_pca_dim"] = int(k_final)
                cfg_local[f"{fam}_ae_dim"] = int(k_final)
                cfg_local[f"{fam}_ae_layers"] = 2
                cfg_local[f"{fam}_ae_activation"] = "gelu"
                cfg_local[f"{fam}_ae_dropout"] = 0.1
                cfg_local[f"{fam}_ae_lr"] = 5e-4
                cfg_local[f"{fam}_dim"] = int(k_final)

            # Audit: persist per-family 3-pillar dims used by this trial.
            try:
                if _full_three_pillar_dims:
                    trial.set_user_attr("phase2_three_pillar_dims", dict(_full_three_pillar_dims))
                    trial.set_user_attr(
                        "phase2_three_pillar_total_dim",
                        int(sum(int(v.get("k_final", 0) or 0) for v in _full_three_pillar_dims.values())),
                    )
            except Exception:
                pass

            # ------------------------------
            # Thresholds: v2 regime-threshold keys only (clean subspace)
            # ------------------------------
            cfg_local["regime_threshold_base"] = float(trial.suggest_float("regime_threshold_base", 0.02, 0.20))
            cfg_local["regime_threshold_bull_mult"] = float(trial.suggest_float("regime_threshold_bull_mult", 0.6, 1.2))
            cfg_local["regime_threshold_bear_mult"] = float(trial.suggest_float("regime_threshold_bear_mult", 0.8, 2.0))
            cfg_local["regime_threshold_crisis_mult"] = float(trial.suggest_float("regime_threshold_crisis_mult", 1.0, 4.0))
            cfg_local["conf_threshold"] = float(trial.suggest_float("conf_threshold", 0.3, 0.8))
            cfg_local["vol_scaler"] = float(trial.suggest_float("vol_scaler", 0.0, 1.0))

            # Provide legacy keys for any older diagnostics/backtest paths (not sampled).
            cfg_local["threshold"] = float(cfg_local["regime_threshold_base"])
            cfg_local["bull_mult"] = float(cfg_local["regime_threshold_bull_mult"])
            cfg_local["bear_mult"] = float(cfg_local["regime_threshold_bear_mult"])
            cfg_local["crisis_mult"] = float(cfg_local["regime_threshold_crisis_mult"])

            cfg_local.update({k: v for k, v in dict(mamba_params).items()})

            # Phase2 v2 forces uncertainty training/inference. Keep cfg explicit so:
            # - trial exports are truthful
            # - downstream code cannot accidentally run with non-gaussian head
            cfg_local["mamba_head_type"] = "gaussian"
            cfg_local["mamba_loss_fn"] = "gaussian_nll"
            try:
                trial.set_user_attr("phase2_forced_mamba_head_type", "gaussian")
                trial.set_user_attr("phase2_forced_mamba_loss_fn", "gaussian_nll")
            except Exception:
                pass

            # Regime-weighted Gaussian NLL + uncertainty regularizers.
            cfg_local["phase2_regime_feature_index"] = int(
                (runtime_overrides or {}).get("phase2_regime_feature_index", -1)
            )
            cfg_local["phase2_regime_loss_alpha"] = float(
                trial.suggest_float("phase2_regime_loss_alpha", 0.0, 1.0)
            )
            cfg_local["phase2_sigma_floor_base"] = float(
                trial.suggest_float("phase2_sigma_floor_base", 1e-6, 1e-3)
            )
            cfg_local["phase2_sigma_floor_stress"] = float(
                trial.suggest_float("phase2_sigma_floor_stress", 1e-6, 1e-2)
            )
            cfg_local["phase2_sigma_floor_strength"] = float(
                trial.suggest_float("phase2_sigma_floor_strength", 0.0, 1e-2)
            )
            cfg_local["phase2_anchor_l2"] = float(
                trial.suggest_float("phase2_anchor_l2", 0.0, 1e-4)
            )
            cfg_local["phase2_sigma_cs_strength"] = float(
                trial.suggest_float("phase2_sigma_cs_strength", 0.0, 1e-3)
            )

            # ─────────────────────────────────────────────────────────────────────
            # Calibration-Gated Learning Thresholds (Optuna-tuned with constraints)
            # ─────────────────────────────────────────────────────────────────────
            # calib_freeze_threshold: FIXED - governance level, not model-sensitive
            cfg_local["calib_freeze_threshold"] = 0.3
            # trust_freeze_threshold: FIXED - governance level
            cfg_local["trust_freeze_threshold"] = 0.3

            # calib_decay_threshold: Optuna-tuned in narrow band [0.5, 0.75]
            # CONSTRAINT: must be > calib_freeze_threshold (enforced below)
            calib_decay_raw = float(trial.suggest_float("calib_decay_threshold", 0.5, 0.75))
            calib_freeze = float(cfg_local["calib_freeze_threshold"])
            # Hard enforce: calib_decay > calib_freeze
            calib_decay_threshold = max(calib_decay_raw, calib_freeze + 0.05)
            cfg_local["calib_decay_threshold"] = float(calib_decay_threshold)

            # overconfidence_penalty_lambda: Optuna-tuned on log scale [1e-3, 3e-1]
            cfg_local["overconfidence_penalty_lambda"] = float(
                trial.suggest_float("overconfidence_penalty_lambda", 1e-3, 3e-1, log=True)
            )

            # Walk-forward portfolio engine knobs (Phase-2 v2).
            # NOTE: update cadence is fixed to U=prune_update_sessions; we do not tune it.
            cfg_local["phase2_engine"] = str(trial.suggest_categorical("phase2_engine", ["v2"]))
            cfg_local["phase2_update_sessions"] = int(trial.suggest_categorical("phase2_update_sessions", [int(prune_update_sessions)]))
            cfg_local["phase2_replay_days"] = 315  # FIXED: max replay window for better regime adaptation
            cfg_local["phase2_update_epochs"] = int(trial.suggest_categorical("phase2_update_epochs", [1, 2, 3, 4, 5]))
            cfg_local["phase2_cov_ewma_lambda"] = float(trial.suggest_float("phase2_cov_ewma_lambda", 0.90, 0.99))
            cfg_local["phase2_shrinkage_alpha"] = float(trial.suggest_float("phase2_shrinkage_alpha", 0.0, 0.30))
            cfg_local["phase2_target_vol"] = float(trial.suggest_float("phase2_target_vol", 0.10, 0.15))
            # GOVERNANCE: phase2_max_gross is now fixed at 1.75, not a tuning knob
            # Hedge fund rule: vol scaler determines exposure, not arbitrary gross caps
            cfg_local["phase2_max_gross"] = 1.75
            cfg_local["phase2_max_name"] = float(trial.suggest_float("phase2_max_name", 0.02, 0.25))
            cfg_local["phase2_max_net"] = float(trial.suggest_float("phase2_max_net", 0.0, 0.50))
            cfg_local["phase2_k_spread"] = float(trial.suggest_float("phase2_k_spread", 5e-5, 3e-4, log=True))
            cfg_local["phase2_k_impact"] = float(trial.suggest_float("phase2_k_impact", 0.0, 1e-3))
            cfg_local["phase2_z_clip"] = float(trial.suggest_categorical("phase2_z_clip", [4.0, 6.0, 8.0, 10.0]))

            # ------------------------------
            # Optional HF-grade overlays: conditional blocks
            # ------------------------------
            rt = dict(runtime_overrides or {})

            has_group_map = bool(rt.get("phase2_group_map_path"))
            has_capital = float(rt.get("phase2_capital_usd", 0.0) or 0.0) > 0.0

            # Turnover overlay
            turnover_enabled = bool(trial.suggest_categorical("phase2_enable_turnover_overlay", [False, True]))
            cfg_local["phase2_enable_turnover_overlay"] = bool(turnover_enabled)
            if turnover_enabled:
                cfg_local["phase2_turnover_cap"] = float(trial.suggest_float("phase2_turnover_cap", 0.05, 1.0))
                kill_enabled = bool(trial.suggest_categorical("phase2_enable_turnover_kill_switch", [False, True]))
                cfg_local["phase2_enable_turnover_kill_switch"] = bool(kill_enabled)
                if kill_enabled:
                    cfg_local["phase2_kill_on_turnover_gt"] = float(trial.suggest_float("phase2_kill_on_turnover_gt", 0.5, 3.0))
                    cfg_local["phase2_flat_cooldown_sessions"] = int(trial.suggest_categorical("phase2_flat_cooldown_sessions", [0, 21, 42, 63]))
            else:
                cfg_local["phase2_enable_turnover_kill_switch"] = False

            # Weight smoothing
            smoothing_enabled = bool(trial.suggest_categorical("phase2_enable_weight_smoothing", [False, True]))
            cfg_local["phase2_enable_weight_smoothing"] = bool(smoothing_enabled)
            if smoothing_enabled:
                cfg_local["phase2_weight_smoothing_alpha"] = float(trial.suggest_float("phase2_weight_smoothing_alpha", 0.05, 0.35))

            # Group caps (only meaningful if we have a group map)
            if has_group_map:
                group_caps_enabled = bool(trial.suggest_categorical("phase2_enable_group_caps", [False, True]))
                cfg_local["phase2_enable_group_caps"] = bool(group_caps_enabled)
                if group_caps_enabled:
                    cfg_local["phase2_group_max_gross"] = float(trial.suggest_float("phase2_group_max_gross", 0.05, 0.50))
                    cfg_local["phase2_group_max_net"] = float(trial.suggest_float("phase2_group_max_net", 0.0, 0.30))
            else:
                cfg_local["phase2_enable_group_caps"] = False

            # Beta neutralization: HARD neutralization with optional cap
            # NOTE: Implementation ALWAYS neutralizes first, then allows capped exposure.
            # This is NOT a "soft guardrail" - it's full neutralization + optional cap.
            beta_enabled = bool(trial.suggest_categorical("phase2_enable_beta_neutral", [False, True]))
            cfg_local["phase2_enable_beta_neutral"] = bool(beta_enabled)
            if beta_enabled:
                cfg_local["phase2_beta_neutral"] = True  # DEPRECATED flag
                # Hedge fund best practice: 0.25-0.35 allows partial beta participation
                # 0.0 = full market neutral, 0.3 = up to ±30% beta exposure allowed
                cfg_local["phase2_beta_max_abs_exposure"] = float(trial.suggest_float("phase2_beta_max_abs_exposure", 0.15, 0.40))
                cfg_local["phase2_beta_lookback_days"] = int(trial.suggest_categorical("phase2_beta_lookback_days", [63, 126, 252]))

            # Liquidity constraints (only meaningful if capital base is provided)
            if has_capital:
                liq_enabled = bool(trial.suggest_categorical("phase2_enable_liquidity_constraints", [False, True]))
                cfg_local["phase2_enable_liquidity_constraints"] = bool(liq_enabled)
                if liq_enabled:
                    cfg_local["phase2_adv_window"] = int(trial.suggest_categorical("phase2_adv_window", [10, 20, 40]))
                    cfg_local["phase2_max_adv_frac_name"] = float(trial.suggest_float("phase2_max_adv_frac_name", 0.001, 0.05, log=True))
                    cfg_local["phase2_max_turnover_adv_frac"] = float(trial.suggest_float("phase2_max_turnover_adv_frac", 0.001, 0.20, log=True))
                    cfg_local["phase2_borrow_fee_bps_annual"] = float(trial.suggest_float("phase2_borrow_fee_bps_annual", 0.0, 300.0))
            else:
                cfg_local["phase2_enable_liquidity_constraints"] = False

            # ─────────────────────────────────────────────────────────────────────
            # Policy Controller v2 (Workstream 8) - Contextual Bandit Knobs
            # ─────────────────────────────────────────────────────────────────────
            policy_enabled = bool(trial.suggest_categorical("phase2_policy_enabled", [False, True]))
            cfg_local["phase2_policy_enabled"] = bool(policy_enabled)
            if policy_enabled:
                # Exploration annealing (half-life in sessions)
                cfg_local["phase2_policy_exploration_half_life"] = int(
                    trial.suggest_categorical("phase2_policy_exploration_half_life", [42, 63, 126, 252])
                )
                # Exploration floor (fraction of initial noise)
                cfg_local["phase2_policy_exploration_floor"] = float(
                    trial.suggest_float("phase2_policy_exploration_floor", 0.05, 0.30)
                )
                # Lin-TS prior variance
                cfg_local["phase2_policy_ts_prior_var"] = float(
                    trial.suggest_float("phase2_policy_ts_prior_var", 0.5, 2.0)
                )
                # Lin-TS noise variance (initial)
                cfg_local["phase2_policy_ts_noise_var"] = float(
                    trial.suggest_float("phase2_policy_ts_noise_var", 0.5, 2.0)
                )
                # Warmup days (random action selection)
                cfg_local["phase2_policy_warmup_days"] = int(
                    trial.suggest_categorical("phase2_policy_warmup_days", [10, 20, 42])
                )
                # Reward function penalty weights
                cfg_local["phase2_policy_reward_lambda_turn"] = float(
                    trial.suggest_float("phase2_policy_reward_lambda_turn", 0.05, 0.5, log=True)
                )
                cfg_local["phase2_policy_reward_lambda_dd"] = float(
                    trial.suggest_float("phase2_policy_reward_lambda_dd", 0.2, 1.0)
                )
                # Reward window (sessions over which to compute reward)
                cfg_local["phase2_policy_reward_window"] = int(
                    trial.suggest_categorical("phase2_policy_reward_window", [5, 10, 21])
                )
            else:
                # Defaults when policy disabled
                cfg_local["phase2_policy_exploration_half_life"] = 63
                cfg_local["phase2_policy_exploration_floor"] = 0.15
                cfg_local["phase2_policy_ts_prior_var"] = 1.0
                cfg_local["phase2_policy_ts_noise_var"] = 1.0
                cfg_local["phase2_policy_warmup_days"] = 20
                cfg_local["phase2_policy_reward_lambda_turn"] = 0.2
                cfg_local["phase2_policy_reward_lambda_dd"] = 0.5
                cfg_local["phase2_policy_reward_window"] = 10

            # ─────────────────────────────────────────────────────────────────────
            # Linear Alpha Combiner (Workstream: z_lin blending)
            # Ridge-regularized linear model that blends with Mamba z-scores
            # ─────────────────────────────────────────────────────────────────────
            linear_enabled = bool(trial.suggest_categorical("phase2_linear_model_enabled", [False, True]))
            cfg_local["phase2_linear_model_enabled"] = bool(linear_enabled)
            if linear_enabled:
                cfg_local["phase2_linear_ridge_lambda"] = float(
                    trial.suggest_float("phase2_linear_ridge_lambda", 1.0, 100.0, log=True)
                )
                cfg_local["phase2_linear_blend_max"] = float(
                    trial.suggest_float("phase2_linear_blend_max", 0.20, 0.50)
                )
                cfg_local["phase2_linear_window"] = int(
                    trial.suggest_categorical("phase2_linear_window", [63, 126, 189, 252])
                )
                cfg_local["phase2_linear_min_samples"] = int(
                    trial.suggest_categorical("phase2_linear_min_samples", [42, 63, 84])
                )
            else:
                # Defaults when disabled (won't affect loop since model is off)
                cfg_local["phase2_linear_ridge_lambda"] = 10.0
                cfg_local["phase2_linear_blend_max"] = 0.5
                cfg_local["phase2_linear_window"] = 126
                cfg_local["phase2_linear_min_samples"] = 63

            # Fixed governance (not sampled)
            cfg_local["phase2_linear_update_interval"] = 21
            cfg_local["phase2_linear_max_window"] = 252

            # Strict audit: sampled params must be consumed (or explicitly helper-only).
            _audit_phase2_trial_params(mode_local="full", trial=trial, cfg_local=cfg_local)

            # Enforce that every sampled Optuna parameter is present in the evaluated cfg.
            # This is a necessary (though not sufficient) condition for "all params are used".
            try:
                trial_params_keys = set((getattr(trial, "params", {}) or {}).keys())
                cfg_keys = set(cfg_local.keys())
                missing_in_cfg = sorted(trial_params_keys - cfg_keys)
                trial.set_user_attr("phase2_trial_param_count", int(len(trial_params_keys)))
                trial.set_user_attr("phase2_trial_cfg_key_count", int(len(cfg_keys)))
                trial.set_user_attr("phase2_trial_missing_params_in_cfg", list(missing_in_cfg))
                if missing_in_cfg:
                    logger.error(
                        "Phase-2 full-mode trial dropped sampled params (count=%d). Examples: %s",
                        len(missing_in_cfg),
                        ", ".join(missing_in_cfg[:20]),
                    )
                    return float("-inf")
            except Exception:
                pass

            # Persist the exact config used for evaluation (so we can later verify
            # what was actually applied, and export a faithful winner payload).
            def _jsonable(x: Any) -> Any:
                try:
                    import numpy as _np

                    if isinstance(x, (_np.integer,)):
                        return int(x)
                    if isinstance(x, (_np.floating,)):
                        return float(x)
                    if isinstance(x, (_np.bool_,)):
                        return bool(x)
                except Exception:
                    pass
                if isinstance(x, dict):
                    return {str(k): _jsonable(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_jsonable(v) for v in x]
                return x

            try:
                # Make label identity explicit in every evaluated cfg.
                cfg_local["phase2_label_id"] = str(label_id)
                try:
                    trial.set_user_attr("phase2_label_id", str(label_id))
                except Exception:
                    pass
            except Exception:
                pass

            # IMPORTANT: runtime_overrides are part of the evaluated config.
            # They must be applied *after* sampling so enforced posture wins, and
            # must be persisted so study artifacts are auditable.
            cfg_eval = dict(cfg_local)
            try:
                cfg_eval.update(dict(runtime_overrides or {}))
            except Exception:
                pass

            try:
                trial.set_user_attr("phase2_trial_cfg", _jsonable(cfg_eval))
                trial.set_user_attr("phase2_prepared_sig", str(_phase2_cfg_signature_for_prepared(cfg_eval)))
            except Exception:
                pass

            res = evaluate_phase2_stateful_once(
                trial_cfg=cfg_eval,
                symbols=eval_symbols,
                horizon=horizon,
                train_start=train_start,
                train_end=train_end,
                oos_start=oos_start,
                oos_end=oos_end,
                portfolio_weights=portfolio_weights,
                aggregation_rule=aggregation_rule,
                objective_spec=objective_spec,
                runtime_overrides=runtime_overrides,
                trial=trial,
                prune_oos_days=None,
                prune_update_sessions=int(prune_update_sessions),
                prune_warmup_folds=int(prune_warmup_folds),
                safety_prune_after_folds=0 if bool(no_prune) else int(safety_prune_after_folds),
                safety_prune_sharpe_floor=float(safety_prune_sharpe_floor),
                safety_prune_maxdd_ceiling=float(safety_prune_maxdd_ceiling),
                deterministic_all=bool(deterministic_all),
            )
            return float(res.objective)

        except optuna.TrialPruned:
            raise
        except NonFiniteError as e:
            logger.exception("Phase-2 non-finite failure: %s", e)
            return float("-inf")
        except Exception as e:
            logger.exception("Phase-2 trial failed: %s", e)
            return float("-inf")

    # Persist aggregation + objective semantics so Phase-2 cannot silently drift.
    try:
        study.set_user_attr("phase2_search_mode", str(mode))
        study.set_user_attr("phase2_aggregation_rule", str(aggregation_rule))
        study.set_user_attr("phase2_portfolio_weights", {str(k).upper(): float(v) for k, v in (portfolio_weights or {}).items()})
        study.set_user_attr("phase2_objective", {
            "type": "sharpe_minus_drawdown",
            "max_drawdown_penalty": float(objective_spec.max_drawdown_penalty),
            "turnover_penalty": float(objective_spec.turnover_penalty),
        })
        study.set_user_attr("phase2_symbols", [str(s).upper() for s in eval_symbols])
        study.set_user_attr("phase2_fixed_blocks", {
            "train_start": str(train_start),
            "train_end": str(train_end),
            "oos_start": str(oos_start),
            "oos_end": str(oos_end),
            "holdout_start": str(holdout_start) if holdout_start is not None else None,
            "holdout_end": str(holdout_end) if holdout_end is not None else None,
            "horizon": int(horizon),
        })
        if mode == "refinement":
            search_keys = [
                "mamba_seq_len",
            ]
            search_keys.extend([
                "mamba_learning_rate",
                "mamba_dropout",
                "mamba_head_dropout",
                "mamba_grad_clip",
            ])
            study.set_user_attr("phase2_search_keys", search_keys)
        elif mode == "full":
            # Full-mode search space intentionally excludes:
            # - track weights
            # - per-family weights
            # - per-family dim/dim_type params
            # These are governed by Stage-A selector + 3-pillar fixed dims.
            study.set_user_attr(
                "phase2_search_keys",
                [
                    # Pipeline (smoothing REMOVED - fixed to none for cache efficiency)
                    # "smoothing_type",
                    # "smoothing_window",
                    "train_fraction",
                    # Mamba (including conditional blocks)
                    "mamba_optimizer",
                    "mamba_lr_scheduler",
                    "mamba_weight_decay",
                    "mamba_warmup_steps",
                    "mamba_d_model",
                    "mamba_n_layers",
                    "mamba_ssm_dim",
                    "mamba_expand_factor",
                    "mamba_seq_len",
                    "mamba_activation",
                    "mamba_norm_type",
                    "mamba_norm_strategy",
                    "mamba_dropout",
                    "mamba_resid_dropout",
                    "mamba_ssm_dropout",
                    "mamba_gate_dropout",
                    "mamba_learning_rate",
                    "mamba_grad_clip",
                    "mamba_max_epochs",
                    "mamba_batch_size",
                    "mamba_head_hidden_dim",
                    "mamba_head_num_layers",
                    "mamba_head_dropout",
                    # Regime/threshold
                    "regime_threshold_base",
                    "regime_threshold_bull_mult",
                    "regime_threshold_bear_mult",
                    "regime_threshold_crisis_mult",
                    "conf_threshold",
                    "vol_scaler",
                    # Phase2 engine knobs
                    "phase2_engine",
                    "phase2_update_sessions",
                    "phase2_replay_days",
                    "phase2_update_epochs",
                    "phase2_cov_ewma_lambda",
                    "phase2_shrinkage_alpha",
                    "phase2_target_vol",
                    "phase2_max_gross",
                    "phase2_max_name",
                    "phase2_max_net",
                    "phase2_k_spread",
                    "phase2_k_impact",
                    "phase2_z_clip",
                    "phase2_regime_feature_index",
                    "phase2_regime_loss_alpha",
                    "phase2_sigma_floor_base",
                    "phase2_sigma_floor_stress",
                    "phase2_sigma_floor_strength",
                    "phase2_anchor_l2",
                    "phase2_sigma_cs_strength",
                    # Optional overlays (conditionally active)
                    "phase2_enable_turnover_overlay",
                    "phase2_turnover_cap",
                    "phase2_enable_turnover_kill_switch",
                    "phase2_kill_on_turnover_gt",
                    "phase2_flat_cooldown_sessions",
                    "phase2_enable_weight_smoothing",
                    "phase2_weight_smoothing_alpha",
                    "phase2_enable_group_caps",
                    "phase2_group_max_gross",
                    "phase2_group_max_net",
                    "phase2_enable_beta_neutral",
                    "phase2_beta_max_abs_exposure",
                    "phase2_beta_lookback_days",
                    "phase2_enable_liquidity_constraints",
                    "phase2_adv_window",
                    "phase2_max_adv_frac_name",
                    "phase2_max_turnover_adv_frac",
                    "phase2_borrow_fee_bps_annual",
                ],
            )
        study.set_user_attr(
            "phase2_pruner",
            {
                "type": "hyperband",
                "min_resource": max(1, int(prune_warmup_folds)),
                "max_resource": int(max_resource),
                "reduction_factor": 3,
                "intermediate_metric": "portfolio_score_prefix_by_fold",
                "step_unit": "fold",
                "update_sessions": int(prune_update_sessions),
                "warmup_folds": int(prune_warmup_folds),
                "final_step": int(max_resource),
            },
        )
        study.set_user_attr(
            "phase2_sampler",
            {
                "type": "tpe",
                "seed": int(seed),
                "n_startup_trials": int(n_startup),
                "top_fraction": float(top_frac),
                "multivariate": True,
                "group": True,
            },
        )
    except Exception:
        pass

    study.optimize(objective, n_trials=int(n_trials))

    best = study.best_trial

    best_overrides: Optional[Dict[str, Any]] = None
    winner_cfg: Optional[Dict[str, Any]] = None
    best_full_cfg: Optional[Dict[str, Any]] = None
    if mode == "refinement":
        # Build a single winner config payload (frozen base + tuned keys).
        best_overrides = {
            "mamba_seq_len": int(best.params.get("mamba_seq_len")),
            "mamba_learning_rate": float(best.params.get("mamba_learning_rate")),
            "mamba_dropout": float(best.params.get("mamba_dropout")),
            "mamba_head_dropout": float(best.params.get("mamba_head_dropout")),
            "mamba_grad_clip": float(best.params.get("mamba_grad_clip")),
        }
        if best_trial_json is None:
            raise ValueError("best_trial_json is required for refinement mode")
        frozen_base = load_stage_b_best_trial_params(best_trial_json)
        winner_cfg = _phase2_apply_refinement_overrides(frozen_base, best_overrides)
    else:
        # Full mode: prefer the exact cfg used during evaluation (stored per-trial).
        try:
            v = (getattr(best, "user_attrs", {}) or {}).get("phase2_trial_cfg")
            if isinstance(v, dict) and v:
                best_full_cfg = dict(v)
        except Exception:
            best_full_cfg = None
        if best_full_cfg is None:
            best_full_cfg = dict(best.params)
        winner_cfg = dict(best_full_cfg)

    # Re-evaluate winner on Tune-OOS for full metrics (objective is scalar only).
    tune_oos_eval = None
    try:
        if mode == "refinement":
            tune_res = evaluate_phase2_stateful_once(
                best_trial_json=best_trial_json,
                symbols=eval_symbols,
                horizon=horizon,
                train_start=train_start,
                train_end=train_end,
                oos_start=oos_start,
                oos_end=oos_end,
                overrides=best_overrides or {},
                portfolio_weights=portfolio_weights,
                aggregation_rule=aggregation_rule,
                objective_spec=objective_spec,
                runtime_overrides=runtime_overrides,
            )
        else:
            tune_res = evaluate_phase2_stateful_once(
                trial_cfg=dict(best_full_cfg or dict(best.params)),
                symbols=eval_symbols,
                horizon=horizon,
                train_start=train_start,
                train_end=train_end,
                oos_start=oos_start,
                oos_end=oos_end,
                portfolio_weights=portfolio_weights,
                aggregation_rule=aggregation_rule,
                objective_spec=objective_spec,
                runtime_overrides=runtime_overrides,
            )
        tune_oos_eval = {
            "objective": float(tune_res.objective),
            "portfolio_metrics": tune_res.portfolio_metrics,
            "per_symbol_metrics": tune_res.per_symbol_metrics,
        }
    except Exception:
        tune_oos_eval = None

    holdout_eval = None
    if holdout_start is not None and holdout_end is not None:
        try:
            if mode == "refinement":
                holdout_res = evaluate_phase2_stateful_once(
                    best_trial_json=best_trial_json,
                    symbols=eval_symbols,
                    horizon=horizon,
                    train_start=train_start,
                    train_end=train_end,
                    oos_start=holdout_start,
                    oos_end=holdout_end,
                    overrides=best_overrides or {},
                    portfolio_weights=portfolio_weights,
                    aggregation_rule=aggregation_rule,
                    objective_spec=objective_spec,
                    runtime_overrides=runtime_overrides,
                )
            else:
                holdout_res = evaluate_phase2_stateful_once(
                    trial_cfg=dict(best_full_cfg or dict(best.params)),
                    symbols=eval_symbols,
                    horizon=horizon,
                    train_start=train_start,
                    train_end=train_end,
                    oos_start=holdout_start,
                    oos_end=holdout_end,
                    portfolio_weights=portfolio_weights,
                    aggregation_rule=aggregation_rule,
                    objective_spec=objective_spec,
                    runtime_overrides=runtime_overrides,
                )
            holdout_eval = {
                "objective": float(holdout_res.objective),
                "portfolio_metrics": holdout_res.portfolio_metrics,
                "per_symbol_metrics": holdout_res.per_symbol_metrics,
            }
        except Exception:
            holdout_eval = None

    export_payload = {
        "type": "stage_b_stateful_phase2_winner",
        "phase2_search_mode": str(mode),
        "study_name": study.study_name,
        "storage": storage,
        "best_trial_number": int(best.number),
        "best_value": float(best.value) if best.value is not None else None,
        "symbols": [str(s).upper() for s in eval_symbols],
        "fixed_blocks": {
            "train_start": str(train_start),
            "train_end": str(train_end),
            "tune_oos_start": str(oos_start),
            "tune_oos_end": str(oos_end),
            "holdout_start": str(holdout_start) if holdout_start is not None else None,
            "holdout_end": str(holdout_end) if holdout_end is not None else None,
            "horizon": int(horizon),
        },
        "aggregation_rule": str(aggregation_rule),
        "portfolio_weights": {str(k).upper(): float(v) for k, v in (portfolio_weights or {}).items()},
        "objective": {
            "type": "sharpe_minus_drawdown",
            "max_drawdown_penalty": float(objective_spec.max_drawdown_penalty),
            "turnover_penalty": float(objective_spec.turnover_penalty),
        },
        "search_keys": study.user_attrs.get("phase2_search_keys"),
        "overrides": best_overrides,
        "winner_cfg": winner_cfg,
        "tune_oos_eval": tune_oos_eval,
        "holdout_eval": holdout_eval,
    }
    try:
        export_winner_path.write_text(json.dumps(export_payload, indent=2, sort_keys=True))
    except Exception:
        pass

    return {
        "study_name": study.study_name,
        "storage": storage,
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_params_count": int(len(getattr(best, "params", {}) or {})),
        "best_trial_has_phase2_trial_cfg": bool(isinstance((getattr(best, "user_attrs", {}) or {}).get("phase2_trial_cfg"), dict)),
        "best_trial_missing_params_in_cfg": list(
            sorted(
                set((getattr(best, "params", {}) or {}).keys())
                - set(((getattr(best, "user_attrs", {}) or {}).get("phase2_trial_cfg") or {}).keys())
            )
        )
        if isinstance((getattr(best, "user_attrs", {}) or {}).get("phase2_trial_cfg"), dict)
        else None,
        "n_trials": int(len(study.trials)),
        "phase2_search_mode": str(mode),
        "aggregation_rule": str(aggregation_rule),
        "symbols": [str(s).upper() for s in eval_symbols],
        "objective": {
            "type": "sharpe_minus_drawdown",
            "max_drawdown_penalty": float(objective_spec.max_drawdown_penalty),
            "turnover_penalty": float(objective_spec.turnover_penalty),
        },
        "winner_export_path": str(export_winner_path),
        "tune_oos_eval": tune_oos_eval,
        "holdout_eval": holdout_eval,
    }
