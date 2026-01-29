"""
Mamba-based Calibration Tracker for Phase-2 Stateful Learning.

This module provides calibration metrics computed directly from Mamba Gaussian
head outputs (μ, σ²) vs realized forward returns. This is the AUTHORITATIVE
calibration source for learning gates, learning decay, and overconfidence
penalties.

Architectural Principle:
- PRIMARY (70-80%): Mamba μ vs realized returns, σ² vs squared error, NLL stability
- SECONDARY (20-30%): Quantile forecasts as cross-check diagnostics only

The Mamba model produces (μ, σ²) predictions during Phase-2 OOS walk-forward.
These predictions mature after H trading days. Once matured, we can compare:
1. μ prediction vs realized return (directional accuracy, correlation)
2. σ² vs realized squared error (uncertainty calibration)
3. Gaussian NLL on matured predictions (probability calibration)

Copyright 2024-2025. All rights reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

# Rolling window for calibration metrics (trading days).
MAMBA_CALIB_WINDOW: int = 63

# Minimum observations to compute calibration scores.
MAMBA_CALIB_MIN_OBS: int = 20

# Blend weights: Mamba primary, quantile secondary.
MAMBA_PRIMARY_WEIGHT: float = 0.75  # 75% Mamba-based calibration
QUANTILE_SECONDARY_WEIGHT: float = 0.25  # 25% quantile diagnostics

# Clipping for numerical stability.
SIGMA_MIN: float = 1e-6
SIGMA_MAX: float = 10.0
NLL_CLIP_MIN: float = -10.0
NLL_CLIP_MAX: float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MambaCalibrationSnapshot:
    """Point-in-time calibration metrics from Mamba (μ, σ²) predictions."""
    
    # Primary Mamba calibration metrics (70-80% weight)
    mu_correlation: float = 0.0  # Corr(μ, realized return)
    mu_directional_accuracy: float = 0.5  # P(sign(μ) == sign(r))
    sigma_calibration: float = 1.0  # Ratio: mean(σ²) / mean(squared_error)
    sigma_coverage_1std: float = 0.683  # Fraction within ±1σ (should be ~68.3%)
    sigma_coverage_2std: float = 0.954  # Fraction within ±2σ (should be ~95.4%)
    nll_mean: float = 0.0  # Mean Gaussian NLL on matured predictions
    nll_std: float = 0.0  # Std of Gaussian NLL (stability)
    
    # Tail calibration metrics (for fat-tailed distribution heads)
    sigma_coverage_3std: float = 0.997  # Fraction within ±3σ (should be ~99.7%)
    tail_excess_left: float = 0.0  # Excess left-tail observations beyond -3σ
    tail_excess_right: float = 0.0  # Excess right-tail observations beyond +3σ
    tail_ratio: float = 1.0  # Ratio of observed/expected tail violations
    pit_uniformity_pvalue: float = 0.5  # PIT uniformity test p-value
    
    # Derived composite scores
    mu_score: float = 0.5  # Composite μ quality [0, 1]
    sigma_score: float = 0.5  # Composite σ² quality [0, 1]
    nll_score: float = 0.5  # Composite NLL quality [0, 1]
    tail_score: float = 0.5  # Tail calibration quality [0, 1]
    mamba_overall: float = 0.5  # Weighted composite [0, 1]
    
    # Secondary quantile diagnostics (20-30% weight, optional)
    quantile_score: float = 0.5  # From external quantile calibration (if available)
    
    # Final blended score (AUTHORITATIVE for learning gates)
    calibration_overall: float = 0.5  # Final blended score [0, 1]
    
    # Observation count for confidence.
    n_matured: int = 0
    is_reliable: bool = False  # True if n_matured >= MAMBA_CALIB_MIN_OBS


@dataclass
class MambaCalibrationTracker:
    """
    Rolling tracker of Mamba calibration metrics during Phase-2 OOS walk-forward.
    
    Maintains buffers of matured predictions and computes rolling calibration
    scores based on Mamba (μ, σ²) vs realized returns.
    
    Usage:
        tracker = MambaCalibrationTracker(horizon=21, window=63)
        
        # During OOS loop:
        for i, day in enumerate(oos_days):
            mu_vec, sigma_vec = model.predict(features)
            tracker.add_prediction(i, mu_vec, sigma_vec)
            
            # After horizon maturity, add realized returns:
            if i >= horizon:
                tracker.add_realized(i - horizon, realized_returns)
            
            # Get current calibration for learning decisions:
            calib = tracker.get_calibration(quantile_score=external_quantile_score)
    """
    
    horizon: int = 21  # Forward return horizon (trading days)
    window: int = MAMBA_CALIB_WINDOW  # Rolling window for metrics
    
    # Prediction buffers: indexed by OOS day index.
    _mu_history: Dict[int, np.ndarray] = field(default_factory=dict)
    _sigma_history: Dict[int, np.ndarray] = field(default_factory=dict)
    _realized_history: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Rolling buffers for efficient computation.
    _mu_buffer: List[np.ndarray] = field(default_factory=list)
    _sigma_buffer: List[np.ndarray] = field(default_factory=list)
    _realized_buffer: List[np.ndarray] = field(default_factory=list)
    _nll_buffer: List[float] = field(default_factory=list)
    
    # Tracking indices.
    _current_idx: int = 0
    _matured_count: int = 0
    
    def add_prediction(
        self,
        idx: int,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> None:
        """
        Record Mamba prediction for a given OOS day.
        
        Args:
            idx: OOS day index (0-based).
            mu: Mamba μ predictions [n_symbols].
            sigma: Mamba σ predictions (std dev) [n_symbols].
        """
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        sigma = np.clip(sigma, SIGMA_MIN, SIGMA_MAX)
        
        self._mu_history[idx] = mu.copy()
        self._sigma_history[idx] = sigma.copy()
        self._current_idx = max(self._current_idx, idx + 1)
    
    def add_realized(
        self,
        idx: int,
        realized: np.ndarray,
    ) -> None:
        """
        Record realized forward returns for a matured prediction.
        
        Args:
            idx: OOS day index when prediction was made.
            realized: Realized H-day forward returns [n_symbols].
        """
        if idx not in self._mu_history:
            logger.debug("[MambaCalib] No prediction at idx=%d to match realized", idx)
            return
        
        realized = np.asarray(realized, dtype=float).reshape(-1)
        self._realized_history[idx] = realized.copy()
        
        # Add to rolling buffers.
        mu = self._mu_history[idx]
        sigma = self._sigma_history[idx]
        
        # Compute NLL for this matured prediction.
        nll = self._compute_nll(mu, sigma, realized)
        
        self._mu_buffer.append(mu)
        self._sigma_buffer.append(sigma)
        self._realized_buffer.append(realized)
        self._nll_buffer.append(nll)
        self._matured_count += 1
        
        # Trim buffers to window size.
        if len(self._mu_buffer) > self.window:
            self._mu_buffer.pop(0)
            self._sigma_buffer.pop(0)
            self._realized_buffer.pop(0)
            self._nll_buffer.pop(0)
    
    def _compute_nll(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        realized: np.ndarray,
    ) -> float:
        """Compute mean Gaussian NLL over symbols."""
        sigma_sq = np.square(sigma) + 1e-12
        nll_per_sym = 0.5 * np.log(2 * np.pi * sigma_sq) + 0.5 * np.square(realized - mu) / sigma_sq
        # Mask invalid.
        mask = np.isfinite(nll_per_sym) & np.isfinite(mu) & np.isfinite(realized)
        if not np.any(mask):
            return 0.0
        nll_mean = float(np.clip(np.mean(nll_per_sym[mask]), NLL_CLIP_MIN, NLL_CLIP_MAX))
        return nll_mean
    
    def get_calibration(
        self,
        quantile_score: Optional[float] = None,
    ) -> MambaCalibrationSnapshot:
        """
        Compute current Mamba calibration metrics from rolling buffers.
        
        Args:
            quantile_score: Optional external quantile calibration score [0, 1]
                           for secondary diagnostic blending.
        
        Returns:
            MambaCalibrationSnapshot with all calibration metrics.
        """
        snap = MambaCalibrationSnapshot()
        snap.n_matured = len(self._mu_buffer)
        snap.is_reliable = snap.n_matured >= MAMBA_CALIB_MIN_OBS
        
        if snap.n_matured < 3:
            # Insufficient data: return neutral defaults.
            snap.calibration_overall = 0.5
            return snap
        
        # Stack buffers into arrays.
        mu_arr = np.concatenate(self._mu_buffer)  # [n_matured * n_symbols]
        sigma_arr = np.concatenate(self._sigma_buffer)
        realized_arr = np.concatenate(self._realized_buffer)
        
        # Mask invalid.
        valid = np.isfinite(mu_arr) & np.isfinite(sigma_arr) & np.isfinite(realized_arr)
        if np.sum(valid) < 10:
            snap.calibration_overall = 0.5
            return snap
        
        mu_v = mu_arr[valid]
        sigma_v = sigma_arr[valid]
        realized_v = realized_arr[valid]
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. μ calibration metrics
        # ─────────────────────────────────────────────────────────────────────
        
        # Correlation: Corr(μ, realized)
        if len(mu_v) >= 10:
            try:
                corr = float(np.corrcoef(mu_v, realized_v)[0, 1])
                if not np.isfinite(corr):
                    corr = 0.0
                snap.mu_correlation = corr
            except Exception:
                snap.mu_correlation = 0.0
        
        # Directional accuracy: P(sign(μ) == sign(realized))
        signs_agree = (mu_v > 0) == (realized_v > 0)
        snap.mu_directional_accuracy = float(np.mean(signs_agree))
        
        # μ composite score: blend correlation (rescaled 0-1) and directional accuracy.
        # Correlation in [-1, 1] → rescale [0, 1].
        corr_scaled = (snap.mu_correlation + 1.0) / 2.0
        snap.mu_score = 0.6 * corr_scaled + 0.4 * snap.mu_directional_accuracy
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. σ² calibration metrics
        # ─────────────────────────────────────────────────────────────────────
        
        squared_err = np.square(realized_v - mu_v)
        sigma_sq = np.square(sigma_v) + 1e-12
        
        # Calibration ratio: mean(σ²) / mean(squared_error)
        # Perfect calibration = 1.0. Under-confident > 1, over-confident < 1.
        mean_sigma_sq = float(np.mean(sigma_sq))
        mean_sq_err = float(np.mean(squared_err))
        if mean_sq_err > 1e-12:
            snap.sigma_calibration = mean_sigma_sq / mean_sq_err
        else:
            snap.sigma_calibration = 1.0
        
        # Coverage: fraction of observations within ±k*σ.
        z_scores = np.abs(realized_v - mu_v) / (sigma_v + 1e-12)
        snap.sigma_coverage_1std = float(np.mean(z_scores <= 1.0))  # Target: 0.683
        snap.sigma_coverage_2std = float(np.mean(z_scores <= 2.0))  # Target: 0.954
        snap.sigma_coverage_3std = float(np.mean(z_scores <= 3.0))  # Target: 0.997
        
        # ─────────────────────────────────────────────────────────────────────
        # 2b. Tail calibration metrics
        # ─────────────────────────────────────────────────────────────────────
        
        # Signed z-scores for tail analysis
        z_signed = (realized_v - mu_v) / (sigma_v + 1e-12)
        
        # Tail excesses (observations beyond ±3σ)
        expected_tail_rate = 0.003  # ~0.3% for ±3σ under Gaussian
        snap.tail_excess_left = float(np.mean(z_signed < -3.0))
        snap.tail_excess_right = float(np.mean(z_signed > 3.0))
        actual_tail_rate = snap.tail_excess_left + snap.tail_excess_right
        snap.tail_ratio = actual_tail_rate / expected_tail_rate if expected_tail_rate > 0 else 1.0
        
        # PIT (Probability Integral Transform) uniformity test
        # Under correct calibration, Φ(z) should be uniform on [0, 1]
        try:
            from scipy import stats
            pit_values = stats.norm.cdf(z_signed)
            # Kolmogorov-Smirnov test against uniform distribution
            ks_stat, ks_pvalue = stats.kstest(pit_values, 'uniform')
            snap.pit_uniformity_pvalue = float(ks_pvalue)
        except Exception:
            snap.pit_uniformity_pvalue = 0.5  # Default if scipy unavailable
        
        # Tail calibration score:
        # - Tail ratio near 1.0 is good (not too many or too few tail events)
        # - PIT uniformity high p-value is good
        tail_ratio_score = 1.0 - min(abs(np.log(snap.tail_ratio + 1e-6)), 2.0) / 2.0
        pit_score = min(1.0, snap.pit_uniformity_pvalue * 5.0)  # Scale up, cap at 1.0
        snap.tail_score = 0.6 * tail_ratio_score + 0.4 * pit_score
        
        # σ² composite score:
        # - Calibration ratio near 1.0 is good.
        # - Coverage near theoretical values is good.
        calib_ratio_score = 1.0 - min(abs(np.log(snap.sigma_calibration + 1e-12)), 2.0) / 2.0
        cov_1std_err = abs(snap.sigma_coverage_1std - 0.683) / 0.683
        cov_2std_err = abs(snap.sigma_coverage_2std - 0.954) / 0.954
        coverage_score = 1.0 - min((cov_1std_err + cov_2std_err) / 2.0, 1.0)
        snap.sigma_score = 0.5 * calib_ratio_score + 0.5 * coverage_score
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. NLL stability metrics
        # ─────────────────────────────────────────────────────────────────────
        
        nll_arr = np.asarray(self._nll_buffer, dtype=float)
        nll_arr = nll_arr[np.isfinite(nll_arr)]
        if len(nll_arr) >= 3:
            snap.nll_mean = float(np.mean(nll_arr))
            snap.nll_std = float(np.std(nll_arr))
            
            # NLL score: lower mean NLL and lower std are better.
            # Transform NLL to [0, 1] score (heuristic: NLL ∈ [-5, 5] typical).
            nll_normalized = (snap.nll_mean + 5.0) / 10.0  # Maps [-5, 5] → [0, 1]
            nll_normalized = 1.0 - np.clip(nll_normalized, 0.0, 1.0)  # Flip so lower NLL = higher score
            
            # Stability penalty: high NLL std is bad.
            stability = 1.0 - min(snap.nll_std / 2.0, 1.0)
            
            snap.nll_score = 0.7 * nll_normalized + 0.3 * stability
        else:
            snap.nll_score = 0.5
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. Composite Mamba calibration (PRIMARY: 70-80%)
        # ─────────────────────────────────────────────────────────────────────
        
        # Weight breakdown:
        # - μ score: 35% (directional quality of mean predictions)
        # - σ² score: 30% (uncertainty calibration quality)
        # - NLL score: 20% (overall probability calibration)
        # - Tail score: 15% (tail/fat-tail calibration)
        snap.mamba_overall = (
            0.35 * snap.mu_score +
            0.30 * snap.sigma_score +
            0.20 * snap.nll_score +
            0.15 * snap.tail_score
        )
        snap.mamba_overall = float(np.clip(snap.mamba_overall, 0.0, 1.0))
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. Blend with secondary quantile diagnostics (20-30%)
        # ─────────────────────────────────────────────────────────────────────
        
        if quantile_score is not None and np.isfinite(quantile_score):
            snap.quantile_score = float(np.clip(quantile_score, 0.0, 1.0))
            snap.calibration_overall = (
                MAMBA_PRIMARY_WEIGHT * snap.mamba_overall +
                QUANTILE_SECONDARY_WEIGHT * snap.quantile_score
            )
        else:
            # No quantile score available: use pure Mamba calibration.
            snap.calibration_overall = snap.mamba_overall
        
        snap.calibration_overall = float(np.clip(snap.calibration_overall, 0.0, 1.0))
        
        return snap
    
    def reset(self) -> None:
        """Clear all buffers and reset tracker state."""
        self._mu_history.clear()
        self._sigma_history.clear()
        self._realized_history.clear()
        self._mu_buffer.clear()
        self._sigma_buffer.clear()
        self._realized_buffer.clear()
        self._nll_buffer.clear()
        self._current_idx = 0
        self._matured_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export tracker state for diagnostics/logging."""
        return {
            "horizon": self.horizon,
            "window": self.window,
            "current_idx": self._current_idx,
            "matured_count": self._matured_count,
            "buffer_size": len(self._mu_buffer),
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize tracker state for persistence.
        
        Returns:
            Dict containing all state needed to restore the tracker.
        """
        return {
            "horizon": self.horizon,
            "window": self.window,
            "current_idx": self._current_idx,
            "matured_count": self._matured_count,
            # Convert dict keys to strings for JSON compatibility
            "mu_history": {str(k): v.tolist() for k, v in self._mu_history.items()},
            "sigma_history": {str(k): v.tolist() for k, v in self._sigma_history.items()},
            "realized_history": {str(k): v.tolist() for k, v in self._realized_history.items()},
            # Rolling buffers
            "mu_buffer": [v.tolist() for v in self._mu_buffer],
            "sigma_buffer": [v.tolist() for v in self._sigma_buffer],
            "realized_buffer": [v.tolist() for v in self._realized_buffer],
            "nll_buffer": list(self._nll_buffer),
        }
    
    @classmethod
    def deserialize(cls, state: Dict[str, Any]) -> "MambaCalibrationTracker":
        """
        Restore tracker from serialized state.
        
        Args:
            state: Serialized state dict from serialize().
        
        Returns:
            Restored MambaCalibrationTracker.
        """
        tracker = cls(
            horizon=int(state.get("horizon", 21)),
            window=int(state.get("window", MAMBA_CALIB_WINDOW)),
        )
        tracker._current_idx = int(state.get("current_idx", 0))
        tracker._matured_count = int(state.get("matured_count", 0))
        
        # Restore history dicts
        mu_hist = state.get("mu_history", {})
        tracker._mu_history = {int(k): np.asarray(v, dtype=float) for k, v in mu_hist.items()}
        
        sigma_hist = state.get("sigma_history", {})
        tracker._sigma_history = {int(k): np.asarray(v, dtype=float) for k, v in sigma_hist.items()}
        
        realized_hist = state.get("realized_history", {})
        tracker._realized_history = {int(k): np.asarray(v, dtype=float) for k, v in realized_hist.items()}
        
        # Restore rolling buffers
        tracker._mu_buffer = [np.asarray(v, dtype=float) for v in state.get("mu_buffer", [])]
        tracker._sigma_buffer = [np.asarray(v, dtype=float) for v in state.get("sigma_buffer", [])]
        tracker._realized_buffer = [np.asarray(v, dtype=float) for v in state.get("realized_buffer", [])]
        tracker._nll_buffer = list(state.get("nll_buffer", []))
        
        return tracker


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_mamba_calibration_tracker(
    horizon: int,
    window: int = MAMBA_CALIB_WINDOW,
) -> MambaCalibrationTracker:
    """
    Factory for creating a MambaCalibrationTracker.
    
    Args:
        horizon: Forward return horizon (trading days).
        window: Rolling window for calibration metrics.
    
    Returns:
        Initialized MambaCalibrationTracker.
    """
    return MambaCalibrationTracker(horizon=horizon, window=window)


def compute_mamba_calibration_score(
    mu: np.ndarray,
    sigma: np.ndarray,
    realized: np.ndarray,
    quantile_score: Optional[float] = None,
) -> float:
    """
    Compute a single-shot Mamba calibration score.
    
    Convenience function for computing calibration from arrays without
    maintaining a rolling tracker.
    
    Args:
        mu: Mamba μ predictions [n_obs].
        sigma: Mamba σ predictions [n_obs].
        realized: Realized returns [n_obs].
        quantile_score: Optional secondary quantile score.
    
    Returns:
        Calibration score [0, 1].
    """
    tracker = MambaCalibrationTracker(horizon=0, window=len(mu))
    
    # Add as single batch.
    for i in range(len(mu)):
        tracker.add_prediction(i, np.array([mu[i]]), np.array([sigma[i]]))
        tracker.add_realized(i, np.array([realized[i]]))
    
    snap = tracker.get_calibration(quantile_score=quantile_score)
    return snap.calibration_overall
