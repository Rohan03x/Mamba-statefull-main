"""Linear Alpha Combiner for Phase-2 signal blending.

This module provides a Ridge-regularized linear model that outputs z_lin directly
(z-space), replacing static quantile blending. The model trains online using
matured forward returns with calibration-gated freeze logic.

Architecture:
- LinearFeatureBuilder: Assembles per-symbol feature matrix from portfolio context
- RidgeModel: Numpy-only Ridge regression with feature standardization
- LinearCombinerState: Manages observation buffering, maturity gating, and online updates

Usage in phase2_stateful.py:
    if linear_state is not None:
        X_day = linear_state.feature_builder.build_day(...)
        linear_state.observe_day(i, X_day, sigma_exec)
        if linear_state.is_ready():
            z_lin = linear_state.predict_z(X_day)
            z = (1 - w_L) * z + w_L * z_lin
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature names for interpretability and persistence
FEATURE_NAMES_PER_SYMBOL = [
    "z_mamba",           # Post-overlay z-score from Mamba
    "quantile_z",        # Quantile forecast z-score
    "risk_scale",        # Risk overlay scale [0,1]
    "regime_multiplier", # Regime overlay [0,1]
    "split_stress",      # Split penalty factor
    "ret_1d",            # 1-day past return
    "ret_5d",            # 5-day cumulative past return
    "ret_21d",           # 21-day cumulative past return
    "rv_21d",            # 21-day realized volatility
    "inv_sigma_exec",    # Inverse execution sigma (precision proxy)
]

FEATURE_NAMES_GLOBAL = [
    "day_calib_score",   # Mamba calibration score
    "online_trust_score",# Online trust metric
    "cboe_panic",        # VIX panic premium
    "cboe_slope",        # VIX term structure slope
    "cboe_vrp",          # Vol risk premium z-score
    "drawdown",          # Current portfolio drawdown
    "realized_vol",      # Portfolio realized vol
    "corr_hhi",          # Correlation HHI (concentration)
    "turnover_prev",     # Previous day turnover
    "cost_prev",         # Previous day cost
]

ALL_FEATURE_NAMES = FEATURE_NAMES_PER_SYMBOL + FEATURE_NAMES_GLOBAL
N_FEATURES = len(ALL_FEATURE_NAMES)


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert to float with fallback."""
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return float(default)
        return float(v)  # type: ignore
    except Exception:
        return float(default)


def _compute_rolling_returns(
    returns_df: Optional[pd.DataFrame],
    i: int,
    syms: Sequence[str],
    window: int,
) -> np.ndarray:
    """Compute cumulative return over past `window` days ending at day i (exclusive of i).
    
    Returns shape (n_assets,).
    """
    n_assets = len(syms)
    result = np.zeros(n_assets, dtype=float)
    
    if returns_df is None or i < 1:
        return result
    
    start_idx = max(0, i - window)
    end_idx = i  # Exclusive of current day
    
    if start_idx >= end_idx:
        return result
    
    try:
        # Get returns for the window
        ret_slice = returns_df.iloc[start_idx:end_idx]
        for j, sym in enumerate(syms):
            if sym in ret_slice.columns:
                r = ret_slice[sym].to_numpy(dtype=float)
                # Cumulative return: prod(1+r) - 1
                r = np.where(np.isfinite(r), r, 0.0)
                cum_ret = float(np.prod(1.0 + r) - 1.0)
                result[j] = cum_ret if np.isfinite(cum_ret) else 0.0
    except Exception:
        pass
    
    return result


def _compute_realized_vol(
    returns_df: Optional[pd.DataFrame],
    i: int,
    syms: Sequence[str],
    window: int = 21,
) -> np.ndarray:
    """Compute realized volatility over past `window` days ending at day i.
    
    Returns shape (n_assets,), annualized.
    """
    n_assets = len(syms)
    result = np.full(n_assets, 0.15, dtype=float)  # Default 15% vol
    
    if returns_df is None or i < window:
        return result
    
    start_idx = max(0, i - window)
    end_idx = i
    
    if end_idx - start_idx < 5:  # Need at least 5 days
        return result
    
    try:
        ret_slice = returns_df.iloc[start_idx:end_idx]
        for j, sym in enumerate(syms):
            if sym in ret_slice.columns:
                r = ret_slice[sym].to_numpy(dtype=float)
                r = r[np.isfinite(r)]
                if len(r) >= 5:
                    vol = float(np.std(r, ddof=1) * np.sqrt(252))
                    result[j] = vol if np.isfinite(vol) and vol > 0 else 0.15
    except Exception:
        pass
    
    return result


@dataclass
class LinearFeatureBuilder:
    """Builds feature matrix for linear alpha combiner.
    
    Features are split into per-symbol (varying across assets) and global
    (same for all assets, broadcast). All features use only past information
    to avoid lookahead bias.
    """
    
    symbols: List[str] = field(default_factory=list)
    n_features: int = N_FEATURES
    feature_names: List[str] = field(default_factory=lambda: list(ALL_FEATURE_NAMES))
    
    def build_day(
        self,
        *,
        i: int,
        syms: Sequence[str],
        z_mamba: np.ndarray,
        sigma_exec: np.ndarray,
        day_ctx: Any = None,  # RoleAwareDayContext
        day_calib_score: float = 1.0,
        day_online_trust: float = 1.0,
        day_cboe_panic: float = 0.0,
        day_cboe_slope: float = 0.0,
        day_cboe_vrp: float = 0.0,
        equity: Optional[np.ndarray] = None,
        turnover: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        returns_df: Optional[pd.DataFrame] = None,
        corr_hhi: float = 0.0,
    ) -> np.ndarray:
        """Build feature matrix for day i.
        
        Args:
            i: Day index in OOS loop
            syms: Symbol list
            z_mamba: Post-overlay z-scores, shape (n_assets,)
            sigma_exec: Execution sigma, shape (n_assets,)
            day_ctx: RoleAwareDayContext for the day (optional)
            day_calib_score: Mamba calibration score [0, 1]
            day_online_trust: Online trust score [0, 1]
            day_cboe_panic: VIX panic premium
            day_cboe_slope: VIX term structure slope
            day_cboe_vrp: Vol risk premium z-score
            equity: Equity curve array (to compute drawdown)
            turnover: Turnover array (for previous day)
            costs: Costs array (for previous day)
            returns_df: Returns DataFrame for momentum features
            corr_hhi: Correlation HHI from covariance matrix (Gap C fix)
        
        Returns:
            X_day: Feature matrix, shape (n_assets, n_features)
        """
        n_assets = len(syms)
        X = np.zeros((n_assets, self.n_features), dtype=float)
        
        # === Per-symbol features ===
        
        # z_mamba (idx 0)
        z_mamba = np.asarray(z_mamba, dtype=float)
        X[:, 0] = np.where(np.isfinite(z_mamba), z_mamba, 0.0)
        
        # quantile_z (idx 1)
        if day_ctx is not None and hasattr(day_ctx, 'quantile_z'):
            qz = np.asarray(day_ctx.quantile_z, dtype=float)
            X[:, 1] = np.where(np.isfinite(qz), qz, 0.0)
        
        # risk_scale (idx 2)
        if day_ctx is not None and hasattr(day_ctx, 'risk_scale'):
            rs = np.asarray(day_ctx.risk_scale, dtype=float)
            X[:, 2] = np.where(np.isfinite(rs), rs, 1.0)
        else:
            X[:, 2] = 1.0
        
        # regime_multiplier (idx 3)
        if day_ctx is not None and hasattr(day_ctx, 'regime_multiplier'):
            rm = np.asarray(day_ctx.regime_multiplier, dtype=float)
            X[:, 3] = np.where(np.isfinite(rm), rm, 1.0)
        else:
            X[:, 3] = 1.0
        
        # split_stress (idx 4)
        if day_ctx is not None and hasattr(day_ctx, 'split_stress'):
            ss = np.asarray(day_ctx.split_stress, dtype=float)
            X[:, 4] = np.where(np.isfinite(ss), ss, 0.0)
        
        # ret_1d, ret_5d, ret_21d (idx 5, 6, 7)
        if returns_df is not None:
            X[:, 5] = _compute_rolling_returns(returns_df, i, syms, window=1)
            X[:, 6] = _compute_rolling_returns(returns_df, i, syms, window=5)
            X[:, 7] = _compute_rolling_returns(returns_df, i, syms, window=21)
        
        # rv_21d (idx 8)
        if returns_df is not None:
            X[:, 8] = _compute_realized_vol(returns_df, i, syms, window=21)
        else:
            X[:, 8] = 0.15  # Default vol
        
        # inv_sigma_exec (idx 9) - precision proxy for robust position sizing
        sigma_exec = np.asarray(sigma_exec, dtype=float)
        inv_sigma = 1.0 / np.clip(sigma_exec, 1e-8, None)
        X[:, 9] = np.where(np.isfinite(inv_sigma), inv_sigma, 1.0)
        
        # === Global features (broadcast to all assets) ===
        
        # day_calib_score (idx 10)
        X[:, 10] = _safe_float(day_calib_score, 1.0)
        
        # online_trust_score (idx 11)
        X[:, 11] = _safe_float(day_online_trust, 1.0)
        
        # cboe_panic (idx 12)
        X[:, 12] = _safe_float(day_cboe_panic, 0.0)
        
        # cboe_slope (idx 13)
        X[:, 13] = _safe_float(day_cboe_slope, 0.0)
        
        # cboe_vrp (idx 14)
        X[:, 14] = _safe_float(day_cboe_vrp, 0.0)
        
        # drawdown (idx 15)
        if equity is not None and i > 0:
            eq_so_far = equity[:i]
            eq_so_far = eq_so_far[np.isfinite(eq_so_far)]
            if len(eq_so_far) > 0:
                peak = float(np.max(eq_so_far))
                current = float(eq_so_far[-1]) if len(eq_so_far) > 0 else 1.0
                dd = (peak - current) / peak if peak > 0 else 0.0
                X[:, 15] = _safe_float(dd, 0.0)
        
        # realized_vol (portfolio-level) (idx 16)
        if returns_df is not None and i >= 21:
            try:
                port_ret = returns_df.iloc[max(0, i-21):i].mean(axis=1).to_numpy(dtype=float)
                port_vol = float(np.std(port_ret[np.isfinite(port_ret)], ddof=1) * np.sqrt(252))
                X[:, 16] = port_vol if np.isfinite(port_vol) else 0.15
            except Exception:
                X[:, 16] = 0.15
        else:
            X[:, 16] = 0.15
        
        # corr_hhi (idx 17) - correlation concentration from eigenvalue HHI
        # FIX Gap C: Use actual value passed from phase2 instead of placeholder
        X[:, 17] = _safe_float(corr_hhi, 0.0)
        
        # turnover_prev (idx 18)
        if turnover is not None and i > 0:
            X[:, 18] = _safe_float(turnover[i - 1], 0.0)
        
        # cost_prev (idx 19)
        if costs is not None and i > 0:
            X[:, 19] = _safe_float(costs[i - 1], 0.0)
        
        return X


@dataclass
class RidgeModel:
    """Numpy-only Ridge regression with feature standardization.
    
    Solves: β = (XᵀX + λI)⁻¹Xᵀy
    
    Features are standardized (mean=0, std=1) before fitting.
    """
    
    n_features: int = N_FEATURES
    lambda_: float = 10.0
    
    # Fitted parameters
    beta_: Optional[np.ndarray] = None  # Coefficients, shape (n_features,)
    intercept_: float = 0.0
    
    # Standardization stats
    mean_: Optional[np.ndarray] = None  # shape (n_features,)
    std_: Optional[np.ndarray] = None   # shape (n_features,)
    
    # Training diagnostics
    r_squared_: float = 0.0
    n_samples_: int = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        """Fit Ridge regression.
        
        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target, shape (n_samples,) - should be z-space (r / sigma_exec)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        # Filter valid samples
        valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) < 10:
            logger.warning("[linear.ridge] Not enough valid samples (%d < 10)", len(y))
            return self
        
        self.n_samples_ = len(y)
        
        # Compute and store standardization stats
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=1)
        self.std_ = np.where(self.std_ > 1e-8, self.std_, 1.0)  # Avoid div by zero
        
        # Standardize features
        X_std = (X - self.mean_) / self.std_
        
        # Center target
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        
        # Ridge regression: β = (XᵀX + λI)⁻¹Xᵀy
        n_features = X_std.shape[1]
        XtX = X_std.T @ X_std
        XtY = X_std.T @ y_centered
        
        # Add regularization
        reg_matrix = self.lambda_ * np.eye(n_features)
        
        try:
            self.beta_ = np.linalg.solve(XtX + reg_matrix, XtY)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            logger.warning("[linear.ridge] Singular matrix, using pseudo-inverse")
            self.beta_ = np.linalg.lstsq(XtX + reg_matrix, XtY, rcond=None)[0]
        
        self.intercept_ = y_mean
        
        # Compute R² for diagnostics
        y_pred = X_std @ self.beta_ + self.intercept_
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        self.r_squared_ = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict z_lin from features.
        
        Args:
            X: Feature matrix, shape (n_assets, n_features)
        
        Returns:
            z_lin: Predicted z-scores, shape (n_assets,)
        """
        if self.beta_ is None or self.mean_ is None or self.std_ is None:
            # Not fitted yet - return zeros
            return np.zeros(X.shape[0], dtype=float)
        
        X = np.asarray(X, dtype=float)
        
        # Handle NaN/Inf in input
        X = np.where(np.isfinite(X), X, 0.0)
        
        # Standardize using stored stats
        X_std = (X - self.mean_) / self.std_
        
        # Predict
        z_lin = X_std @ self.beta_ + self.intercept_
        
        # Clip extreme predictions
        z_lin = np.clip(z_lin, -10.0, 10.0)
        
        return z_lin
    
    def get_coefficients_report(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get coefficient report for logging."""
        if self.beta_ is None:
            return {"status": "not_fitted"}
        
        if feature_names is None:
            feature_names = list(ALL_FEATURE_NAMES)
        
        # Top coefficients by absolute value
        abs_coefs = np.abs(self.beta_)
        top_indices = np.argsort(abs_coefs)[::-1][:5]
        
        top_coefs: List[Dict[str, Any]] = [
            {
                "feature": feature_names[idx] if idx < len(feature_names) else f"feat_{idx}",
                "coef": float(self.beta_[idx]),
                "abs_coef": float(abs_coefs[idx]),
            }
            for idx in top_indices
        ]
        
        # Key coefficients
        z_mamba_coef = float(self.beta_[0]) if len(self.beta_) > 0 else 0.0
        quantile_z_coef = float(self.beta_[1]) if len(self.beta_) > 1 else 0.0
        
        return {
            "status": "fitted",
            "n_samples": self.n_samples_,
            "r_squared": float(self.r_squared_),
            "intercept": float(self.intercept_),
            "z_mamba_coef": z_mamba_coef,
            "quantile_z_coef": quantile_z_coef,
            "z_mamba_sign": "positive" if z_mamba_coef >= 0 else "negative",
            "quantile_z_sign": "positive" if quantile_z_coef >= 0 else "negative",
            "top_coefficients": top_coefs,
        }


@dataclass
class LinearCombinerState:
    """Manages linear alpha combiner state, buffering, and online updates.
    
    Key responsibilities:
    1. Buffer daily observations (X, sigma_exec) for maturity processing
    2. Compute z-space targets when predictions mature (y = r / sigma_exec)
    3. Maintain rolling training window
    4. Refit model at specified intervals with freeze logic
    """
    
    # Config
    horizon: int = 21
    update_interval: int = 21
    ridge_lambda: float = 10.0
    window: int = 126
    max_window: int = 252
    min_samples: int = 63
    
    # Components
    feature_builder: LinearFeatureBuilder = field(default_factory=LinearFeatureBuilder)
    model: RidgeModel = field(default_factory=RidgeModel)
    
    # Observation buffers (keyed by day index)
    _X_buffer: Dict[int, np.ndarray] = field(default_factory=dict)
    _sigma_exec_buffer: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Training data (rolling window)
    # FIX Gap D: Add day_idx tracking for proper rolling window and persistence
    _X_train: List[np.ndarray] = field(default_factory=list)
    _y_train: List[np.ndarray] = field(default_factory=list)
    _day_idx_train: List[int] = field(default_factory=list)
    
    # State tracking
    _last_fit_day: int = -1
    _n_updates: int = 0
    _prev_beta: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.model.lambda_ = self.ridge_lambda
    
    def observe_day(self, i: int, X_day: np.ndarray, sigma_exec: np.ndarray) -> None:
        """Store observation for day i.
        
        Called during the main loop BEFORE prediction. Stores features
        and sigma_exec for later maturity processing.
        
        Args:
            i: Day index
            X_day: Feature matrix, shape (n_assets, n_features)
            sigma_exec: Execution sigma, shape (n_assets,)
        """
        self._X_buffer[i] = X_day.copy()
        self._sigma_exec_buffer[i] = np.asarray(sigma_exec, dtype=float).copy()
        
        # Garbage collect old buffers (beyond max_window + horizon)
        cutoff = i - self.max_window - self.horizon - 10
        for old_idx in list(self._X_buffer.keys()):
            if old_idx < cutoff:
                del self._X_buffer[old_idx]
                if old_idx in self._sigma_exec_buffer:
                    del self._sigma_exec_buffer[old_idx]
    
    def update_if_matured(
        self,
        i: int,
        fwd_ret_mat: np.ndarray,
        freeze: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process matured observations and optionally refit.
        
        Called after main loop prediction. For day i, the prediction
        made at day (i - horizon) has now matured.
        
        Args:
            i: Current day index
            fwd_ret_mat: Forward returns matrix, shape (n_days, n_assets)
            freeze: If True, buffer but don't refit (calibration < 0.55)
        
        Returns:
            Refit report dict if refit occurred, else None
        """
        matured_idx = i - self.horizon
        
        if matured_idx < 0:
            return None
        
        # Get buffered observation from matured day
        if matured_idx not in self._X_buffer:
            return None
        
        X_matured = self._X_buffer[matured_idx]
        sigma_matured = self._sigma_exec_buffer.get(matured_idx)
        
        if sigma_matured is None:
            return None
        
        # Get realized returns for matured prediction
        if matured_idx >= fwd_ret_mat.shape[0]:
            return None
        
        realized_ret = fwd_ret_mat[matured_idx, :]
        
        # Compute z-space target: y = r / (sigma_exec + eps)
        eps = 1e-8
        y_matured = realized_ret / (sigma_matured + eps)
        
        # Filter valid samples
        valid_mask = np.isfinite(y_matured) & np.all(np.isfinite(X_matured), axis=1)
        
        if np.sum(valid_mask) > 0:
            # Add to training buffer (per-sample for rolling window)
            # FIX Gap D: Store day_idx for proper window enforcement
            self._X_train.append(X_matured[valid_mask])
            self._y_train.append(y_matured[valid_mask])
            self._day_idx_train.append(matured_idx)
            
            # Enforce rolling window by day count (not sample count)
            # Use 'window' for training, 'max_window' for storage retention
            while len(self._day_idx_train) > 0:
                oldest_day = self._day_idx_train[0]
                if (matured_idx - oldest_day) > self.max_window:
                    self._X_train.pop(0)
                    self._y_train.pop(0)
                    self._day_idx_train.pop(0)
                else:
                    break
        
        # Check if refit is due
        should_refit = (
            (i - self._last_fit_day) >= self.update_interval
            and not freeze
        )
        
        if should_refit:
            return self._do_refit(i)
        
        return None
    
    def _do_refit(self, i: int) -> Dict[str, Any]:
        """Perform model refit on accumulated training data.
        
        FIX Gap D: Use 'window' parameter to select training subset from buffer.
        Buffer stores up to 'max_window' days, but training uses recent 'window' days.
        
        Returns:
            Refit report with R², drift, coefficients, etc.
        """
        if len(self._X_train) == 0:
            return {"status": "no_data"}
        
        # Apply training window: use only recent 'window' days
        # (buffer may hold more for warm restarts)
        matured_idx = i - self.horizon
        train_cutoff_day = matured_idx - self.window
        
        X_chunks = []
        y_chunks = []
        for j in range(len(self._day_idx_train)):
            if self._day_idx_train[j] >= train_cutoff_day:
                X_chunks.append(self._X_train[j])
                y_chunks.append(self._y_train[j])
        
        if len(X_chunks) == 0:
            return {"status": "no_data", "n_samples": 0}
        
        # Concatenate training data
        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks)
        
        n_samples = len(y_all)
        
        if n_samples < self.min_samples:
            return {
                "status": "insufficient_samples",
                "n_samples": n_samples,
                "min_required": self.min_samples,
            }
        
        # Store previous beta for drift calculation
        self._prev_beta = self.model.beta_.copy() if self.model.beta_ is not None else None
        
        # Fit model
        self.model.fit(X_all, y_all)
        
        self._last_fit_day = i
        self._n_updates += 1
        
        # Compute drift
        drift = 0.0
        if self._prev_beta is not None and self.model.beta_ is not None:
            drift = float(np.linalg.norm(self.model.beta_ - self._prev_beta))
        
        # Build report
        report = self.model.get_coefficients_report()
        report.update({
            "day_idx": i,
            "n_updates": self._n_updates,
            "drift": drift,
        })
        
        logger.info(
            "[linear.refit] day=%d n_samples=%d R²=%.3f drift=%.4f z_mamba_coef=%.3f",
            i, n_samples, report.get("r_squared", 0.0), drift,
            report.get("z_mamba_coef", 0.0),
        )
        
        return report
    
    def predict_z(self, X_day: np.ndarray) -> np.ndarray:
        """Predict z_lin for current day.
        
        Args:
            X_day: Feature matrix, shape (n_assets, n_features)
        
        Returns:
            z_lin: Predicted z-scores, shape (n_assets,)
        """
        return self.model.predict(X_day)
    
    def is_ready(self) -> bool:
        """Check if model is trained and ready for blending.
        
        Returns True only after first successful fit with min_samples.
        """
        return self.model.beta_ is not None
    
    def get_daily_diagnostics(
        self,
        z_lin: np.ndarray,
        z_mamba: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute daily diagnostic metrics.
        
        Args:
            z_lin: Linear model predictions, shape (n_assets,)
            z_mamba: Original Mamba z-scores, shape (n_assets,)
        
        Returns:
            Dict with diagnostic metrics
        """
        z_lin = np.asarray(z_lin, dtype=float)
        z_mamba = np.asarray(z_mamba, dtype=float)
        
        valid_mask = np.isfinite(z_lin) & np.isfinite(z_mamba)
        
        if np.sum(valid_mask) < 2:
            return {"status": "insufficient_valid"}
        
        z_lin_v = z_lin[valid_mask]
        z_mamba_v = z_mamba[valid_mask]
        
        # Correlation
        corr = float(np.corrcoef(z_lin_v, z_mamba_v)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        
        # Sign disagreement
        sign_disagree = float(np.mean(np.sign(z_lin_v) != np.sign(z_mamba_v)))
        
        # Scale impact: mean(|z_lin| / |z_mamba|)
        z_mamba_abs = np.abs(z_mamba_v)
        z_lin_abs = np.abs(z_lin_v)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(z_mamba_abs > 1e-6, z_lin_abs / z_mamba_abs, 1.0)
            scale_impact = float(np.clip(np.mean(ratio), 0.0, 10.0))
        
        return {
            "corr_z_lin_z_mamba": corr,
            "sign_disagreement_frac": sign_disagree,
            "scale_impact": scale_impact,
            "is_ready": self.is_ready(),
            "n_updates": self._n_updates,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence.
        
        FIX Gap D: Persist training buffer (X, y, day_idx) for warm restarts.
        """
        # Concatenate buffer samples for compact storage
        buffer_X = np.vstack(self._X_train) if len(self._X_train) > 0 else None
        buffer_y = np.concatenate(self._y_train) if len(self._y_train) > 0 else None
        buffer_days = np.array(self._day_idx_train, dtype=int) if len(self._day_idx_train) > 0 else None
        
        # Track sample counts per day for reconstruction
        sample_counts = [len(y) for y in self._y_train] if len(self._y_train) > 0 else None
        
        return {
            "horizon": self.horizon,
            "update_interval": self.update_interval,
            "ridge_lambda": self.ridge_lambda,
            "window": self.window,
            "max_window": self.max_window,
            "min_samples": self.min_samples,
            "n_updates": self._n_updates,
            "last_fit_day": self._last_fit_day,
            "beta": self.model.beta_.tolist() if self.model.beta_ is not None else None,
            "intercept": float(self.model.intercept_),
            "mean": self.model.mean_.tolist() if self.model.mean_ is not None else None,
            "std": self.model.std_.tolist() if self.model.std_ is not None else None,
            "r_squared": float(self.model.r_squared_),
            "n_samples": int(self.model.n_samples_),
            "feature_names": list(ALL_FEATURE_NAMES),
            # Buffer persistence (Gap D fix)
            "buffer_X": buffer_X.tolist() if buffer_X is not None else None,
            "buffer_y": buffer_y.tolist() if buffer_y is not None else None,
            "buffer_days": buffer_days.tolist() if buffer_days is not None else None,
            "buffer_sample_counts": sample_counts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "LinearCombinerState":
        """Deserialize state from persistence.
        
        FIX Gap D: Restore training buffer for warm restarts.
        """
        state = cls(
            horizon=int(data.get("horizon", 21)),
            update_interval=int(data.get("update_interval", 21)),
            ridge_lambda=float(data.get("ridge_lambda", 10.0)),
            window=int(data.get("window", 126)),
            max_window=int(data.get("max_window", 252)),
            min_samples=int(data.get("min_samples", 63)),
        )
        
        state._n_updates = int(data.get("n_updates", 0))
        state._last_fit_day = int(data.get("last_fit_day", -1))
        
        # Restore model
        if data.get("beta") is not None:
            state.model.beta_ = np.array(data["beta"], dtype=float)
            state.model.intercept_ = float(data.get("intercept", 0.0))
            state.model.mean_ = np.array(data["mean"], dtype=float) if data.get("mean") else None
            state.model.std_ = np.array(data["std"], dtype=float) if data.get("std") else None
            state.model.r_squared_ = float(data.get("r_squared", 0.0))
            state.model.n_samples_ = int(data.get("n_samples", 0))
        
        # Restore buffer (Gap D fix)
        if data.get("buffer_X") is not None and data.get("buffer_y") is not None:
            buffer_X_flat = np.array(data["buffer_X"], dtype=float)
            buffer_y_flat = np.array(data["buffer_y"], dtype=float)
            buffer_days = data.get("buffer_days", [])
            sample_counts = data.get("buffer_sample_counts", [])
            
            if len(buffer_days) == len(sample_counts) and len(buffer_days) > 0:
                # Reconstruct per-day chunks
                X_chunks = []
                y_chunks = []
                
                X_offset = 0
                y_offset = 0
                for count in sample_counts:
                    if count > 0:
                        X_chunk = buffer_X_flat[X_offset:X_offset + count, :]
                        y_chunk = buffer_y_flat[y_offset:y_offset + count]
                        X_chunks.append(X_chunk)
                        y_chunks.append(y_chunk)
                        X_offset += count
                        y_offset += count
                
                state._X_train = X_chunks
                state._y_train = y_chunks
                state._day_idx_train = list(buffer_days)
        
        return state


def create_linear_combiner(
    *,
    horizon: int = 21,
    update_interval: int = 21,
    ridge_lambda: float = 10.0,
    window: int = 126,
    max_window: int = 252,
    min_samples: int = 63,
    symbols: Sequence[str] = (),
) -> LinearCombinerState:
    """Factory function to create LinearCombinerState with config.
    
    Args:
        horizon: Prediction horizon (days)
        update_interval: Days between refits
        ridge_lambda: Ridge regularization strength
        window: Rolling training window size
        max_window: Maximum samples to retain
        min_samples: Minimum samples before first fit
        symbols: Symbol list (for feature builder)
    
    Returns:
        Configured LinearCombinerState
    """
    feature_builder = LinearFeatureBuilder(symbols=list(symbols))
    
    model = RidgeModel(
        n_features=N_FEATURES,
        lambda_=ridge_lambda,
    )
    
    return LinearCombinerState(
        horizon=horizon,
        update_interval=update_interval,
        ridge_lambda=ridge_lambda,
        window=window,
        max_window=max_window,
        min_samples=min_samples,
        feature_builder=feature_builder,
        model=model,
    )
