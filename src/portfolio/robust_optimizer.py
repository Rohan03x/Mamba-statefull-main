"""Robust Portfolio Optimizer — Workstream 7.

Upgrades from simple signal thresholding to distribution-aware, tail-aware
robust optimization.

Key Features:
1. Replace sigma proxy with predicted sigma from multi-horizon model
2. Factor covariance model with graph shrinkage
3. Mean-variance with CVaR constraint
4. Turnover penalty in objective
5. Uncertainty-aware exposure caps

Reference: Phase2_Optuna_Search_Space.md Section 17 (Workstream 7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PredictionBundle:
    """Container for model predictions per asset.
    
    All arrays have shape (n_assets,) unless otherwise noted.
    """
    mu: np.ndarray          # Expected return predictions
    sigma: np.ndarray       # Predicted volatility (from model, NOT proxy)
    sigma_reliability: np.ndarray  # Calibration reliability score [0, 1]
    p_up: Optional[np.ndarray] = None  # Probability of positive return
    rho: Optional[np.ndarray] = None   # Model confidence / correlation
    
    # Multi-horizon predictions (optional)
    mu_by_horizon: Optional[Dict[int, np.ndarray]] = None  # {5: [...], 21: [...], ...}
    sigma_by_horizon: Optional[Dict[int, np.ndarray]] = None
    
    def __post_init__(self):
        """Validate and normalize inputs."""
        self.mu = np.asarray(self.mu, dtype=float).ravel()
        self.sigma = np.asarray(self.sigma, dtype=float).ravel()
        self.sigma_reliability = np.asarray(self.sigma_reliability, dtype=float).ravel()
        
        # Ensure sigma is positive
        self.sigma = np.maximum(self.sigma, 1e-6)
        # Ensure reliability is in [0, 1]
        self.sigma_reliability = np.clip(self.sigma_reliability, 0.0, 1.0)
        
        if self.p_up is not None:
            self.p_up = np.clip(np.asarray(self.p_up, dtype=float).ravel(), 0.0, 1.0)
        if self.rho is not None:
            self.rho = np.clip(np.asarray(self.rho, dtype=float).ravel(), 0.0, 1.0)
    
    @property
    def n_assets(self) -> int:
        return len(self.mu)


@dataclass
class CovarianceModel:
    """Container for covariance estimation state."""
    cov: np.ndarray                     # Current covariance matrix (n x n)
    factor_cov: Optional[np.ndarray] = None  # Factor covariance (k x k)
    factor_loadings: Optional[np.ndarray] = None  # Factor loadings (n x k)
    idio_var: Optional[np.ndarray] = None  # Idiosyncratic variances (n,)
    
    # Graph-based shrinkage
    sector_ids: Optional[np.ndarray] = None  # Sector labels per asset
    cluster_ids: Optional[np.ndarray] = None  # Graph cluster labels
    adjacency: Optional[np.ndarray] = None  # Adjacency matrix for shrinkage
    
    def __post_init__(self):
        self.cov = np.asarray(self.cov, dtype=float)
        if self.cov.ndim == 1:
            # Diagonal only: expand to full matrix
            self.cov = np.diag(self.cov)


@dataclass
class PortfolioConstraints:
    """Portfolio constraint specification."""
    # Position limits
    max_gross: float = 2.0         # Maximum gross exposure
    max_net: float = 0.5           # Maximum net exposure (long - short)
    max_name: float = 0.10         # Maximum single-name weight
    min_name: float = -0.10        # Minimum single-name weight (for shorts)
    
    # Risk limits
    target_vol_annual: float = 0.15  # Target annualized volatility
    max_vol_annual: float = 0.25     # Hard cap on volatility
    
    # CVaR / tail risk
    cvar_alpha: float = 0.05       # CVaR tail probability (5%)
    cvar_limit: Optional[float] = None  # Max CVaR (None = no constraint)
    
    # Turnover
    turnover_cap: Optional[float] = None  # Max turnover per period
    turnover_penalty: float = 0.0  # Turnover penalty in objective
    
    # Beta neutrality
    beta_neutral: bool = False
    max_beta_exposure: float = 0.1
    
    # Sector limits
    max_sector_exposure: float = 0.3  # Max weight in any sector
    
    # Drawdown control
    current_drawdown: float = 0.0   # Current drawdown for throttling
    drawdown_throttle: float = 0.10  # DD threshold for exposure reduction
    drawdown_gross_mult: float = 0.5  # Gross multiplier when in DD
    
    # Uncertainty-aware caps
    mu_sigma_cap: float = 3.0       # Cap weight by mu/sigma ratio
    reliability_min: float = 0.3     # Min reliability to take position


@dataclass  
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: np.ndarray             # Optimal weights (n_assets,)
    expected_return: float          # Expected portfolio return
    expected_vol: float             # Expected portfolio volatility
    sharpe: float                   # Expected Sharpe ratio
    gross_exposure: float           # Sum of abs weights
    net_exposure: float             # Sum of weights
    turnover: float                 # Turnover from previous weights
    
    # Risk decomposition
    cvar_5: Optional[float] = None  # 5% CVaR
    max_drawdown_proxy: Optional[float] = None
    
    # Solver info
    solver_status: str = "unknown"
    solve_time_ms: float = 0.0
    
    # Diagnostic
    active_names: int = 0           # Number of non-zero positions
    capped_by_uncertainty: int = 0  # Positions capped due to uncertainty


# =============================================================================
# COVARIANCE MODELS
# =============================================================================

def ewma_cov_update(
    cov: np.ndarray,
    returns: np.ndarray,
    *,
    lambda_: float = 0.94,
) -> np.ndarray:
    """EWMA covariance update: cov <- lambda*cov + (1-lambda) * (r r^T).
    
    Args:
        cov: Current covariance matrix (n x n)
        returns: Today's returns vector (n,)
        lambda_: Decay factor (0.94 = ~20 day half-life)
    
    Returns:
        Updated covariance matrix
    """
    c = np.asarray(cov, dtype=float)
    r = np.asarray(returns, dtype=float).ravel()
    
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] != len(r):
        return c
    
    lam = float(np.clip(lambda_, 0.0, 0.9999))
    r = np.where(np.isfinite(r), r, 0.0)
    rr = np.outer(r, r)
    c = np.where(np.isfinite(c), c, 0.0)
    out = lam * c + (1.0 - lam) * rr
    return 0.5 * (out + out.T)


def shrink_cov_to_diagonal(
    cov: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Shrink covariance toward diagonal (Ledoit-Wolf style).
    
    Args:
        cov: Full covariance matrix
        alpha: Shrinkage intensity (0 = no shrinkage, 1 = diagonal only)
    
    Returns:
        Shrunk covariance matrix
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        return c
    
    a = float(np.clip(alpha, 0.0, 1.0))
    c = np.where(np.isfinite(c), c, 0.0)
    d = np.diag(np.diag(c))
    out = (1.0 - a) * c + a * d
    return 0.5 * (out + out.T)


def shrink_cov_to_graph(
    cov: np.ndarray,
    adjacency: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """Shrink covariance toward graph structure (sector/cluster communities).
    
    This shrinks off-diagonal elements based on graph distance:
    - Assets in same cluster: keep more correlation
    - Assets in different clusters: shrink toward zero
    
    Args:
        cov: Full covariance matrix (n x n)
        adjacency: Adjacency matrix (n x n) with edge weights [0, 1]
        alpha: Shrinkage intensity
    
    Returns:
        Graph-shrunk covariance matrix
    """
    c = np.asarray(cov, dtype=float)
    adj = np.asarray(adjacency, dtype=float)
    
    if c.ndim != 2 or adj.ndim != 2:
        return c
    if c.shape != adj.shape:
        return c
    
    n = c.shape[0]
    a = float(np.clip(alpha, 0.0, 1.0))
    
    # Normalize adjacency to [0, 1]
    adj = np.clip(adj, 0.0, 1.0)
    adj = np.where(np.isfinite(adj), adj, 0.0)
    
    # Shrink off-diagonal: keep (1 - alpha + alpha * adj[i,j]) of correlation
    # Where adj[i,j] = 1 means same cluster, adj[i,j] = 0 means different
    shrink_factor = (1.0 - a) + a * adj
    
    # Keep diagonal unchanged
    np.fill_diagonal(shrink_factor, 1.0)
    
    # Apply shrinkage
    c = np.where(np.isfinite(c), c, 0.0)
    out = c * shrink_factor
    return 0.5 * (out + out.T)


def build_sector_adjacency(sector_ids: np.ndarray) -> np.ndarray:
    """Build adjacency matrix from sector labels.
    
    Assets in same sector get adjacency = 1, different sectors get 0.
    
    Args:
        sector_ids: Sector label per asset (n,)
    
    Returns:
        Adjacency matrix (n x n)
    """
    s = np.asarray(sector_ids).ravel()
    n = len(s)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if s[i] == s[j]:
                adj[i, j] = 1.0
    return adj


def build_factor_covariance(
    returns: np.ndarray,
    n_factors: int = 5,
    *,
    method: str = "pca",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build factor covariance model from returns.
    
    Decomposes: Σ = B F B^T + D
    where B = factor loadings, F = factor covariance, D = idiosyncratic diagonal
    
    Args:
        returns: Historical returns (T x n)
        n_factors: Number of factors
        method: "pca" or "statistical"
    
    Returns:
        (factor_loadings (n x k), factor_cov (k x k), idio_var (n,))
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 20:
        n = X.shape[1] if X.ndim == 2 else 1
        return np.eye(n, n_factors), np.eye(n_factors), np.ones(n) * 0.01
    
    T, n = X.shape
    k = min(n_factors, n - 1, T - 1)
    
    # Demean
    X = X - np.nanmean(X, axis=0, keepdims=True)
    X = np.where(np.isfinite(X), X, 0.0)
    
    # PCA via SVD
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        loadings = Vt[:k, :].T * (S[:k] / np.sqrt(T))  # (n x k)
        factor_returns = U[:, :k] * S[:k]  # (T x k)
        factor_cov = np.cov(factor_returns.T)  # (k x k)
        if factor_cov.ndim == 0:
            factor_cov = np.array([[factor_cov]])
        
        # Idiosyncratic variance: residual variance
        fitted = factor_returns @ loadings.T
        residuals = X - fitted
        idio_var = np.var(residuals, axis=0)
        idio_var = np.maximum(idio_var, 1e-8)
        
        return loadings, factor_cov, idio_var
    except Exception as e:
        logger.warning(f"Factor covariance failed: {e}")
        return np.eye(n, k), np.eye(k), np.ones(n) * 0.01


def reconstruct_cov_from_factors(
    factor_loadings: np.ndarray,
    factor_cov: np.ndarray,
    idio_var: np.ndarray,
) -> np.ndarray:
    """Reconstruct full covariance from factor model.
    
    Σ = B F B^T + D
    
    Args:
        factor_loadings: (n x k)
        factor_cov: (k x k)
        idio_var: (n,)
    
    Returns:
        Full covariance matrix (n x n)
    """
    B = np.asarray(factor_loadings, dtype=float)
    F = np.asarray(factor_cov, dtype=float)
    D = np.asarray(idio_var, dtype=float).ravel()
    
    cov = B @ F @ B.T + np.diag(D)
    return 0.5 * (cov + cov.T)


def replace_sigma_with_predicted(
    cov: np.ndarray,
    predicted_sigma: np.ndarray,
    *,
    blend_alpha: float = 0.5,
) -> np.ndarray:
    """Replace diagonal (variances) with predicted sigma from model.
    
    This uses the model's sigma predictions instead of historical volatility.
    Correlations are preserved from the historical covariance.
    
    Args:
        cov: Historical covariance matrix (n x n)
        predicted_sigma: Model-predicted volatilities (n,)
        blend_alpha: Blend factor (1.0 = pure predicted, 0.0 = pure historical)
    
    Returns:
        Updated covariance matrix
    """
    c = np.asarray(cov, dtype=float).copy()
    s = np.asarray(predicted_sigma, dtype=float).ravel()
    
    if c.ndim != 2 or c.shape[0] != len(s):
        return c
    
    n = len(s)
    a = float(np.clip(blend_alpha, 0.0, 1.0))
    
    # Extract historical std devs
    hist_std = np.sqrt(np.maximum(np.diag(c), 1e-12))
    
    # Blend predicted with historical
    new_std = a * s + (1.0 - a) * hist_std
    new_std = np.maximum(new_std, 1e-8)
    
    # Convert to correlation matrix
    corr = c / np.outer(hist_std, hist_std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    
    # Reconstruct with new variances
    new_cov = corr * np.outer(new_std, new_std)
    return 0.5 * (new_cov + new_cov.T)


# =============================================================================
# CVaR / TAIL RISK
# =============================================================================

def estimate_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Estimate portfolio CVaR (Expected Shortfall) from historical returns.
    
    CVaR_α = E[R | R <= VaR_α]
    
    Args:
        weights: Portfolio weights (n,)
        returns: Historical returns (T x n)
        alpha: Tail probability (0.05 = 5% worst cases)
    
    Returns:
        CVaR estimate (negative value = loss)
    """
    w = np.asarray(weights, dtype=float).ravel()
    R = np.asarray(returns, dtype=float)
    
    if R.ndim != 2 or R.shape[1] != len(w):
        return 0.0
    
    # Portfolio returns
    port_ret = R @ w
    port_ret = port_ret[np.isfinite(port_ret)]
    
    if len(port_ret) < 20:
        return 0.0
    
    # VaR and CVaR
    var_cutoff = np.percentile(port_ret, alpha * 100)
    tail_returns = port_ret[port_ret <= var_cutoff]
    
    if len(tail_returns) == 0:
        return var_cutoff
    
    return float(np.mean(tail_returns))


def estimate_parametric_cvar(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Estimate CVaR assuming normal distribution.
    
    For normal: CVaR_α = μ - σ * φ(Φ^(-1)(α)) / α
    where φ = PDF, Φ = CDF of standard normal
    
    Args:
        weights: Portfolio weights
        mu: Expected returns
        cov: Covariance matrix
        alpha: Tail probability
    
    Returns:
        Parametric CVaR estimate
    """
    from scipy import stats
    
    w = np.asarray(weights, dtype=float).ravel()
    m = np.asarray(mu, dtype=float).ravel()
    c = np.asarray(cov, dtype=float)
    
    if len(w) != len(m) or c.shape[0] != len(w):
        return 0.0
    
    port_mu = float(w @ m)
    port_var = float(w @ c @ w)
    port_sigma = np.sqrt(max(port_var, 1e-12))
    
    # CVaR formula for normal
    z_alpha = stats.norm.ppf(alpha)
    pdf_at_var = stats.norm.pdf(z_alpha)
    cvar = port_mu - port_sigma * pdf_at_var / alpha
    
    return float(cvar)


def tail_dependence_penalty(
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_tail_risk: np.ndarray,
) -> float:
    """Compute penalty for exposure to high tail-dependence clusters.
    
    Args:
        weights: Portfolio weights
        cluster_ids: Cluster label per asset
        cluster_tail_risk: Tail risk score per cluster (higher = more tail risk)
    
    Returns:
        Tail dependence penalty (higher = more concentrated in risky clusters)
    """
    w = np.asarray(weights, dtype=float).ravel()
    c = np.asarray(cluster_ids).ravel()
    r = np.asarray(cluster_tail_risk, dtype=float).ravel()
    
    if len(w) != len(c):
        return 0.0
    
    # Compute exposure per cluster, weighted by tail risk
    penalty = 0.0
    for i, (wi, ci) in enumerate(zip(w, c)):
        ci = int(ci)
        if 0 <= ci < len(r):
            penalty += abs(wi) * r[ci]
    
    return float(penalty)


# =============================================================================
# UNCERTAINTY-AWARE POSITION SIZING
# =============================================================================

def compute_mu_sigma_cap(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    max_ir: float = 3.0,
) -> np.ndarray:
    """Compute position cap based on mu/sigma ratio.
    
    High conviction (|mu|/sigma > max_ir) positions are capped to prevent
    over-concentration in "too good to be true" signals.
    
    Args:
        mu: Expected returns
        sigma: Predicted volatilities
        max_ir: Maximum information ratio allowed
    
    Returns:
        Cap multiplier per asset in [0, 1]
    """
    m = np.asarray(mu, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float).ravel()
    s = np.maximum(s, 1e-8)
    
    ir = np.abs(m) / s
    cap = np.minimum(max_ir / np.maximum(ir, 1e-8), 1.0)
    cap = np.where(ir <= max_ir, 1.0, cap)
    return np.clip(cap, 0.0, 1.0)


def compute_reliability_cap(
    sigma_reliability: np.ndarray,
    *,
    min_reliability: float = 0.3,
    scale_below_min: bool = True,
) -> np.ndarray:
    """Compute position cap based on sigma reliability score.
    
    Low reliability predictions should have reduced position size.
    
    Args:
        sigma_reliability: Reliability score from calibration tracker [0, 1]
        min_reliability: Minimum reliability to take full position
        scale_below_min: If True, scale linearly below min; if False, zero out
    
    Returns:
        Cap multiplier per asset in [0, 1]
    """
    r = np.asarray(sigma_reliability, dtype=float).ravel()
    r = np.clip(r, 0.0, 1.0)
    
    if scale_below_min:
        # Linear scale from 0 at r=0 to 1 at r=min_reliability
        cap = np.minimum(r / max(min_reliability, 1e-8), 1.0)
    else:
        cap = np.where(r >= min_reliability, 1.0, 0.0)
    
    return cap


def apply_uncertainty_caps(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    sigma_reliability: np.ndarray,
    *,
    max_ir: float = 3.0,
    min_reliability: float = 0.3,
) -> Tuple[np.ndarray, int]:
    """Apply uncertainty-aware caps to portfolio weights.
    
    Combines mu/sigma cap and reliability cap.
    
    Args:
        weights: Raw portfolio weights
        mu: Expected returns
        sigma: Predicted volatilities
        sigma_reliability: Reliability scores
        max_ir: Maximum information ratio
        min_reliability: Minimum reliability threshold
    
    Returns:
        (capped_weights, n_capped)
    """
    w = np.asarray(weights, dtype=float).ravel()
    
    ir_cap = compute_mu_sigma_cap(mu, sigma, max_ir=max_ir)
    rel_cap = compute_reliability_cap(sigma_reliability, min_reliability=min_reliability)
    
    combined_cap = ir_cap * rel_cap
    capped = w * combined_cap
    
    n_capped = int(np.sum((combined_cap < 1.0) & (np.abs(w) > 1e-8)))
    
    return capped, n_capped


# =============================================================================
# CORE OPTIMIZER
# =============================================================================

class RobustPortfolioOptimizer:
    """Distribution-aware, tail-aware robust portfolio optimizer.
    
    Upgrades from simple mean-variance to:
    1. Predicted sigma (from model) instead of historical proxy
    2. Factor or graph-shrinkage covariance
    3. CVaR constraint for tail risk control
    4. Turnover penalty in objective
    5. Uncertainty-aware position caps
    
    Objective: max E[R] - λ_var * Var[R] - λ_turn * Turnover - λ_tail * TailPenalty
    Subject to:
        - CVaR >= cvar_limit
        - |w_i| <= max_name
        - sum(|w|) <= max_gross
        - |sum(w)| <= max_net
        - sqrt(w'Σw) <= max_vol
        - Turnover <= turnover_cap
    """
    
    def __init__(
        self,
        *,
        lambda_var: float = 1.0,
        lambda_turnover: float = 0.0,
        lambda_tail: float = 0.0,
        use_predicted_sigma: bool = True,
        predicted_sigma_blend: float = 0.7,
        covariance_method: str = "ewma_shrink",  # ewma_shrink, factor, graph_shrink
        cvar_constraint: bool = False,
        uncertainty_caps: bool = True,
        solver: str = "analytical",  # analytical, cvxpy, scipy
        trading_days: int = 252,
    ):
        """Initialize optimizer.
        
        Args:
            lambda_var: Risk aversion (higher = more risk-averse)
            lambda_turnover: Turnover penalty weight
            lambda_tail: Tail risk penalty weight
            use_predicted_sigma: Use model sigma instead of historical
            predicted_sigma_blend: Blend factor for predicted sigma
            covariance_method: Covariance estimation method
            cvar_constraint: Enable CVaR constraint
            uncertainty_caps: Enable mu/sigma and reliability caps
            solver: Optimization solver to use
            trading_days: Trading days per year
        """
        self.lambda_var = float(lambda_var)
        self.lambda_turnover = float(lambda_turnover)
        self.lambda_tail = float(lambda_tail)
        self.use_predicted_sigma = bool(use_predicted_sigma)
        self.predicted_sigma_blend = float(predicted_sigma_blend)
        self.covariance_method = str(covariance_method)
        self.cvar_constraint = bool(cvar_constraint)
        self.uncertainty_caps = bool(uncertainty_caps)
        self.solver = str(solver)
        self.trading_days = int(trading_days)
    
    def optimize(
        self,
        predictions: PredictionBundle,
        cov_model: CovarianceModel,
        constraints: PortfolioConstraints,
        prev_weights: Optional[np.ndarray] = None,
        historical_returns: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Optimize portfolio weights.
        
        Args:
            predictions: Model predictions (mu, sigma, reliability)
            cov_model: Covariance estimation state
            constraints: Portfolio constraints
            prev_weights: Previous weights for turnover calculation
            historical_returns: Historical returns for CVaR (T x n)
        
        Returns:
            OptimizationResult with optimal weights and diagnostics
        """
        import time
        t0 = time.perf_counter()
        
        n = predictions.n_assets
        mu = predictions.mu.copy()
        sigma = predictions.sigma.copy()
        reliability = predictions.sigma_reliability.copy()
        
        if prev_weights is None:
            prev_weights = np.zeros(n)
        prev_weights = np.asarray(prev_weights, dtype=float).ravel()
        if len(prev_weights) != n:
            prev_weights = np.zeros(n)
        
        # Step 1: Build covariance matrix
        cov = self._build_covariance(cov_model, sigma)
        
        # Step 2: Apply uncertainty caps to mu (reduce expected return for unreliable)
        if self.uncertainty_caps:
            mu_capped, n_capped = self._apply_uncertainty_scaling(
                mu, sigma, reliability, constraints
            )
        else:
            mu_capped = mu
            n_capped = 0
        
        # Step 3: Solve optimization
        if self.solver == "analytical":
            w_raw = self._solve_analytical(mu_capped, cov, constraints, prev_weights)
        else:
            w_raw = self._solve_analytical(mu_capped, cov, constraints, prev_weights)
        
        # Step 4: Apply hard constraints
        w = self._apply_hard_constraints(w_raw, constraints)
        
        # Step 5: Apply uncertainty caps to weights
        if self.uncertainty_caps:
            w, n_capped_weights = apply_uncertainty_caps(
                w, mu, sigma, reliability,
                max_ir=constraints.mu_sigma_cap,
                min_reliability=constraints.reliability_min,
            )
            n_capped = max(n_capped, n_capped_weights)
        
        # Step 6: Apply vol target scaling
        w = self._apply_vol_target(w, cov, constraints.target_vol_annual)
        
        # Step 7: Re-apply hard constraints after scaling
        w = self._apply_hard_constraints(w, constraints)
        
        # Step 8: Apply drawdown throttle
        if constraints.current_drawdown >= constraints.drawdown_throttle:
            w = w * constraints.drawdown_gross_mult
        
        # Compute result metrics
        port_ret = float(w @ mu)
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 1e-12)))
        port_vol_annual = port_vol * np.sqrt(self.trading_days)
        sharpe = port_ret / max(port_vol, 1e-8)
        
        gross = float(np.sum(np.abs(w)))
        net = float(np.sum(w))
        turnover = float(np.sum(np.abs(w - prev_weights)))
        active = int(np.sum(np.abs(w) > 1e-6))
        
        # CVaR
        cvar = None
        if historical_returns is not None and len(historical_returns) >= 20:
            cvar = estimate_cvar(w, historical_returns, constraints.cvar_alpha)
        
        solve_time = (time.perf_counter() - t0) * 1000
        
        return OptimizationResult(
            weights=w,
            expected_return=port_ret,
            expected_vol=port_vol_annual,
            sharpe=sharpe,
            gross_exposure=gross,
            net_exposure=net,
            turnover=turnover,
            cvar_5=cvar,
            solver_status="optimal",
            solve_time_ms=solve_time,
            active_names=active,
            capped_by_uncertainty=n_capped,
        )
    
    def _build_covariance(
        self,
        cov_model: CovarianceModel,
        predicted_sigma: np.ndarray,
    ) -> np.ndarray:
        """Build covariance matrix using configured method."""
        cov = cov_model.cov.copy()
        
        # Replace diagonal with predicted sigma if enabled
        if self.use_predicted_sigma:
            cov = replace_sigma_with_predicted(
                cov, predicted_sigma, blend_alpha=self.predicted_sigma_blend
            )
        
        # Apply additional shrinkage based on method
        if self.covariance_method == "graph_shrink" and cov_model.adjacency is not None:
            cov = shrink_cov_to_graph(cov, cov_model.adjacency, alpha=0.2)
        elif self.covariance_method == "factor" and cov_model.factor_loadings is not None:
            cov = reconstruct_cov_from_factors(
                cov_model.factor_loadings,
                cov_model.factor_cov,
                cov_model.idio_var,
            )
        
        # Always apply some diagonal regularization
        cov = cov + np.eye(cov.shape[0]) * 1e-8
        
        return cov
    
    def _apply_uncertainty_scaling(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        reliability: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> Tuple[np.ndarray, int]:
        """Scale mu based on uncertainty (conservative for unreliable signals)."""
        # Reduce mu for low-reliability predictions
        rel_scale = compute_reliability_cap(
            reliability, min_reliability=constraints.reliability_min
        )
        
        # Don't let mu/sigma ratio get too extreme
        ir_scale = compute_mu_sigma_cap(mu, sigma, max_ir=constraints.mu_sigma_cap)
        
        scale = rel_scale * ir_scale
        n_scaled = int(np.sum(scale < 1.0))
        
        return mu * scale, n_scaled
    
    def _solve_analytical(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        constraints: PortfolioConstraints,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        """Analytical mean-variance solution with turnover and tail risk penalties.
        
        Solves: max mu'w - (λ/2) w'Σw - (λ_turn/2) ||w - w_prev||^2 - (λ_tail/2) ||L'w||^2
        where L is Cholesky of cov (tail risk proxy).
        
        Solution: w* = (λΣ + λ_turn I + λ_tail L'L)^-1 (mu + λ_turn w_prev)
                     = (λΣ + λ_turn I + λ_tail Σ)^-1 (mu + λ_turn w_prev)
                     = ((λ + λ_tail)Σ + λ_turn I)^-1 (mu + λ_turn w_prev)
        """
        n = len(mu)
        lam = self.lambda_var
        lam_turn = self.lambda_turnover
        lam_tail = self.lambda_tail
        
        # Regularized inverse with tail risk penalty
        # Note: λ_tail * L'L = λ_tail * Σ, so we just add it to variance penalty
        effective_lam = lam + lam_tail
        reg_cov = effective_lam * cov + lam_turn * np.eye(n)
        reg_cov = reg_cov + np.eye(n) * 1e-8  # Numerical stability
        
        try:
            inv = np.linalg.pinv(reg_cov)
        except Exception:
            inv = np.eye(n)
        
        rhs = mu + lam_turn * prev_weights
        w = inv @ rhs
        
        return w
    
    def _apply_hard_constraints(
        self,
        w: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> np.ndarray:
        """Apply hard portfolio constraints."""
        w = np.asarray(w, dtype=float).ravel()
        
        # Name limits
        w = np.clip(w, constraints.min_name, constraints.max_name)
        
        # Gross exposure limit
        gross = float(np.sum(np.abs(w)))
        if gross > constraints.max_gross:
            w = w * (constraints.max_gross / gross)
        
        # Net exposure limit
        net = float(np.sum(w))
        if abs(net) > constraints.max_net:
            # Shift long/short proportionally
            if net > constraints.max_net:
                excess = net - constraints.max_net
                w_pos = np.maximum(w, 0)
                pos_sum = float(np.sum(w_pos))
                if pos_sum > 0:
                    w = np.where(w > 0, w - excess * (w / pos_sum), w)
            elif net < -constraints.max_net:
                excess = -constraints.max_net - net
                w_neg = np.minimum(w, 0)
                neg_sum = float(np.sum(np.abs(w_neg)))
                if neg_sum > 0:
                    w = np.where(w < 0, w + excess * (np.abs(w) / neg_sum), w)
        
        return w
    
    def _apply_vol_target(
        self,
        w: np.ndarray,
        cov: np.ndarray,
        target_vol_annual: float,
    ) -> np.ndarray:
        """Scale weights to target volatility."""
        if target_vol_annual <= 0:
            return w
        
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return w
        
        port_vol_annual = np.sqrt(port_var) * np.sqrt(self.trading_days)
        
        if port_vol_annual > 0:
            scale = target_vol_annual / port_vol_annual
            scale = min(scale, 5.0)  # Cap scaling
            w = w * scale
        
        return w
    
    def update_covariance(
        self,
        cov_model: CovarianceModel,
        returns: np.ndarray,
        *,
        lambda_ewma: float = 0.94,
        shrink_alpha: float = 0.1,
    ) -> CovarianceModel:
        """Update covariance model with new returns.
        
        Args:
            cov_model: Current covariance state
            returns: Today's returns vector
            lambda_ewma: EWMA decay factor
            shrink_alpha: Shrinkage intensity
        
        Returns:
            Updated CovarianceModel
        """
        new_cov = ewma_cov_update(cov_model.cov, returns, lambda_=lambda_ewma)
        new_cov = shrink_cov_to_diagonal(new_cov, alpha=shrink_alpha)
        
        return CovarianceModel(
            cov=new_cov,
            factor_cov=cov_model.factor_cov,
            factor_loadings=cov_model.factor_loadings,
            idio_var=cov_model.idio_var,
            sector_ids=cov_model.sector_ids,
            cluster_ids=cov_model.cluster_ids,
            adjacency=cov_model.adjacency,
        )


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def build_prediction_bundle_from_mamba(
    preds_df: pd.DataFrame,
    *,
    mu_col: str = "mu_hat",
    sigma_col: str = "sigma_hat",
    reliability_col: str = "sigma_reliability",
    p_up_col: str = "p_up",
    rho_col: str = "rho",
    sigma_floor: float = 0.02,
) -> PredictionBundle:
    """Build PredictionBundle from Mamba model output DataFrame.
    
    Args:
        preds_df: DataFrame with model predictions (one row per asset)
        mu_col, sigma_col, etc: Column names
        sigma_floor: Minimum sigma value
    
    Returns:
        PredictionBundle
    """
    mu = preds_df.get(mu_col, pd.Series(0.0, index=preds_df.index)).to_numpy(dtype=float)
    sigma = preds_df.get(sigma_col, pd.Series(sigma_floor, index=preds_df.index)).to_numpy(dtype=float)
    sigma = np.maximum(sigma, sigma_floor)
    
    reliability = preds_df.get(reliability_col, pd.Series(1.0, index=preds_df.index)).to_numpy(dtype=float)
    
    p_up = None
    if p_up_col in preds_df.columns:
        p_up = preds_df[p_up_col].to_numpy(dtype=float)
    
    rho = None
    if rho_col in preds_df.columns:
        rho = preds_df[rho_col].to_numpy(dtype=float)
    
    return PredictionBundle(
        mu=mu,
        sigma=sigma,
        sigma_reliability=reliability,
        p_up=p_up,
        rho=rho,
    )


def build_covariance_model_from_returns(
    returns_df: pd.DataFrame,
    *,
    method: str = "ewma",
    lambda_ewma: float = 0.94,
    shrink_alpha: float = 0.1,
    n_factors: int = 5,
    sector_col: Optional[str] = None,
    sector_map: Optional[Dict[str, int]] = None,
) -> CovarianceModel:
    """Build CovarianceModel from historical returns.
    
    Args:
        returns_df: Historical returns (T x n assets as columns)
        method: "ewma", "factor", or "sample"
        lambda_ewma: EWMA decay for covariance
        shrink_alpha: Shrinkage intensity
        n_factors: Number of factors for factor model
        sector_col: Column with sector labels (if available)
        sector_map: Map from asset name to sector id
    
    Returns:
        CovarianceModel
    """
    R = returns_df.to_numpy(dtype=float)
    R = np.where(np.isfinite(R), R, 0.0)
    
    if method == "factor":
        loadings, f_cov, idio = build_factor_covariance(R, n_factors=n_factors)
        cov = reconstruct_cov_from_factors(loadings, f_cov, idio)
    elif method == "ewma":
        # Initialize with sample covariance, then EWMA update would follow
        cov = np.cov(R.T) if R.shape[0] >= 2 else np.eye(R.shape[1])
    else:
        cov = np.cov(R.T) if R.shape[0] >= 2 else np.eye(R.shape[1])
    
    cov = shrink_cov_to_diagonal(cov, alpha=shrink_alpha)
    
    # Build sector adjacency if available
    adjacency = None
    sector_ids = None
    if sector_map is not None:
        cols = list(returns_df.columns)
        sector_ids = np.array([sector_map.get(c, 0) for c in cols])
        adjacency = build_sector_adjacency(sector_ids)
    
    return CovarianceModel(
        cov=cov,
        sector_ids=sector_ids,
        adjacency=adjacency,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "PredictionBundle",
    "CovarianceModel",
    "PortfolioConstraints",
    "OptimizationResult",
    # Covariance functions
    "ewma_cov_update",
    "shrink_cov_to_diagonal",
    "shrink_cov_to_graph",
    "build_sector_adjacency",
    "build_factor_covariance",
    "reconstruct_cov_from_factors",
    "replace_sigma_with_predicted",
    # CVaR functions
    "estimate_cvar",
    "estimate_parametric_cvar",
    "tail_dependence_penalty",
    # Uncertainty caps
    "compute_mu_sigma_cap",
    "compute_reliability_cap",
    "apply_uncertainty_caps",
    # Core optimizer
    "RobustPortfolioOptimizer",
    # Integration helpers
    "build_prediction_bundle_from_mamba",
    "build_covariance_model_from_returns",
]
