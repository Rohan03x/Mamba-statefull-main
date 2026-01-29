"""Policy Controller v2 - Contextual Bandit for Trading Desk Control.

This module implements a contextual bandit that learns to select trading policies
based on market conditions and model reliability. It acts as a "meta-layer" that
keeps the strategy out of bad trades during calibration failures and regime breaks.

Key features:
- Expanded action space with risk-off controls, leverage scheduling, sector caps
- Risk-adjusted reward with turnover and drawdown penalties
- Exploration annealing for long-run stability
- Gradual risk-off ramp with hysteresis
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# STATE VECTOR DIMENSION (v2: 25 dims)
# =============================================================================
POLICY_STATE_DIM_V2 = 25


@dataclass(frozen=True)
class PolicyAction:
    """Policy knob set selected by the contextual bandit.
    
    v2 expansion includes:
    - Regime-aware thresholding parameters
    - Risk/exposure constraints
    - Quantile forecast blending
    - Dynamic risk controls (risk-off, leverage schedule, sector caps)
    - Robust optimizer integration (lambda multipliers)
    """
    name: str
    
    # === REGIME THRESHOLDING ===
    base_threshold: float
    regime_mult_bull: float
    regime_mult_bear: float
    regime_mult_crisis: float
    
    # === POSITION CONSTRAINTS ===
    target_vol: float
    turnover_cap: float
    max_gross: float
    max_net: float
    max_name: float
    weight_smoothing_alpha: float = 0.0
    vol_scaler: float = 1.0
    
    # === QUANTILE BLENDING ===
    # z = (1-w_q)*z_mamba + w_q*z_quantile
    quantile_blend_weight: float = 0.0  # 0 = pure Mamba, 1 = pure quantile
    
    # === LINEAR META-MODEL BLENDING ===
    # z = (1-w_L)*z + w_L*z_lin (separate from quantile blending)
    linear_blend_weight: float = 0.0  # 0 = no linear correction, 1 = full linear
    
    # === DYNAMIC RISK CONTROLS (v2 NEW) ===
    # Risk-off duration: sessions to stay flat/low after trigger
    risk_off_duration: int = 0
    # Leverage schedule: multiplier on max_gross (0.5 = half leverage)
    max_leverage_schedule: float = 1.0
    # Sector cap strength: 0 = no sector caps, 1 = strict caps
    sector_cap_strength: float = 1.0
    # Minimum mu_reliability to trade (below this, reduce exposure)
    confidence_floor: float = 0.3
    
    # === ROBUST OPTIMIZER INTEGRATION (v2 NEW) ===
    # Multiplier on λ_var (>1 = more risk averse)
    robust_lambda_var_mult: float = 1.0
    # Override predicted_sigma_blend (-1 = use config default)
    robust_sigma_blend_override: float = -1.0
    # Multiplier on CVaR limit (<1 = tighter constraint)
    cvar_strictness: float = 1.0


@dataclass
class RiskOffState:
    """Tracks risk-off ramp state for gradual re-entry.
    
    When triggered, the controller gradually ramps back exposure:
    - Day 0-1: max_leverage_schedule = 0.0 (flat)
    - Then: 0.25 → 0.50 → 0.75 → 1.00 over remaining cooldown
    
    Hysteresis: do not re-risk until drawdown_velocity < 0 AND vol_of_vol is declining.
    """
    is_active: bool = False
    trigger_session: int = -1
    duration: int = 5  # Total cooldown sessions
    current_session: int = 0
    
    # Hysteresis conditions (must improve before full re-risk)
    last_drawdown_velocity: float = 0.0
    last_vol_of_vol: float = 0.0
    
    def get_leverage_mult(self) -> float:
        """Get current leverage multiplier based on ramp schedule."""
        if not self.is_active:
            return 1.0
        
        # Sessions elapsed since trigger
        elapsed = self.current_session - self.trigger_session
        if elapsed < 0:
            return 1.0
        
        # Day 0-1: flat
        if elapsed <= 1:
            return 0.0
        
        # Ramp schedule: 0.25 → 0.50 → 0.75 → 1.00
        ramp_steps = max(1, self.duration - 2)
        step = min(elapsed - 2, ramp_steps)
        ramp_schedule = [0.25, 0.50, 0.75, 1.00]
        
        if step >= len(ramp_schedule):
            return 1.0
        return ramp_schedule[min(step, len(ramp_schedule) - 1)]
    
    def should_exit_risk_off(
        self, drawdown_velocity: float, vol_of_vol: float
    ) -> bool:
        """Check hysteresis conditions for exiting risk-off mode."""
        if not self.is_active:
            return True
        
        # Must complete minimum duration
        elapsed = self.current_session - self.trigger_session
        if elapsed < self.duration:
            return False
        
        # Hysteresis: both conditions must be improving
        dd_improving = drawdown_velocity < self.last_drawdown_velocity
        vov_improving = vol_of_vol < self.last_vol_of_vol
        
        return dd_improving and vov_improving
    
    def trigger(self, session: int, duration: int, dd_vel: float, vov: float) -> None:
        """Trigger risk-off mode."""
        self.is_active = True
        self.trigger_session = session
        self.duration = max(2, duration)
        self.current_session = session
        self.last_drawdown_velocity = dd_vel
        self.last_vol_of_vol = vov
    
    def advance(self, session: int, dd_vel: float, vov: float) -> None:
        """Advance to next session and update hysteresis state."""
        self.current_session = session
        
        # Check if we can exit
        if self.should_exit_risk_off(dd_vel, vov):
            self.is_active = False
        
        # Update last values for next hysteresis check
        self.last_drawdown_velocity = dd_vel
        self.last_vol_of_vol = vov


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

def compute_policy_reward(
    *,
    net_return: float,
    realized_vol: float,
    turnover: float,
    drawdown_delta: float,
    target_vol: float,
    # Penalty weights
    turnover_penalty: float = 0.1,
    drawdown_penalty: float = 0.5,
    vol_tracking_penalty: float = 0.2,
    # Scaling
    sharpe_window_scale: float = 1.0,  # √(252/window) for annualization
) -> float:
    """Compute risk-adjusted reward for policy update.
    
    Reward = α * r_t / σ_realized                   # IR-like
           - β * |τ_t|                               # Turnover penalty
           - γ * max(0, DD_t - DD_{t-1})             # Drawdown penalty
           - δ * |σ_realized - σ_target|             # Vol tracking error
    
    This aligns the bandit reward with the same utility form optimized at
    the Phase-2 level (Sharpe/return with drawdown + turnover penalties).
    
    Args:
        net_return: Net return for this session
        realized_vol: Realized annualized volatility
        turnover: Absolute turnover this session
        drawdown_delta: Change in drawdown (positive = worsening)
        target_vol: Target volatility for this action
        turnover_penalty: Weight on turnover term
        drawdown_penalty: Weight on drawdown worsening term
        vol_tracking_penalty: Weight on vol tracking error
        sharpe_window_scale: Scaling factor for annualization
    
    Returns:
        Scalar reward value
    """
    # IR-like component: return / vol
    if realized_vol < 1e-8:
        ir_component = float(net_return) * 100.0  # Scale up for low-vol periods
    else:
        ir_component = float(net_return) / float(realized_vol)
    
    # Scale for annualization
    ir_component *= float(sharpe_window_scale)
    
    # Turnover penalty (absolute)
    turn_penalty = float(turnover_penalty) * abs(float(turnover))
    
    # Drawdown penalty (only penalize worsening)
    dd_penalty = float(drawdown_penalty) * max(0.0, float(drawdown_delta))
    
    # Vol tracking error (penalize deviation from target)
    vol_error = float(vol_tracking_penalty) * abs(float(realized_vol) - float(target_vol))
    
    reward = float(ir_component) - float(turn_penalty) - float(dd_penalty) - float(vol_error)
    
    # Clip to prevent extreme values from destabilizing the bandit
    return float(np.clip(reward, -10.0, 10.0))


def compute_reward_rolling_sharpe(
    returns: np.ndarray,
    turnover: float,
    drawdown: float,
    *,
    turnover_penalty: float = 0.1,
    drawdown_penalty: float = 0.3,
    annualization: float = 252.0,
) -> float:
    """Compute reward based on rolling window Sharpe ratio.
    
    This is an alternative to instantaneous reward, using a short rolling
    window to reduce noise.
    
    Args:
        returns: Array of recent returns (e.g., last 5-10 sessions)
        turnover: Current session turnover
        drawdown: Current drawdown level
        turnover_penalty: Weight on turnover
        drawdown_penalty: Weight on drawdown level
        annualization: Annualization factor
    
    Returns:
        Scalar reward value
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    
    if r.size < 2:
        sharpe = 0.0
    else:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        if sigma < 1e-8:
            sharpe = float(mu) * 100.0
        else:
            sharpe = float(mu) / float(sigma) * math.sqrt(float(annualization))
    
    reward = (
        sharpe
        - float(turnover_penalty) * abs(float(turnover))
        - float(drawdown_penalty) * abs(float(drawdown))
    )
    
    return float(np.clip(reward, -10.0, 10.0))


# =============================================================================
# POLICY CONTROLLER (CONTEXTUAL BANDIT)
# =============================================================================

class PolicyController:
    """Lightweight contextual bandit over a small action space.

    Uses linear Thompson sampling by default with exploration annealing.
    Each action maintains its own Bayesian linear regression posterior.
    
    v2 enhancements:
    - Exploration annealing: ts_noise_var decays with configurable half-life
    - Risk-off state tracking for gradual ramp
    - Improved numerical stability
    """

    def __init__(
        self,
        actions: Sequence[PolicyAction],
        feature_dim: int,
        *,
        method: str = "lin_ts",  # lin_ts | ewa
        seed: Optional[int] = None,
        ts_prior_var: float = 1.0,
        ts_noise_var: float = 1.0,
        ewa_eta: float = 0.5,
        ewa_temperature: float = 1.0,
        warmup_steps: int = 0,
        # v2: Exploration annealing
        exploration_half_life: int = 63,  # Sessions for noise to decay by half
        exploration_floor: float = 0.1,   # Minimum exploration (10% of initial)
    ) -> None:
        self.actions = list(actions)
        if not self.actions:
            raise ValueError("PolicyController requires at least one action")
        self.feature_dim = int(feature_dim)
        self.method = str(method or "lin_ts").lower().strip()
        self.rng = np.random.default_rng(int(seed) if seed is not None else None)
        self.ts_prior_var = float(ts_prior_var)
        self.ts_noise_var_initial = float(ts_noise_var)
        self.ts_noise_var = float(ts_noise_var)
        self.ewa_eta = float(ewa_eta)
        self.ewa_temperature = float(ewa_temperature)
        self.warmup_steps = int(max(0, warmup_steps))
        
        # Exploration annealing
        self.exploration_half_life = int(max(1, exploration_half_life))
        self.exploration_floor = float(np.clip(exploration_floor, 0.01, 0.99))

        n = len(self.actions)
        d = self.feature_dim
        if self.method == "ewa":
            self._weights = np.ones(n, dtype=float)
        else:
            # Linear Thompson sampling state per action.
            self._A = [np.eye(d, dtype=float) * (1.0 / max(1e-6, self.ts_prior_var)) for _ in range(n)]
            self._b = [np.zeros(d, dtype=float) for _ in range(n)]
        self._t = 0
        
        # Risk-off state tracking
        self._risk_off_state = RiskOffState()
        
        # Action selection history for diagnostics
        self._action_counts = np.zeros(n, dtype=int)
        self._cumulative_rewards = np.zeros(n, dtype=float)

    @property
    def risk_off_state(self) -> RiskOffState:
        """Get current risk-off state."""
        return self._risk_off_state

    def _compute_annealed_noise_var(self) -> float:
        """Compute exploration noise with exponential decay.
        
        noise_var(t) = max(floor, initial * 0.5^(t / half_life))
        """
        if self._t <= self.warmup_steps:
            return self.ts_noise_var_initial
        
        effective_t = self._t - self.warmup_steps
        decay = 0.5 ** (effective_t / self.exploration_half_life)
        noise_var = self.ts_noise_var_initial * decay
        
        floor_noise = self.ts_noise_var_initial * self.exploration_floor
        return max(noise_var, floor_noise)

    def select_action(self, x: np.ndarray) -> Tuple[int, PolicyAction]:
        """Select action based on current state.
        
        Args:
            x: State vector (must match feature_dim)
        
        Returns:
            Tuple of (action_index, PolicyAction)
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.feature_dim):
            raise ValueError(f"policy state dim mismatch: expected {self.feature_dim}, got {x.size}")

        n = len(self.actions)
        if self._t < self.warmup_steps:
            idx = int(self.rng.integers(0, n))
            self._t += 1
            self._action_counts[idx] += 1
            return idx, self.actions[idx]

        if self.method == "ewa":
            w = np.asarray(self._weights, dtype=float)
            # Softmax with temperature for exploration.
            logits = w / max(1e-6, float(self.ewa_temperature))
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs = probs / float(np.sum(probs)) if float(np.sum(probs)) > 0 else np.full(n, 1.0 / n)
            idx = int(self.rng.choice(np.arange(n), p=probs))
            self._t += 1
            self._action_counts[idx] += 1
            return idx, self.actions[idx]

        # Linear Thompson sampling with annealed exploration.
        noise_var = self._compute_annealed_noise_var()
        
        best_idx = 0
        best_score = -1e18
        for i in range(n):
            A = self._A[i]
            b = self._b[i]
            try:
                A_inv = np.linalg.pinv(A)
            except Exception:
                A_inv = np.eye(self.feature_dim, dtype=float)
            mu = A_inv @ b
            try:
                cov = float(noise_var) * A_inv
                # Add small regularization for numerical stability
                cov = cov + np.eye(self.feature_dim) * 1e-8
                theta = self.rng.multivariate_normal(mean=mu, cov=cov)
            except Exception:
                theta = mu
            score = float(np.dot(theta, x))
            if score > best_score:
                best_score = score
                best_idx = i

        self._t += 1
        self._action_counts[best_idx] += 1
        return int(best_idx), self.actions[int(best_idx)]

    def update(self, *, action_index: int, x: np.ndarray, reward: float) -> None:
        """Update bandit posterior with observed reward.
        
        Args:
            action_index: Index of action that was taken
            x: State vector at time of action
            reward: Observed reward
        """
        idx = int(action_index)
        if idx < 0 or idx >= len(self.actions):
            return
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != int(self.feature_dim):
            return
        r = float(reward)
        
        # Track cumulative rewards
        self._cumulative_rewards[idx] += r

        if self.method == "ewa":
            # Exponentially weighted experts update.
            self._weights[idx] = float(self._weights[idx]) * float(np.exp(self.ewa_eta * r))
            return

        # Linear Thompson sampling update.
        mat_a = self._A[idx]
        vec_b = self._b[idx]
        mat_a = mat_a + np.outer(x, x)
        vec_b = vec_b + r * x
        self._A[idx] = mat_a
        self._b[idx] = vec_b

    def trigger_risk_off(
        self,
        session: int,
        duration: int,
        drawdown_velocity: float,
        vol_of_vol: float,
    ) -> None:
        """Trigger risk-off mode with gradual ramp-back."""
        self._risk_off_state.trigger(session, duration, drawdown_velocity, vol_of_vol)

    def advance_risk_off(
        self, session: int, drawdown_velocity: float, vol_of_vol: float
    ) -> float:
        """Advance risk-off state and return current leverage multiplier.
        
        Returns:
            Leverage multiplier in [0, 1]
        """
        self._risk_off_state.advance(session, drawdown_velocity, vol_of_vol)
        return self._risk_off_state.get_leverage_mult()

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics for logging."""
        total_selections = int(np.sum(self._action_counts))
        
        diagnostics = {
            "policy_total_steps": float(self._t),
            "policy_noise_var": float(self._compute_annealed_noise_var()),
            "policy_risk_off_active": float(self._risk_off_state.is_active),
            "policy_num_actions": float(len(self.actions)),
        }
        
        # Per-action statistics
        for i, action in enumerate(self.actions):
            prefix = f"policy_action_{action.name}"
            count = int(self._action_counts[i])
            diagnostics[f"{prefix}_count"] = float(count)
            diagnostics[f"{prefix}_frac"] = float(count) / max(1, total_selections)
            diagnostics[f"{prefix}_avg_reward"] = (
                float(self._cumulative_rewards[i]) / max(1, count)
            )
        
        return diagnostics


# =============================================================================
# DEFAULT ACTION MENU
# =============================================================================

def build_default_policy_actions_v2(
    *,
    base_threshold: float = 0.05,
    regime_mult_bull: float = 0.9,
    regime_mult_bear: float = 1.5,
    regime_mult_crisis: float = 3.0,
    target_vol: float = 0.15,
    turnover_cap: float = 0.5,
    max_gross: float = 1.5,
    max_net: float = 0.15,
    max_name: float = 0.10,
) -> List[PolicyAction]:
    """Build default action menu for policy controller v2.
    
    Returns 7 actions covering the key trading desk decisions:
    1. normal: Baseline calibrated settings
    2. conservative: Low vol, tight caps, high risk aversion
    3. aggressive: Higher vol target, looser caps
    4. risk_off: Complete flat with gradual ramp-back
    5. reduce_exposure: Half exposure, higher λ_var
    6. trust_quantile: Lean on quantile forecast when Mamba unreliable
    7. tight_sectors: Strict sector neutrality
    """
    actions = [
        # 1. Normal baseline
        PolicyAction(
            name="normal",
            base_threshold=base_threshold,
            regime_mult_bull=regime_mult_bull,
            regime_mult_bear=regime_mult_bear,
            regime_mult_crisis=regime_mult_crisis,
            target_vol=target_vol,
            turnover_cap=turnover_cap,
            max_gross=max_gross,
            max_net=max_net,
            max_name=max_name,
            quantile_blend_weight=0.0,
            risk_off_duration=0,
            max_leverage_schedule=1.0,
            sector_cap_strength=1.0,
            confidence_floor=0.3,
            robust_lambda_var_mult=1.0,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=1.0,
        ),
        
        # 2. Conservative
        PolicyAction(
            name="conservative",
            base_threshold=base_threshold * 1.5,
            regime_mult_bull=regime_mult_bull * 0.8,
            regime_mult_bear=regime_mult_bear * 1.3,
            regime_mult_crisis=regime_mult_crisis * 1.5,
            target_vol=target_vol * 0.6,
            turnover_cap=turnover_cap * 0.7,
            max_gross=max_gross * 0.6,
            max_net=max_net * 0.5,
            max_name=max_name * 0.7,
            quantile_blend_weight=0.2,
            risk_off_duration=0,
            max_leverage_schedule=0.7,
            sector_cap_strength=1.2,
            confidence_floor=0.5,
            robust_lambda_var_mult=2.0,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=0.8,
        ),
        
        # 3. Aggressive
        PolicyAction(
            name="aggressive",
            base_threshold=base_threshold * 0.7,
            regime_mult_bull=regime_mult_bull * 1.2,
            regime_mult_bear=regime_mult_bear * 0.8,
            regime_mult_crisis=regime_mult_crisis * 0.7,
            target_vol=target_vol * 1.4,
            turnover_cap=turnover_cap * 1.3,
            max_gross=max_gross * 1.3,
            max_net=max_net * 1.5,
            max_name=max_name * 1.3,
            quantile_blend_weight=0.0,
            risk_off_duration=0,
            max_leverage_schedule=1.3,
            sector_cap_strength=0.7,
            confidence_floor=0.2,
            robust_lambda_var_mult=0.5,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=1.2,
        ),
        
        # 4. Risk-off (flat with gradual ramp-back)
        PolicyAction(
            name="risk_off",
            base_threshold=base_threshold * 3.0,
            regime_mult_bull=1.0,
            regime_mult_bear=1.0,
            regime_mult_crisis=1.0,
            target_vol=0.0,
            turnover_cap=0.0,
            max_gross=0.0,
            max_net=0.0,
            max_name=0.0,
            quantile_blend_weight=0.0,
            risk_off_duration=5,
            max_leverage_schedule=0.0,
            sector_cap_strength=1.0,
            confidence_floor=1.0,  # Never trade
            robust_lambda_var_mult=10.0,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=0.5,
        ),
        
        # 5. Reduce exposure (half leverage)
        PolicyAction(
            name="reduce_exposure",
            base_threshold=base_threshold * 1.3,
            regime_mult_bull=regime_mult_bull,
            regime_mult_bear=regime_mult_bear * 1.2,
            regime_mult_crisis=regime_mult_crisis * 1.3,
            target_vol=target_vol * 0.5,
            turnover_cap=turnover_cap * 0.6,
            max_gross=max_gross * 0.5,
            max_net=max_net * 0.5,
            max_name=max_name * 0.6,
            quantile_blend_weight=0.1,
            risk_off_duration=0,
            max_leverage_schedule=0.5,
            sector_cap_strength=1.1,
            confidence_floor=0.4,
            robust_lambda_var_mult=2.0,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=0.9,
        ),
        
        # 6. Trust quantile (lean on quantile forecast)
        PolicyAction(
            name="trust_quantile",
            base_threshold=base_threshold * 1.1,
            regime_mult_bull=regime_mult_bull,
            regime_mult_bear=regime_mult_bear,
            regime_mult_crisis=regime_mult_crisis,
            target_vol=target_vol * 0.8,
            turnover_cap=turnover_cap * 0.8,
            max_gross=max_gross * 0.8,
            max_net=max_net,
            max_name=max_name,
            quantile_blend_weight=0.7,  # Heavy quantile blend
            risk_off_duration=0,
            max_leverage_schedule=0.8,
            sector_cap_strength=1.0,
            confidence_floor=0.5,  # Higher floor
            robust_lambda_var_mult=1.5,
            robust_sigma_blend_override=0.8,  # Higher predicted sigma blend
            cvar_strictness=1.0,
        ),
        
        # 7. Tight sectors (strict sector neutrality)
        PolicyAction(
            name="tight_sectors",
            base_threshold=base_threshold,
            regime_mult_bull=regime_mult_bull,
            regime_mult_bear=regime_mult_bear,
            regime_mult_crisis=regime_mult_crisis,
            target_vol=target_vol * 0.9,
            turnover_cap=turnover_cap * 0.9,
            max_gross=max_gross * 0.9,
            max_net=max_net * 0.5,
            max_name=max_name * 0.8,
            quantile_blend_weight=0.1,
            risk_off_duration=0,
            max_leverage_schedule=0.9,
            sector_cap_strength=1.5,  # Strict sector caps
            confidence_floor=0.35,
            robust_lambda_var_mult=1.2,
            robust_sigma_blend_override=-1.0,
            cvar_strictness=1.0,
        ),
    ]
    
    return actions


# =============================================================================
# STATE VECTOR UTILITIES
# =============================================================================

def normalize_state_feature(
    value: float,
    *,
    low: float = 0.0,
    high: float = 1.0,
    clip: bool = True,
) -> float:
    """Normalize a feature to [-1, 1] range.
    
    Args:
        value: Raw feature value
        low: Expected minimum value
        high: Expected maximum value
        clip: Whether to clip to [-1, 1]
    
    Returns:
        Normalized value
    """
    if not np.isfinite(value):
        return 0.0
    
    # Scale to [0, 1]
    range_val = float(high) - float(low)
    if range_val < 1e-8:
        scaled = 0.5
    else:
        scaled = (float(value) - float(low)) / range_val
    
    # Map to [-1, 1]
    normalized = 2.0 * scaled - 1.0
    
    if clip:
        normalized = float(np.clip(normalized, -1.0, 1.0))
    
    return normalized


def compute_vol_of_vol(
    returns: np.ndarray,
    *,
    vol_window: int = 5,
    vov_window: int = 20,
) -> float:
    """Compute volatility-of-volatility (vol-of-vol).
    
    Args:
        returns: Return series
        vol_window: Window for rolling volatility
        vov_window: Window for std of rolling vol
    
    Returns:
        Vol-of-vol (std of rolling volatility)
    """
    r = np.asarray(returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    
    if r.size < vol_window + vov_window:
        return 0.0
    
    # Rolling volatility
    rolling_vol = []
    for i in range(vol_window, len(r) + 1):
        window = r[i - vol_window:i]
        vol = float(np.std(window))
        rolling_vol.append(vol)
    
    if len(rolling_vol) < vov_window:
        return 0.0
    
    # Std of rolling vol
    rolling_vol = np.array(rolling_vol[-vov_window:])
    return float(np.std(rolling_vol))


def compute_correlation_hhi(corr_matrix: np.ndarray) -> float:
    """Compute HHI (Herfindahl-Hirschman Index) of correlation eigenvalues.
    
    Higher HHI = more concentrated = more systemic risk.
    
    Args:
        corr_matrix: Correlation matrix (N x N)
    
    Returns:
        HHI in [0, 1], where 1 = single dominant factor
    """
    corr = np.asarray(corr_matrix, dtype=float)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return 0.0
    
    try:
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.where(np.isfinite(eigvals), eigvals, 0.0)
        eigvals = np.maximum(eigvals, 0.0)  # Ensure non-negative
        
        total = float(np.sum(eigvals))
        if total < 1e-8:
            return 0.0
        
        # Normalize to get "shares"
        shares = eigvals / total
        
        # HHI = sum of squared shares
        hhi = float(np.sum(shares ** 2))
        
        return hhi
    except Exception:
        return 0.0


def compute_expected_shortfall(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute Expected Shortfall (CVaR) at given confidence level.
    
    Args:
        returns: Return series
        alpha: Tail percentile (0.05 = 5% worst cases)
    
    Returns:
        Expected shortfall (average of worst alpha% returns, positive = loss)
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    
    if r.size < 5:
        return 0.0
    
    # Sort returns (worst first)
    sorted_r = np.sort(r)
    
    # Take worst alpha% of returns
    n_tail = max(1, int(len(sorted_r) * alpha))
    tail_returns = sorted_r[:n_tail]
    
    # ES = -E[R | R <= VaR] (positive = loss)
    es = -float(np.mean(tail_returns))
    
    return max(0.0, es)


def compute_drawdown_velocity(
    equity: np.ndarray,
    window: int = 5,
) -> float:
    """Compute rate of drawdown change (momentum).
    
    Positive = drawdown worsening (bad)
    Negative = drawdown improving (good)
    
    Args:
        equity: Equity curve
        window: Window for velocity calculation
    
    Returns:
        Drawdown velocity (change rate)
    """
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    
    if eq.size < window + 1:
        return 0.0
    
    # Compute running drawdown
    running_max = np.maximum.accumulate(eq)
    drawdown = (running_max - eq) / np.maximum(running_max, 1e-8)
    
    # Velocity = change in drawdown over window
    if len(drawdown) < window:
        return 0.0
    
    dd_change = float(drawdown[-1]) - float(drawdown[-window])
    velocity = dd_change / window
    
    return velocity


def compute_sector_imbalance(
    weights: np.ndarray,
    sector_map: Dict[int, str],
    target_sector_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute max sector weight deviation from target.
    
    Args:
        weights: Current portfolio weights (N,)
        sector_map: Map from asset index to sector name
        target_sector_weights: Target weights per sector (default: equal)
    
    Returns:
        Max absolute deviation from target
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    
    if not sector_map:
        return 0.0
    
    # Compute sector weights
    sector_weights: Dict[str, float] = {}
    for i, wt in enumerate(w):
        sector = sector_map.get(i, "unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(wt)
    
    sectors = list(sector_weights.keys())
    if not sectors:
        return 0.0
    
    # Default target: equal weight
    if target_sector_weights is None:
        equal_weight = 1.0 / len(sectors)
        target_sector_weights = {s: equal_weight for s in sectors}
    
    # Max deviation
    max_dev = 0.0
    for sector, actual in sector_weights.items():
        target = target_sector_weights.get(sector, 0.0)
        dev = abs(actual - target)
        max_dev = max(max_dev, dev)
    
    return max_dev


def compute_mu_sigma_dispersion(
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Compute dispersion of |μ|/σ ratio across assets.
    
    High dispersion = model very confident about some assets, uncertain about others.
    
    Args:
        mu: Predicted means (N,)
        sigma: Predicted stds (N,)
    
    Returns:
        Std of |μ|/σ ratios
    """
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    
    # Avoid division by zero
    s = np.where(s > 1e-8, s, 1e-8)
    
    ratio = np.abs(m) / s
    ratio = ratio[np.isfinite(ratio)]
    
    if ratio.size < 2:
        return 0.0
    
    return float(np.std(ratio))
