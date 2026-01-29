"""
Risk Latch State Machine.

Unified risk-control system with clear precedence hierarchy.
All risk mechanisms (policy, kill-switches, overlays) are consolidated
into a single state machine that determines portfolio behavior.

Precedence Order (hard overrides):
    1. EMERGENCY_STOP  - Flatten immediately, latch for N sessions, manual release
    2. SAFE_FALLBACK   - Hold last known good holdings OR benchmark basket
    3. FLATTEN         - Force w_target = 0 (kill-switches: NaN/Inf, DD, vol, turnover)
    4. THROTTLE        - Scale exposure (soft risk-off, policy stress scaling)
    5. NORMAL          - Everything runs normally

This ensures "policy says risk-on" while a kill-switch says "flat"
never creates ambiguous behavior.

Copyright 2024-2026. All rights reserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Latch Modes (ordered by precedence, highest first)
# ─────────────────────────────────────────────────────────────────────────────

class RiskLatchMode(IntEnum):
    """
    Risk latch modes ordered by precedence (highest = strongest).
    
    Higher values override lower values. The state machine always
    uses the highest active mode.
    """
    NORMAL = 0          # Everything runs (policy + overlays + optimizer)
    THROTTLE = 1        # Scale exposure (multiply z or shrink caps)
    FLATTEN = 2         # Force w_target = 0 (keep execution bookkeeping)
    SAFE_FALLBACK = 3   # Hold last known good holdings OR benchmark basket
    EMERGENCY_STOP = 4  # Flatten immediately + latch for N sessions


# ─────────────────────────────────────────────────────────────────────────────
# Trigger Conditions
# ─────────────────────────────────────────────────────────────────────────────

class TriggerReason(IntEnum):
    """Reasons for triggering a risk latch mode."""
    
    # EMERGENCY_STOP triggers (mode 4)
    MANUAL_EMERGENCY = auto()       # Manual operator intervention
    SYSTEM_FAILURE = auto()         # Critical system error
    EXCHANGE_HALT = auto()          # Exchange-wide trading halt
    
    # SAFE_FALLBACK triggers (mode 3)
    DATA_FAILURE = auto()           # Data feed failure / stale data
    MODEL_ANOMALY = auto()          # Model output anomaly (z-score explosion)
    CALIBRATION_COLLAPSE = auto()   # Calibration score collapsed
    INPUT_HEALTH_FAIL = auto()      # Input data health check failed
    OUTPUT_SANITY_FAIL = auto()     # Output sanity check failed
    ELIGIBLE_MASK_COLLAPSE = auto() # Eligible mask collapsed (80% drop)
    
    # FLATTEN triggers (mode 2)
    NAN_INF_DETECTED = auto()       # NaN/Inf in predictions or weights
    DRAWDOWN_KILL = auto()          # Max drawdown threshold breached
    VOL_KILL = auto()               # Realized vol exceeds limit
    TURNOVER_KILL = auto()          # Turnover exceeds limit
    ONE_DAY_CRASH = auto()          # Single-day loss exceeds threshold
    ONE_DAY_LOSS_KILL = auto()      # Immediate 1-day loss kill (EMERGENCY)
    GAP_SHOCK = auto()              # Overnight gap / open shock
    HYGIENE_KILL = auto()           # Portfolio hygiene check failed
    CORRELATION_SPIKE = auto()      # Correlation HHI spike
    
    # THROTTLE triggers (mode 1)
    POLICY_RISK_OFF = auto()        # Policy mandates risk-off
    DRAWDOWN_STRESS = auto()        # Approaching drawdown limit
    VOL_STRESS = auto()             # Vol approaching limit
    REGIME_STRESS = auto()          # Crisis/bear regime
    LIQUIDITY_STRESS = auto()       # Low liquidity conditions
    CBOE_PANIC = auto()             # High VIX / panic premium
    COOLDOWN_ACTIVE = auto()        # Post-event cooldown period
    SIGMA_COLLAPSE = auto()         # Sigma vector collapsed (too many near zero)
    Z_SATURATION = auto()           # Z-scores saturating at clip boundary
    MU_CONSTANT = auto()            # Mu vector near-constant across symbols
    SIGN_FLIP_EXTREME = auto()      # Extreme sign flip rate (>90%)


@dataclass
class TriggerEvent:
    """A trigger event that can change the risk latch state."""
    
    reason: TriggerReason
    mode: RiskLatchMode
    severity: float = 1.0           # 0-1, used for throttle scaling
    message: str = ""
    details: "Dict[str, Any]" = field(default_factory=dict)
    latch_sessions: int = 0         # How long to latch (0 = single-day)
    
    def __post_init__(self):
        if not self.message:
            self.message = f"{self.reason.name} triggered {self.mode.name}"


# ─────────────────────────────────────────────────────────────────────────────
# Trigger Condition Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskLatchThresholds:
    """Configurable thresholds for risk latch triggers."""
    
    # FLATTEN thresholds
    max_drawdown_kill: float = 0.15         # Kill at 15% drawdown
    max_vol_kill: float = 0.40              # Kill at 40% annualized vol
    max_turnover_kill: float = 3.0          # Kill at 300% daily turnover
    one_day_crash_pct: float = -0.05        # Kill on -5% single day
    max_correlation_hhi: float = 0.5        # Kill on HHI > 0.5
    
    # FAST CRASH TRIGGERS (1-day loss kill → EMERGENCY_STOP)
    one_day_loss_kill_pct: float = 0.03     # Kill on -3% single day loss (EMERGENCY)
    emergency_cooldown_sessions: int = 5    # Sessions to stay flat after emergency
    
    # GAP/OVERNIGHT SHOCK TRIGGERS
    gap_shock_mult: float = 4.0             # Gap > 4x rolling vol triggers
    gap_shock_min_pct: float = 0.05         # Minimum gap % to trigger (5%)
    gap_shock_n_symbols: int = 3            # Number of symbols with gaps to trigger portfolio-level
    gap_shock_portfolio_pct: float = 0.02   # Portfolio gap > 2% triggers
    gap_shock_mode: str = "throttle"        # "throttle" or "flatten"
    gap_shock_throttle_scale: float = 0.3   # Scale to 30% on gap shock
    
    # THROTTLE thresholds
    drawdown_stress_start: float = 0.08     # Start throttling at 8% DD
    vol_stress_start: float = 0.25          # Start throttling at 25% vol
    cboe_panic_threshold: float = 25.0      # VIX > 25 triggers throttle
    calibration_min: float = 0.3            # Calibration < 0.3 is stress
    
    # SAFE_FALLBACK thresholds
    model_z_explosion: float = 10.0         # |z| > 10 is anomaly
    calibration_collapse: float = 0.1       # Calibration < 0.1 is collapse
    data_staleness_hours: float = 2.0       # Data older than 2h is stale
    
    # EMERGENCY_STOP parameters
    emergency_latch_sessions: int = 5       # Latch for 5 sessions
    post_flatten_cooldown: int = 3          # Cooldown after flatten
    
    # Throttle scaling
    throttle_min_scale: float = 0.1         # Minimum exposure scale
    throttle_max_scale: float = 1.0         # Maximum exposure scale
    
    # ═══════════════════════════════════════════════════════════════════════
    # INPUT HEALTH THRESHOLDS (Point 1 - start-of-day prechecks)
    # ═══════════════════════════════════════════════════════════════════════
    hygiene_ok_min_pct: float = 0.30        # Min % hygiene_ok to proceed (30%); use 0.50 for live
    role_ctx_stale_days: int = 2            # role_ctx stale if > N days old
    eligible_mask_collapse_pct: float = 0.50  # Collapse if 50% fewer eligible vs yesterday
    
    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT SANITY THRESHOLDS (Point 2 - post-signal, pre-optimizer)
    # ═══════════════════════════════════════════════════════════════════════
    sigma_collapse_eps: float = 1e-6        # Sigma considered zero if < eps
    sigma_collapse_pct: float = 0.50        # SAFE_FALLBACK if > 50% sigmas collapsed; use 0.30 for live
    z_saturation_clip: float = 3.0          # z-clip boundary
    z_saturation_pct: float = 0.20          # THROTTLE if > 20% z's at clip (model saturating)
    mu_constant_std_min: float = 0.001      # Mu near-constant if std < this
    sign_flip_extreme_pct: float = 0.95     # Extreme if > 95% signs flip (likely broken, not regime)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Latch State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskLatchState:
    """Current state of the risk latch."""
    
    mode: RiskLatchMode = RiskLatchMode.NORMAL
    primary_reason: Optional[TriggerReason] = None
    all_triggers: "List[TriggerEvent]" = field(default_factory=list)
    
    # Throttle parameters
    exposure_scale: float = 1.0             # Applied to z or weights
    max_gross_scale: float = 1.0            # Applied to max_gross constraint
    max_name_scale: float = 1.0             # Applied to max_name constraint
    
    # Latch tracking
    latch_remaining_sessions: int = 0       # Sessions until auto-release
    cooldown_remaining_sessions: int = 0    # Post-event cooldown
    
    # Safe fallback state
    last_good_weights: Optional[np.ndarray] = None
    benchmark_weights: Optional[np.ndarray] = None
    
    # Audit trail
    mode_since_session: int = 0             # Session when mode was entered
    total_flatten_count: int = 0            # Total flattens this run
    total_emergency_count: int = 0          # Total emergencies this run
    
    def is_active(self) -> bool:
        """True if any risk control is active (not NORMAL)."""
        return self.mode != RiskLatchMode.NORMAL
    
    def should_flatten(self) -> bool:
        """
        True if we should flatten positions.
        
        Note: SAFE_FALLBACK (mode 3) does NOT flatten - it holds last good weights.
        Only FLATTEN (mode 2) and EMERGENCY_STOP (mode 4) cause flattening.
        """
        return self.mode == RiskLatchMode.FLATTEN or self.mode == RiskLatchMode.EMERGENCY_STOP
    
    def should_hold_fallback(self) -> bool:
        """True if we should hold last good weights or benchmark."""
        return self.mode == RiskLatchMode.SAFE_FALLBACK
    
    def should_throttle(self) -> bool:
        """True if we should throttle exposure."""
        return self.mode >= RiskLatchMode.THROTTLE
    
    def get_effective_weights(
        self,
        target_weights: np.ndarray,
        n_assets: int,
    ) -> np.ndarray:
        """
        Get effective weights after applying risk latch.
        
        Args:
            target_weights: Optimizer-computed target weights
            n_assets: Number of assets
        
        Returns:
            Effective weights after risk latch
        """
        if self.mode == RiskLatchMode.EMERGENCY_STOP:
            # Immediate flatten
            return np.zeros(n_assets, dtype=float)
        
        elif self.mode == RiskLatchMode.SAFE_FALLBACK:
            # Use last known good or benchmark
            if self.last_good_weights is not None and len(self.last_good_weights) == n_assets:
                return self.last_good_weights.copy()
            elif self.benchmark_weights is not None and len(self.benchmark_weights) == n_assets:
                return self.benchmark_weights.copy()
            else:
                # No fallback available, flatten
                return np.zeros(n_assets, dtype=float)
        
        elif self.mode == RiskLatchMode.FLATTEN:
            # Force flat
            return np.zeros(n_assets, dtype=float)
        
        elif self.mode == RiskLatchMode.THROTTLE:
            # Scale target weights
            return target_weights * self.exposure_scale
        
        else:  # NORMAL
            return target_weights
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging."""
        return {
            "mode": self.mode.name,
            "mode_value": int(self.mode),
            "primary_reason": self.primary_reason.name if self.primary_reason else None,
            "exposure_scale": round(self.exposure_scale, 4),
            "max_gross_scale": round(self.max_gross_scale, 4),
            "max_name_scale": round(self.max_name_scale, 4),
            "latch_remaining": self.latch_remaining_sessions,
            "cooldown_remaining": self.cooldown_remaining_sessions,
            "is_active": self.is_active(),
            "should_flatten": self.should_flatten(),
            "should_hold_fallback": self.mode == RiskLatchMode.SAFE_FALLBACK,
            "has_fallback_weights": self.last_good_weights is not None,
            "trigger_count": len(self.all_triggers),
            "total_flatten_count": self.total_flatten_count,
            "total_emergency_count": self.total_emergency_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Risk Latch State Machine
# ─────────────────────────────────────────────────────────────────────────────

class RiskLatch:
    """
    Unified risk-control state machine.
    
    Consolidates all risk mechanisms into a single state machine with
    clear precedence. Higher modes always override lower modes.
    
    Usage:
        latch = RiskLatch(thresholds)
        
        # In Phase-2 loop, before computing weights:
        latch.begin_session(session_idx)
        
        # Check conditions
        latch.check_nan_inf(predictions, weights)
        latch.check_drawdown(current_dd)
        latch.check_volatility(realized_vol)
        latch.check_turnover(turnover)
        latch.check_model_health(calibration_score, z_scores)
        latch.check_policy(policy_action)
        
        # Get effective state
        state = latch.resolve()
        
        # Apply to weights
        if state.should_flatten():
            w_target = np.zeros(n_assets)
        else:
            w_target = optimizer.solve()
            w_target = state.get_effective_weights(w_target, n_assets)
        
        # End session (update latch counters, store good weights)
        latch.end_session(w_target if not state.should_flatten() else None)
    """
    
    def __init__(
        self,
        thresholds: Optional[RiskLatchThresholds] = None,
        n_assets: int = 0,
    ):
        """
        Args:
            thresholds: Configurable thresholds for triggers
            n_assets: Number of assets (for weight arrays)
        """
        self.thresholds = thresholds or RiskLatchThresholds()
        self.n_assets = n_assets
        
        # Current state
        self.state = RiskLatchState()
        
        # Session tracking
        self.current_session: int = 0
        self.session_active: bool = False
        
        # Pending triggers for current session
        self._pending_triggers: List[TriggerEvent] = []
        
        # History
        self.mode_history: List[Tuple[int, RiskLatchMode, str]] = []
        
        # Manual controls
        self._manual_emergency: bool = False
        self._manual_release_requested: bool = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    
    def begin_session(self, session_idx: int) -> None:
        """Begin a new trading session."""
        self.current_session = session_idx
        self.session_active = True
        self._pending_triggers.clear()
        
        # Decrement latch counters (but NOT for SAFE_FALLBACK - it persists until conditions improve)
        if self.state.mode != RiskLatchMode.SAFE_FALLBACK:
            if self.state.latch_remaining_sessions > 0:
                self.state.latch_remaining_sessions -= 1
        
        if self.state.cooldown_remaining_sessions > 0:
            self.state.cooldown_remaining_sessions -= 1
            # Add cooldown trigger if still active
            if self.state.cooldown_remaining_sessions > 0:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.COOLDOWN_ACTIVE,
                    mode=RiskLatchMode.THROTTLE,
                    severity=0.5,
                    message=f"Cooldown active: {self.state.cooldown_remaining_sessions} sessions remaining",
                ))
        
        # Check if latched mode should release
        # SAFE_FALLBACK doesn't auto-release on latch counter - it releases when conditions improve
        if self.state.mode >= RiskLatchMode.FLATTEN and self.state.mode != RiskLatchMode.SAFE_FALLBACK:
            if self.state.latch_remaining_sessions <= 0 and not self._manual_emergency:
                # Transition to cooldown
                self.state.cooldown_remaining_sessions = self.thresholds.post_flatten_cooldown
                logger.info(
                    "[RiskLatch] Session %d: Released from %s, entering cooldown (%d sessions)",
                    session_idx, self.state.mode.name, self.state.cooldown_remaining_sessions
                )
    
    def end_session(self, good_weights: Optional[np.ndarray] = None) -> None:
        """
        End the current trading session.
        
        Args:
            good_weights: If provided, store as last known good weights
        """
        if good_weights is not None and len(good_weights) == self.n_assets:
            if np.all(np.isfinite(good_weights)):
                self.state.last_good_weights = good_weights.copy()
        
        self.session_active = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Trigger Addition
    # ─────────────────────────────────────────────────────────────────────────
    
    def _add_trigger(self, trigger: TriggerEvent) -> None:
        """Add a trigger event."""
        self._pending_triggers.append(trigger)
        
        if trigger.mode >= RiskLatchMode.FLATTEN:
            logger.warning(
                "[RiskLatch] Session %d: %s → %s: %s",
                self.current_session, trigger.reason.name, trigger.mode.name, trigger.message
            )
        else:
            logger.info(
                "[RiskLatch] Session %d: %s → %s: %s",
                self.current_session, trigger.reason.name, trigger.mode.name, trigger.message
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Check Methods (call before resolve)
    # ─────────────────────────────────────────────────────────────────────────
    
    def trigger_manual_emergency(self, message: str = "Manual emergency stop") -> None:
        """Trigger manual emergency stop (highest precedence)."""
        self._manual_emergency = True
        self._add_trigger(TriggerEvent(
            reason=TriggerReason.MANUAL_EMERGENCY,
            mode=RiskLatchMode.EMERGENCY_STOP,
            severity=1.0,
            message=message,
            latch_sessions=self.thresholds.emergency_latch_sessions,
        ))
    
    def release_manual_emergency(self) -> None:
        """Release manual emergency stop."""
        self._manual_emergency = False
        self._manual_release_requested = True
        logger.info("[RiskLatch] Manual emergency release requested")
    
    def check_nan_inf(
        self,
        arrays: "List[Optional[np.ndarray]]",
        names: Optional[List[str]] = None,
    ) -> bool:
        """
        Check for NaN/Inf in arrays.
        
        Returns:
            True if NaN/Inf detected
        """
        names = names or [f"array_{i}" for i in range(len(arrays))]
        
        for arr, name in zip(arrays, names):
            if arr is None:
                continue
            if not np.all(np.isfinite(arr)):
                nan_count = int(np.sum(~np.isfinite(arr)))
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.NAN_INF_DETECTED,
                    mode=RiskLatchMode.FLATTEN,
                    severity=1.0,
                    message=f"NaN/Inf in {name}: {nan_count} values",
                    details={"array_name": name, "nan_count": nan_count},
                    latch_sessions=1,
                ))
                return True
        
        return False
    
    def check_drawdown(self, current_dd: float) -> bool:
        """
        Check drawdown conditions.
        
        Args:
            current_dd: Current drawdown (positive value, e.g., 0.1 = 10%)
        
        Returns:
            True if kill triggered
        """
        if not np.isfinite(current_dd):
            return False
        
        if current_dd >= self.thresholds.max_drawdown_kill:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DRAWDOWN_KILL,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"Max drawdown kill: {current_dd:.1%} >= {self.thresholds.max_drawdown_kill:.1%}",
                details={"current_dd": current_dd, "threshold": self.thresholds.max_drawdown_kill},
                latch_sessions=self.thresholds.post_flatten_cooldown,
            ))
            return True
        
        elif current_dd >= self.thresholds.drawdown_stress_start:
            # Linear scale from 1.0 at stress_start to throttle_min at kill
            stress_range = self.thresholds.max_drawdown_kill - self.thresholds.drawdown_stress_start
            stress_pct = (current_dd - self.thresholds.drawdown_stress_start) / stress_range
            scale = 1.0 - stress_pct * (1.0 - self.thresholds.throttle_min_scale)
            
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DRAWDOWN_STRESS,
                mode=RiskLatchMode.THROTTLE,
                severity=stress_pct,
                message=f"Drawdown stress: {current_dd:.1%}, scale={scale:.2f}",
                details={"current_dd": current_dd, "scale": scale},
            ))
        
        return False
    
    def check_volatility(self, realized_vol: float) -> bool:
        """
        Check realized volatility conditions.
        
        Args:
            realized_vol: Annualized realized volatility
        
        Returns:
            True if kill triggered
        """
        if not np.isfinite(realized_vol):
            return False
        
        if realized_vol >= self.thresholds.max_vol_kill:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.VOL_KILL,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"Max vol kill: {realized_vol:.1%} >= {self.thresholds.max_vol_kill:.1%}",
                details={"realized_vol": realized_vol, "threshold": self.thresholds.max_vol_kill},
                latch_sessions=self.thresholds.post_flatten_cooldown,
            ))
            return True
        
        elif realized_vol >= self.thresholds.vol_stress_start:
            stress_range = self.thresholds.max_vol_kill - self.thresholds.vol_stress_start
            stress_pct = (realized_vol - self.thresholds.vol_stress_start) / stress_range
            scale = 1.0 - stress_pct * (1.0 - self.thresholds.throttle_min_scale)
            
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.VOL_STRESS,
                mode=RiskLatchMode.THROTTLE,
                severity=stress_pct,
                message=f"Vol stress: {realized_vol:.1%}, scale={scale:.2f}",
                details={"realized_vol": realized_vol, "scale": scale},
            ))
        
        return False
    
    def check_turnover(self, turnover: float) -> bool:
        """
        Check turnover conditions.
        
        Args:
            turnover: Daily turnover (1.0 = 100%)
        
        Returns:
            True if kill triggered
        """
        if not np.isfinite(turnover):
            return False
        
        if turnover >= self.thresholds.max_turnover_kill:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.TURNOVER_KILL,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"Max turnover kill: {turnover:.0%} >= {self.thresholds.max_turnover_kill:.0%}",
                details={"turnover": turnover, "threshold": self.thresholds.max_turnover_kill},
                latch_sessions=self.thresholds.post_flatten_cooldown,
            ))
            return True
        
        return False
    
    def check_daily_return(self, daily_return: float) -> bool:
        """
        Check for single-day crash.
        
        Args:
            daily_return: Single-day return (e.g., -0.05 = -5%)
        
        Returns:
            True if kill triggered
        """
        if not np.isfinite(daily_return):
            return False
        
        if daily_return <= self.thresholds.one_day_crash_pct:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.ONE_DAY_CRASH,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"One-day crash kill: {daily_return:.1%} <= {self.thresholds.one_day_crash_pct:.1%}",
                details={"daily_return": daily_return, "threshold": self.thresholds.one_day_crash_pct},
                latch_sessions=self.thresholds.post_flatten_cooldown,
            ))
            return True
        
        return False
    
    def check_one_day_loss_kill(
        self,
        pnl_net_today: float,
        equity_today: float,
        equity_prev: float,
    ) -> bool:
        """
        Check for immediate 1-day loss kill → EMERGENCY_STOP.
        
        This is a fast-reacting trigger for catastrophic single-day losses.
        Should be called RIGHT AFTER computing pnl_net (post-cost).
        
        Args:
            pnl_net_today: Net PnL for today (e.g., -0.03 = -3%)
            equity_today: Equity at end of day
            equity_prev: Equity at start of day (previous close)
        
        Returns:
            True if EMERGENCY_STOP triggered
        """
        if not np.isfinite(pnl_net_today) or not np.isfinite(equity_today) or not np.isfinite(equity_prev):
            return False
        
        # Check PnL directly
        loss_pnl = -pnl_net_today  # Convert to positive loss
        
        # Also check equity ratio
        equity_loss = 0.0
        if equity_prev > 0:
            equity_loss = 1.0 - (equity_today / equity_prev)
        
        # Take worst of both
        actual_loss = max(loss_pnl, equity_loss)
        threshold = self.thresholds.one_day_loss_kill_pct
        
        if actual_loss >= threshold:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.ONE_DAY_LOSS_KILL,
                mode=RiskLatchMode.EMERGENCY_STOP,
                severity=1.0,
                message=f"1-DAY LOSS KILL: {actual_loss:.2%} loss >= {threshold:.2%} threshold → EMERGENCY_STOP",
                details={
                    "pnl_net_today": float(pnl_net_today),
                    "equity_loss": float(equity_loss),
                    "actual_loss": float(actual_loss),
                    "threshold": float(threshold),
                },
                latch_sessions=self.thresholds.emergency_cooldown_sessions,
            ))
            # Set emergency cooldown
            self.state.cooldown_remaining_sessions = self.thresholds.emergency_cooldown_sessions
            return True
        
        return False
    
    def check_gap_shock(
        self,
        r_vec: "Optional[np.ndarray]",
        rolling_vol: Optional[np.ndarray] = None,
        prev_weights: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None,
    ) -> bool:
        """
        Check for overnight gap / open shock (pre-trade).
        
        Detects dangerous gaps that may indicate regime shifts or
        overnight news that invalidates yesterday's positions.
        
        Should be called at START OF DAY, before optimization.
        
        Args:
            r_vec: Today's returns (close-to-close or open-to-prev-close)
            rolling_vol: Per-asset rolling volatility (annualized, optional)
            prev_weights: Previous day's weights for portfolio gap calc (optional)
            symbols: Symbol names for logging (optional)
        
        Returns:
            True if gap shock triggered
        """
        if r_vec is None or len(r_vec) == 0:
            return False
        
        n_assets = len(r_vec)
        thresholds = self.thresholds
        
        # ─────────────────────────────────────────────────────────────────────
        # A) Check individual symbol gaps: |r| > gap_mult * rolling_vol
        # ─────────────────────────────────────────────────────────────────────
        n_shocked = 0
        shocked_symbols: List[str] = []
        
        if rolling_vol is not None and len(rolling_vol) == n_assets:
            # Daily vol from annualized
            daily_vol = rolling_vol / np.sqrt(252.0)
            gap_threshold = thresholds.gap_shock_mult * daily_vol
            
            for j in range(n_assets):
                if not np.isfinite(r_vec[j]) or not np.isfinite(gap_threshold[j]):
                    continue
                # Also check minimum absolute gap
                abs_r = abs(r_vec[j])
                if abs_r > gap_threshold[j] and abs_r >= thresholds.gap_shock_min_pct:
                    n_shocked += 1
                    if symbols is not None and j < len(symbols):
                        shocked_symbols.append(symbols[j])
        else:
            # Without rolling vol, just use minimum absolute threshold
            for j in range(n_assets):
                if np.isfinite(r_vec[j]) and abs(r_vec[j]) >= thresholds.gap_shock_min_pct:
                    n_shocked += 1
                    if symbols is not None and j < len(symbols):
                        shocked_symbols.append(symbols[j])
        
        # ─────────────────────────────────────────────────────────────────────
        # B) Check portfolio-level gap: w' @ r
        # ─────────────────────────────────────────────────────────────────────
        portfolio_gap = 0.0
        if prev_weights is not None and len(prev_weights) == n_assets:
            portfolio_gap = float(np.dot(prev_weights, r_vec))
        
        # ─────────────────────────────────────────────────────────────────────
        # C) Determine if trigger fires
        # ─────────────────────────────────────────────────────────────────────
        symbol_shock = n_shocked >= thresholds.gap_shock_n_symbols
        portfolio_shock = abs(portfolio_gap) >= thresholds.gap_shock_portfolio_pct
        
        if symbol_shock or portfolio_shock:
            # Determine mode based on config
            if thresholds.gap_shock_mode == "flatten":
                mode = RiskLatchMode.FLATTEN
                severity = 1.0
            else:
                mode = RiskLatchMode.THROTTLE
                severity = thresholds.gap_shock_throttle_scale
            
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.GAP_SHOCK,
                mode=mode,
                severity=severity,
                message=f"GAP SHOCK: {n_shocked} symbols shocked, portfolio_gap={portfolio_gap:.2%}",
                details={
                    "n_shocked": n_shocked,
                    "portfolio_gap": float(portfolio_gap),
                    "shocked_symbols": shocked_symbols[:5],  # Top 5 for logging
                    "mode": thresholds.gap_shock_mode,
                },
                latch_sessions=1,  # Single day latch
            ))
            return True
        
        return False
    
    def check_correlation_hhi(self, corr_hhi: float) -> bool:
        """
        Check correlation concentration.
        
        Args:
            corr_hhi: Correlation HHI (0-1)
        
        Returns:
            True if kill triggered
        """
        if not np.isfinite(corr_hhi):
            return False
        
        if corr_hhi >= self.thresholds.max_correlation_hhi:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.CORRELATION_SPIKE,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message=f"Correlation spike: HHI={corr_hhi:.2f} >= {self.thresholds.max_correlation_hhi:.2f}",
                details={"corr_hhi": corr_hhi, "threshold": self.thresholds.max_correlation_hhi},
                latch_sessions=1,
            ))
            return True
        
        return False
    
    def check_model_health(
        self,
        calibration_score: float,
        z_scores: Optional[np.ndarray] = None,
    ) -> RiskLatchMode:
        """
        Check model health (calibration, z-score explosion).
        
        Args:
            calibration_score: Mamba calibration score (0-1)
            z_scores: Current z-scores (optional)
        
        Returns:
            Triggered mode (NORMAL if no issue)
        """
        # Check for calibration collapse → SAFE_FALLBACK
        if calibration_score < self.thresholds.calibration_collapse:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.CALIBRATION_COLLAPSE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"Calibration collapse: {calibration_score:.3f} < {self.thresholds.calibration_collapse:.3f}",
                details={"calibration": calibration_score},
                latch_sessions=2,
            ))
            return RiskLatchMode.SAFE_FALLBACK
        
        # Check for z-score explosion → SAFE_FALLBACK
        if z_scores is not None and len(z_scores) > 0:
            max_abs_z = float(np.max(np.abs(z_scores[np.isfinite(z_scores)]))) if np.any(np.isfinite(z_scores)) else 0.0
            if max_abs_z > self.thresholds.model_z_explosion:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.MODEL_ANOMALY,
                    mode=RiskLatchMode.SAFE_FALLBACK,
                    severity=1.0,
                    message=f"Model anomaly: max|z|={max_abs_z:.1f} > {self.thresholds.model_z_explosion:.1f}",
                    details={"max_abs_z": max_abs_z},
                    latch_sessions=1,
                ))
                return RiskLatchMode.SAFE_FALLBACK
        
        # Check for calibration stress → THROTTLE
        if calibration_score < self.thresholds.calibration_min:
            stress_pct = 1.0 - (calibration_score / self.thresholds.calibration_min)
            scale = 1.0 - stress_pct * (1.0 - self.thresholds.throttle_min_scale)
            
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.REGIME_STRESS,
                mode=RiskLatchMode.THROTTLE,
                severity=stress_pct,
                message=f"Low calibration stress: {calibration_score:.3f}, scale={scale:.2f}",
                details={"calibration": calibration_score, "scale": scale},
            ))
            return RiskLatchMode.THROTTLE
        
        return RiskLatchMode.NORMAL
    
    def check_cboe_panic(self, vix: float, panic_premium: float = 0.0) -> RiskLatchMode:
        """
        Check CBOE panic conditions.
        
        Args:
            vix: Current VIX level
            panic_premium: VIX panic premium z-score
        
        Returns:
            Triggered mode
        """
        if vix >= self.thresholds.cboe_panic_threshold:
            stress_pct = min(1.0, (vix - self.thresholds.cboe_panic_threshold) / 20.0)
            scale = 1.0 - stress_pct * (1.0 - self.thresholds.throttle_min_scale)
            
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.CBOE_PANIC,
                mode=RiskLatchMode.THROTTLE,
                severity=stress_pct,
                message=f"CBOE panic: VIX={vix:.1f}, scale={scale:.2f}",
                details={"vix": vix, "panic_premium": panic_premium, "scale": scale},
            ))
            return RiskLatchMode.THROTTLE
        
        return RiskLatchMode.NORMAL
    
    def check_policy_action(
        self,
        risk_off: bool = False,
        flat: bool = False,
        scale: float = 1.0,
    ) -> RiskLatchMode:
        """
        Check policy action from strategy layer.
        
        Args:
            risk_off: Policy mandates risk-off
            flat: Policy mandates flat
            scale: Policy exposure scale
        
        Returns:
            Triggered mode
        """
        if flat:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.POLICY_RISK_OFF,
                mode=RiskLatchMode.FLATTEN,
                severity=1.0,
                message="Policy mandates FLAT",
                latch_sessions=1,
            ))
            return RiskLatchMode.FLATTEN
        
        elif risk_off or scale < 1.0:
            effective_scale = min(scale, 0.5 if risk_off else 1.0)
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.POLICY_RISK_OFF,
                mode=RiskLatchMode.THROTTLE,
                severity=1.0 - effective_scale,
                message=f"Policy risk-off: scale={effective_scale:.2f}",
                details={"scale": effective_scale, "risk_off": risk_off},
            ))
            return RiskLatchMode.THROTTLE
        
        return RiskLatchMode.NORMAL
    
    def check_data_health(
        self,
        data_staleness_hours: float = 0.0,
        data_coverage: float = 1.0,
    ) -> RiskLatchMode:
        """
        Check data health conditions.
        
        Args:
            data_staleness_hours: Hours since last data update
            data_coverage: Fraction of assets with valid data
        
        Returns:
            Triggered mode
        """
        if data_staleness_hours > self.thresholds.data_staleness_hours:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DATA_FAILURE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"Data stale: {data_staleness_hours:.1f}h > {self.thresholds.data_staleness_hours:.1f}h",
                details={"staleness_hours": data_staleness_hours},
                latch_sessions=1,
            ))
            return RiskLatchMode.SAFE_FALLBACK
        
        if data_coverage < 0.5:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DATA_FAILURE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"Low data coverage: {data_coverage:.0%}",
                details={"coverage": data_coverage},
                latch_sessions=1,
            ))
            return RiskLatchMode.SAFE_FALLBACK
        
        return RiskLatchMode.NORMAL
    
    # ─────────────────────────────────────────────────────────────────────────
    # Input Health Checks (Point 1 - start-of-day prechecks)
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_input_health(
        self,
        hygiene_ok_pct: float,
        role_ctx_available: bool = True,
        role_ctx_days_stale: int = 0,
        eligible_count_today: int = 0,
        eligible_count_yesterday: int = 0,
    ) -> RiskLatchMode:
        """
        Check input data health before inference.
        
        This is Point 1 in the risk control flow: start-of-day prechecks.
        
        Args:
            hygiene_ok_pct: Percentage of assets with hygiene_ok=True (0-1)
            role_ctx_available: Whether role_ctx is available
            role_ctx_days_stale: Days since last role_ctx update
            eligible_count_today: Number of eligible symbols today
            eligible_count_yesterday: Number of eligible symbols yesterday
        
        Returns:
            Triggered mode
        """
        triggered_mode = RiskLatchMode.NORMAL
        
        # Check hygiene_ok percentage
        if hygiene_ok_pct < self.thresholds.hygiene_ok_min_pct:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.INPUT_HEALTH_FAIL,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"Hygiene check failed: {hygiene_ok_pct:.0%} < {self.thresholds.hygiene_ok_min_pct:.0%}",
                details={"hygiene_ok_pct": hygiene_ok_pct, "threshold": self.thresholds.hygiene_ok_min_pct},
                latch_sessions=1,
            ))
            triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        # Check role_ctx availability
        if not role_ctx_available:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DATA_FAILURE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message="role_ctx not available",
                latch_sessions=1,
            ))
            triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        # Check role_ctx staleness
        elif role_ctx_days_stale > self.thresholds.role_ctx_stale_days:
            self._add_trigger(TriggerEvent(
                reason=TriggerReason.DATA_FAILURE,
                mode=RiskLatchMode.SAFE_FALLBACK,
                severity=1.0,
                message=f"role_ctx stale: {role_ctx_days_stale} days > {self.thresholds.role_ctx_stale_days}",
                details={"days_stale": role_ctx_days_stale},
                latch_sessions=1,
            ))
            triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        # Check eligible mask collapse
        if eligible_count_yesterday > 0 and eligible_count_today > 0:
            drop_pct = 1.0 - (eligible_count_today / eligible_count_yesterday)
            if drop_pct >= self.thresholds.eligible_mask_collapse_pct:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.ELIGIBLE_MASK_COLLAPSE,
                    mode=RiskLatchMode.SAFE_FALLBACK,
                    severity=1.0,
                    message=f"Eligible mask collapsed: {drop_pct:.0%} drop ({eligible_count_yesterday} → {eligible_count_today})",
                    details={
                        "drop_pct": drop_pct,
                        "today": eligible_count_today,
                        "yesterday": eligible_count_yesterday,
                    },
                    latch_sessions=1,
                ))
                triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        return triggered_mode
    
    def check_output_sanity(
        self,
        mu_vec: np.ndarray,
        sigma_vec: np.ndarray,
        z_vec: np.ndarray,
        z_prev: Optional[np.ndarray] = None,
        z_clip: Optional[float] = None,
    ) -> RiskLatchMode:
        """
        Check output sanity after signal generation, before optimizer.
        
        This is Point 2 in the risk control flow: post-signal, pre-optimizer.
        
        Args:
            mu_vec: Predicted mean returns
            sigma_vec: Predicted volatilities
            z_vec: Z-scores (signals)
            z_prev: Previous day's z-scores (for sign flip detection)
            z_clip: Z-score clip boundary (default: from thresholds)
        
        Returns:
            Triggered mode
        """
        triggered_mode = RiskLatchMode.NORMAL
        z_clip = z_clip or self.thresholds.z_saturation_clip
        
        n_valid = int(np.sum(np.isfinite(mu_vec)))
        if n_valid == 0:
            # No valid data at all → NaN check should catch this
            return RiskLatchMode.NORMAL
        
        # ───────────────────────────────────────────────────────────────────
        # Check 1: Sigma collapse (too many sigmas near zero)
        # ───────────────────────────────────────────────────────────────────
        sigma_valid = sigma_vec[np.isfinite(sigma_vec)]
        if len(sigma_valid) > 0:
            n_collapsed = int(np.sum(sigma_valid < self.thresholds.sigma_collapse_eps))
            collapse_pct = n_collapsed / len(sigma_valid)
            
            if collapse_pct >= self.thresholds.sigma_collapse_pct:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.SIGMA_COLLAPSE,
                    mode=RiskLatchMode.SAFE_FALLBACK,
                    severity=1.0,
                    message=f"Sigma collapsed: {collapse_pct:.0%} < eps ({n_collapsed}/{len(sigma_valid)})",
                    details={"collapse_pct": collapse_pct, "n_collapsed": n_collapsed},
                    latch_sessions=1,
                ))
                triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        # ───────────────────────────────────────────────────────────────────
        # Check 2: Z-score saturation (too many at clip boundary)
        # ───────────────────────────────────────────────────────────────────
        z_valid = z_vec[np.isfinite(z_vec)]
        if len(z_valid) > 0:
            n_saturated = int(np.sum(np.abs(z_valid) >= z_clip * 0.99))
            saturation_pct = n_saturated / len(z_valid)
            
            if saturation_pct >= self.thresholds.z_saturation_pct:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.Z_SATURATION,
                    mode=RiskLatchMode.THROTTLE,
                    severity=saturation_pct,
                    message=f"Z saturating: {saturation_pct:.0%} at ±{z_clip} ({n_saturated}/{len(z_valid)})",
                    details={"saturation_pct": saturation_pct, "n_saturated": n_saturated, "z_clip": z_clip},
                ))
                triggered_mode = max(triggered_mode, RiskLatchMode.THROTTLE)
        
        # ───────────────────────────────────────────────────────────────────
        # Check 3: Mu near-constant (all predictions identical)
        # ───────────────────────────────────────────────────────────────────
        mu_valid = mu_vec[np.isfinite(mu_vec)]
        if len(mu_valid) > 1:
            mu_std = float(np.std(mu_valid))
            
            if mu_std < self.thresholds.mu_constant_std_min:
                self._add_trigger(TriggerEvent(
                    reason=TriggerReason.MU_CONSTANT,
                    mode=RiskLatchMode.SAFE_FALLBACK,
                    severity=1.0,
                    message=f"Mu near-constant: std={mu_std:.2e} < {self.thresholds.mu_constant_std_min:.2e}",
                    details={"mu_std": mu_std, "mu_mean": float(np.mean(mu_valid))},
                    latch_sessions=1,
                ))
                triggered_mode = max(triggered_mode, RiskLatchMode.SAFE_FALLBACK)
        
        # ───────────────────────────────────────────────────────────────────
        # Check 4: Extreme sign flip rate (>90% of z's flipped from yesterday)
        # ───────────────────────────────────────────────────────────────────
        if z_prev is not None and len(z_prev) == len(z_vec):
            # Only compare where both are valid and non-zero
            valid_mask = (
                np.isfinite(z_vec) & np.isfinite(z_prev) &
                (np.abs(z_vec) > 0.01) & (np.abs(z_prev) > 0.01)
            )
            n_comparable = int(np.sum(valid_mask))
            
            if n_comparable > 5:  # Need enough data points
                signs_today = np.sign(z_vec[valid_mask])
                signs_prev = np.sign(z_prev[valid_mask])
                n_flipped = int(np.sum(signs_today != signs_prev))
                flip_rate = n_flipped / n_comparable
                
                if flip_rate >= self.thresholds.sign_flip_extreme_pct:
                    self._add_trigger(TriggerEvent(
                        reason=TriggerReason.SIGN_FLIP_EXTREME,
                        mode=RiskLatchMode.THROTTLE,
                        severity=flip_rate,
                        message=f"Extreme sign flip: {flip_rate:.0%} flipped ({n_flipped}/{n_comparable})",
                        details={"flip_rate": flip_rate, "n_flipped": n_flipped, "n_comparable": n_comparable},
                    ))
                    triggered_mode = max(triggered_mode, RiskLatchMode.THROTTLE)
        
        return triggered_mode
    
    # ─────────────────────────────────────────────────────────────────────────
    # Resolution
    # ─────────────────────────────────────────────────────────────────────────
    
    def resolve(self) -> RiskLatchState:
        """
        Resolve pending triggers into final state.
        
        Applies precedence: highest mode wins.
        For THROTTLE, exposure_scale is the minimum of all throttle scales.
        
        Returns:
            Resolved RiskLatchState
        """
        # Start with current state (may be latched)
        new_mode = RiskLatchMode.NORMAL
        primary_reason = None
        exposure_scale = 1.0
        max_gross_scale = 1.0
        max_name_scale = 1.0
        max_latch = 0
        
        # Handle manual release FIRST (before checking manual emergency)
        # This allows release to take effect immediately
        if self._manual_release_requested:
            self._manual_release_requested = False
            if not self._manual_emergency:
                self.state.latch_remaining_sessions = 0
                logger.info("[RiskLatch] Manual release completed")
        
        # If manual emergency is active, that overrides everything
        if self._manual_emergency:
            new_mode = RiskLatchMode.EMERGENCY_STOP
            primary_reason = TriggerReason.MANUAL_EMERGENCY
            max_latch = self.thresholds.emergency_latch_sessions
        
        # If currently latched at high mode, maintain it
        elif self.state.latch_remaining_sessions > 0 and self.state.mode >= RiskLatchMode.FLATTEN:
            new_mode = self.state.mode
            primary_reason = self.state.primary_reason
        
        # Otherwise, resolve from pending triggers
        else:
            for trigger in self._pending_triggers:
                if trigger.mode > new_mode:
                    new_mode = trigger.mode
                    primary_reason = trigger.reason
                
                if trigger.latch_sessions > max_latch:
                    max_latch = trigger.latch_sessions
                
                # Accumulate throttle scales
                if trigger.mode == RiskLatchMode.THROTTLE:
                    # Extract scale from details or compute from severity
                    scale = trigger.details.get("scale", 1.0 - 0.5 * trigger.severity)
                    exposure_scale = min(exposure_scale, scale)
        
        # Clamp scales
        exposure_scale = max(self.thresholds.throttle_min_scale, min(1.0, exposure_scale))
        
        # If not throttling, reset scales
        if new_mode < RiskLatchMode.THROTTLE:
            exposure_scale = 1.0
        
        # Track mode transitions
        if new_mode != self.state.mode:
            self.mode_history.append((self.current_session, new_mode, primary_reason.name if primary_reason else ""))
            
            if new_mode == RiskLatchMode.FLATTEN:
                self.state.total_flatten_count += 1
            elif new_mode == RiskLatchMode.EMERGENCY_STOP:
                self.state.total_emergency_count += 1
            
            self.state.mode_since_session = self.current_session
        
        # Update state
        self.state.mode = new_mode
        self.state.primary_reason = primary_reason
        self.state.all_triggers = self._pending_triggers.copy()
        self.state.exposure_scale = exposure_scale
        self.state.max_gross_scale = max_gross_scale if new_mode == RiskLatchMode.THROTTLE else 1.0
        self.state.max_name_scale = max_name_scale if new_mode == RiskLatchMode.THROTTLE else 1.0
        
        if max_latch > self.state.latch_remaining_sessions:
            self.state.latch_remaining_sessions = max_latch
        
        return self.state
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_benchmark_weights(self, weights: np.ndarray) -> None:
        """Set benchmark weights for SAFE_FALLBACK mode."""
        if len(weights) == self.n_assets:
            self.state.benchmark_weights = weights.copy()
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary."""
        s = self.state
        lines = [
            f"═══ RiskLatch State ═══",
            f"  Mode: {s.mode.name} (value={s.mode})",
        ]
        
        if s.primary_reason:
            lines.append(f"  Primary Reason: {s.primary_reason.name}")
        
        if s.mode == RiskLatchMode.THROTTLE:
            lines.append(f"  Exposure Scale: {s.exposure_scale:.2f}")
        
        if s.latch_remaining_sessions > 0:
            lines.append(f"  Latch Remaining: {s.latch_remaining_sessions} sessions")
        
        if s.cooldown_remaining_sessions > 0:
            lines.append(f"  Cooldown Remaining: {s.cooldown_remaining_sessions} sessions")
        
        if s.all_triggers:
            lines.append(f"  Active Triggers ({len(s.all_triggers)}):")
            for t in s.all_triggers[:5]:
                lines.append(f"    - {t.reason.name} → {t.mode.name}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "state": self.state.to_dict(),
            "current_session": self.current_session,
            "manual_emergency": self._manual_emergency,
            "mode_history_count": len(self.mode_history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────

def create_risk_latch(
    cfg: Dict[str, Any],
    n_assets: int,
) -> RiskLatch:
    """
    Create RiskLatch from config dict.
    
    Config keys (all optional, with defaults):
        phase2_risk_latch_max_dd_kill: float = 0.15
        phase2_risk_latch_max_vol_kill: float = 0.60  # High last-resort, prefer relative vol triggers
        phase2_risk_latch_max_turnover_kill: float = 3.0
        phase2_risk_latch_one_day_crash: float = -0.05
        phase2_risk_latch_dd_stress_start: float = 0.08
        phase2_risk_latch_vol_stress_start: float = 0.25
        phase2_risk_latch_cboe_panic: float = 25.0
        phase2_risk_latch_throttle_min: float = 0.1
        phase2_risk_latch_emergency_sessions: int = 5
        phase2_risk_latch_cooldown_sessions: int = 3
        
        # Fast crash triggers
        phase2_kill_1day_loss_pct: float = 0.03
        phase2_emergency_cooldown_sessions: int = 5
        phase2_gap_shock_mult: float = 4.0
        phase2_gap_shock_min_pct: float = 0.03  # 3% gaps catch real shocks in large caps
        phase2_gap_shock_n_symbols: int = 3
        phase2_gap_shock_portfolio_pct: float = 0.02
        phase2_gap_shock_mode: str = "throttle"
        phase2_gap_shock_throttle_scale: float = 0.3
        
        # Input health triggers (Point 1)
        phase2_risk_latch_hygiene_min_pct: float = 0.30
        phase2_risk_latch_role_ctx_stale_days: int = 2
        phase2_risk_latch_eligible_collapse_pct: float = 0.50
        
        # Output sanity triggers (Point 2)
        phase2_risk_latch_sigma_collapse_eps: float = 1e-6
        phase2_risk_latch_sigma_collapse_pct: float = 0.50
        phase2_risk_latch_z_saturation_pct: float = 0.20
        phase2_risk_latch_mu_constant_std_min: float = 0.001
        phase2_risk_latch_sign_flip_extreme_pct: float = 0.95
    """
    thresholds = RiskLatchThresholds(
        max_drawdown_kill=float(cfg.get("phase2_risk_latch_max_dd_kill", 0.15)),
        max_vol_kill=float(cfg.get("phase2_risk_latch_max_vol_kill", 0.60)),
        max_turnover_kill=float(cfg.get("phase2_risk_latch_max_turnover_kill", 3.0)),
        one_day_crash_pct=float(cfg.get("phase2_risk_latch_one_day_crash", -0.05)),
        max_correlation_hhi=float(cfg.get("phase2_risk_latch_max_corr_hhi", 0.5)),
        drawdown_stress_start=float(cfg.get("phase2_risk_latch_dd_stress_start", 0.08)),
        vol_stress_start=float(cfg.get("phase2_risk_latch_vol_stress_start", 0.25)),
        cboe_panic_threshold=float(cfg.get("phase2_risk_latch_cboe_panic", 25.0)),
        calibration_min=float(cfg.get("phase2_risk_latch_calib_min", 0.3)),
        calibration_collapse=float(cfg.get("phase2_risk_latch_calib_collapse", 0.1)),
        model_z_explosion=float(cfg.get("phase2_risk_latch_z_explosion", 10.0)),
        data_staleness_hours=float(cfg.get("phase2_risk_latch_data_stale_hours", 2.0)),
        emergency_latch_sessions=int(cfg.get("phase2_risk_latch_emergency_sessions", 5)),
        post_flatten_cooldown=int(cfg.get("phase2_risk_latch_cooldown_sessions", 3)),
        throttle_min_scale=float(cfg.get("phase2_risk_latch_throttle_min", 0.1)),
        throttle_max_scale=float(cfg.get("phase2_risk_latch_throttle_max", 1.0)),
        # Fast crash triggers
        one_day_loss_kill_pct=float(cfg.get("phase2_kill_1day_loss_pct", 0.03)),
        emergency_cooldown_sessions=int(cfg.get("phase2_emergency_cooldown_sessions", 5)),
        gap_shock_mult=float(cfg.get("phase2_gap_shock_mult", 4.0)),
        gap_shock_min_pct=float(cfg.get("phase2_gap_shock_min_pct", 0.03)),
        gap_shock_n_symbols=int(cfg.get("phase2_gap_shock_n_symbols", 3)),
        gap_shock_portfolio_pct=float(cfg.get("phase2_gap_shock_portfolio_pct", 0.02)),
        gap_shock_mode=str(cfg.get("phase2_gap_shock_mode", "throttle")),
        gap_shock_throttle_scale=float(cfg.get("phase2_gap_shock_throttle_scale", 0.3)),
        # Input health triggers (Point 1)
        hygiene_ok_min_pct=float(cfg.get("phase2_risk_latch_hygiene_min_pct", 0.30)),
        role_ctx_stale_days=int(cfg.get("phase2_risk_latch_role_ctx_stale_days", 2)),
        eligible_mask_collapse_pct=float(cfg.get("phase2_risk_latch_eligible_collapse_pct", 0.50)),
        # Output sanity triggers (Point 2)
        sigma_collapse_eps=float(cfg.get("phase2_risk_latch_sigma_collapse_eps", 1e-6)),
        sigma_collapse_pct=float(cfg.get("phase2_risk_latch_sigma_collapse_pct", 0.50)),
        z_saturation_clip=float(cfg.get("phase2_z_clip", 3.0)),
        z_saturation_pct=float(cfg.get("phase2_risk_latch_z_saturation_pct", 0.20)),
        mu_constant_std_min=float(cfg.get("phase2_risk_latch_mu_constant_std_min", 0.001)),
        sign_flip_extreme_pct=float(cfg.get("phase2_risk_latch_sign_flip_extreme_pct", 0.95)),
    )
    
    return RiskLatch(thresholds=thresholds, n_assets=n_assets)
