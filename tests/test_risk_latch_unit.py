"""
Unit Tests for RiskLatch State Machine.

These are deterministic tests that verify the risk control behavior
in isolation. Run with: pytest tests/test_risk_latch_unit.py -v

Copyright 2024-2026. All rights reserved.
"""

import numpy as np
import pytest
from typing import Dict, Any

from src.stage_b_stateful.risk_latch import (
    RiskLatch,
    RiskLatchMode,
    RiskLatchThresholds,
    TriggerReason,
    create_risk_latch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_thresholds() -> RiskLatchThresholds:
    """Default thresholds for testing."""
    return RiskLatchThresholds()


@pytest.fixture
def default_latch(default_thresholds: RiskLatchThresholds) -> RiskLatch:
    """Default RiskLatch with 10 assets."""
    return RiskLatch(thresholds=default_thresholds, n_assets=10)


@pytest.fixture
def cfg_dict() -> Dict[str, Any]:
    """Config dict for create_risk_latch."""
    return {
        "phase2_risk_latch_max_dd_kill": 0.15,
        "phase2_risk_latch_max_vol_kill": 0.60,  # High last-resort threshold
        "phase2_kill_1day_loss_pct": 0.03,
        "phase2_risk_latch_hygiene_min_pct": 0.30,
        "phase2_risk_latch_sigma_collapse_pct": 0.50,
        "phase2_risk_latch_z_saturation_pct": 0.20,
        "phase2_risk_latch_sign_flip_extreme_pct": 0.95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test: Emergency Stop from Intraday Monitor
# ─────────────────────────────────────────────────────────────────────────────

class TestEmergencyStop:
    """Test EMERGENCY_STOP triggers and behavior."""
    
    def test_manual_emergency_triggers_emergency_stop(self, default_latch: RiskLatch):
        """Intraday JSON says emergency → RiskLatch=EMERGENCY_STOP."""
        default_latch.begin_session(0)
        
        # Trigger manual emergency (simulating intraday monitor)
        default_latch.trigger_manual_emergency("Intraday monitor EMERGENCY")
        
        state = default_latch.resolve()
        
        assert state.mode == RiskLatchMode.EMERGENCY_STOP
        assert state.primary_reason == TriggerReason.MANUAL_EMERGENCY
        assert state.should_flatten() is True
    
    def test_emergency_stop_forces_zero_weights(self, default_latch: RiskLatch):
        """EMERGENCY_STOP → w_target == 0."""
        default_latch.begin_session(0)
        default_latch.trigger_manual_emergency("Test emergency")
        state = default_latch.resolve()
        
        target_weights = np.array([0.1, 0.2, -0.1, 0.15, -0.05, 0.1, 0.0, 0.0, 0.0, 0.0])
        effective = state.get_effective_weights(target_weights, n_assets=10)
        
        assert np.allclose(effective, 0.0)
    
    def test_emergency_latches_for_n_sessions(self, default_latch: RiskLatch):
        """EMERGENCY_STOP latches for configured sessions."""
        latch_sessions = default_latch.thresholds.emergency_latch_sessions
        
        default_latch.begin_session(0)
        default_latch.trigger_manual_emergency("Test latch")
        state = default_latch.resolve()
        default_latch.end_session(None)
        
        # Should remain latched for N sessions
        for session in range(1, latch_sessions):
            default_latch.begin_session(session)
            state = default_latch.resolve()
            
            assert state.mode == RiskLatchMode.EMERGENCY_STOP, f"Session {session} should still be latched"
            assert state.latch_remaining_sessions > 0
            
            default_latch.end_session(None)
        
        # After latch expires, should release (if manual emergency is released)
        default_latch.release_manual_emergency()
        default_latch.begin_session(latch_sessions)
        state = default_latch.resolve()
        
        # Should be in cooldown now
        assert state.mode < RiskLatchMode.EMERGENCY_STOP


# ─────────────────────────────────────────────────────────────────────────────
# Test: Data Failure → Safe Fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFailure:
    """Test DATA_FAILURE triggers SAFE_FALLBACK."""
    
    def test_stale_data_triggers_safe_fallback(self, default_latch: RiskLatch):
        """Data staleness > threshold → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        stale_hours = default_latch.thresholds.data_staleness_hours + 1.0
        mode = default_latch.check_data_health(data_staleness_hours=stale_hours)
        
        assert mode == RiskLatchMode.SAFE_FALLBACK
        
        state = default_latch.resolve()
        assert state.mode == RiskLatchMode.SAFE_FALLBACK
        assert state.primary_reason == TriggerReason.DATA_FAILURE
    
    def test_low_data_coverage_triggers_safe_fallback(self, default_latch: RiskLatch):
        """Data coverage < 50% → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        mode = default_latch.check_data_health(data_coverage=0.3)
        
        assert mode == RiskLatchMode.SAFE_FALLBACK
    
    def test_safe_fallback_uses_last_good_weights(self, default_latch: RiskLatch):
        """SAFE_FALLBACK returns last_good_weights if available."""
        last_good = np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Session 0: Normal, store good weights
        default_latch.begin_session(0)
        default_latch.end_session(good_weights=last_good)
        
        # Session 1: Data failure
        default_latch.begin_session(1)
        default_latch.check_data_health(data_staleness_hours=10.0)
        state = default_latch.resolve()
        
        target = np.ones(10) * 0.2
        effective = state.get_effective_weights(target, n_assets=10)
        
        assert np.allclose(effective, last_good)
    
    def test_safe_fallback_uses_benchmark_if_no_last_good(self, default_latch: RiskLatch):
        """SAFE_FALLBACK uses benchmark_weights if last_good unavailable."""
        benchmark = np.ones(10) * 0.1  # Equal weight benchmark
        default_latch.set_benchmark_weights(benchmark)
        
        default_latch.begin_session(0)
        default_latch.check_data_health(data_staleness_hours=10.0)
        state = default_latch.resolve()
        
        target = np.ones(10) * 0.2
        effective = state.get_effective_weights(target, n_assets=10)
        
        assert np.allclose(effective, benchmark)
    
    def test_safe_fallback_flattens_if_no_fallback_available(self, default_latch: RiskLatch):
        """SAFE_FALLBACK → flat if no last_good or benchmark."""
        default_latch.begin_session(0)
        default_latch.check_data_health(data_staleness_hours=10.0)
        state = default_latch.resolve()
        
        target = np.ones(10) * 0.2
        effective = state.get_effective_weights(target, n_assets=10)
        
        assert np.allclose(effective, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Input Health Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestInputHealth:
    """Test input health checks (Point 1)."""
    
    def test_hygiene_below_threshold_triggers_safe_fallback(self, default_latch: RiskLatch):
        """hygiene_ok % < threshold → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        hygiene_pct = default_latch.thresholds.hygiene_ok_min_pct - 0.1
        mode = default_latch.check_input_health(hygiene_ok_pct=hygiene_pct)
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_role_ctx_missing_triggers_safe_fallback(self, default_latch: RiskLatch):
        """role_ctx not available → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        mode = default_latch.check_input_health(
            hygiene_ok_pct=1.0,
            role_ctx_available=False,
        )
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_role_ctx_stale_triggers_safe_fallback(self, default_latch: RiskLatch):
        """role_ctx stale > N days → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        stale_days = default_latch.thresholds.role_ctx_stale_days + 1
        mode = default_latch.check_input_health(
            hygiene_ok_pct=1.0,
            role_ctx_available=True,
            role_ctx_days_stale=stale_days,
        )
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_eligible_mask_collapse_triggers_safe_fallback(self, default_latch: RiskLatch):
        """80% drop in eligible symbols → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        mode = default_latch.check_input_health(
            hygiene_ok_pct=1.0,
            role_ctx_available=True,
            eligible_count_today=10,
            eligible_count_yesterday=100,  # 90% drop
        )
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_healthy_input_returns_normal(self, default_latch: RiskLatch):
        """Healthy inputs → NORMAL."""
        default_latch.begin_session(0)
        
        mode = default_latch.check_input_health(
            hygiene_ok_pct=0.9,
            role_ctx_available=True,
            role_ctx_days_stale=0,
            eligible_count_today=100,
            eligible_count_yesterday=100,
        )
        
        assert mode == RiskLatchMode.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# Test: Output Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputSanity:
    """Test output sanity checks (Point 2)."""
    
    def test_sigma_collapse_triggers_safe_fallback(self, default_latch: RiskLatch):
        """z saturation (sigma collapse) → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        # 60% of sigmas collapsed (> 50% threshold)
        sigma_vec = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 0.02, 0.02, 0.02, 0.02])
        mu_vec = np.random.randn(10) * 0.01
        z_vec = np.random.randn(10)
        
        mode = default_latch.check_output_sanity(
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            z_vec=z_vec,
        )
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_z_saturation_triggers_throttle(self, default_latch: RiskLatch):
        """>30% z's at clip boundary → THROTTLE."""
        default_latch.begin_session(0)
        
        z_clip = 3.0
        # 40% at clip boundary
        z_vec = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        sigma_vec = np.ones(10) * 0.02
        mu_vec = np.random.randn(10) * 0.01
        
        mode = default_latch.check_output_sanity(
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            z_vec=z_vec,
            z_clip=z_clip,
        )
        
        assert mode >= RiskLatchMode.THROTTLE
    
    def test_mu_constant_triggers_safe_fallback(self, default_latch: RiskLatch):
        """mu near-constant across symbols → SAFE_FALLBACK."""
        default_latch.begin_session(0)
        
        # All mu values identical
        mu_vec = np.ones(10) * 0.001
        sigma_vec = np.ones(10) * 0.02
        z_vec = np.random.randn(10)
        
        mode = default_latch.check_output_sanity(
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            z_vec=z_vec,
        )
        
        assert mode >= RiskLatchMode.SAFE_FALLBACK
    
    def test_extreme_sign_flip_triggers_throttle(self, default_latch: RiskLatch):
        """>90% sign flip → THROTTLE."""
        default_latch.begin_session(0)
        
        # Previous z all positive
        z_prev = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # Current z all negative (100% flip)
        z_vec = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        
        mu_vec = np.random.randn(10) * 0.01
        sigma_vec = np.ones(10) * 0.02
        
        mode = default_latch.check_output_sanity(
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            z_vec=z_vec,
            z_prev=z_prev,
        )
        
        assert mode >= RiskLatchMode.THROTTLE
    
    def test_healthy_output_returns_normal(self, default_latch: RiskLatch):
        """Healthy outputs → NORMAL."""
        default_latch.begin_session(0)
        
        mu_vec = np.random.randn(10) * 0.01
        sigma_vec = np.ones(10) * 0.02
        z_vec = np.random.randn(10) * 0.5  # Well within clip
        z_prev = z_vec + np.random.randn(10) * 0.1  # Small changes
        
        mode = default_latch.check_output_sanity(
            mu_vec=mu_vec,
            sigma_vec=sigma_vec,
            z_vec=z_vec,
            z_prev=z_prev,
        )
        
        assert mode == RiskLatchMode.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# Test: Stress/Throttle Triggers
# ─────────────────────────────────────────────────────────────────────────────

class TestThrottleTriggers:
    """Test throttle triggers."""
    
    def test_drawdown_stress_triggers_throttle(self, default_latch: RiskLatch):
        """Drawdown approaching kill → THROTTLE with scale."""
        default_latch.begin_session(0)
        
        # DD at stress threshold (halfway between stress_start and kill)
        dd = (default_latch.thresholds.drawdown_stress_start + 
              default_latch.thresholds.max_drawdown_kill) / 2
        
        kill_triggered = default_latch.check_drawdown(dd)
        state = default_latch.resolve()
        
        assert kill_triggered is False
        assert state.mode == RiskLatchMode.THROTTLE
        assert 0.0 < state.exposure_scale < 1.0
    
    def test_vol_stress_triggers_throttle(self, default_latch: RiskLatch):
        """Vol approaching kill → THROTTLE with scale."""
        default_latch.begin_session(0)
        
        # Vol at stress threshold
        vol = (default_latch.thresholds.vol_stress_start + 
               default_latch.thresholds.max_vol_kill) / 2
        
        kill_triggered = default_latch.check_volatility(vol)
        state = default_latch.resolve()
        
        assert kill_triggered is False
        assert state.mode == RiskLatchMode.THROTTLE
        assert 0.0 < state.exposure_scale < 1.0
    
    def test_one_day_loss_triggers_emergency(self, default_latch: RiskLatch):
        """1-day loss > cap → EMERGENCY_STOP."""
        default_latch.begin_session(0)
        
        loss_pct = default_latch.thresholds.one_day_loss_kill_pct + 0.01
        
        emergency = default_latch.check_one_day_loss_kill(
            pnl_net_today=-loss_pct,
            equity_today=1.0 - loss_pct,
            equity_prev=1.0,
        )
        
        assert emergency is True
        
        state = default_latch.resolve()
        assert state.mode == RiskLatchMode.EMERGENCY_STOP


# ─────────────────────────────────────────────────────────────────────────────
# Test: NaN/Inf Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestNanInfHandling:
    """Test NaN/Inf kill-switch."""
    
    def test_nan_triggers_flatten(self, default_latch: RiskLatch):
        """NaN in predictions → FLATTEN."""
        default_latch.begin_session(0)
        
        mu_with_nan = np.array([0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        
        triggered = default_latch.check_nan_inf([mu_with_nan], ["mu"])
        
        assert triggered is True
        
        state = default_latch.resolve()
        assert state.mode >= RiskLatchMode.FLATTEN
    
    def test_inf_triggers_flatten(self, default_latch: RiskLatch):
        """Inf in weights → FLATTEN."""
        default_latch.begin_session(0)
        
        w_with_inf = np.array([0.1, np.inf, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        triggered = default_latch.check_nan_inf([w_with_inf], ["weights"])
        
        assert triggered is True
        
        state = default_latch.resolve()
        assert state.mode >= RiskLatchMode.FLATTEN


# ─────────────────────────────────────────────────────────────────────────────
# Test: Factory Function
# ─────────────────────────────────────────────────────────────────────────────

class TestFactory:
    """Test create_risk_latch factory."""
    
    def test_create_from_config(self, cfg_dict: Dict[str, Any]):
        """create_risk_latch respects config values."""
        latch = create_risk_latch(cfg_dict, n_assets=10)
        
        assert latch.thresholds.max_drawdown_kill == 0.15
        assert latch.thresholds.max_vol_kill == 0.60  # Updated to match fixture
        assert latch.thresholds.one_day_loss_kill_pct == 0.03
        assert latch.thresholds.hygiene_ok_min_pct == 0.30
        assert latch.n_assets == 10
    
    def test_create_with_defaults(self):
        """create_risk_latch uses defaults for missing config."""
        latch = create_risk_latch({}, n_assets=5)
        
        assert latch.thresholds.max_drawdown_kill == 0.15  # Default
        assert latch.n_assets == 5


# ─────────────────────────────────────────────────────────────────────────────
# Test: Precedence
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecedence:
    """Test mode precedence (higher modes override lower)."""
    
    def test_emergency_overrides_all(self, default_latch: RiskLatch):
        """EMERGENCY_STOP overrides all other triggers."""
        default_latch.begin_session(0)
        
        # Trigger throttle (lower precedence)
        default_latch.check_drawdown(0.10)  # Stress
        
        # Trigger emergency (highest precedence)
        default_latch.trigger_manual_emergency("Override test")
        
        state = default_latch.resolve()
        
        assert state.mode == RiskLatchMode.EMERGENCY_STOP
    
    def test_flatten_overrides_throttle(self, default_latch: RiskLatch):
        """FLATTEN overrides THROTTLE."""
        default_latch.begin_session(0)
        
        # Trigger throttle
        default_latch.check_drawdown(0.10)
        
        # Trigger flatten (higher precedence)
        default_latch.check_nan_inf([np.array([np.nan])], ["test"])
        
        state = default_latch.resolve()
        
        assert state.mode >= RiskLatchMode.FLATTEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
