"""
Unit tests for Risk Latch SAFE_FALLBACK mode.

Tests that SAFE_FALLBACK properly holds last good weights instead of flattening.
"""

import numpy as np
import pytest

from src.stage_b_stateful.risk_latch import (
    RiskLatch,
    RiskLatchMode,
    RiskLatchState,
    RiskLatchThresholds,
    TriggerReason,
)


class TestSafeFallbackMode:
    """Test SAFE_FALLBACK mode behavior."""
    
    def test_should_flatten_excludes_safe_fallback(self):
        """SAFE_FALLBACK should NOT trigger should_flatten()."""
        state = RiskLatchState()
        
        # NORMAL: should not flatten
        state.mode = RiskLatchMode.NORMAL
        assert not state.should_flatten()
        
        # THROTTLE: should not flatten
        state.mode = RiskLatchMode.THROTTLE
        assert not state.should_flatten()
        
        # FLATTEN: should flatten
        state.mode = RiskLatchMode.FLATTEN
        assert state.should_flatten()
        
        # SAFE_FALLBACK: should NOT flatten (this is the fix)
        state.mode = RiskLatchMode.SAFE_FALLBACK
        assert not state.should_flatten()
        assert state.should_hold_fallback()
        
        # EMERGENCY_STOP: should flatten
        state.mode = RiskLatchMode.EMERGENCY_STOP
        assert state.should_flatten()
    
    def test_get_effective_weights_holds_last_good(self):
        """SAFE_FALLBACK should return last_good_weights."""
        n_assets = 5
        target_weights = np.array([0.2, -0.1, 0.15, -0.05, 0.0])
        last_good = np.array([0.1, 0.2, -0.1, 0.05, -0.15])
        
        state = RiskLatchState()
        state.mode = RiskLatchMode.SAFE_FALLBACK
        state.last_good_weights = last_good.copy()
        
        effective = state.get_effective_weights(target_weights, n_assets)
        
        # Should return last good weights, NOT target weights
        np.testing.assert_array_equal(effective, last_good)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, effective, target_weights)
    
    def test_get_effective_weights_uses_benchmark_fallback(self):
        """SAFE_FALLBACK uses benchmark if last_good unavailable."""
        n_assets = 5
        target_weights = np.array([0.2, -0.1, 0.15, -0.05, 0.0])
        benchmark = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weight
        
        state = RiskLatchState()
        state.mode = RiskLatchMode.SAFE_FALLBACK
        state.last_good_weights = None  # No last good available
        state.benchmark_weights = benchmark.copy()
        
        effective = state.get_effective_weights(target_weights, n_assets)
        
        # Should return benchmark weights
        np.testing.assert_array_equal(effective, benchmark)
    
    def test_get_effective_weights_flattens_if_no_fallback(self):
        """SAFE_FALLBACK flattens if no last_good or benchmark available."""
        n_assets = 5
        target_weights = np.array([0.2, -0.1, 0.15, -0.05, 0.0])
        
        state = RiskLatchState()
        state.mode = RiskLatchMode.SAFE_FALLBACK
        state.last_good_weights = None
        state.benchmark_weights = None
        
        effective = state.get_effective_weights(target_weights, n_assets)
        
        # Should flatten (no other option)
        np.testing.assert_array_equal(effective, np.zeros(n_assets))
    
    def test_end_session_stores_good_weights(self):
        """end_session should store good weights for later fallback."""
        latch = RiskLatch(n_assets=3)
        good_weights = np.array([0.3, -0.2, 0.1])
        
        latch.end_session(good_weights)
        
        np.testing.assert_array_equal(latch.state.last_good_weights, good_weights)
    
    def test_end_session_rejects_nan_weights(self):
        """end_session should not store weights with NaN."""
        latch = RiskLatch(n_assets=3)
        bad_weights = np.array([0.3, np.nan, 0.1])
        
        latch.end_session(bad_weights)
        
        # Should not store bad weights
        assert latch.state.last_good_weights is None
    
    def test_safe_fallback_no_turnover(self):
        """SAFE_FALLBACK should produce zero turnover if holding same weights."""
        n_assets = 3
        last_good = np.array([0.3, -0.2, 0.1])
        
        state = RiskLatchState()
        state.mode = RiskLatchMode.SAFE_FALLBACK
        state.last_good_weights = last_good.copy()
        
        # Get effective weights
        effective = state.get_effective_weights(np.zeros(n_assets), n_assets)
        
        # Compute turnover vs last good
        turnover = np.sum(np.abs(effective - last_good))
        
        assert turnover == 0.0, "Holding last good should have zero turnover"
    
    def test_flatten_vs_safe_fallback_turnover(self):
        """FLATTEN should have higher turnover than SAFE_FALLBACK."""
        n_assets = 3
        last_good = np.array([0.3, -0.2, 0.1])
        
        # FLATTEN case
        state_flatten = RiskLatchState()
        state_flatten.mode = RiskLatchMode.FLATTEN
        w_flatten = state_flatten.get_effective_weights(np.zeros(n_assets), n_assets)
        turnover_flatten = np.sum(np.abs(w_flatten - last_good))
        
        # SAFE_FALLBACK case
        state_fallback = RiskLatchState()
        state_fallback.mode = RiskLatchMode.SAFE_FALLBACK
        state_fallback.last_good_weights = last_good.copy()
        w_fallback = state_fallback.get_effective_weights(np.zeros(n_assets), n_assets)
        turnover_fallback = np.sum(np.abs(w_fallback - last_good))
        
        assert turnover_flatten > turnover_fallback, "FLATTEN should have higher turnover than SAFE_FALLBACK"
        assert turnover_fallback == 0.0, "SAFE_FALLBACK should have zero turnover"
        assert turnover_flatten == np.sum(np.abs(last_good)), "FLATTEN turnover = |last_good|"


class TestSafeFallbackPersistence:
    """Test that SAFE_FALLBACK persists until conditions improve."""
    
    def test_safe_fallback_does_not_decrement_latch(self):
        """SAFE_FALLBACK should not decrement latch counter."""
        latch = RiskLatch(n_assets=3)
        latch.state.mode = RiskLatchMode.SAFE_FALLBACK
        latch.state.latch_remaining_sessions = 5
        
        # Begin session
        latch.begin_session(1)
        
        # Latch counter should NOT decrement for SAFE_FALLBACK
        assert latch.state.latch_remaining_sessions == 5
    
    def test_flatten_decrements_latch(self):
        """FLATTEN should decrement latch counter."""
        latch = RiskLatch(n_assets=3)
        latch.state.mode = RiskLatchMode.FLATTEN
        latch.state.latch_remaining_sessions = 5
        
        # Begin session
        latch.begin_session(1)
        
        # Latch counter SHOULD decrement for FLATTEN
        assert latch.state.latch_remaining_sessions == 4
    
    def test_emergency_decrements_latch(self):
        """EMERGENCY_STOP should decrement latch counter."""
        latch = RiskLatch(n_assets=3)
        latch.state.mode = RiskLatchMode.EMERGENCY_STOP
        latch.state.latch_remaining_sessions = 5
        
        # Begin session
        latch.begin_session(1)
        
        # Latch counter SHOULD decrement for EMERGENCY
        assert latch.state.latch_remaining_sessions == 4


class TestSafeFallbackTriggers:
    """Test conditions that trigger SAFE_FALLBACK."""
    
    def test_input_health_fail_triggers_safe_fallback(self):
        """check_input_health with data failure should trigger SAFE_FALLBACK."""
        latch = RiskLatch(n_assets=5)
        latch.begin_session(0)
        
        # Simulate data failure: hygiene_ok very low
        hygiene_ok_pct = 0.15  # < 0.30 threshold
        
        mode = latch.check_input_health(
            hygiene_ok_pct=hygiene_ok_pct,
            role_ctx_available=True,
            role_ctx_days_stale=0,
            eligible_count_today=5,
            eligible_count_yesterday=5,
        )
        
        assert mode == RiskLatchMode.SAFE_FALLBACK
        
        state = latch.resolve()
        assert state.mode == RiskLatchMode.SAFE_FALLBACK
        assert state.should_hold_fallback()
        assert not state.should_flatten()
    
    def test_output_sanity_sigma_collapse_triggers_safe_fallback(self):
        """Sigma collapse should trigger SAFE_FALLBACK."""
        thresholds = RiskLatchThresholds()
        thresholds.sigma_collapse_pct = 0.50
        thresholds.sigma_collapse_eps = 1e-6
        
        latch = RiskLatch(thresholds=thresholds, n_assets=5)
        latch.begin_session(0)
        
        # Simulate sigma collapse: 80% of sigmas near zero
        mu = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        sigma = np.array([1e-7, 1e-7, 1e-7, 1e-7, 0.02])  # 4/5 = 80% collapsed
        z = np.array([1.0, 2.0, -1.0, 1.5, -0.5])
        
        mode = latch.check_output_sanity(
            mu_vec=mu,
            sigma_vec=sigma,
            z_vec=z,
            z_prev=None,
            z_clip=4.0,
        )
        
        assert mode == RiskLatchMode.SAFE_FALLBACK
        
        state = latch.resolve()
        assert state.mode == RiskLatchMode.SAFE_FALLBACK


class TestSafeFallbackIntegration:
    """Integration test simulating Phase-2 loop with SAFE_FALLBACK."""
    
    def test_safe_fallback_preserves_position_vs_flatten(self):
        """
        Simulate 3-day period with data failure.
        SAFE_FALLBACK should preserve positions, FLATTEN would churn.
        """
        n_assets = 3
        latch = RiskLatch(n_assets=n_assets)
        
        # Day 0: Normal operation, establish good weights
        latch.begin_session(0)
        state_0 = latch.resolve()
        good_weights = np.array([0.3, -0.2, 0.1])
        latch.end_session(good_weights)
        
        assert state_0.mode == RiskLatchMode.NORMAL
        np.testing.assert_array_equal(latch.state.last_good_weights, good_weights)
        
        # Day 1-3: Data failure triggers SAFE_FALLBACK
        turnover_safe_fallback = 0.0
        
        for day in [1, 2, 3]:
            latch.begin_session(day)
            
            # Trigger SAFE_FALLBACK via input health fail
            latch.check_input_health(
                hygiene_ok_pct=0.15,  # Below 0.30 threshold
                role_ctx_available=True,
                role_ctx_days_stale=0,
                eligible_count_today=3,
                eligible_count_yesterday=3,
            )
            
            state = latch.resolve()
            assert state.mode == RiskLatchMode.SAFE_FALLBACK
            
            # Get effective weights
            target = np.zeros(n_assets)  # Optimizer would return this
            effective = state.get_effective_weights(target, n_assets)
            
            # Should hold last good weights
            np.testing.assert_array_equal(effective, good_weights)
            
            # Compute turnover
            if day == 1:
                prev_w = good_weights
            else:
                prev_w = effective
            
            turnover_safe_fallback += np.sum(np.abs(effective - prev_w))
            
            latch.end_session(None)  # Don't update last_good during fallback
        
        # SAFE_FALLBACK should have zero turnover (held position)
        assert turnover_safe_fallback == 0.0
        
        # Compare to FLATTEN behavior (would have churned)
        # If we had flattened on day 1, then re-entered on day 4:
        flatten_turnover = np.sum(np.abs(good_weights))  # Day 1: exit
        flatten_turnover += np.sum(np.abs(good_weights))  # Day 4: re-enter
        
        assert turnover_safe_fallback < flatten_turnover, \
            "SAFE_FALLBACK should have lower turnover than FLATTEN"
    
    def test_to_dict_includes_fallback_info(self):
        """State serialization should include fallback info."""
        state = RiskLatchState()
        state.mode = RiskLatchMode.SAFE_FALLBACK
        state.last_good_weights = np.array([0.1, 0.2, 0.3])
        
        state_dict = state.to_dict()
        
        assert state_dict["mode"] == "SAFE_FALLBACK"
        assert state_dict["should_hold_fallback"] is True
        assert state_dict["has_fallback_weights"] is True
        assert state_dict["should_flatten"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
