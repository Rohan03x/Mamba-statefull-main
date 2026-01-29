"""
Comprehensive tests for Gap 4 fix: Partial fill participation constraint.

Tests verify that:
1. Partial fills are applied correctly when trades exceed ADV participation limits
2. Residuals are computed correctly
3. Multi-day scenarios work correctly (no double-counting)
4. Edge cases are handled properly
"""

import numpy as np
import pytest
from dataclasses import dataclass, field

from src.stage_b_stateful.phase2_stateful import (
    _apply_liquidity_participation_constraint,
    ExecutionContext,
)


class TestPartialFillBasics:
    """Test basic partial fill functionality."""
    
    def test_no_constraint_when_within_limits(self):
        """When trade is within ADV limits, full execution should occur."""
        n = 5
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e7, 1e7, 1e7, 1e7, 1e7], dtype=float)
        
        target_w = np.array([0.10, 0.05, -0.05, 0.00, 0.08], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # With $10M ADV and 10% participation, max trade = $1M
        # All trades are < $1M, so full execution
        np.testing.assert_array_almost_equal(executed_w, target_w)
        np.testing.assert_array_almost_equal(residual, np.zeros(n))
    
    def test_partial_fill_when_exceeds_limits(self):
        """When trade exceeds ADV limits, partial fill should occur."""
        n = 3
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        # Asset 0: $1M ADV (very illiquid)
        # Asset 1: $10M ADV (liquid)
        # Asset 2: $5M ADV (medium)
        exec_ctx.adv_usd = np.array([1e6, 1e7, 5e6], dtype=float)
        
        target_w = np.array([0.15, 0.10, 0.08], dtype=float)  # Want 15%, 10%, 8%
        prev_w = np.zeros(n, dtype=float)
        
        # Max participation = 10%
        # Asset 0: max trade = 10% * $1M = $100K = 0.10 weight, but want 0.15
        # Asset 1: max trade = 10% * $10M = $1M = 1.00 weight, want 0.10 → OK
        # Asset 2: max trade = 10% * $5M = $500K = 0.50 weight, want 0.08 → OK
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # Asset 0 should be capped at 0.10
        assert abs(executed_w[0] - 0.10) < 1e-9, f"Expected 0.10, got {executed_w[0]}"
        assert abs(residual[0] - 0.05) < 1e-9, f"Expected 0.05 residual, got {residual[0]}"
        
        # Assets 1 and 2 should execute fully
        assert abs(executed_w[1] - 0.10) < 1e-9
        assert abs(residual[1]) < 1e-9
        assert abs(executed_w[2] - 0.08) < 1e-9
        assert abs(residual[2]) < 1e-9
    
    def test_partial_fill_sell_order(self):
        """Partial fills should work for sell orders (negative weights)."""
        n = 2
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6, 5e6], dtype=float)
        
        # Currently hold 20%, want to reduce to 0% (sell 20%)
        target_w = np.array([0.00, 0.00], dtype=float)
        prev_w = np.array([0.20, 0.15], dtype=float)
        
        # Asset 0: want to sell $200K, max = $100K → partial fill
        # Asset 1: want to sell $150K, max = $500K → full fill
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # Asset 0: can only sell 10% → executed = 20% - 10% = 10%
        assert abs(executed_w[0] - 0.10) < 1e-9
        assert abs(residual[0] - (-0.10)) < 1e-9  # Still want to sell 10% more
        
        # Asset 1: full sell
        assert abs(executed_w[1] - 0.00) < 1e-9
        assert abs(residual[1]) < 1e-9


class TestMultiDayScenarios:
    """Test multi-day partial fill scenarios (Gap 4 bug focus)."""
    
    def test_two_day_accumulation_constant_target(self):
        """
        Test that positions accumulate correctly over multiple days.
        
        Scenario: Want 15% position but can only trade 10% per day due to ADV.
        Day 1: 0% → 10% (execute 10%, residual 5%)
        Day 2: 10% → 15% (execute 5% more to reach target)
        
        This tests the Gap 4 fix: without the fix, day 2 would try to trade 10%
        (double-counting), but with the fix it should only trade 5%.
        """
        n = 1
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6], dtype=float)  # $1M ADV
        
        # Day 1: Want 15% from 0%
        target_w_day1 = np.array([0.15], dtype=float)
        prev_w_day1 = np.array([0.00], dtype=float)
        
        executed_w_day1, residual_day1 = _apply_liquidity_participation_constraint(
            target_w_day1, prev_w_day1, exec_ctx, max_participation_rate=0.10
        )
        
        # Day 1: Can only execute 10% (max = 10% * $1M = $100K)
        assert abs(executed_w_day1[0] - 0.10) < 1e-9, f"Day 1 executed: {executed_w_day1[0]}"
        assert abs(residual_day1[0] - 0.05) < 1e-9, f"Day 1 residual: {residual_day1[0]}"
        
        # Store residual (this happens in main loop)
        exec_ctx.partial_fill_residual = residual_day1
        
        # Day 2: Still want 15%, but now starting from 10%
        target_w_day2 = np.array([0.15], dtype=float)  # Same target
        prev_w_day2 = executed_w_day1.copy()  # Start from day 1 execution
        
        executed_w_day2, residual_day2 = _apply_liquidity_participation_constraint(
            target_w_day2, prev_w_day2, exec_ctx, max_participation_rate=0.10
        )
        
        # Day 2: Should try to trade 5% (from 10% to 15%)
        # With the bug fix, delta = 15% - 10% = 5% (correct)
        # Without fix, delta = 15% - 10% + 5% = 10% (wrong!)
        assert abs(executed_w_day2[0] - 0.15) < 1e-9, \
            f"Day 2 executed: {executed_w_day2[0]}, expected 0.15"
        assert abs(residual_day2[0]) < 1e-9, \
            f"Day 2 residual: {residual_day2[0]}, expected 0.00 (target reached)"
    
    def test_three_day_accumulation_very_illiquid(self):
        """Test accumulation over 3 days for very illiquid asset."""
        n = 1
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=10_000_000.0,  # $10M capital
        )
        exec_ctx.adv_usd = np.array([2e6], dtype=float)  # $2M ADV
        
        # Want 30% position = $3M, but max trade per day = 10% * $2M = $200K = 2%
        target_w = np.array([0.30], dtype=float)
        
        # Day 1
        prev_w_day1 = np.array([0.00], dtype=float)
        executed_w_day1, residual_day1 = _apply_liquidity_participation_constraint(
            target_w, prev_w_day1, exec_ctx, max_participation_rate=0.10
        )
        
        # Max trade = 2%, so execute from 0% to 2%
        assert abs(executed_w_day1[0] - 0.02) < 1e-9
        assert abs(residual_day1[0] - 0.28) < 1e-9
        
        exec_ctx.partial_fill_residual = residual_day1
        
        # Day 2
        prev_w_day2 = executed_w_day1.copy()
        executed_w_day2, residual_day2 = _apply_liquidity_participation_constraint(
            target_w, prev_w_day2, exec_ctx, max_participation_rate=0.10
        )
        
        # Delta = 30% - 2% = 28%, cap at 2% → execute to 4%
        assert abs(executed_w_day2[0] - 0.04) < 1e-9
        assert abs(residual_day2[0] - 0.26) < 1e-9
        
        exec_ctx.partial_fill_residual = residual_day2
        
        # Day 3
        prev_w_day3 = executed_w_day2.copy()
        executed_w_day3, residual_day3 = _apply_liquidity_participation_constraint(
            target_w, prev_w_day3, exec_ctx, max_participation_rate=0.10
        )
        
        # Delta = 30% - 4% = 26%, cap at 2% → execute to 6%
        assert abs(executed_w_day3[0] - 0.06) < 1e-9
        assert abs(residual_day3[0] - 0.24) < 1e-9
    
    def test_changing_target_over_days(self):
        """
        Test that changing targets work correctly.
        
        Day 1: Want 20%, execute 10%
        Day 2: Want 15% (changed!), should trade -5% (from 10% to 15%, not 15% more)
        """
        n = 1
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6], dtype=float)
        
        # Day 1: Want 20%
        target_w_day1 = np.array([0.20], dtype=float)
        prev_w_day1 = np.array([0.00], dtype=float)
        
        executed_w_day1, residual_day1 = _apply_liquidity_participation_constraint(
            target_w_day1, prev_w_day1, exec_ctx, max_participation_rate=0.10
        )
        
        # Max = 10%, execute to 10%
        assert abs(executed_w_day1[0] - 0.10) < 1e-9
        assert abs(residual_day1[0] - 0.10) < 1e-9
        
        exec_ctx.partial_fill_residual = residual_day1
        
        # Day 2: Target changes to 15%
        target_w_day2 = np.array([0.15], dtype=float)
        prev_w_day2 = executed_w_day1.copy()  # 10%
        
        executed_w_day2, residual_day2 = _apply_liquidity_participation_constraint(
            target_w_day2, prev_w_day2, exec_ctx, max_participation_rate=0.10
        )
        
        # Delta = 15% - 10% = 5% (NOT 15% - 10% + 10% = 15%!)
        # Can execute full 5% since it's under limit
        assert abs(executed_w_day2[0] - 0.15) < 1e-9, \
            f"Expected 0.15, got {executed_w_day2[0]} (bug would give 0.20 or higher)"
        assert abs(residual_day2[0]) < 1e-9
    
    def test_target_decreases_over_days(self):
        """
        Test selling down a position over multiple days.
        
        Day 1: Have 30%, want 20%, can sell 10% → execute to 20%
        Day 2: Have 20%, want 10%, can sell 10% → execute to 10%
        """
        n = 1
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6], dtype=float)
        
        # Day 1: Reduce from 30% to 20%
        target_w_day1 = np.array([0.20], dtype=float)
        prev_w_day1 = np.array([0.30], dtype=float)
        
        executed_w_day1, residual_day1 = _apply_liquidity_participation_constraint(
            target_w_day1, prev_w_day1, exec_ctx, max_participation_rate=0.10
        )
        
        # Want to sell 10%, max = 10% → can execute fully
        assert abs(executed_w_day1[0] - 0.20) < 1e-9
        assert abs(residual_day1[0]) < 1e-9
        
        exec_ctx.partial_fill_residual = residual_day1
        
        # Day 2: Further reduce from 20% to 10%
        target_w_day2 = np.array([0.10], dtype=float)
        prev_w_day2 = executed_w_day1.copy()
        
        executed_w_day2, residual_day2 = _apply_liquidity_participation_constraint(
            target_w_day2, prev_w_day2, exec_ctx, max_participation_rate=0.10
        )
        
        # Want to sell 10%, can execute fully
        assert abs(executed_w_day2[0] - 0.10) < 1e-9
        assert abs(residual_day2[0]) < 1e-9


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_capital(self):
        """With zero capital, should return target unchanged."""
        n = 3
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=0.0,  # Zero capital
        )
        
        target_w = np.array([0.10, 0.05, -0.05], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        np.testing.assert_array_almost_equal(executed_w, target_w)
        np.testing.assert_array_almost_equal(residual, np.zeros(n))
    
    def test_zero_participation_rate(self):
        """With zero participation rate, should return target unchanged."""
        n = 3
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e7, 1e7, 1e7], dtype=float)
        
        target_w = np.array([0.10, 0.05, -0.05], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.0
        )
        
        np.testing.assert_array_almost_equal(executed_w, target_w)
        np.testing.assert_array_almost_equal(residual, np.zeros(n))
    
    def test_very_small_trades_ignored(self):
        """Very small trades (< 1e-12) should be ignored."""
        n = 3
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6, 1e6, 1e6], dtype=float)
        
        target_w = np.array([1e-15, -1e-14, 0.10], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # Tiny trades should be ignored
        assert abs(executed_w[0]) < 1e-12
        assert abs(executed_w[1]) < 1e-12
        assert abs(residual[0]) < 1e-12
        assert abs(residual[1]) < 1e-12
        
        # Normal trade should work
        assert abs(executed_w[2] - 0.10) < 1e-9
    
    def test_mixed_long_short_portfolio(self):
        """Test with mixed long/short positions."""
        n = 4
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        # Different ADV for each asset
        exec_ctx.adv_usd = np.array([1e6, 5e6, 2e6, 10e6], dtype=float)
        
        target_w = np.array([0.15, -0.10, 0.08, -0.12], dtype=float)
        prev_w = np.array([0.05, -0.03, 0.00, -0.05], dtype=float)
        
        # Trades: +10%, -7%, +8%, -7%
        # Max: 10%, 50%, 20%, 100% (all in weight space)
        # Asset 0: want +10%, max = 10% → can execute fully
        # Asset 1: want -7%, max = 50% → can execute fully
        # Asset 2: want +8%, max = 20% → can execute fully
        # Asset 3: want -7%, max = 100% → can execute fully
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # All should execute fully (none exceed their limits)
        np.testing.assert_array_almost_equal(executed_w, target_w)
        np.testing.assert_array_almost_equal(residual, np.zeros(n))
    
    def test_default_adv_when_missing(self):
        """When ADV data is missing, should use default (allowing trade)."""
        n = 2
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        # Asset 0 has ADV, asset 1 doesn't (will use default $10M)
        exec_ctx.adv_usd = np.array([1e6], dtype=float)  # Only 1 element!
        
        target_w = np.array([0.15, 0.15], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # Asset 0: ADV = $1M, max = $100K = 10% → partial fill
        assert abs(executed_w[0] - 0.10) < 1e-9
        assert abs(residual[0] - 0.05) < 1e-9
        
        # Asset 1: default ADV = $10M, max = $1M = 100% → full fill
        assert abs(executed_w[1] - 0.15) < 1e-9
        assert abs(residual[1]) < 1e-9


class TestResidualTracking:
    """Test that residuals are tracked correctly for diagnostics."""
    
    def test_residual_stored_correctly(self):
        """Verify residual is stored in exec_ctx for tracking."""
        n = 2
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6, 5e6], dtype=float)
        
        target_w = np.array([0.15, 0.08], dtype=float)
        prev_w = np.zeros(n, dtype=float)
        
        executed_w, residual_out = _apply_liquidity_participation_constraint(
            target_w, prev_w, exec_ctx, max_participation_rate=0.10
        )
        
        # Simulate what main loop does: store residual
        exec_ctx.partial_fill_residual = residual_out
        
        # Verify it's stored
        np.testing.assert_array_almost_equal(
            exec_ctx.partial_fill_residual,
            residual_out
        )
        
        # Asset 0 should have residual
        assert abs(exec_ctx.partial_fill_residual[0] - 0.05) < 1e-9
        # Asset 1 should have no residual
        assert abs(exec_ctx.partial_fill_residual[1]) < 1e-9
    
    def test_residual_not_used_in_computation(self):
        """
        Verify that residual is NOT used in delta_w computation (Gap 4 fix).
        
        This is the core test for the bug fix.
        """
        n = 1
        exec_ctx = ExecutionContext(
            n_assets=n,
            capital_usd=1_000_000.0,
        )
        exec_ctx.adv_usd = np.array([1e6], dtype=float)
        
        # Day 1: partial fill
        target_day1 = np.array([0.20], dtype=float)
        prev_day1 = np.array([0.00], dtype=float)
        
        exec_day1, res_day1 = _apply_liquidity_participation_constraint(
            target_day1, prev_day1, exec_ctx, max_participation_rate=0.10
        )
        
        # Execute 10%, residual 10%
        assert abs(exec_day1[0] - 0.10) < 1e-9
        assert abs(res_day1[0] - 0.10) < 1e-9
        
        # Store residual (simulate main loop)
        exec_ctx.partial_fill_residual = res_day1
        
        # Day 2: same target, verify residual is NOT added to delta
        target_day2 = np.array([0.20], dtype=float)
        prev_day2 = exec_day1.copy()  # Start from 10%
        
        exec_day2, res_day2 = _apply_liquidity_participation_constraint(
            target_day2, prev_day2, exec_ctx, max_participation_rate=0.10
        )
        
        # With fix: delta = 20% - 10% = 10%, execute all → reach 20%
        # Without fix: delta = 20% - 10% + 10% = 20%, capped at 10% → reach 20% but wrong reason
        # Better test: use very illiquid asset where cap matters
        
        # Actually, let's verify by the residual on day 2:
        # If residual was added, day 2 would try to trade 20% (10% target + 10% residual)
        # But only 10% allowed, so would execute 10% → end at 20% with 10% residual
        # With fix, day 2 tries to trade 10% (just the target difference)
        # Can execute all 10% → end at 20% with 0% residual
        
        assert abs(exec_day2[0] - 0.20) < 1e-9, \
            f"Should reach target 20%, got {exec_day2[0]}"
        assert abs(res_day2[0]) < 1e-9, \
            f"Should have no residual (target reached), got {res_day2[0]}"
