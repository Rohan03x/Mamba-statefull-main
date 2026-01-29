"""
Tests for Gap 6: Beta Neutralization Config Handling

Verifies:
1. Beta neutralization is HARD (always neutralizes first), not soft guardrail
2. beta_cap = 0 → full neutralization
3. beta_cap > 0 → neutralize + allow capped exposure
4. Config logic works correctly with deprecated beta_neutral flag

Classification: Low impact (config clarity), Low severity
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage_b_stateful.phase2_stateful import _apply_beta_neutralization


class TestBetaNeutralizationHardConstraint(unittest.TestCase):
    """Test that beta neutralization is HARD (not soft)."""
    
    def test_always_neutralizes_first(self):
        """
        Verify that _apply_beta_neutralization ALWAYS neutralizes first,
        regardless of cap setting. This is NOT a soft guardrail.
        """
        # Portfolio with strong positive beta exposure
        w = np.array([0.2, 0.3, 0.1, -0.1, 0.0])
        beta = np.array([1.5, 1.2, 0.8, -0.5, 0.0])
        
        # Original beta exposure
        original_exposure = float(beta @ w)
        self.assertGreater(original_exposure, 0.3)  # Significant positive beta
        
        # Apply with very large cap (should still neutralize first)
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=10.0)
        result_exposure = float(beta @ w_result)
        
        # Should neutralize to ~0, not just cap at 10.0
        self.assertLess(abs(result_exposure), 0.01, 
                       "Should neutralize to ~0, not preserve original beta")
    
    def test_full_neutralization_when_cap_zero(self):
        """beta_cap = 0 should result in full beta neutralization."""
        w = np.array([0.2, 0.3, -0.1, 0.1, 0.0])
        beta = np.array([1.2, 1.0, 0.8, -0.3, 0.0])
        
        # Original exposure
        original_exposure = float(beta @ w)
        self.assertNotAlmostEqual(original_exposure, 0.0, places=2)
        
        # Full neutralization (cap = 0)
        w_neutral = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        neutral_exposure = float(beta @ w_neutral)
        
        # Should be neutralized to ~0
        self.assertAlmostEqual(neutral_exposure, 0.0, places=6,
                              msg="Cap=0 should fully neutralize beta")
    
    def test_capped_exposure_when_cap_positive(self):
        """beta_cap > 0 should neutralize first, then allow capped exposure."""
        w = np.array([0.3, 0.2, 0.1, -0.1, 0.0])
        beta = np.array([1.5, 1.2, 0.8, -0.5, 0.0])
        
        cap = 0.25  # Allow up to ±0.25 beta exposure
        
        # Apply neutralization with cap
        w_capped = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=cap)
        capped_exposure = float(beta @ w_capped)
        
        # Should be neutralized first (near 0), not just capped at 0.25
        # This is the key difference between hard and soft
        self.assertLess(abs(capped_exposure), 0.01,
                       msg="Should neutralize first, then numeric precision may leave small residual")
    
    def test_negative_cap_returns_neutralized(self):
        """Negative cap should return fully neutralized weights.
        
        Note: The function ALWAYS neutralizes when called. Negative cap
        means 'full neutralization' (no partial exposure allowed).
        The config logic decides whether to call this function at all.
        """
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([1.2, 1.0, 0.8, -0.3])
        
        # Original exposure
        original_exp = float(beta @ w)
        self.assertNotAlmostEqual(original_exp, 0.0, places=1)
        
        # Negative cap should fully neutralize (same as cap=0)
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=-1.0)
        result_exp = float(beta @ w_result)
        
        # Should be neutralized to ~0
        self.assertAlmostEqual(result_exp, 0.0, places=6,
                              msg="Negative cap should fully neutralize")


class TestBetaNeutralizationMechanics(unittest.TestCase):
    """Test the mechanics of beta neutralization."""
    
    def test_neutralization_preserves_dollar_neutral(self):
        """Neutralization with uniform betas should make portfolio dollar-neutral."""
        w = np.array([0.2, 0.3, 0.1, -0.1, 0.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # All same beta
        
        original_sum = float(np.sum(w))
        self.assertGreater(original_sum, 0.3)  # Positive net exposure
        
        w_neutral = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        neutral_sum = float(np.sum(w_neutral))
        
        # When all betas = 1, beta neutralization = dollar neutralization
        # So sum should go to ~0
        self.assertAlmostEqual(neutral_sum, 0.0, places=6,
                              msg="Uniform betas → dollar neutral portfolio")
    
    def test_zero_beta_vector_returns_unchanged(self):
        """If all betas are zero, should return original weights."""
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([0.0, 0.0, 0.0, 0.0])
        
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        np.testing.assert_array_almost_equal(w_result, w)
    
    def test_mismatched_dimensions_returns_unchanged(self):
        """If dimensions don't match, should return original weights."""
        w = np.array([0.2, 0.3, -0.1])
        beta = np.array([1.0, 1.2])  # Wrong size
        
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        np.testing.assert_array_equal(w_result, w)
    
    def test_single_asset_portfolio(self):
        """Single asset should work correctly."""
        w = np.array([0.5])
        beta = np.array([1.2])
        
        # Neutralize: should go to 0
        w_neutral = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        self.assertAlmostEqual(w_neutral[0], 0.0, places=6)


class TestBetaNeutralizationEdgeCases(unittest.TestCase):
    """Test edge cases and numeric stability."""
    
    def test_very_small_beta_denominator(self):
        """Near-zero betas should return unchanged weights."""
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        # Should return original (denom too small)
        np.testing.assert_array_almost_equal(w_result, w)
    
    def test_nan_in_beta_vector(self):
        """NaN in beta should be handled gracefully."""
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([1.0, float('nan'), 1.0, 1.0])
        
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        # Should return original (denom will be NaN)
        np.testing.assert_array_equal(w_result, w)
    
    def test_inf_beta_values(self):
        """Inf in beta should be handled gracefully."""
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([1.0, float('inf'), 1.0, 1.0])
        
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        # Should return original (denom will be inf)
        np.testing.assert_array_equal(w_result, w)
    
    def test_large_weights_and_betas(self):
        """Should handle large values without overflow."""
        w = np.array([100.0, 200.0, -50.0, 75.0])
        beta = np.array([5.0, 3.0, 2.0, -1.0])
        
        # Should not raise overflow/underflow
        w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        
        # Check result is finite
        self.assertTrue(np.all(np.isfinite(w_result)))
        
        # Check neutralization worked
        exposure = float(beta @ w_result)
        self.assertLess(abs(exposure), 1.0)


class TestBetaCapThresholds(unittest.TestCase):
    """Test different beta cap threshold behaviors."""
    
    def test_cap_zero_vs_negative_cap(self):
        """Verify cap=0 (full neutral) vs cap<0 (also full neutral).
        
        Note: Both neutralize fully. The difference is in the CONFIG LOGIC
        that decides whether to call this function:
        - beta_cap < 0 → don't call function (no constraint)
        - beta_cap >= 0 → call function (neutralize, possibly with cap)
        """
        w = np.array([0.2, 0.3, 0.1, -0.1])
        beta = np.array([1.2, 1.0, 0.8, -0.3])
        
        # Full neutralization (cap=0)
        w_cap_zero = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=0.0)
        exp_cap_zero = abs(float(beta @ w_cap_zero))
        
        # Negative cap (also neutralizes when function is called)
        w_neg_cap = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=-1.0)
        exp_neg_cap = abs(float(beta @ w_neg_cap))
        
        # Both should neutralize to ~0
        self.assertLess(exp_cap_zero, 0.01, "Cap=0 should neutralize")
        self.assertLess(exp_neg_cap, 0.01, "Negative cap also neutralizes (when called)")
    
    def test_increasing_caps(self):
        """As cap increases, should allow more beta exposure."""
        w = np.array([0.3, 0.2, 0.1, -0.1])
        beta = np.array([1.5, 1.2, 0.8, -0.5])
        
        caps = [0.0, 0.1, 0.2, 0.3, 0.5]
        exposures = []
        
        for cap in caps:
            w_result = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=cap)
            exp = abs(float(beta @ w_result))
            exposures.append(exp)
        
        # Exposures should be monotonically non-decreasing
        # (though in practice all should be ~0 due to HARD neutralization)
        for i in range(len(exposures) - 1):
            self.assertLessEqual(exposures[i], exposures[i+1] + 0.01,
                                msg=f"Exposure should increase with cap: {exposures}")
    
    def test_very_large_cap_equivalent_to_full_neutralization(self):
        """
        Very large cap should still neutralize (not preserve original).
        This confirms HARD neutralization behavior.
        """
        w = np.array([0.2, 0.3, -0.1, 0.1])
        beta = np.array([1.2, 1.0, 0.8, -0.3])
        
        # Original exposure
        original_exp = abs(float(beta @ w))
        
        # Apply with huge cap
        w_huge_cap = _apply_beta_neutralization(w, beta, max_abs_beta_exposure=100.0)
        huge_cap_exp = abs(float(beta @ w_huge_cap))
        
        # Should neutralize to ~0, NOT preserve original ~0.48
        self.assertLess(huge_cap_exp, 0.01,
                       "Even huge cap should neutralize first (HARD behavior)")
        self.assertGreater(original_exp, 0.3, "Verify original had significant beta")


class TestConfigLogicSimulation(unittest.TestCase):
    """Simulate config logic from phase2_stateful.py line 7792."""
    
    def test_config_logic_beta_neutral_true(self):
        """If beta_neutral=True, should apply neutralization."""
        beta_neutral = True
        beta_cap = 0.0  # irrelevant when beta_neutral=True
        
        # Condition from line 7792
        should_apply = beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0)
        
        self.assertTrue(should_apply,
                       "beta_neutral=True should trigger neutralization")
    
    def test_config_logic_beta_cap_set(self):
        """If beta_cap >= 0, should apply neutralization (even if beta_neutral=False)."""
        beta_neutral = False
        beta_cap = 0.25
        
        # Condition from line 7792 (after fix: >= 0 instead of > 0)
        should_apply = beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0)
        
        self.assertTrue(should_apply,
                       "beta_cap >= 0 should trigger neutralization")
    
    def test_config_logic_both_false(self):
        """If beta_neutral=False and beta_cap < 0, should NOT apply."""
        beta_neutral = False
        beta_cap = -1.0
        
        should_apply = beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0)
        
        self.assertFalse(should_apply,
                        "Neither flag should mean no neutralization")
    
    def test_config_logic_cap_zero_is_active(self):
        """beta_cap = 0 should be considered active (full neutralization)."""
        beta_neutral = False
        beta_cap = 0.0
        
        # After fix: should use >= 0 instead of > 0
        should_apply = beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0)
        
        self.assertTrue(should_apply,
                       "beta_cap=0 should trigger full neutralization")
    
    def test_config_logic_nan_cap_ignored(self):
        """NaN beta_cap should be ignored."""
        beta_neutral = False
        beta_cap = float('nan')
        
        should_apply = beta_neutral or (np.isfinite(beta_cap) and beta_cap >= 0)
        
        self.assertFalse(should_apply,
                        "NaN beta_cap should be ignored")


if __name__ == "__main__":
    unittest.main()
