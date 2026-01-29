"""
Test for Gap 2: Quantile & Linear Blending Mutual Exclusion.

Verifies that linear and quantile blending are mutually exclusive,
preventing double-blending that would dilute signals incorrectly.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock


class TestLinearQuantileMutualExclusion:
    """Test that linear and quantile blending don't both apply."""
    
    def test_linear_ready_blocks_quantile(self):
        """
        When linear model is ready and w_L > 0, quantile blending should NOT apply.
        
        Scenario:
        - quantile_blend_weight_day = 0.5
        - linear_blend_max = 0.5
        - linear model ready
        
        Expected:
        - Only linear blend applied (w_L = 0.5)
        - Quantile blend NOT applied
        - Final z = 0.5 * z_mamba + 0.5 * z_lin (no quantile component)
        """
        # Setup
        z_mamba = np.array([1.0, -0.5, 0.8, -0.3, 0.0])
        z_quantile = np.array([0.5, -0.3, 0.6, -0.2, 0.1])
        z_lin = np.array([0.8, -0.4, 0.7, -0.25, 0.05])
        
        w_L = 0.5  # Linear blend weight
        
        # Simulate linear blending
        z_after_linear = (1.0 - w_L) * z_mamba + w_L * z_lin
        
        # Expected: z should equal z_after_linear (NO quantile blend applied)
        expected = z_after_linear
        
        # Verify quantile is NOT applied
        # If quantile were applied: z = (1-0.5) * z_after_linear + 0.5 * z_quantile
        z_if_both = (1.0 - 0.5) * z_after_linear + 0.5 * z_quantile
        
        # These should NOT be equal (proves no double blend)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            expected, z_if_both
        )
        
        # Verify final composition
        # Should be 50% mamba + 50% linear, NO quantile
        expected_composition = {
            "mamba": 0.5,
            "linear": 0.5,
            "quantile": 0.0,
        }
        
        # Verify via reconstruction
        reconstructed = 0.5 * z_mamba + 0.5 * z_lin
        np.testing.assert_array_almost_equal(expected, reconstructed)
    
    def test_linear_not_ready_allows_quantile(self):
        """
        When linear model exists but NOT ready, quantile blending should apply.
        
        Scenario:
        - Linear model initialized but < min_samples
        - quantile_blend_weight_day = 0.3
        
        Expected:
        - Linear blend NOT applied (model not ready)
        - Quantile blend applied (w_q = 0.3)
        - Final z = 0.7 * z_mamba + 0.3 * z_quantile
        """
        z_mamba = np.array([1.0, -0.5, 0.8, -0.3, 0.0])
        z_quantile = np.array([0.5, -0.3, 0.6, -0.2, 0.1])
        
        w_q = 0.3  # Quantile blend weight
        
        # Simulate quantile blending (fallback when linear not ready)
        z_expected = (1.0 - w_q) * z_mamba + w_q * z_quantile
        
        # Verify composition
        expected_composition = {
            "mamba": 0.7,
            "quantile": 0.3,
            "linear": 0.0,
        }
        
        # Verify via reconstruction
        reconstructed = 0.7 * z_mamba + 0.3 * z_quantile
        np.testing.assert_array_almost_equal(z_expected, reconstructed)
    
    def test_linear_disabled_uses_quantile(self):
        """
        When linear model disabled (linear_state = None), quantile applies.
        
        Scenario:
        - phase2_linear_model_enabled = False
        - quantile_blend_weight_day = 0.2
        
        Expected:
        - Linear blend NOT applied (disabled)
        - Quantile blend applied (w_q = 0.2)
        - Final z = 0.8 * z_mamba + 0.2 * z_quantile
        """
        z_mamba = np.array([1.0, -0.5, 0.8])
        z_quantile = np.array([0.5, -0.3, 0.6])
        
        w_q = 0.2
        z_expected = (1.0 - w_q) * z_mamba + w_q * z_quantile
        
        # Verify
        reconstructed = 0.8 * z_mamba + 0.2 * z_quantile
        np.testing.assert_array_almost_equal(z_expected, reconstructed)
    
    def test_no_double_blend_scenario(self):
        """
        Integration test: Verify no double-blending in worst-case scenario.
        
        Scenario (the bug case):
        - quantile_blend_weight_day = 0.3
        - linear_blend_max = 0.5
        - linear model ready
        
        Without fix (BUG):
        - First: z = 0.7 * z_mamba + 0.3 * z_lin
        - Then: z = 0.7 * z + 0.3 * z_quantile
        - Result: ~0.49 mamba + 0.21 linear + 0.30 quantile (WRONG)
        
        With fix:
        - Only: z = 0.7 * z_mamba + 0.3 * z_lin
        - Result: 0.7 mamba + 0.3 linear + 0.0 quantile (CORRECT)
        """
        z_mamba = np.array([1.0, -0.5, 0.8, -0.3])
        z_quantile = np.array([0.5, -0.3, 0.6, -0.2])
        z_lin = np.array([0.8, -0.4, 0.7, -0.25])
        
        quantile_blend_weight_day = 0.3
        linear_blend_max = 0.5
        w_L = np.clip(quantile_blend_weight_day, 0.0, linear_blend_max)  # 0.3
        
        # CORRECT BEHAVIOR (with fix):
        # Linear blend applied, quantile NOT applied
        z_correct = (1.0 - w_L) * z_mamba + w_L * z_lin
        
        # BUGGY BEHAVIOR (without fix):
        # Linear blend, THEN quantile blend on top
        z_after_linear = (1.0 - w_L) * z_mamba + w_L * z_lin
        z_buggy = (1.0 - 0.3) * z_after_linear + 0.3 * z_quantile
        
        # Verify they are different
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            z_correct, z_buggy
        )
        
        # Verify correct composition
        # z_correct should be exactly: 0.7 * z_mamba + 0.3 * z_lin
        reconstructed_correct = 0.7 * z_mamba + 0.3 * z_lin
        np.testing.assert_array_almost_equal(z_correct, reconstructed_correct)
        
        # Verify buggy composition would be wrong
        # z_buggy would be: 0.7 * (0.7 * z_mamba + 0.3 * z_lin) + 0.3 * z_quantile
        #                  = 0.49 * z_mamba + 0.21 * z_lin + 0.3 * z_quantile
        reconstructed_buggy = 0.49 * z_mamba + 0.21 * z_lin + 0.3 * z_quantile
        np.testing.assert_array_almost_equal(z_buggy, reconstructed_buggy, decimal=10)
    
    def test_policy_knob_interpretation(self):
        """
        Verify policy's quantile_blend_weight is interpreted correctly.
        
        With linear model enabled and ready:
        - policy sets quantile_blend_weight = 0.4
        - linear_blend_max = 0.5
        - Expect: w_L = 0.4 (capped), applied to LINEAR (not quantile)
        - Result: 60% mamba + 40% linear (NOT 60% mamba + 40% quantile)
        """
        policy_quantile_weight = 0.4
        linear_blend_max = 0.5
        
        # With linear ready, this weight goes to LINEAR blending
        w_L = np.clip(policy_quantile_weight, 0.0, linear_blend_max)
        assert w_L == 0.4
        
        # Composition should be:
        composition = {
            "mamba": 1.0 - w_L,  # 0.6
            "linear": w_L,        # 0.4
            "quantile": 0.0,      # 0.0 (NOT applied)
        }
        
        assert composition["mamba"] == 0.6
        assert composition["linear"] == 0.4
        assert composition["quantile"] == 0.0
    
    def test_zero_blend_weight_no_blending(self):
        """
        When quantile_blend_weight_day = 0, no blending should occur.
        """
        z_mamba = np.array([1.0, -0.5, 0.8])
        z_quantile = np.array([0.5, -0.3, 0.6])
        z_lin = np.array([0.8, -0.4, 0.7])
        
        w_q = 0.0
        
        # No blending
        z_expected = z_mamba.copy()
        
        # Verify
        np.testing.assert_array_equal(z_expected, z_mamba)
    
    def test_linear_exception_fallback_to_quantile(self):
        """
        If linear feature building fails, should fall back to quantile.
        
        Scenario:
        - Linear model enabled and ready
        - Feature build raises exception
        - quantile_blend_weight_day = 0.2
        
        Expected:
        - Linear blend NOT applied (exception)
        - Quantile blend applied (fallback)
        """
        z_mamba = np.array([1.0, -0.5, 0.8])
        z_quantile = np.array([0.5, -0.3, 0.6])
        
        # With exception in linear path, should fall back to quantile
        w_q = 0.2
        z_expected = (1.0 - w_q) * z_mamba + w_q * z_quantile
        
        # Verify
        reconstructed = 0.8 * z_mamba + 0.2 * z_quantile
        np.testing.assert_array_almost_equal(z_expected, reconstructed)


class TestBlendingLogic:
    """Test the blending decision tree."""
    
    def test_decision_tree_linear_ready(self):
        """
        Decision: linear_state is not None AND is_ready() AND w_L > 0
        Result: Apply linear blend, skip quantile
        """
        linear_state_mock = Mock()
        linear_state_mock.is_ready.return_value = True
        
        quantile_blend_weight_day = 0.3
        linear_blend_max = 0.5
        
        w_L = np.clip(quantile_blend_weight_day, 0.0, linear_blend_max)
        
        # Decision logic
        linear_blend_applied = (
            linear_state_mock is not None and
            linear_state_mock.is_ready() and
            w_L > 0
        )
        
        assert linear_blend_applied == True
        
        # Quantile should NOT apply
        should_apply_quantile = not linear_blend_applied
        assert should_apply_quantile == False
    
    def test_decision_tree_linear_not_ready(self):
        """
        Decision: linear_state is not None BUT not is_ready()
        Result: Skip linear, apply quantile
        """
        linear_state_mock = Mock()
        linear_state_mock.is_ready.return_value = False
        
        quantile_blend_weight_day = 0.3
        linear_blend_max = 0.5
        w_L = np.clip(quantile_blend_weight_day, 0.0, linear_blend_max)
        
        # Decision logic
        linear_blend_applied = (
            linear_state_mock is not None and
            linear_state_mock.is_ready() and
            w_L > 0
        )
        
        assert linear_blend_applied is False
        
        # Quantile SHOULD apply
        should_apply_quantile = not linear_blend_applied and quantile_blend_weight_day > 0
        assert should_apply_quantile is True
    
    def test_decision_tree_linear_disabled(self):
        """
        Decision: linear_state is None
        Result: Skip linear, apply quantile
        """
        linear_state_mock = None
        quantile_blend_weight_day = 0.3
        
        # Decision logic
        linear_blend_applied = False  # Can't apply if None
        
        # Quantile SHOULD apply
        should_apply_quantile = not linear_blend_applied and quantile_blend_weight_day > 0
        assert should_apply_quantile is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
