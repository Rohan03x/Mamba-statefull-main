"""
Comprehensive tests for Gap 3 fix: ADV integration into policy liquidity stress.

Tests verify that:
1. liquidity_stress is computed correctly when ADV data is provided
2. liquidity_stress reflects portfolio illiquidity (high stress when holding low-ADV assets)
3. liquidity_stress is 0.0 when ADV data is missing or invalid
4. Edge cases (NaN, zero ADV, single asset, etc.) are handled properly
"""

import numpy as np
import pytest

from src.stage_b_stateful.phase2_stateful import _policy_state_vector
import pandas as pd


class TestLiquidityStressComputation:
    """Test liquidity_stress calculation in policy state vector (dimension 17)."""
    
    def test_liquidity_stress_with_uniform_adv(self):
        """When all assets have same ADV, liquidity_stress should be 0.0."""
        # Setup: uniform ADV across all assets
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)  # All assets have $1M ADV
        weights = np.ones(n_assets, dtype=float) / n_assets  # Equal weight portfolio
        
        # Minimal test data
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        # Dimension 17 is liquidity_stress
        liquidity_stress = state_vec[17]
        
        # With uniform ADV: min_adv = mean_adv, so stress = 1 - 1 = 0
        assert liquidity_stress == 0.0, f"Expected 0.0 for uniform ADV, got {liquidity_stress}"
    
    def test_liquidity_stress_with_one_illiquid_asset(self):
        """When one asset has very low ADV, liquidity_stress should be high (~1.0)."""
        # Setup: one asset with very low ADV, others high
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)
        adv_arr[0] = 1e3  # First asset has only $1K ADV (0.1% of others)
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        
        # min_adv = 1e3, mean_adv ≈ (1e3 + 9*1e6)/10 = 901e3
        # stress = 1 - (1e3 / 901e3) ≈ 1 - 0.00111 ≈ 0.999
        assert liquidity_stress > 0.99, f"Expected high stress (>0.99), got {liquidity_stress}"
        assert liquidity_stress <= 1.0, f"Stress should be clipped to 1.0, got {liquidity_stress}"
    
    def test_liquidity_stress_without_adv_data(self):
        """When adv_arr is None, liquidity_stress should default to 0.0."""
        n_assets = 10
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=None,  # No ADV data
        )
        
        liquidity_stress = state_vec[17]
        
        # Should default to 0.0 when no ADV data
        assert liquidity_stress == 0.0, f"Expected 0.0 when adv_arr=None, got {liquidity_stress}"
    
    def test_liquidity_stress_with_nan_adv_values(self):
        """When ADV has NaN values, they should be filtered out."""
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)
        adv_arr[0] = np.nan
        adv_arr[1] = np.nan
        adv_arr[2] = 1e3  # One illiquid asset after filtering NaNs
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        
        # After filtering NaNs: min_adv = 1e3, mean_adv ≈ (1e3 + 7*1e6)/8
        # Should still compute valid stress
        assert liquidity_stress > 0.9, f"Expected high stress after filtering NaNs, got {liquidity_stress}"
        assert liquidity_stress <= 1.0
    
    def test_liquidity_stress_with_all_nan_adv(self):
        """When all ADV values are NaN, stress should be 0.0."""
        n_assets = 10
        adv_arr = np.full(n_assets, np.nan, dtype=float)
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        
        # No valid ADV data after filtering -> default to 0.0
        assert liquidity_stress == 0.0, f"Expected 0.0 when all ADV is NaN, got {liquidity_stress}"
    
    def test_liquidity_stress_with_zero_adv(self):
        """When some assets have zero ADV, stress should be 1.0 (max)."""
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)
        adv_arr[0] = 0.0  # Zero ADV = completely illiquid
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        
        # min_adv = 0, so stress = 1 - (0 / mean) = 1.0
        assert liquidity_stress == 1.0, f"Expected 1.0 for zero ADV, got {liquidity_stress}"
    
    def test_liquidity_stress_with_single_asset(self):
        """With single asset, stress should be 0.0 (min = mean)."""
        n_assets = 1
        adv_arr = np.array([1e6], dtype=float)
        weights = np.array([1.0], dtype=float)
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        
        # Single asset: min = mean, so stress = 0
        assert liquidity_stress == 0.0, f"Expected 0.0 for single asset, got {liquidity_stress}"
    
    def test_liquidity_stress_gradient(self):
        """Stress should increase monotonically as min ADV decreases."""
        n_assets = 10
        base_adv = 1e6
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        stress_values = []
        min_adv_fractions = [1.0, 0.5, 0.1, 0.01, 0.001]  # Decreasing min ADV
        
        for frac in min_adv_fractions:
            adv_arr = np.full(n_assets, base_adv, dtype=float)
            adv_arr[0] = base_adv * frac
            
            state_vec = _policy_state_vector(
                i=50,
                horizon=21,
                returns_df=returns_df,
                net_ret=np.random.randn(100) * 0.01,
                turnover=np.random.rand(100) * 0.1,
                costs=np.random.rand(100) * 0.001,
                equity=np.cumprod(1 + np.random.randn(100) * 0.01),
                mu_mat=np.random.randn(100, n_assets) * 0.001,
                sigma_mat=np.random.rand(100, n_assets) * 0.02,
                fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
                market_regime=pd.Series(np.zeros(100), index=returns_df.index),
                weights=weights,
                adv_arr=adv_arr,
            )
            
            stress_values.append(state_vec[17])
        
        # Stress should increase as min ADV fraction decreases
        for i in range(len(stress_values) - 1):
            assert stress_values[i] < stress_values[i + 1], \
                f"Stress should increase as ADV decreases: {stress_values}"
    
    def test_liquidity_stress_is_in_valid_range(self):
        """Stress should always be in [0, 1] regardless of ADV distribution."""
        n_assets = 20
        
        # Test various ADV distributions
        test_cases = [
            np.random.uniform(1e3, 1e7, n_assets),  # Random uniform
            np.logspace(3, 7, n_assets),  # Log-spaced (exponential)
            np.array([1e3] + [1e7] * (n_assets - 1)),  # One tiny, rest huge
            np.array([1e7, 1e6, 1e5, 1e4] + [1e3] * (n_assets - 4)),  # Gradual decline
        ]
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        for adv_arr in test_cases:
            state_vec = _policy_state_vector(
                i=50,
                horizon=21,
                returns_df=returns_df,
                net_ret=np.random.randn(100) * 0.01,
                turnover=np.random.rand(100) * 0.1,
                costs=np.random.rand(100) * 0.001,
                equity=np.cumprod(1 + np.random.randn(100) * 0.01),
                mu_mat=np.random.randn(100, n_assets) * 0.001,
                sigma_mat=np.random.rand(100, n_assets) * 0.02,
                fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
                market_regime=pd.Series(np.zeros(100), index=returns_df.index),
                weights=weights,
                adv_arr=adv_arr,
            )
            
            liquidity_stress = state_vec[17]
            assert 0.0 <= liquidity_stress <= 1.0, \
                f"Stress {liquidity_stress} outside [0, 1] for ADV distribution {adv_arr}"


class TestADVIntegrationEdgeCases:
    """Test edge cases and error handling in ADV integration."""
    
    def test_empty_adv_array(self):
        """Empty ADV array should default stress to 0.0."""
        n_assets = 10
        adv_arr = np.array([], dtype=float)  # Empty
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        assert liquidity_stress == 0.0
    
    def test_negative_adv_values(self):
        """Negative ADV values should be handled gracefully."""
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)
        adv_arr[0] = -1e6  # Invalid negative value
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        # Should not crash, compute stress from valid values
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        # Should compute stress from 9 positive values (min = 0 after clipping negatives)
        assert 0.0 <= liquidity_stress <= 1.0
    
    def test_inf_adv_values(self):
        """Infinite ADV values should be filtered out."""
        n_assets = 10
        adv_arr = np.full(n_assets, 1e6, dtype=float)
        adv_arr[0] = np.inf
        adv_arr[1] = -np.inf
        
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        state_vec = _policy_state_vector(
            i=50,
            horizon=21,
            returns_df=returns_df,
            net_ret=np.random.randn(100) * 0.01,
            turnover=np.random.rand(100) * 0.1,
            costs=np.random.rand(100) * 0.001,
            equity=np.cumprod(1 + np.random.randn(100) * 0.01),
            mu_mat=np.random.randn(100, n_assets) * 0.001,
            sigma_mat=np.random.rand(100, n_assets) * 0.02,
            fwd_ret_mat=np.random.randn(100, n_assets) * 0.01,
            market_regime=pd.Series(np.zeros(100), index=returns_df.index),
            weights=weights,
            adv_arr=adv_arr,
        )
        
        liquidity_stress = state_vec[17]
        # Should compute from 8 finite values
        assert 0.0 <= liquidity_stress <= 1.0
