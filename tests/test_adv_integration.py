"""
Integration test for Gap 3 fix: ADV data flow through Phase-2 policy state.

Verifies that ADV data computed in the main loop is correctly passed to
the policy state builder and produces non-zero liquidity stress values.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.stage_b_stateful.phase2_stateful import _policy_state_vector


class TestADVIntegration:
    """Integration tests for ADV data flow through policy state."""
    
    def test_adv_vector_computation_from_dict(self):
        """
        Simulate the ADV vector computation as done in phase2_stateful.py.
        
        This mirrors the code block around line 6670 that builds adv_vec_for_policy
        from adv_usd_by_sym dictionary.
        """
        # Setup: Mock adv_usd_by_sym as it would be in real execution
        syms = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        day = pd.Timestamp('2024-01-15')
        
        # Create mock ADV series for each symbol
        date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        adv_usd_by_sym = {
            'AAPL': pd.Series(1e7, index=date_range),  # $10M ADV
            'MSFT': pd.Series(8e6, index=date_range),  # $8M ADV
            'GOOGL': pd.Series(5e6, index=date_range),  # $5M ADV
            'TSLA': pd.Series(2e6, index=date_range),  # $2M ADV
            'NVDA': pd.Series(1e5, index=date_range),  # $100K ADV (illiquid!)
        }
        
        # Simulate the computation block from phase2_stateful.py
        adv_vec_for_policy = None
        if adv_usd_by_sym:
            try:
                adv_vec_for_policy = np.zeros(len(syms), dtype=float)
                for j, sym in enumerate(syms):
                    adv_s = adv_usd_by_sym.get(str(sym).upper())
                    if adv_s is not None and not adv_s.empty:
                        try:
                            adv_val = float(pd.to_numeric(adv_s.reindex([day]).iloc[0], errors="coerce"))
                            if np.isfinite(adv_val):
                                adv_vec_for_policy[j] = adv_val
                            else:
                                adv_vec_for_policy[j] = 0.0
                        except Exception:
                            adv_vec_for_policy[j] = 0.0
                    else:
                        adv_vec_for_policy[j] = 0.0
                
                if not np.all(adv_vec_for_policy == 0.0):
                    pass  # Keep the computed vector
                else:
                    adv_vec_for_policy = None
            except Exception:
                adv_vec_for_policy = None
        
        # Verify computation
        assert adv_vec_for_policy is not None
        assert len(adv_vec_for_policy) == len(syms)
        assert adv_vec_for_policy[0] == 1e7  # AAPL
        assert adv_vec_for_policy[4] == 1e5  # NVDA (illiquid)
        
        # Pass to policy state vector
        n_assets = len(syms)
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2023-01-01', periods=100),
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
            adv_arr=adv_vec_for_policy,
        )
        
        liquidity_stress = state_vec[17]
        
        # With NVDA at 100K and mean ~5M, stress should be high
        # stress = 1 - (1e5 / mean) where mean ≈ (1e7 + 8e6 + 5e6 + 2e6 + 1e5)/5 = 5.02e6
        # stress ≈ 1 - (1e5 / 5.02e6) ≈ 1 - 0.02 ≈ 0.98
        assert liquidity_stress > 0.95, f"Expected high stress, got {liquidity_stress}"
        assert liquidity_stress <= 1.0
    
    def test_adv_missing_symbols_handled(self):
        """Test that missing ADV data for some symbols is handled gracefully."""
        syms = ['AAPL', 'MSFT', 'UNKNOWN', 'TSLA', 'NVDA']
        day = pd.Timestamp('2024-01-15')
        
        date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        adv_usd_by_sym = {
            'AAPL': pd.Series(1e7, index=date_range),
            'MSFT': pd.Series(8e6, index=date_range),
            # 'UNKNOWN' not in dict
            'TSLA': pd.Series(2e6, index=date_range),
            'NVDA': pd.Series(6e6, index=date_range),
        }
        
        # Build ADV vector
        adv_vec_for_policy = np.zeros(len(syms), dtype=float)
        for j, sym in enumerate(syms):
            adv_s = adv_usd_by_sym.get(str(sym).upper())
            if adv_s is not None and not adv_s.empty:
                try:
                    adv_val = float(pd.to_numeric(adv_s.reindex([day]).iloc[0], errors="coerce"))
                    if np.isfinite(adv_val):
                        adv_vec_for_policy[j] = adv_val
                except Exception:
                    pass
        
        # UNKNOWN symbol should have 0.0 ADV
        assert adv_vec_for_policy[2] == 0.0
        
        # Other symbols should have valid ADV
        assert adv_vec_for_policy[0] == 1e7
        assert adv_vec_for_policy[1] == 8e6
        assert adv_vec_for_policy[3] == 2e6
        assert adv_vec_for_policy[4] == 6e6
        
        # Pass to policy state (should handle zero ADV gracefully)
        n_assets = len(syms)
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2023-01-01', periods=100),
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
            adv_arr=adv_vec_for_policy,
        )
        
        liquidity_stress = state_vec[17]
        
        # Should compute stress from non-zero ADV values
        # Note: Zero values are filtered during isfinite() check in _policy_state_vector
        # After filtering: min_adv = 0 (from UNKNOWN), so stress = 1.0
        # This is correct behavior - zero ADV means completely illiquid
        assert liquidity_stress == 1.0, f"Expected 1.0 stress (zero ADV), got {liquidity_stress}"
    
    def test_adv_empty_dict_returns_none(self):
        """Test that empty adv_usd_by_sym dict results in None ADV vector."""
        syms = ['AAPL', 'MSFT']
        adv_usd_by_sym = {}  # Empty dict
        
        adv_vec_for_policy = None
        if adv_usd_by_sym:
            # Block won't execute because dict is empty
            adv_vec_for_policy = np.zeros(len(syms), dtype=float)
        
        assert adv_vec_for_policy is None
        
        # Policy state should handle None gracefully
        n_assets = len(syms)
        weights = np.ones(n_assets, dtype=float) / n_assets
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2023-01-01', periods=100),
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
            adv_arr=adv_vec_for_policy,
        )
        
        liquidity_stress = state_vec[17]
        
        # Should default to 0.0
        assert liquidity_stress == 0.0
    
    def test_adv_stale_date_handled(self):
        """Test that requesting ADV for a date outside the series range is handled."""
        syms = ['AAPL']
        day = pd.Timestamp('2025-12-31')  # Future date not in series
        
        date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        adv_usd_by_sym = {
            'AAPL': pd.Series(1e7, index=date_range),
        }
        
        # Build ADV vector - reindex will forward fill
        adv_vec_for_policy = np.zeros(len(syms), dtype=float)
        for j, sym in enumerate(syms):
            adv_s = adv_usd_by_sym.get(str(sym).upper())
            if adv_s is not None and not adv_s.empty:
                try:
                    # reindex to future date - will have NaN
                    adv_val = float(pd.to_numeric(adv_s.reindex([day]).iloc[0], errors="coerce"))
                    if np.isfinite(adv_val):
                        adv_vec_for_policy[j] = adv_val
                    else:
                        adv_vec_for_policy[j] = 0.0
                except Exception:
                    adv_vec_for_policy[j] = 0.0
        
        # Should get 0.0 for out-of-range date
        assert adv_vec_for_policy[0] == 0.0
    
    def test_policy_state_vector_shape_unchanged(self):
        """Verify that adding ADV doesn't change state vector dimension."""
        n_assets = 10
        adv_arr = np.random.uniform(1e5, 1e7, n_assets)
        weights = np.ones(n_assets, dtype=float) / n_assets
        
        returns_df = pd.DataFrame(
            np.random.randn(100, n_assets) * 0.01,
            index=pd.date_range('2023-01-01', periods=100),
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
        
        # Should be 25-dimensional (POLICY_STATE_DIM_V2 = 25)
        assert state_vec.shape == (25,), f"Expected (25,), got {state_vec.shape}"
        
        # All dimensions should be finite
        assert np.all(np.isfinite(state_vec))
        
        # Dimensions that can be negative (clipped to [-1, 1]):
        # - Dim 2: Sharpe ratio
        # - Dim 5: avg_corr (correlation)
        # - Dim 10, 11, 12: CBOE features
        negative_allowed = {2, 5, 10, 11, 12}
        
        for i, val in enumerate(state_vec):
            if i in negative_allowed:
                assert -1.0 <= val <= 1.0, f"Dim {i} outside [-1, 1]: {val}"
            else:
                assert 0.0 <= val <= 1.0, f"Dim {i} outside [0, 1]: {val}"
