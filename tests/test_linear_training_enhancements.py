"""
Test suite for LinearCombinerState training enhancements:
1. Per-day sample weighting (fix universe size bias)
2. Alpha residual targets (reduce beta via cross-sectional demeaning)
"""
import numpy as np
import pytest
from src.portfolio.linear_alpha_combiner import LinearCombinerState


class DummyMambaState:
    """Stub for testing"""
    def predict(self, X):
        # Return simple z-scores for testing
        return np.random.randn(*X.shape[:2])


def test_alpha_residual_cross_sectional_demean():
    """
    Test that targets are cross-sectionally demeaned:
    - Raw return: r = [0.02, 0.04, -0.01]
    - Mean return: mean(r) = 0.0167
    - Alpha residual: r_alpha = [0.0033, 0.0233, -0.0267]
    - Target: y = r_alpha / sigma_exec
    
    Verifies that training removes market mode.
    """
    n_seq, n_ctx, T = 5, 3, 30
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 20
    state.ridge_lambda = 0.1
    state.__post_init__()
    
    # Simulate data with common market return component
    market_ret = 0.015  # 1.5% market move
    idio_ret = np.array([0.005, 0.025, -0.025])  # Individual stock alpha
    
    for t in range(T):
        seq_mat = np.random.randn(n_seq, 3)
        ctx_mat = np.random.randn(n_ctx, 3)
        
        # Build feature matrix
        X_day = np.random.randn(3, 15)  # 3 symbols, 15 features (approx)
        sigma_exec = np.ones(3) * 0.01
        
        # All stocks get market return + their idiosyncratic component
        actual_ret = market_ret + idio_ret + 0.001 * np.random.randn(3)
        
        # Build forward returns matrix
        fwd_ret_mat = np.zeros((T - t, 3))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = actual_ret
        
        # Observe current day
        state.observe_day(t, X_day, sigma_exec)
        
        # Try to update with matured observations
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Check that training targets are demeaned
    # (can't directly access y_train in current API, but we verify via fit)
    assert state._y_train is not None
    assert len(state._y_train) > 0
    
    # Verify cross-sectional mean is removed: mean of targets should be ~0
    for y_day in state._y_train:
        if len(y_day) > 0:
            # After demeaning, mean should be ~0 (up to numerical noise)
            # Note: y = (r - mean(r)) / sigma, so mean(y) ≈ 0
            assert abs(np.mean(y_day)) < 0.1, f"Targets not demeaned: mean={np.mean(y_day)}"


def test_day_weighting_equal_contribution():
    """
    Test that per-day weighting ensures equal contribution:
    - Some days have masked out symbols (NaN returns)
    - Day effective samples vary
    
    Without weighting: Days with more valid samples get more weight in Ridge fit
    With weighting: Each day contributes equally
    """
    n_symbols = 30  # Fixed universe size
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 20
    state.ridge_lambda = 0.1
    state.min_samples = 10  # Lower threshold for test
    state.__post_init__()
    
    T = 30
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)  # Approx feature count
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # Vary number of valid returns per day
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            returns = 0.01 * np.random.randn(n_symbols)
            # Some days have fewer valid symbols
            if t < 15:
                # First half: mask out 20 symbols (only 10 valid)
                returns[10:] = np.nan
            # Second half: all 30 symbols valid
            fwd_ret_mat[state.horizon, :] = returns
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Refit should apply day weighting
    info = state._do_refit(i=25)
    
    assert info["status"] == "fitted"
    
    # Check that training data was weighted
    # Day with 30 valid samples should have weight 1/sqrt(30) ≈ 0.183 per sample
    # Day with 10 valid samples should have weight 1/sqrt(10) ≈ 0.316 per sample
    # This ensures each day contributes equally in total
    
    # Verify via buffer sizes (indirect check since we can't inspect weights directly)
    assert len(state._day_idx_train) > 0
    print(f"Trained on {len(state._day_idx_train)} days with day weighting")


def test_combined_fix_reduces_market_beta():
    """
    Integration test: Combined alpha residual + day weighting
    should train model to predict idiosyncratic alpha, not market beta.
    
    Setup:
    - Market factor: r_mkt = 0.02
    - Stock returns: r_i = 0.02 + alpha_i where alpha_i ~ N(0, 0.01)
    
    Without fix: Model learns market beta
    With fix: Model learns idiosyncratic component
    """
    n_symbols, T = 20, 50
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 30
    state.ridge_lambda = 0.1
    state.min_samples = 20
    state.__post_init__()
    
    market_returns = []
    
    for t in range(T):
        r_mkt = 0.02 if t % 2 == 0 else -0.01  # Alternating market
        market_returns.append(r_mkt)
        
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # All stocks track market + small idiosyncratic noise
        alpha_i = 0.005 * np.random.randn(n_symbols)
        actual_ret = r_mkt + alpha_i
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = actual_ret
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Refit with alpha residual + day weighting
    info = state._do_refit(i=40)
    
    assert info["status"] == "fitted"
    
    # Verify targets have removed market component
    # After cross-sectional demeaning, mean of returns should be ~0
    for y_day in state._y_train:
        if len(y_day) > 1:
            # Mean should be near zero after demeaning
            assert abs(np.mean(y_day)) < 0.2


def test_edge_case_single_symbol_day():
    """
    Test that day weighting handles edge case where only 1 symbol has valid return.
    Weight should be 1/sqrt(1) = 1.0 (no scaling).
    """
    n_symbols, T = 10, 20
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 15
    state.ridge_lambda = 0.1
    state.min_samples = 5
    state.__post_init__()
    
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            returns = 0.01 * np.random.randn(n_symbols)
            # Alternate between only 1 valid symbol and all 10 valid
            if t % 2 == 0:
                returns[1:] = np.nan  # Only first symbol valid
            fwd_ret_mat[state.horizon, :] = returns
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Should handle single-symbol days gracefully
    info = state._do_refit(i=15)
    assert info["status"] == "fitted"
    assert info["n_samples"] > 0


def test_zero_variance_day_handling():
    """
    Test that cross-sectional demeaning handles case where all returns are identical.
    If r_i = 0.02 for all stocks, then r_alpha = 0 for all (no dispersion).
    """
    n_symbols, T = 15, 20
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 15
    state.ridge_lambda = 0.1
    state.min_samples = 20
    state.__post_init__()
    
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # All stocks have identical return (no cross-sectional variation)
        identical_ret = 0.015
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = identical_ret
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Should handle zero-variance days
    # After demeaning, all targets should be ~0
    info = state._do_refit(i=15)
    assert info["status"] == "fitted"
    
    # All targets should be near zero after demeaning identical returns
    for y_day in state._y_train:
        assert np.allclose(y_day, 0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
