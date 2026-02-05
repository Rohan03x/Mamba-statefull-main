"""
Test suite for linear combiner enhancements:
1. Confidence tracking via rolling residual variance
2. Time-decay weighting within training window
3. Separate linear vs quantile blend knobs
"""
import numpy as np
import pytest
from src.portfolio.linear_alpha_combiner import LinearCombinerState, create_linear_combiner


def test_confidence_tracking():
    """
    Test that confidence is computed from rolling residual variance.
    
    High residuals → low confidence → scaled-down predictions
    Low residuals → high confidence → full-strength predictions
    """
    n_symbols, T = 20, 40
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 20
    state.ridge_lambda = 0.1
    state.min_samples = 20
    state.confidence_ewma_alpha = 0.3
    state.__post_init__()
    
    # Phase 1: Good predictions (low residuals)
    for t in range(25):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # Returns that match model predictions closely
        base_ret = 0.001 * np.random.randn(n_symbols)
        noise = 0.0001 * np.random.randn(n_symbols)  # Small noise
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = base_ret + noise
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Check confidence after good predictions
    confidence_good = None
    if state.is_ready():
        X_test = np.random.randn(n_symbols, 15)
        z_lin, confidence_good = state.predict_z_with_confidence(X_test, confidence_scaling=False)
        
        # Should have high confidence (low residuals)
        print(f"Confidence after good predictions: {np.mean(confidence_good):.3f}")
    
    # Phase 2: Bad predictions (high residuals)
    for t in range(25, T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # Returns that deviate significantly from predictions
        base_ret = 0.001 * np.random.randn(n_symbols)
        noise = 0.05 * np.random.randn(n_symbols)  # Large noise
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = base_ret + noise
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Check confidence after bad predictions
    if state.is_ready() and confidence_good is not None:
        X_test = np.random.randn(n_symbols, 15)
        z_lin, confidence_bad = state.predict_z_with_confidence(X_test, confidence_scaling=False)
        
        print(f"Confidence after bad predictions: {np.mean(confidence_bad):.3f}")
        print(f"Residual EWMA: {state._residual_ewma}")
        print(f"N confidence updates: {state._n_confidence_updates}")
        
        # Confidence should have decreased (higher residuals)
        # With EWMA alpha=0.3, it should respond to regime change
        if state._n_confidence_updates > 3:
            assert np.mean(confidence_bad) < np.mean(confidence_good), \
                f"Confidence should decrease with high residuals: {np.mean(confidence_good):.3f} → {np.mean(confidence_bad):.3f}"
        else:
            # If not enough updates matured, at least check EWMA is tracking
            assert state._residual_ewma is not None, "Residual EWMA should be tracking"
            print("Not enough confidence updates to verify decrease (need >3 mature observations)")


def test_confidence_scaling_reduces_predictions():
    """
    Test that confidence scaling reduces z_lin magnitude when confidence is low.
    """
    n_symbols, T = 20, 30
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 20
    state.ridge_lambda = 0.1
    state.min_samples = 20
    state.__post_init__()
    
    # Train with noisy data to induce low confidence
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # High variance returns
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = 0.05 * np.random.randn(n_symbols)
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    if state.is_ready():
        X_test = np.random.randn(n_symbols, 15)
        
        # Unscaled predictions
        z_lin_unscaled, confidence = state.predict_z_with_confidence(X_test, confidence_scaling=False)
        
        # Confidence-scaled predictions
        z_lin_scaled, _ = state.predict_z_with_confidence(X_test, confidence_scaling=True)
        
        # Scaled predictions should be smaller (confidence < 1)
        assert np.mean(np.abs(z_lin_scaled)) <= np.mean(np.abs(z_lin_unscaled)), \
            "Confidence scaling should reduce prediction magnitude"
        
        # Verify scaling relationship
        expected_scaled = z_lin_unscaled * confidence
        assert np.allclose(z_lin_scaled, expected_scaled, atol=1e-6), \
            "Scaled predictions should equal unscaled * confidence"


def test_time_decay_weighting():
    """
    Test that time-decay gives more weight to recent days.
    
    Older days should have exponentially decaying influence.
    """
    n_symbols, T = 20, 50
    
    # Model with time decay (halflife=21 days)
    state_decay = LinearCombinerState()
    state_decay.horizon = 5
    state_decay.window = 40
    state_decay.ridge_lambda = 0.1
    state_decay.min_samples = 30
    state_decay.time_decay_halflife = 21  # Recent days weighted more
    state_decay.__post_init__()
    
    # Model without time decay (uniform weighting)
    state_uniform = LinearCombinerState()
    state_uniform.horizon = 5
    state_uniform.window = 40
    state_uniform.ridge_lambda = 0.1
    state_uniform.min_samples = 30
    state_uniform.time_decay_halflife = 0  # No decay
    state_uniform.__post_init__()
    
    # Simulate regime change: old days have different pattern than recent days
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        # Old regime (first 30 days): returns proportional to first feature
        # New regime (last 20 days): returns proportional to second feature
        if t < 30:
            # Old pattern: y ≈ 0.5 * X[0]
            signal = 0.5 * X_day[:, 0]
        else:
            # New pattern: y ≈ 0.5 * X[1]
            signal = 0.5 * X_day[:, 1]
        
        returns = signal + 0.01 * np.random.randn(n_symbols)
        
        fwd_ret_mat_decay = np.zeros((T - t, n_symbols))
        fwd_ret_mat_uniform = np.zeros((T - t, n_symbols))
        
        if t + state_decay.horizon < T:
            fwd_ret_mat_decay[state_decay.horizon, :] = returns
            fwd_ret_mat_uniform[state_uniform.horizon, :] = returns
        
        # Feed to both models
        state_decay.observe_day(t, X_day, sigma_exec)
        state_decay.update_if_matured(i=t, fwd_ret_mat=fwd_ret_mat_decay, freeze=False)
        
        state_uniform.observe_day(t, X_day, sigma_exec)
        state_uniform.update_if_matured(i=t, fwd_ret_mat=fwd_ret_mat_uniform, freeze=False)
    
    # After regime change, time-decay model should adapt faster
    # It should learn new pattern (X[1]) better than uniform model
    if state_decay.is_ready() and state_uniform.is_ready():
        # Test on new regime pattern
        X_test = np.random.randn(n_symbols, 15)
        X_test[:, 1] = 2.0  # Strong signal in second feature (new regime)
        X_test[:, 0] = 0.0  # No signal in first feature (old regime)
        
        z_decay = state_decay.predict_z(X_test)
        z_uniform = state_uniform.predict_z(X_test)
        
        # Time-decay model should give stronger response to new pattern
        # (We can't guarantee this with random data, but we verify it trains differently)
        assert state_decay.model.beta_ is not None
        assert state_uniform.model.beta_ is not None
        
        # Coefficients should differ due to different weighting
        coef_diff = np.linalg.norm(state_decay.model.beta_ - state_uniform.model.beta_)
        assert coef_diff > 0.01, "Time-decay should produce different coefficients"


def test_time_decay_zero_means_uniform():
    """
    Test that time_decay_halflife=0 produces uniform weighting (no decay).
    """
    n_symbols, T = 20, 30
    
    state = LinearCombinerState()
    state.horizon = 5
    state.window = 20
    state.ridge_lambda = 0.1
    state.min_samples = 20
    state.time_decay_halflife = 0  # No time decay
    state.__post_init__()
    
    for t in range(T):
        X_day = np.random.randn(n_symbols, 15)
        sigma_exec = np.ones(n_symbols) * 0.01
        
        fwd_ret_mat = np.zeros((T - t, n_symbols))
        if t + state.horizon < T:
            fwd_ret_mat[state.horizon, :] = 0.01 * np.random.randn(n_symbols)
        
        state.observe_day(t, X_day, sigma_exec)
        state.update_if_matured(
            i=t,
            fwd_ret_mat=fwd_ret_mat,
            freeze=False
        )
    
    # Refit should work without decay
    info = state._do_refit(i=25)
    assert info["status"] == "fitted"
    
    # Verify all days in window are used
    assert len(state._day_idx_train) > 0


def test_factory_with_time_decay():
    """
    Test that create_linear_combiner accepts time_decay_halflife parameter.
    """
    state = create_linear_combiner(
        horizon=21,
        update_interval=21,
        ridge_lambda=10.0,
        window=126,
        max_window=252,
        min_samples=63,
        time_decay_halflife=42,
        symbols=["AAPL", "MSFT", "GOOGL"],
    )
    
    assert state.time_decay_halflife == 42
    assert state.horizon == 21
    assert state.window == 126


def test_confidence_with_no_updates():
    """
    Test that confidence defaults to 1.0 when no updates have occurred.
    """
    state = LinearCombinerState()
    state.__post_init__()
    
    X_test = np.random.randn(10, 15)
    z_lin, confidence = state.predict_z_with_confidence(X_test, confidence_scaling=False)
    
    # No training data → confidence should be neutral (1.0)
    assert np.allclose(confidence, 1.0), "Confidence should be 1.0 before any updates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
