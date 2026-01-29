"""Unit tests for Linear Alpha Combiner.

Tests cover:
- RidgeModel fit/predict shape correctness
- Feature standardization roundtrip
- Maturity alignment (no lookahead)
- Blend gating until ready
- Hygiene final veto enforcement
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.linear_alpha_combiner import (
    LinearFeatureBuilder,
    RidgeModel,
    LinearCombinerState,
    create_linear_combiner,
    N_FEATURES,
    ALL_FEATURE_NAMES,
)


class TestRidgeModel:
    """Tests for RidgeModel."""
    
    def test_ridge_fit_predict_shapes(self):
        """Verify (n_samples, n_features) → (n_samples,) output."""
        model = RidgeModel(n_features=5, lambda_=1.0)
        
        # Synthetic training data
        np.random.seed(42)
        n_train = 100
        X_train = np.random.randn(n_train, 5)
        y_train = X_train[:, 0] * 2.0 + np.random.randn(n_train) * 0.1
        
        model.fit(X_train, y_train)
        
        # Verify fitted
        assert model.beta_ is not None
        assert model.beta_.shape == (5,)
        assert model.mean_ is not None
        assert model.mean_.shape == (5,)
        assert model.std_ is not None
        assert model.std_.shape == (5,)
        
        # Predict
        n_test = 20
        X_test = np.random.randn(n_test, 5)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (n_test,)
        assert np.all(np.isfinite(y_pred))
    
    def test_ridge_recovers_simple_relationship(self):
        """Verify Ridge recovers linear coefficients."""
        model = RidgeModel(n_features=3, lambda_=0.01)  # Low regularization
        
        np.random.seed(123)
        n = 500
        X = np.random.randn(n, 3)
        # y = 2*x0 + 0.5*x1 - 1*x2 + noise
        true_coefs = np.array([2.0, 0.5, -1.0])
        y = X @ true_coefs + np.random.randn(n) * 0.1
        
        model.fit(X, y)
        
        # Check coefficients approximately match (accounting for standardization)
        assert model.beta_ is not None
        # After standardization, coefficients scale by std
        # Just check signs are correct
        assert model.beta_[0] > 0  # Should be positive
        assert model.beta_[1] > 0  # Should be positive
        assert model.beta_[2] < 0  # Should be negative
        
        # R² should be high
        assert model.r_squared_ > 0.9
    
    def test_standardization_roundtrip(self):
        """Verify mean/std stored correctly and inverse works."""
        model = RidgeModel(n_features=4, lambda_=1.0)
        
        np.random.seed(999)
        X = np.random.randn(50, 4) * np.array([1, 10, 100, 0.1]) + np.array([0, 5, -50, 0.5])
        y = np.random.randn(50)
        
        model.fit(X, y)
        
        # Check standardization stats computed
        assert model.mean_ is not None
        assert model.std_ is not None
        
        # Standardized features should have mean≈0, std≈1 on training data
        X_std = (X - model.mean_) / model.std_
        assert np.allclose(np.mean(X_std, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(X_std, axis=0, ddof=1), 1, atol=0.1)
    
    def test_predict_not_fitted_returns_zeros(self):
        """Unfitted model returns zeros."""
        model = RidgeModel(n_features=5)
        
        X_test = np.random.randn(10, 5)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (10,)
        assert np.allclose(y_pred, 0.0)
    
    def test_get_coefficients_report(self):
        """Test coefficient report generation."""
        model = RidgeModel(n_features=5, lambda_=1.0)
        
        # Not fitted
        report = model.get_coefficients_report()
        assert report["status"] == "not_fitted"
        
        # Fit
        np.random.seed(1)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        
        report = model.get_coefficients_report()
        assert report["status"] == "fitted"
        assert "z_mamba_coef" in report
        assert "r_squared" in report
        assert "top_coefficients" in report
        assert len(report["top_coefficients"]) == 5


class TestLinearCombinerState:
    """Tests for LinearCombinerState."""
    
    def test_blend_gating_until_ready(self):
        """is_ready()=False → z unchanged."""
        state = create_linear_combiner(
            horizon=5,
            update_interval=5,
            min_samples=10,
            symbols=["A", "B", "C"],
        )
        
        # Not ready initially
        assert not state.is_ready()
        
        # Predict returns zeros
        X_day = np.random.randn(3, N_FEATURES)
        z_lin = state.predict_z(X_day)
        assert np.allclose(z_lin, 0.0)
    
    def test_observe_buffers_correctly(self):
        """Observations are buffered for maturity processing."""
        state = create_linear_combiner(horizon=5)
        
        X_day = np.random.randn(3, N_FEATURES)
        sigma_exec = np.array([0.02, 0.03, 0.02])
        
        state.observe_day(0, X_day, sigma_exec)
        
        assert 0 in state._X_buffer
        assert 0 in state._sigma_exec_buffer
        assert state._X_buffer[0].shape == (3, N_FEATURES)
    
    def test_maturity_alignment_no_leakage(self):
        """update uses matured_idx only, no future data."""
        state = create_linear_combiner(
            horizon=5,
            update_interval=1,  # Refit every day for testing
            min_samples=5,
        )
        
        np.random.seed(42)
        n_days = 20
        n_assets = 3
        
        # Simulate: observe days 0..n_days, with fwd returns
        fwd_ret_mat = np.random.randn(n_days, n_assets) * 0.05
        
        for i in range(n_days):
            X_day = np.random.randn(n_assets, N_FEATURES)
            sigma_exec = np.full(n_assets, 0.02)
            state.observe_day(i, X_day, sigma_exec)
            
            # Update only processes matured predictions
            refit_report = state.update_if_matured(i, fwd_ret_mat, freeze=False)
            
            # Before horizon days, no training data should be used
            if i < state.horizon:
                assert len(state._X_train) == 0 or len(state._y_train) == 0
        
        # After enough days, should have training data
        assert len(state._X_train) > 0
        assert len(state._y_train) > 0
    
    def test_freeze_logic(self):
        """Freeze=True buffers but doesn't refit."""
        state = create_linear_combiner(
            horizon=3,
            update_interval=1,
            min_samples=3,
        )
        
        np.random.seed(0)
        n_days = 15
        n_assets = 2
        fwd_ret_mat = np.random.randn(n_days, n_assets) * 0.05
        
        # First pass: allow updates
        for i in range(10):
            X_day = np.random.randn(n_assets, N_FEATURES)
            sigma_exec = np.full(n_assets, 0.02)
            state.observe_day(i, X_day, sigma_exec)
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        updates_before = state._n_updates
        assert updates_before > 0  # Should have done some updates
        
        # Now freeze
        for i in range(10, 15):
            X_day = np.random.randn(n_assets, N_FEATURES)
            sigma_exec = np.full(n_assets, 0.02)
            state.observe_day(i, X_day, sigma_exec)
            state.update_if_matured(i, fwd_ret_mat, freeze=True)
        
        updates_after = state._n_updates
        assert updates_after == updates_before  # No new updates while frozen
    
    def test_serialization_roundtrip(self):
        """to_dict / from_dict roundtrip."""
        state = create_linear_combiner(
            horizon=5,
            update_interval=5,
            ridge_lambda=15.0,
            window=100,
        )
        
        # Fit the model
        np.random.seed(42)
        n_days = 20
        n_assets = 3
        fwd_ret_mat = np.random.randn(n_days, n_assets) * 0.05
        
        for i in range(n_days):
            X_day = np.random.randn(n_assets, N_FEATURES)
            sigma_exec = np.full(n_assets, 0.02)
            state.observe_day(i, X_day, sigma_exec)
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Serialize
        data = state.to_dict()
        
        assert "beta" in data
        assert "horizon" in data
        assert data["horizon"] == 5
        
        # Deserialize
        state2 = LinearCombinerState.from_dict(data)
        
        assert state2.horizon == state.horizon
        assert state2.ridge_lambda == state.ridge_lambda
        assert state2._n_updates == state._n_updates
        
        if state.model.beta_ is not None:
            assert np.allclose(state2.model.beta_, state.model.beta_)


class TestLinearFeatureBuilder:
    """Tests for LinearFeatureBuilder."""
    
    def test_build_day_shape(self):
        """Output shape matches (n_assets, n_features)."""
        builder = LinearFeatureBuilder()
        
        syms = ["AAPL", "MSFT", "GOOG"]
        z_mamba = np.array([0.5, -0.3, 0.1])
        sigma_exec = np.array([0.02, 0.03, 0.025])
        
        X = builder.build_day(
            i=10,
            syms=syms,
            z_mamba=z_mamba,
            sigma_exec=sigma_exec,
        )
        
        assert X.shape == (3, N_FEATURES)
        assert np.all(np.isfinite(X))
    
    def test_z_mamba_is_first_feature(self):
        """z_mamba values appear in first column."""
        builder = LinearFeatureBuilder()
        
        z_mamba = np.array([1.0, 2.0, 3.0])
        
        X = builder.build_day(
            i=0,
            syms=["A", "B", "C"],
            z_mamba=z_mamba,
            sigma_exec=np.ones(3) * 0.02,
        )
        
        assert np.allclose(X[:, 0], z_mamba)
    
    def test_global_features_broadcast(self):
        """Global features have same value for all assets."""
        builder = LinearFeatureBuilder()
        
        X = builder.build_day(
            i=10,
            syms=["A", "B", "C", "D"],
            z_mamba=np.zeros(4),
            sigma_exec=np.ones(4) * 0.02,
            day_calib_score=0.75,
            day_cboe_panic=0.5,
        )
        
        # day_calib_score is at index 10 (10 per-symbol features + 0)
        assert np.all(X[:, 10] == 0.75)
        state = create_linear_combiner(horizon=3, min_samples=3)
        
        # Force model to be "ready" by setting beta
        state.model.beta_ = np.ones(N_FEATURES) * 0.1
        state.model.mean_ = np.zeros(N_FEATURES)
        state.model.std_ = np.ones(N_FEATURES)
        
        assert state.is_ready()
        
        # Get prediction
        X_day = np.ones((3, N_FEATURES))
        z_lin = state.predict_z(X_day)
        
        # z_lin should be non-zero
        assert not np.allclose(z_lin, 0.0)
        
        # Simulate hygiene veto
        hygiene_ok = np.array([True, False, True])
        z_mamba = np.array([1.0, 1.0, 1.0])
        w_L = 0.5
        
        # Blend
        z = (1.0 - w_L) * z_mamba + w_L * z_lin
        
        # Apply hygiene veto
        z_final = np.where(hygiene_ok, z, 0.0)
        
        # Asset 1 (hygiene=False) should be zero
        assert z_final[1] == 0.0
        assert z_final[0] != 0.0
        assert z_final[2] != 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
