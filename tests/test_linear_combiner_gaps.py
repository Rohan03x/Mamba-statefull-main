"""
Test Gap C (corr_hhi) and Gap D (buffer persistence) fixes.

Gap C: corr_hhi should be computed from covariance properly and passed to linear builder.
Gap D: Training buffer should persist across restarts with proper rolling window enforcement.
"""

import numpy as np
import pytest
from src.portfolio.linear_alpha_combiner import (
    LinearCombinerState,
    LinearFeatureBuilder,
    create_linear_combiner,
)
from src.portfolio.policy_controller import compute_correlation_hhi


class TestGapC_CorrHHI:
    """Test proper correlation HHI computation and usage."""
    
    def test_covariance_to_correlation_conversion(self):
        """Verify cov → corr conversion using D^-1/2 * Cov * D^-1/2."""
        # Create a simple covariance matrix
        cov = np.array([
            [4.0, 1.2, 0.8],
            [1.2, 9.0, 2.4],
            [0.8, 2.4, 16.0]
        ])
        
        # Convert to correlation manually
        diag_vals = np.sqrt(np.diag(cov))
        diag_inv = 1.0 / diag_vals
        corr = (cov * diag_inv[:, None]) * diag_inv[None, :]
        
        # Verify diagonal is 1.0
        assert np.allclose(np.diag(corr), 1.0), "Correlation diagonal should be 1.0"
        
        # Verify symmetry
        assert np.allclose(corr, corr.T), "Correlation should be symmetric"
        
        # Verify off-diagonals are in [-1, 1]
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        assert np.all(off_diag >= -1.0) and np.all(off_diag <= 1.0), "Correlations in [-1, 1]"
    
    def test_corrcoef_on_cov_is_wrong(self):
        """Demonstrate that np.corrcoef(cov) is WRONG - it correlates rows of cov."""
        cov = np.array([
            [4.0, 1.2, 0.8],
            [1.2, 9.0, 2.4],
            [0.8, 2.4, 16.0]
        ])
        
        # WRONG approach (what was in the code)
        wrong_corr = np.corrcoef(cov)
        
        # RIGHT approach
        diag_vals = np.sqrt(np.diag(cov))
        diag_inv = 1.0 / diag_vals
        right_corr = (cov * diag_inv[:, None]) * diag_inv[None, :]
        
        # They should NOT be the same
        assert not np.allclose(wrong_corr, right_corr), "np.corrcoef(cov) gives wrong result"
        
        # Right approach has diagonal = 1.0
        assert np.allclose(np.diag(right_corr), 1.0), "Right approach: diag = 1.0"
        
        # Wrong approach may not (it correlates rows, which are arbitrary)
        # In this case, wrong_corr diagonal may not be exactly 1.0 if rows are not normalized
    
    def test_correlation_hhi_computation(self):
        """Test eigenvalue-based HHI computation."""
        # Perfect concentration (single factor)
        corr_perfect = np.ones((5, 5))
        hhi_perfect = compute_correlation_hhi(corr_perfect)
        assert hhi_perfect > 0.9, "Perfect correlation → high HHI"
        
        # No correlation (identity)
        corr_identity = np.eye(5)
        hhi_identity = compute_correlation_hhi(corr_identity)
        assert hhi_identity < 0.3, "No correlation → low HHI"
        
        # Moderate correlation
        corr_moderate = np.eye(5) + 0.3 * (np.ones((5, 5)) - np.eye(5))
        hhi_moderate = compute_correlation_hhi(corr_moderate)
        assert 0.2 < hhi_moderate < 0.7, "Moderate correlation → medium HHI"
    
    def test_linear_builder_receives_corr_hhi(self):
        """Verify linear feature builder accepts and uses corr_hhi parameter."""
        builder = LinearFeatureBuilder()
        n_assets = 10
        
        # Build features with corr_hhi = 0.0 (low concentration)
        X_low = builder.build_day(
            i=100,
            syms=[f"SYM{i}" for i in range(n_assets)],
            z_mamba=np.random.randn(n_assets),
            sigma_exec=np.ones(n_assets) * 0.02,
            corr_hhi=0.0,
        )
        
        # Build features with corr_hhi = 0.8 (high concentration)
        X_high = builder.build_day(
            i=100,
            syms=[f"SYM{i}" for i in range(n_assets)],
            z_mamba=np.random.randn(n_assets),
            sigma_exec=np.ones(n_assets) * 0.02,
            corr_hhi=0.8,
        )
        
        # Feature index 16 is corr_hhi
        assert X_low.shape == (n_assets, 19), "19 features total"
        assert np.allclose(X_low[:, 16], 0.0), "corr_hhi=0.0 should be stored"
        assert np.allclose(X_high[:, 16], 0.8), "corr_hhi=0.8 should be stored"
        assert not np.allclose(X_low[:, 16], X_high[:, 16]), "Different corr_hhi values"


class TestGapD_BufferPersistence:
    """Test rolling buffer persistence and proper window enforcement."""
    
    def test_day_idx_tracking(self):
        """Verify day_idx is tracked when adding to buffer."""
        state = create_linear_combiner(horizon=21, window=126, max_window=252)
        
        n_assets = 5
        n_features = 19
        
        # Observe several days
        for i in range(10):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state.observe_day(i, X_day, sigma_day)
        
        # Create fake forward returns
        fwd_ret_mat = np.random.randn(100, n_assets) * 0.01
        
        # Process matured observations (starting from day 21)
        for i in range(21, 30):
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Check buffer has day_idx tracking
        assert len(state._day_idx_train) > 0, "day_idx_train should be populated"
        assert len(state._day_idx_train) == len(state._X_train), "day_idx matches X buffer"
        assert len(state._day_idx_train) == len(state._y_train), "day_idx matches y buffer"
    
    def test_rolling_window_enforcement_by_day(self):
        """Verify buffer enforces max_window by day count, not sample count."""
        state = create_linear_combiner(horizon=21, window=63, max_window=126)
        
        n_assets = 10
        n_features = 19
        
        # Observe many days
        for i in range(200):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state.observe_day(i, X_day, sigma_day)
        
        # Create fake forward returns
        fwd_ret_mat = np.random.randn(250, n_assets) * 0.01
        
        # Process all matured observations
        for i in range(21, 200):
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Check buffer respects max_window (126 days)
        if len(state._day_idx_train) > 0:
            oldest_day = state._day_idx_train[0]
            newest_day = state._day_idx_train[-1]
            day_span = newest_day - oldest_day
            
            # Buffer should not exceed max_window
            assert day_span <= state.max_window, f"Day span {day_span} exceeds max_window {state.max_window}"
    
    def test_window_vs_max_window_usage(self):
        """Verify 'window' selects training subset from 'max_window' buffer."""
        state = create_linear_combiner(horizon=21, window=63, max_window=126)
        
        n_assets = 5
        n_features = 19
        
        # Observe 150 days (more than max_window)
        for i in range(150):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state.observe_day(i, X_day, sigma_day)
        
        # Create fake forward returns
        fwd_ret_mat = np.random.randn(200, n_assets) * 0.01
        
        # Process matured observations
        for i in range(21, 150):
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Trigger a refit at day 149
        state._last_fit_day = -1  # Force refit
        report = state.update_if_matured(149, fwd_ret_mat, freeze=False)
        
        if report is not None and report.get("status") == "fitted":
            # Check that training used 'window' days, not full buffer
            n_samples = report.get("n_samples", 0)
            
            # Training should use recent 'window' days (63)
            # Each day contributes ~n_assets samples (assuming most are valid)
            # So n_samples should be roughly (window * n_assets), not (max_window * n_assets)
            expected_samples_rough = state.window * n_assets
            max_samples_rough = state.max_window * n_assets
            
            # Samples should be closer to window estimate than max_window
            assert n_samples < max_samples_rough * 0.7, "Should use window subset, not full buffer"
    
    def test_buffer_persistence_to_dict(self):
        """Verify buffer is serialized in to_dict()."""
        state = create_linear_combiner(horizon=21, window=63, max_window=126)
        
        n_assets = 5
        n_features = 19
        
        # Add some observations
        for i in range(30):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state.observe_day(i, X_day, sigma_day)
        
        # Process matured observations
        fwd_ret_mat = np.random.randn(100, n_assets) * 0.01
        for i in range(21, 30):
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Serialize
        data = state.to_dict()
        
        # Check buffer fields exist
        assert "buffer_X" in data, "buffer_X should be serialized"
        assert "buffer_y" in data, "buffer_y should be serialized"
        assert "buffer_days" in data, "buffer_days should be serialized"
        assert "buffer_sample_counts" in data, "buffer_sample_counts should be serialized"
        
        # Check non-empty if buffer has data
        if len(state._X_train) > 0:
            assert data["buffer_X"] is not None, "buffer_X should have data"
            assert data["buffer_y"] is not None, "buffer_y should have data"
            assert data["buffer_days"] is not None, "buffer_days should have data"
            assert data["buffer_sample_counts"] is not None, "buffer_sample_counts should have data"
    
    def test_buffer_persistence_from_dict(self):
        """Verify buffer is restored from from_dict()."""
        # Create and populate state
        state1 = create_linear_combiner(horizon=21, window=63, max_window=126)
        
        n_assets = 5
        n_features = 19
        
        # Add observations
        for i in range(30):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state1.observe_day(i, X_day, sigma_day)
        
        # Process matured observations
        fwd_ret_mat = np.random.randn(100, n_assets) * 0.01
        for i in range(21, 30):
            state1.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Store buffer state
        orig_X_count = len(state1._X_train)
        orig_y_count = len(state1._y_train)
        orig_days = list(state1._day_idx_train)
        
        # Serialize and deserialize
        data = state1.to_dict()
        state2 = LinearCombinerState.from_dict(data)
        
        # Check buffer was restored
        assert len(state2._X_train) == orig_X_count, "X buffer count should match"
        assert len(state2._y_train) == orig_y_count, "y buffer count should match"
        assert len(state2._day_idx_train) == len(orig_days), "day_idx buffer count should match"
        
        if len(orig_days) > 0:
            assert state2._day_idx_train == orig_days, "day_idx values should match"
    
    def test_buffer_persistence_warm_restart(self):
        """Verify warm restart: restored buffer enables immediate training."""
        # Create state and train it
        state1 = create_linear_combiner(horizon=21, window=63, max_window=126, min_samples=30)
        
        n_assets = 5
        n_features = 19
        
        # Populate with enough data to train
        for i in range(80):
            X_day = np.random.randn(n_assets, n_features)
            sigma_day = np.ones(n_assets) * 0.02
            state1.observe_day(i, X_day, sigma_day)
        
        fwd_ret_mat = np.random.randn(150, n_assets) * 0.01
        for i in range(21, 80):
            state1.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Trigger fit
        state1._last_fit_day = -1
        report1 = state1.update_if_matured(80, fwd_ret_mat, freeze=False)
        
        assert state1.is_ready(), "State1 should be ready after training"
        
        # Serialize and restore
        data = state1.to_dict()
        state2 = LinearCombinerState.from_dict(data)
        
        # Check state2 is ready immediately (warm restart)
        assert state2.is_ready(), "State2 should be ready after warm restart"
        
        # Check model coefficients were restored
        assert state2.model.beta_ is not None, "Model should have coefficients"
        assert len(state2.model.beta_) == n_features, "Coefficients should match feature count"


class TestIntegration:
    """Integration tests combining Gap C and Gap D fixes."""
    
    def test_end_to_end_with_corr_hhi_and_persistence(self):
        """Full workflow: build features with corr_hhi, train, persist, restore."""
        state = create_linear_combiner(horizon=21, window=63, max_window=126, min_samples=30)
        
        n_assets = 10
        builder = state.feature_builder
        
        # Simulate 100 days
        for i in range(100):
            # Build features with varying corr_hhi
            corr_hhi_val = 0.2 + 0.6 * np.sin(i / 10.0)  # Oscillate between 0.2 and 0.8
            
            X_day = builder.build_day(
                i=i,
                syms=[f"SYM{j}" for j in range(n_assets)],
                z_mamba=np.random.randn(n_assets),
                sigma_exec=np.ones(n_assets) * 0.02,
                day_calib_score=0.8,
                day_online_trust=0.9,
                corr_hhi=corr_hhi_val,
            )
            
            sigma_day = np.ones(n_assets) * 0.02
            state.observe_day(i, X_day, sigma_day)
            
            # Check corr_hhi was stored in features
            assert np.allclose(X_day[:, 16], corr_hhi_val), f"Day {i}: corr_hhi should be {corr_hhi_val}"
        
        # Process matured observations
        fwd_ret_mat = np.random.randn(150, n_assets) * 0.01
        for i in range(21, 100):
            state.update_if_matured(i, fwd_ret_mat, freeze=False)
        
        # Trigger refit
        state._last_fit_day = -1
        report = state.update_if_matured(100, fwd_ret_mat, freeze=False)
        
        assert state.is_ready(), "State should be trained"
        
        # Serialize
        data = state.to_dict()
        
        # Verify persistence includes buffer
        assert data["buffer_X"] is not None, "Buffer should be persisted"
        assert len(data["buffer_days"]) > 0, "Buffer days should be persisted"
        
        # Restore
        state2 = LinearCombinerState.from_dict(data)
        
        # Verify warm restart
        assert state2.is_ready(), "Restored state should be ready"
        assert len(state2._X_train) > 0, "Buffer should be restored"
        assert len(state2._day_idx_train) > 0, "day_idx should be restored"
        
        # Verify predictions work
        X_new = builder.build_day(
            i=101,
            syms=[f"SYM{j}" for j in range(n_assets)],
            z_mamba=np.random.randn(n_assets),
            sigma_exec=np.ones(n_assets) * 0.02,
            corr_hhi=0.5,
        )
        
        z_pred = state2.predict_z(X_new)
        assert z_pred.shape == (n_assets,), "Prediction should work after restore"
        assert np.all(np.isfinite(z_pred)), "Predictions should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
