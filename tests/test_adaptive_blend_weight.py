"""Test quality-adaptive linear blend weight computation.

This test verifies that the LinearCombinerState properly adjusts blend weights
based on model quality metrics (R², drift, correlation, sign disagreement).
"""

import numpy as np
import pytest

from src.portfolio.linear_alpha_combiner import LinearCombinerState, LinearFeatureBuilder


def test_adaptive_weight_model_not_ready():
    """When model not ready, adaptive weight should be zero."""
    state = LinearCombinerState(horizon=5, min_samples=10)
    
    z_lin = np.array([0.5, -0.3, 0.8])
    z_mamba = np.array([0.6, -0.2, 0.7])
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.5,
        z_lin=z_lin,
        z_mamba=z_mamba,
    )
    
    assert w_adaptive == 0.0
    assert diag["reason"] == "model_not_ready"


def test_adaptive_weight_low_r_squared():
    """Low R² should down-weight the linear model."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Directly set model as fitted with poor R²
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = -0.1  # Poor R²
    state.model.n_samples_ = 30
    state._n_updates = 5
    
    z_lin = np.array([0.5, -0.3, 0.8, 0.2])
    z_mamba = np.array([0.6, -0.2, 0.7, 0.1])
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.5,
        z_lin=z_lin,
        z_mamba=z_mamba,
        r_squared_threshold=0.1,  # High threshold for test
    )
    
    # Should be down-weighted due to negative R²
    assert w_adaptive < 0.5
    assert "r_squared_penalty" in diag


def test_adaptive_weight_low_correlation():
    """Low correlation between z_lin and z_mamba should down-weight."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Directly set model as fitted
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = 0.3  # Decent R²
    state.model.n_samples_ = 50
    state._n_updates = 5  # Bypass stability gate
    
    # Create z_lin and z_mamba with low/negative correlation
    z_lin = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    z_mamba = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # Opposite signs
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.8,
        z_lin=z_lin,
        z_mamba=z_mamba,
        corr_threshold=0.5,
        min_stable_updates=3,
    )
    
    # Should be heavily down-weighted due to negative correlation
    assert w_adaptive < 0.8
    assert "corr_penalty" in diag
    assert diag.get("current_corr", 0.0) < 0.5


def test_adaptive_weight_high_sign_disagreement():
    """High sign disagreement should down-weight."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Directly set model as fitted
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = 0.3
    state.model.n_samples_ = 50
    state._n_updates = 5
    
    # Create predictions with 100% sign disagreement
    z_lin = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    z_mamba = np.array([-1.0, -1.0, -1.0, 1.0, 1.0])  # All opposite
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.7,
        z_lin=z_lin,
        z_mamba=z_mamba,
        sign_disagree_threshold=0.3,  # 30% threshold
        min_stable_updates=3,
    )
    
    # Should be down-weighted
    assert w_adaptive < 0.7
    assert "sign_disagree_penalty" in diag
    assert diag.get("current_sign_disagree", 0.0) > 0.3


def test_adaptive_weight_stability_gate():
    """Should down-weight during warmup period (min_stable_updates)."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Directly set model as fitted with only 2 updates
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = 0.3
    state.model.n_samples_ = 30
    state._n_updates = 2  # Below threshold
    
    z_lin = np.array([0.5, -0.3, 0.8])
    z_mamba = np.array([0.6, -0.2, 0.7])
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.6,
        z_lin=z_lin,
        z_mamba=z_mamba,
        min_stable_updates=5,  # Require 5 updates
    )
    
    # Should be scaled by 2/5 = 0.4
    assert w_adaptive < 0.6
    assert "stability_penalty" in diag
    assert diag.get("stability_penalty", 1.0) == 0.4  # 2/5


def test_adaptive_weight_good_quality():
    """With good quality metrics, should preserve base weight."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Set model with good R²
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = 0.35  # Good R²
    state.model.n_samples_ = 100
    state._n_updates = 10  # Bypass stability gate
    
    # Create highly correlated predictions with low sign disagreement
    z_mamba = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    z_lin = z_mamba * 0.9 + np.random.randn(5) * 0.05  # High correlation
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.7,
        z_lin=z_lin,
        z_mamba=z_mamba,
        r_squared_threshold=0.05,
        drift_threshold=0.5,
        corr_threshold=0.3,
        sign_disagree_threshold=0.4,
        min_stable_updates=3,
    )
    
    # Should preserve most of base weight (minimal penalties)
    assert w_adaptive >= 0.6  # Allow some numerical tolerance
    assert diag.get("current_r_squared", 0.0) > 0.05
    

def test_adaptive_weight_drift_penalty():
    """High coefficient drift should down-weight."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Directly set model as fitted
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = 0.3
    state.model.n_samples_ = 30
    state._n_updates = 5
    
    # Manually inject high drift value
    state._drift_history.append(0.5)  # High drift
    
    z_lin = np.array([0.5, -0.3, 0.8])
    z_mamba = np.array([0.6, -0.2, 0.7])
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.6,
        z_lin=z_lin,
        z_mamba=z_mamba,
        drift_threshold=0.15,  # Low threshold
        min_stable_updates=3,
    )
    
    # Should be down-weighted due to drift
    assert w_adaptive < 0.6
    assert "drift_penalty" in diag


def test_adaptive_weight_multiplicative_penalties():
    """Multiple penalties should compound multiplicatively."""
    state = LinearCombinerState(horizon=5, min_samples=3)
    
    # Set up model with multiple bad conditions
    state.model.beta_ = np.random.randn(20)
    state.model.intercept_ = 0.0
    state.model.mean_ = np.zeros(20)
    state.model.std_ = np.ones(20)
    state.model.r_squared_ = -0.1  # Negative R²
    state.model.n_samples_ = 30
    state._n_updates = 1  # Low update count
    state._drift_history.append(0.3)  # High drift
    
    # Low correlation
    z_lin = np.array([1.0, -1.0, 1.0])
    z_mamba = np.array([-1.0, 1.0, -1.0])
    
    w_adaptive, diag = state.compute_adaptive_blend_weight(
        base_weight=0.8,
        z_lin=z_lin,
        z_mamba=z_mamba,
        r_squared_threshold=0.1,
        drift_threshold=0.15,
        corr_threshold=0.5,
        min_stable_updates=5,
    )
    
    # Multiple penalties should drive weight very low
    assert w_adaptive < 0.3
    assert "combined_penalty" in diag
    assert len(diag.get("penalties_applied", [])) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
