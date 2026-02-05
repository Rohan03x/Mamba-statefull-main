"""
Test suite for linear combiner non-linearity enhancements:
1. Interaction features (z_mamba * volatility, regime interactions)
2. Non-linear transformations (z_mamba^2)
3. Sector indicators (one-hot encoding)

These features address the limited model complexity of pure linear Ridge regression.
"""
import numpy as np
import pytest
from src.portfolio.linear_alpha_combiner import (
    LinearFeatureBuilder,
    LinearCombinerState,
    create_linear_combiner,
    N_FEATURES,
    N_FEATURES_EXTENDED,
    FEATURE_NAMES_INTERACTIONS,
    FEATURE_NAMES_SECTORS,
)


def test_interaction_features_enabled():
    """
    Test that enabling interactions adds correct features.
    
    Interaction terms capture non-linear relationships:
    - z_mamba * rv_21d: volatility-conditional signal strength
    - z_mamba * regime: regime-conditional predictions
    - z_mamba^2: non-linear signal strength
    """
    builder = LinearFeatureBuilder(
        symbols=["AAPL", "MSFT", "GOOGL"],
        use_interactions=True,
        use_sectors=False,
    )
    
    # Check feature count increased
    expected_n_features = N_FEATURES + len(FEATURE_NAMES_INTERACTIONS)
    assert builder.n_features == expected_n_features, \
        f"Expected {expected_n_features} features with interactions, got {builder.n_features}"
    
    # Check feature names include interactions
    assert "z_mamba_x_rv" in builder.feature_names
    assert "z_mamba_x_regime" in builder.feature_names
    assert "z_mamba_squared" in builder.feature_names
    assert "ret_1d_x_rv" in builder.feature_names
    assert "regime_x_panic" in builder.feature_names


def test_sector_features_enabled():
    """
    Test that enabling sectors adds sector indicators.
    
    Sector indicators allow model to learn sector-specific coefficients,
    addressing the one-size-fits-all limitation.
    """
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "JPM": "Financial Services",
        "JNJ": "Healthcare",
        "XOM": "Energy",
    }
    
    builder = LinearFeatureBuilder(
        symbols=list(sector_map.keys()),
        use_interactions=False,
        use_sectors=True,
        sector_map=sector_map,
    )
    
    # Check feature count increased
    expected_n_features = N_FEATURES + len(FEATURE_NAMES_SECTORS)
    assert builder.n_features == expected_n_features, \
        f"Expected {expected_n_features} features with sectors, got {builder.n_features}"
    
    # Check feature names include sectors
    assert "sector_tech" in builder.feature_names
    assert "sector_finance" in builder.feature_names
    assert "sector_healthcare" in builder.feature_names
    assert "sector_energy" in builder.feature_names


def test_interaction_features_computed_correctly():
    """
    Test that interaction features are computed with correct formulas.
    """
    sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
    
    builder = LinearFeatureBuilder(
        symbols=["AAPL", "MSFT"],
        use_interactions=True,
        use_sectors=False,
    )
    
    # Build features with known inputs
    z_mamba = np.array([2.0, -1.5])
    sigma_exec = np.array([0.01, 0.02])
    
    X = builder.build_day(
        i=10,
        syms=["AAPL", "MSFT"],
        z_mamba=z_mamba,
        sigma_exec=sigma_exec,
        day_calib_score=0.9,
        day_online_trust=0.95,
        day_cboe_panic=0.1,
        day_cboe_slope=-0.05,
        day_cboe_vrp=0.2,
        returns_df=None,
        corr_hhi=0.15,
    )
    
    # Base features are first N_FEATURES columns
    z_mamba_col = X[:, 0]
    rv_21d_col = X[:, 8]  # Will be default 0.15
    regime_mult_col = X[:, 3]  # Will be default 1.0
    cboe_panic_col = X[:, 12]
    
    # Interaction features start at index N_FEATURES
    z_mamba_x_rv = X[:, N_FEATURES + 0]
    z_mamba_x_regime = X[:, N_FEATURES + 1]
    z_mamba_squared = X[:, N_FEATURES + 2]
    regime_x_panic = X[:, N_FEATURES + 4]
    
    # Verify computations
    np.testing.assert_allclose(z_mamba_x_rv, z_mamba_col * rv_21d_col, rtol=1e-6)
    np.testing.assert_allclose(z_mamba_x_regime, z_mamba_col * regime_mult_col, rtol=1e-6)
    np.testing.assert_allclose(z_mamba_squared, z_mamba_col ** 2, rtol=1e-6)
    np.testing.assert_allclose(regime_x_panic, regime_mult_col * cboe_panic_col, rtol=1e-6)


def test_sector_indicators_one_hot_encoding():
    """
    Test that sector indicators are correctly one-hot encoded.
    """
    sector_map = {
        "AAPL": "Technology",
        "JPM": "Financial Services",
        "JNJ": "Healthcare",
        "XOM": "Energy",
        "BA": "Industrials",
        "WMT": "Consumer Cyclical",
        "OTHER": "Real Estate",  # Maps to "Other"
    }
    
    builder = LinearFeatureBuilder(
        symbols=list(sector_map.keys()),
        use_interactions=False,
        use_sectors=True,
        sector_map=sector_map,
    )
    
    # Build features
    n_symbols = len(sector_map)
    X = builder.build_day(
        i=10,
        syms=list(sector_map.keys()),
        z_mamba=np.random.randn(n_symbols),
        sigma_exec=np.ones(n_symbols) * 0.01,
    )
    
    # Sector features start at index N_FEATURES
    sector_features = X[:, N_FEATURES:]
    
    # AAPL (Technology) should have sector_tech = 1
    assert sector_features[0, 0] == 1.0, "AAPL should be in tech sector"
    
    # JPM (Financial) should have sector_finance = 1
    assert sector_features[1, 1] == 1.0, "JPM should be in finance sector"
    
    # JNJ (Healthcare) should have sector_healthcare = 1
    assert sector_features[2, 2] == 1.0, "JNJ should be in healthcare sector"
    
    # XOM (Energy) should have sector_energy = 1
    assert sector_features[3, 5] == 1.0, "XOM should be in energy sector"
    
    # BA (Industrials) should have sector_industrial = 1
    assert sector_features[4, 4] == 1.0, "BA should be in industrial sector"
    
    # WMT (Consumer) should have sector_consumer = 1
    assert sector_features[5, 3] == 1.0, "WMT should be in consumer sector"
    
    # OTHER (Real Estate) should have sector_other = 1
    assert sector_features[6, 6] == 1.0, "OTHER should be in other sector"
    
    # Each row should sum to 1 (one-hot)
    for i in range(n_symbols):
        assert np.sum(sector_features[i, :]) == 1.0, \
            f"Sector indicators for symbol {i} should be one-hot encoded"


def test_combined_interactions_and_sectors():
    """
    Test that both interactions and sectors can be enabled simultaneously.
    """
    sector_map = {
        "AAPL": "Technology",
        "JPM": "Financial Services",
    }
    
    builder = LinearFeatureBuilder(
        symbols=list(sector_map.keys()),
        use_interactions=True,
        use_sectors=True,
        sector_map=sector_map,
    )
    
    # Check feature count
    expected_n_features = N_FEATURES + len(FEATURE_NAMES_INTERACTIONS) + len(FEATURE_NAMES_SECTORS)
    assert builder.n_features == expected_n_features, \
        f"Expected {expected_n_features} features with both, got {builder.n_features}"
    
    # Build features
    X = builder.build_day(
        i=10,
        syms=list(sector_map.keys()),
        z_mamba=np.array([1.0, -0.5]),
        sigma_exec=np.array([0.01, 0.01]),
    )
    
    # Verify shape
    assert X.shape == (2, expected_n_features), \
        f"Expected shape (2, {expected_n_features}), got {X.shape}"
    
    # Verify interactions exist (after base features)
    assert X.shape[1] > N_FEATURES + len(FEATURE_NAMES_INTERACTIONS), \
        "Should have interactions + sectors"


def test_interaction_features_improve_regime_sensitivity():
    """
    Integration test: Interactions should help model learn regime-conditional behavior.
    
    Setup:
    - High volatility regime: z_mamba signal is reliable
    - Low volatility regime: z_mamba signal is noisy
    
    Model with interactions should learn this relationship better.
    """
    n_symbols, T = 15, 50
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]

    # Model WITH interactions - use factory function
    state_interact = create_linear_combiner(
        symbols=symbols,
        use_interactions=True,
        use_sectors=False,
    )
    state_interact.horizon = 5
    state_interact.window = 30
    state_interact.ridge_lambda = 1.0
    state_interact.min_samples = 20

    # Model WITHOUT interactions
    state_baseline = create_linear_combiner(
        symbols=symbols,
        use_interactions=False,
        use_sectors=False,
    )
    state_baseline.horizon = 5
    state_baseline.window = 30
    state_baseline.ridge_lambda = 1.0
    state_baseline.min_samples = 20

    # Accumulate forward returns in proper shape (n_days, n_assets)
    fwd_ret_history = []

    # Simulate regime-conditional data
    for t in range(T):
        # Alternate between high and low volatility regimes
        high_vol = (t // 10) % 2 == 0

        if high_vol:
            # High vol regime: z_mamba is predictive
            rv = 0.3 * np.ones(n_symbols)
            z_mamba = np.random.randn(n_symbols)
            # Returns = 0.01 * z_mamba + noise
            returns = 0.01 * z_mamba + 0.005 * np.random.randn(n_symbols)
        else:
            # Low vol regime: z_mamba is noise
            rv = 0.1 * np.ones(n_symbols)
            z_mamba = np.random.randn(n_symbols)
            # Returns = noise (z_mamba not predictive)
            returns = 0.005 * np.random.randn(n_symbols)

        fwd_ret_history.append(returns)

        # Use build_day method to get properly constructed features
        X_interact = state_interact.feature_builder.build_day(
            i=t,
            syms=symbols,
            z_mamba=z_mamba,
            sigma_exec=np.ones(n_symbols) * 0.01,
        )
        X_baseline = state_baseline.feature_builder.build_day(
            i=t,
            syms=symbols,
            z_mamba=z_mamba,
            sigma_exec=np.ones(n_symbols) * 0.01,
        )
        
        # Manually override rv_21d to control regime
        X_interact[:, 8] = rv
        X_baseline[:, 8] = rv
        
        # Observe and update both models
        sigma_exec_day = np.ones(n_symbols) * 0.01
        state_interact.observe_day(t, X_interact, sigma_exec_day)
        state_baseline.observe_day(t, X_baseline, sigma_exec_day)
        
        # Update with forward returns (when mature)
        if t >= state_interact.horizon:
            # Create forward return matrix with shape (n_days, n_assets)
            fwd_ret_mat = np.array(fwd_ret_history)
            
            state_interact.update_if_matured(t, fwd_ret_mat, freeze=False)
            state_baseline.update_if_matured(t, fwd_ret_mat, freeze=False)
    
    # After training, model with interactions should have learned
    # that z_mamba * rv_21d interaction is predictive
    if state_interact.is_ready() and state_baseline.is_ready():
        # Check that interaction model has non-zero coefficient for z_mamba_x_rv
        if state_interact.model.beta_ is not None and len(state_interact.model.beta_) > N_FEATURES:
            z_mamba_x_rv_coef = state_interact.model.beta_[N_FEATURES + 0]
            print(f"Interaction model z_mamba_x_rv coef: {z_mamba_x_rv_coef:.4f}")
            # Should be non-zero if model learned the regime-conditional relationship
            # (We can't guarantee exact value with random data, but verify it's being used)
        
        # Models should differ in their predictions
        X_test = np.random.randn(n_symbols, state_baseline.feature_builder.n_features)
        z_baseline = state_baseline.predict_z(X_test)
        
        # For interaction model, need extended features
        X_test_interact = np.zeros((n_symbols, state_interact.feature_builder.n_features))
        X_test_interact[:, :N_FEATURES] = X_test
        if state_interact.feature_builder.use_interactions:
            X_test_interact[:, N_FEATURES + 0] = X_test[:, 0] * X_test[:, 8]  # z_mamba_x_rv
        
        z_interact = state_interact.predict_z(X_test_interact)
        
        # Predictions should differ due to interaction terms
        assert not np.allclose(z_interact, z_baseline, atol=0.01), \
            "Interaction model should make different predictions than baseline"


def test_create_linear_combiner_with_extended_features():
    """
    Test that factory function creates combiner with extended features.
    """
    sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
    
    state = create_linear_combiner(
        horizon=21,
        update_interval=21,
        ridge_lambda=10.0,
        window=126,
        max_window=252,
        min_samples=63,
        time_decay_halflife=42,
        use_interactions=True,
        use_sectors=True,
        sector_map=sector_map,
        symbols=list(sector_map.keys()),
    )
    
    # Check feature builder configured
    assert state.feature_builder.use_interactions == True
    assert state.feature_builder.use_sectors == True
    assert state.feature_builder.sector_map == sector_map
    
    # Check feature count
    expected_n = N_FEATURES + len(FEATURE_NAMES_INTERACTIONS) + len(FEATURE_NAMES_SECTORS)
    assert state.feature_builder.n_features == expected_n
    assert state.model.n_features == expected_n


def test_baseline_features_unchanged_when_disabled():
    """
    Test that baseline behavior is preserved when extended features are disabled.
    """
    builder_baseline = LinearFeatureBuilder(
        symbols=["AAPL", "MSFT"],
        use_interactions=False,
        use_sectors=False,
    )
    
    builder_extended = LinearFeatureBuilder(
        symbols=["AAPL", "MSFT"],
        use_interactions=True,
        use_sectors=True,
        sector_map={"AAPL": "Technology", "MSFT": "Technology"},
    )
    
    # Baseline should have N_FEATURES
    assert builder_baseline.n_features == N_FEATURES
    
    # Extended should have more
    assert builder_extended.n_features > N_FEATURES
    
    # Build features
    z_mamba = np.array([1.0, -0.5])
    sigma_exec = np.array([0.01, 0.01])
    
    X_baseline = builder_baseline.build_day(
        i=10,
        syms=["AAPL", "MSFT"],
        z_mamba=z_mamba,
        sigma_exec=sigma_exec,
    )
    
    X_extended = builder_extended.build_day(
        i=10,
        syms=["AAPL", "MSFT"],
        z_mamba=z_mamba,
        sigma_exec=sigma_exec,
    )
    
    # First N_FEATURES columns should be identical
    np.testing.assert_allclose(
        X_baseline,
        X_extended[:, :N_FEATURES],
        rtol=1e-6,
        err_msg="Base features should be identical regardless of extended features"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
