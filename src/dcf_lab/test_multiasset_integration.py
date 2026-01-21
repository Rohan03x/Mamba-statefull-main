"""
Test Multi-Asset Correlation Integration with AI Price Forecasting

This script tests the integration of multi-asset correlation modeling
with the main AI price forecasting system.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    # Try relative imports first
    from .ai_price_forecast import ai_price_forecast
    from .ml_advanced import MultiAssetForecaster
except ImportError:
    try:
        # Try importing with sys.path manipulation
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from .ai_price_forecast import ai_price_forecast
        from .ml_advanced import MultiAssetForecaster
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory")
        exit(1)


def create_synthetic_multiasset_data(n_assets=5, n_days=500):
    """Create synthetic multi-asset return data"""
    # Use modern numpy random generator instead of deprecated global state
    rng = np.random.default_rng(42)
    # Also set legacy global seed for compatibility
    np.random.seed(42)

    # Asset names
    asset_names = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'][:n_assets]

    # Create correlation structure
    base_correlation = 0.3
    correlation_matrix = np.full((n_assets, n_assets), base_correlation)
    np.fill_diagonal(correlation_matrix, 1.0)

    # Add some variation
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation_matrix[i, j] = correlation_matrix[j,
                                                          i] = base_correlation + rng.normal(0, 0.1)

    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    # Normalize to correlation matrix
    std_dev = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix = correlation_matrix / np.outer(std_dev, std_dev)
    np.fill_diagonal(correlation_matrix, 1.0)

    # Generate correlated returns
    L = np.linalg.cholesky(correlation_matrix)
    independent_returns = rng.normal(0, 0.02, (n_days, n_assets))
    correlated_returns = independent_returns @ L.T

    # Create dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create DataFrame
    returns_df = pd.DataFrame(
        correlated_returns,
        index=dates,
        columns=asset_names)

    return returns_df


def create_synthetic_price_data(ticker='AAPL', n_days=500):
    """Create synthetic price data for forecasting"""
    # Use modern numpy random generator instead of deprecated global state
    rng = np.random.default_rng(42)
    # Also set legacy global seed for compatibility
    np.random.seed(42)

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate price path
    initial_price = 150.0
    returns = rng.normal(0.001, 0.02, n_days)  # Slight positive drift
    log_returns = np.cumsum(returns)
    prices = initial_price * np.exp(log_returns)

    # Create basic technical features
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['high'] = df['close'] * (1 + np.abs(rng.normal(0, 0.01, len(df))))
    df['low'] = df['close'] * (1 - np.abs(rng.normal(0, 0.01, len(df))))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['volume'] = rng.integers(1000000, 10000000, len(df))

    # Add some technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = 50 + rng.normal(0, 15, len(df))  # Simplified RSI
    df['macd'] = rng.normal(0, 2, len(df))  # Simplified MACD

    # Forward fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


def test_standalone_multiasset():
    """Test standalone multi-asset correlation forecasting"""
    print("=== Testing Standalone Multi-Asset System ===")

    # Create synthetic data
    multiasset_data = create_synthetic_multiasset_data(n_assets=4, n_days=300)
    print(f"Created multi-asset data: {multiasset_data.shape}")
    print(f"Assets: {list(multiasset_data.columns)}")

    # Test correlation forecasting
    try:
        forecaster = MultiAssetForecaster(
            correlation_method='ewma',
            n_factors=3,
            lookback_window=200
        )

        forecaster.fit(multiasset_data)
        print("✓ Multi-asset model fitted successfully")

        # Generate correlation forecast
        correlation_forecast = forecaster.forecast_correlation(horizon=5)
        print("✓ Correlation forecast generated")
        print(
            f"  Correlation matrix shape: {
                correlation_forecast.correlation_matrix.shape}")
        print(f"  Methodology: {correlation_forecast.methodology}")

        # Portfolio risk forecast
        n_assets = len(multiasset_data.columns)
        equal_weights = np.ones(n_assets) / n_assets
        portfolio_risk = forecaster.portfolio_risk_forecast(
            equal_weights, horizon=5)
        print("✓ Portfolio risk forecast generated")
        print(
            f"  Annual volatility: {
                portfolio_risk['portfolio_volatility_annual']:.2%}")

        return True

    except Exception as e:
        print(f"✗ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_forecasting():
    """Test integrated AI forecasting with multi-asset correlation"""
    print("\n=== Testing Integrated AI Forecasting with Multi-Asset ===")

    # Create synthetic data
    ticker = 'AAPL'
    price_data = create_synthetic_price_data(ticker=ticker, n_days=300)
    multiasset_data = create_synthetic_multiasset_data(n_assets=4, n_days=300)

    print(f"Created price data for {ticker}: {price_data.shape}")
    print(f"Created multi-asset data: {multiasset_data.shape}")

    # Test without multi-asset correlation
    try:
        print("\n--- Testing without multi-asset correlation ---")
        forecast_simple, results_simple = ai_price_forecast(
            df=price_data,
            ticker=ticker,
            horizon=5,
            model_type='ensemble',
            use_quantiles=True,
            use_options_anchoring=False,  # Disable for testing
            use_probability_calibration=False,  # Disable for testing
            use_multiasset_correlation=False
        )

        print("✓ Simple AI forecast completed")
        print(f"  Forecast length: {len(forecast_simple)}")
        print(f"  Model type: {results_simple.get('model_type', 'unknown')}")

    except Exception as e:
        print(f"✗ Simple forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with multi-asset correlation
    try:
        print("\n--- Testing with multi-asset correlation ---")
        forecast_multiasset, results_multiasset = ai_price_forecast(
            df=price_data,
            ticker=ticker,
            horizon=5,
            model_type='ensemble',
            use_quantiles=True,
            use_options_anchoring=False,  # Disable for testing
            use_probability_calibration=False,  # Disable for testing
            use_multiasset_correlation=True,
            multiasset_data=multiasset_data,
            correlation_method='ewma'
        )

        print("✓ Multi-asset AI forecast completed")
        print(f"  Forecast length: {len(forecast_multiasset)}")
        print(
            f"  Model type: {
                results_multiasset.get(
                    'model_type',
                    'unknown')}")

        # Check for multi-asset results
        if 'correlation_forecast' in results_multiasset:
            print("✓ Correlation forecast included in results")
            corr_info = results_multiasset['correlation_forecast']
            print(
                f"  Correlation matrix size: {
                    len(
                        corr_info['correlation_matrix'])}x{
                    len(
                        corr_info['correlation_matrix'][0])}")
            print(f"  Methodology: {corr_info['methodology']}")
            print(f"  Number of assets: {corr_info['n_assets']}")

        if 'multiasset_risk' in results_multiasset:
            print("✓ Multi-asset risk metrics included")
            risk_info = results_multiasset['multiasset_risk']
            print(
                f"  Portfolio volatility: {
                    risk_info.get(
                        'portfolio_volatility_annual',
                        'N/A')}")
            print(
                f"  Correlation method: {
                    risk_info.get(
                        'correlation_method',
                        'N/A')}")

        return True

    except Exception as e:
        print(f"✗ Multi-asset forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Testing Multi-Asset Correlation Integration")
    print("=" * 50)

    # Test 1: Standalone multi-asset system
    test1_success = test_standalone_multiasset()

    # Test 2: Integrated forecasting system
    test2_success = test_integrated_forecasting()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Standalone Multi-Asset: {'✓ PASS' if test1_success else '✗ FAIL'}")
    print(f"Integrated Forecasting: {'✓ PASS' if test2_success else '✗ FAIL'}")

    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED! Multi-asset integration successful.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
