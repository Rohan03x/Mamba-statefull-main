"""
Demo script for ensemble forecasting system
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def demo_ensemble_forecasting():
    """
    Demonstrate the ensemble forecasting system with sample data
    """
    print("=== ENSEMBLE FORECASTING DEMO ===")

    # Create sample data (you would replace this with real data)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    rng = np.random.default_rng(42)

    # Simulate realistic price data with trend and volatility
    returns = rng.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [100.0]  # Starting price

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create DataFrame with features
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': rng.lognormal(15, 0.5, len(dates)),
        'high': [p * (1 + abs(rng.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(rng.normal(0, 0.01))) for p in prices],
    })

    # Add technical indicators
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
    df['rsi'] = 50 + rng.normal(0, 10, len(df))  # Simplified RSI
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()

    # Add moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['price_to_sma_10'] = df['close'] / df['sma_10']
    df['price_to_sma_20'] = df['close'] / df['sma_20']
    df['price_to_sma_50'] = df['close'] / df['sma_50']

    # Add log returns
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5)) / 5

    # Add sentiment features (simulated)
    df['sentiment_score'] = rng.normal(0, 0.1, len(df))
    df['sentiment_positive'] = rng.uniform(0.2, 0.8, len(df))
    df['sentiment_negative'] = 1 - df['sentiment_positive']
    df['news_volume'] = rng.poisson(5, len(df))
    df['sentiment_volatility'] = rng.uniform(0.1, 0.5, len(df))

    df.set_index('date', inplace=True)
    df.dropna(inplace=True)

    print(f"Created sample dataset with {len(df)} observations")
    print(f"Features available: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Test ensemble forecasting
    try:
        from .ai_price_forecast import ai_price_forecast

        print("\n=== TESTING ENSEMBLE FORECASTING ===")

        # Import config
        from .ai_price_forecast import ForecastConfig

        # Create configuration for ensemble forecasting
        config = ForecastConfig(
            horizon=10,
            model_type='ensemble',
            use_multihorizon=True,
            use_quantiles=True
        )

        # Generate forecast using ensemble
        forecast_series, results = ai_price_forecast(
            df=df,
            ticker="DEMO",
            config=config
        )

        print("\nForecast Results:")
        print(f"Model type: {results.get('model_type', 'Unknown')}")
        print(f"Forecast horizon: {results.get('horizon', 'Unknown')} days")
        print(
            f"Training samples: {
                results.get(
                    'training_samples',
                    'Unknown')}")
        print(f"Features used: {len(results.get('features_used', []))}")
        print(f"Last price: ${results.get('last_price', 0):.2f}")

        if 'ensemble_components' in results:
            print(
                f"Ensemble components: {
                    ', '.join(
                        results['ensemble_components'])}")

        # Print forecast
        print("\nPrice Forecasts:")
        for date, price in forecast_series.head(5).items():
            print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")

        # Visualize results
        plot_ensemble_results(df, forecast_series, results)

    except Exception as e:
        print(f"Error during ensemble testing: {e}")
        import traceback
        traceback.print_exc()


def plot_ensemble_results(historical_df, forecast_series, results):
    """
    Plot ensemble forecasting results with uncertainty bands
    """
    try:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Price forecast with historical data
        _plot_price_forecast(ax1, historical_df, forecast_series, results)

        # Plot 2: Log returns forecast
        _plot_returns_forecast(ax2, results)

        plt.tight_layout()
        plt.show()

        print("\nVisualization completed!")

    except Exception as e:
        print(f"Error creating visualization: {e}")


def _plot_price_forecast(ax, historical_df, forecast_series, results):
    """Plot price forecast with historical data and uncertainty bands"""
    historical_prices = historical_df['close'].tail(60)  # Last 60 days

    ax.plot(historical_prices.index, historical_prices.values,
            label='Historical Prices', color='blue', linewidth=2)
    ax.plot(
        forecast_series.index,
        forecast_series.values,
        label='Ensemble Forecast',
        color='red',
        linewidth=2,
        linestyle='--')

    # Add uncertainty bands if available
    _add_uncertainty_bands(ax, forecast_series, results)

    ax.set_title('Ensemble Price Forecast', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _add_uncertainty_bands(ax, forecast_series, results):
    """Add uncertainty bands to price forecast plot"""
    if 'quantile_forecasts' not in results:
        return

    quantiles = results['quantile_forecasts']

    if 'q_10' not in quantiles or 'q_90' not in quantiles:
        return

    # Convert quantile log returns to prices
    last_price = results.get('last_price', 100.0)

    lower_prices = []
    upper_prices = []
    current_lower = last_price
    current_upper = last_price

    max_steps = min(
        len(forecast_series), len(
            quantiles['q_10'][0]), len(
            quantiles['q_90'][0]))

    for i in range(max_steps):
        current_lower *= np.exp(quantiles['q_10'][0][i])
        current_upper *= np.exp(quantiles['q_90'][0][i])
        lower_prices.append(current_lower)
        upper_prices.append(current_upper)

    if lower_prices and upper_prices:
        ax.fill_between(forecast_series.index[:len(lower_prices)],
                        lower_prices, upper_prices,
                        alpha=0.3, color='red', label='80% Confidence Band')


def _plot_returns_forecast(ax, results):
    """Plot forecasted log returns"""
    if 'forecast_returns' not in results:
        return

    returns = results['forecast_returns']
    ax.bar(range(len(returns)), returns, alpha=0.7, color='green')
    ax.set_title('Forecasted Log Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=12)
    ax.set_ylabel('Log Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)


if __name__ == "__main__":
    demo_ensemble_forecasting()
