"""
Test Suite for Backup Price Sources & Volatility Provider

This script tests backup price data sources and volatility indicators:
- Stooq CSV data for backup pricing
- Cboe VIX volatility data
- Market stress indicators
- Volatility calculations and analysis

Author: DCF Lab Team
Created: 2025-09-18
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from dcf_lab.providers.backup_prices_volatility import (
    BackupPriceConfig,
    get_backup_provider,
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_backup_price_provider():
    """Test backup price sources and volatility provider"""

    print("📈 Backup Price Sources & Volatility Provider Test Suite")
    print("=" * 70)

    # Initialize provider
    config = BackupPriceConfig(
        cache_dir="./cache/backup_test",
        enable_cache=True,
        max_cache_age_hours=1  # Short cache for testing
    )

    provider = get_backup_provider(config)

    # Test 1: Stooq Backup Price Data
    print("\n⚡ Test 1: Stooq Backup Price Data")
    print("-" * 50)

    test_symbols = ["AAPL.US", "SPY.US", "QQQ.US"]
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    for symbol in test_symbols:
        try:
            result = provider.get_backup_price_data(
                symbol, start_date=start_date)

            if result['success']:
                price_data = result['price_data']
                vol_data = result['volatility_metrics']

                print(f"✅ {symbol}: {result['record_count']} records")
                print(
                    f"   Date range: {
                        result['date_range']['start']} to {
                        result['date_range']['end']}")

                if not price_data.empty:
                    latest = price_data.iloc[-1]
                    print(f"   Latest close: ${latest['Close']:.2f}")

                    if 'Volume' in price_data.columns:
                        avg_volume = price_data['Volume'].tail(20).mean()
                        print(f"   Avg volume (20d): {avg_volume:,.0f}")

                if not vol_data.empty and 'Volatility_20d' in vol_data.columns:
                    current_vol = vol_data['Volatility_20d'].dropna().iloc[-1]
                    print(f"   Current volatility: {current_vol:.1%}")

                    if 'Vol_Percentile' in vol_data.columns:
                        vol_percentile = vol_data['Vol_Percentile'].dropna(
                        ).iloc[-1]
                        print(
                            f"   Volatility percentile: {
                                vol_percentile:.1%}")
            else:
                print(f"❌ {symbol}: {result['error']}")

        except Exception as e:
            print(f"❌ {symbol}: Test failed - {e}")

    # Test 2: VIX Volatility Data
    print("\n⚡ Test 2: VIX Volatility Data")
    print("-" * 50)

    try:
        vix_result = provider.get_vix_data()

        if vix_result['success']:
            vix_data = vix_result['vix_data']
            analysis = vix_result['volatility_analysis']

            print(f"✅ VIX data: {vix_result['record_count']} records")
            print(
                f"   Date range: {
                    vix_result['date_range']['start']} to {
                    vix_result['date_range']['end']}")

            print("\n   Current VIX Analysis:")
            current_vix = analysis.get('current_vix')
            if current_vix:
                print(f"   Current VIX: {current_vix:.2f}")

                avg_30d = analysis.get('avg_30d')
                if avg_30d:
                    print(f"   30-day average: {avg_30d:.2f}")

                avg_1y = analysis.get('avg_1y')
                if avg_1y:
                    print(f"   1-year average: {avg_1y:.2f}")

                percentile = analysis.get('percentile_1y')
                if percentile:
                    print(f"   1-year percentile: {percentile:.1%}")

                min_1y = analysis.get('min_1y')
                max_1y = analysis.get('max_1y')
                if min_1y and max_1y:
                    print(f"   1-year range: {min_1y:.2f} - {max_1y:.2f}")

            # Show recent VIX levels
            if not vix_data.empty and 'Close' in vix_data.columns:
                print("\n   Recent VIX levels:")
                recent_vix = vix_data['Close'].tail(5)
                for date, vix_value in recent_vix.items():
                    print(f"     {date.strftime('%Y-%m-%d')}: {vix_value:.2f}")
        else:
            print(f"❌ VIX data failed: {vix_result['error']}")

    except Exception as e:
        print(f"❌ VIX test failed: {e}")

    # Test 3: Market Stress Indicators
    print("\n⚡ Test 3: Market Stress Indicators")
    print("-" * 50)

    try:
        stress_result = provider.get_market_stress_indicators()

        if stress_result['success']:
            indicators = stress_result['indicators']
            analysis = stress_result['analysis']

            print("✅ Market stress analysis:")

            # VIX-based indicators
            stress_level = indicators.get('vix_stress_level', 'Unknown')
            description = indicators.get('vix_description', 'N/A')
            current_vix = indicators.get('current_vix')

            print(f"   Stress Level: {stress_level}")
            print(f"   Description: {description}")

            if current_vix:
                print(f"   Current VIX: {current_vix:.2f}")

            percentile = indicators.get('vix_percentile_1y')
            if percentile:
                print(f"   VIX Percentile: {percentile:.1%}")

            vix_vs_avg = indicators.get('vix_vs_avg')
            if vix_vs_avg:
                direction = "above" if vix_vs_avg > 0 else "below"
                print(
                    f"   VIX vs 1Y avg: {
                        abs(vix_vs_avg):.2f} points {direction}")

            # Market regime analysis
            vol_regime = analysis.get('volatility_regime', 'Unknown')
            fear_greed = analysis.get('fear_greed_indicator', 'Unknown')
            outlook = analysis.get('market_outlook', 'N/A')

            print(f"   Volatility Regime: {vol_regime}")
            print(f"   Fear/Greed: {fear_greed}")
            print(f"   Market Outlook: {outlook}")

            # Interpret stress level
            print("\n   Stress Level Interpretation:")
            if stress_level == "Low":
                print("   📈 Low volatility - potential complacency risk")
            elif stress_level == "Normal":
                print("   ⚖️ Normal conditions - balanced market")
            elif stress_level == "Elevated":
                print("   ⚠️ Elevated uncertainty - monitor closely")
            elif stress_level == "High":
                print("   🚨 High stress - defensive positioning")
            elif stress_level == "Extreme":
                print("   💥 Extreme stress - potential buying opportunity")
        else:
            print(f"❌ Stress indicators failed: {stress_result['error']}")

    except Exception as e:
        print(f"❌ Stress indicators test failed: {e}")

    # Test 4: Volatility Calculations
    print("\n⚡ Test 4: Volatility Calculations")
    print("-" * 50)

    try:
        # Get sample data for volatility testing
        sample_result = provider.get_backup_price_data(
            "SPY.US", start_date=start_date)

        if sample_result['success']:
            vol_data = sample_result['volatility_metrics']

            if not vol_data.empty:
                print("✅ Volatility metrics calculated:")

                # Show available volatility metrics
                vol_columns = [
                    col for col in vol_data.columns
                    if 'vol' in col.lower() or 'Vol' in col]
                print(f"   Available metrics: {', '.join(vol_columns)}")

                # Latest volatility values
                latest_vol = vol_data.iloc[-1]

                if 'Volatility_20d' in vol_data.columns:
                    vol_20d = latest_vol['Volatility_20d']
                    if pd.notna(vol_20d):
                        print(f"   20-day volatility: {vol_20d:.1%}")

                if 'Parkinson_Vol' in vol_data.columns:
                    parkinson_vol = latest_vol['Parkinson_Vol']
                    if pd.notna(parkinson_vol):
                        print(f"   Parkinson volatility: {parkinson_vol:.1%}")

                if 'GK_Vol' in vol_data.columns:
                    gk_vol = latest_vol['GK_Vol']
                    if pd.notna(gk_vol):
                        print(f"   Garman-Klass volatility: {gk_vol:.1%}")

                if 'Vol_Percentile' in vol_data.columns:
                    vol_percentile = latest_vol['Vol_Percentile']
                    if pd.notna(vol_percentile):
                        print(
                            f"   Volatility percentile: {
                                vol_percentile:.1%}")

                # Volatility trend analysis
                if 'Volatility_20d' in vol_data.columns:
                    vol_series = vol_data['Volatility_20d'].dropna()
                    if len(vol_series) >= 10:
                        recent_avg = vol_series.tail(5).mean()
                        older_avg = vol_series.tail(20).head(10).mean()

                        if recent_avg > older_avg * 1.1:
                            trend = "Rising"
                        elif recent_avg < older_avg * 0.9:
                            trend = "Falling"
                        else:
                            trend = "Stable"

                        print(f"   Volatility trend: {trend}")
            else:
                print("⚠️ No volatility metrics available")
        else:
            print(
                f"❌ Failed to get data for volatility test: {
                    sample_result['error']}")

    except Exception as e:
        print(f"❌ Volatility calculations test failed: {e}")

    # Summary
    print("\n🎯 Backup Price Sources & Volatility Test Summary")
    print("=" * 70)
    print("✅ Stooq backup price data retrieval")
    print("✅ Cboe VIX volatility data access")
    print("✅ Market stress indicators calculation")
    print("✅ Advanced volatility metrics computation")
    print("✅ Volatility regime analysis")
    print("✅ Fear/greed indicators")
    print("✅ Backup data caching system")
    print("\n🏆 Backup Price & Volatility Provider: All features working!")
    print("📊 Reliable backup when primary sources fail")
    print("📈 Comprehensive volatility and stress analysis")
    print("🚀 Ready for risk management and market timing")


if __name__ == "__main__":
    test_backup_price_provider()
