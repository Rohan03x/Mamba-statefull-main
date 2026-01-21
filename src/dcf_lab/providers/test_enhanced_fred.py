"""
Test Suite for Enhanced FRED Economic Data Provider

This test suite validates all components of the enhanced FRED provider:
- Yield curve construction and analysis
- Economic indicators data
- Risk-free rate calculation
- Economic cycle detection
- Integration tests

Author: DCF Lab Team
Created: 2025-09-18
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dcf_lab.providers.enhanced_fred import (
    EconomicIndicatorsProvider,
    EnhancedFREDConfig,
    EnhancedFREDProvider,
    YieldCurveBuilder,
    get_enhanced_fred_provider,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEnhancedFREDConfig:
    """Test enhanced FRED configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EnhancedFREDConfig()

        assert config.api_key is None
        assert config.base_url == "https://api.stlouisfed.org/fred"
        assert config.enable_cache is True
        assert config.max_cache_age_hours == 6
        assert config.timeout == 30
        assert config.retries == 3
        assert abs(config.rate_limit_delay - 0.1) < 1e-10

    def test_custom_config(self):
        """Test custom configuration"""
        config = EnhancedFREDConfig(
            api_key="test_key",
            enable_cache=False,
            timeout=60
        )

        assert config.api_key == "test_key"
        assert config.enable_cache is False
        assert config.timeout == 60


class TestYieldCurveBuilder:
    """Test yield curve building and analysis"""

    def setup_method(self):
        """Set up test environment"""
        self.config = EnhancedFREDConfig()
        self.yield_curve = YieldCurveBuilder(self.config)

    def test_treasury_series_mapping(self):
        """Test Treasury series ID mapping"""
        assert '1_month' in self.yield_curve.treasury_series
        assert '10_year' in self.yield_curve.treasury_series
        assert '30_year' in self.yield_curve.treasury_series

        assert self.yield_curve.treasury_series['10_year'] == 'DGS10'
        assert self.yield_curve.treasury_series['30_year'] == 'DGS30'

    def test_maturity_years_mapping(self):
        """Test maturity years conversion"""
        assert abs(self.yield_curve.maturity_years['3_month'] - 0.25) < 1e-10
        assert self.yield_curve.maturity_years['1_year'] == 1
        assert self.yield_curve.maturity_years['10_year'] == 10
        assert self.yield_curve.maturity_years['30_year'] == 30

    def test_get_treasury_yields(self):
        """Test Treasury yields data retrieval"""
        start_date = '2023-01-01'
        end_date = '2023-12-31'

        yields = self.yield_curve.get_treasury_yields(start_date, end_date)

        assert isinstance(yields, pd.DataFrame)
        assert not yields.empty
        assert '10_year' in yields.columns
        assert '30_year' in yields.columns

        # Check data types and ranges
        for col in yields.columns:
            assert yields[col].dtype in [np.float64, float]
            # Treasury yields should be positive and reasonable
            assert yields[col].min() >= 0
            assert yields[col].max() <= 20  # Reasonable upper bound

    def test_interpolate_yield_curve(self):
        """Test yield curve interpolation"""
        # Create test yield data
        test_yields = pd.Series({
            '3_month': 1.5,
            '1_year': 2.0,
            '5_year': 3.0,
            '10_year': 3.5,
            '30_year': 4.0
        })

        target_maturities = [0.5, 2, 7, 15]
        interpolated = self.yield_curve.interpolate_yield_curve(
            test_yields, target_maturities)

        assert isinstance(interpolated, pd.Series)
        assert len(interpolated) == len(target_maturities)

        # Check interpolated values are reasonable
        assert 1.5 <= interpolated[0.5] <= 2.0  # Between 3M and 1Y
        assert 2.0 <= interpolated[2] <= 3.0    # Between 1Y and 5Y

    def test_interpolate_yield_curve_insufficient_data(self):
        """Test interpolation with insufficient data points"""
        test_yields = pd.Series({'10_year': 3.5})  # Only one data point

        target_maturities = [1, 5, 20]
        interpolated = self.yield_curve.interpolate_yield_curve(
            test_yields, target_maturities)

        assert isinstance(interpolated, pd.Series)
        assert len(interpolated) == len(target_maturities)
        # Should return NaN values due to insufficient data
        assert interpolated.isna().all()

    def test_calculate_forward_rates(self):
        """Test forward rate calculation"""
        # Create test spot rates
        maturities = [1, 2, 5, 10]
        spot_rates = pd.Series([2.0, 2.5, 3.0, 3.5], index=maturities)

        forward_rates = self.yield_curve.calculate_forward_rates(
            spot_rates, maturities)

        assert isinstance(forward_rates, pd.Series)
        assert len(forward_rates) == len(maturities) - 1

        # Check forward rates are calculated correctly
        # Forward rate from year 1 to 2: (2.5*2 - 2.0*1) / (2-1) = 3.0
        expected_1y1y = (2.5 * 2 - 2.0 * 1) / (2 - 1)
        assert abs(forward_rates.iloc[0] - expected_1y1y) < 0.01

    def test_analyze_yield_curve_shape_normal(self):
        """Test yield curve shape analysis - normal curve"""
        test_yields = pd.Series({
            '3_month': 1.5,
            '2_year': 2.5,
            '10_year': 3.5,
            '30_year': 4.0
        })

        analysis = self.yield_curve.analyze_yield_curve_shape(test_yields)

        assert isinstance(analysis, dict)
        assert 'curve_type' in analysis
        assert 'slope' in analysis
        assert 'level' in analysis
        assert 'inversion_points' in analysis
        assert 'steepness_measures' in analysis

        # Should be normal curve (10Y > 3M)
        assert analysis['curve_type'] in ['normal', 'steep_normal']
        assert abs(analysis['slope'] - 2.0) < 1e-10  # 3.5 - 1.5
        assert abs(analysis['level'] - 2.875) < 1e-10  # Average of rates
        assert analysis['inversion_points'] == []  # No inversions

    def test_analyze_yield_curve_shape_inverted(self):
        """Test yield curve shape analysis - inverted curve"""
        test_yields = pd.Series({
            '3_month': 4.5,
            '2_year': 3.5,
            '10_year': 3.0,
            '30_year': 3.2
        })

        analysis = self.yield_curve.analyze_yield_curve_shape(test_yields)

        assert analysis['curve_type'] == 'inverted'
        assert analysis['slope'] == -1.5  # 3.0 - 4.5
        # Should detect inversions
        assert len(analysis['inversion_points']) > 0


class TestEconomicIndicatorsProvider:
    """Test economic indicators provider"""

    def setup_method(self):
        """Set up test environment"""
        self.config = EnhancedFREDConfig()
        self.indicators = EconomicIndicatorsProvider(self.config)

    def test_indicator_series_mapping(self):
        """Test economic indicator series mapping"""
        assert 'gdp_real' in self.indicators.indicator_series
        assert 'unemployment_rate' in self.indicators.indicator_series
        assert 'cpi_all' in self.indicators.indicator_series
        assert 'fed_funds_rate' in self.indicators.indicator_series

        assert self.indicators.indicator_series['gdp_real'] == 'GDPC1'
        assert self.indicators.indicator_series['unemployment_rate'] == 'UNRATE'

    def test_get_indicator_data(self):
        """Test economic indicator data retrieval"""
        indicators = ['gdp_real', 'unemployment_rate', 'cpi_all']
        start_date = '2023-01-01'
        end_date = '2023-12-31'

        data = self.indicators.get_indicator_data(
            indicators, start_date, end_date)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

        for indicator in indicators:
            assert indicator in data.columns
            assert data[indicator].dtype in [np.float64, float]
            # Should have some non-null values
            assert data[indicator].notna().any()

    def test_get_indicator_data_invalid_indicators(self):
        """Test with invalid indicator names"""
        indicators = ['invalid_indicator', 'gdp_real']

        data = self.indicators.get_indicator_data(indicators)

        assert isinstance(data, pd.DataFrame)
        # Should only include valid indicators
        assert 'gdp_real' in data.columns
        assert 'invalid_indicator' not in data.columns

    def test_calculate_economic_cycles(self):
        """Test economic cycle calculation"""
        # Create test GDP data with a recession pattern
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')

        # Simulate recession in 2020 Q2-Q3
        gdp_values = [
            100,
            95,
            92,
            96,
            102,
            105,
            108,
            110,
            112,
            115,
            118,
            120,
            122,
            125,
            128,
            130]
        gdp_data = pd.Series(gdp_values, index=dates[:len(gdp_values)])

        cycles = self.indicators.calculate_economic_cycles(gdp_data)

        assert isinstance(cycles, dict)
        assert 'recession_periods' in cycles
        assert 'expansion_periods' in cycles
        assert 'current_cycle_phase' in cycles
        assert 'cycle_statistics' in cycles

        assert isinstance(cycles['recession_periods'], list)
        assert isinstance(cycles['expansion_periods'], list)
        assert cycles['current_cycle_phase'] in [
            'recession', 'expansion', 'slowdown', 'unknown']


class TestEnhancedFREDProvider:
    """Test main enhanced FRED provider"""

    def setup_method(self):
        """Set up test environment"""
        self.provider = get_enhanced_fred_provider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert isinstance(self.provider, EnhancedFREDProvider)
        assert isinstance(self.provider.config, EnhancedFREDConfig)
        assert isinstance(self.provider.yield_curve, YieldCurveBuilder)
        assert isinstance(self.provider.indicators, EconomicIndicatorsProvider)

    def test_get_yield_curve_data(self):
        """Test yield curve data retrieval"""
        result = self.provider.get_yield_curve_data()

        assert isinstance(result, dict)
        assert 'date' in result
        assert 'retrieved_at' in result
        assert 'treasury_yields' in result
        assert 'interpolated_curve' in result
        assert 'forward_rates' in result
        assert 'curve_analysis' in result
        assert 'success' in result

        if result['success']:
            assert isinstance(result['treasury_yields'], dict)
            assert isinstance(result['interpolated_curve'], dict)
            assert isinstance(result['curve_analysis'], dict)

            # Check curve analysis structure
            curve_analysis = result['curve_analysis']
            assert 'curve_type' in curve_analysis
            assert 'slope' in curve_analysis
            assert 'level' in curve_analysis
            assert 'inversion_points' in curve_analysis

    def test_get_yield_curve_data_specific_date(self):
        """Test yield curve data for specific date"""
        test_date = '2023-12-01'
        result = self.provider.get_yield_curve_data(test_date)

        assert isinstance(result, dict)
        assert result['date'] == test_date

    def test_get_economic_dashboard(self):
        """Test economic dashboard"""
        result = self.provider.get_economic_dashboard(lookback_months=6)

        assert isinstance(result, dict)
        assert 'retrieved_at' in result
        assert 'lookback_months' in result
        assert 'indicators' in result
        assert 'growth_metrics' in result
        assert 'inflation_metrics' in result
        assert 'employment_metrics' in result
        assert 'monetary_policy' in result
        assert 'cycle_analysis' in result
        assert 'success' in result

        assert result['lookback_months'] == 6

        if result['success']:
            assert isinstance(result['indicators'], dict)
            assert isinstance(result['growth_metrics'], dict)
            assert isinstance(result['inflation_metrics'], dict)
            assert isinstance(result['employment_metrics'], dict)
            assert isinstance(result['monetary_policy'], dict)

    def test_calculate_risk_free_rate_default(self):
        """Test risk-free rate calculation with default maturity"""
        result = self.provider.calculate_risk_free_rate()

        assert isinstance(result, dict)
        assert 'maturity_years' in result
        assert 'calculated_at' in result
        assert 'risk_free_rate' in result
        assert 'calculation_method' in result
        assert 'curve_data' in result
        assert 'success' in result

        assert result['maturity_years'] == 10  # Default

        if result['success']:
            assert isinstance(result['risk_free_rate'], (int, float))
            assert result['risk_free_rate'] >= 0
            assert result['risk_free_rate'] <= 20  # Reasonable upper bound
            assert result['calculation_method'] in [
                'direct_10_year', 'interpolated', 'fallback']

    def test_calculate_risk_free_rate_custom_maturity(self):
        """Test risk-free rate calculation with custom maturity"""
        result = self.provider.calculate_risk_free_rate(maturity_years=5)

        assert result['maturity_years'] == 5

    def test_determine_policy_stance(self):
        """Test monetary policy stance determination"""
        # Test tightening scenario
        tightening_data = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0])
        stance = self.provider._determine_policy_stance(tightening_data)
        assert stance == 'tightening'

        # Test easing scenario
        easing_data = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        stance = self.provider._determine_policy_stance(easing_data)
        assert stance == 'easing'

        # Test neutral scenario
        neutral_data = pd.Series([2.0, 2.1, 2.0, 1.9, 2.0])
        stance = self.provider._determine_policy_stance(neutral_data)
        assert stance == 'neutral'

        # Test insufficient data
        short_data = pd.Series([2.0, 2.1])
        stance = self.provider._determine_policy_stance(short_data)
        assert stance == 'unknown'


class TestIntegration:
    """Integration tests for enhanced FRED provider"""

    def setup_method(self):
        """Set up test environment"""
        self.provider = get_enhanced_fred_provider()

    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow"""
        # Get yield curve
        curve_result = self.provider.get_yield_curve_data()
        assert 'success' in curve_result

        # Get economic dashboard
        dashboard_result = self.provider.get_economic_dashboard()
        assert 'success' in dashboard_result

        # Calculate risk-free rate
        rf_result = self.provider.calculate_risk_free_rate()
        assert 'success' in rf_result

        # If all successful, check consistency
        if (curve_result['success'] and dashboard_result['success']
                and rf_result['success']):
            # Risk-free rate should be consistent with yield curve
            treasury_yields = curve_result['treasury_yields']
            risk_free_rate = rf_result['risk_free_rate']

            if '10_year' in treasury_yields:
                # Should be reasonably close (within 1% for test data)
                assert abs(treasury_yields['10_year'] - risk_free_rate) <= 1.0

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with invalid config
        config = EnhancedFREDConfig(api_key="invalid_key")
        provider = EnhancedFREDProvider(config)

        # Should handle errors gracefully
        result = provider.get_yield_curve_data()
        assert 'success' in result
        assert 'error' in result

        # Test with invalid date
        result = provider.get_yield_curve_data('invalid-date')
        assert isinstance(result, dict)
        assert 'error' in result or 'success' in result

    def test_data_consistency(self):
        """Test data consistency across multiple calls"""
        # Multiple calls should return consistent data structure
        result1 = self.provider.get_yield_curve_data()
        result2 = self.provider.get_yield_curve_data()

        # Should have same structure
        assert result1.keys() == result2.keys()

        if result1['success'] and result2['success']:
            # Treasury yields structure should be consistent
            assert result1['treasury_yields'].keys(
            ) == result2['treasury_yields'].keys()

    def test_factory_function(self):
        """Test factory function"""
        # Test with default config
        provider1 = get_enhanced_fred_provider()
        assert isinstance(provider1, EnhancedFREDProvider)

        # Test with custom config
        config = EnhancedFREDConfig(timeout=60)
        provider2 = get_enhanced_fred_provider(config)
        assert isinstance(provider2, EnhancedFREDProvider)
        assert provider2.config.timeout == 60


def test_example_usage():
    """Test the example usage from the module"""
    # This tests the if __name__ == "__main__" block functionality
    provider = get_enhanced_fred_provider()

    # Test yield curve
    curve_result = provider.get_yield_curve_data()
    assert isinstance(curve_result, dict)

    # Test economic dashboard
    dashboard = provider.get_economic_dashboard()
    assert isinstance(dashboard, dict)

    # Test risk-free rate
    rf_result = provider.calculate_risk_free_rate(10)
    assert isinstance(rf_result, dict)


if __name__ == "__main__":
    # Run a simple test when executed directly
    print("Running Enhanced FRED Provider Tests...")

    # Create provider
    provider = get_enhanced_fred_provider()

    # Test basic functionality
    print("\n1. Testing yield curve...")
    curve_result = provider.get_yield_curve_data()
    print(f"   Success: {curve_result['success']}")

    print("\n2. Testing economic dashboard...")
    dashboard = provider.get_economic_dashboard()
    print(f"   Success: {dashboard['success']}")

    print("\n3. Testing risk-free rate...")
    rf_result = provider.calculate_risk_free_rate()
    print(f"   Success: {rf_result['success']}")

    print("\n✅ Enhanced FRED Provider tests completed!")
    print("💡 Run with pytest for comprehensive testing")
