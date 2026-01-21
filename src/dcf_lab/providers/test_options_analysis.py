"""
Test Suite for Options Analysis Provider

This module provides comprehensive tests for the options analysis functionality
including Black-Scholes pricing, Greeks calculation, and options flow analysis.
"""

import os

# Import the components to test
import sys

import numpy as np
import pytest
from options_analysis import (
    BlackScholesCalculator,
    OptionsAnalysisConfig,
    OptionsDataAnalyzer,
    get_options_analysis_provider,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOptionsAnalysisConfig:
    """Test options analysis configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = OptionsAnalysisConfig()

        assert config.cache_dir == "./cache/options_data"
        assert config.enable_cache is True
        assert config.max_cache_age_hours == 1
        assert config.timeout == 30
        assert config.retries == 3
        assert config.rate_limit_delay == 0.5

        # URLs
        assert "cboe.com" in config.cboe_base_url
        assert "finance.yahoo.com" in config.options_data_url

        # Analysis settings
        assert config.risk_free_rate == 0.05
        assert config.dividend_yield == 0.02
        assert config.min_volume_threshold == 100
        assert config.min_open_interest == 50
        assert config.max_days_to_expiry == 365

    def test_custom_config(self):
        """Test custom configuration"""
        config = OptionsAnalysisConfig(
            cache_dir="./custom_options_cache",
            enable_cache=False,
            max_cache_age_hours=2,
            risk_free_rate=0.03,
            dividend_yield=0.015
        )

        assert config.cache_dir == "./custom_options_cache"
        assert config.enable_cache is False
        assert config.max_cache_age_hours == 2
        assert config.risk_free_rate == 0.03
        assert config.dividend_yield == 0.015


class TestBlackScholesCalculator:
    """Test Black-Scholes calculator functionality"""

    @pytest.fixture
    def bs_calc(self):
        """Create Black-Scholes calculator for testing"""
        return BlackScholesCalculator()

    def test_call_option_pricing(self, bs_calc):
        """Test call option pricing"""
        # Test parameters
        S = 100.0  # Current price
        K = 100.0  # Strike price
        T = 0.25   # 3 months to expiration
        r = 0.05   # 5% risk-free rate
        sigma = 0.20  # 20% volatility

        call_price = bs_calc.black_scholes_call(S, K, T, r, sigma)

        # ATM call with 3 months should have positive time value
        assert call_price > 0
        assert call_price < S  # Should be less than stock price

        # Test ITM call
        itm_call = bs_calc.black_scholes_call(S, K - 10, T, r, sigma)
        assert itm_call > call_price  # ITM should be worth more

        # Test OTM call
        otm_call = bs_calc.black_scholes_call(S, K + 10, T, r, sigma)
        assert otm_call < call_price  # OTM should be worth less

    def test_put_option_pricing(self, bs_calc):
        """Test put option pricing"""
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma = 0.20

        put_price = bs_calc.black_scholes_put(S, K, T, r, sigma)

        # ATM put should have positive time value
        assert put_price > 0
        assert put_price < K  # Should be less than strike

        # Test ITM put
        itm_put = bs_calc.black_scholes_put(S, K + 10, T, r, sigma)
        assert itm_put > put_price  # ITM should be worth more

        # Test OTM put
        otm_put = bs_calc.black_scholes_put(S, K - 10, T, r, sigma)
        assert otm_put < put_price  # OTM should be worth less

    def test_put_call_parity(self, bs_calc):
        """Test put-call parity relationship"""
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma = 0.20
        q = 0.02  # Dividend yield

        call_price = bs_calc.black_scholes_call(S, K, T, r, sigma, q)
        put_price = bs_calc.black_scholes_put(S, K, T, r, sigma, q)

        # Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
        left_side = call_price - put_price
        right_side = S * np.exp(-q * T) - K * np.exp(-r * T)

        # Should be approximately equal (within small tolerance)
        assert abs(left_side - right_side) < 0.01

    def test_implied_volatility_calculation(self, bs_calc):
        """Test implied volatility calculation"""
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma_true = 0.25

        # Calculate theoretical price with known volatility
        market_price = bs_calc.black_scholes_call(S, K, T, r, sigma_true)

        # Calculate implied volatility from market price
        implied_vol = bs_calc.calculate_implied_volatility(
            market_price, S, K, T, r, 'call'
        )

        # Should recover the original volatility
        assert abs(implied_vol - sigma_true) < 0.001

    def test_greeks_calculation(self, bs_calc):
        """Test Greeks calculations"""
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma = 0.20

        # Calculate all Greeks
        call_delta = bs_calc.calculate_delta(S, K, T, r, sigma, 'call')
        put_delta = bs_calc.calculate_delta(S, K, T, r, sigma, 'put')
        gamma = bs_calc.calculate_gamma(S, K, T, r, sigma)
        call_theta = bs_calc.calculate_theta(S, K, T, r, sigma, 'call')
        put_theta = bs_calc.calculate_theta(S, K, T, r, sigma, 'put')
        vega = bs_calc.calculate_vega(S, K, T, r, sigma)
        call_rho = bs_calc.calculate_rho(S, K, T, r, sigma, 'call')
        put_rho = bs_calc.calculate_rho(S, K, T, r, sigma, 'put')

        # Test delta ranges and relationship
        assert 0 < call_delta < 1  # Call delta between 0 and 1
        assert -1 < put_delta < 0  # Put delta between -1 and 0
        assert abs(call_delta - put_delta - 1) < 0.01  # Delta relationship

        # Test gamma (should be positive for both calls and puts)
        assert gamma > 0

        # Test theta (time decay - should be negative for long options)
        assert call_theta < 0
        assert put_theta < 0

        # Test vega (should be positive for both calls and puts)
        assert vega > 0

        # Test rho
        assert call_rho > 0  # Call rho should be positive
        assert put_rho < 0   # Put rho should be negative

    def test_edge_cases(self, bs_calc):
        """Test edge cases"""
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.20

        # Test zero time to expiration
        T = 0.0
        call_price_zero_time = bs_calc.black_scholes_call(S, K, T, r, sigma)
        put_price_zero_time = bs_calc.black_scholes_put(S, K, T, r, sigma)

        # Should equal intrinsic value
        assert call_price_zero_time == max(S - K, 0)
        assert put_price_zero_time == max(K - S, 0)

        # Test very high volatility
        high_vol_call = bs_calc.black_scholes_call(S, K, 0.25, r, 2.0)
        normal_vol_call = bs_calc.black_scholes_call(S, K, 0.25, r, 0.20)

        # Higher volatility should result in higher option prices
        assert high_vol_call > normal_vol_call


class TestOptionsDataAnalyzer:
    """Test options data analyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create options data analyzer for testing"""
        config = OptionsAnalysisConfig()
        return OptionsDataAnalyzer(config)

    def test_options_chain_retrieval(self, analyzer):
        """Test options chain data retrieval"""
        result = analyzer.get_options_chain("AAPL")

        assert result['success'] is True
        assert result['symbol'] == "AAPL"
        assert 'retrieved_at' in result
        assert result['current_price'] > 0
        assert isinstance(result['options_data'], dict)
        assert len(result['options_data']) > 0

        # Check first expiration data
        first_exp = next(iter(result['options_data'].values()))
        assert 'calls' in first_exp
        assert 'puts' in first_exp
        assert 'expiration' in first_exp
        assert 'days_to_expiry' in first_exp

        # Check option structure
        if first_exp['calls']:
            call_option = first_exp['calls'][0]
            required_fields = [
                'strike', 'bid', 'ask', 'last', 'volume', 'open_interest',
                'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'
            ]

            for field in required_fields:
                assert field in call_option

    def test_volatility_surface_analysis(self, analyzer):
        """Test volatility surface analysis"""
        result = analyzer.get_options_chain("TSLA")

        assert result['success'] is True
        analysis = result.get('analysis', {})
        vol_analysis = analysis.get('volatility_analysis', {})

        # Check volatility analysis components
        expected_keys = [
            'atm_volatility', 'volatility_skew', 'volatility_smile',
            'term_structure', 'vol_surface_metrics'
        ]

        for key in expected_keys:
            assert key in vol_analysis

        # Check ATM volatility is reasonable
        atm_vol = vol_analysis['atm_volatility']
        assert 0.05 <= atm_vol <= 2.0  # Between 5% and 200%

        # Check volatility surface metrics
        vol_metrics = vol_analysis['vol_surface_metrics']
        assert 'avg_implied_vol' in vol_metrics
        assert 'vol_of_vol' in vol_metrics
        assert 'min_iv' in vol_metrics
        assert 'max_iv' in vol_metrics

    def test_options_flow_analysis(self, analyzer):
        """Test options flow analysis"""
        result = analyzer.get_options_chain("SPY")

        assert result['success'] is True
        analysis = result.get('analysis', {})
        flow_analysis = analysis.get('flow_analysis', {})

        # Check flow analysis components
        expected_keys = [
            'volume_weighted_iv', 'delta_weighted_flow', 'gamma_exposure',
            'vega_exposure', 'flow_direction'
        ]

        for key in expected_keys:
            assert key in flow_analysis

        # Check flow direction values
        valid_directions = ['bullish', 'bearish', 'neutral']
        assert flow_analysis['flow_direction'] in valid_directions

        # Check exposure values are numbers
        assert isinstance(flow_analysis['gamma_exposure'], (int, float))
        assert isinstance(flow_analysis['vega_exposure'], (int, float))

    def test_sentiment_indicators(self, analyzer):
        """Test sentiment indicators calculation"""
        result = analyzer.get_options_chain("QQQ")

        assert result['success'] is True
        analysis = result.get('analysis', {})
        sentiment = analysis.get('sentiment_indicators', {})

        # Check sentiment indicators
        expected_keys = [
            'put_call_ratio_sentiment',
            'volatility_sentiment',
            'flow_sentiment',
            'overall_sentiment',
            'sentiment_score',
            'fear_greed_index']

        for key in expected_keys:
            assert key in sentiment

        # Check sentiment values
        valid_sentiments = [
            'bullish',
            'bearish',
            'neutral',
            'fearful',
            'complacent']
        assert sentiment['overall_sentiment'] in [
            'bullish', 'bearish', 'neutral']
        assert sentiment['put_call_ratio_sentiment'] in valid_sentiments

        # Check score ranges
        assert -1 <= sentiment['sentiment_score'] <= 1
        assert 0 <= sentiment['fear_greed_index'] <= 100

    def test_unusual_activity_detection(self, analyzer):
        """Test unusual activity detection"""
        result = analyzer.get_options_chain("GOOGL")

        assert result['success'] is True
        analysis = result.get('analysis', {})
        unusual_activity = analysis.get('unusual_activity', [])

        # Should be a list
        assert isinstance(unusual_activity, list)

        # If there's unusual activity, check structure
        if unusual_activity:
            activity = unusual_activity[0]
            expected_fields = [
                'symbol',
                'option_type',
                'strike',
                'volume',
                'open_interest',
                'implied_volatility',
                'moneyness',
                'unusual_factor',
                'activity_type']

            for field in expected_fields:
                assert field in activity

            # Check option type
            assert activity['option_type'] in ['call', 'put']

            # Check unusual factor
            assert activity['unusual_factor'] >= 1.0


class TestOptionsAnalysisProvider:
    """Test comprehensive options analysis provider"""

    @pytest.fixture
    def provider(self):
        """Create options analysis provider for testing"""
        return get_options_analysis_provider()

    def test_initialization(self, provider):
        """Test provider initialization"""
        assert provider.config is not None
        assert provider.analyzer is not None
        assert hasattr(provider, '_session')

    def test_comprehensive_analysis(self, provider):
        """Test comprehensive options analysis"""
        result = provider.get_comprehensive_options_analysis("AAPL")

        assert result['success'] is True
        assert result['symbol'] == "AAPL"
        assert 'analyzed_at' in result

        # Check main components
        main_components = [
            'options_chain', 'market_sentiment', 'volatility_analysis',
            'risk_assessment', 'trading_recommendations'
        ]

        for component in main_components:
            assert component in result

    def test_sentiment_analysis_enhancement(self, provider):
        """Test enhanced sentiment analysis"""
        result = provider.get_comprehensive_options_analysis("MSFT")

        sentiment = result['market_sentiment']

        # Check enhanced sentiment features
        assert 'sentiment_strength' in sentiment
        assert 'market_regime' in sentiment
        assert 'implications' in sentiment

        # Check sentiment strength
        strength = sentiment['sentiment_strength']
        assert 'score' in strength
        assert 'level' in strength
        assert strength['level'] in ['weak', 'moderate', 'strong']

        # Check market regime
        regime = sentiment['market_regime']
        assert 'fear_level' in regime
        assert 'greed_level' in regime
        assert 'regime' in regime
        assert regime['regime'] in ['fear', 'greed', 'neutral']

    def test_volatility_analysis_enhancement(self, provider):
        """Test enhanced volatility analysis"""
        result = provider.get_comprehensive_options_analysis("TSLA")

        vol_analysis = result['volatility_analysis']

        # Check enhanced volatility features
        assert 'volatility_regime' in vol_analysis
        assert 'skew_analysis' in vol_analysis
        assert 'trading_opportunities' in vol_analysis

        # Check volatility regime
        vol_regime = vol_analysis['volatility_regime']
        assert 'level' in vol_regime
        assert 'percentile_estimate' in vol_regime
        assert 'regime_type' in vol_regime
        assert vol_regime['level'] in ['low', 'moderate', 'high']

        # Check trading opportunities
        opportunities = vol_analysis['trading_opportunities']
        assert isinstance(opportunities['volatility_trading'], bool)
        assert isinstance(opportunities['mean_reversion'], bool)

    def test_risk_assessment(self, provider):
        """Test risk assessment functionality"""
        result = provider.get_comprehensive_options_analysis("SPY")

        risk = result['risk_assessment']

        # Check risk assessment components
        expected_keys = [
            'overall_risk_level', 'risk_factors', 'risk_score',
            'portfolio_impact', 'hedging_recommendations'
        ]

        for key in expected_keys:
            assert key in risk

        # Check risk level
        assert risk['overall_risk_level'] in ['low', 'moderate', 'high']

        # Check risk score range
        assert 0 <= risk['risk_score'] <= 1

        # Check portfolio impact
        portfolio_impact = risk['portfolio_impact']
        assert 'delta_neutrality' in portfolio_impact
        assert 'gamma_scalping_opportunity' in portfolio_impact
        assert 'volatility_exposure' in portfolio_impact

    def test_trading_recommendations(self, provider):
        """Test trading recommendations generation"""
        result = provider.get_comprehensive_options_analysis("QQQ")

        recommendations = result['trading_recommendations']

        # Check recommendations structure
        expected_keys = [
            'strategy_suggestions', 'volatility_plays', 'risk_management',
            'market_outlook', 'confidence_level'
        ]

        for key in expected_keys:
            assert key in recommendations

        # Check outlook and confidence
        assert recommendations['market_outlook'] in [
            'positive', 'negative', 'neutral']
        assert recommendations['confidence_level'] in [
            'low', 'moderate', 'high']

        # Check recommendation lists
        assert isinstance(recommendations['strategy_suggestions'], list)
        assert isinstance(recommendations['volatility_plays'], list)
        assert isinstance(recommendations['risk_management'], list)

    def test_different_symbols(self, provider):
        """Test provider with different symbols"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']

        for symbol in symbols:
            result = provider.get_comprehensive_options_analysis(symbol)
            assert result['success'] is True
            assert result['symbol'] == symbol
            assert result['options_chain']['current_price'] > 0


class TestIntegration:
    """Integration tests for options analysis provider"""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Create provider
        provider = get_options_analysis_provider()

        # Run comprehensive analysis
        result = provider.get_comprehensive_options_analysis("AAPL")

        # Verify complete workflow
        assert result['success'] is True

        # Check options chain has data
        chain = result['options_chain']
        assert chain['success'] is True
        assert len(chain['options_data']) > 0

        # Check analysis components
        analysis = chain.get('analysis', {})
        assert len(analysis.get('summary', {})) > 0
        assert len(analysis.get('volatility_analysis', {})) > 0
        assert len(analysis.get('sentiment_indicators', {})) > 0

        # Check enhanced analysis
        assert len(result['market_sentiment']) > 0
        assert len(result['volatility_analysis']) > 0
        assert len(result['risk_assessment']) > 0
        assert len(result['trading_recommendations']) > 0

    def test_data_consistency(self):
        """Test data consistency across analysis"""
        provider = get_options_analysis_provider()
        result = provider.get_comprehensive_options_analysis("SPY")

        # Extract data points
        current_price = result['options_chain']['current_price']
        sentiment_score = result['market_sentiment']['sentiment_score']
        vol_regime = result['volatility_analysis']['volatility_regime'][
            'level']
        risk_level = result['risk_assessment']['overall_risk_level']

        # Validate data types and ranges
        assert isinstance(current_price, (int, float))
        assert current_price > 0

        assert isinstance(sentiment_score, (int, float))
        assert -1 <= sentiment_score <= 1

        assert vol_regime in ['low', 'moderate', 'high']
        assert risk_level in ['low', 'moderate', 'high']

    def test_edge_case_handling(self):
        """Test handling of edge cases"""
        provider = get_options_analysis_provider()

        # Test with different symbols
        # Include some that might not exist
        symbols = ['AAPL', 'XYZ123', 'TEST']

        for symbol in symbols:
            result = provider.get_comprehensive_options_analysis(symbol)

            # Should handle gracefully (our mock data will work for any symbol)
            assert result is not None
            assert 'success' in result

# Performance tests


class TestPerformance:
    """Performance tests for options analysis provider"""

    def test_analysis_speed(self):
        """Test analysis completion speed"""
        import time

        provider = get_options_analysis_provider()

        start_time = time.time()
        result = provider.get_comprehensive_options_analysis("AAPL")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert execution_time < 3.0  # 3 seconds max for mock data
        assert result['success'] is True


if __name__ == "__main__":
    # Run basic tests
    print("=== Running Options Analysis Tests ===")

    # Test configuration
    print("\n1. Testing Configuration...")
    config = OptionsAnalysisConfig()
    assert config.risk_free_rate == 0.05
    print("✅ Configuration test passed")

    # Test Black-Scholes calculator
    print("\n2. Testing Black-Scholes Calculator...")
    bs_calc = BlackScholesCalculator()

    # Test call pricing
    call_price = bs_calc.black_scholes_call(100, 100, 0.25, 0.05, 0.20)
    assert call_price > 0
    print(f"✅ Black-Scholes call price: ${call_price:.2f}")

    # Test Greeks
    delta = bs_calc.calculate_delta(100, 100, 0.25, 0.05, 0.20, 'call')
    gamma = bs_calc.calculate_gamma(100, 100, 0.25, 0.05, 0.20)
    assert 0 < delta < 1
    assert gamma > 0
    print(f"✅ Greeks calculation: Delta={delta:.3f}, Gamma={gamma:.3f}")

    # Test options data analyzer
    print("\n3. Testing Options Data Analyzer...")
    config = OptionsAnalysisConfig()
    analyzer = OptionsDataAnalyzer(config)
    chain_result = analyzer.get_options_chain("AAPL")
    assert chain_result['success'] is True
    assert len(chain_result['options_data']) > 0
    print(
        f"✅ Options chain analysis - Current price: ${chain_result['current_price']:.2f}")

    # Test provider
    print("\n4. Testing Options Analysis Provider...")
    provider = get_options_analysis_provider()
    analysis_result = provider.get_comprehensive_options_analysis("AAPL")
    assert analysis_result['success'] is True
    print("✅ Comprehensive analysis completed")

    # Test sentiment and volatility
    print("\n5. Testing Sentiment & Volatility Analysis...")
    sentiment = analysis_result['market_sentiment']
    volatility = analysis_result['volatility_analysis']

    assert 'overall_sentiment' in sentiment
    assert 'volatility_regime' in volatility

    print(f"✅ Sentiment: {sentiment.get('overall_sentiment', 'unknown')}")
    print(
        f"✅ Vol Regime: {
            volatility.get(
                'volatility_regime',
                {}).get(
                'level',
                'unknown')}")

    print("\n=== All Options Analysis Tests Passed ===")
    print("🚀 Options analysis provider is ready for production use!")
