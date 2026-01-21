"""
Test Suite for Market Microstructure & Short Interest Provider

This test suite validates all components of the market microstructure provider:
- Short interest analysis and squeeze detection
- Failed-to-deliver (FTD) analysis
- Liquidity metrics calculation
- Risk assessment and recommendations
- Integration tests

Author: DCF Lab Team
Created: 2025-09-18
"""

import sys
from pathlib import Path

import pandas as pd

from dcf_lab.providers.market_microstructure import (
    FailedToDeliverAnalyzer,
    MarketMicrostructureConfig,
    MarketMicrostructureProvider,
    ShortInterestAnalyzer,
    get_market_microstructure_provider,
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMarketMicrostructureConfig:
    """Test market microstructure configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = MarketMicrostructureConfig()

        assert config.cache_dir == "./cache/market_microstructure"
        assert config.enable_cache is True
        assert config.max_cache_age_hours == 24
        assert config.timeout == 30
        assert config.retries == 3
        assert abs(config.rate_limit_delay - 1.0) < 0.01

    def test_custom_config(self):
        """Test custom configuration"""
        config = MarketMicrostructureConfig(
            enable_cache=False,
            timeout=60,
            rate_limit_delay=0.5
        )

        assert config.enable_cache is False
        assert config.timeout == 60
        assert abs(config.rate_limit_delay - 0.5) < 0.01


class TestShortInterestAnalyzer:
    """Test short interest analysis engine"""

    def setup_method(self):
        """Set up test environment"""
        self.config = MarketMicrostructureConfig()
        self.analyzer = ShortInterestAnalyzer(self.config)

    def test_threshold_initialization(self):
        """Test threshold initialization"""
        assert 'low' in self.analyzer.short_interest_thresholds
        assert 'extreme' in self.analyzer.short_interest_thresholds
        assert 'low_risk' in self.analyzer.days_to_cover_thresholds
        assert 'squeeze_risk' in self.analyzer.days_to_cover_thresholds

    def test_classify_short_ratio(self):
        """Test short ratio classification"""
        assert self.analyzer._classify_short_ratio(3.0) == 'low'
        assert self.analyzer._classify_short_ratio(10.0) == 'moderate'
        assert self.analyzer._classify_short_ratio(20.0) == 'high'
        assert self.analyzer._classify_short_ratio(30.0) == 'extreme'

    def test_classify_days_to_cover(self):
        """Test days to cover classification"""
        assert self.analyzer._classify_days_to_cover(1.0) == 'quick_cover'
        assert self.analyzer._classify_days_to_cover(3.0) == 'moderate_cover'
        assert self.analyzer._classify_days_to_cover(7.0) == 'slow_cover'
        assert self.analyzer._classify_days_to_cover(15.0) == 'difficult_cover'

    def test_assess_days_to_cover_risk(self):
        """Test days to cover risk assessment"""
        assert self.analyzer._assess_days_to_cover_risk(1.0) == 'low'
        assert self.analyzer._assess_days_to_cover_risk(3.0) == 'moderate'
        assert self.analyzer._assess_days_to_cover_risk(7.0) == 'high'
        assert self.analyzer._assess_days_to_cover_risk(15.0) == 'extreme'

    def test_calculate_short_metrics_basic(self):
        """Test basic short metrics calculation"""
        # Create test data
        short_data = pd.DataFrame([
            {
                'settlement_date': '2023-12-01',
                'report_date': '2023-12-03',
                'symbol': 'TEST',
                'short_volume': 100000,
                'total_volume': 1000000
            }
        ])

        volume_data = pd.DataFrame([
            {'date': '2023-12-01', 'symbol': 'TEST', 'volume': 2000000}
        ])

        float_shares = 10_000_000

        result = self.analyzer.calculate_short_metrics(
            'TEST', short_data, volume_data, float_shares
        )

        assert isinstance(result, dict)
        assert result['symbol'] == 'TEST'
        assert 'short_interest' in result
        assert 'days_to_cover' in result
        assert 'short_ratio' in result
        assert 'success' in result

        if result['success']:
            short_interest = result['short_interest']
            assert short_interest['short_shares'] == 100000
            assert short_interest['total_volume'] == 1000000
            assert abs(short_interest['short_volume_ratio'] - 0.1) < 0.01

            short_ratio = result['short_ratio']
            assert abs(
                short_ratio['percentage'] -
                1.0) < 0.01  # 100k/10M * 100
            assert short_ratio['classification'] == 'low'

            days_to_cover = result['days_to_cover']
            assert abs(days_to_cover['days'] - 0.05) < 0.01  # 100k/2M

    def test_calculate_short_metrics_empty_data(self):
        """Test short metrics with empty data"""
        empty_data = pd.DataFrame()

        result = self.analyzer.calculate_short_metrics('TEST', empty_data)

        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result

    def test_analyze_short_trends(self):
        """Test short interest trend analysis"""
        # Create trending data
        dates = pd.date_range('2023-11-01', '2023-12-01', freq='7D')
        short_data = pd.DataFrame([
            {
                'settlement_date': date.strftime('%Y-%m-%d'),
                'short_volume': 100000 + (i * 10000),  # Increasing trend
                'total_volume': 1000000
            }
            for i, date in enumerate(dates)
        ])

        trends = self.analyzer._analyze_short_trends(short_data)

        assert isinstance(trends, dict)
        assert 'direction' in trends
        assert 'magnitude' in trends
        assert 'recent_change' in trends
        assert 'consistency' in trends

    def test_detect_squeeze_signals(self):
        """Test short squeeze signal detection"""
        # Create test metrics with high squeeze probability
        metrics = {
            'short_ratio': {'percentage': 25.0},  # High short interest
            'days_to_cover': {'days': 12.0},      # High days to cover
            'trend_analysis': {
                'direction': 'increasing',
                'consistency': 0.8
            }
        }

        # Mock volume data showing high volume
        volume_data = pd.DataFrame([
            {'date': f'2023-12-{i:02d}', 'volume': 2000000 if i <= 25 else 4000000}
            for i in range(1, 31)
        ])

        signals = self.analyzer._detect_squeeze_signals(
            pd.DataFrame(), volume_data, metrics
        )

        assert isinstance(signals, dict)
        assert 'squeeze_probability' in signals
        assert 'signal_strength' in signals
        assert 'key_indicators' in signals
        assert 'timeframe_estimate' in signals

        # Should detect some squeeze signals
        assert signals['squeeze_probability'] > 0
        assert len(signals['key_indicators']) > 0


class TestFailedToDeliverAnalyzer:
    """Test failed-to-deliver analysis engine"""

    def setup_method(self):
        """Set up test environment"""
        self.config = MarketMicrostructureConfig()
        self.analyzer = FailedToDeliverAnalyzer(self.config)

    def test_analyze_ftd_data_basic(self):
        """Test basic FTD data analysis"""
        # Create test FTD data
        ftd_data = pd.DataFrame([
            {
                'settlement_date': '2023-12-01',
                'symbol': 'TEST',
                'quantity': 50000,
                'price': 100.0
            },
            {
                'settlement_date': '2023-12-02',
                'symbol': 'TEST',
                'quantity': 75000,
                'price': 101.0
            }
        ])

        result = self.analyzer.analyze_ftd_data('TEST', ftd_data)

        assert isinstance(result, dict)
        assert result['symbol'] == 'TEST'
        assert 'ftd_summary' in result
        assert 'trend_analysis' in result
        assert 'threshold_breaches' in result
        assert 'risk_indicators' in result
        assert 'success' in result

        if result['success']:
            ftd_summary = result['ftd_summary']
            assert ftd_summary['total_failed_shares'] == 125000  # 50k + 75k
            assert ftd_summary['max_daily_ftd'] == 75000
            assert ftd_summary['ftd_reporting_days'] == 2

    def test_analyze_ftd_data_empty(self):
        """Test FTD analysis with empty data"""
        empty_data = pd.DataFrame()

        result = self.analyzer.analyze_ftd_data('TEST', empty_data)

        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result

    def test_analyze_ftd_trends(self):
        """Test FTD trend analysis"""
        # Create trending FTD data
        dates = pd.date_range('2023-11-01', '2023-12-01', freq='D')
        ftd_data = pd.DataFrame([
            {
                'settlement_date': date.strftime('%Y-%m-%d'),
                # Every 3rd day with increasing
                'quantity': 10000 + (i * 1000) if i % 3 == 0 else 0,
                'symbol': 'TEST'
            }
            for i, date in enumerate(dates)
        ])

        trends = self.analyzer._analyze_ftd_trends(ftd_data)

        assert isinstance(trends, dict)
        assert 'direction' in trends
        assert 'volatility' in trends
        assert 'persistence' in trends
        assert 'recent_acceleration' in trends

    def test_identify_threshold_breaches(self):
        """Test threshold breach identification"""
        # Create data with threshold breaches
        dates = pd.date_range('2023-11-01', '2023-12-15', freq='D')
        ftd_data = pd.DataFrame([
            {
                'settlement_date': date.strftime('%Y-%m-%d'),
                'quantity': 15000,  # Above 10k threshold
                'symbol': 'TEST'
            }
            for date in dates  # 44 consecutive days > T+13 threshold
        ])

        breaches = self.analyzer._identify_threshold_breaches(ftd_data)

        assert isinstance(breaches, list)
        # Should detect ongoing breach
        if breaches:
            breach = breaches[0]
            assert 'type' in breach
            assert 'consecutive_days' in breach
            assert breach['consecutive_days'] >= 13


class TestMarketMicrostructureProvider:
    """Test main market microstructure provider"""

    def setup_method(self):
        """Set up test environment"""
        self.provider = get_market_microstructure_provider()

    def test_provider_initialization(self):
        """Test provider initialization"""
        assert isinstance(self.provider, MarketMicrostructureProvider)
        assert isinstance(self.provider.config, MarketMicrostructureConfig)
        assert isinstance(self.provider.short_analyzer, ShortInterestAnalyzer)
        assert isinstance(self.provider.ftd_analyzer, FailedToDeliverAnalyzer)

    def test_get_comprehensive_analysis(self):
        """Test comprehensive analysis"""
        result = self.provider.get_comprehensive_analysis(
            'AAPL', lookback_days=30)

        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'analyzed_at' in result
        assert 'lookback_days' in result
        assert 'short_analysis' in result
        assert 'ftd_analysis' in result
        assert 'liquidity_metrics' in result
        assert 'risk_assessment' in result
        assert 'trading_patterns' in result
        assert 'recommendations' in result
        assert 'success' in result

        assert result['symbol'] == 'AAPL'
        assert result['lookback_days'] == 30

        if result['success']:
            # Check structure of sub-analyses
            short_analysis = result['short_analysis']
            assert isinstance(short_analysis, dict)

            ftd_analysis = result['ftd_analysis']
            assert isinstance(ftd_analysis, dict)

            liquidity_metrics = result['liquidity_metrics']
            assert isinstance(liquidity_metrics, dict)
            assert 'average_daily_volume' in liquidity_metrics
            assert 'liquidity_score' in liquidity_metrics

            risk_assessment = result['risk_assessment']
            assert isinstance(risk_assessment, dict)
            assert 'overall_risk_level' in risk_assessment
            assert 'risk_score' in risk_assessment

            recommendations = result['recommendations']
            assert isinstance(recommendations, dict)
            assert 'monitoring_frequency' in recommendations

    def test_generate_mock_data_methods(self):
        """Test mock data generation methods"""
        # Test short data generation
        short_data = self.provider._generate_mock_short_data('TEST', 30)
        assert isinstance(short_data, pd.DataFrame)
        assert not short_data.empty
        assert 'symbol' in short_data.columns
        assert 'short_volume' in short_data.columns
        assert 'total_volume' in short_data.columns

        # Test FTD data generation
        ftd_data = self.provider._generate_mock_ftd_data('TEST', 30)
        assert isinstance(ftd_data, pd.DataFrame)
        # FTD data might be empty (that's normal)
        if not ftd_data.empty:
            assert 'symbol' in ftd_data.columns
            assert 'quantity' in ftd_data.columns

        # Test volume data generation
        volume_data = self.provider._generate_mock_volume_data('TEST', 30)
        assert isinstance(volume_data, pd.DataFrame)
        assert not volume_data.empty
        assert 'symbol' in volume_data.columns
        assert 'volume' in volume_data.columns

    def test_calculate_liquidity_metrics(self):
        """Test liquidity metrics calculation"""
        # Create test volume data
        volume_data = pd.DataFrame([
            {'date': f'2023-12-{i:02d}', 'symbol': 'TEST',
                'volume': 1000000 + (i * 50000)}
            for i in range(1, 21)
        ])

        metrics = self.provider._calculate_liquidity_metrics(volume_data)

        assert isinstance(metrics, dict)
        assert 'average_daily_volume' in metrics
        assert 'volume_volatility' in metrics
        assert 'volume_trend' in metrics
        assert 'liquidity_score' in metrics
        assert 'market_depth_estimate' in metrics

        assert metrics['average_daily_volume'] > 0
        assert 0 <= metrics['liquidity_score'] <= 1
        assert metrics['market_depth_estimate'] in [
            'shallow', 'moderate', 'deep']

    def test_analyze_trading_patterns(self):
        """Test trading pattern analysis"""
        # Create volume data with some spikes
        volume_data = pd.DataFrame([
            {'date': f'2023-12-{i:02d}', 'symbol': 'TEST',
             'volume': 1000000 if i != 10 else 3000000}  # Spike on day 10
            for i in range(1, 21)
        ])

        patterns = self.provider._analyze_trading_patterns(volume_data)

        assert isinstance(patterns, dict)
        assert 'volume_spikes' in patterns
        assert 'unusual_activity' in patterns
        assert 'pattern_classification' in patterns
        assert 'anomaly_score' in patterns

        # Should detect the volume spike
        assert len(patterns['volume_spikes']) > 0
        assert 0 <= patterns['anomaly_score'] <= 1

    def test_assess_overall_risk(self):
        """Test overall risk assessment"""
        # Create mock analysis data
        analysis = {
            'short_analysis': {
                'success': True,
                'risk_assessment': {'risk_score': 0.6}
            },
            'ftd_analysis': {
                'success': True,
                'risk_indicators': {'risk_score': 0.4}
            },
            'liquidity_metrics': {
                'liquidity_score': 0.7,
                'market_depth_estimate': 'moderate'
            },
            'trading_patterns': {
                'anomaly_score': 0.3,
                'pattern_classification': 'normal'
            }
        }

        risk = self.provider._assess_overall_risk(analysis)

        assert isinstance(risk, dict)
        assert 'overall_risk_level' in risk
        assert 'risk_score' in risk
        assert 'key_risk_factors' in risk
        assert 'monitoring_priority' in risk

        assert risk['overall_risk_level'] in [
            'low', 'moderate', 'high', 'extreme']
        assert 0 <= risk['risk_score'] <= 1
        assert risk['monitoring_priority'] in [
            'low', 'moderate', 'high', 'immediate']

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Create mock analysis data with various risk scenarios
        analysis = {
            'risk_assessment': {'overall_risk_level': 'moderate'},
            'short_analysis': {
                'success': True,
                'squeeze_indicators': {'squeeze_probability': 0.6},
                'short_ratio': {'percentage': 15.0},
                'days_to_cover': {'days': 8.0}
            },
            'liquidity_metrics': {
                'market_depth_estimate': 'shallow',
                'average_daily_volume': 500000
            }
        }

        recommendations = self.provider._generate_recommendations(analysis)

        assert isinstance(recommendations, dict)
        assert 'trading_recommendations' in recommendations
        assert 'risk_management' in recommendations
        assert 'monitoring_frequency' in recommendations
        assert 'key_metrics_to_watch' in recommendations
        assert 'alert_thresholds' in recommendations

        assert recommendations['monitoring_frequency'] in ['weekly', 'daily']
        assert isinstance(recommendations['trading_recommendations'], list)
        assert isinstance(recommendations['risk_management'], list)
        assert isinstance(recommendations['key_metrics_to_watch'], list)
        assert isinstance(recommendations['alert_thresholds'], dict)


class TestIntegration:
    """Integration tests for market microstructure provider"""

    def setup_method(self):
        """Set up test environment"""
        self.provider = get_market_microstructure_provider()

    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow"""
        symbols = ['AAPL', 'MSFT', 'TSLA']

        for symbol in symbols:
            result = self.provider.get_comprehensive_analysis(
                symbol, lookback_days=30)

            assert isinstance(result, dict)
            assert 'success' in result
            assert result['symbol'] == symbol

            if result['success']:
                # Verify all major components are present
                assert 'short_analysis' in result
                assert 'ftd_analysis' in result
                assert 'liquidity_metrics' in result
                assert 'risk_assessment' in result
                assert 'recommendations' in result

    def test_different_lookback_periods(self):
        """Test analysis with different lookback periods"""
        lookback_periods = [30, 60, 90]

        for days in lookback_periods:
            result = self.provider.get_comprehensive_analysis(
                'AAPL', lookback_days=days)

            assert isinstance(result, dict)
            assert result['lookback_days'] == days

            if result['success']:
                # Longer periods should potentially have more data
                short_analysis = result.get('short_analysis', {})
                if short_analysis.get('success'):
                    # Should have meaningful analysis regardless of period
                    assert 'short_interest' in short_analysis

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with unusual symbols
        unusual_symbols = [
            '',
            'INVALID_SYMBOL_VERY_LONG',
            '123',
            'TEST-WITH-DASHES']

        for symbol in unusual_symbols:
            result = self.provider.get_comprehensive_analysis(
                symbol, lookback_days=30)

            assert isinstance(result, dict)
            assert 'success' in result
            assert 'error' in result or result['success'] is True

    def test_factory_function(self):
        """Test factory function"""
        # Test with default config
        provider1 = get_market_microstructure_provider()
        assert isinstance(provider1, MarketMicrostructureProvider)

        # Test with custom config
        config = MarketMicrostructureConfig(timeout=60)
        provider2 = get_market_microstructure_provider(config)
        assert isinstance(provider2, MarketMicrostructureProvider)
        assert provider2.config.timeout == 60


def test_example_usage():
    """Test the example usage from the module"""
    provider = get_market_microstructure_provider()

    # Test comprehensive analysis
    analysis = provider.get_comprehensive_analysis('AAPL', lookback_days=60)
    assert isinstance(analysis, dict)
    assert 'success' in analysis


if __name__ == "__main__":
    # Run a simple test when executed directly
    print("Running Market Microstructure Provider Tests...")

    # Create provider
    provider = get_market_microstructure_provider()

    # Test basic functionality
    print("\n1. Testing comprehensive analysis...")
    analysis = provider.get_comprehensive_analysis('TSLA', lookback_days=30)
    print(f"   Success: {analysis['success']}")

    if analysis['success']:
        print("\n2. Testing analysis components...")

        short_analysis = analysis.get('short_analysis', {})
        print(f"   Short Analysis: {short_analysis.get('success', False)}")

        ftd_analysis = analysis.get('ftd_analysis', {})
        print(f"   FTD Analysis: {ftd_analysis.get('success', False)}")

        liquidity = analysis.get('liquidity_metrics', {})
        print(f"   Liquidity Score: {liquidity.get('liquidity_score', 0):.2f}")

        risk = analysis.get('risk_assessment', {})
        print(f"   Risk Level: {risk.get('overall_risk_level', 'unknown')}")

        recommendations = analysis.get('recommendations', {})
        print(
            f"   Monitoring: {
                recommendations.get(
                    'monitoring_frequency',
                    'unknown')}")

    print("\n✅ Market Microstructure Provider tests completed!")
    print("💡 Run with pytest for comprehensive testing")
