"""
Test Alternative Signals Provider

Tests the alternative signals provider including Google Trends analysis,
social sentiment tracking, economic nowcasting, and integrated signals analysis.

Author: DCF Lab Team
Created: 2025-01-20
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from providers.alternative_signals import (
    AlternativeSignalsConfig,
    AlternativeSignalsProvider,
    EconomicNowcastingAnalyzer,
    GoogleTrendsAnalyzer,
    SocialSentimentAnalyzer,
    get_alternative_signals_provider,
)

# Import the provider and its components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAlternativeSignalsConfig(unittest.TestCase):
    """Test AlternativeSignalsConfig"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = AlternativeSignalsConfig()

        self.assertEqual(config.cache_dir, "./cache/alternative_signals")
        self.assertTrue(config.enable_cache)
        self.assertEqual(config.max_cache_age_hours, 4)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.retries, 3)
        self.assertEqual(config.rate_limit_delay, 1.0)
        self.assertEqual(config.sentiment_lookback_days, 30)
        self.assertEqual(config.trends_lookback_weeks, 12)
        self.assertEqual(config.correlation_threshold, 0.3)
        self.assertEqual(config.signal_confidence_threshold, 0.6)

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = AlternativeSignalsConfig(
            cache_dir="./custom_cache",
            enable_cache=False,
            max_cache_age_hours=2,
            timeout=60,
            sentiment_lookback_days=7,
            correlation_threshold=0.5
        )

        self.assertEqual(config.cache_dir, "./custom_cache")
        self.assertFalse(config.enable_cache)
        self.assertEqual(config.max_cache_age_hours, 2)
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.sentiment_lookback_days, 7)
        self.assertEqual(config.correlation_threshold, 0.5)


class TestGoogleTrendsAnalyzer(unittest.TestCase):
    """Test GoogleTrendsAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = AlternativeSignalsConfig()
        self.analyzer = GoogleTrendsAnalyzer(self.config)

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer.config, AlternativeSignalsConfig)
        self.assertIsInstance(self.analyzer.economic_keywords, dict)
        self.assertIsInstance(self.analyzer.stress_indicators, list)

        # Check some expected keywords
        self.assertIn('recession', self.analyzer.economic_keywords)
        self.assertIn('inflation', self.analyzer.economic_keywords)
        self.assertIn('unemployment', self.analyzer.economic_keywords)

        # Check stress indicators
        self.assertIn('bankruptcy', self.analyzer.stress_indicators)
        self.assertIn('debt relief', self.analyzer.stress_indicators)

    def test_trends_analysis_default_keywords(self):
        """Test trends analysis with default keywords"""
        result = self.analyzer.get_trends_analysis()

        self.assertIsInstance(result, dict)
        self.assertIn('analyzed_at', result)
        self.assertIn('timeframe', result)
        self.assertIn('keyword_trends', result)
        self.assertIn('economic_signals', result)
        self.assertIn('stress_indicators', result)
        self.assertIn('trend_correlations', result)
        self.assertTrue(result['success'])

        # Check keyword trends data
        keyword_trends = result['keyword_trends']
        self.assertIn('keywords', keyword_trends)
        self.assertIn('summary', keyword_trends)

        # Check economic signals
        economic_signals = result['economic_signals']
        self.assertIn('recession_sentiment', economic_signals)
        self.assertIn('inflation_concern', economic_signals)
        self.assertIn('employment_stress', economic_signals)
        self.assertIn('economic_outlook', economic_signals)

        # Validate signal ranges
        for signal in [
            'recession_sentiment',
            'inflation_concern',
                'employment_stress']:
            value = economic_signals[signal]
            self.assertGreaterEqual(value, -1)
            self.assertLessEqual(value, 1)

    def test_trends_analysis_custom_keywords(self):
        """Test trends analysis with custom keywords"""
        custom_keywords = ['bitcoin', 'housing market', 'federal reserve']
        result = self.analyzer.get_trends_analysis(keywords=custom_keywords)

        self.assertTrue(result['success'])

        # Check that all custom keywords are analyzed
        keywords_data = result['keyword_trends']['keywords']
        for keyword in custom_keywords:
            self.assertIn(keyword, keywords_data)

            keyword_data = keywords_data[keyword]
            self.assertIn('search_volume', keyword_data)
            self.assertIn('average_interest', keyword_data)
            self.assertIn('trend_direction', keyword_data)
            self.assertIn('volatility', keyword_data)

    def test_trends_analysis_different_timeframes(self):
        """Test trends analysis with different timeframes"""
        timeframes = ['1d', '7d', '1m', '3m', '6m', '1y']

        for timeframe in timeframes:
            result = self.analyzer.get_trends_analysis(timeframe=timeframe)

            self.assertTrue(result['success'])
            self.assertEqual(result['timeframe'], timeframe)

            # Check date range is appropriate for timeframe
            date_range = result['keyword_trends']['date_range']
            start_date = datetime.fromisoformat(date_range['start'])
            end_date = datetime.fromisoformat(date_range['end'])

            # Verify timeframe duration is reasonable
            duration = (end_date - start_date).days
            if timeframe == '1d':
                self.assertLessEqual(duration, 2)
            elif timeframe == '7d':
                self.assertLessEqual(duration, 8)
            elif timeframe == '1m':
                self.assertLessEqual(duration, 32)

    def test_stress_indicators_analysis(self):
        """Test stress indicators analysis"""
        result = self.analyzer.get_trends_analysis()
        stress_indicators = result['stress_indicators']

        self.assertIn('financial_stress_level', stress_indicators)
        self.assertIn('stress_trending', stress_indicators)
        self.assertIn('stress_categories', stress_indicators)
        self.assertIn('early_warning_signals', stress_indicators)

        # Validate stress level range
        stress_level = stress_indicators['financial_stress_level']
        self.assertGreaterEqual(stress_level, 0)
        self.assertLessEqual(stress_level, 1)

        # Check stress trending values
        self.assertIn(stress_indicators['stress_trending'],
                      ['increasing', 'decreasing', 'stable'])

        # Check stress categories
        stress_categories = stress_indicators['stress_categories']
        expected_categories = [
            'bankruptcy_searches', 'debt_relief_searches',
            'financial_help_searches', 'credit_problems_searches'
        ]
        for category in expected_categories:
            self.assertIn(category, stress_categories)

    def test_trend_correlations(self):
        """Test trend correlations calculation"""
        keywords = ['recession', 'unemployment', 'stock market']
        result = self.analyzer.get_trends_analysis(keywords=keywords)

        correlations = result['trend_correlations']
        self.assertIn('correlation_matrix', correlations)
        self.assertIn('strong_correlations', correlations)
        self.assertIn('correlation_insights', correlations)

        # Check correlation matrix structure
        correlation_matrix = correlations['correlation_matrix']
        for keyword in keywords:
            self.assertIn(keyword, correlation_matrix)
            for other_keyword in keywords:
                self.assertIn(other_keyword, correlation_matrix[keyword])

                # Diagonal should be 1.0
                if keyword == other_keyword:
                    self.assertEqual(
                        correlation_matrix[keyword][other_keyword], 1.0)
                else:
                    # Correlation should be between -1 and 1
                    corr_value = correlation_matrix[keyword][other_keyword]
                    self.assertGreaterEqual(corr_value, -1)
                    self.assertLessEqual(corr_value, 1)


class TestSocialSentimentAnalyzer(unittest.TestCase):
    """Test SocialSentimentAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = AlternativeSignalsConfig()
        self.analyzer = SocialSentimentAnalyzer(self.config)

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer.config, AlternativeSignalsConfig)
        self.assertIsInstance(self.analyzer.financial_keywords, list)
        self.assertIsInstance(self.analyzer.positive_indicators, list)
        self.assertIsInstance(self.analyzer.negative_indicators, list)

        # Check some expected keywords
        self.assertIn('market', self.analyzer.financial_keywords)
        self.assertIn('stocks', self.analyzer.financial_keywords)
        self.assertIn('economy', self.analyzer.financial_keywords)

        # Check sentiment indicators
        self.assertIn('bullish', self.analyzer.positive_indicators)
        self.assertIn('bearish', self.analyzer.negative_indicators)

    def test_social_sentiment_analysis_default(self):
        """Test social sentiment analysis with defaults"""
        result = self.analyzer.get_social_sentiment_analysis()

        self.assertIsInstance(result, dict)
        self.assertIn('analyzed_at', result)
        self.assertIn('platforms', result)
        self.assertIn('keywords', result)
        self.assertIn('sentiment_scores', result)
        self.assertIn('platform_analysis', result)
        self.assertIn('trending_topics', result)
        self.assertIn('sentiment_trends', result)
        self.assertTrue(result['success'])

        # Check default platforms and keywords
        self.assertEqual(result['platforms'], ['reddit', 'twitter'])
        # Top 5 financial keywords
        self.assertEqual(len(result['keywords']), 5)

        # Check sentiment scores structure
        sentiment_scores = result['sentiment_scores']
        for keyword in result['keywords']:
            self.assertIn(keyword, sentiment_scores)

            keyword_data = sentiment_scores[keyword]
            self.assertIn('sentiment_score', keyword_data)
            self.assertIn('confidence', keyword_data)
            self.assertIn('volume', keyword_data)
            self.assertIn('mentions', keyword_data)

            # Validate ranges
            self.assertGreaterEqual(keyword_data['sentiment_score'], -1)
            self.assertLessEqual(keyword_data['sentiment_score'], 1)
            self.assertGreaterEqual(keyword_data['confidence'], 0)
            self.assertLessEqual(keyword_data['confidence'], 1)

    def test_social_sentiment_analysis_custom(self):
        """Test social sentiment analysis with custom parameters"""
        platforms = ['reddit']
        keywords = ['bitcoin', 'housing']

        result = self.analyzer.get_social_sentiment_analysis(
            platforms=platforms, keywords=keywords
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['platforms'], platforms)
        self.assertEqual(result['keywords'], keywords)

        # Check that custom keywords are analyzed
        sentiment_scores = result['sentiment_scores']
        for keyword in keywords:
            self.assertIn(keyword, sentiment_scores)

    def test_platform_analysis(self):
        """Test platform-specific analysis"""
        result = self.analyzer.get_social_sentiment_analysis()
        platform_analysis = result['platform_analysis']

        for platform in result['platforms']:
            self.assertIn(platform, platform_analysis)

            platform_data = platform_analysis[platform]
            for keyword in result['keywords']:
                self.assertIn(keyword, platform_data)

                keyword_data = platform_data[keyword]
                self.assertIn('sentiment_score', keyword_data)
                self.assertIn('posts_analyzed', keyword_data)
                self.assertIn('engagement_score', keyword_data)
                self.assertIn('trending_rank', keyword_data)

    def test_trending_topics(self):
        """Test trending topics analysis"""
        result = self.analyzer.get_social_sentiment_analysis()
        trending_topics = result['trending_topics']

        for platform in result['platforms']:
            self.assertIn(platform, trending_topics)

            topics = trending_topics[platform]
            self.assertIsInstance(topics, list)
            self.assertEqual(len(topics), 5)  # Top 5 trending

            for topic_data in topics:
                self.assertIn('topic', topic_data)
                self.assertIn('sentiment', topic_data)
                self.assertIn('volume', topic_data)
                self.assertIn('rank', topic_data)
                self.assertIn('change_24h', topic_data)

    def test_sentiment_trends(self):
        """Test sentiment trends over time"""
        result = self.analyzer.get_social_sentiment_analysis()
        sentiment_trends = result['sentiment_trends']

        # Should have trends for top 3 keywords
        self.assertLessEqual(len(sentiment_trends), 3)

        for keyword, trend_data in sentiment_trends.items():
            self.assertIsInstance(trend_data, list)
            self.assertEqual(len(trend_data), 7)  # 7 days

            for daily_data in trend_data:
                self.assertIn('date', daily_data)
                self.assertIn('sentiment', daily_data)
                self.assertIn('volume', daily_data)

                # Validate date format
                datetime.fromisoformat(daily_data['date'])

                # Validate sentiment range
                self.assertGreaterEqual(daily_data['sentiment'], -1)
                self.assertLessEqual(daily_data['sentiment'], 1)


class TestEconomicNowcastingAnalyzer(unittest.TestCase):
    """Test EconomicNowcastingAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = AlternativeSignalsConfig()
        self.analyzer = EconomicNowcastingAnalyzer(self.config)

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer.config, AlternativeSignalsConfig)
        self.assertIsInstance(self.analyzer.nowcast_indicators, dict)
        self.assertIsInstance(self.analyzer.alt_data_sources, dict)

        # Check expected indicators
        expected_indicators = [
            'gdp_growth', 'employment', 'inflation', 'consumer_spending',
            'business_investment', 'trade_activity', 'financial_conditions'
        ]
        for indicator in expected_indicators:
            self.assertIn(indicator, self.analyzer.nowcast_indicators)

        # Check alternative data sources
        expected_sources = [
            'satellite_data', 'shipping_data', 'energy_consumption',
            'mobile_location', 'credit_card_spending', 'job_postings'
        ]
        for source in expected_sources:
            self.assertIn(source, self.analyzer.alt_data_sources)

    def test_nowcasting_analysis_default(self):
        """Test nowcasting analysis with default indicators"""
        result = self.analyzer.get_nowcasting_analysis()

        self.assertIsInstance(result, dict)
        self.assertIn('analyzed_at', result)
        self.assertIn('nowcast_period', result)
        self.assertIn('indicators', result)
        self.assertIn('nowcast_values', result)
        self.assertIn('confidence_scores', result)
        self.assertIn('data_sources', result)
        self.assertIn('economic_outlook', result)
        self.assertTrue(result['success'])

        # Check default indicators
        # All default indicators
        self.assertEqual(len(result['indicators']), 7)

        # Check nowcast values structure
        nowcast_values = result['nowcast_values']
        confidence_scores = result['confidence_scores']
        data_sources = result['data_sources']

        for indicator in result['indicators']:
            # Check nowcast estimate
            self.assertIn(indicator, nowcast_values)
            estimate = nowcast_values[indicator]
            self.assertIn('value', estimate)
            self.assertIn('unit', estimate)
            self.assertIn('last_updated', estimate)

            # Check confidence score
            self.assertIn(indicator, confidence_scores)
            confidence = confidence_scores[indicator]
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)

            # Check data sources
            self.assertIn(indicator, data_sources)
            sources = data_sources[indicator]
            self.assertIsInstance(sources, list)
            self.assertGreater(len(sources), 0)

    def test_nowcasting_analysis_custom_indicators(self):
        """Test nowcasting analysis with custom indicators"""
        custom_indicators = ['gdp_growth', 'employment', 'inflation']
        result = self.analyzer.get_nowcasting_analysis(
            indicators=custom_indicators)

        self.assertTrue(result['success'])
        self.assertEqual(result['indicators'], custom_indicators)

        # Check that only custom indicators are analyzed
        nowcast_values = result['nowcast_values']
        self.assertEqual(len(nowcast_values), len(custom_indicators))

        for indicator in custom_indicators:
            self.assertIn(indicator, nowcast_values)

    def test_economic_outlook(self):
        """Test economic outlook generation"""
        result = self.analyzer.get_nowcasting_analysis()
        economic_outlook = result['economic_outlook']

        # Check required outlook fields
        required_fields = [
            'overall_assessment', 'growth_outlook', 'inflation_outlook',
            'risk_factors', 'positive_factors', 'outlook_confidence'
        ]
        for field in required_fields:
            self.assertIn(field, economic_outlook)

        # Check valid values
        self.assertIn(economic_outlook['overall_assessment'],
                      ['positive', 'negative', 'neutral'])
        self.assertIn(economic_outlook['growth_outlook'],
                      ['strong', 'moderate', 'weak'])
        self.assertIn(economic_outlook['inflation_outlook'],
                      ['elevated', 'moderate', 'low'])

        # Check confidence range
        self.assertGreaterEqual(economic_outlook['outlook_confidence'], 0)
        self.assertLessEqual(economic_outlook['outlook_confidence'], 1)

        # Check factors are lists
        self.assertIsInstance(economic_outlook['risk_factors'], list)
        self.assertIsInstance(economic_outlook['positive_factors'], list)

    def test_indicator_units(self):
        """Test indicator units are correctly assigned"""
        result = self.analyzer.get_nowcasting_analysis()
        nowcast_values = result['nowcast_values']

        expected_units = {
            'gdp_growth': 'percent_annual',
            'employment': 'index_50_neutral',
            'inflation': 'percent_annual',
            'consumer_spending': 'percent_growth',
            'business_investment': 'index_50_neutral',
            'trade_activity': 'index_50_neutral',
            'financial_conditions': 'index_50_neutral'
        }

        for indicator, expected_unit in expected_units.items():
            if indicator in nowcast_values:
                actual_unit = nowcast_values[indicator]['unit']
                self.assertEqual(actual_unit, expected_unit)


class TestAlternativeSignalsProvider(unittest.TestCase):
    """Test AlternativeSignalsProvider"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AlternativeSignalsConfig(
            cache_dir=os.path.join(self.temp_dir, "cache"),
            enable_cache=True
        )
        self.provider = AlternativeSignalsProvider(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_provider_initialization(self):
        """Test provider initialization"""
        self.assertIsInstance(self.provider.config, AlternativeSignalsConfig)
        self.assertIsInstance(
            self.provider.trends_analyzer,
            GoogleTrendsAnalyzer)
        self.assertIsInstance(
            self.provider.sentiment_analyzer,
            SocialSentimentAnalyzer)
        self.assertIsInstance(
            self.provider.nowcasting_analyzer,
            EconomicNowcastingAnalyzer)

        # Check cache directory creation
        self.assertTrue(os.path.exists(self.config.cache_dir))

    def test_factory_function(self):
        """Test factory function"""
        provider = get_alternative_signals_provider()
        self.assertIsInstance(provider, AlternativeSignalsProvider)

        # Test with custom config
        custom_config = AlternativeSignalsConfig(timeout=60)
        provider_custom = get_alternative_signals_provider(custom_config)
        self.assertEqual(provider_custom.config.timeout, 60)

    def test_comprehensive_analysis_all_components(self):
        """Test comprehensive analysis with all components"""
        result = self.provider.get_comprehensive_alternative_analysis(
            include_trends=True,
            include_sentiment=True,
            include_nowcasting=True
        )

        self.assertIsInstance(result, dict)
        self.assertIn('analyzed_at', result)
        self.assertIn('analysis_scope', result)
        self.assertIn('trends_analysis', result)
        self.assertIn('sentiment_analysis', result)
        self.assertIn('nowcasting_analysis', result)
        self.assertIn('integrated_signals', result)
        self.assertIn('early_warning_indicators', result)
        self.assertIn('alternative_outlook', result)
        self.assertTrue(result['success'])

        # Check analysis scope
        scope = result['analysis_scope']
        self.assertTrue(scope['trends_analysis'])
        self.assertTrue(scope['sentiment_analysis'])
        self.assertTrue(scope['nowcasting_analysis'])

        # Check that all analyses succeeded
        self.assertTrue(result['trends_analysis']['success'])
        self.assertTrue(result['sentiment_analysis']['success'])
        self.assertTrue(result['nowcasting_analysis']['success'])

    def test_comprehensive_analysis_selective_components(self):
        """Test comprehensive analysis with selective components"""
        # Test with only trends and sentiment
        result = self.provider.get_comprehensive_alternative_analysis(
            include_trends=True,
            include_sentiment=True,
            include_nowcasting=False
        )

        self.assertTrue(result['success'])

        scope = result['analysis_scope']
        self.assertTrue(scope['trends_analysis'])
        self.assertTrue(scope['sentiment_analysis'])
        self.assertFalse(scope['nowcasting_analysis'])

        # Check that trends and sentiment are populated
        self.assertNotEqual(result['trends_analysis'], {})
        self.assertNotEqual(result['sentiment_analysis'], {})
        self.assertEqual(result['nowcasting_analysis'], {})

    def test_integrated_signals(self):
        """Test integrated signals analysis"""
        result = self.provider.get_comprehensive_alternative_analysis()
        integrated_signals = result['integrated_signals']

        # Check integrated signals structure
        expected_fields = [
            'composite_scores', 'signal_consistency',
            'signal_strength', 'cross_validation'
        ]
        for field in expected_fields:
            self.assertIn(field, integrated_signals)

        # Check composite scores
        composite_scores = integrated_signals['composite_scores']
        expected_composites = [
            'economic_sentiment',
            'employment_conditions',
            'inflation_pressures']
        for composite in expected_composites:
            if composite in composite_scores:
                score = composite_scores[composite]
                self.assertGreaterEqual(score, -1)
                self.assertLessEqual(score, 1)

        # Check cross validation
        cross_validation = integrated_signals['cross_validation']
        self.assertIn('signal_count', cross_validation)
        self.assertIn('overall_confidence', cross_validation)
        self.assertGreaterEqual(cross_validation['overall_confidence'], 0)
        self.assertLessEqual(cross_validation['overall_confidence'], 1)

    def test_early_warning_indicators(self):
        """Test early warning indicators"""
        result = self.provider.get_comprehensive_alternative_analysis()
        early_warnings = result['early_warning_indicators']

        # Check warning structure
        expected_warnings = [
            'recession_warnings', 'inflation_warnings',
            'employment_warnings', 'financial_stress_warnings'
        ]
        for warning_type in expected_warnings:
            self.assertIn(warning_type, early_warnings)
            self.assertIsInstance(early_warnings[warning_type], list)

        # Check overall warning level
        self.assertIn('overall_warning_level', early_warnings)
        self.assertIn(early_warnings['overall_warning_level'],
                      ['low', 'moderate', 'high'])

        # Check total warnings
        self.assertIn('total_warnings', early_warnings)
        self.assertIsInstance(early_warnings['total_warnings'], int)
        self.assertGreaterEqual(early_warnings['total_warnings'], 0)

    def test_alternative_outlook(self):
        """Test alternative outlook generation"""
        result = self.provider.get_comprehensive_alternative_analysis()
        alt_outlook = result['alternative_outlook']

        # Check outlook structure
        expected_fields = [
            'economic_outlook', 'confidence_level', 'key_themes',
            'risk_assessment', 'opportunity_areas', 'outlook_summary'
        ]
        for field in expected_fields:
            self.assertIn(field, alt_outlook)

        # Check valid values
        self.assertIn(alt_outlook['economic_outlook'],
                      ['positive', 'negative', 'neutral'])
        self.assertIn(alt_outlook['confidence_level'],
                      ['high', 'moderate', 'low'])
        self.assertIn(alt_outlook['risk_assessment'],
                      ['high', 'moderate', 'low'])

        # Check that lists are properly formatted
        self.assertIsInstance(alt_outlook['key_themes'], list)
        self.assertIsInstance(alt_outlook['opportunity_areas'], list)

        # Check summary is a string
        self.assertIsInstance(alt_outlook['outlook_summary'], str)
        self.assertGreater(len(alt_outlook['outlook_summary']), 0)

    def test_session_configuration(self):
        """Test HTTP session configuration"""
        session = self.provider._get_session()

        self.assertIsNotNone(session)
        self.assertIn('User-Agent', session.headers)
        self.assertEqual(session.headers['User-Agent'],
                         'DCF Lab Alternative Signals Provider')

    def test_error_handling(self):
        """Test error handling in provider"""
        # Test with invalid configuration that might cause errors
        with patch.object(self.provider.trends_analyzer, 'get_trends_analysis',
                          side_effect=Exception("Test error")):
            result = self.provider.get_comprehensive_alternative_analysis(
                include_sentiment=False,
                include_nowcasting=False
            )

            # Should handle the error gracefully
            self.assertFalse(result['success'])
            self.assertIsNotNone(result['error'])


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestAlternativeSignalsConfig,
        TestGoogleTrendsAnalyzer,
        TestSocialSentimentAnalyzer,
        TestEconomicNowcastingAnalyzer,
        TestAlternativeSignalsProvider
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print("Alternative Signals Provider Tests Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) ==
                      0 and len(result.errors) == 0) else 1
    exit(exit_code)
