"""
Test Suite for News & Sentiment Analysis Provider

This module provides comprehensive tests for the news and sentiment analysis
functionality including GDELT analysis, news aggregation, and sentiment scoring.
"""

import os

# Import the components to test
import sys
from datetime import datetime

import pytest
from news_sentiment import (
    GDELTAnalyzer,
    NewsAggregator,
    NewsAnalysisConfig,
    get_news_analysis_provider,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNewsAnalysisConfig:
    """Test news analysis configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = NewsAnalysisConfig()

        assert config.cache_dir == "./cache/news_sentiment"
        assert config.enable_cache is True
        assert config.max_cache_age_hours == 6
        assert config.timeout == 30
        assert config.retries == 3
        assert config.rate_limit_delay == 1.0

        # URLs
        assert "gdeltproject.org" in config.gdelt_base_url
        assert "newsapi.org" in config.news_api_base_url

        # Sentiment settings
        assert config.sentiment_window_hours == 24
        assert config.event_impact_threshold == 0.5
        assert config.regime_change_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration"""
        config = NewsAnalysisConfig(
            cache_dir="./custom_cache",
            enable_cache=False,
            max_cache_age_hours=12,
            sentiment_window_hours=48
        )

        assert config.cache_dir == "./custom_cache"
        assert config.enable_cache is False
        assert config.max_cache_age_hours == 12
        assert config.sentiment_window_hours == 48


class TestGDELTAnalyzer:
    """Test GDELT analyzer functionality"""

    @pytest.fixture
    def gdelt_analyzer(self):
        """Create GDELT analyzer for testing"""
        config = NewsAnalysisConfig()
        return GDELTAnalyzer(config)

    def test_initialization(self, gdelt_analyzer):
        """Test GDELT analyzer initialization"""
        assert gdelt_analyzer.config is not None
        assert len(gdelt_analyzer.financial_event_codes) > 0
        assert len(gdelt_analyzer.market_themes) > 0

        # Check key event codes
        assert '01' in gdelt_analyzer.financial_event_codes
        assert '20' in gdelt_analyzer.financial_event_codes

        # Check key market themes
        assert 'ECON_RECESSION' in gdelt_analyzer.market_themes
        assert 'ECON_STOCKMARKET' in gdelt_analyzer.market_themes

    def test_get_gdelt_events(self, gdelt_analyzer):
        """Test GDELT events retrieval"""
        result = gdelt_analyzer.get_gdelt_events("financial markets", "1d")

        assert result['success'] is True
        assert result['query'] == "financial markets"
        assert result['timespan'] == "1d"
        assert 'retrieved_at' in result
        assert isinstance(result['events'], list)
        assert len(result['events']) > 0

        # Check event structure
        event = result['events'][0]
        required_fields = [
            'event_id', 'date', 'event_type', 'actor1', 'actor2',
            'event_code', 'goldstein_scale', 'avg_tone', 'num_mentions',
            'num_sources', 'num_articles', 'url', 'source_name',
            'themes', 'locations', 'relevance_score'
        ]

        for field in required_fields:
            assert field in event

        # Check data types
        assert isinstance(event['goldstein_scale'], (int, float))
        assert isinstance(event['avg_tone'], (int, float))
        assert isinstance(event['themes'], list)
        assert isinstance(event['locations'], list)
        assert -10 <= event['goldstein_scale'] <= 10  # GDELT scale
        assert -10 <= event['avg_tone'] <= 10  # GDELT tone scale

    def test_timespan_variations(self, gdelt_analyzer):
        """Test different timespan options"""
        timespans = ['1h', '6h', '1d', '3d', '1w']

        for timespan in timespans:
            result = gdelt_analyzer.get_gdelt_events("economy", timespan)
            assert result['success'] is True
            assert result['timespan'] == timespan
            assert len(result['events']) > 0

    def test_event_summary_analysis(self, gdelt_analyzer):
        """Test event summary analysis"""
        result = gdelt_analyzer.get_gdelt_events("markets", "1d")

        assert 'summary' in result
        summary = result['summary']

        # Check summary structure
        expected_keys = [
            'total_events', 'event_types', 'sentiment_distribution',
            'top_themes', 'geographic_distribution', 'source_distribution',
            'key_insights'
        ]

        for key in expected_keys:
            assert key in summary

        # Check sentiment distribution
        sentiment_dist = summary['sentiment_distribution']
        assert 'mean_tone' in sentiment_dist
        assert 'positive_events' in sentiment_dist
        assert 'negative_events' in sentiment_dist

        assert isinstance(sentiment_dist['mean_tone'], (int, float))
        assert sentiment_dist['positive_events'] >= 0
        assert sentiment_dist['negative_events'] >= 0

    def test_sentiment_metrics_calculation(self, gdelt_analyzer):
        """Test sentiment metrics calculation"""
        result = gdelt_analyzer.get_gdelt_events("financial crisis", "1d")

        assert 'sentiment_analysis' in result
        sentiment = result['sentiment_analysis']

        # Check sentiment metrics
        expected_keys = [
            'sentiment_score',
            'sentiment_momentum',
            'sentiment_volatility',
            'sentiment_trend',
            'regime_change_probability',
            'sentiment_extremes',
            'weighted_sentiment']

        for key in expected_keys:
            assert key in sentiment

        # Check value ranges
        assert -1 <= sentiment['sentiment_score'] <= 1
        assert sentiment['sentiment_volatility'] >= 0
        assert 0 <= sentiment['regime_change_probability'] <= 1
        assert sentiment['sentiment_trend'] in [
            'improving', 'deteriorating', 'stable', 'neutral']


class TestNewsAggregator:
    """Test news aggregator functionality"""

    @pytest.fixture
    def news_aggregator(self):
        """Create news aggregator for testing"""
        config = NewsAnalysisConfig()
        return NewsAggregator(config)

    def test_initialization(self, news_aggregator):
        """Test news aggregator initialization"""
        assert news_aggregator.config is not None
        assert len(news_aggregator.news_feeds) > 0

        # Check key news feeds
        assert 'reuters_business' in news_aggregator.news_feeds
        assert 'yahoo_finance' in news_aggregator.news_feeds

    def test_get_financial_news(self, news_aggregator):
        """Test financial news retrieval"""
        result = news_aggregator.get_financial_news(hours_back=24)

        assert result['success'] is True
        assert result['hours_back'] == 24
        assert 'retrieved_at' in result
        assert isinstance(result['articles'], list)
        assert len(result['articles']) > 0

        # Check article structure
        article = result['articles'][0]
        required_fields = [
            'id', 'title', 'source', 'published', 'url',
            'sentiment_score', 'relevance_score', 'category',
            'keywords', 'summary'
        ]

        for field in required_fields:
            assert field in article

        # Check data types and ranges
        assert isinstance(article['sentiment_score'], (int, float))
        assert isinstance(article['relevance_score'], (int, float))
        assert isinstance(article['keywords'], list)
        assert -1 <= article['sentiment_score'] <= 1
        assert 0 <= article['relevance_score'] <= 1

    def test_news_sentiment_analysis(self, news_aggregator):
        """Test news sentiment analysis"""
        result = news_aggregator.get_financial_news(hours_back=24)

        assert 'sentiment_analysis' in result
        sentiment = result['sentiment_analysis']

        # Check sentiment analysis structure
        expected_keys = [
            'overall_sentiment',
            'sentiment_distribution',
            'sentiment_trend',
            'sentiment_by_category',
            'sentiment_by_source',
            'market_sentiment_score']

        for key in expected_keys:
            assert key in sentiment

        # Check sentiment distribution
        dist = sentiment['sentiment_distribution']
        assert 'positive' in dist
        assert 'negative' in dist
        assert 'neutral' in dist
        assert 'positive_pct' in dist
        assert 'negative_pct' in dist

        # Check percentages sum to 1
        total_pct = dist['positive_pct'] + dist['negative_pct'] + \
            (dist['neutral'] / (dist['positive'] +
             dist['negative'] + dist['neutral']))
        assert abs(total_pct - 1.0) < 0.01  # Allow small floating point errors

    def test_topic_analysis(self, news_aggregator):
        """Test news topic analysis"""
        result = news_aggregator.get_financial_news(hours_back=24)

        assert 'topic_analysis' in result
        topics = result['topic_analysis']

        # Check topic analysis structure
        expected_keys = [
            'top_categories', 'trending_keywords', 'topic_sentiment',
            'emerging_themes'
        ]

        for key in expected_keys:
            assert key in topics

        # Check categories
        if topics['top_categories']:
            assert isinstance(topics['top_categories'], dict)

        # Check keywords
        if topics['trending_keywords']:
            assert isinstance(topics['trending_keywords'], dict)

        # Check emerging themes
        assert isinstance(topics['emerging_themes'], list)


class TestNewsAnalysisProvider:
    """Test comprehensive news analysis provider"""

    @pytest.fixture
    def provider(self):
        """Create news analysis provider for testing"""
        return get_news_analysis_provider()

    def test_initialization(self, provider):
        """Test provider initialization"""
        assert provider.config is not None
        assert provider.gdelt is not None
        assert provider.news_aggregator is not None
        assert hasattr(provider, '_session')

    def test_comprehensive_sentiment_analysis(self, provider):
        """Test comprehensive sentiment analysis"""
        result = provider.get_comprehensive_sentiment_analysis(
            "financial markets", "1d")

        assert result['success'] is True
        assert result['query'] == "financial markets"
        assert result['timespan'] == "1d"
        assert 'analyzed_at' in result

        # Check main components
        main_components = [
            'gdelt_analysis', 'news_analysis', 'combined_sentiment',
            'regime_indicators', 'risk_assessment', 'recommendations'
        ]

        for component in main_components:
            assert component in result

    def test_combined_sentiment_analysis(self, provider):
        """Test combined sentiment analysis"""
        result = provider.get_comprehensive_sentiment_analysis("economy", "1d")

        combined = result['combined_sentiment']

        # Check combined sentiment structure
        expected_keys = [
            'overall_sentiment_score',
            'confidence_level',
            'sentiment_sources',
            'sentiment_consistency',
            'momentum_indicators',
            'volatility_measures']

        for key in expected_keys:
            assert key in combined

        # Check value ranges
        assert -1 <= combined['overall_sentiment_score'] <= 1
        assert 0 <= combined['confidence_level'] <= 1
        assert 0 <= combined['sentiment_consistency'] <= 1

        # Check sentiment sources
        sources = combined['sentiment_sources']
        assert 'gdelt_score' in sources
        assert 'news_score' in sources
        assert 'gdelt_weight' in sources
        assert 'news_weight' in sources

    def test_regime_indicators_analysis(self, provider):
        """Test regime change indicators"""
        result = provider.get_comprehensive_sentiment_analysis(
            "market volatility", "1d")

        regime = result['regime_indicators']

        # Check regime indicators structure
        expected_keys = [
            'regime_change_probability', 'key_indicators', 'risk_factors',
            'stability_measures', 'early_warning_signals'
        ]

        for key in expected_keys:
            assert key in regime

        # Check probability range
        assert 0 <= regime['regime_change_probability'] <= 1

        # Check indicators are lists
        assert isinstance(regime['key_indicators'], list)
        assert isinstance(regime['risk_factors'], list)
        assert isinstance(regime['early_warning_signals'], list)

        # Check stability measures
        stability = regime['stability_measures']
        assert 'sentiment_stability' in stability
        assert 'source_agreement' in stability
        assert 'information_flow' in stability

    def test_risk_assessment(self, provider):
        """Test risk assessment functionality"""
        result = provider.get_comprehensive_sentiment_analysis("crisis", "1d")

        risk = result['risk_assessment']

        # Check risk assessment structure
        expected_keys = [
            'overall_risk_level', 'risk_score', 'primary_risks',
            'risk_mitigation', 'monitoring_priority'
        ]

        for key in expected_keys:
            assert key in risk

        # Check risk level values
        valid_risk_levels = ['low', 'moderate', 'high', 'extreme']
        assert risk['overall_risk_level'] in valid_risk_levels

        # Check risk score range
        assert 0 <= risk['risk_score'] <= 1

        # Check monitoring priority
        valid_priorities = ['low', 'moderate', 'high', 'immediate']
        assert risk['monitoring_priority'] in valid_priorities

        # Check lists
        assert isinstance(risk['primary_risks'], list)
        assert isinstance(risk['risk_mitigation'], list)

    def test_recommendations_generation(self, provider):
        """Test recommendations generation"""
        result = provider.get_comprehensive_sentiment_analysis(
            "bull market", "1d")

        recommendations = result['recommendations']

        # Check recommendations structure
        expected_keys = [
            'trading_implications',
            'risk_management',
            'monitoring_suggestions',
            'market_outlook',
            'confidence_assessment']

        for key in expected_keys:
            assert key in recommendations

        # Check outlook values
        valid_outlooks = ['positive', 'negative', 'neutral']
        assert recommendations['market_outlook'] in valid_outlooks

        # Check confidence assessment
        valid_confidence = ['low', 'moderate', 'high']
        assert recommendations['confidence_assessment'] in valid_confidence

        # Check recommendation lists
        assert isinstance(recommendations['trading_implications'], list)
        assert isinstance(recommendations['risk_management'], list)
        assert isinstance(recommendations['monitoring_suggestions'], list)

    def test_different_queries(self, provider):
        """Test provider with different query types"""
        queries = [
            "stock market",
            "inflation",
            "interest rates",
            "recession",
            "trade war",
            "cryptocurrency"
        ]

        for query in queries:
            result = provider.get_comprehensive_sentiment_analysis(query, "6h")
            assert result['success'] is True
            assert result['query'] == query
            assert len(result['gdelt_analysis']['events']) > 0
            assert len(result['news_analysis']['articles']) > 0

    def test_different_timespans(self, provider):
        """Test provider with different time spans"""
        timespans = ['1h', '6h', '1d', '3d']

        for timespan in timespans:
            result = provider.get_comprehensive_sentiment_analysis(
                "markets", timespan)
            assert result['success'] is True
            assert result['timespan'] == timespan


class TestIntegration:
    """Integration tests for news analysis provider"""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Create provider
        provider = get_news_analysis_provider()

        # Run comprehensive analysis
        result = provider.get_comprehensive_sentiment_analysis(
            "financial markets", "1d")

        # Verify complete workflow
        assert result['success'] is True

        # Check all major components completed
        assert result['gdelt_analysis']['success'] is True
        assert result['news_analysis']['success'] is True
        assert len(result['combined_sentiment']) > 0
        assert len(result['regime_indicators']) > 0
        assert len(result['risk_assessment']) > 0
        assert len(result['recommendations']) > 0

        # Verify data consistency
        gdelt_events = len(result['gdelt_analysis']['events'])
        news_articles = len(result['news_analysis']['articles'])

        assert gdelt_events > 0
        assert news_articles > 0

        # Verify sentiment scores are reasonable
        combined_sentiment = result['combined_sentiment'][
            'overall_sentiment_score']
        gdelt_sentiment = result['gdelt_analysis']['sentiment_analysis'][
            'sentiment_score']
        news_sentiment = result['news_analysis']['sentiment_analysis'][
            'market_sentiment_score']

        assert -1 <= combined_sentiment <= 1
        assert -1 <= gdelt_sentiment <= 1
        assert -1 <= news_sentiment <= 1

    def test_data_quality_validation(self):
        """Test data quality and validation"""
        provider = get_news_analysis_provider()
        result = provider.get_comprehensive_sentiment_analysis(
            "validation test", "1d")

        # Check GDELT data quality
        gdelt_events = result['gdelt_analysis']['events']
        for event in gdelt_events[:5]:  # Check first 5 events
            # Validate required fields
            assert 'event_id' in event
            assert 'date' in event
            assert 'avg_tone' in event

            # Validate date format
            datetime.fromisoformat(event['date'].replace('Z', '+00:00'))

            # Validate tone range
            assert -10 <= event['avg_tone'] <= 10

        # Check news data quality
        news_articles = result['news_analysis']['articles']
        for article in news_articles[:5]:  # Check first 5 articles
            # Validate required fields
            assert 'title' in article
            assert 'published' in article
            assert 'sentiment_score' in article

            # Validate sentiment range
            assert -1 <= article['sentiment_score'] <= 1

    def test_error_handling(self):
        """Test error handling capabilities"""
        # Test with empty query
        provider = get_news_analysis_provider()
        result = provider.get_comprehensive_sentiment_analysis("", "1d")

        # Should still work with empty query
        assert result is not None

        # Test with invalid timespan (should default to valid behavior)
        result = provider.get_comprehensive_sentiment_analysis(
            "test", "invalid")
        assert result is not None

# Performance and benchmarking tests


class TestPerformance:
    """Performance tests for news analysis provider"""

    def test_analysis_speed(self):
        """Test analysis completion speed"""
        import time

        provider = get_news_analysis_provider()

        start_time = time.time()
        result = provider.get_comprehensive_sentiment_analysis(
            "performance test", "1d")
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 5.0  # 5 seconds max for mock data
        assert result['success'] is True

    def test_memory_efficiency(self):
        """Test memory usage during analysis"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        provider = get_news_analysis_provider()
        result = provider.get_comprehensive_sentiment_analysis(
            "memory test", "1w")

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Should not use excessive memory (adjust threshold as needed)
        assert memory_increase < 100 * 1024 * 1024  # 100MB max increase
        assert result['success'] is True


if __name__ == "__main__":
    # Run basic tests
    print("=== Running News & Sentiment Analysis Tests ===")

    # Test configuration
    print("\n1. Testing Configuration...")
    config = NewsAnalysisConfig()
    assert config.cache_dir == "./cache/news_sentiment"
    print("✅ Configuration test passed")

    # Test GDELT analyzer
    print("\n2. Testing GDELT Analyzer...")
    gdelt = GDELTAnalyzer(config)
    gdelt_result = gdelt.get_gdelt_events("test", "1d")
    assert gdelt_result['success'] is True
    assert len(gdelt_result['events']) > 0
    print(
        f"✅ GDELT analyzer test passed - {len(gdelt_result['events'])} events")

    # Test news aggregator
    print("\n3. Testing News Aggregator...")
    news_agg = NewsAggregator(config)
    news_result = news_agg.get_financial_news(24)
    assert news_result['success'] is True
    assert len(news_result['articles']) > 0
    print(
        f"✅ News aggregator test passed - {len(news_result['articles'])} articles")

    # Test provider
    print("\n4. Testing News Analysis Provider...")
    provider = get_news_analysis_provider()
    analysis_result = provider.get_comprehensive_sentiment_analysis(
        "test", "1d")
    assert analysis_result['success'] is True
    print("✅ Provider test passed")

    # Test sentiment metrics
    print("\n5. Testing Sentiment Metrics...")
    combined = analysis_result['combined_sentiment']
    assert 'overall_sentiment_score' in combined
    assert 'confidence_level' in combined
    assert -1 <= combined['overall_sentiment_score'] <= 1
    assert 0 <= combined['confidence_level'] <= 1
    print("✅ Sentiment metrics test passed")

    print("\n=== All News & Sentiment Analysis Tests Passed ===")
    print("🚀 News analysis provider is ready for production use!")
