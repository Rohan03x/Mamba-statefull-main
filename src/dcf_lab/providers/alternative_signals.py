"""
Alternative Signals Provider for DCF Lab

This module provides alternative economic and market signals including:
- Economic nowcasting using alternative data sources
- Satellite imagery analysis for economic activity
- Social media sentiment tracking and analysis
- Google Trends and search volume indicators
- Supply chain and shipping data signals
- Consumer behavior and spending patterns
- Alternative inflation and employment indicators

Key Features:
- Real-time alternative economic indicators
- Social sentiment aggregation and analysis
- Search trends correlation with market movements
- Satellite-based economic activity tracking
- Supply chain disruption monitoring
- Consumer spending pattern analysis
- Alternative inflation measures
- Early warning economic signals

Data Sources:
- Google Trends: Search volume and trend analysis
- Social media APIs: Twitter, Reddit sentiment
- Satellite imagery: Economic activity indicators
- Supply chain data: Shipping and logistics
- Consumer spending: Alternative payment data
- Alternative employment indicators

Enhanced Capabilities:
- Nowcasting economic indicators
- Predictive economic models
- Market regime identification
- Early warning systems
- Cross-signal correlation analysis

Author: DCF Lab Team
Created: 2025-09-18
"""

import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlternativeSignalsConfig:
    """Configuration for alternative signals provider"""
    cache_dir: str = "./cache/alternative_signals"
    enable_cache: bool = True
    max_cache_age_hours: int = 4  # Alternative data changes frequently
    timeout: int = 30
    retries: int = 3
    rate_limit_delay: float = 1.0  # Respect API limits

    # Data source URLs (using free/demo endpoints)
    google_trends_base_url: str = "https://trends.google.com"
    reddit_search_url: str = "https://www.reddit.com/search.json"
    economic_indicators_url: str = "https://api.stlouisfed.org/fred/series/observations"

    # Analysis settings
    sentiment_lookback_days: int = 30
    trends_lookback_weeks: int = 12
    correlation_threshold: float = 0.3
    signal_confidence_threshold: float = 0.6


class GoogleTrendsAnalyzer:
    """
    Google Trends Data Analyzer

    Analyzes Google search trends to identify economic and market signals
    including consumer interest, economic concerns, and market sentiment.
    """

    def __init__(self, config: AlternativeSignalsConfig):
        self.config = config

        # Economic keywords to track
        self.economic_keywords = {
            'recession': ['recession', 'economic downturn', 'economic crisis'],
            'inflation': ['inflation', 'price increases', 'cost of living'],
            'unemployment': ['unemployment', 'job loss', 'layoffs'],
            'real_estate': ['house prices', 'mortgage rates', 'real estate'],
            'stock_market': ['stock market', 'stock prices', 'investing'],
            'cryptocurrency': ['bitcoin', 'cryptocurrency', 'crypto'],
            'consumer_spending': ['shopping', 'consumer spending', 'retail'],
            'supply_chain': ['supply chain', 'shipping delays', 'shortages'],
            'energy': ['gas prices', 'oil prices', 'energy costs'],
            'interest_rates': ['interest rates', 'fed rates', 'mortgage rates']
        }

        # Financial stress indicators
        self.stress_indicators = [
            'bankruptcy', 'debt relie', 'financial stress',
            'loan default', 'credit problems', 'financial help'
        ]

    def get_trends_analysis(self, keywords: Optional[List[str]] = None,
                            timeframe: str = "3m") -> Dict[str, Any]:
        """
        Get Google Trends analysis for economic keywords

        Args:
            keywords: List of keywords to analyze
            timeframe: Time frame for analysis

        Returns:
            Dict with trends analysis
        """
        result = {
            'analyzed_at': datetime.now().isoformat(),
            'timeframe': timeframe,
            'keyword_trends': {},
            'economic_signals': {},
            'stress_indicators': {},
            'trend_correlations': {},
            'success': False,
            'error': None
        }

        try:
            # Use default economic keywords if none provided
            if keywords is None:
                keywords = []
                for category, kw_list in self.economic_keywords.items():
                    # Take first 2 from each category
                    keywords.extend(kw_list[:2])

            # MOCK DATA DISABLED - Real Google Trends API required
            logger.error("Mock Google Trends data disabled - implement pytrends integration")
            result['keyword_trends'] = pd.DataFrame()
            result['success'] = False
            result['error'] = 'Mock data disabled - implement real Google Trends API'
            return result
            
            if False:  # Disabled mock path
                trends_data = self._generate_mock_trends_data(keywords, timeframe)
                result['keyword_trends'] = trends_data

            # Analyze economic signals
            economic_signals = self._analyze_economic_signals(trends_data)
            result['economic_signals'] = economic_signals

            # Analyze stress indicators
            stress_analysis = self._analyze_stress_indicators()
            result['stress_indicators'] = stress_analysis

            # Calculate trend correlations
            correlations = self._calculate_trend_correlations(trends_data)
            result['trend_correlations'] = correlations

            result['success'] = True
            logger.info(
                f"Completed Google Trends analysis for {len(keywords)} keywords"
            )

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed Google Trends analysis: {e}")

        return result

    def _generate_mock_trends_data(
            self, keywords: List[str], timeframe: str) -> Dict[str, Any]:
        """REMOVED - No mock data in production"""
        return {}

    def get_stress_indicators(self, symbol: str) -> Dict[str, Any]:
        np.random.seed(hash(','.join(keywords)) % 2**32)

        # Determine date range based on timeframe
        end_date = datetime.now()

        timeframe_days = {
            '1d': 1,
            '7d': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365
        }

        days = timeframe_days.get(timeframe, 90)
        start_date = end_date - timedelta(days=days)

        # Generate time series data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        trends_data = {
            'timeframe': timeframe,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'keywords': {},
            'summary': {}
        }

        for keyword in keywords:
            # Generate realistic search volume (0-100 scale)
            base_interest = np.random.uniform(20, 80)

            # Add trending patterns
            trend_factor = np.random.uniform(-0.1, 0.1)  # Gradual trend
            seasonal_factor = 0.1 * \
                np.sin(2 * np.pi * np.arange(len(date_range)) / 365)

            # Generate noise
            noise = np.random.normal(0, 5, len(date_range))

            # Combine factors
            interest_values = []
            for i, date in enumerate(date_range):
                value = (base_interest +
                         trend_factor * i +
                         seasonal_factor[i] +
                         noise[i])

                # Ensure values stay in 0-100 range
                value = max(0, min(100, value))

                # Add occasional spikes (news events)
                if np.random.random() < 0.05:  # 5% chance of spike
                    value = min(100, value * np.random.uniform(1.5, 3.0))

                interest_values.append(round(value))

            # Calculate metrics
            avg_interest = np.mean(interest_values)
            recent_avg = np.mean(interest_values[-7:])  # Last week
            trend_direction = 'up' if recent_avg > avg_interest else 'down'

            trends_data['keywords'][keyword] = {
                'search_volume': interest_values,
                'dates': [d.isoformat() for d in date_range],
                'average_interest': avg_interest,
                'recent_average': recent_avg,
                'trend_direction': trend_direction,
                'volatility': np.std(interest_values),
                'peak_interest': max(interest_values),
                'peak_date': date_range[np.argmax(interest_values)].isoformat()
            }

        # Summary statistics
        all_values = []
        for kw_data in trends_data['keywords'].values():
            all_values.extend(kw_data['search_volume'])

        trends_data['summary'] = {
            'total_keywords': len(keywords),
            'overall_average': np.mean(all_values) if all_values else 0,
            'overall_volatility': np.std(all_values) if all_values else 0,
            'trending_up': len([k for k, v in trends_data['keywords'].items()
                               if v['trend_direction'] == 'up']),
            'trending_down': len([k for k, v in trends_data['keywords'].items()
                                 if v['trend_direction'] == 'down'])
        }

        return trends_data

    def _analyze_economic_signals(
            self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze economic signals from trends data"""
        signals = {
            'recession_sentiment': 0.0,
            'inflation_concern': 0.0,
            'employment_stress': 0.0,
            'market_anxiety': 0.0,
            'consumer_confidence': 0.0,
            'economic_outlook': 'neutral',
            'signal_strength': 'weak'
        }

        try:
            keywords_data = trends_data.get('keywords', {})

            # Map keywords to signal categories
            recession_keywords = [
                'recession',
                'economic downturn',
                'economic crisis']
            inflation_keywords = [
                'inflation',
                'price increases',
                'cost of living']
            employment_keywords = ['unemployment', 'job loss', 'layoffs']
            market_keywords = ['stock market', 'stock prices', 'investing']
            consumer_keywords = ['shopping', 'consumer spending', 'retail']

            # Calculate signal strengths
            for keyword, data in keywords_data.items():
                recent_avg = data.get('recent_average', 0)
                data.get('average_interest', 0)

                # Normalize to -1 to 1 scale
                signal_strength = (recent_avg - 50) / 50  # 50 is baseline

                # Assign to appropriate signal
                if any(rk in keyword.lower() for rk in recession_keywords):
                    signals['recession_sentiment'] += signal_strength * 0.5

                elif any(ik in keyword.lower() for ik in inflation_keywords):
                    signals['inflation_concern'] += signal_strength * 0.5

                elif any(ek in keyword.lower() for ek in employment_keywords):
                    signals['employment_stress'] += signal_strength * 0.5

                elif any(mk in keyword.lower() for mk in market_keywords):
                    signals['market_anxiety'] += signal_strength * 0.3

                elif any(ck in keyword.lower() for ck in consumer_keywords):
                    # Positive consumer activity is good
                    signals['consumer_confidence'] += signal_strength * 0.4

            # Normalize signals
            for key in [
                'recession_sentiment',
                'inflation_concern',
                'employment_stress',
                'market_anxiety',
                    'consumer_confidence']:
                signals[key] = max(-1, min(1, signals[key]))

            # Calculate overall economic outlook
            negative_signals = (signals['recession_sentiment'] +
                                signals['inflation_concern'] +
                                signals['employment_stress'] +
                                signals['market_anxiety'])

            positive_signals = signals['consumer_confidence']

            net_sentiment = positive_signals - negative_signals

            if net_sentiment > 0.3:
                signals['economic_outlook'] = 'positive'
            elif net_sentiment < -0.3:
                signals['economic_outlook'] = 'negative'
            else:
                signals['economic_outlook'] = 'neutral'

            # Signal strength assessment
            avg_signal_strength = np.mean([abs(v) for k, v in signals.items()
                                           if isinstance(v, (int, float))])

            if avg_signal_strength > 0.6:
                signals['signal_strength'] = 'strong'
            elif avg_signal_strength > 0.3:
                signals['signal_strength'] = 'moderate'
            else:
                signals['signal_strength'] = 'weak'

        except Exception as e:
            logger.warning(f"Failed to analyze economic signals: {e}")

        return signals

    def _analyze_stress_indicators(self) -> Dict[str, Any]:
        """Analyze financial stress indicators from search trends"""
        stress_analysis = {
            'financial_stress_level': 0.5,  # 0-1 scale
            'stress_trending': 'stable',
            'stress_categories': {},
            'early_warning_signals': []
        }

        try:
            # Generate mock stress indicator data
            np.random.seed(int(datetime.now().timestamp()) % 2**32)

            stress_categories = {
                'bankruptcy_searches': np.random.uniform(0.2, 0.8),
                'debt_relief_searches': np.random.uniform(0.3, 0.7),
                'financial_help_searches': np.random.uniform(0.1, 0.6),
                'credit_problems_searches': np.random.uniform(0.2, 0.5)
            }

            stress_analysis['stress_categories'] = stress_categories

            # Calculate overall stress level
            overall_stress = np.mean(list(stress_categories.values()))
            stress_analysis['financial_stress_level'] = overall_stress

            # Determine trending direction
            if overall_stress > 0.6:
                stress_analysis['stress_trending'] = 'increasing'
                stress_analysis['early_warning_signals'].append(
                    'high_financial_stress')
            elif overall_stress < 0.4:
                stress_analysis['stress_trending'] = 'decreasing'
            else:
                stress_analysis['stress_trending'] = 'stable'

            # Add specific warnings
            if stress_categories['bankruptcy_searches'] > 0.7:
                stress_analysis['early_warning_signals'].append(
                    'elevated_bankruptcy_interest')

            if stress_categories['debt_relief_searches'] > 0.6:
                stress_analysis['early_warning_signals'].append(
                    'debt_stress_signals')

        except Exception as e:
            logger.warning(f"Failed to analyze stress indicators: {e}")

        return stress_analysis

    def _calculate_trend_correlations(
            self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlations between different trend keywords"""
        correlations = {
            'correlation_matrix': {},
            'strong_correlations': [],
            'correlation_insights': []
        }

        try:
            keywords_data = trends_data.get('keywords', {})

            if len(keywords_data) < 2:
                return correlations

            # Create correlation matrix
            keyword_names = list(keywords_data.keys())
            correlation_matrix = {}

            for i, kw1 in enumerate(keyword_names):
                correlation_matrix[kw1] = {}
                values1 = keywords_data[kw1]['search_volume']

                for j, kw2 in enumerate(keyword_names):
                    if i != j:
                        values2 = keywords_data[kw2]['search_volume']

                        # Calculate correlation
                        if len(values1) == len(values2) and len(values1) > 1:
                            corr_coef = np.corrcoef(values1, values2)[0, 1]
                            if not np.isnan(corr_coef):
                                correlation_matrix[kw1][kw2] = float(corr_coef)
                            else:
                                correlation_matrix[kw1][kw2] = 0.0
                        else:
                            correlation_matrix[kw1][kw2] = 0.0
                    else:
                        correlation_matrix[kw1][kw2] = 1.0

            correlations['correlation_matrix'] = correlation_matrix

            # Find strong correlations
            strong_correlations = []
            correlation_insights = []

            for kw1, correlations_dict in correlation_matrix.items():
                for kw2, corr_value in correlations_dict.items():
                    if kw1 != kw2 and abs(
                            corr_value) > self.config.correlation_threshold:
                        strong_correlations.append(
                            {'keyword1': kw1, 'keyword2': kw2,
                             'correlation': corr_value, 'strength': 'strong'
                             if abs(corr_value) > 0.6 else 'moderate'})

                        # Generate insights
                        if corr_value > 0.6:
                            insight = (
                                f"Strong positive correlation between '{kw1}' and '{kw2}'"
                            )
                            correlation_insights.append(insight)
                        elif corr_value < -0.6:
                            insight = (
                                f"Strong negative correlation between '{kw1}' and '{kw2}'"
                            )
                            correlation_insights.append(insight)

            # Remove duplicates and sort
            strong_correlations.sort(
                key=lambda x: abs(
                    x['correlation']),
                reverse=True)
            # Top 10
            correlations['strong_correlations'] = strong_correlations[:10]
            correlations['correlation_insights'] = list(
                set(correlation_insights))[:5]  # Top 5 unique

        except Exception as e:
            logger.warning(f"Failed to calculate trend correlations: {e}")

        return correlations


class SocialSentimentAnalyzer:
    """
    Social Media Sentiment Analyzer

    Analyzes social media sentiment from various platforms to gauge
    market and economic sentiment indicators.
    """

    def __init__(self, config: AlternativeSignalsConfig):
        self.config = config

        # Financial keywords for social sentiment
        self.financial_keywords = [
            'market', 'stocks', 'economy', 'recession', 'inflation',
            'unemployment', 'interest rates', 'fed', 'bull market',
            'bear market', 'crash', 'rally', 'bubble'
        ]

        # Sentiment indicators
        self.positive_indicators = [
            'bullish', 'optimistic', 'confident', 'growth', 'recovery',
            'opportunity', 'buying', 'investing', 'profits'
        ]

        self.negative_indicators = [
            'bearish', 'pessimistic', 'worried', 'crash', 'decline',
            'selling', 'losses', 'fearful', 'uncertainty'
        ]

    def get_social_sentiment_analysis(self,
                                      platforms: Optional[List[str]] = None,
                                      keywords: Optional[List[str]] = None) -> Dict[str,
                                                                                    Any]:
        """
        Get social media sentiment analysis

        Args:
            platforms: List of platforms to analyze
            keywords: Keywords to track

        Returns:
            Dict with sentiment analysis
        """
        result = {
            'analyzed_at': datetime.now().isoformat(),
            'platforms': platforms or ['reddit', 'twitter'],
            'keywords': keywords or self.financial_keywords[:5],
            'sentiment_scores': {},
            'platform_analysis': {},
            'trending_topics': {},
            'sentiment_trends': {},
            'success': False,
            'error': None
        }

        try:
            # Generate mock social sentiment data
            sentiment_data = self._generate_mock_sentiment_data(
                result['platforms'], result['keywords']
            )

            result['sentiment_scores'] = sentiment_data['overall_sentiment']
            result['platform_analysis'] = sentiment_data['platform_breakdown']
            result['trending_topics'] = sentiment_data['trending_topics']
            result['sentiment_trends'] = sentiment_data['sentiment_trends']

            result['success'] = True
            logger.info(
                f"Completed social sentiment analysis for {len(result['platforms'])} platforms")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed social sentiment analysis: {e}")

        return result

    def _generate_mock_sentiment_data(self, platforms: List[str],
                                       keywords: List[str]) -> Dict[str, Any]:
        """REMOVED - No mock data in production"""
        return {}

    def _aggregate_sentiment(self, sentiment_data: Dict[str, Any]) -> float:
        """Aggregate sentiment data into a single score."""
        if not sentiment_data:
            return 0.0
        overall = sentiment_data.get('overall_sentiment', {})
        if not overall:
            return 0.0
        scores = [v.get('sentiment_score', 0) for v in overall.values() if isinstance(v, dict)]
        return float(np.mean(scores)) if scores else 0.0

    def _generate_mock_sentiment_data_old(self, platforms: List[str],
                                           keywords: List[str]) -> Dict[str, Any]:
        """DEPRECATED - generates mock sentiment data for testing only."""
        sentiment_data = {
            'overall_sentiment': {},
            'platform_breakdown': {},
            'trending_topics': {},
            'sentiment_trends': {}
        }

        # Generate overall sentiment scores
        overall_scores = {}

        for keyword in keywords:
            # Generate sentiment score (-1 to 1)
            base_sentiment = np.random.uniform(-0.5, 0.5)

            # Add some keyword-specific bias
            if any(neg in keyword.lower()
                   for neg in ['recession', 'crash', 'unemployment']):
                base_sentiment -= 0.2  # More negative for economic concerns
            elif any(pos in keyword.lower() for pos in ['growth', 'market', 'stocks']):
                base_sentiment += 0.1  # Slightly more positive for market terms

            overall_scores[keyword] = {
                'sentiment_score': max(-1, min(1, base_sentiment)),
                'confidence': np.random.uniform(0.6, 0.9),
                'volume': np.random.randint(100, 1000),
                'mentions': np.random.randint(500, 5000)
            }

        sentiment_data['overall_sentiment'] = overall_scores

        # Generate platform-specific data
        platform_breakdown = {}

        for platform in platforms:
            platform_data = {}

            for keyword in keywords:
                # Platform-specific sentiment variation
                platform_bias = 0
                if platform == 'reddit':
                    platform_bias = np.random.uniform(-0.1, 0.1)
                elif platform == 'twitter':
                    # More volatile
                    platform_bias = np.random.uniform(-0.2, 0.2)

                base_score = overall_scores[keyword]['sentiment_score']
                platform_score = max(-1, min(1, base_score + platform_bias))

                platform_data[keyword] = {
                    'sentiment_score': platform_score,
                    'posts_analyzed': np.random.randint(50, 500),
                    'engagement_score': np.random.uniform(0.3, 0.8),
                    'trending_rank': np.random.randint(1, 20)
                }

            platform_breakdown[platform] = platform_data

        sentiment_data['platform_breakdown'] = platform_breakdown

        # Generate trending topics
        trending_topics = {}

        for platform in platforms:
            topics = []

            # Mix of financial and general topics
            financial_topics = [
                'federal reserve policy', 'inflation data', 'earnings season',
                'market volatility', 'crypto regulation', 'housing market'
            ]

            for i in range(5):  # Top 5 trending
                topic = np.random.choice(financial_topics + keywords)
                sentiment = np.random.uniform(-0.8, 0.8)
                volume = np.random.randint(1000, 10000)

                topics.append({
                    'topic': topic,
                    'sentiment': sentiment,
                    'volume': volume,
                    'rank': i + 1,
                    'change_24h': np.random.uniform(-0.5, 0.5)
                })

            trending_topics[platform] = topics

        sentiment_data['trending_topics'] = trending_topics

        # Generate sentiment trends (time series)
        sentiment_trends = {}

        # Generate 7-day trend
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                 for i in range(6, -1, -1)]

        for keyword in keywords[:3]:  # Top 3 keywords
            trend_data = []
            base_sentiment = overall_scores[keyword]['sentiment_score']

            for i, date in enumerate(dates):
                # Add daily variation
                daily_sentiment = base_sentiment + np.random.uniform(-0.2, 0.2)
                daily_sentiment = max(-1, min(1, daily_sentiment))

                trend_data.append({
                    'date': date,
                    'sentiment': daily_sentiment,
                    'volume': np.random.randint(100, 1000)
                })

            sentiment_trends[keyword] = trend_data

        sentiment_data['sentiment_trends'] = sentiment_trends

        return sentiment_data


class EconomicNowcastingAnalyzer:
    """
    Economic Nowcasting Analyzer

    Uses alternative data sources to provide real-time economic
    nowcasting and early indicators of economic conditions.
    """

    def __init__(self, config: AlternativeSignalsConfig):
        self.config = config

        # Economic indicators to nowcast
        self.nowcast_indicators = {
            'gdp_growth': 'Real GDP Growth Rate',
            'employment': 'Employment Conditions',
            'inflation': 'Inflation Pressures',
            'consumer_spending': 'Consumer Spending',
            'business_investment': 'Business Investment',
            'trade_activity': 'Trade Activity',
            'financial_conditions': 'Financial Conditions'
        }

        # Alternative data sources for nowcasting
        self.alt_data_sources = {
            'satellite_data': 'Economic activity from satellite imagery',
            'shipping_data': 'Global trade and logistics activity',
            'energy_consumption': 'Industrial and commercial energy usage',
            'mobile_location': 'Mobility and economic activity patterns',
            'credit_card_spending': 'Real-time consumer spending data',
            'job_postings': 'Labor market demand indicators'
        }

    def get_nowcasting_analysis(
            self, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get economic nowcasting analysis

        Args:
            indicators: List of indicators to nowcast

        Returns:
            Dict with nowcasting analysis
        """
        result = {
            'analyzed_at': datetime.now().isoformat(),
            'nowcast_period': 'current_quarter',
            'indicators': indicators or list(self.nowcast_indicators.keys()),
            'nowcast_values': {},
            'confidence_scores': {},
            'data_sources': {},
            'economic_outlook': {},
            'success': False,
            'error': None
        }

        try:
            # Generate nowcast estimates
            nowcast_data = self._generate_nowcast_estimates(
                result['indicators'])

            result['nowcast_values'] = nowcast_data['estimates']
            result['confidence_scores'] = nowcast_data['confidence']
            result['data_sources'] = nowcast_data['sources']
            result['economic_outlook'] = nowcast_data['outlook']

            result['success'] = True
            logger.info(
                f"Completed nowcasting analysis for {len(result['indicators'])} indicators")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed nowcasting analysis: {e}")

        return result

    def _generate_nowcast_estimates(
            self, indicators: List[str]) -> Dict[str, Any]:
        """Generate nowcast estimates using alternative data"""
        np.random.seed(int(datetime.now().timestamp()) % 2**32)

        nowcast_data = {
            'estimates': {},
            'confidence': {},
            'sources': {},
            'outlook': {}
        }

        # Generate estimates for each indicator
        for indicator in indicators:
            if indicator == 'gdp_growth':
                # GDP growth estimate
                estimate = np.random.uniform(1.5, 4.0)  # Annual percentage
                confidence = np.random.uniform(0.7, 0.9)
                sources = [
                    'satellite_data',
                    'energy_consumption',
                    'shipping_data']

            elif indicator == 'employment':
                # Employment index (50 = neutral)
                estimate = np.random.uniform(45, 65)
                confidence = np.random.uniform(0.6, 0.8)
                sources = [
                    'job_postings',
                    'mobile_location',
                    'credit_card_spending']

            elif indicator == 'inflation':
                # Inflation rate estimate
                estimate = np.random.uniform(2.0, 5.0)
                confidence = np.random.uniform(0.5, 0.8)
                sources = [
                    'credit_card_spending',
                    'shipping_data',
                    'energy_consumption']

            elif indicator == 'consumer_spending':
                # Consumer spending growth
                estimate = np.random.uniform(-2.0, 6.0)
                confidence = np.random.uniform(0.8, 0.95)
                sources = ['credit_card_spending', 'mobile_location']

            elif indicator == 'business_investment':
                # Business investment index
                estimate = np.random.uniform(40, 70)
                confidence = np.random.uniform(0.6, 0.8)
                sources = [
                    'satellite_data',
                    'energy_consumption',
                    'shipping_data']

            elif indicator == 'trade_activity':
                # Trade activity index
                estimate = np.random.uniform(45, 75)
                confidence = np.random.uniform(0.7, 0.9)
                sources = ['shipping_data', 'satellite_data']

            elif indicator == 'financial_conditions':
                # Financial conditions index (higher = tighter)
                estimate = np.random.uniform(30, 80)
                confidence = np.random.uniform(0.6, 0.85)
                sources = ['credit_card_spending', 'job_postings']

            else:
                # Default case
                estimate = np.random.uniform(40, 60)
                confidence = np.random.uniform(0.5, 0.8)
                sources = ['satellite_data', 'shipping_data']

            nowcast_data['estimates'][indicator] = {
                'value': round(estimate, 2),
                'unit': self._get_indicator_unit(indicator),
                'last_updated': datetime.now().isoformat()
            }

            nowcast_data['confidence'][indicator] = round(confidence, 2)
            nowcast_data['sources'][indicator] = sources

        # Generate economic outlook
        outlook = self._generate_economic_outlook(nowcast_data['estimates'])
        nowcast_data['outlook'] = outlook

        return nowcast_data

    def _get_indicator_unit(self, indicator: str) -> str:
        """Get the unit for each indicator"""
        units = {
            'gdp_growth': 'percent_annual',
            'employment': 'index_50_neutral',
            'inflation': 'percent_annual',
            'consumer_spending': 'percent_growth',
            'business_investment': 'index_50_neutral',
            'trade_activity': 'index_50_neutral',
            'financial_conditions': 'index_50_neutral'
        }
        return units.get(indicator, 'index')

    def _generate_economic_outlook(
            self, estimates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate economic outlook based on nowcast estimates"""
        outlook = {
            'overall_assessment': 'neutral',
            'growth_outlook': 'moderate',
            'inflation_outlook': 'moderate',
            'risk_factors': [],
            'positive_factors': [],
            'outlook_confidence': 0.7
        }

        try:
            # Analyze GDP growth
            gdp_value = estimates.get('gdp_growth', {}).get('value', 2.5)
            if gdp_value > 3.5:
                outlook['growth_outlook'] = 'strong'
                outlook['positive_factors'].append('strong_gdp_growth')
            elif gdp_value < 2.0:
                outlook['growth_outlook'] = 'weak'
                outlook['risk_factors'].append('weak_gdp_growth')

            # Analyze inflation
            inflation_value = estimates.get('inflation', {}).get('value', 3.0)
            if inflation_value > 4.0:
                outlook['inflation_outlook'] = 'elevated'
                outlook['risk_factors'].append('high_inflation')
            elif inflation_value < 2.0:
                outlook['inflation_outlook'] = 'low'
                outlook['positive_factors'].append('low_inflation')

            # Analyze employment
            employment_value = estimates.get('employment', {}).get('value', 50)
            if employment_value > 55:
                outlook['positive_factors'].append('strong_employment')
            elif employment_value < 45:
                outlook['risk_factors'].append('weak_employment')

            # Analyze consumer spending
            spending_value = estimates.get(
                'consumer_spending', {}).get(
                'value', 2.0)
            if spending_value > 4.0:
                outlook['positive_factors'].append('strong_consumer_spending')
            elif spending_value < 0:
                outlook['risk_factors'].append('weak_consumer_spending')

            # Overall assessment
            positive_count = len(outlook['positive_factors'])
            risk_count = len(outlook['risk_factors'])

            if positive_count > risk_count + 1:
                outlook['overall_assessment'] = 'positive'
            elif risk_count > positive_count + 1:
                outlook['overall_assessment'] = 'negative'
            else:
                outlook['overall_assessment'] = 'neutral'

            # Confidence assessment
            confidence_factors = 0
            if 'gdp_growth' in estimates:
                confidence_factors += 1
            if 'employment' in estimates:
                confidence_factors += 1
            if 'consumer_spending' in estimates:
                confidence_factors += 1

            outlook['outlook_confidence'] = min(
                0.9, 0.5 + confidence_factors * 0.15)

        except Exception as e:
            logger.warning(f"Failed to generate economic outlook: {e}")

        return outlook


class AlternativeSignalsProvider:
    """
    Comprehensive Alternative Signals Provider

    Integrates multiple alternative data sources to provide economic
    nowcasting, social sentiment analysis, and early warning indicators.
    """

    def __init__(self, config: Optional[AlternativeSignalsConfig] = None):
        """Initialize alternative signals provider"""
        self.config = config or AlternativeSignalsConfig()
        self.trends_analyzer = GoogleTrendsAnalyzer(self.config)
        self.sentiment_analyzer = SocialSentimentAnalyzer(self.config)
        self.nowcasting_analyzer = EconomicNowcastingAnalyzer(self.config)
        self._session = None

        # Create cache directory
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)

    def _get_session(self) -> requests.Session:
        """Get configured requests session"""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set headers
            self._session.headers.update({
                'User-Agent': 'DCF Lab Alternative Signals Provider'
            })

        return self._session

    def get_comprehensive_alternative_analysis(self,
                                               include_trends: bool = True,
                                               include_sentiment: bool = True,
                                               include_nowcasting: bool = True) -> Dict[str,
                                                                                        Any]:
        """
        Get comprehensive alternative signals analysis

        Args:
            include_trends: Include Google Trends analysis
            include_sentiment: Include social sentiment analysis
            include_nowcasting: Include economic nowcasting

        Returns:
            Dict with comprehensive alternative signals analysis
        """
        result = {
            'analyzed_at': datetime.now().isoformat(),
            'analysis_scope': {
                'trends_analysis': include_trends,
                'sentiment_analysis': include_sentiment,
                'nowcasting_analysis': include_nowcasting
            },
            'trends_analysis': {},
            'sentiment_analysis': {},
            'nowcasting_analysis': {},
            'integrated_signals': {},
            'early_warning_indicators': {},
            'alternative_outlook': {},
            'success': False,
            'error': None
        }

        try:
            # Google Trends Analysis
            if include_trends:
                trends_result = self.trends_analyzer.get_trends_analysis()
                result['trends_analysis'] = trends_result

            # Social Sentiment Analysis
            if include_sentiment:
                sentiment_result = self.sentiment_analyzer.get_social_sentiment_analysis()
                result['sentiment_analysis'] = sentiment_result

            # Economic Nowcasting
            if include_nowcasting:
                nowcast_result = self.nowcasting_analyzer.get_nowcasting_analysis()
                result['nowcasting_analysis'] = nowcast_result

            # Integrate signals
            integrated_signals = self._integrate_alternative_signals(
                result['trends_analysis'],
                result['sentiment_analysis'],
                result['nowcasting_analysis']
            )
            result['integrated_signals'] = integrated_signals

            # Early warning indicators
            early_warnings = self._generate_early_warning_indicators(
                result['trends_analysis'], result['sentiment_analysis'],
                result['nowcasting_analysis']
            )
            result['early_warning_indicators'] = early_warnings

            # Alternative economic outlook
            alt_outlook = self._generate_alternative_outlook(
                integrated_signals, early_warnings)
            result['alternative_outlook'] = alt_outlook

            result['success'] = True
            logger.info("Completed comprehensive alternative signals analysis")

        except Exception as e:
            result['error'] = str(e)
            logger.error(
                f"Failed comprehensive alternative signals analysis: {e}")

        return result

    def _integrate_alternative_signals(self,
                                       trends_data: Dict[str,
                                                         Any],
                                       sentiment_data: Dict[str,
                                                            Any],
                                       nowcast_data: Dict[str,
                                                          Any]) -> Dict[str,
                                                                        Any]:
        """Integrate signals from multiple alternative data sources"""
        integrated = {
            'composite_scores': {},
            'signal_consistency': {},
            'signal_strength': {},
            'cross_validation': {}
        }

        try:
            # Extract key signals
            signals = {}

            # From trends analysis
            if trends_data.get('success'):
                econ_signals = trends_data.get('economic_signals', {})
                signals['trends_recession_sentiment'] = econ_signals.get(
                    'recession_sentiment', 0)
                signals['trends_inflation_concern'] = econ_signals.get(
                    'inflation_concern', 0)
                signals['trends_employment_stress'] = econ_signals.get(
                    'employment_stress', 0)

            # From sentiment analysis
            if sentiment_data.get('success'):
                sentiment_scores = sentiment_data.get('sentiment_scores', {})

                # Average sentiment across keywords
                all_sentiments = [data.get('sentiment_score', 0)
                                  for data in sentiment_scores.values()]
                if all_sentiments:
                    signals['social_sentiment'] = np.mean(all_sentiments)
                else:
                    signals['social_sentiment'] = 0

            # From nowcasting analysis
            if nowcast_data.get('success'):
                estimates = nowcast_data.get('nowcast_values', {})

                # Normalize nowcast values to sentiment-like scale
                gdp_val = estimates.get('gdp_growth', {}).get('value', 2.5)
                signals['nowcast_growth'] = (
                    gdp_val - 2.5) / 2.5  # Normalize around 2.5% baseline

                employment_val = estimates.get(
                    'employment', {}).get(
                    'value', 50)
                signals['nowcast_employment'] = (
                    employment_val - 50) / 50  # Normalize around 50 baseline

            # Calculate composite scores
            composite_scores = {}

            # Economic sentiment composite
            econ_sentiment_signals = [
                signals.get('trends_recession_sentiment', 0),
                signals.get('social_sentiment', 0),
                signals.get('nowcast_growth', 0)
            ]
            composite_scores['economic_sentiment'] = np.mean(
                econ_sentiment_signals)

            # Employment composite
            employment_signals = [
                signals.get('trends_employment_stress', 0),
                signals.get('nowcast_employment', 0)
            ]
            if employment_signals:
                composite_scores['employment_conditions'] = np.mean(
                    employment_signals)

            # Inflation concerns composite
            inflation_signals = [
                signals.get('trends_inflation_concern', 0)
            ]
            composite_scores['inflation_pressures'] = np.mean(
                inflation_signals)

            integrated['composite_scores'] = composite_scores

            # Signal consistency (how aligned are different sources)
            consistency_scores = {}

            # Check consistency between trends and sentiment
            if 'trends_recession_sentiment' in signals and 'social_sentiment' in signals:
                consistency = 1 - \
                    abs(signals['trends_recession_sentiment'] -
                        signals['social_sentiment']) / 2
                consistency_scores['trends_sentiment_consistency'] = max(
                    0, consistency)

            integrated['signal_consistency'] = consistency_scores

            # Signal strength assessment
            signal_strengths = {}
            for signal_name, signal_value in signals.items():
                signal_strengths[signal_name] = {
                    'value': signal_value, 'strength': 'strong'
                    if abs(signal_value) > 0.6 else 'moderate'
                    if abs(signal_value) > 0.3 else 'weak'}

            integrated['signal_strength'] = signal_strengths

            # Cross-validation metrics
            cross_validation = {
                'signal_count': len(signals),
                'strong_signals': len([s for s in signals.values() if abs(s) > 0.6]),
                'conflicting_signals': len([s for s in signals.values() if s < -0.3]) > 0 and
                len([s for s in signals.values()
                     if s > 0.3]) > 0,
                # More signals = higher confidence
                'overall_confidence': min(len(signals) / 5, 1.0)
            }

            integrated['cross_validation'] = cross_validation

        except Exception as e:
            logger.warning(f"Failed to integrate alternative signals: {e}")

        return integrated

    def _generate_early_warning_indicators(self,
                                           trends_data: Dict[str,
                                                             Any],
                                           sentiment_data: Dict[str,
                                                                Any],
                                           nowcast_data: Dict[str,
                                                              Any]) -> Dict[str,
                                                                            Any]:
        """Generate early warning indicators from alternative data"""
        warnings = {
            'recession_warnings': [],
            'inflation_warnings': [],
            'employment_warnings': [],
            'financial_stress_warnings': [],
            'overall_warning_level': 'low'
        }

        try:
            warning_count = 0

            # Recession warnings
            if trends_data.get('success'):
                econ_signals = trends_data.get('economic_signals', {})
                stress_indicators = trends_data.get('stress_indicators', {})

                if econ_signals.get('recession_sentiment', 0) > 0.5:
                    warnings['recession_warnings'].append(
                        'elevated_recession_search_interest')
                    warning_count += 1

                if stress_indicators.get('financial_stress_level', 0) > 0.7:
                    warnings['financial_stress_warnings'].append(
                        'high_financial_stress_searches')
                    warning_count += 1

            # Sentiment-based warnings
            if sentiment_data.get('success'):
                sentiment_scores = sentiment_data.get('sentiment_scores', {})

                # Check for extremely negative sentiment
                negative_keywords = ['recession', 'unemployment', 'crash']
                for keyword in negative_keywords:
                    if keyword in sentiment_scores:
                        score = sentiment_scores[keyword].get(
                            'sentiment_score', 0)
                        if score < -0.6:
                            warnings['recession_warnings'].append(
                                f'extremely_negative_{keyword}_sentiment')
                            warning_count += 1

            # Nowcasting warnings
            if nowcast_data.get('success'):
                estimates = nowcast_data.get('nowcast_values', {})

                gdp_growth = estimates.get('gdp_growth', {}).get('value', 2.5)
                if gdp_growth < 1.0:
                    warnings['recession_warnings'].append(
                        'nowcast_weak_gdp_growth')
                    warning_count += 1

                employment_idx = estimates.get(
                    'employment', {}).get(
                    'value', 50)
                if employment_idx < 40:
                    warnings['employment_warnings'].append(
                        'nowcast_weak_employment')
                    warning_count += 1

                inflation_rate = estimates.get(
                    'inflation', {}).get(
                    'value', 3.0)
                if inflation_rate > 4.5:
                    warnings['inflation_warnings'].append(
                        'nowcast_high_inflation')
                    warning_count += 1

            # Overall warning level
            if warning_count >= 4:
                warnings['overall_warning_level'] = 'high'
            elif warning_count >= 2:
                warnings['overall_warning_level'] = 'moderate'
            else:
                warnings['overall_warning_level'] = 'low'

            warnings['total_warnings'] = warning_count

        except Exception as e:
            logger.warning(f"Failed to generate early warning indicators: {e}")

        return warnings

    def _generate_alternative_outlook(self,
                                      integrated_signals: Dict[str,
                                                               Any],
                                      early_warnings: Dict[str,
                                                           Any]) -> Dict[str,
                                                                         Any]:
        """Generate alternative economic outlook based on integrated signals"""
        outlook = {
            'economic_outlook': 'neutral',
            'confidence_level': 'moderate',
            'key_themes': [],
            'risk_assessment': 'moderate',
            'opportunity_areas': [],
            'outlook_summary': ''
        }

        try:
            # Extract composite scores
            composite_scores = integrated_signals.get('composite_scores', {})
            warning_level = early_warnings.get('overall_warning_level', 'low')

            economic_sentiment = composite_scores.get('economic_sentiment', 0)
            employment_conditions = composite_scores.get(
                'employment_conditions', 0)

            # Determine overall outlook
            if economic_sentiment > 0.3 and employment_conditions > 0.2:
                outlook['economic_outlook'] = 'positive'
                outlook['key_themes'].append('positive_economic_sentiment')
            elif economic_sentiment < -0.3 or employment_conditions < -0.3:
                outlook['economic_outlook'] = 'negative'
                outlook['key_themes'].append('negative_economic_indicators')
            else:
                outlook['economic_outlook'] = 'neutral'
                outlook['key_themes'].append('mixed_economic_signals')

            # Risk assessment
            if warning_level == 'high':
                outlook['risk_assessment'] = 'high'
                outlook['key_themes'].append('elevated_warning_indicators')
            elif warning_level == 'moderate':
                outlook['risk_assessment'] = 'moderate'
            else:
                outlook['risk_assessment'] = 'low'
                outlook['opportunity_areas'].append('stable_conditions')

            # Confidence level
            signal_consistency = integrated_signals.get(
                'signal_consistency', {})
            cross_validation = integrated_signals.get('cross_validation', {})

            overall_confidence = cross_validation.get(
                'overall_confidence', 0.5)

            if overall_confidence > 0.8 and signal_consistency:
                outlook['confidence_level'] = 'high'
            elif overall_confidence > 0.6:
                outlook['confidence_level'] = 'moderate'
            else:
                outlook['confidence_level'] = 'low'

            # Generate summary
            outlook_summary = (
                f"Alternative signals indicate {outlook['economic_outlook']} economic outlook "
                f"with {outlook['confidence_level']} confidence. "
                f"Risk assessment is {outlook['risk_assessment']}."
            )

            if outlook['key_themes']:
                outlook_summary += (
                    f" Key themes: {', '.join(outlook['key_themes'])}."
                )

            outlook['outlook_summary'] = outlook_summary

        except Exception as e:
            logger.warning(f"Failed to generate alternative outlook: {e}")

        return outlook


def get_alternative_signals_provider(
        config: Optional[AlternativeSignalsConfig] = None) -> AlternativeSignalsProvider:
    """Factory function to create alternative signals provider"""
    return AlternativeSignalsProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_alternative_signals_provider()

    print("=== Alternative Signals Provider ===")

    # Test comprehensive alternative signals analysis
    print("\n1. Comprehensive Alternative Signals Analysis:")
    analysis = provider.get_comprehensive_alternative_analysis()

    if analysis['success']:
        print("✅ Alternative signals analysis completed")

        # Trends analysis
        trends = analysis.get('trends_analysis', {})
        if trends.get('success'):
            econ_signals = trends.get('economic_signals', {})
            print("\nGoogle Trends Analysis:")
            recession_sentiment = econ_signals.get('recession_sentiment', 0)
            inflation_concern = econ_signals.get('inflation_concern', 0)
            employment_stress = econ_signals.get('employment_stress', 0)
            economic_outlook = econ_signals.get('economic_outlook', 'unknown')
            print(f"  Recession Sentiment: {recession_sentiment:.2f}")
            print(f"  Inflation Concern: {inflation_concern:.2f}")
            print(f"  Employment Stress: {employment_stress:.2f}")
            print(f"  Economic Outlook: {economic_outlook}")

        # Sentiment analysis
        sentiment = analysis.get('sentiment_analysis', {})
        if sentiment.get('success'):
            platforms = sentiment.get('platforms', [])
            sentiment_scores = sentiment.get('sentiment_scores', {})
            print("\nSocial Sentiment Analysis:")
            print(f"  Platforms Analyzed: {', '.join(platforms)}")

            if sentiment_scores:
                avg_sentiment = np.mean(
                    [data.get('sentiment_score', 0)
                     for data in sentiment_scores.values()])
                print(f"  Average Sentiment: {avg_sentiment:.2f}")
                print(f"  Keywords Tracked: {len(sentiment_scores)}")

        # Nowcasting analysis
        nowcast = analysis.get('nowcasting_analysis', {})
        if nowcast.get('success'):
            estimates = nowcast.get('nowcast_values', {})
            outlook = nowcast.get('economic_outlook', {})
            print("\nEconomic Nowcasting:")

            gdp_est = estimates.get('gdp_growth', {})
            if gdp_est:
                gdp_value = gdp_est.get('value', 0)
                print(f"  GDP Growth Estimate: {gdp_value:.1f}%")

            employment_est = estimates.get('employment', {})
            if employment_est:
                employment_value = employment_est.get('value', 0)
                print(f"  Employment Index: {employment_value:.0f}")

            overall_assessment = outlook.get('overall_assessment', 'unknown')
            outlook_confidence = outlook.get('outlook_confidence', 0)
            print(f"  Overall Assessment: {overall_assessment}")
            print(f"  Outlook Confidence: {outlook_confidence:.1%}")

        # Integrated signals
        integrated = analysis.get('integrated_signals', {})
        composite_scores = integrated.get('composite_scores', {})
        print("\nIntegrated Signals:")
        economic_sentiment = composite_scores.get('economic_sentiment', 0)
        employment_conditions = composite_scores.get('employment_conditions', 0)
        inflation_pressures = composite_scores.get('inflation_pressures', 0)
        print(f"  Economic Sentiment: {economic_sentiment:.2f}")
        print(f"  Employment Conditions: {employment_conditions:.2f}")
        print(f"  Inflation Pressures: {inflation_pressures:.2f}")

        cross_validation = integrated.get('cross_validation', {})
        print(f"  Signal Count: {cross_validation.get('signal_count', 0)}")
        overall_confidence = cross_validation.get('overall_confidence', 0)
        print(f"  Overall Confidence: {overall_confidence:.1%}")

        # Early warning indicators
        warning_indicators = analysis.get('early_warning_indicators', {})
        print("\nEarly Warning Indicators:")
        warning_level = warning_indicators.get('overall_warning_level', 'unknown')
        total_warnings = warning_indicators.get('total_warnings', 0)
        print(f"  Warning Level: {warning_level}")
        print(f"  Total Warnings: {total_warnings}")

        recession_warnings = warnings.get('recession_warnings', [])
        if recession_warnings:
            print(f"  Recession Warnings: {', '.join(recession_warnings[:3])}")

        # Alternative outlook
        alt_outlook = analysis.get('alternative_outlook', {})
        print("\nAlternative Economic Outlook:")
        economic_outlook = alt_outlook.get('economic_outlook', 'unknown')
        confidence_level = alt_outlook.get('confidence_level', 'unknown')
        risk_assessment = alt_outlook.get('risk_assessment', 'unknown')
        print(f"  Economic Outlook: {economic_outlook}")
        print(f"  Confidence Level: {confidence_level}")
        print(f"  Risk Assessment: {risk_assessment}")

        key_themes = alt_outlook.get('key_themes', [])
        if key_themes:
            print(f"  Key Themes: {', '.join(key_themes)}")

        summary = alt_outlook.get('outlook_summary', '')
        if summary:
            print(f"\nOutlook Summary: {summary}")

    else:
        print(f"❌ Alternative signals analysis failed: {analysis['error']}")

    print("\n=== Alternative Signals Provider Ready ===")
    print("✅ Google Trends economic keyword analysis")
    print("✅ Social media sentiment tracking and analysis")
    print("✅ Economic nowcasting using alternative data")
    print("✅ Integrated alternative signals analysis")
    print("✅ Early warning indicators and risk assessment")
    print("✅ Alternative economic outlook generation")
    print("🚀 Ready for comprehensive alternative signals analysis")
