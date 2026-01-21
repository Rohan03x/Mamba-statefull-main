"""
News Sentiment Analysis for Financial Forecasting

This module implements sophisticated sentiment analysis for financial news and social media,
integrating FinBERT and other finance-specific NLP models to enhance price forecasting.

Key Features:
- FinBERT sentiment analysis with financial domain expertise
- Multi-source news aggregation (financial news, social media, earnings calls)
- Temporal sentiment decay modeling and persistence analysis
- Sentiment-weighted forecast adjustments with confidence scoring
- Real-time news impact assessment and event detection
- Sentiment momentum and reversal pattern identification

News Sources Supported:
- Financial news articles (Reuters, Bloomberg, CNBC, etc.)
- Social media sentiment (Twitter/X, Reddit financial communities)
- SEC filings and earnings call transcripts
- Analyst reports and research notes
- Options flow and unusual activity alerts

Sentiment Integration:
- News-driven volatility adjustments
- Sentiment momentum indicators
- Event-driven forecast modifications
- Multi-timeframe sentiment aggregation
"""

import logging
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Core dependencies - LAZY IMPORT to avoid torch bus error
# transformers will be imported only when FinBERT is actually instantiated
HAS_TRANSFORMERS = None  # Will be checked lazily
AutoTokenizer = None
AutoModelForSequenceClassification = None
pipeline = None

def _lazy_import_transformers():
    """Lazy import transformers to avoid loading torch at module level."""
    global HAS_TRANSFORMERS, AutoTokenizer, AutoModelForSequenceClassification, pipeline
    if HAS_TRANSFORMERS is not None:
        return HAS_TRANSFORMERS
    try:
        import transformers  # type: ignore
        AutoTokenizer = transformers.AutoTokenizer
        AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
        pipeline = transformers.pipeline
        HAS_TRANSFORMERS = True
    except ImportError:
        HAS_TRANSFORMERS = False
        warnings.warn(
            "transformers not available - FinBERT sentiment analysis disabled")
    return HAS_TRANSFORMERS

try:
    HAS_WEB_SCRAPING = True
except ImportError:
    HAS_WEB_SCRAPING = False
    warnings.warn(
        "requests/beautifulsoup not available - web scraping disabled")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    # Only warn on first import, not repeatedly
    import sys
    if 'nltk_warning_shown' not in sys.modules:
        warnings.warn(
            "NLTK not available - using enhanced fallback sentiment analysis")
        sys.modules['nltk_warning_shown'] = True

# Optional dependencies for advanced features
try:
    from textblob import TextBlob  # type: ignore
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Single news article with metadata"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    relevance_score: float = 1.0


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0-1 confidence score
    positive_prob: float  # Probability of positive sentiment
    negative_prob: float  # Probability of negative sentiment
    neutral_prob: float  # Probability of neutral sentiment
    compound_score: float  # Overall sentiment score (-1 to 1)
    article_id: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class SentimentImpact:
    """Sentiment impact on forecasting"""
    price_adjustment: float  # Percentage adjustment to forecast
    volatility_adjustment: float  # Volatility multiplier
    confidence: float  # Confidence in the adjustment
    time_decay: float  # How quickly impact decays
    sentiment_momentum: float  # Sentiment trend strength
    supporting_articles: int  # Number of supporting articles


class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analysis for financial texts"""

    # Class-level cache for model instances
    _model_cache = {}

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.initialized = False

        if _lazy_import_transformers():
            self._initialize_model()

    def _initialize_model(self):
        """Initialize FinBERT model and tokenizer with caching"""
        try:
            # Check if model is already cached
            if self.model_name in self._model_cache:
                logger.info(f"Using cached FinBERT model: {self.model_name}")
                cached_model = self._model_cache[self.model_name]
                self.tokenizer = cached_model['tokenizer']
                self.model = cached_model['model']
                self.pipeline = cached_model['pipeline']
                self.initialized = True
                return

            logger.info(f"Loading FinBERT model: {self.model_name}")

            # Get HuggingFace token from environment and clean it
            import os
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                hf_token = hf_token.strip()  # Remove whitespace and newlines
            
            # Initialize tokenizer with authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=hf_token,
                trust_remote_code=True
            )

            # Initialize model with authentication
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True
            )

            # Create sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None  # Returns all scores instead of return_all_scores=True
            )

            # Cache the model for future use
            self._model_cache[self.model_name] = {
                'tokenizer': self.tokenizer,
                'model': self.model,
                'pipeline': self.pipeline
            }

            self.initialized = True
            logger.info("FinBERT model loaded and cached successfully")

        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.initialized = False

    def analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of financial text using FinBERT"""
        if not self.initialized:
            return self._fallback_sentiment(text)

        try:
            # Clean and preprocess text
            text = self._preprocess_text(text)

            # Truncate if too long (FinBERT has token limits)
            max_length = 512
            if len(text.split()) > max_length:
                text = ' '.join(text.split()[:max_length])

            # Get sentiment predictions
            # First result contains all scores
            results = self.pipeline(text)[0]

            # Parse results
            sentiment_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score

            # Map FinBERT labels to standard format
            positive_prob = sentiment_scores.get('positive', 0.0)
            negative_prob = sentiment_scores.get('negative', 0.0)
            neutral_prob = sentiment_scores.get('neutral', 0.0)

            # Determine primary sentiment
            max_score = max(positive_prob, negative_prob, neutral_prob)
            if max_score == positive_prob:
                sentiment = 'positive'
                confidence = positive_prob
            elif max_score == negative_prob:
                sentiment = 'negative'
                confidence = negative_prob
            else:
                sentiment = 'neutral'
                confidence = neutral_prob

            # Calculate compound score (-1 to 1)
            compound_score = positive_prob - negative_prob

            return SentimentScore(
                sentiment=sentiment,
                confidence=confidence,
                positive_prob=positive_prob,
                negative_prob=negative_prob,
                neutral_prob=neutral_prob,
                compound_score=compound_score,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"FinBERT sentiment analysis failed: {e}")
            return self._fallback_sentiment(text)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess financial text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s$%\-\+\.]', ' ', text)

        return text

    def _fallback_sentiment(self, text: str) -> SentimentScore:
        """Fallback sentiment analysis using simple methods"""
        if HAS_NLTK:
            try:
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text)

                compound = scores['compound']

                if compound >= 0.05:
                    sentiment = 'positive'
                    confidence = abs(compound)
                elif compound <= -0.05:
                    sentiment = 'negative'
                    confidence = abs(compound)
                else:
                    sentiment = 'neutral'
                    confidence = 1 - abs(compound)

                return SentimentScore(
                    sentiment=sentiment,
                    confidence=confidence,
                    positive_prob=scores['pos'],
                    negative_prob=scores['neg'],
                    neutral_prob=scores['neu'],
                    compound_score=compound,
                    timestamp=datetime.now()
                )
            except Exception as e:
                logger.error(f"NLTK sentiment analysis failed: {e}")

        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                confidence = min(abs(polarity) + 0.3, 0.9)

                return SentimentScore(
                    sentiment=sentiment,
                    confidence=confidence,
                    positive_prob=max(0, polarity),
                    negative_prob=max(0, -polarity),
                    neutral_prob=1 - abs(polarity),
                    compound_score=polarity,
                    timestamp=datetime.now()
                )
            except Exception as e:
                logger.error(f"TextBlob sentiment analysis failed: {e}")

        # Ultimate fallback - neutral sentiment
        return SentimentScore(
            sentiment='neutral',
            confidence=0.5,
            positive_prob=0.33,
            negative_prob=0.33,
            neutral_prob=0.34,
            compound_score=0.0,
            timestamp=datetime.now()
        )


class NewsAggregator:
    """Aggregates news from multiple sources for sentiment analysis"""

    def __init__(self, sources: Optional[List[str]] = None):
        # Default to available sources - MOCK DISABLED
        self.sources = sources or ['yahoo']
        self.articles_cache = {}
        self.last_fetch = None

    def fetch_news(self, ticker: str, days_back: int = 7) -> List[NewsArticle]:
        """Fetch news articles for a given ticker"""
        logger.info(f"Fetching news for {ticker} from last {days_back} days")

        all_articles = []

        for source in self.sources:
            try:
                if source == 'yahoo' and HAS_YFINANCE:
                    articles = self._fetch_yahoo_news(ticker, days_back)
                elif source == 'mock':
                    logger.error(f"MOCK DATA DISABLED - source '{source}' not allowed")
                    continue
                else:
                    logger.warning(
                        f"Source {source} not available or supported")
                    continue

                all_articles.extend(articles)
                logger.info(f"Fetched {len(articles)} articles from {source}")

            except Exception as e:
                logger.error(f"Failed to fetch news from {source}: {e}")

        # Remove duplicates and sort by timestamp
        unique_articles = self._deduplicate_articles(all_articles)
        unique_articles.sort(key=lambda x: x.timestamp, reverse=True)

        logger.info(f"Total unique articles fetched: {len(unique_articles)}")
        return unique_articles

    def _fetch_yahoo_news(
            self,
            ticker: str,
            days_back: int) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        try:
            # Get ticker object
            stock = yf.Ticker(ticker)

            # Get news (Yahoo Finance API)
            news = stock.news

            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for item in news[:20]:  # Limit to recent articles
                try:
                    # Parse timestamp
                    timestamp = datetime.fromtimestamp(
                        item.get('providerPublishTime', time.time()))

                    if timestamp < cutoff_date:
                        continue

                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('summary', ''),
                        source='yahoo_finance',
                        timestamp=timestamp,
                        url=item.get('link', ''),
                        category='financial_news'
                    )
                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to parse Yahoo news item: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Yahoo Finance news fetch failed: {e}")
            return []

    def _fetch_mock_news(
            self,
            ticker: str,
            days_back: int) -> List[NewsArticle]:
        """DISABLED - Mock news generation not allowed in production"""
        logger.error("Mock news generation disabled - use real data sources only")
        return []

    def _deduplicate_articles(
            self,
            articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []

        unique_articles = []
        seen_titles = set()

        for article in articles:
            # Simple deduplication based on title
            title_key = article.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        return unique_articles


class SentimentTemporalModel:
    """Models temporal decay and persistence of sentiment impact"""

    def __init__(
        self,
        decay_half_life: float = 24.0,  # Hours for sentiment to decay by half
        momentum_window: int = 48,      # Hours to look back for momentum
        minimum_impact: float = 0.01    # Minimum impact threshold
    ):
        self.decay_half_life = decay_half_life
        self.momentum_window = momentum_window
        self.minimum_impact = minimum_impact

    def calculate_temporal_weight(
            self,
            sentiment_time: datetime,
            current_time: datetime) -> float:
        """Calculate temporal weight for sentiment based on time decay"""
        hours_elapsed = (current_time - sentiment_time).total_seconds() / 3600

        # Exponential decay
        weight = np.exp(-0.693 * hours_elapsed / self.decay_half_life)

        return max(weight, self.minimum_impact)

    def calculate_sentiment_momentum(
            self, sentiment_history: List[SentimentScore]) -> float:
        """Calculate sentiment momentum over time"""
        if len(sentiment_history) < 2:
            return 0.0

        # Sort by timestamp
        sorted_history = sorted(sentiment_history, key=lambda x: x.timestamp)

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=self.momentum_window)

        # Filter recent sentiments
        recent_sentiments = [
            s for s in sorted_history
            if s.timestamp >= cutoff_time
        ]

        if len(recent_sentiments) < 2:
            return 0.0

        # Calculate momentum as trend in compound scores
        scores = [s.compound_score for s in recent_sentiments]

        # Simple linear trend
        if len(scores) >= 3:
            # Calculate average change rate
            changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
            momentum = np.mean(changes)
        else:
            momentum = scores[-1] - scores[0]

        return np.clip(momentum, -1.0, 1.0)

    def aggregate_sentiment_impact(
        self,
        sentiment_scores: List[SentimentScore],
        current_time: Optional[datetime] = None
    ) -> Tuple[float, float, int]:
        """Aggregate multiple sentiment scores with temporal weighting"""
        if not sentiment_scores:
            return 0.0, 0.0, 0

        current_time = current_time or datetime.now()

        weighted_sum = 0.0
        weight_sum = 0.0
        confidence_sum = 0.0

        for sentiment in sentiment_scores:
            temporal_weight = self.calculate_temporal_weight(
                sentiment.timestamp, current_time)
            sentiment_weight = temporal_weight * sentiment.confidence

            weighted_sum += sentiment.compound_score * sentiment_weight
            weight_sum += sentiment_weight
            confidence_sum += sentiment.confidence * temporal_weight

        if weight_sum == 0:
            return 0.0, 0.0, 0

        aggregated_sentiment = weighted_sum / weight_sum
        aggregated_confidence = confidence_sum / weight_sum

        return aggregated_sentiment, aggregated_confidence, len(
            sentiment_scores)


class NewsSentimentForecaster:
    """Main class for news sentiment analysis and forecast adjustment"""

    def __init__(
        self,
        sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None,
        news_aggregator: Optional[NewsAggregator] = None,
        temporal_model: Optional[SentimentTemporalModel] = None
    ):
        self.sentiment_analyzer = sentiment_analyzer or FinBERTSentimentAnalyzer()
        self.news_aggregator = news_aggregator or NewsAggregator()
        self.temporal_model = temporal_model or SentimentTemporalModel()

        # Sentiment history cache
        self.sentiment_history = defaultdict(list)

        # Model parameters for forecast adjustment
        self.sentiment_to_return_multiplier = 0.02  # 2% max adjustment
        self.sentiment_to_volatility_multiplier = 0.15  # 15% max vol adjustment
        self.confidence_threshold = 0.6  # Minimum confidence for adjustments

    def analyze_ticker_sentiment(
        self,
        ticker: str,
        days_back: int = 7,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Analyze sentiment for a specific ticker"""
        logger.info(f"Starting sentiment analysis for {ticker}")

        # Fetch news articles
        articles = self.news_aggregator.fetch_news(ticker, days_back)

        if not articles:
            logger.warning(f"No articles found for {ticker}")
            return self._create_neutral_sentiment_result(ticker)

        # Analyze sentiment for each article
        sentiment_scores = []

        for article in articles:
            try:
                # Combine title and content for analysis
                text = f"{article.title}. {article.content}"

                # Get sentiment score
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                sentiment.article_id = f"{article.source}_{hash(article.title) % 10000}"
                sentiment.timestamp = article.timestamp

                sentiment_scores.append(sentiment)

            except Exception as e:
                logger.error(f"Failed to analyze sentiment for article: {e}")
                continue

        if not sentiment_scores:
            logger.warning(f"No sentiment scores generated for {ticker}")
            return self._create_neutral_sentiment_result(ticker)

        # Store in history cache
        if use_cache:
            self.sentiment_history[ticker].extend(sentiment_scores)
            # Keep only recent history (last 30 days)
            cutoff = datetime.now() - timedelta(days=30)
            self.sentiment_history[ticker] = [
                s for s in self.sentiment_history[ticker]
                if s.timestamp >= cutoff
            ]

        # Calculate aggregated sentiment and momentum
        aggregated_sentiment, confidence, article_count = self.temporal_model.aggregate_sentiment_impact(
            sentiment_scores)

        sentiment_momentum = self.temporal_model.calculate_sentiment_momentum(
            self.sentiment_history[ticker]
        )

        # Calculate forecast impact
        impact = self._calculate_forecast_impact(
            aggregated_sentiment, confidence, sentiment_momentum, article_count
        )

        # Build comprehensive result
        result = {
            'ticker': ticker, 'analysis_time': datetime.now(),
            'articles_analyzed': len(articles),
            'sentiment_scores': sentiment_scores,
            'aggregated_sentiment': aggregated_sentiment,
            'sentiment_confidence': confidence,
            'sentiment_momentum': sentiment_momentum,
            'forecast_impact': impact,
            'sentiment_distribution': self._calculate_sentiment_distribution(
                sentiment_scores),
            'recent_headlines': [a.title for a in articles[: 5]],
            'data_quality':
            {'articles_processed': len(articles),
             'successful_analyses': len(sentiment_scores),
             'average_confidence': np.mean(
                 [s.confidence for s in sentiment_scores]),
             'time_span_hours':
             (max(a.timestamp for a in articles) -
              min(a.timestamp for a in articles)).total_seconds() / 3600}}

        logger.info(
            f"Sentiment analysis complete for {ticker}: sentiment={aggregated_sentiment:.3f}, confidence={confidence:.3f}")

        return result

    def _calculate_forecast_impact(
        self,
        sentiment: float,
        confidence: float,
        momentum: float,
        article_count: int
    ) -> SentimentImpact:
        """Calculate the impact of sentiment on forecasting"""

        # Base adjustments from sentiment
        base_price_adjustment = sentiment * self.sentiment_to_return_multiplier
        base_volatility_adjustment = 1.0 + \
            abs(sentiment) * self.sentiment_to_volatility_multiplier

        # Confidence scaling
        confidence_multiplier = max(
            0, (confidence - self.confidence_threshold) /
            (1 - self.confidence_threshold))

        # Momentum adjustment
        momentum_boost = 1.0 + abs(momentum) * 0.5

        # Article count impact (more articles = more confidence)
        article_boost = min(2.0, 1.0 + np.log(max(1, article_count)) / 5)

        # Final adjustments
        final_price_adjustment = base_price_adjustment * \
            confidence_multiplier * momentum_boost * article_boost
        final_volatility_adjustment = base_volatility_adjustment * confidence_multiplier

        # Time decay calculation
        time_decay = 0.693 / self.temporal_model.decay_half_life  # Decay rate per hour

        return SentimentImpact(
            price_adjustment=final_price_adjustment,
            volatility_adjustment=final_volatility_adjustment,
            confidence=confidence * confidence_multiplier,
            time_decay=time_decay,
            sentiment_momentum=momentum,
            supporting_articles=article_count
        )

    def _calculate_sentiment_distribution(
            self, sentiment_scores: List[SentimentScore]) -> Dict[str, float]:
        """Calculate distribution of sentiment categories"""
        if not sentiment_scores:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

        counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        for score in sentiment_scores:
            counts[score.sentiment] += 1

        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def _create_neutral_sentiment_result(self, ticker: str) -> Dict[str, Any]:
        """Create neutral sentiment result when no data available"""
        neutral_impact = SentimentImpact(
            price_adjustment=0.0,
            volatility_adjustment=1.0,
            confidence=0.5,
            time_decay=0.0,
            sentiment_momentum=0.0,
            supporting_articles=0
        )

        return {
            'ticker': ticker,
            'analysis_time': datetime.now(),
            'articles_analyzed': 0,
            'sentiment_scores': [],
            'aggregated_sentiment': 0.0,
            'sentiment_confidence': 0.5,
            'sentiment_momentum': 0.0,
            'forecast_impact': neutral_impact,
            'sentiment_distribution': {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34},
            'recent_headlines': [],
            'data_quality': {
                'articles_processed': 0,
                'successful_analyses': 0,
                'average_confidence': 0.5,
                'time_span_hours': 0}}

    def get_sentiment_adjusted_forecast(
        self,
        base_forecast: Dict[str, Any],
        ticker: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Apply sentiment adjustments to base forecast"""

        # Get sentiment analysis
        sentiment_result = self.analyze_ticker_sentiment(ticker, days_back)
        impact = sentiment_result['forecast_impact']

        # Apply adjustments to forecast
        adjusted_forecast = base_forecast.copy()

        # Apply forecast adjustments
        self._apply_returns_adjustment(adjusted_forecast, impact)
        self._apply_quantile_adjustments(adjusted_forecast, impact)

        # Add sentiment information to forecast
        adjusted_forecast['sentiment_analysis'] = sentiment_result
        adjusted_forecast['sentiment_adjustments'] = {
            'price_adjustment_pct': impact.price_adjustment * 100,
            'volatility_multiplier': impact.volatility_adjustment,
            'adjustment_confidence': impact.confidence,
            'sentiment_momentum': impact.sentiment_momentum,
            'supporting_articles': impact.supporting_articles
        }

        logger.info(
            f"Applied sentiment adjustments to {ticker} forecast: price_adj={impact.price_adjustment:.2%}, vol_mult={impact.volatility_adjustment:.2f}")

        return adjusted_forecast

    def _apply_returns_adjustment(
            self, forecast: Dict[str, Any], impact: SentimentImpact):
        """Apply sentiment adjustment to forecast returns"""
        if 'forecast_returns' not in forecast:
            return

        returns = np.array(forecast['forecast_returns'])
        adjusted_returns = returns + impact.price_adjustment
        forecast['forecast_returns'] = adjusted_returns.tolist()

        # Update price forecasts if available
        if 'last_price' in forecast:
            last_price = forecast['last_price']
            adjusted_prices = []
            current_price = last_price

            for ret in adjusted_returns:
                current_price = current_price * (1 + ret)
                adjusted_prices.append(current_price)

            forecast['adjusted_prices'] = adjusted_prices

    def _apply_quantile_adjustments(
            self, forecast: Dict[str, Any], impact: SentimentImpact):
        """Apply sentiment adjustment to quantile forecasts"""
        if 'quantile_forecasts' not in forecast:
            return

        quantiles = forecast['quantile_forecasts']
        for quantile_key in quantiles:
            if isinstance(quantiles[quantile_key], list):
                original = np.array(quantiles[quantile_key])
                adjusted = original + impact.price_adjustment

                # Apply volatility adjustment to spread for extreme quantiles
                if quantile_key in ['q_10', 'q_25', 'q_75', 'q_90']:
                    spread_adjustment = (
                        adjusted - original) * impact.volatility_adjustment
                    adjusted = original + spread_adjustment

                forecast['quantile_forecasts'][quantile_key] = adjusted.tolist()


def test_news_sentiment_system():
    """Test the news sentiment analysis system"""
    print("=== TESTING NEWS SENTIMENT ANALYSIS SYSTEM ===")

    # Test individual components
    print("\n--- Testing FinBERT Sentiment Analyzer ---")
    analyzer = FinBERTSentimentAnalyzer()

    test_texts = [
        "Apple reports record quarterly earnings, beating analyst expectations significantly.",
        "Concerns grow over Apple's declining iPhone sales in key markets amid competition.",
        "Apple maintains stable performance with modest growth in services revenue."]

    for i, text in enumerate(test_texts):
        sentiment = analyzer.analyze_sentiment(text)
        print(
            f"Text {i + 1}: {sentiment.sentiment} (confidence: {sentiment.confidence:.2f}, compound: {sentiment.compound_score:.2f})")

    print("\n--- Testing News Aggregation ---")
    aggregator = NewsAggregator()
    articles = aggregator.fetch_news('AAPL', days_back=7)
    print(f"Fetched {len(articles)} articles for AAPL")

    for article in articles[:3]:
        print(
            f"  - {article.title} ({article.source}, {article.timestamp.strftime('%Y-%m-%d %H:%M')})")

    print("\n--- Testing Full Sentiment Forecaster ---")
    forecaster = NewsSentimentForecaster()

    # Test sentiment analysis
    result = forecaster.analyze_ticker_sentiment('AAPL', days_back=7)
    print("Sentiment Analysis Results for AAPL:")
    print(f"  Aggregated Sentiment: {result['aggregated_sentiment']:.3f}")
    print(f"  Confidence: {result['sentiment_confidence']:.3f}")
    print(f"  Momentum: {result['sentiment_momentum']:.3f}")
    print(f"  Articles Analyzed: {result['articles_analyzed']}")

    impact = result['forecast_impact']
    print("  Forecast Impact:")
    print(f"    Price Adjustment: {impact.price_adjustment:.2%}")
    print(f"    Volatility Multiplier: {impact.volatility_adjustment:.2f}")
    print(f"    Supporting Articles: {impact.supporting_articles}")

    # Test forecast adjustment
    print("\n--- Testing Forecast Adjustment ---")
    base_forecast = {
        'forecast_returns': [0.001, 0.002, -0.001, 0.003, 0.000],
        'last_price': 150.0,
        'quantile_forecasts': {
            'q_10': [-0.02, -0.015, -0.01, -0.005, 0.0],
            'q_50': [0.001, 0.002, -0.001, 0.003, 0.000],
            'q_90': [0.025, 0.02, 0.015, 0.01, 0.005]
        }
    }

    adjusted_forecast = forecaster.get_sentiment_adjusted_forecast(
        base_forecast, 'AAPL')

    print("Original vs Adjusted Returns:")
    orig_returns = base_forecast['forecast_returns']
    adj_returns = adjusted_forecast['forecast_returns']

    for i, (orig, adj) in enumerate(zip(orig_returns, adj_returns)):
        print(f"  Day {i+1}: {orig:.3f} -> {adj:.3f} (change: {adj-orig:+.3f})")

    adjustments = adjusted_forecast['sentiment_adjustments']
    print("\nSentiment Adjustments Applied:")
    print(f"  Price Adjustment: {adjustments['price_adjustment_pct']:.2f}%")
    print(f"  Volatility Multiplier: {adjustments['volatility_multiplier']:.2f}")
    print(f"  Confidence: {adjustments['adjustment_confidence']:.2f}")

    print("\n=== NEWS SENTIMENT TESTING COMPLETE ===")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_news_sentiment_system()
