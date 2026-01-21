"""
News and Earnings NLP Analysis

This module implements NLP-based sentiment analysis for news articles and 
earnings transcripts to extract insights that can improve financial forecasts.
"""

import logging
import os
import re
import string
import warnings
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local dependencies
try:
    from ..utils import get_data_path
except ImportError:
    def get_data_path():
        """Fallback function to get data path"""
        return os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))),
            "data")


# Simple text processing utilities to replace NLTK
def simple_tokenize(text):
    """Simple word tokenization"""
    # Replace punctuation with spaces, then split on whitespace
    for char in string.punctuation:
        text = text.replace(char, ' ')
    return text.split()


def simple_sentence_tokenize(text):
    """Simple sentence tokenization"""
    # Split on common sentence endings
    sentences = []
    for potential_sentence in re.split(r'(?<=[.!?])\s+', text):
        if potential_sentence:  # Ignore empty strings
            sentences.append(potential_sentence)
    return sentences


# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'mysel', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yoursel', 'yourselves', 'he', 'him', 'his', 'himsel', 'she',
    'her', 'hers', 'hersel', 'it', 'its', 'itsel', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'i', 'or', 'because', 'as', 'until', 'while', 'o',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'of', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}


class NewsArticle:
    """Class representing a news article with metadata and content"""

    def __init__(self,
                 headline: str,
                 content: str,
                 date: Union[str, datetime],
                 source: str,
                 url: Optional[str] = None):
        """
        Initialize a news article

        Args:
            headline: Article headline
            content: Article content/body
            date: Publication date
            source: News source
            url: Optional URL to the article
        """
        self.headline = headline
        self.content = content
        self.source = source
        self.url = url

        # Parse date if needed
        if isinstance(date, str):
            try:
                self.date = datetime.fromisoformat(date)
            except ValueError:
                try:
                    # Try another common format
                    self.date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    warnings.warn(
                        f"Could not parse date: {date}. Using current date.")
                    self.date = datetime.now()
        else:
            self.date = date

        # Processed content
        self._clean_content = None
        self._tokens = None
        self._sentences = None

        # Analysis results
        self.sentiment = None
        self.topics = None
        self.entities = None
        self.summary = None

    def __repr__(self):
        date_str = self.date.strftime('%Y-%m-%d')
        return f"NewsArticle(headline='{self.headline}', date={date_str}, source='{self.source}')"

    def clean_content(self) -> str:
        """Clean and normalize the article content"""
        if self._clean_content is not None:
            return self._clean_content

        # Combine headline and content
        full_text = f"{self.headline}\n\n{self.content}"

        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', full_text)

        # Remove special characters and digits
        text = re.sub(r'[^\w\s.]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        self._clean_content = text
        return text

    def get_tokens(self) -> List[str]:
        """Tokenize article content into words"""
        if self._tokens is not None:
            return self._tokens

        text = self.clean_content().lower()

        # Tokenize using our simple tokenizer
        tokens = simple_tokenize(text)

        # Remove stopwords and punctuation
        tokens = [word for word in tokens
                  if word not in STOP_WORDS and word not in string.punctuation]

        self._tokens = tokens
        return tokens

    def get_sentences(self) -> List[str]:
        """Split article into sentences"""
        if self._sentences is not None:
            return self._sentences

        text = self.clean_content()
        self._sentences = simple_sentence_tokenize(text)
        return self._sentences


class FinbertSentimentAnalyzer:
    """
    Rule-based sentiment analyzer for financial text

    This class uses a dictionary-based approach to analyze sentiment
    in financial news and earnings transcripts.
    """

    def __init__(self):
        """Initialize the rule-based sentiment analyzer"""
        # Financial sentiment dictionaries
        self.positive_words = {
            'increase',
            'growth',
            'profit',
            'gain',
            'improve',
            'positive',
            'strong',
            'upside',
            'rise',
            'exceed',
            'beat',
            'above',
            'better',
            'higher',
            'up',
            'bullish',
            'successful',
            'opportunity',
            'outperform',
            'recovery',
            'surpass',
            'win',
            'robust',
            'confidence',
            'optimistic',
            'pleased',
            'excited',
            'happy',
            'record',
            'strengthen',
            'advantage',
            'excellent',
            'favorable',
            'impressive',
            'progress',
            'momentum',
            'accelerate'}

        self.negative_words = {
            'decrease',
            'decline',
            'loss',
            'lose',
            'negative',
            'weak',
            'downside',
            'fall',
            'miss',
            'below',
            'worse',
            'lower',
            'down',
            'bearish',
            'fail',
            'disadvantage',
            'challenge',
            'underperform',
            'recession',
            'drop',
            'disappoint',
            'risk',
            'concern',
            'uncertain',
            'worry',
            'problem',
            'threat',
            'difficult',
            'headwind',
            'struggle',
            'cut',
            'reduce',
            'eliminate',
            'unfavorable',
            'slowdown',
            'pressure',
            'adverse'}

        # Intensifiers affect sentiment strength
        self.intensifiers = {
            'very',
            'highly',
            'extremely',
            'significantly',
            'substantially',
            'considerably',
            'exceptionally',
            'remarkably',
            'notably',
            'greatly',
            'vastly',
            'hugely',
            'tremendously',
            'extraordinarily',
            'immensely',
            'especially',
            'particularly'}

        # Negations reverse sentiment
        self.negations = {
            'not',
            'no',
            'never',
            'neither',
            'nor',
            'none',
            'nobody',
            'nothing',
            'nowhere',
            'hardly',
            'barely',
            'scarcely',
            'isn\'t',
            'wasn\'t',
            'weren\'t',
            'haven\'t',
            'hasn\'t',
            'hadn\'t',
            'won\'t',
            'wouldn\'t',
            'don\'t',
            'doesn\'t',
            'didn\'t',
            'can\'t',
            'couldn\'t',
            'shouldn\'t',
            'mightn\'t',
            'without'}

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using rule-based approach

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        # Split text into chunks (simplified sentences)
        chunks = simple_sentence_tokenize(text.lower())

        # Process each chunk for sentiment
        positive_score = 0
        negative_score = 0
        total_chunks = len(chunks)

        for chunk in chunks:
            # Check for negations in this chunk
            has_negation = any(neg in chunk for neg in self.negations)

            # Check for intensifiers
            intensifier_count = sum(
                1 for intensifier in self.intensifiers if intensifier in chunk)
            # Increase intensity by 20% per intensifier
            intensity_factor = 1.0 + (0.2 * intensifier_count)

            # Count positive and negative words
            chunk_tokens = simple_tokenize(chunk)

            pos_in_chunk = sum(
                1 for token in chunk_tokens if token in self.positive_words)
            neg_in_chunk = sum(
                1 for token in chunk_tokens if token in self.negative_words)

            # Apply negation (reverse sentiment)
            if has_negation:
                # Swap positive and negative counts
                pos_in_chunk, neg_in_chunk = neg_in_chunk, pos_in_chunk

            # Apply intensity
            pos_in_chunk *= intensity_factor
            neg_in_chunk *= intensity_factor

            # Add to total scores
            positive_score += pos_in_chunk
            negative_score += neg_in_chunk

        # Normalize scores
        if total_chunks > 0:
            positive_score /= total_chunks
            negative_score /= total_chunks

        # Calculate neutral and compound scores
        neutral_score = max(0, 1.0 - (positive_score + negative_score))
        compound_score = positive_score - negative_score

        # Ensure scores are between 0 and 1 (except compound which is -1 to 1)
        positive_score = min(1.0, max(0.0, positive_score))
        negative_score = min(1.0, max(0.0, negative_score))
        neutral_score = min(1.0, max(0.0, neutral_score))
        compound_score = min(1.0, max(-1.0, compound_score))

        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'compound': compound_score
        }

    def analyze_article(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article

        Args:
            article: NewsArticle object

        Returns:
            Dictionary with sentiment analysis results
        """
        # Get headline sentiment
        headline_sentiment = self.analyze_text(article.headline)

        # Get content sentiment
        content_sentiment = self.analyze_text(article.content)

        # Combined sentiment (weighted average)
        combined = {
            'positive': headline_sentiment['positive'] *
            0.3 +
            content_sentiment['positive'] *
            0.7,
            'negative': headline_sentiment['negative'] *
            0.3 +
            content_sentiment['negative'] *
            0.7,
            'neutral': headline_sentiment['neutral'] *
            0.3 +
            content_sentiment['neutral'] *
            0.7,
            'compound': headline_sentiment['compound'] *
            0.3 +
            content_sentiment['compound'] *
            0.7}

        # Store results in article
        article.sentiment = {
            'headline': headline_sentiment,
            'content': content_sentiment,
            'combined': combined
        }

        return article.sentiment

    def analyze_articles(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple news articles

        Args:
            articles: List of NewsArticle objects

        Returns:
            Dictionary with aggregated sentiment analysis results
        """
        # Analyze each article
        for article in articles:
            if not hasattr(article, 'sentiment') or article.sentiment is None:
                self.analyze_article(article)

        # Calculate average sentiment
        compound_scores = [article.sentiment['combined']
                           ['compound'] for article in articles]
        avg_compound = sum(compound_scores) / \
            len(compound_scores) if compound_scores else 0

        # Identify most positive and negative articles
        article_sentiments = [
            (article, article.sentiment['combined']['compound'])
            for article in articles]
        most_positive = max(
            article_sentiments,
            key=lambda x: x[1]) if article_sentiments else (
            None,
            0)
        most_negative = min(
            article_sentiments,
            key=lambda x: x[1]) if article_sentiments else (
            None,
            0)

        return {
            'average_sentiment': avg_compound,
            'article_scores': compound_scores,
            'most_positive_article': most_positive[0],
            'most_negative_article': most_negative[0]
        }


class EarningsCallAnalyzer:
    """
    Analyzer for earnings call transcripts

    This class extracts insights from earnings call transcripts,
    such as sentiment, tone shift, and key topics.
    """

    def __init__(self):
        """Initialize the earnings call analyzer"""
        self.sentiment_analyzer = FinbertSentimentAnalyzer()

        # Important financial phrases to track
        self.guidance_phrases = [
            "guidance", "outlook", "expect", "anticipate", "forecast",
            "project", "estimate", "target", "goal"
        ]

        self.risk_phrases = [
            "risk", "uncertainty", "challenge", "headwind", "difficult",
            "concern", "issue", "problem", "recession", "inflation"
        ]

        self.strength_phrases = [
            "growth", "strong", "increase", "improve", "opportunity",
            "confident", "excited", "pleased", "progress", "success"
        ]

    def _extract_sections(self, transcript: str) -> Dict[str, str]:
        """
        Extract different sections from an earnings call transcript

        Args:
            transcript: Full transcript text

        Returns:
            Dictionary with sections (prepared_remarks, qa_session)
        """
        sections = {}

        # Look for common section markers
        prepared_pattern = re.compile(
            r"(?:prepared remarks|opening remarks|operator|management discussion|"
            r"management presentation).*?(?=questions?|q&a|operator|the conference)",
            re.IGNORECASE | re.DOTALL)

        qa_pattern = re.compile(
            r"(?:questions?|q&a|question-and-answer).*",
            re.IGNORECASE | re.DOTALL
        )

        # Extract sections
        prepared_match = prepared_pattern.search(transcript)
        qa_match = qa_pattern.search(transcript)

        if prepared_match:
            sections['prepared_remarks'] = prepared_match.group(0)
        else:
            # If no clear separation, use first half
            half_point = len(transcript) // 2
            sections['prepared_remarks'] = transcript[:half_point]

        if qa_match:
            sections['qa_session'] = qa_match.group(0)
        else:
            # If no clear separation, use second half
            half_point = len(transcript) // 2
            sections['qa_session'] = transcript[half_point:]

        return sections

    def _extract_management_tone(
            self, prepared_remarks: str) -> Dict[str, float]:
        """
        Extract management tone from prepared remarks

        Args:
            prepared_remarks: Prepared remarks section text

        Returns:
            Dictionary with tone metrics
        """
        # Get overall sentiment
        sentiment = self.sentiment_analyzer.analyze_text(prepared_remarks)

        # Count phrase occurrences
        guidance_count = sum(
            1 for phrase in self.guidance_phrases if re.search(
                r'\b' + phrase + r'\b',
                prepared_remarks,
                re.IGNORECASE))

        risk_count = sum(1 for phrase in self.risk_phrases if re.search(
            r'\b' + phrase + r'\b', prepared_remarks, re.IGNORECASE))

        strength_count = sum(
            1 for phrase in self.strength_phrases if re.search(
                r'\b' + phrase + r'\b',
                prepared_remarks,
                re.IGNORECASE))

        # Calculate confidence ratio
        if risk_count > 0:
            confidence_ratio = strength_count / risk_count
        else:
            confidence_ratio = strength_count

        return {
            'sentiment': sentiment,
            'guidance_mentions': guidance_count,
            'risk_mentions': risk_count,
            'strength_mentions': strength_count,
            'confidence_ratio': confidence_ratio
        }

    def _extract_analyst_concerns(self, qa_session: str) -> Dict[str, Any]:
        """
        Extract analyst concerns from Q&A session

        Args:
            qa_session: Q&A session text

        Returns:
            Dictionary with analyst concern metrics
        """
        # Split into questions (approximate)
        question_pattern = re.compile(
            r"(?:question|q:).*(?=question|q:|$)",
            re.IGNORECASE | re.DOTALL)
        questions = question_pattern.findall(qa_session)

        if not questions:
            # Fallback: try to split by speaker changes
            questions = re.split(
                r"\n+(?=[A-Z][a-z]+ [A-Z][a-z]+:)", qa_session)

        # Analyze each question
        question_sentiments = []
        for question in questions:
            sentiment = self.sentiment_analyzer.analyze_text(question)
            question_sentiments.append(sentiment)

        # Calculate average sentiment
        if question_sentiments:
            avg_sentiment = {
                'positive': np.mean(
                    [s['positive'] for s in question_sentiments]),
                'negative': np.mean(
                    [s['negative'] for s in question_sentiments]),
                'neutral': np.mean(
                    [s['neutral'] for s in question_sentiments]),
                'compound': np.mean(
                    [s['compound'] for s in question_sentiments])}
        else:
            # Fallback
            avg_sentiment = self.sentiment_analyzer.analyze_text(qa_session)

        return {
            'num_questions': len(questions),
            'question_sentiments': question_sentiments,
            'avg_sentiment': avg_sentiment
        }

    def _extract_guidance_changes(self, transcript: str) -> Dict[str, Any]:
        """
        Extract guidance changes from the transcript

        Args:
            transcript: Full transcript text

        Returns:
            Dictionary with guidance change metrics
        """
        # Look for guidance statements
        # Break down the pattern into simpler components
        number_with_units = r"\$?\d+(?:\.\d+)?(?:\s*(?:billion|million|thousand|percent|%))?"
        guidance_terms = r"guidance|outlook|expect|project|forecast|anticipate"
        guidance_pattern = re.compile(
            fr"(?:{guidance_terms}).*?{number_with_units}",
            re.IGNORECASE
        )

        guidance_matches = guidance_pattern.findall(transcript)

        # Look for guidance revisions
        revision_verbs = r"revise|update|change|adjust|raise|lower|increase|decrease"
        forecast_terms = r"guidance|outlook|forecast"
        revision_pattern = re.compile(
            fr"(?:{revision_verbs}).*?(?:{forecast_terms})",
            re.IGNORECASE
        )

        revision_matches = revision_pattern.findall(transcript)

        # Determine direction
        increase_pattern = re.compile(
            r"(?:raise|increase|improve|higher)", re.IGNORECASE)
        decrease_pattern = re.compile(
            r"(?:lower|decrease|reduce|cut|below)", re.IGNORECASE)

        revision_text = " ".join(revision_matches)
        increase_matches = increase_pattern.findall(revision_text)
        decrease_matches = decrease_pattern.findall(revision_text)

        if len(increase_matches) > len(decrease_matches):
            direction = "increase"
        elif len(decrease_matches) > len(increase_matches):
            direction = "decrease"
        else:
            direction = "unchanged"

        return {
            'guidance_statements': guidance_matches,
            'guidance_revisions': revision_matches,
            'revision_direction': direction
        }

    def analyze_transcript(self,
                           transcript: str,
                           date: Union[str,
                                       datetime,
                                       None] = None) -> Dict[str,
                                                             Any]:
        """
        Analyze an earnings call transcript

        Args:
            transcript: Full transcript text
            date: Call date (optional, defaults to current date)

        Returns:
            Dictionary with analysis results
        """
        # Parse date if needed
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            try:
                date = datetime.fromisoformat(date)
            except ValueError:
                try:
                    date = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    date = datetime.now()

        # Extract sections
        sections = self._extract_sections(transcript)
        prepared_remarks = sections.get('prepared_remarks', '')
        qa_session = sections.get('qa_session', '')

        # Analyze tone
        management_tone = self._extract_management_tone(prepared_remarks)

        # Analyze Q&A
        analyst_concerns = self._extract_analyst_concerns(qa_session)

        # Analyze guidance
        guidance_changes = self._extract_guidance_changes(transcript)

        # Overall sentiment
        overall_sentiment = self.sentiment_analyzer.analyze_text(transcript)

        # Extract key topics using simple keyword extraction
        words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', transcript.lower())
        word_freq = Counter(words)
        # Filter out common stop words
        stop_words = {
            'the',
            'and',
            'is',
            'in',
            'to',
            'we',
            'our',
            'o',
            'for',
            'a',
            'on',
            'with',
            'as',
            'that'}
        topic_words = [(word, count) for word, count in word_freq.most_common(
            20) if word not in stop_words and len(word) > 3]
        key_topics = [word for word, _ in topic_words[:5]]

        # Determine guidance tone
        if guidance_changes['revision_direction'] == 'increase':
            guidance_tone = 'positive'
        elif guidance_changes['revision_direction'] == 'decrease':
            guidance_tone = 'negative'
        else:
            guidance_tone = 'neutral'

        # Determine confidence level based on management tone
        confidence_ratio = management_tone['confidence_ratio']
        if confidence_ratio > 2:
            confidence_level = 'high'
        elif confidence_ratio > 1:
            confidence_level = 'moderate'
        else:
            confidence_level = 'cautious'

        # Calculate overall sentiment score (-1 to 1 scale)
        sentiment_score = overall_sentiment['compound']

        # Compile detailed results
        detailed_results = {
            'date': date,
            'overall_sentiment': overall_sentiment,
            'management_tone': management_tone,
            'analyst_concerns': analyst_concerns,
            'guidance_changes': guidance_changes
        }

        # Compile simplified results for backward compatibility with examples
        simplified_results = {
            'date': date,
            'sentiment': sentiment_score,
            'key_topics': key_topics,
            'guidance_tone': guidance_tone,
            'confidence_level': confidence_level,
            # Include detailed results as a sub-dictionary
            'detailed_analysis': detailed_results
        }

        return simplified_results


class SentimentTimeSeriesBuilder:
    """
    Build time series of sentiment scores from news articles

    This class aggregates sentiment scores from news articles
    into a time series that can be used for forecasting.
    """

    def __init__(self):
        """Initialize the sentiment time series builder"""
        self.analyzer = FinbertSentimentAnalyzer()

    def build_sentiment_series(self,
                               articles: List[NewsArticle],
                               freq: str = 'D') -> pd.DataFrame:
        """
        Build sentiment time series from news articles

        Args:
            articles: List of NewsArticle objects
            freq: Frequency for resampling ('D' for daily, 'W' for weekly)

        Returns:
            DataFrame with sentiment time series
        """
        # Sort articles by date
        articles.sort(key=lambda x: x.date)

        # Analyze articles without sentiment yet
        for article in articles:
            if article.sentiment is None:
                self.analyzer.analyze_article(article)

        # Create DataFrame
        data = []
        for article in articles:
            sentiment = article.sentiment['combined']
            data.append({
                'date': article.date,
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'neutral': sentiment['neutral'],
                'compound': sentiment['compound'],
                'source': article.source
            })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Resample to desired frequency
        resampled = df.resample(freq).agg({
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'compound': 'mean',
            'source': 'count'
        })

        # Rename count column
        resampled.rename(columns={'source': 'article_count'}, inplace=True)

        # Forward fill missing dates
        resampled = resampled.fillna(method='ffill')

        # Add momentum features (change in sentiment)
        resampled['compound_change'] = resampled['compound'].diff()
        resampled['sentiment_momentum'] = resampled['compound'].rolling(
            window=3).mean().diff()

        return resampled


class NewsImpactAnalyzer:
    """
    Analyze the impact of news on stock prices

    This class combines sentiment analysis with price movements
    to identify how news affects stock prices.
    """

    def __init__(self):
        """Initialize the news impact analyzer"""
        self.sentiment_builder = SentimentTimeSeriesBuilder()

    def analyze(self, news_sentiment, earnings_insights):
        """
        Analyze the combined impact of news sentiment and earnings insights

        Args:
            news_sentiment: Dictionary with news sentiment results
            earnings_insights: Dictionary with earnings call insights

        Returns:
            Dictionary with adjustments for financial forecasts
        """
        # Calculate revenue growth adjustment based on sentiment
        # Scale to reasonable range
        news_factor = news_sentiment['average_sentiment'] * 0.5

        # Use earnings tone for additional factor
        earnings_factor = 0.0
        if earnings_insights['guidance_tone'] == 'positive':
            earnings_factor = 0.02
        elif earnings_insights['guidance_tone'] == 'negative':
            earnings_factor = -0.02

        # Adjust based on management confidence
        confidence_factor = 0.0
        if earnings_insights['confidence_level'] == 'high':
            confidence_factor = 0.01
        elif earnings_insights['confidence_level'] == 'cautious':
            confidence_factor = -0.01

        # Calculate combined adjustments
        revenue_growth_adj = news_factor + earnings_factor
        margin_adj = confidence_factor + (news_factor * 0.5)
        # Lower risk premium for positive sentiment
        risk_premium_adj = -news_factor * 0.01

        return {
            'revenue_growth_adj': revenue_growth_adj,
            'margin_adj': margin_adj,
            'risk_premium_adj': risk_premium_adj,
            'news_sentiment': news_sentiment,
            'earnings_insights': earnings_insights
        }

    def analyze_news_impact(self,
                            articles: List[NewsArticle],
                            price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how news sentiment impacts stock prices

        Args:
            articles: List of NewsArticle objects
            price_data: DataFrame with stock prices (must have 'date' and 'close' columns)

        Returns:
            Dictionary with impact analysis results
        """
        # Build sentiment series
        sentiment_series = self.sentiment_builder.build_sentiment_series(
            articles, freq='D')

        # Ensure price data has datetime index
        price_df = price_data.copy()
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df.set_index('date', inplace=True)
        else:
            price_df.index = pd.to_datetime(price_df.index)

        # Calculate returns
        price_df['return'] = price_df['close'].pct_change()

        # Join sentiment and price data
        combined = sentiment_series.join(
            price_df[['close', 'return']], how='inner')

        # Calculate correlations
        correlations = {
            'compound_return': combined['compound'].corr(
                combined['return']),
            'compound_change_return': combined['compound_change'].corr(
                combined['return']),
            'sentiment_momentum_return': combined['sentiment_momentum'].corr(
                combined['return'])}

        # Look at next-day impact (sentiment today → return tomorrow)
        combined['next_return'] = combined['return'].shift(-1)
        correlations['compound_next_return'] = combined['compound'].corr(
            combined['next_return'])

        # Calculate prediction accuracy (simple model)
        combined['sentiment_signal'] = np.sign(combined['compound'])
        combined['return_direction'] = np.sign(combined['next_return'])
        accurate = (combined['sentiment_signal'] ==
                    combined['return_direction'])
        accuracy = accurate.sum() / accurate.count() if accurate.count() > 0 else 0.5

        # Detect significant news days
        significant_news = combined[combined['article_count'] >= 5].copy()
        if not significant_news.empty:
            sig_accuracy = (
                significant_news['sentiment_signal'] == significant_news['return_direction']
            ).sum() / len(significant_news)
        else:
            sig_accuracy = None

        # Calculate impact score
        impact_score = abs(
            correlations['compound_next_return']) * (accuracy - 0.5) * 2

        return {
            'correlations': correlations,
            'accuracy': accuracy,
            'significant_news_accuracy': sig_accuracy,
            'impact_score': impact_score,
            'sample_size': len(combined),
            'sentiment_data': sentiment_series
        }


class NewsAndEarningsForecaster:
    """
    Enhances financial forecasts using news and earnings sentiment

    This class integrates news sentiment and earnings call analysis
    into financial forecasts to improve accuracy.
    """

    def __init__(self):
        """Initialize the forecaster"""
        self.sentiment_analyzer = FinbertSentimentAnalyzer()
        self.earnings_analyzer = EarningsCallAnalyzer()
        self.impact_analyzer = NewsImpactAnalyzer()

    def analyze_news_and_earnings(self,
                                  ticker: str,
                                  articles: List[NewsArticle],
                                  earnings_transcripts: Optional[List[Tuple[str, datetime]]] = None,
                                  price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze news and earnings for a stock

        Args:
            ticker: Stock ticker symbol
            articles: List of NewsArticle objects
            earnings_transcripts: Optional list of (transcript_text, date) tuples
            price_data: Optional DataFrame with price data

        Returns:
            Dictionary with combined analysis results
        """
        # Analyze news sentiment
        sentiment_series = self.impact_analyzer.sentiment_builder.build_sentiment_series(
            articles, freq='D')

        # Analyze news impact if price data available
        news_impact = None
        if price_data is not None:
            news_impact = self.impact_analyzer.analyze_news_impact(
                articles, price_data)

        # Analyze earnings transcripts
        earnings_analysis = []
        if earnings_transcripts:
            for transcript, date in earnings_transcripts:
                analysis = self.earnings_analyzer.analyze_transcript(
                    transcript, date)
                earnings_analysis.append(analysis)

        # Combine results
        results = {
            'ticker': ticker, 'sentiment_series': sentiment_series,
            'news_impact': news_impact, 'earnings_analysis': earnings_analysis,
            'latest_sentiment': sentiment_series.iloc[-1].to_dict()
            if not sentiment_series.empty else None}

        return results

    def generate_sample_earnings_call(
            self,
            ticker: str,
            sentiment: str = 'neutral') -> str:
        """
        Generate a sample earnings call transcript for testing purposes

        Args:
            ticker: Company ticker symbol
            sentiment: The sentiment tone ('positive', 'neutral', 'negative')

        Returns:
            Sample earnings call transcript text
        """
        company_name = f"{ticker} Corporation"

        # Template structure for earnings call
        templates = {
    'positive': {
        'intro': (
            f"Operator: Good day, and welcome to the " f"{company_name} quarterly earnings conference call. " f"At this time, all participants are in a listen-only " f"mode. After the speakers' presentation, there will " f"be a question-and-answer session."), 'prepared': """
                CEO: Thank you, operator. Good afternoon everyone, and thank you for joining {company_name}'s quarterly earnings call.

                We are extremely pleased to report exceptional results for the quarter, with revenue of $1.25 billion, up 18% year-over-year, significantly exceeding our guidance.

                Our gross margin reached 52%, up 200 basis points from last year, driven by a favorable product mix and continued operational efficiencies.

                We're particularly excited about the strong performance in our core markets, with growth across all major regions. The investments we've made in product innovation are clearly paying off, with new product lines contributing over $200 million in revenue this quarter.

                Looking ahead, we're raising our full-year guidance based on the strong momentum we're seeing across our business. We now expect full-year revenue growth of 15-18%, up from our previous guidance of 12-15%.

                CFO: Thank you. I'll now walk through our financial results in more detail.

                As mentioned, revenue was $1.25 billion, with operating income of $320 million, up 22% year-over-year. Earnings per share came in at $1.15, compared to $0.92 in the prior year period.

                Cash flow from operations was $350 million, and we ended the quarter with $2.8 billion in cash and investments. Our board has authorized an increase in our share repurchase program to $3 billion.

                We're in an excellent position to continue investing in growth opportunities while returning capital to shareholders.

                Back to you for Q&A.
                """, 'qa': """
                Analyst 1: Congratulations on the strong results. Could you elaborate on the drivers behind the margin expansion? Do you see this as sustainable going forward?

                CEO: Thank you for the question. The margin expansion reflects both our strategic shift toward higher-margin products and the efficiency initiatives we implemented last year. We believe these are structural improvements, and while there may be some quarterly fluctuations, we expect to sustain gross margins in the 50-52% range for the foreseeable future.

                CFO: I'll add that our automation investments have also contributed significantly, reducing production costs by approximately 5% year-over-year on a per-unit basis.

                Analyst 2: Can you discuss your strategy for capital allocation given the strong cash position?

                CFO: We're maintaining our balanced approach. In addition to the increased share repurchase authorization, we're allocating approximately $500 million for R&D and $300 million for strategic M&A opportunities that would strengthen our product portfolio or expand our technological capabilities.

                Analyst 3: Are you seeing any signs of slowing demand in any regions or product categories?

                CEO: Actually, we're seeing strength across all major regions. Even in markets where there are broader economic concerns, our products continue to gain market share. Our newer product categories are growing at over 25% year-over-year, and our established categories remain robust with mid-single-digit growth.

                Operator: This concludes today's question and answer session. I would now like to turn the call back to the CEO for closing remarks.

                CEO: Thank you everyone for joining us today. We're extremely pleased with our performance and excited about our opportunities ahead. We'll continue executing on our strategy and look forward to updating you on our progress next quarter.
                """}, 'neutral': {
                'intro': f"Operator: Good day, and welcome to the {company_name} quarterly earnings conference call. At this time, all participants are in a listen-only mode. After the speakers' presentation, there will be a question-and-answer session.", 'prepared': """
                CEO: Thank you, operator. Good afternoon everyone, and thank you for joining {company_name}'s quarterly earnings call.

                We're reporting solid results for the quarter, with revenue of $1.05 billion, up 5% year-over-year, in line with our guidance.

                Our gross margin was 48%, consistent with the same period last year, as cost improvements were offset by pricing pressures in certain markets.

                We saw mixed performance across regions, with strong growth in North America and Asia, while Europe was relatively flat due to ongoing economic uncertainties.

                Looking ahead, we're maintaining our full-year guidance of 4-6% revenue growth as we continue to navigate a competitive market environment.

                CFO: Thank you. I'll now walk through our financial results in more detail.

                As mentioned, revenue was $1.05 billion, with operating income of $250 million, up 3% year-over-year. Earnings per share came in at $0.95, compared to $0.92 in the prior year period.

                Cash flow from operations was $280 million, and we ended the quarter with $2.2 billion in cash and investments. We continue to allocate capital toward our $1.5 billion share repurchase program.

                We remain focused on balancing investments for future growth with prudent expense management.

                Back to you for Q&A.
                """, 'qa': """
                Analyst 1: Thanks for taking my question. Could you provide more color on the regional performance differences, particularly the slowdown in Europe?

                CEO: In Europe, we're seeing customers extend their purchasing cycles due to economic uncertainty. We don't view this as a loss of market share but rather a temporary slowdown in the overall market. Our product positioning remains strong, and we expect growth to resume as conditions improve.

                Analyst 2: How are you addressing the pricing pressures mentioned earlier?

                CFO: We're taking a balanced approach. In some segments, we're holding pricing to maintain market share, while in others, particularly our premium products, we have more pricing power. Simultaneously, we're accelerating cost efficiency programs to protect margins without compromising quality.

                Analyst 3: Could you comment on your inventory levels given the uneven demand environment?

                CFO: We've been cautious with inventory management. Current levels are about 5% higher than last year, which we consider appropriate given our growth projections. We've implemented more sophisticated forecasting tools to better align inventory with regional demand patterns.

                Operator: This concludes today's question and answer session. I would now like to turn the call back to the CEO for closing remarks.

                CEO: Thank you everyone for joining us today. While we face some challenges in certain markets, our overall business remains healthy, and we're confident in our strategy. We appreciate your continued support and look forward to speaking with you next quarter.
                """}, 'negative': {
                     'intro': f"Operator: Good day, and welcome to the {company_name} quarterly earnings conference call. At this time, all participants are in a listen-only mode. After the speakers' presentation, there will be a question-and-answer session.", 'prepared': """
                CEO: Thank you, operator. Good afternoon everyone, and thank you for joining {company_name}'s quarterly earnings call.

                This was a challenging quarter for our company. Revenue came in at $920 million, down 8% year-over-year and below our guidance of $980 million to $1 billion.

                Our gross margin decreased to 44%, down from 48% in the same period last year, primarily due to increased competitive pressure and unfavorable foreign exchange movements.

                We experienced significant headwinds across multiple regions, particularly in our international markets, where revenue declined by 12%.

                Given these challenges, we are revising our full-year guidance downward. We now expect a revenue decline of 2-5% for the year, compared to our previous expectation of 1-3% growth.

                CFO: I'll now walk through our financial results in more detail.

                As mentioned, revenue was $920 million, with operating income of $180 million, down 15% year-over-year. Earnings per share came in at $0.78, compared to $0.92 in the prior year period.

                Cash flow from operations decreased to $200 million, and we ended the quarter with $1.8 billion in cash and investments. In light of current conditions, we're suspending our share repurchase program to preserve capital.

                We've initiated a comprehensive cost reduction program targeting $120 million in annual savings, which will include some workforce reductions.

                Back to you for Q&A.
                """, 'qa': """
                Analyst 1: Thank you for taking my question. Can you elaborate on the competitive pressures you're facing? Are these primarily related to pricing or are you losing share to new entrants?

                CEO: It's a combination of factors. In several key markets, we're seeing more aggressive pricing from established competitors, and yes, some emerging players are gaining traction in the lower end of our market. We're conducting a strategic review of our product portfolio and pricing structure to address these challenges.

                Analyst 2: Could you quantify the impact of the foreign exchange movements versus the underlying business performance?

                CFO: Foreign exchange negatively impacted revenue by approximately $45 million or about 5 percentage points of the year-over-year decline. Even excluding this impact, however, we still faced organic challenges in our core business.

                Analyst 3: Regarding the cost reduction program, which areas of the business will be most affected, and do you expect any impact on your product development roadmap?

                CEO: The reductions will be across most functional areas, with an emphasis on non-customer-facing roles and administrative functions. We're protecting our core R&D investments, though we have reprioritized some projects. We've delayed two product launches that were planned for next year to ensure we can deliver the quality and features our customers expect.

                Operator: This concludes today's question and answer session. I would now like to turn the call back to the CEO for closing remarks.

                CEO: Thank you for your questions today. While we're disappointed with our performance this quarter, we believe the actions we're taking will position us for improved results as we move forward. We remain committed to our long-term strategy and are confident in our ability to navigate through these challenges. We appreciate your patience and support during this period.
                """}}

        # Ensure sentiment is valid
        if sentiment not in templates:
            sentiment = 'neutral'

        template = templates[sentiment]

        # Combine sections into full transcript
        full_transcript = (
            template['intro'] + "\n\n" +
            template['prepared'] + "\n\n" +
            template['qa']
        )

        return full_transcript

    def extract_forecast_adjustments(
            self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract forecast adjustments from news and earnings analysis

        Args:
            analysis_results: Output from analyze_news_and_earnings

        Returns:
            Dictionary with adjustment factors for different financial drivers
        """
        # Default adjustments (no change)
        adjustments = {
            'revenue_growth': 0.0,
            'profit_margin': 0.0,
            'capex': 0.0,
            'nwc': 0.0,
            'wacc': 0.0,
            'terminal_growth': 0.0
        }

        # Extract sentiment trend
        sentiment_series = analysis_results.get('sentiment_series')
        if sentiment_series is not None and not sentiment_series.empty:
            # Look at recent sentiment trend (last 30 days if available)
            recent = sentiment_series.tail(30)
            if len(recent) >= 5:
                sentiment_trend = recent['compound'].mean()
                momentum = recent['sentiment_momentum'].mean()

                # Adjust growth expectations based on sentiment
                # Stronger positive sentiment → higher growth
                adjustments['revenue_growth'] = sentiment_trend * 0.02

                # Adjust margins based on sentiment momentum
                # Improving sentiment → improving margins
                adjustments['profit_margin'] = momentum * 0.01

                # Terminal growth expectations
                adjustments['terminal_growth'] = sentiment_trend * 0.005

                # Risk perception (impacts WACC)
                volatility = recent['compound'].std()
                adjustments['wacc'] = -sentiment_trend * \
                    0.01 + volatility * 0.02

        # Extract earnings guidance impact
        earnings_analysis = analysis_results.get('earnings_analysis', [])
        if earnings_analysis:
            # Use most recent earnings call
            latest = earnings_analysis[-1]

            # Extract guidance direction
            guidance_dir = latest['guidance_changes']['revision_direction']

            # Management confidence ratio
            confidence_ratio = latest['management_tone']['confidence_ratio']

            # Adjust growth based on guidance
            if guidance_dir == 'increase':
                adjustments['revenue_growth'] += 0.01
                adjustments['profit_margin'] += 0.005
            elif guidance_dir == 'decrease':
                adjustments['revenue_growth'] -= 0.015
                adjustments['profit_margin'] -= 0.01

            # Adjust capex based on confidence
            if confidence_ratio > 1.5:  # High confidence
                adjustments['capex'] -= 0.01  # More efficient capex
            elif confidence_ratio < 0.8:  # Low confidence
                adjustments['capex'] += 0.015  # Less efficient capex

            # Adjust WACC based on analyst concerns
            analyst_sentiment = latest['analyst_concerns']['avg_sentiment'][
                'compound']
            adjustments['wacc'] -= analyst_sentiment * 0.005

        # Extract news impact
        news_impact = analysis_results.get('news_impact')
        if news_impact:
            impact_score = news_impact['impact_score']

            # If news has significant impact on stock
            if abs(impact_score) > 0.1:
                # Adjust terminal growth based on impact
                adjustments['terminal_growth'] += impact_score * 0.01

        return adjustments

    def adjust_forecast_distributions(self,
                                      base_distributions: Dict[str,
                                                               Any],
                                      adjustments: Dict[str,
                                                        float]) -> Dict[str,
                                                                        Any]:
        """
        Adjust forecast distributions based on news and earnings analysis

        Args:
            base_distributions: Base forecast distributions
            adjustments: Adjustment factors from extract_forecast_adjustments

        Returns:
            Adjusted forecast distributions
        """
        adjusted = {}

        for key, dist in base_distributions.items():
            # Skip if we don't have an adjustment for this parameter
            if key not in adjustments:
                adjusted[key] = dist
                continue

            # Get adjustment factor
            adj_factor = adjustments[key]

            # Process based on distribution type
            adjusted[key] = self._adjust_distribution(dist, adj_factor)

        return adjusted

    def _adjust_distribution(self, dist: Any, adj_factor: float) -> Any:
        """
        Helper method to adjust a single distribution by the adjustment factor

        Args:
            dist: The distribution to adjust
            adj_factor: Amount to adjust by

        Returns:
            Adjusted distribution
        """
        # For our DistributionSpec objects
        if hasattr(dist, 'dist_type') and hasattr(dist, 'params'):
            return self._adjust_distribution_spec(dist, adj_factor)

        # For mean, stddev tuples
        elif isinstance(dist, tuple) and len(dist) == 2:
            mean, stddev = dist
            return (mean + adj_factor, stddev)

        # For constant values
        elif isinstance(dist, (int, float)):
            return dist + adj_factor

        # Unknown format - keep as is
        else:
            return dist

    def _adjust_distribution_spec(self, dist: Any, adj_factor: float) -> Any:
        """
        Adjust a distribution spec object

        Args:
            dist: Distribution spec object
            adj_factor: Amount to adjust by

        Returns:
            Adjusted distribution spec
        """
        new_dist = dist.copy()

        if dist.dist_type == 'triangular':
            # Adjust triangular distribution
            params = dist.params
            new_dist.params = {
                'left': params['left'] + adj_factor,
                'mode': params['mode'] + adj_factor,
                'right': params['right'] + adj_factor
            }

        elif dist.dist_type == 'normal':
            # Adjust normal distribution
            params = dist.params
            new_dist.params = {
                'mean': params['mean'] + adj_factor,
                'stddev': params['stddev']  # Keep same variability
            }

        return new_dist


def fetch_news_articles(ticker: str, days_back: int = 30) -> List[NewsArticle]:
    """
    Fetch news articles for a ticker (placeholder)

    In a real implementation, this would connect to a news API
    to fetch actual articles. This version returns placeholders.

    Args:
        ticker: Stock ticker symbol
        days_back: Number of days to look back

    Returns:
        List of NewsArticle objects
    """
    articles = []

    # Generate placeholder articles
    today = datetime.now()

    # Create some sample headlines and sentiments
    samples = [
        ("positive", f"{ticker} Reports Strong Quarterly Results, Raises Guidance",
         "The company reported earnings that exceeded analyst expectations and raised its full-year guidance."),
        ("positive", f"{ticker} Announces New Product Launch",
         "The company unveiled its latest product line, which analysts expect to drive significant revenue growth."),
        ("negative", f"{ticker} Faces Supply Chain Challenges",
         "The company reported difficulties with suppliers that may impact production targets for the quarter."),
        ("neutral", f"{ticker} CEO Speaks at Industry Conference",
         "The chief executive discussed industry trends and the company's positioning in the market."),
        ("negative", f"Analyst Downgrades {ticker} Citing Valuation Concerns",
         "A leading Wall Street analyst downgraded the stock from Buy to Hold, citing valuation concerns after the recent rally.")
    ]

    # Generate articles with different dates
    for i in range(min(days_back, 15)):
        # Select a random sample
        _, headline, content = samples[i % len(samples)]

        # Create article date (spread throughout the period)
        article_date = today - timedelta(days=int(i * days_back / 15))

        # Create article
        article = NewsArticle(
            headline=headline,
            content=content +
            f" The company's management expressed confidence in the long-term strategy for {ticker}.",
            date=article_date,
            source=["Bloomberg", "Reuters", "CNBC",
                    "Wall Street Journal", "Financial Times"][i % 5],
                        url=f"https://example.com/news/{ticker}/{article_date.strftime('%Y%m%d')}"
        )

        articles.append(article)

    return articles
