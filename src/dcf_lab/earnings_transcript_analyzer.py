"""
Earnings Transcript NLP Analysis for AI Forecasting

This module provides comprehensive analysis of earnings call transcripts
including:
- Management sentiment and tone analysis
- Forward guidance extraction and classification
- Key topic identification and trending
- Management confidence scoring
- Revenue/earnings forecast extraction
- Risk factor identification
- Competitive positioning analysis

Key Features:
- Real-time transcript ingestion from earnings calls
- Advanced NLP with transformer models for sentiment
- Financial entity recognition (FinBERT integration)
- Management forward guidance classification
- Earnings surprise prediction based on tone
- Competitive intelligence extraction
- Risk assessment from management commentary

Data Sources:
- SEC filings (earnings reports)
- Earnings call transcript providers
- Financial news and press releases
- Company investor relations pages
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# NLP and text analysis
try:
    import spacy
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Financial text processing
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
MARKET_SHARE = 'market share'


@dataclass
class TranscriptSegment:
    """Represents a segment of an earnings transcript"""
    speaker: str  # CEO, CFO, Analyst, etc.
    role: str  # management, analyst, moderator
    content: str
    timestamp: Optional[str] = None
    segment_type: str = "discussion"  # presentation, qa, discussion


@dataclass
class ForwardGuidance:
    """Forward-looking guidance extracted from transcripts"""
    guidance_type: str  # revenue, earnings, margin, capex
    period: str  # Q1, Q2, FY2024, etc.
    value_range: Optional[Tuple[float, float]] = None
    confidence_level: str = "moderate"  # low, moderate, high
    sentiment: str = "neutral"  # positive, negative, neutral
    context: str = ""


@dataclass
class SentimentMetrics:
    """Sentiment analysis metrics for transcript"""
    overall_sentiment: float  # -1 to 1
    management_confidence: float  # 0 to 1
    forward_optimism: float  # -1 to 1
    risk_concern_level: float  # 0 to 1
    analyst_sentiment: float  # -1 to 1
    uncertainty_level: float  # 0 to 1


@dataclass
class TopicAnalysis:
    """Topic analysis results"""
    primary_topics: List[str]
    topic_sentiment: Dict[str, float]
    trending_topics: List[str]
    risk_topics: List[str]
    growth_topics: List[str]


@dataclass
class EarningsTranscriptAnalysis:
    """Complete earnings transcript analysis results"""
    ticker: str
    quarter: str
    analysis_date: str
    sentiment_metrics: SentimentMetrics
    forward_guidance: List[ForwardGuidance]
    topic_analysis: TopicAnalysis
    key_insights: List[str]
    surprise_indicators: Dict[str, float]
    competitive_mentions: Dict[str, int]
    risk_factors: List[str]


class EarningsTranscriptAnalyzer:
    """
    Earnings Transcript NLP Analysis Engine
    
    Provides comprehensive analysis of earnings call transcripts using
    advanced NLP techniques for financial sentiment and guidance extraction.
    """
    
    def __init__(self):
        """Initialize the earnings transcript analyzer"""
        self.sentiment_pipeline = None
        self.finbert_pipeline = None
        self.nlp = None
        self.sia = None
        
        # Initialize NLP models
        self._initialize_models()
        
        # Financial keywords and patterns
        self._initialize_financial_patterns()
        
    def _initialize_models(self):
        """Initialize NLP models and pipelines"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Financial sentiment analysis with FinBERT
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                
                # General sentiment for comparison
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("✅ Transformer models loaded successfully")
            
            if TRANSFORMERS_AVAILABLE:
                # Load spaCy model for entity recognition
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("SpaCy model 'en_core_web_sm' not found")
                    self.nlp = None
            
            if NLTK_AVAILABLE:
                # NLTK sentiment analyzer as fallback
                self.sia = SentimentIntensityAnalyzer()
                logger.info("✅ NLTK sentiment analyzer loaded")
                
        except Exception as e:
            logger.warning(f"⚠️ Error initializing NLP models: {e}")
    
    def _initialize_financial_patterns(self):
        """Initialize financial keyword patterns and regex"""
        # Guidance keywords
        self.guidance_patterns = {
            'revenue': [
                r'revenue.*guidance', r'sales.*outlook', r'top.*line.*expect',
                r'revenue.*forecast', r'sales.*projection'
            ],
            'earnings': [
                r'earnings.*guidance', r'eps.*outlook', r'profit.*expect',
                r'earnings.*forecast', r'bottom.*line.*projection'
            ],
            'margin': [
                r'margin.*guidance', r'margin.*outlook', r'margin.*expect',
                r'operating.*margin', r'gross.*margin.*expect'
            ],
            'capex': [
                r'capital.*expenditure', r'capex.*guidance',
                r'investment.*plan', r'spending.*outlook', r'capex.*expect'
            ]
        }
        
        # Sentiment keywords
        self.positive_indicators = [
            'strong', 'growth', 'optimistic', 'confident', 'exceed',
            'outperform', 'momentum', 'upside', 'positive', 'robust'
        ]
        
        self.negative_indicators = [
            'challenging', 'headwinds', 'pressure', 'decline', 'weak',
            'concern', 'risk', 'uncertainty', 'volatile', 'difficult'
        ]
        
        # Risk-related keywords
        self.risk_keywords = [
            'risk', 'uncertainty', 'volatility', 'headwind', 'challenge',
            'concern', 'pressure', 'competition', 'regulation', 'supply chain'
        ]
        
        # Growth keywords
        self.growth_keywords = [
            'growth', 'expansion', 'opportunity', 'investment', 'innovation',
            MARKET_SHARE, 'new product', 'digital transformation'
        ]
    
    def get_earnings_transcript(self, ticker: str, quarter: str) -> str:
        """
        Retrieve earnings transcript for analysis
        
        Args:
            ticker: Stock ticker symbol
            quarter: Quarter (e.g., "Q1 2024")
            
        Returns:
            Transcript text
        """
        # In a real implementation, this would fetch from:
        # - Earnings call transcript APIs
        # - SEC filings
        # - Financial news services
        # - Company investor relations pages
        
        # Mock transcript for demonstration
        return f"""
        {ticker} Q1 2024 Earnings Call Transcript
        
        CEO John Smith: Thank you for joining us today. I'm pleased to report that Q1 2024 was a strong quarter for {ticker}. We delivered revenue of $2.5 billion, representing 15% year-over-year growth, which exceeded our guidance of 12-14% growth.
        
        Our margins expanded by 200 basis points driven by operational efficiency and favorable product mix. We're seeing strong momentum in our core business segments and remain optimistic about the growth trajectory.
        
        Looking ahead to Q2, we expect revenue growth of 10-12% and continued margin expansion. However, we are monitoring supply chain pressures and competitive dynamics closely.
        
        CFO Jane Doe: Our balance sheet remains strong with $500M in cash. We're confident in our ability to invest in growth while maintaining financial discipline. We're raising our full-year guidance based on Q1 performance.
        
        Analyst Question: Can you provide more color on the competitive landscape?
        
        CEO: We're seeing increased competition in certain segments, but our differentiated products and strong customer relationships position us well. We continue to gain market share in key markets.
        
        Analyst: What are the main risks you're watching?
        
        CEO: The main risks include supply chain disruptions, inflation pressure on costs, and potential economic slowdown. However, we have contingency plans in place and remain confident in our strategy.
        """
    
    def parse_transcript_segments(self, transcript: str) -> List[TranscriptSegment]:
        """
        Parse transcript into structured segments
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            List of transcript segments
        """
        segments = []
        
        # Simple parsing logic - in practice would be more sophisticated
        lines = transcript.split('\n')
        current_speaker = ""
        current_content = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains speaker designation
            if ':' in line and any(title in line.lower() for title in ['ceo', 'cfo', 'analyst', 'question']):
                # Save previous segment
                if current_speaker and current_content:
                    role = self._classify_speaker_role(current_speaker)
                    segments.append(TranscriptSegment(
                        speaker=current_speaker,
                        role=role,
                        content=current_content.strip()
                    ))
                
                # Start new segment
                parts = line.split(':', 1)
                current_speaker = parts[0].strip()
                current_content = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Continue current segment
                current_content += " " + line
        
        # Add final segment
        if current_speaker and current_content:
            role = self._classify_speaker_role(current_speaker)
            segments.append(TranscriptSegment(
                speaker=current_speaker,
                role=role,
                content=current_content.strip()
            ))
        
        return segments
    
    def _classify_speaker_role(self, speaker: str) -> str:
        """Classify speaker role based on title"""
        speaker_lower = speaker.lower()
        if any(title in speaker_lower for title in ['ceo', 'chief executive', 'president']):
            return 'management'
        elif any(title in speaker_lower for title in ['cfo', 'chief financial']):
            return 'management'
        elif any(title in speaker_lower for title in ['analyst', 'question']):
            return 'analyst'
        else:
            return 'other'
    
    def analyze_sentiment(self, segments: List[TranscriptSegment]) -> SentimentMetrics:
        """
        Analyze sentiment across transcript segments
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Sentiment metrics
        """
        management_segments = [s for s in segments if s.role == 'management']
        analyst_segments = [s for s in segments if s.role == 'analyst']
        
        # Analyze management sentiment
        mgmt_sentiments = []
        confidence_scores = []
        risk_scores = []
        
        for segment in management_segments:
            sentiment = self._analyze_segment_sentiment(segment.content)
            mgmt_sentiments.append(sentiment)
            
            # Confidence scoring based on language patterns
            confidence = self._calculate_confidence_score(segment.content)
            confidence_scores.append(confidence)
            
            # Risk concern level
            risk_score = self._calculate_risk_concern(segment.content)
            risk_scores.append(risk_score)
        
        # Analyze analyst sentiment
        analyst_sentiments = []
        for segment in analyst_segments:
            sentiment = self._analyze_segment_sentiment(segment.content)
            analyst_sentiments.append(sentiment)
        
        # Calculate forward-looking optimism
        forward_optimism = self._analyze_forward_guidance_sentiment(management_segments)
        
        # Calculate uncertainty level
        uncertainty = self._calculate_uncertainty_level(management_segments)
        
        return SentimentMetrics(
            overall_sentiment=np.mean(mgmt_sentiments) if mgmt_sentiments else 0.0,
            management_confidence=np.mean(confidence_scores) if confidence_scores else 0.5,
            forward_optimism=forward_optimism,
            risk_concern_level=np.mean(risk_scores) if risk_scores else 0.0,
            analyst_sentiment=np.mean(analyst_sentiments) if analyst_sentiments else 0.0,
            uncertainty_level=uncertainty
        )
    
    def _analyze_segment_sentiment(self, text: str) -> float:
        """Analyze sentiment of a text segment"""
        try:
            if self.finbert_pipeline:
                # Use FinBERT for financial sentiment
                result = self.finbert_pipeline(text[:512])  # Truncate for model limits
                label = result[0]['label']
                score = result[0]['score']
                
                # Convert to -1 to 1 scale
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:
                    return 0.0
            
            elif self.sia:
                # Fallback to NLTK
                scores = self.sia.polarity_scores(text)
                return scores['compound']
            
            else:
                # Basic keyword-based sentiment
                return self._basic_sentiment_analysis(text)
                
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_indicators if word in text_lower)
        negative_count = sum(1 for word in self.negative_indicators if word in text_lower)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_count
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate management confidence score"""
        confidence_indicators = [
            'confident', 'strong', 'solid', 'robust', 'optimistic',
            'expect', 'believe', 'committed', 'disciplined'
        ]
        
        uncertainty_indicators = [
            'uncertain', 'challenging', 'difficult', 'volatile',
            'cautious', 'monitoring', 'watching', 'concerned'
        ]
        
        text_lower = text.lower()
        
        confidence_count = sum(1 for word in confidence_indicators if word in text_lower)
        uncertainty_count = sum(1 for word in uncertainty_indicators if word in text_lower)
        
        # Base confidence is 0.5, adjust based on language
        base_confidence = 0.5
        adjustment = (confidence_count - uncertainty_count) * 0.1
        
        return max(0.0, min(1.0, base_confidence + adjustment))
    
    def _calculate_risk_concern(self, text: str) -> float:
        """Calculate risk concern level"""
        text_lower = text.lower()
        risk_mentions = sum(1 for keyword in self.risk_keywords if keyword in text_lower)
        
        # Normalize by text length (approximate word count)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        risk_density = risk_mentions / (word_count / 100)  # Risk mentions per 100 words
        return min(1.0, risk_density)
    
    def _analyze_forward_guidance_sentiment(self, segments: List[TranscriptSegment]) -> float:
        """Analyze sentiment of forward-looking statements"""
        forward_texts = []
        
        for segment in segments:
            # Look for forward-looking language
            sentences = sent_tokenize(segment.content) if NLTK_AVAILABLE else [segment.content]
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['expect', 'outlook', 'guidance', 'forward', 'next', 'future']):
                    forward_texts.append(sentence)
        
        if not forward_texts:
            return 0.0
        
        sentiments = [self._analyze_segment_sentiment(text) for text in forward_texts]
        return np.mean(sentiments)
    
    def _calculate_uncertainty_level(self, segments: List[TranscriptSegment]) -> float:
        """Calculate overall uncertainty level"""
        uncertainty_words = [
            'uncertain', 'volatility', 'unclear', 'depends', 'monitor',
            'watching', 'cautious', 'careful', 'wait and see'
        ]
        
        total_words = 0
        uncertainty_count = 0
        
        for segment in segments:
            words = segment.content.lower().split()
            total_words += len(words)
            uncertainty_count += sum(1 for word in words if any(uw in word for uw in uncertainty_words))
        
        if total_words == 0:
            return 0.0
        
        return min(1.0, uncertainty_count / (total_words / 100))
    
    def extract_forward_guidance(self, segments: List[TranscriptSegment]) -> List[ForwardGuidance]:
        """
        Extract forward guidance from transcript segments
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of forward guidance items
        """
        guidance_items = []
        management_segments = [s for s in segments if s.role == 'management']
        
        for segment in management_segments:
            text = segment.content
            
            # Look for each type of guidance
            for guidance_type, patterns in self.guidance_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract surrounding context
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end]
                        
                        # Try to extract numerical guidance
                        value_range = self._extract_numerical_guidance(context)
                        
                        # Determine time period
                        period = self._extract_time_period(context)
                        
                        # Analyze sentiment of guidance
                        sentiment = self._classify_guidance_sentiment(context)
                        
                        # Assess confidence level
                        confidence = self._assess_guidance_confidence(context)
                        
                        guidance_items.append(ForwardGuidance(
                            guidance_type=guidance_type,
                            period=period,
                            value_range=value_range,
                            confidence_level=confidence,
                            sentiment=sentiment,
                            context=context.strip()
                        ))
        
        return guidance_items
    
    def _extract_numerical_guidance(self, text: str) -> Optional[Tuple[float, float]]:
        """Extract numerical ranges from guidance text"""
        # Look for percentage ranges like "10-12%" or "between 10% and 12%"
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)[-–](\d+(?:\.\d+)?)%',
            r'between (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)%'
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return (float(match.group(1)), float(match.group(2)))
                except ValueError:
                    continue
        
        # Look for single percentage values
        single_percentage = re.search(r'(\d+(?:\.\d+)?)%', text)
        if single_percentage:
            try:
                value = float(single_percentage.group(1))
                return (value, value)
            except ValueError:
                pass
        
        return None
    
    def _extract_time_period(self, text: str) -> str:
        """Extract time period from guidance text"""
        text_lower = text.lower()
        
        # Quarterly periods
        if 'q1' in text_lower or 'first quarter' in text_lower:
            return 'Q1'
        elif 'q2' in text_lower or 'second quarter' in text_lower:
            return 'Q2'
        elif 'q3' in text_lower or 'third quarter' in text_lower:
            return 'Q3'
        elif 'q4' in text_lower or 'fourth quarter' in text_lower:
            return 'Q4'
        
        # Annual periods
        elif 'full year' in text_lower or 'full-year' in text_lower or 'fy' in text_lower:
            return 'FY'
        elif 'next year' in text_lower:
            return 'Next Year'
        
        return 'Unspecified'
    
    def _classify_guidance_sentiment(self, text: str) -> str:
        """Classify sentiment of guidance"""
        sentiment_score = self._analyze_segment_sentiment(text)
        
        if sentiment_score > 0.1:
            return 'positive'
        elif sentiment_score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_guidance_confidence(self, text: str) -> str:
        """Assess confidence level of guidance"""
        high_confidence_words = ['confident', 'expect', 'will', 'committed']
        low_confidence_words = [
            'hope', 'may', 'could', 'potential', 'uncertain'
        ]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in high_confidence_words):
            return 'high'
        elif any(word in text_lower for word in low_confidence_words):
            return 'low'
        else:
            return 'moderate'
    
    def analyze_topics(
            self, segments: List[TranscriptSegment]) -> TopicAnalysis:
        """
        Analyze key topics discussed in transcript
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Topic analysis results
        """
        # Combine all management content
        management_segments = [s for s in segments if s.role == 'management']
        management_text = " ".join([s.content for s in management_segments])
        
        # Extract key topics using keyword analysis
        # In a real implementation, would use topic modeling
        # (LDA, BERTopic, etc.)
        
        primary_topics = self._extract_primary_topics(management_text)
        topic_sentiment = self._analyze_topic_sentiment(
            management_text, primary_topics)
        trending_topics = self._identify_trending_topics(primary_topics)
        risk_topics = self._identify_risk_topics(management_text)
        growth_topics = self._identify_growth_topics(management_text)
        
        return TopicAnalysis(
            primary_topics=primary_topics,
            topic_sentiment=topic_sentiment,
            trending_topics=trending_topics,
            risk_topics=risk_topics,
            growth_topics=growth_topics
        )
    
    def _extract_primary_topics(self, text: str) -> List[str]:
        """Extract primary business topics"""
        # Business topic keywords
        topic_keywords = {
            'revenue': ['revenue', 'sales', 'top line'],
            'margins': ['margin', 'profitability', 'operating income'],
            'growth': ['growth', 'expansion', MARKET_SHARE],
            'costs': ['costs', 'expenses', 'efficiency'],
            'market': ['market', 'customer', 'demand'],
            'product': ['product', 'innovation', 'development'],
            'competition': ['competition', 'competitive', 'rivals'],
            'operations': ['operations', 'supply chain', 'manufacturing'],
            'strategy': ['strategy', 'strategic', 'transformation'],
            'investment': ['investment', 'capex', 'spending']
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # Return top topics
        sorted_topics = sorted(
            topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:8]]
    
    def _analyze_topic_sentiment(
            self, text: str, topics: List[str]) -> Dict[str, float]:
        """Analyze sentiment for each topic"""
        topic_sentiment = {}
        
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else [text]
        
        for topic in topics:
            sentences_for_topic = [
                s for s in sentences if topic.lower() in s.lower()
            ]
            if sentences_for_topic:
                sentiments = [
                    self._analyze_segment_sentiment(s)
                    for s in sentences_for_topic
                ]
                topic_sentiment[topic] = np.mean(sentiments)
            else:
                topic_sentiment[topic] = 0.0
        
        return topic_sentiment
    
    def _identify_trending_topics(self, topics: List[str]) -> List[str]:
        """Identify trending topics (simplified)"""
        # In practice, would compare to historical transcripts
        trend_indicators = ['digital', 'cloud', 'AI', 'sustainability', 'ESG']
        return [topic for topic in topics if any(indicator in topic.lower() for indicator in trend_indicators)]
    
    def _identify_risk_topics(self, text: str) -> List[str]:
        """Identify risk-related topics"""
        risk_topics = []
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else [text]
        
        for sentence in sentences:
            if any(risk_word in sentence.lower() for risk_word in self.risk_keywords):
                # Extract topic from sentence (simplified)
                words = sentence.lower().split()
                for i, word in enumerate(words):
                    if word in self.risk_keywords and i > 0:
                        context = ' '.join(words[max(0, i-2):i+3])
                        risk_topics.append(context)
        
        return list(set(risk_topics))[:5]  # Top 5 unique risk topics
    
    def _identify_growth_topics(self, text: str) -> List[str]:
        """Identify growth-related topics"""
        growth_topics = []
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else [text]
        
        for sentence in sentences:
            if any(growth_word in sentence.lower() for growth_word in self.growth_keywords):
                # Extract topic from sentence (simplified)
                words = sentence.lower().split()
                for i, word in enumerate(words):
                    if any(gw in word for gw in self.growth_keywords) and i > 0:
                        context = ' '.join(words[max(0, i-2):i+3])
                        growth_topics.append(context)
        
        return list(set(growth_topics))[:5]  # Top 5 unique growth topics
    
    def calculate_surprise_indicators(self, analysis: 'EarningsTranscriptAnalysis') -> Dict[str, float]:
        """Calculate earnings surprise prediction indicators"""
        indicators = {}
        
        # Management confidence vs. guidance sentiment
        confidence_guidance_delta = (
            analysis.sentiment_metrics.management_confidence - 
            (analysis.sentiment_metrics.forward_optimism + 1) / 2  # Normalize to 0-1
        )
        indicators['confidence_guidance_alignment'] = confidence_guidance_delta
        
        # Positive guidance count
        positive_guidance = sum(1 for g in analysis.forward_guidance if g.sentiment == 'positive')
        total_guidance = len(analysis.forward_guidance)
        indicators['positive_guidance_ratio'] = positive_guidance / total_guidance if total_guidance > 0 else 0.5
        
        # Risk concern vs. forward optimism
        risk_optimism_balance = analysis.sentiment_metrics.forward_optimism - analysis.sentiment_metrics.risk_concern_level
        indicators['risk_optimism_balance'] = risk_optimism_balance
        
        # Uncertainty level (lower = more surprise potential)
        indicators['certainty_level'] = 1 - analysis.sentiment_metrics.uncertainty_level
        
        # Overall surprise indicator (weighted combination)
        surprise_score = (
            0.3 * indicators['confidence_guidance_alignment'] +
            0.3 * indicators['positive_guidance_ratio'] +
            0.2 * indicators['risk_optimism_balance'] +
            0.2 * indicators['certainty_level']
        )
        indicators['overall_surprise_probability'] = max(0, min(1, (surprise_score + 1) / 2))
        
        return indicators
    
    def analyze_earnings_transcript(self, ticker: str, quarter: str) -> EarningsTranscriptAnalysis:
        """
        Complete earnings transcript analysis
        
        Args:
            ticker: Stock ticker symbol
            quarter: Quarter (e.g., "Q1 2024")
            
        Returns:
            Complete analysis results
        """
        try:
            # Get transcript
            transcript = self.get_earnings_transcript(ticker, quarter)
            
            # Parse into segments
            segments = self.parse_transcript_segments(transcript)
            
            # Analyze sentiment
            sentiment_metrics = self.analyze_sentiment(segments)
            
            # Extract forward guidance
            forward_guidance = self.extract_forward_guidance(segments)
            
            # Analyze topics
            topic_analysis = self.analyze_topics(segments)
            
            # Create analysis object
            analysis = EarningsTranscriptAnalysis(
                ticker=ticker,
                quarter=quarter,
                analysis_date=datetime.now().strftime('%Y-%m-%d'),
                sentiment_metrics=sentiment_metrics,
                forward_guidance=forward_guidance,
                topic_analysis=topic_analysis,
                key_insights=[],
                surprise_indicators={},
                competitive_mentions={},
                risk_factors=topic_analysis.risk_topics
            )
            
            # Calculate surprise indicators
            analysis.surprise_indicators = self.calculate_surprise_indicators(analysis)
            
            # Extract key insights
            analysis.key_insights = self._generate_key_insights(analysis)
            
            # Analyze competitive mentions
            analysis.competitive_mentions = self._extract_competitive_mentions(segments)
            
            logger.info(f"✅ Completed earnings transcript analysis for {ticker} {quarter}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing earnings transcript: {e}")
            # Return empty analysis
            return EarningsTranscriptAnalysis(
                ticker=ticker,
                quarter=quarter,
                analysis_date=datetime.now().strftime('%Y-%m-%d'),
                sentiment_metrics=SentimentMetrics(0, 0.5, 0, 0, 0, 0),
                forward_guidance=[],
                topic_analysis=TopicAnalysis([], {}, [], [], []),
                key_insights=[],
                surprise_indicators={},
                competitive_mentions={},
                risk_factors=[]
            )
    
    def _generate_key_insights(self, analysis: EarningsTranscriptAnalysis) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Sentiment insights
        if analysis.sentiment_metrics.overall_sentiment > 0.3:
            insights.append("Management sentiment is notably positive")
        elif analysis.sentiment_metrics.overall_sentiment < -0.3:
            insights.append("Management sentiment shows concerning negativity")
        
        # Confidence insights
        if analysis.sentiment_metrics.management_confidence > 0.7:
            insights.append("Management demonstrates high confidence in guidance")
        elif analysis.sentiment_metrics.management_confidence < 0.3:
            insights.append("Management confidence appears low")
        
        # Forward guidance insights
        positive_guidance = sum(1 for g in analysis.forward_guidance if g.sentiment == 'positive')
        if positive_guidance > len(analysis.forward_guidance) / 2:
            insights.append("Forward guidance skews positive")
        
        # Risk insights
        if analysis.sentiment_metrics.risk_concern_level > 0.5:
            insights.append("Elevated risk concerns expressed by management")
        
        # Surprise potential
        if analysis.surprise_indicators.get('overall_surprise_probability', 0) > 0.7:
            insights.append("High potential for positive earnings surprise")
        elif analysis.surprise_indicators.get('overall_surprise_probability', 0) < 0.3:
            insights.append("Low expectations suggest limited surprise potential")
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_competitive_mentions(self, segments: List[TranscriptSegment]) -> Dict[str, int]:
        """Extract mentions of competitors"""
        # Common competitor keywords
        competitive_keywords = [
            'competitor', 'competition', 'rival', MARKET_SHARE,
            'competitive landscape', 'peers', 'industry'
        ]
        
        mentions = {}
        management_text = " ".join([s.content for s in segments if s.role == 'management'])
        
        for keyword in competitive_keywords:
            count = management_text.lower().count(keyword)
            if count > 0:
                mentions[keyword] = count
        
        return mentions
    
    def get_earnings_transcript_features(self, ticker: str, quarter: str) -> Dict[str, float]:
        """
        Generate earnings transcript features for ML models
        
        Args:
            ticker: Stock ticker symbol
            quarter: Quarter for analysis
            
        Returns:
            Dictionary of transcript features
        """
        try:
            analysis = self.analyze_earnings_transcript(ticker, quarter)
            
            features = {}
            
            # Sentiment features
            features['mgmt_sentiment'] = analysis.sentiment_metrics.overall_sentiment
            features['mgmt_confidence'] = analysis.sentiment_metrics.management_confidence
            features['forward_optimism'] = analysis.sentiment_metrics.forward_optimism
            features['risk_concern_level'] = analysis.sentiment_metrics.risk_concern_level
            features['analyst_sentiment'] = analysis.sentiment_metrics.analyst_sentiment
            features['uncertainty_level'] = analysis.sentiment_metrics.uncertainty_level
            
            # Guidance features
            features['guidance_count'] = len(analysis.forward_guidance)
            positive_guidance = sum(1 for g in analysis.forward_guidance if g.sentiment == 'positive')
            features['positive_guidance_ratio'] = positive_guidance / len(analysis.forward_guidance) if analysis.forward_guidance else 0
            
            high_confidence_guidance = sum(1 for g in analysis.forward_guidance if g.confidence_level == 'high')
            features['high_confidence_guidance_ratio'] = high_confidence_guidance / len(analysis.forward_guidance) if analysis.forward_guidance else 0
            
            # Topic features
            features['topic_count'] = len(analysis.topic_analysis.primary_topics)
            features['growth_topic_count'] = len(analysis.topic_analysis.growth_topics)
            features['risk_topic_count'] = len(analysis.topic_analysis.risk_topics)
            
            # Average topic sentiment
            if analysis.topic_analysis.topic_sentiment:
                features['avg_topic_sentiment'] = np.mean(list(analysis.topic_analysis.topic_sentiment.values()))
            else:
                features['avg_topic_sentiment'] = 0.0
            
            # Surprise indicators
            features.update(analysis.surprise_indicators)
            
            # Competitive features
            features['competitive_mentions'] = sum(analysis.competitive_mentions.values())
            
            logger.info(f"📊 Generated {len(features)} earnings transcript features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error generating earnings transcript features: {e}")
            return {}