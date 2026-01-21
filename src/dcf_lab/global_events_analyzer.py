"""
Quick fix for global events analyzer to restore functionality
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GlobalEvent:
    """Represents a global event from GDELT"""
    event_id: str
    event_date: str
    event_type: str
    actors: List[str]
    location: str
    goldstein_scale: float
    num_mentions: int
    avg_tone: float
    source_url: Optional[str] = None

@dataclass
class EventSignal:
    """Event-driven trading signal"""
    signal_date: str
    signal_type: str
    event_category: str
    affected_regions: List[str]
    sentiment_score: float
    magnitude: float
    confidence: float
    description: str
    related_events: List[str]

class GlobalEventsAnalyzer:
    """Simple global events analyzer for testing"""
    
    def __init__(self):
        self.region_market_weights = {
            'US': 0.25, 'CHINA': 0.20, 'EUROPE': 0.18, 'JAPAN': 0.08,
            'UK': 0.06, 'GERMANY': 0.05, 'FRANCE': 0.04, 'CANADA': 0.03,
            'OTHER': 0.01
        }
        
        # Initialize GDELT auto-fetcher for real data
        try:
            from src.dcf_lab.gdelt_auto_fetcher import GDELTAutoFetcher
            self.gdelt_fetcher = GDELTAutoFetcher()
            self._gdelt_available = True
            logger.info("✅ GDELT auto-fetcher initialized for global_events")
        except Exception as e:
            self.gdelt_fetcher = None
            self._gdelt_available = False
            logger.warning(f"⚠️ GDELT auto-fetcher not available: {e}")
    
    def get_global_events(self, start_date: str, end_date: str, symbol: str = "AAPL") -> List[GlobalEvent]:
        """Get real GDELT events from auto-fetcher"""
        if not self._gdelt_available or self.gdelt_fetcher is None:
            logger.warning("GDELT not available - returning empty events")
            return []
        
        try:
            # Fetch GDELT data using auto-fetcher
            gdelt_df = self.gdelt_fetcher.fetch(symbol, start_date, end_date)
            
            if gdelt_df is None or gdelt_df.empty:
                logger.warning(f"No GDELT data for {symbol} in range {start_date} to {end_date}")
                return []
            
            # Convert GDELT headlines to GlobalEvent objects
            events = []
            for date, row in gdelt_df.iterrows():
                headline = row.get('headline', '')
                if not headline:
                    continue
                
                # Parse GDELT themes/headline into event structure
                # Simple sentiment extraction from headline tone
                tone = self._extract_tone_from_headline(headline)
                goldstein = self._estimate_goldstein_from_tone(tone)
                
                event = GlobalEvent(
                    event_id=f"gdelt_{date.strftime('%Y%m%d')}",
                    event_date=date.strftime('%Y-%m-%d'),
                    actors=['GLOBAL'],  # Simplified - would parse from GDELT themes
                    event_type='NEWS',
                    location='GLOBAL',  # Use location instead of regions
                    goldstein_scale=goldstein,
                    avg_tone=tone,
                    num_mentions=1  # Each headline counts as 1 mention
                )
                events.append(event)
            
            logger.info(f"✅ Loaded {len(events)} GDELT events for {symbol}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to fetch GDELT events: {e}")
            return []
    
    def _extract_tone_from_headline(self, headline: str) -> float:
        """Extract sentiment tone from headline (-100 to +100 scale)"""
        # Simple keyword-based tone estimation
        headline_lower = headline.lower()
        
        positive_words = ['rise', 'gain', 'up', 'growth', 'profit', 'success', 'win', 'positive']
        negative_words = ['fall', 'loss', 'down', 'decline', 'crisis', 'risk', 'negative', 'concern']
        
        pos_count = sum(1 for word in positive_words if word in headline_lower)
        neg_count = sum(1 for word in negative_words if word in headline_lower)
        
        if pos_count == 0 and neg_count == 0:
            return 0.0
        
        # Map to -100 to +100 scale
        net_sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        return net_sentiment * 50.0  # Scale to roughly -50 to +50 range
    
    def _estimate_goldstein_from_tone(self, tone: float) -> float:
        """Estimate Goldstein scale from tone (-10 to +10 scale)"""
        # Goldstein scale: -10 (very negative) to +10 (very positive)
        return tone / 10.0  # Map from -100/+100 to -10/+10
    
    def detect_regime_changes(self, events: List[GlobalEvent]) -> List[EventSignal]:
        """Simple regime change detection"""
        return []
    
    def get_global_events_features(self, start_date: str, end_date: str, ticker: str = "AAPL") -> Dict[str, float]:
        """Generate global events features"""
        events = self.get_global_events(start_date, end_date, symbol=ticker)
        
        if not events:
            return self._get_default_features()
        
        # Calculate basic features
        total_events = len(events)
        avg_tone = np.mean([e.avg_tone for e in events])
        avg_goldstein = np.mean([e.goldstein_scale for e in events])
        mentions = sum([e.num_mentions for e in events])
        
        # Corporate mentions (simplified)
        corporate_mentions = sum(1 for e in events if ticker.upper() in str(e.actors))
        corporate_sentiment = np.mean([e.avg_tone for e in events if ticker.upper() in str(e.actors)]) if corporate_mentions > 0 else 0
        
        features = {
            'events_event_count': total_events,
            'events_total_mentions': mentions,
            'events_avg_goldstein_scale': avg_goldstein,
            'events_avg_tone': avg_tone,
            'events_sentiment_trend_score': avg_tone / 100.0,  # Normalize
            'events_negative_event_ratio': sum(1 for e in events if e.avg_tone < -10) / max(total_events, 1),
            'events_positive_event_ratio': sum(1 for e in events if e.avg_tone > 10) / max(total_events, 1),
            'events_regime_change_signals': 0,  # Simplified
            'events_max_signal_magnitude': max([abs(e.goldstein_scale) for e in events]) if events else 0,
            'events_crisis_risk_signals': sum(1 for e in events if e.goldstein_scale < -5),
            'events_corporate_mentions': corporate_mentions,
            'events_corporate_sentiment': corporate_sentiment / 100.0,  # Normalize
            'events_geographic_risk_exposure': np.mean([abs(e.goldstein_scale) for e in events]) if events else 0
        }
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Default features when no events available"""
        return {
            'events_event_count': 0,
            'events_total_mentions': 0,
            'events_avg_goldstein_scale': 0,
            'events_avg_tone': 0,
            'events_sentiment_trend_score': 0,
            'events_negative_event_ratio': 0,
            'events_positive_event_ratio': 0,
            'events_regime_change_signals': 0,
            'events_max_signal_magnitude': 0,
            'events_crisis_risk_signals': 0,
            'events_corporate_mentions': 0,
            'events_corporate_sentiment': 0,
            'events_geographic_risk_exposure': 0
        }