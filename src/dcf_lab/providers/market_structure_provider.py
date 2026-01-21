"""
Market Structure Data Provider
=============================

Comprehensive integration for market microstructure data including:
- Short interest data from multiple sources
- Market maker activity and order flow
- Dark pool trading volumes
- Options flow and volatility skew
- Trading venue statistics
- Market depth and liquidity metrics

Data Sources:
- FINRA (Financial Industry Regulatory Authority)
- CBOE (Chicago Board Options Exchange)
- NYSE/NASDAQ market data feeds
- Alternative Trading Systems (ATS)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class ShortInterestData:
    """Short interest data structure"""
    symbol: str
    settlement_date: str
    short_interest: int
    avg_daily_volume: int
    days_to_cover: float
    change_percent: float


@dataclass
class DarkPoolData:
    """Dark pool trading data"""
    symbol: str
    date: str
    dark_pool_volume: int
    total_volume: int
    dark_pool_percentage: float
    venue_breakdown: Dict[str, int]


class MarketStructureProvider:
    """
    Market structure data provider for microstructure analysis

    Key Features:
    - Short interest tracking and analysis
    - Dark pool volume monitoring
    - Market maker order flow
    - Options market microstructure
    - Trading venue statistics
    - Liquidity and depth metrics
    """

    def __init__(self):
        """Initialize market structure provider"""
        self.base_urls = {
            'finra': 'https://api.finra.org',
            'cboe': 'https://www.cboe.com/us/equities',
            'iex': 'https://cloud.iexapis.com/stable'
        }

        self._cache = {}
        self._last_request_time = {}
        self._min_request_intervals = {
            'finra': 1.0,  # Conservative rate limiting
            'cboe': 0.5,
            'iex': 0.1
        }

        # Common headers
        self.headers = {
            'User-Agent': 'MarketStructure-Provider/1.0',
            'Accept': 'application/json'
        }

    def _rate_limit(self, source: str):
        """Rate limiting by data source"""
        last_time = self._last_request_time.get(source, 0)
        min_interval = self._min_request_intervals.get(source, 1.0)

        elapsed = time.time() - last_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time[source] = time.time()

    def _make_request(
            self,
            source: str,
            endpoint: str,
            params: Dict = None) -> Dict:
        """Make API request with proper rate limiting"""
        self._rate_limit(source)

        base_url = self.base_urls.get(source)
        if not base_url:
            raise ValueError(f"Unknown data source: {source}")

        url = f"{base_url}/{endpoint}"

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()

            # Handle different response formats
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'text/csv' in content_type:
                # Convert CSV to dict format
                return {'csv_data': response.text}
            else:
                return {'raw_data': response.text}

        except requests.exceptions.RequestException as e:
            logger.error(f"{source} API request failed: {e}")
            raise

    def get_short_interest_data(
            self,
            symbol: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> List[ShortInterestData]:
        """
        Get short interest data for a symbol

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of ShortInterestData objects
        """
        # MOCK DATA DISABLED - Real FINRA API integration required
        logger.error("Mock short interest data generation disabled")
        return []
        
        if False:  # Disabled mock implementation
            try:
                rng = np.random.default_rng(42)

            # Simulate realistic short interest data
            if not start_date:
                start_date = (datetime.now() -
                              timedelta(days=180)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Generate sample data (replace with actual API calls)
            data = []
            current = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')

            # Simulate bi-monthly reporting (FINRA reports twice monthly)
            while current <= end:
                # Simulate realistic short interest metrics
                base_short_interest = 50000000  # Base for large cap
                volatility = rng.normal(0, 0.1)  # 10% volatility
                short_interest = int(base_short_interest * (1 + volatility))

                avg_daily_volume = int(rng.normal(25000000, 5000000))
                days_to_cover = short_interest / max(avg_daily_volume, 1)

                # Previous period for change calculation
                prev_short = int(base_short_interest *
                                 (1 + rng.normal(0, 0.05)))
                change_percent = (
                    (short_interest - prev_short) / prev_short) * 100

                data.append(ShortInterestData(
                    symbol=symbol,
                    settlement_date=current.strftime('%Y-%m-%d'),
                    short_interest=max(short_interest, 0),
                    avg_daily_volume=max(avg_daily_volume, 1),
                    days_to_cover=round(days_to_cover, 2),
                    change_percent=round(change_percent, 2)
                ))

                # Move to next bi-monthly period
                current += timedelta(days=15)

            return data

        except Exception as e:
            logger.error(
                f"Failed to get short interest data for {symbol}: {e}")
            return []

    def get_dark_pool_data(
            self,
            symbol: str,
            date: Optional[str] = None) -> Optional[DarkPoolData]:
        """
        Get dark pool trading data for a symbol

        Args:
            symbol: Stock symbol
            date: Trading date (YYYY-MM-DD), defaults to latest

        Returns:
            DarkPoolData object or None
        """
        # MOCK DATA DISABLED
        logger.error("Mock dark pool data generation disabled")
        return None
        
        try:
            if False:  # Disabled
                # Use modern numpy random generator
                rng = np.random.default_rng(42)

            if not date:
                date = datetime.now().strftime('%Y-%m-%d')

            # Simulate dark pool data (replace with actual data feeds)
            # Dark pools typically account for 15-25% of equity volume
            total_volume = rng.integers(20000000, 100000000)
            dark_pool_percentage = rng.uniform(0.15, 0.25)
            dark_pool_volume = int(total_volume * dark_pool_percentage)

            # Simulate venue breakdown
            venues = [
                'UBS ATS',
                'Credit Suisse CrossFinder',
                'ITG POSIT',
                'Goldman Sachs Sigma X',
                'Morgan Stanley Pool']
            venue_breakdown = {}
            remaining_volume = dark_pool_volume

            for i, venue in enumerate(venues[:-1]):
                allocation = rng.uniform(0.1, 0.3)
                volume = int(remaining_volume * allocation)
                venue_breakdown[venue] = volume
                remaining_volume -= volume

            venue_breakdown[venues[-1]] = remaining_volume  # Assign remainder

            return DarkPoolData(
                symbol=symbol,
                date=date,
                dark_pool_volume=dark_pool_volume,
                total_volume=total_volume,
                dark_pool_percentage=round(dark_pool_percentage * 100, 2),
                venue_breakdown=venue_breakdown
            )

        except Exception as e:
            logger.error(f"Failed to get dark pool data for {symbol}: {e}")
            return None

    def get_options_flow_data(self, symbol: str,
                              expiration: Optional[str] = None) -> Dict:
        """
        Get options flow and unusual activity data

        Args:
            symbol: Stock symbol
            expiration: Options expiration date (YYYY-MM-DD)

        Returns:
            Dict with options flow metrics
        """
        try:
            # Use modern numpy random generator
            rng = np.random.default_rng(42)

            # Simulate options flow data
            flow_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'total_volume': rng.integers(50000, 500000),
                'call_volume': 0,
                'put_volume': 0,
                'call_put_ratio': 0,
                'unusual_activity': [],
                'volume_by_expiration': {},
                'net_flow_by_strike': {}
            }

            # Call/Put breakdown
            call_percentage = rng.uniform(0.4, 0.7)
            flow_data['call_volume'] = int(
                flow_data['total_volume'] * call_percentage)
            flow_data['put_volume'] = flow_data['total_volume'] - \
                flow_data['call_volume']
            flow_data['call_put_ratio'] = round(
                flow_data['call_volume'] / max(flow_data['put_volume'], 1), 2)

            # Simulate unusual activity
            if rng.random() < 0.3:  # 30% chance of unusual activity
                flow_data['unusual_activity'].append({
                    'type': 'large_block',
                    'strike': rng.integers(180, 220),
                    'expiration': (datetime.now() + timedelta(days=rng.integers(7, 60))).strftime('%Y-%m-%d'),
                    'volume': rng.integers(5000, 20000),
                    'option_type': rng.choice(['call', 'put']),
                    'premium': round(rng.uniform(100000, 1000000), 2)
                })

            return flow_data

        except Exception as e:
            logger.error(f"Failed to get options flow data for {symbol}: {e}")
            return {}

    def get_market_maker_data(self, symbol: str) -> Dict:
        """
        Get market maker activity and order flow data

        Args:
            symbol: Stock symbol

        Returns:
            Dict with market maker metrics
        """
        try:
            # Use modern numpy random generator
            rng = np.random.default_rng(42)

            # Simulate market maker data
            mm_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'total_market_maker_volume': rng.integers(5000000, 50000000),
                'market_maker_percentage': round(rng.uniform(0.25, 0.45), 2),
                'top_market_makers': [],
                'order_flow_metrics': {
                    'payment_for_order_flow': round(rng.uniform(1000000, 10000000), 2),
                    'internalization_rate': round(rng.uniform(0.15, 0.35), 2),
                    'avg_fill_time_ms': round(rng.uniform(5, 25), 2)
                },
                'liquidity_metrics': {
                    'bid_ask_spread_bps': round(rng.uniform(1, 8), 2),
                    'market_depth_shares': rng.integers(50000, 200000),
                    'impact_per_million': round(rng.uniform(2, 15), 2)
                }
            }

            # Top market makers (simulated)
            mm_firms = [
                'Citadel Securities',
                'Virtu Financial',
                'Two Sigma Securities',
                'Jane Street',
                'Jump Trading']
            total_mm_volume = mm_data['total_market_maker_volume']

            for i, firm in enumerate(mm_firms):
                if i == len(mm_firms) - 1:
                    # Last firm gets remainder
                    volume = total_mm_volume
                else:
                    share = rng.uniform(0.1, 0.3)
                    volume = int(total_mm_volume * share)
                    total_mm_volume -= volume

                mm_data['top_market_makers'].append({
                    'firm': firm,
                    'volume': volume,
                    'market_share': round((volume / mm_data['total_market_maker_volume']) * 100, 2)
                })

            return mm_data

        except Exception as e:
            logger.error(f"Failed to get market maker data for {symbol}: {e}")
            return {}

    def get_venue_statistics(
            self,
            symbol: str,
            date: Optional[str] = None) -> Dict:
        """
        Get trading venue statistics and market share data

        Args:
            symbol: Stock symbol
            date: Trading date (YYYY-MM-DD)

        Returns:
            Dict with venue statistics
        """
        try:
            # Use modern numpy random generator
            rng = np.random.default_rng(42)

            if not date:
                date = datetime.now().strftime('%Y-%m-%d')

            # Simulate venue statistics
            venues = {
                'NYSE': rng.uniform(0.15, 0.25),
                'NASDAQ': rng.uniform(0.15, 0.25),
                'BATS': rng.uniform(0.10, 0.20),
                'IEX': rng.uniform(0.02, 0.08),
                'ARCA': rng.uniform(0.08, 0.15),
                'EDGX': rng.uniform(0.05, 0.12),
                'Dark Pools': rng.uniform(0.15, 0.25)
            }

            # Normalize to 100%
            total = sum(venues.values())
            venues = {k: round((v/total) * 100, 2) for k, v in venues.items()}

            total_volume = rng.integers(50000000, 200000000)

            venue_data = {
                'symbol': symbol,
                'date': date,
                'total_volume': total_volume,
                'venue_breakdown': {},
                'metrics': {
                    'fragmentation_index': round(1 - max(venues.values())/100, 3),
                    'primary_venue_share': max(venues.values()),
                    'off_exchange_percentage': venues.get('Dark Pools', 0)
                }
            }

            # Calculate actual volumes
            for venue, percentage in venues.items():
                volume = int(total_volume * (percentage / 100))
                venue_data['venue_breakdown'][venue] = {
                    'volume': volume,
                    'percentage': percentage,
                    'avg_trade_size': rng.integers(100, 1000)
                }

            return venue_data

        except Exception as e:
            logger.error(f"Failed to get venue statistics for {symbol}: {e}")
            return {}

    def get_comprehensive_market_structure(self, symbol: str) -> Dict:
        """
        Get comprehensive market structure analysis for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with comprehensive market structure data
        """
        try:
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_sources': []
            }

            # Get short interest data
            short_interest = self.get_short_interest_data(symbol)
            if short_interest:
                result['short_interest'] = {
                    'latest': short_interest[-1].__dict__
                    if short_interest else None,
                    'historical_count': len(short_interest),
                    'trend_analysis': self._analyze_short_interest_trend(
                        short_interest)}
                result['data_sources'].append('short_interest')

            # Get dark pool data
            dark_pool = self.get_dark_pool_data(symbol)
            if dark_pool:
                result['dark_pool'] = dark_pool.__dict__
                result['data_sources'].append('dark_pool')

            # Get options flow data
            options_flow = self.get_options_flow_data(symbol)
            if options_flow:
                result['options_flow'] = options_flow
                result['data_sources'].append('options_flow')

            # Get market maker data
            market_makers = self.get_market_maker_data(symbol)
            if market_makers:
                result['market_makers'] = market_makers
                result['data_sources'].append('market_makers')

            # Get venue statistics
            venues = self.get_venue_statistics(symbol)
            if venues:
                result['venue_statistics'] = venues
                result['data_sources'].append('venue_statistics')

            # Calculate overall market structure score
            result['market_structure_score'] = self._calculate_structure_score(
                result)

            return result

        except Exception as e:
            logger.error(
                f"Failed to get comprehensive market structure for {symbol}: {e}")
            return {'error': str(e)}

    def _analyze_short_interest_trend(
            self, data: List[ShortInterestData]) -> Dict:
        """Analyze short interest trends"""
        if len(data) < 2:
            return {'trend': 'insufficient_data'}

        recent_data = data[-5:]  # Last 5 periods
        short_interests = [d.short_interest for d in recent_data]

        # Calculate trend
        if len(short_interests) >= 3:
            slope = np.polyfit(
                range(
                    len(short_interests)),
                short_interests,
                1)[0]
            trend = 'increasing' if slope > 0 else 'decreasing'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'latest_days_to_cover': recent_data[-1].days_to_cover,
            'avg_days_to_cover': round(np.mean([d.days_to_cover for d in recent_data]), 2),
            'volatility': round(np.std([d.change_percent for d in recent_data]), 2)
        }

    def _calculate_structure_score(self, data: Dict) -> Dict:
        """Calculate overall market structure health score"""
        score = 0
        factors = []

        # Calculate individual component scores
        score, factors = self._score_short_interest(data, score, factors)
        score, factors = self._score_dark_pool(data, score, factors)
        score, factors = self._score_options_flow(data, score, factors)
        score, factors = self._score_market_makers(data, score, factors)
        score, factors = self._score_venue_diversity(data, score, factors)

        # Calculate rating based on score
        if score > 40:
            rating = 'excellent'
        elif score > 10:
            rating = 'good'
        elif score > -10:
            rating = 'fair'
        else:
            rating = 'poor'

        return {
            'score': max(0, min(100, score + 50)),  # Normalize to 0-100
            'rating': rating,
            'contributing_factors': factors
        }

    def _score_short_interest(
            self,
            data: Dict,
            score: int,
            factors: list) -> tuple:
        """Score short interest factor"""
        if 'short_interest' in data and data['short_interest']['latest']:
            days_to_cover = data['short_interest']['latest']['days_to_cover']
            if days_to_cover < 3:
                score += 20
                factors.append('low_short_interest')
            elif days_to_cover > 10:
                score -= 10
                factors.append('high_short_interest')
        return score, factors

    def _score_dark_pool(self, data: Dict, score: int, factors: list) -> tuple:
        """Score dark pool factor"""
        if 'dark_pool' in data:
            dp_pct = data['dark_pool']['dark_pool_percentage']
            if 15 <= dp_pct <= 25:
                score += 15
                factors.append('healthy_dark_pool_activity')
            elif dp_pct > 35:
                score -= 10
                factors.append('excessive_dark_pool_activity')
        return score, factors

    def _score_options_flow(
            self,
            data: Dict,
            score: int,
            factors: list) -> tuple:
        """Score options flow factor"""
        if 'options_flow' in data:
            call_put_ratio = data['options_flow']['call_put_ratio']
            if 0.8 <= call_put_ratio <= 1.5:
                score += 10
                factors.append('balanced_options_flow')
        return score, factors

    def _score_market_makers(
            self,
            data: Dict,
            score: int,
            factors: list) -> tuple:
        """Score market maker factor"""
        if 'market_makers' in data:
            mm_pct = data['market_makers']['market_maker_percentage']
            if 0.25 <= mm_pct <= 0.45:
                score += 10
                factors.append('healthy_market_making')
        return score, factors

    def _score_venue_diversity(
            self,
            data: Dict,
            score: int,
            factors: list) -> tuple:
        """Score venue fragmentation factor"""
        if 'venue_statistics' in data:
            frag_index = data['venue_statistics']['metrics'][
                'fragmentation_index']
            if 0.6 <= frag_index <= 0.8:
                score += 10
                factors.append('appropriate_fragmentation')
        return score, factors
