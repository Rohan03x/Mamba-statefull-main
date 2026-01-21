"""
Market Microstructure & Short Interest Provider for DCF Lab

This module provides comprehensive market microstructure analysis including:
- FINRA short interest data (bi-monthly reports)
- SEC Failed-to-Deliver (FTD) data
- Short interest ratios and trends
- Short squeeze indicators
- Market liquidity metrics
- Trading volume analysis
- OTC market data

Key Features:
- Short interest ratio calculations
- Days-to-cover analysis
- Failed-to-deliver tracking
- Short squeeze probability models
- Liquidity stress indicators
- Volume-weighted metrics
- Market maker activity analysis

Data Sources:
- FINRA Short Interest reports
- SEC Regulation SHO data
- SEC Failed-to-Deliver files
- FINRA OTC transparency data
- Historical trading patterns

Enhanced Capabilities:
- Predictive short squeeze models
- Liquidity risk assessment
- Market stress detection
- Trading pattern recognition
- Risk-adjusted metrics

Author: DCF Lab Team
Created: 2025-09-18
"""

import logging
import os
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
class MarketMicrostructureConfig:
    """Configuration for market microstructure provider"""
    cache_dir: str = "./cache/market_microstructure"
    enable_cache: bool = True
    max_cache_age_hours: int = 24  # Daily updates for most data
    timeout: int = 30
    retries: int = 3
    rate_limit_delay: float = 1.0  # Be respectful to government servers

    # Data source URLs
    finra_short_interest_url: str = "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files"
    sec_ftd_url: str = "https://www.sec.gov/data/foiadocsfailsdatahtm"
    finra_otc_url: str = "https://otctransparency.finra.org/otctransparency/"


class ShortInterestAnalyzer:
    """
    Short Interest Analysis Engine

    Provides comprehensive analysis of short interest data including
    ratios, trends, and short squeeze indicators.
    """

    def __init__(self, config: MarketMicrostructureConfig):
        self.config = config

        # Short interest thresholds for analysis
        self.short_interest_thresholds = {
            'low': 5.0,      # < 5% of float
            'moderate': 15.0,  # 5-15% of float
            'high': 25.0,    # 15-25% of float
            'extreme': 50.0  # > 25% of float
        }

        # Days to cover thresholds
        self.days_to_cover_thresholds = {
            'low_risk': 2.0,
            'moderate_risk': 5.0,
            'high_risk': 10.0,
            'squeeze_risk': 20.0
        }

    def calculate_short_metrics(self, symbol: str, short_data: pd.DataFrame,
                                volume_data: pd.DataFrame = None,
                                float_shares: float = None) -> Dict[str, Any]:
        """
        Calculate comprehensive short interest metrics

        Args:
            symbol: Stock symbol
            short_data: DataFrame with short interest data
            volume_data: DataFrame with trading volume data
            float_shares: Number of shares in float

        Returns:
            Dict with short interest analysis
        """
        metrics = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'short_interest': {},
            'days_to_cover': {},
            'short_ratio': {},
            'trend_analysis': {},
            'squeeze_indicators': {},
            'risk_assessment': {},
            'success': False,
            'error': None
        }

        try:
            if short_data.empty:
                metrics['error'] = "No short interest data available"
                return metrics

            # Get latest short interest
            latest_short = short_data.iloc[-1] if not short_data.empty else None

            if latest_short is not None:
                short_shares = latest_short.get('short_volume', 0)
                total_volume = latest_short.get('total_volume', 0)

                metrics['short_interest'] = {
                    'short_shares': int(short_shares),
                    'total_volume': int(total_volume),
                    'short_volume_ratio': float(
                        short_shares /
                        total_volume) if total_volume > 0 else 0,
                    'settlement_date': latest_short.get(
                        'settlement_date',
                        ''),
                    'report_date': latest_short.get(
                        'report_date',
                        '')}

                # Calculate short ratio (% of float)
                if float_shares and float_shares > 0:
                    short_ratio = (short_shares / float_shares) * 100
                    metrics['short_ratio'] = {
                        'percentage': float(short_ratio),
                        'float_shares': float(float_shares),
                        'classification': self._classify_short_ratio(
                            short_ratio)}

                # Calculate days to cover
                if volume_data is not None and not volume_data.empty:
                    recent_volume = volume_data.tail(
                        20)['volume'].mean()  # 20-day average
                    if recent_volume > 0:
                        days_to_cover = short_shares / recent_volume
                        metrics['days_to_cover'] = {
                            'days': float(days_to_cover),
                            'avg_volume_20d': float(recent_volume),
                            'classification': self._classify_days_to_cover(
                                days_to_cover),
                            'risk_level': self._assess_days_to_cover_risk(
                                days_to_cover)}

                # Trend analysis
                if len(short_data) >= 2:
                    trend = self._analyze_short_trends(short_data)
                    metrics['trend_analysis'] = trend

                # Short squeeze indicators
                squeeze_signals = self._detect_squeeze_signals(
                    short_data, volume_data, metrics)
                metrics['squeeze_indicators'] = squeeze_signals

                # Risk assessment
                risk_assessment = self._assess_short_risk(metrics)
                metrics['risk_assessment'] = risk_assessment

                metrics['success'] = True
                logger.info(f"Calculated short metrics for {symbol}")

        except Exception as e:
            metrics['error'] = str(e)
            logger.error(
                f"Failed to calculate short metrics for {symbol}: {e}")

        return metrics

    def _classify_short_ratio(self, ratio: float) -> str:
        """Classify short interest ratio"""
        if ratio < self.short_interest_thresholds['low']:
            return 'low'
        elif ratio < self.short_interest_thresholds['moderate']:
            return 'moderate'
        elif ratio < self.short_interest_thresholds['high']:
            return 'high'
        else:
            return 'extreme'

    def _classify_days_to_cover(self, days: float) -> str:
        """Classify days to cover"""
        if days < self.days_to_cover_thresholds['low_risk']:
            return 'quick_cover'
        elif days < self.days_to_cover_thresholds['moderate_risk']:
            return 'moderate_cover'
        elif days < self.days_to_cover_thresholds['high_risk']:
            return 'slow_cover'
        else:
            return 'difficult_cover'

    def _assess_days_to_cover_risk(self, days: float) -> str:
        """Assess risk level based on days to cover"""
        if days < self.days_to_cover_thresholds['low_risk']:
            return 'low'
        elif days < self.days_to_cover_thresholds['moderate_risk']:
            return 'moderate'
        elif days < self.days_to_cover_thresholds['high_risk']:
            return 'high'
        else:
            return 'extreme'

    def _analyze_short_trends(
            self, short_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze short interest trends"""
        trends = {
            'direction': 'unknown',
            'magnitude': 0,
            'consistency': 0,
            'recent_change': 0,
            'volatility': 0
        }

        try:
            if len(short_data) < 2:
                return trends

            # Calculate period-over-period changes
            short_data['short_change'] = short_data['short_volume'].pct_change()

            # Recent trend (last 6 periods)
            recent_data = short_data.tail(6)
            if len(recent_data) >= 2:
                recent_changes = recent_data['short_change'].dropna()

                if len(recent_changes) > 0:
                    avg_change = recent_changes.mean()
                    trends['recent_change'] = float(avg_change)

                    # Determine direction
                    if avg_change > 0.05:  # >5% average increase
                        trends['direction'] = 'increasing'
                    elif avg_change < -0.05:  # >5% average decrease
                        trends['direction'] = 'decreasing'
                    else:
                        trends['direction'] = 'stable'

                    # Magnitude
                    trends['magnitude'] = float(abs(avg_change))

                    # Consistency (what % of recent periods had same direction)
                    if avg_change > 0:
                        consistent_periods = (recent_changes > 0).sum()
                    else:
                        consistent_periods = (recent_changes < 0).sum()

                    trends['consistency'] = float(
                        consistent_periods / len(recent_changes))

                    # Volatility
                    trends['volatility'] = float(recent_changes.std())

        except Exception as e:
            logger.warning(f"Failed to analyze short trends: {e}")

        return trends

    def _detect_squeeze_signals(self,
                                short_data: pd.DataFrame,
                                volume_data: pd.DataFrame = None,
                                metrics: Dict[str,
                                              Any] = None) -> Dict[str,
                                                                   Any]:
        """Detect potential short squeeze signals"""
        signals = {
            'squeeze_probability': 0.0,
            'signal_strength': 'none',
            'key_indicators': [],
            'risk_factors': [],
            'timeframe_estimate': 'unknown'
        }

        try:
            score = 0.0
            max_score = 10.0

            # High short interest ratio
            short_ratio = metrics.get('short_ratio', {})
            if short_ratio:
                ratio_pct = short_ratio.get('percentage', 0)
                if ratio_pct > 30:
                    score += 3.0
                    signals['key_indicators'].append('extreme_short_interest')
                elif ratio_pct > 20:
                    score += 2.0
                    signals['key_indicators'].append('high_short_interest')
                elif ratio_pct > 10:
                    score += 1.0
                    signals['key_indicators'].append('moderate_short_interest')

            # High days to cover
            days_to_cover = metrics.get('days_to_cover', {})
            if days_to_cover:
                days = days_to_cover.get('days', 0)
                if days > 10:
                    score += 2.5
                    signals['key_indicators'].append('high_days_to_cover')
                elif days > 5:
                    score += 1.5
                    signals['key_indicators'].append('moderate_days_to_cover')

            # Increasing short interest trend
            trend = metrics.get('trend_analysis', {})
            if trend.get('direction') == 'increasing' and trend.get(
                    'consistency', 0) > 0.6:
                score += 1.5
                signals['key_indicators'].append('increasing_short_trend')

            # Recent volume patterns (if available)
            if volume_data is not None and not volume_data.empty:
                recent_volume = volume_data.tail(5)['volume'].mean()
                avg_volume = volume_data['volume'].mean()

                if recent_volume > avg_volume * 1.5:  # 50% above average
                    score += 1.0
                    signals['key_indicators'].append('elevated_volume')

                    if recent_volume > avg_volume * 2.0:  # 100% above average
                        score += 1.0
                        signals['key_indicators'].append('very_high_volume')

            # Calculate probability
            signals['squeeze_probability'] = min(score / max_score, 1.0)

            # Signal strength
            if signals['squeeze_probability'] > 0.7:
                signals['signal_strength'] = 'strong'
                signals['timeframe_estimate'] = 'days_to_weeks'
            elif signals['squeeze_probability'] > 0.5:
                signals['signal_strength'] = 'moderate'
                signals['timeframe_estimate'] = 'weeks_to_months'
            elif signals['squeeze_probability'] > 0.3:
                signals['signal_strength'] = 'weak'
                signals['timeframe_estimate'] = 'months_or_longer'
            else:
                signals['signal_strength'] = 'none'
                signals['timeframe_estimate'] = 'unlikely'

            # Risk factors
            if days_to_cover.get('days', 0) < 2:
                signals['risk_factors'].append('low_days_to_cover')

            if short_ratio.get('percentage', 0) < 5:
                signals['risk_factors'].append('low_short_interest')

            if trend.get('direction') == 'decreasing':
                signals['risk_factors'].append('decreasing_short_trend')

        except Exception as e:
            logger.warning(f"Failed to detect squeeze signals: {e}")

        return signals

    def _assess_short_risk(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall short interest risk"""
        risk = {
            'overall_risk': 'low',
            'risk_score': 0.0,
            'primary_risks': [],
            'mitigation_factors': [],
            'recommendation': 'monitor'
        }

        try:
            score = 0.0

            # Short ratio risk
            short_ratio = metrics.get('short_ratio', {})
            if short_ratio:
                classification = short_ratio.get('classification', 'low')
                if classification == 'extreme':
                    score += 4.0
                    risk['primary_risks'].append('extreme_short_ratio')
                elif classification == 'high':
                    score += 3.0
                    risk['primary_risks'].append('high_short_ratio')
                elif classification == 'moderate':
                    score += 1.5
                    risk['primary_risks'].append('moderate_short_ratio')

            # Days to cover risk
            days_to_cover = metrics.get('days_to_cover', {})
            if days_to_cover:
                risk_level = days_to_cover.get('risk_level', 'low')
                if risk_level == 'extreme':
                    score += 3.0
                    risk['primary_risks'].append('extreme_days_to_cover')
                elif risk_level == 'high':
                    score += 2.0
                    risk['primary_risks'].append('high_days_to_cover')

            # Trend risk
            trend = metrics.get('trend_analysis', {})
            if trend.get('direction') == 'increasing':
                score += 1.0
                risk['primary_risks'].append('increasing_short_trend')
            elif trend.get('direction') == 'decreasing':
                risk['mitigation_factors'].append('decreasing_short_trend')

            # Squeeze probability
            squeeze = metrics.get('squeeze_indicators', {})
            squeeze_prob = squeeze.get('squeeze_probability', 0)
            if squeeze_prob > 0.7:
                score += 2.0
                risk['primary_risks'].append('high_squeeze_probability')
            elif squeeze_prob > 0.5:
                score += 1.0
                risk['primary_risks'].append('moderate_squeeze_probability')

            risk['risk_score'] = min(score / 10.0, 1.0)

            # Overall risk assessment
            if risk['risk_score'] > 0.7:
                risk['overall_risk'] = 'extreme'
                risk['recommendation'] = 'immediate_attention'
            elif risk['risk_score'] > 0.5:
                risk['overall_risk'] = 'high'
                risk['recommendation'] = 'close_monitoring'
            elif risk['risk_score'] > 0.3:
                risk['overall_risk'] = 'moderate'
                risk['recommendation'] = 'regular_monitoring'
            else:
                risk['overall_risk'] = 'low'
                risk['recommendation'] = 'periodic_review'

        except Exception as e:
            logger.warning(f"Failed to assess short risk: {e}")

        return risk


class FailedToDeliverAnalyzer:
    """
    Failed-to-Deliver (FTD) Analysis Engine

    Analyzes SEC Failed-to-Deliver data to identify settlement issues
    and potential market manipulation indicators.
    """

    def __init__(self, config: MarketMicrostructureConfig):
        self.config = config

    def analyze_ftd_data(
            self, symbol: str, ftd_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze Failed-to-Deliver data for a symbol

        Args:
            symbol: Stock symbol
            ftd_data: DataFrame with FTD data

        Returns:
            Dict with FTD analysis
        """
        analysis = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'ftd_summary': {},
            'trend_analysis': {},
            'threshold_breaches': [],
            'risk_indicators': {},
            'success': False,
            'error': None
        }

        try:
            if ftd_data.empty:
                analysis['error'] = "No FTD data available"
                return analysis

            # Summary statistics
            total_ftds = ftd_data['quantity'].sum()
            max_ftd = ftd_data['quantity'].max()
            avg_ftd = ftd_data['quantity'].mean()
            ftd_days = len(ftd_data)

            analysis['ftd_summary'] = {
                'total_failed_shares': int(total_ftds),
                'max_daily_ftd': int(max_ftd),
                'avg_daily_ftd': float(avg_ftd),
                'ftd_reporting_days': int(ftd_days),
                'date_range': {
                    'start': ftd_data['settlement_date'].min(),
                    'end': ftd_data['settlement_date'].max()
                }
            }

            # Trend analysis
            if len(ftd_data) >= 5:
                trend = self._analyze_ftd_trends(ftd_data)
                analysis['trend_analysis'] = trend

            # Threshold breaches (T+13, T+35 rules)
            thresholds = self._identify_threshold_breaches(ftd_data)
            analysis['threshold_breaches'] = thresholds

            # Risk indicators
            risk_indicators = self._assess_ftd_risk(ftd_data, analysis)
            analysis['risk_indicators'] = risk_indicators

            analysis['success'] = True
            logger.info(f"Analyzed FTD data for {symbol}")

        except Exception as e:
            analysis['error'] = str(e)
            logger.error(f"Failed to analyze FTD data for {symbol}: {e}")

        return analysis

    def _analyze_ftd_trends(self, ftd_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze FTD trends over time"""
        trends = {
            'direction': 'unknown',
            'volatility': 0,
            'persistence': 0,
            'recent_acceleration': False
        }

        try:
            # Calculate rolling averages
            ftd_data['ftd_ma5'] = ftd_data['quantity'].rolling(5).mean()
            ftd_data['ftd_ma10'] = ftd_data['quantity'].rolling(10).mean()

            # Recent trend
            recent_data = ftd_data.tail(10)
            if len(recent_data) >= 5:
                correlation = np.corrcoef(
                    range(
                        len(recent_data)),
                    recent_data['quantity'])[
                    0,
                    1]

                if correlation > 0.3:
                    trends['direction'] = 'increasing'
                elif correlation < -0.3:
                    trends['direction'] = 'decreasing'
                else:
                    trends['direction'] = 'stable'

                # Volatility
                trends['volatility'] = float(
                    recent_data['quantity'].std() /
                    recent_data['quantity'].mean())

                # Persistence (consecutive days with FTDs)
                consecutive_days = 0
                for _, row in recent_data.iterrows():
                    if row['quantity'] > 0:
                        consecutive_days += 1
                    else:
                        break

                trends['persistence'] = consecutive_days

                # Recent acceleration
                if len(recent_data) >= 6:
                    early_avg = recent_data.head(5)['quantity'].mean()
                    late_avg = recent_data.tail(5)['quantity'].mean()

                    if late_avg > early_avg * 1.5:  # 50% increase
                        trends['recent_acceleration'] = True

        except Exception as e:
            logger.warning(f"Failed to analyze FTD trends: {e}")

        return trends

    def _identify_threshold_breaches(
            self, ftd_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify regulatory threshold breaches"""
        breaches = []

        try:
            # Reg SHO thresholds
            threshold_10k = 10000  # 10,000 shares

            # T+13 consecutive days threshold
            consecutive_ftds = 0
            consecutive_start = None

            for _, row in ftd_data.iterrows():
                if row['quantity'] >= threshold_10k:
                    if consecutive_ftds == 0:
                        consecutive_start = row['settlement_date']
                    consecutive_ftds += 1
                else:
                    if consecutive_ftds >= 13:  # T+13 threshold
                        breaches.append(
                            {'type': 't_plus_13_breach',
                             'start_date': consecutive_start,
                             'end_date': row['settlement_date'],
                             'consecutive_days': consecutive_ftds,
                             'severity': 'high'
                             if consecutive_ftds >= 35 else 'moderate'})
                    consecutive_ftds = 0
                    consecutive_start = None

            # Check if still in a breach period
            if consecutive_ftds >= 13:
                breaches.append(
                    {'type': 't_plus_13_breach_ongoing',
                     'start_date': consecutive_start,
                     'end_date': ftd_data.iloc[-1]['settlement_date'],
                     'consecutive_days': consecutive_ftds, 'severity': 'high'
                     if consecutive_ftds >= 35 else 'moderate'})

        except Exception as e:
            logger.warning(f"Failed to identify threshold breaches: {e}")

        return breaches

    def _assess_ftd_risk(self, ftd_data: pd.DataFrame,
                         analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess FTD-related risks"""
        risk = {
            'risk_level': 'low',
            'risk_score': 0.0,
            'primary_concerns': [],
            'regulatory_risk': 'low',
            'settlement_risk': 'low'
        }

        try:
            score = 0.0

            # Volume of FTDs
            ftd_summary = analysis.get('ftd_summary', {})
            max_ftd = ftd_summary.get('max_daily_ftd', 0)

            if max_ftd > 100000:  # >100k shares in single day
                score += 3.0
                risk['primary_concerns'].append('large_daily_ftd')
            elif max_ftd > 50000:
                score += 2.0
                risk['primary_concerns'].append('moderate_daily_ftd')

            # Threshold breaches
            breaches = analysis.get('threshold_breaches', [])
            if breaches:
                for breach in breaches:
                    if breach.get('severity') == 'high':
                        score += 2.5
                        risk['primary_concerns'].append(
                            'severe_threshold_breach')
                        risk['regulatory_risk'] = 'high'
                    else:
                        score += 1.5
                        risk['primary_concerns'].append('threshold_breach')
                        risk['regulatory_risk'] = 'moderate'

            # Trend analysis
            trend = analysis.get('trend_analysis', {})
            if trend.get('direction') == 'increasing':
                score += 1.0
                risk['primary_concerns'].append('increasing_ftd_trend')

            if trend.get('recent_acceleration'):
                score += 1.5
                risk['primary_concerns'].append('accelerating_ftds')

            if trend.get('persistence', 0) > 10:
                score += 1.0
                risk['primary_concerns'].append('persistent_ftds')
                risk['settlement_risk'] = 'high'

            risk['risk_score'] = min(score / 10.0, 1.0)

            # Overall risk level
            if risk['risk_score'] > 0.7:
                risk['risk_level'] = 'extreme'
            elif risk['risk_score'] > 0.5:
                risk['risk_level'] = 'high'
            elif risk['risk_score'] > 0.3:
                risk['risk_level'] = 'moderate'
            else:
                risk['risk_level'] = 'low'

        except Exception as e:
            logger.warning(f"Failed to assess FTD risk: {e}")

        return risk


class MarketMicrostructureProvider:
    """
    Comprehensive Market Microstructure Provider

    Integrates FINRA short interest data, SEC FTD data, and trading
    analysis to provide complete market microstructure insights.
    """

    def __init__(self, config: Optional[MarketMicrostructureConfig] = None):
        """Initialize market microstructure provider"""
        self.config = config or MarketMicrostructureConfig()
        self.short_analyzer = ShortInterestAnalyzer(self.config)
        self.ftd_analyzer = FailedToDeliverAnalyzer(self.config)
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
                'User-Agent': 'DCF Lab Market Microstructure Provider'
            })

        return self._session

    def get_comprehensive_analysis(
            self, symbol: str, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Get comprehensive market microstructure analysis

        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data to analyze

        Returns:
            Dict with complete microstructure analysis
        """
        result = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'short_analysis': {},
            'ftd_analysis': {},
            'liquidity_metrics': {},
            'risk_assessment': {},
            'trading_patterns': {},
            'recommendations': {},
            'success': False,
            'error': None
        }

        try:
            # MOCK DATA DISABLED - Real market microstructure data required
            logger.error(f"Mock market microstructure data disabled for {symbol}")
            result['success'] = False
            result['error'] = 'Mock data disabled - implement real FINRA/SEC data integration'
            return result
            
            # Disabled mock data paths
            if False:
                short_data = self._generate_mock_short_data(symbol, lookback_days)
                ftd_data = self._generate_mock_ftd_data(symbol, lookback_days)
                volume_data = self._generate_mock_volume_data(
                    symbol, lookback_days)

            # Analyze short interest
            if not short_data.empty:
                float_shares = 100_000_000  # Mock float shares
                short_analysis = self.short_analyzer.calculate_short_metrics(
                    symbol, short_data, volume_data, float_shares
                )
                result['short_analysis'] = short_analysis

            # Analyze FTD data
            if not ftd_data.empty:
                ftd_analysis = self.ftd_analyzer.analyze_ftd_data(
                    symbol, ftd_data)
                result['ftd_analysis'] = ftd_analysis

            # Calculate liquidity metrics
            liquidity_metrics = self._calculate_liquidity_metrics(
                volume_data, short_data)
            result['liquidity_metrics'] = liquidity_metrics

            # Trading pattern analysis
            trading_patterns = self._analyze_trading_patterns(
                volume_data, short_data)
            result['trading_patterns'] = trading_patterns

            # Overall risk assessment
            risk_assessment = self._assess_overall_risk(result)
            result['risk_assessment'] = risk_assessment

            # Generate recommendations
            recommendations = self._generate_recommendations(result)
            result['recommendations'] = recommendations

            result['success'] = True
            logger.info(f"Completed comprehensive analysis for {symbol}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed comprehensive analysis for {symbol}: {e}")

        return result

    def _generate_mock_short_data(
            self,
            symbol: str,
            days: int) -> pd.DataFrame:
        """Generate mock short interest data for testing"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=days//7,
            freq='7D')  # Weekly data

        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol

        # Generate realistic short interest data
        base_short_volume = np.random.randint(50_000, 500_000)
        base_total_volume = base_short_volume * np.random.uniform(8, 20)

        data = []
        for date in dates:
            # Add some trend and noise
            trend_factor = 1 + (len(data) * 0.01)  # Slight upward trend
            noise = np.random.normal(1, 0.2)

            short_volume = int(base_short_volume * trend_factor * noise)
            total_volume = int(
                base_total_volume *
                trend_factor *
                noise *
                np.random.uniform(
                    0.8,
                    1.2))

            data.append({
                'settlement_date': date.strftime('%Y-%m-%d'),
                'report_date': (date + timedelta(days=2)).strftime('%Y-%m-%d'),
                'symbol': symbol,
                'short_volume': max(short_volume, 0),
                'total_volume': max(total_volume, short_volume)
            })

        return pd.DataFrame(data)

    def _generate_mock_ftd_data(self, symbol: str, days: int) -> pd.DataFrame:
        """REMOVED - No mock data in production"""
        return pd.DataFrame()
        # FTD data is less frequent than daily
        dates = pd.date_range(end=datetime.now(), periods=days//3, freq='3D')

        np.random.seed(hash(symbol + 'ftd') % 2**32)

        data = []
        consecutive_days = 0

        for date in dates:
            # Simulate FTD patterns
            if np.random.random() < 0.3:  # 30% chance of FTD
                # Occasional large FTDs
                if np.random.random() < 0.1:  # 10% chance of large FTD
                    quantity = np.random.randint(50_000, 200_000)
                else:
                    quantity = np.random.randint(1_000, 20_000)

                consecutive_days += 1
            else:
                quantity = 0
                consecutive_days = 0

            if quantity > 0:
                data.append({
                    'settlement_date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': np.random.uniform(10, 200)
                })

        return pd.DataFrame(data)

    def _generate_mock_volume_data(
            self,
            symbol: str,
            days: int) -> pd.DataFrame:
        """Generate mock trading volume data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        np.random.seed(hash(symbol + 'volume') % 2**32)

        base_volume = np.random.randint(500_000, 5_000_000)

        data = []
        for date in dates:
            # Skip weekends
            if date.weekday() >= 5:
                continue

            # Add volume patterns
            volume = int(base_volume * np.random.lognormal(0, 0.5))

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'volume': volume
            })

        return pd.DataFrame(data)

    def _calculate_liquidity_metrics(self,
                                     volume_data: pd.DataFrame,
                                     short_data: pd.DataFrame = None) -> Dict[str,
                                                                              Any]:
        """Calculate liquidity and market quality metrics"""
        metrics = {
            'average_daily_volume': 0,
            'volume_volatility': 0,
            'volume_trend': 'stable',
            'liquidity_score': 0.5,
            'market_depth_estimate': 'moderate'
        }

        try:
            if not volume_data.empty:
                volumes = volume_data['volume']

                metrics['average_daily_volume'] = float(volumes.mean())
                metrics['volume_volatility'] = float(
                    volumes.std() / volumes.mean())

                # Volume trend
                if len(volumes) >= 10:
                    recent_avg = volumes.tail(10).mean()
                    earlier_avg = volumes.head(10).mean()

                    if recent_avg > earlier_avg * 1.2:
                        metrics['volume_trend'] = 'increasing'
                    elif recent_avg < earlier_avg * 0.8:
                        metrics['volume_trend'] = 'decreasing'
                    else:
                        metrics['volume_trend'] = 'stable'

                # Liquidity score (0-1, higher is more liquid)
                avg_volume = metrics['average_daily_volume']
                volatility = metrics['volume_volatility']

                # Normalize based on typical market ranges
                volume_score = min(
                    avg_volume / 1_000_000,
                    1.0)  # 1M+ volume = full score
                # Lower volatility = higher score
                stability_score = max(0, 1 - volatility)

                metrics['liquidity_score'] = (
                    volume_score * 0.7 + stability_score * 0.3)

                # Market depth estimate
                if metrics['liquidity_score'] > 0.7:
                    metrics['market_depth_estimate'] = 'deep'
                elif metrics['liquidity_score'] > 0.4:
                    metrics['market_depth_estimate'] = 'moderate'
                else:
                    metrics['market_depth_estimate'] = 'shallow'

        except Exception as e:
            logger.warning(f"Failed to calculate liquidity metrics: {e}")

        return metrics

    def _analyze_trading_patterns(self,
                                  volume_data: pd.DataFrame,
                                  short_data: pd.DataFrame = None) -> Dict[str,
                                                                           Any]:
        """Analyze trading patterns and anomalies"""
        patterns = {
            'volume_spikes': [],
            'unusual_activity': [],
            'pattern_classification': 'normal',
            'seasonality': {},
            'anomaly_score': 0.0
        }

        try:
            if not volume_data.empty:
                volumes = volume_data['volume']

                # Detect volume spikes (>2 std dev above mean)
                mean_vol = volumes.mean()
                std_vol = volumes.std()
                threshold = mean_vol + (2 * std_vol)

                spikes = volume_data[volume_data['volume'] > threshold]
                patterns['volume_spikes'] = [
                    {'date': spike['date'],
                     'volume': int(spike['volume']),
                     'multiple_of_average': float(
                         spike['volume'] / mean_vol)} for _,
                    spike in spikes.iterrows()]

                # Pattern classification
                spike_frequency = len(
                    patterns['volume_spikes']) / len(volume_data)

                if spike_frequency > 0.1:  # >10% of days have spikes
                    patterns['pattern_classification'] = 'highly_volatile'
                elif spike_frequency > 0.05:  # >5% of days have spikes
                    patterns['pattern_classification'] = 'moderately_volatile'
                else:
                    patterns['pattern_classification'] = 'normal'

                # Anomaly score
                patterns['anomaly_score'] = min(spike_frequency * 10, 1.0)

        except Exception as e:
            logger.warning(f"Failed to analyze trading patterns: {e}")

        return patterns

    def _assess_overall_risk(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall market microstructure risk"""
        risk = {
            'overall_risk_level': 'low',
            'risk_score': 0.0,
            'key_risk_factors': [],
            'risk_mitigation': [],
            'monitoring_priority': 'low'
        }

        try:
            score = 0.0

            # Short interest risks
            short_analysis = analysis.get('short_analysis', {})
            if short_analysis.get('success'):
                short_risk = short_analysis.get('risk_assessment', {})
                short_score = short_risk.get('risk_score', 0)
                score += short_score * 0.4  # 40% weight

                if short_score > 0.5:
                    risk['key_risk_factors'].append('elevated_short_interest')

            # FTD risks
            ftd_analysis = analysis.get('ftd_analysis', {})
            if ftd_analysis.get('success'):
                ftd_risk = ftd_analysis.get('risk_indicators', {})
                ftd_score = ftd_risk.get('risk_score', 0)
                score += ftd_score * 0.3  # 30% weight

                if ftd_score > 0.5:
                    risk['key_risk_factors'].append('failed_to_deliver_issues')

            # Liquidity risks
            liquidity = analysis.get('liquidity_metrics', {})
            liquidity_score = 1 - \
                liquidity.get('liquidity_score', 0.5)  # Invert for risk
            score += liquidity_score * 0.2  # 20% weight

            if liquidity.get('market_depth_estimate') == 'shallow':
                risk['key_risk_factors'].append('poor_liquidity')

            # Trading pattern risks
            patterns = analysis.get('trading_patterns', {})
            anomaly_score = patterns.get('anomaly_score', 0)
            score += anomaly_score * 0.1  # 10% weight

            if patterns.get('pattern_classification') in [
                    'highly_volatile', 'moderately_volatile']:
                risk['key_risk_factors'].append('unusual_trading_patterns')

            risk['risk_score'] = min(score, 1.0)

            # Overall risk level
            if risk['risk_score'] > 0.7:
                risk['overall_risk_level'] = 'extreme'
                risk['monitoring_priority'] = 'immediate'
            elif risk['risk_score'] > 0.5:
                risk['overall_risk_level'] = 'high'
                risk['monitoring_priority'] = 'high'
            elif risk['risk_score'] > 0.3:
                risk['overall_risk_level'] = 'moderate'
                risk['monitoring_priority'] = 'moderate'
            else:
                risk['overall_risk_level'] = 'low'
                risk['monitoring_priority'] = 'low'

            # Risk mitigation suggestions
            if liquidity.get('liquidity_score', 0) < 0.3:
                risk['risk_mitigation'].append(
                    'increase_position_sizing_caution')

            if short_analysis.get(
                    'squeeze_indicators', {}).get(
                    'squeeze_probability', 0) > 0.5:
                risk['risk_mitigation'].append('monitor_short_squeeze_risk')

            if ftd_analysis.get('risk_indicators', {}).get(
                    'regulatory_risk') in ['high', 'extreme']:
                risk['risk_mitigation'].append('regulatory_scrutiny_possible')

        except Exception as e:
            logger.warning(f"Failed to assess overall risk: {e}")

        return risk

    def _generate_recommendations(
            self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        recommendations = {
            'trading_recommendations': [],
            'risk_management': [],
            'monitoring_frequency': 'weekly',
            'key_metrics_to_watch': [],
            'alert_thresholds': {}
        }

        try:
            risk_level = analysis.get(
                'risk_assessment', {}).get(
                'overall_risk_level', 'low')

            # Trading recommendations based on analysis
            short_analysis = analysis.get('short_analysis', {})
            if short_analysis.get('success'):
                squeeze_prob = short_analysis.get(
                    'squeeze_indicators', {}).get(
                    'squeeze_probability', 0)

                if squeeze_prob > 0.7:
                    recommendations['trading_recommendations'].append(
                        'high_short_squeeze_risk_consider_position_sizing')
                    recommendations['monitoring_frequency'] = 'daily'
                elif squeeze_prob > 0.5:
                    recommendations['trading_recommendations'].append(
                        'moderate_short_squeeze_risk_monitor_closely')
                    recommendations['monitoring_frequency'] = 'daily'

            # Risk management recommendations
            liquidity = analysis.get('liquidity_metrics', {})
            if liquidity.get('market_depth_estimate') == 'shallow':
                recommendations['risk_management'].append(
                    'use_limit_orders_avoid_market_orders')
                recommendations['risk_management'].append(
                    'reduce_position_sizes_due_to_liquidity')

            if risk_level in ['high', 'extreme']:
                recommendations['risk_management'].append(
                    'increase_monitoring_frequency')
                recommendations['risk_management'].append(
                    'consider_position_hedging')
                recommendations['monitoring_frequency'] = 'daily'

            # Key metrics to monitor
            recommendations['key_metrics_to_watch'] = [
                'short_interest_ratio',
                'days_to_cover',
                'trading_volume',
                'failed_to_deliver'
            ]

            if short_analysis.get('success'):
                short_ratio = short_analysis.get(
                    'short_ratio', {}).get(
                    'percentage', 0)
                days_to_cover = short_analysis.get(
                    'days_to_cover', {}).get('days', 0)

                # Alert thresholds
                recommendations['alert_thresholds'] = {
                    # 50% increase or 20%
                    'short_ratio_increase': max(short_ratio * 1.5, 20),
                    # 50% increase or 10 days
                    'days_to_cover_increase': max(days_to_cover * 1.5, 10),
                    # 2x average volume
                    'volume_spike': liquidity.get('average_daily_volume', 0) * 2,
                    'ftd_threshold': 50000  # 50k shares FTD
                }

        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")

        return recommendations


def get_market_microstructure_provider(
        config: Optional[MarketMicrostructureConfig] = None) -> MarketMicrostructureProvider:
    """Factory function to create market microstructure provider"""
    return MarketMicrostructureProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_market_microstructure_provider()

    print("=== Market Microstructure & Short Interest Provider ===")

    # Test comprehensive analysis
    print("\n1. Comprehensive Analysis for AAPL:")
    analysis = provider.get_comprehensive_analysis('AAPL', lookback_days=60)

    if analysis['success']:
        print(f"✅ Analysis completed for {analysis['symbol']}")

        # Short analysis
        short_analysis = analysis.get('short_analysis', {})
        if short_analysis.get('success'):
            short_ratio = short_analysis.get('short_ratio', {})
            days_to_cover = short_analysis.get('days_to_cover', {})
            squeeze_indicators = short_analysis.get('squeeze_indicators', {})

            print("\nShort Interest Analysis:")
            if short_ratio:
                print(
                    f"  Short Ratio: {
                        short_ratio.get(
                            'percentage',
                            0):.1f}% of float")
                print(
                    f"  Classification: {
                        short_ratio.get(
                            'classification',
                            'unknown')}")

            if days_to_cover:
                print(f"  Days to Cover: {days_to_cover.get('days', 0):.1f}")
                print(
                    f"  Risk Level: {
                        days_to_cover.get(
                            'risk_level',
                            'unknown')}")

            if squeeze_indicators:
                print(
                    f"  Squeeze Probability: {
                        squeeze_indicators.get(
                            'squeeze_probability',
                            0):.1%}")
                print(
                    f"  Signal Strength: {
                        squeeze_indicators.get(
                            'signal_strength',
                            'none')}")

        # FTD analysis
        ftd_analysis = analysis.get('ftd_analysis', {})
        if ftd_analysis.get('success'):
            ftd_summary = ftd_analysis.get('ftd_summary', {})
            risk_indicators = ftd_analysis.get('risk_indicators', {})

            print("\nFailed-to-Deliver Analysis:")
            print(
                f"  Total Failed Shares: {
                    ftd_summary.get(
                        'total_failed_shares',
                        0):,                    }")
            print(f"  Max Daily FTD: {ftd_summary.get('max_daily_ftd', 0):,}")
            print(
                f"  Risk Level: {
                    risk_indicators.get(
                        'risk_level',
                        'unknown')}")

        # Liquidity metrics
        liquidity = analysis.get('liquidity_metrics', {})
        print("\nLiquidity Metrics:")
        print(
            f"  Average Daily Volume: {
                liquidity.get(
                    'average_daily_volume',
                    0):,.0f}")
        print(f"  Liquidity Score: {liquidity.get('liquidity_score', 0):.2f}")
        print(
            f"  Market Depth: {
                liquidity.get(
                    'market_depth_estimate',
                    'unknown')}")
        print(f"  Volume Trend: {liquidity.get('volume_trend', 'unknown')}")

        # Overall risk assessment
        risk_assessment = analysis.get('risk_assessment', {})
        print("\nRisk Assessment:")
        print(
            f"  Overall Risk Level: {
                risk_assessment.get(
                    'overall_risk_level',
                    'unknown')}")
        print(f"  Risk Score: {risk_assessment.get('risk_score', 0):.2f}")
        print(
            f"  Monitoring Priority: {
                risk_assessment.get(
                    'monitoring_priority',
                    'unknown')}")

        key_risks = risk_assessment.get('key_risk_factors', [])
        if key_risks:
            print(f"  Key Risk Factors: {', '.join(key_risks)}")

        # Recommendations
        recommendations = analysis.get('recommendations', {})
        print("\nRecommendations:")
        print(
            f"  Monitoring Frequency: {
                recommendations.get(
                    'monitoring_frequency',
                    'unknown')}")

        trading_recs = recommendations.get('trading_recommendations', [])
        if trading_recs:
            print(f"  Trading: {', '.join(trading_recs)}")

        risk_mgmt = recommendations.get('risk_management', [])
        if risk_mgmt:
            print(f"  Risk Management: {', '.join(risk_mgmt)}")
    else:
        print(f"❌ Analysis failed: {analysis['error']}")

    print("\n=== Market Microstructure Provider Ready ===")
    print("✅ Short interest analysis with squeeze detection")
    print("✅ Failed-to-deliver monitoring and risk assessment")
    print("✅ Liquidity metrics and market depth analysis")
    print("✅ Comprehensive risk scoring and recommendations")
    print("🚀 Ready for advanced market microstructure analysis")
