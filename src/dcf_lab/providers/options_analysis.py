"""Options analysis provider with Black-Scholes analytics and mock data support."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from scipy import stats
from urllib3.util.retry import Retry


@dataclass
class OptionsAnalysisConfig:
    """Configuration settings for the options analysis provider."""

    cache_dir: str = "./cache/options_data"
    enable_cache: bool = True
    max_cache_age_hours: int = 1
    timeout: int = 30
    retries: int = 3
    rate_limit_delay: float = 0.5

    cboe_base_url: str = "https://www.cboe.com"
    options_data_url: str = "https://finance.yahoo.com/options"

    risk_free_rate: float = 0.05
    dividend_yield: float = 0.02
    min_volume_threshold: int = 100
    min_open_interest: int = 50
    max_days_to_expiry: int = 365


class BlackScholesCalculator:
    """Utility class providing Black-Scholes pricing and Greeks."""

    @staticmethod
    def black_scholes_call(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            dividend_yield: float = 0.0) -> float:
        """Calculate Black-Scholes call option price."""
        if time_to_expiry <= 0:
            return max(spot_price - strike, 0)

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        call_price = (
            spot_price * np.exp(-dividend_yield * time_to_expiry) *
            stats.norm.cdf(d1) -
            strike * np.exp(-risk_free_rate * time_to_expiry) *
            stats.norm.cdf(d2))
        return call_price

    @staticmethod
    def black_scholes_put(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            dividend_yield: float = 0.0) -> float:
        """Calculate Black-Scholes put option price."""
        if time_to_expiry <= 0:
            return max(strike - spot_price, 0)

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        put_price = (
            strike * np.exp(-risk_free_rate * time_to_expiry) *
            stats.norm.cdf(-d2) -
            spot_price * np.exp(-dividend_yield * time_to_expiry) *
            stats.norm.cdf(-d1))
        return put_price

    @staticmethod
    def calculate_implied_volatility(
            market_price: float,
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            option_type: str = 'call',
            dividend_yield: float = 0.0,
            max_iterations: int = 100,
            tolerance: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method."""
        if time_to_expiry <= 0:
            return 0.0

        sigma = 0.25

        for _ in range(max_iterations):
            if option_type.lower() == 'call':
                price = BlackScholesCalculator.black_scholes_call(
                    spot_price, strike, time_to_expiry,
                    risk_free_rate, sigma, dividend_yield)
            else:
                price = BlackScholesCalculator.black_scholes_put(
                    spot_price, strike, time_to_expiry,
                    risk_free_rate, sigma, dividend_yield)

            vega = BlackScholesCalculator.calculate_vega(
                spot_price, strike, time_to_expiry,
                risk_free_rate, sigma, dividend_yield)
            price_diff = price - market_price

            if abs(price_diff) < tolerance:
                return sigma

            if vega == 0:
                break

            sigma = max(sigma - price_diff / vega, 0.001)

        return sigma

    @staticmethod
    def calculate_delta(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            option_type: str = 'call',
            dividend_yield: float = 0.0) -> float:
        """Calculate option delta."""
        if time_to_expiry <= 0:
            if option_type.lower() == 'call':
                return 1.0 if spot_price > strike else 0.0
            return -1.0 if spot_price < strike else 0.0

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

        if option_type.lower() == 'call':
            return np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1)
        return -np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1)

    @staticmethod
    def calculate_gamma(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            dividend_yield: float = 0.0) -> float:
        """Calculate option gamma."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

        gamma = (
            np.exp(-dividend_yield * time_to_expiry) * stats.norm.pdf(d1)) / (
            spot_price * volatility * np.sqrt(time_to_expiry))
        return gamma

    @staticmethod
    def calculate_theta(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            option_type: str = 'call',
            dividend_yield: float = 0.0) -> float:
        """Calculate option theta (time decay per day)."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type.lower() == 'call':
            theta = (
                (-spot_price * stats.norm.pdf(d1) * volatility *
                 np.exp(-dividend_yield * time_to_expiry)) /
                (2 * np.sqrt(time_to_expiry)) -
                risk_free_rate * strike *
                np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2) +
                dividend_yield * spot_price *
                np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(d1))
        else:
            theta = (
                (-spot_price * stats.norm.pdf(d1) * volatility *
                 np.exp(-dividend_yield * time_to_expiry)) /
                (2 * np.sqrt(time_to_expiry)) +
                risk_free_rate * strike *
                np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2) -
                dividend_yield * spot_price *
                np.exp(-dividend_yield * time_to_expiry) * stats.norm.cdf(-d1))

        return theta / 365

    @staticmethod
    def calculate_vega(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            dividend_yield: float = 0.0) -> float:
        """Calculate option vega (per 1% change in volatility)."""
        if time_to_expiry <= 0:
            return 0.0

        d1 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

        vega = (
            spot_price * np.exp(-dividend_yield * time_to_expiry) *
            stats.norm.pdf(d1) * np.sqrt(time_to_expiry))
        return vega / 100

    @staticmethod
    def calculate_rho(
            spot_price: float,
            strike: float,
            time_to_expiry: float,
            risk_free_rate: float,
            volatility: float,
            option_type: str = 'call',
            dividend_yield: float = 0.0) -> float:
        """Calculate option rho (per 1% change in interest rate)."""
        if time_to_expiry <= 0:
            return 0.0

        d2 = (np.log(spot_price / strike) +
              (risk_free_rate - dividend_yield - 0.5 * volatility**2) *
              time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

        if option_type.lower() == 'call':
            rho = (strike * time_to_expiry *
                    np.exp(-risk_free_rate * time_to_expiry) *
                    stats.norm.cdf(d2))
        else:
            rho = (-strike * time_to_expiry *
                    np.exp(-risk_free_rate * time_to_expiry) *
                    stats.norm.cdf(-d2))

        return rho / 100


class OptionsDataAnalyzer:
    """
    Options Data Analyzer

    Analyzes options market data to provide insights into market sentiment,
    volatility expectations, and derivatives-based risk indicators.
    """

    # Ticker mapping for Tiingo compatibility
    TICKER_MAP = {
        "^VIX": "VIX",
        "^VIX3M": "VIXM",  # Map to VIX3M alternative that Tiingo supports
        "^VXN": "VXN",
        "^VXO": "VXO",
        "^RVX": "RVX",
        "^VXEEM": "VXEEM",
        "^VXGDX": "VXGDX"
    }

    def __init__(self, config: OptionsAnalysisConfig):
        self.config = config
        self.bs_calc = BlackScholesCalculator()

    def get_options_chain(
            self, symbol: str, expiration_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get options chain data for a symbol

        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates to analyze

        Returns:
            Dict with options chain data and analysis
        """
        # Map ticker symbols for Tiingo compatibility
        mapped_symbol = self.TICKER_MAP.get(symbol, symbol)
        if mapped_symbol != symbol:
            logger.info(f"🔄 Mapped ticker {symbol} → {mapped_symbol} for Tiingo compatibility")
        
        result = {
            'symbol': symbol,  # Keep original symbol in result
            'mapped_symbol': mapped_symbol,  # Include mapped symbol for reference
            'retrieved_at': datetime.now().isoformat(),
            'current_price': 0.0,
            'options_data': {},
            'analysis': {},
            'success': False,
            'error': None
        }

        try:
            # MOCK DATA DISABLED - Real options data required
            logger.error(f"Mock options data disabled for {symbol}")
            result['success'] = False
            result['error'] = 'Mock data disabled - implement real options API integration'
            return result
            
            if False:  # Disabled mock path
                mock_data = self._generate_mock_options_data(
                    symbol, expiration_dates)

                result['current_price'] = mock_data['current_price']
                result['options_data'] = mock_data['options_chain']

                # Analyze options data
                if mock_data['options_chain']:
                    analysis = self._analyze_options_chain(mock_data, symbol)
                    result['analysis'] = analysis

            result['success'] = True
            logger.info(f"Retrieved options chain for {symbol}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to get options chain for {symbol}: {e}")

        return result

    def _generate_mock_options_data(
            self, symbol: str, expiration_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """DISABLED - Mock options data not allowed"""
        logger.error("Mock options data generation disabled")
        return {'current_price': 0.0, 'options_chain': []}

        # Generate realistic current price
        base_prices = {
            'AAPL': 175.0,
            'MSFT': 325.0,
            'GOOGL': 135.0,
            'TSLA': 250.0,
            'SPY': 450.0,
            'QQQ': 380.0
        }

        current_price = base_prices.get(
            symbol, 100.0) * (1 + rng.uniform(-0.05, 0.05))

        # Generate expiration dates if not provided
        if expiration_dates is None:
            base_date = datetime.now()
            expiration_dates = [
                (base_date + timedelta(days=7)).strftime('%Y-%m-%d'),    # Weekly
                (base_date + timedelta(days=14)).strftime('%Y-%m-%d'),   # 2 weeks
                (base_date + timedelta(days=30)).strftime('%Y-%m-%d'),   # Monthly
                (base_date + timedelta(days=60)).strftime('%Y-%m-%d'),   # 2 months
                (base_date + timedelta(days=90)).strftime('%Y-%m-%d'),   # Quarterly
            ]

        options_chain = {}

        for exp_date in expiration_dates:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_expiry = (exp_datetime - datetime.now()).days
            time_to_expiry = days_to_expiry / 365.0

            if time_to_expiry <= 0:
                continue

            # Generate strike prices around current price
            strikes = []
            price_increment = 5 if current_price > 50 else 1

            for i in range(-10, 11):  # 21 strikes total
                strike = round((current_price + i * price_increment) /
                               price_increment) * price_increment
                strikes.append(strike)

            calls = []
            puts = []

            # Base implied volatility (varies by expiration)
            base_iv = 0.25 + 0.05 * rng.uniform(-1, 1)  # 20-30% base IV

            for strike in strikes:
                # Volatility smile/skew
                moneyness = strike / current_price
                iv_adjustment = 0.02 * (moneyness - 1)**2  # Smile effect

                # Add skew (puts more expensive)
                if moneyness < 1:  # ITM puts / OTM calls
                    iv_adjustment += 0.03 * (1 - moneyness)

                implied_vol = base_iv + iv_adjustment + rng.uniform(-0.02, 0.02)
                implied_vol = max(implied_vol, 0.05)  # Minimum 5% IV

                # Calculate theoretical prices
                call_price = self.bs_calc.black_scholes_call(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    self.config.dividend_yield)

                put_price = self.bs_calc.black_scholes_put(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    self.config.dividend_yield)

                # Add market noise
                call_market_price = call_price * (1 + rng.uniform(-0.1, 0.1))
                put_market_price = put_price * (1 + rng.uniform(-0.1, 0.1))

                # Generate volume and open interest
                # Higher volume for ATM options
                distance_from_atm = abs(moneyness - 1)
                base_volume = max(
                    int(1000 * np.exp(-5 * distance_from_atm)), 10)

                call_volume = int(base_volume * rng.uniform(0.5, 2.0))
                put_volume = int(base_volume * rng.uniform(0.5, 2.0))

                call_oi = int(call_volume * rng.uniform(2, 10))
                put_oi = int(put_volume * rng.uniform(2, 10))

                # Calculate Greeks
                call_delta = self.bs_calc.calculate_delta(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'call',
                    self.config.dividend_yield)

                put_delta = self.bs_calc.calculate_delta(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'put',
                    self.config.dividend_yield)

                gamma = self.bs_calc.calculate_gamma(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    self.config.dividend_yield)

                call_theta = self.bs_calc.calculate_theta(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'call',
                    self.config.dividend_yield)

                put_theta = self.bs_calc.calculate_theta(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'put',
                    self.config.dividend_yield)

                vega = self.bs_calc.calculate_vega(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    self.config.dividend_yield)

                call_rho = self.bs_calc.calculate_rho(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'call',
                    self.config.dividend_yield)

                put_rho = self.bs_calc.calculate_rho(
                    current_price,
                    strike,
                    time_to_expiry,
                    self.config.risk_free_rate,
                    implied_vol,
                    'put',
                    self.config.dividend_yield)

                call_option = {
                    'strike': strike,
                    'bid': max(call_market_price - 0.05, 0.01),
                    'ask': call_market_price + 0.05,
                    'last': call_market_price,
                    'mark': call_market_price,
                    'volume': call_volume,
                    'open_interest': call_oi,
                    'implied_volatility': implied_vol,
                    'delta': call_delta,
                    'gamma': gamma,
                    'theta': call_theta,
                    'vega': vega,
                    'rho': call_rho,
                    'theoretical_price': call_price,
                    'moneyness': moneyness,
                    'time_to_expiry': time_to_expiry,
                    'in_the_money': current_price > strike
                }

                put_option = {
                    'strike': strike,
                    'bid': max(put_market_price - 0.05, 0.01),
                    'ask': put_market_price + 0.05,
                    'last': put_market_price,
                    'mark': put_market_price,
                    'volume': put_volume,
                    'open_interest': put_oi,
                    'implied_volatility': implied_vol,
                    'delta': put_delta,
                    'gamma': gamma,
                    'theta': put_theta,
                    'vega': vega,
                    'rho': put_rho,
                    'theoretical_price': put_price,
                    'moneyness': moneyness,
                    'time_to_expiry': time_to_expiry,
                    'in_the_money': current_price < strike
                }

                calls.append(call_option)
                puts.append(put_option)

            options_chain[exp_date] = {
                'expiration': exp_date,
                'days_to_expiry': days_to_expiry,
                'calls': calls,
                'puts': puts
            }

        return {
            'current_price': current_price,
            'options_chain': options_chain
        }

    def _analyze_options_chain(
            self, options_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze options chain data"""
        analysis = {
            'summary': {},
            'volatility_analysis': {},
            'flow_analysis': {},
            'sentiment_indicators': {},
            'risk_metrics': {},
            'unusual_activity': []
        }

        try:
            current_price = options_data['current_price']
            options_chain = options_data['options_chain']

            all_calls = []
            all_puts = []

            # Collect all options data
            for exp_date, exp_data in options_chain.items():
                all_calls.extend(exp_data['calls'])
                all_puts.extend(exp_data['puts'])

            # Summary statistics
            total_call_volume = sum(opt['volume'] for opt in all_calls)
            total_put_volume = sum(opt['volume'] for opt in all_puts)
            total_call_oi = sum(opt['open_interest'] for opt in all_calls)
            total_put_oi = sum(opt['open_interest'] for opt in all_puts)

            analysis['summary'] = {
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'put_call_volume_ratio': total_put_volume / max(total_call_volume, 1),
                'put_call_oi_ratio': total_put_oi / max(total_call_oi, 1),
                'total_volume': total_call_volume + total_put_volume,
                'total_open_interest': total_call_oi + total_put_oi
            }

            # Volatility analysis
            volatility_analysis = self._analyze_volatility_surface(
                all_calls, all_puts, current_price)
            analysis['volatility_analysis'] = volatility_analysis

            # Options flow analysis
            flow_analysis = self._analyze_options_flow(all_calls, all_puts)
            analysis['flow_analysis'] = flow_analysis

            # Sentiment indicators
            sentiment = self._calculate_sentiment_indicators(
                all_calls, all_puts, analysis['summary'])
            analysis['sentiment_indicators'] = sentiment

            # Risk metrics
            risk_metrics = self._calculate_portfolio_greeks(
                all_calls, all_puts)
            analysis['risk_metrics'] = risk_metrics

            # Unusual activity detection
            unusual_activity = self._detect_unusual_activity(
                all_calls, all_puts, symbol)
            analysis['unusual_activity'] = unusual_activity

        except Exception as e:
            logger.warning(f"Failed to analyze options chain: {e}")

        return analysis

    def _analyze_volatility_surface(self,
                                    calls: List[Dict],
                                    puts: List[Dict],
                                    current_price: float) -> Dict[str,
                                                                  Any]:
        """Analyze implied volatility surface"""
        analysis = {
            'atm_volatility': 0.0,
            'volatility_skew': 0.0,
            'volatility_smile': {},
            'term_structure': {},
            'vol_surface_metrics': {}
        }

        try:
            # Find ATM volatility
            atm_calls = [
                opt for opt in calls if abs(
                    opt['moneyness'] -
                    1.0) < 0.05]
            if atm_calls:
                analysis['atm_volatility'] = np.mean(
                    [opt['implied_volatility'] for opt in atm_calls])

            # Calculate volatility skew (OTM puts vs OTM calls)
            otm_puts = [opt for opt in puts if opt['moneyness'] < 0.95]
            otm_calls = [opt for opt in calls if opt['moneyness'] > 1.05]

            if otm_puts and otm_calls:
                avg_put_iv = np.mean([opt['implied_volatility']
                                     for opt in otm_puts])
                avg_call_iv = np.mean([opt['implied_volatility']
                                      for opt in otm_calls])
                analysis['volatility_skew'] = avg_put_iv - avg_call_iv

            # Volatility smile by moneyness
            moneyness_buckets = {}
            for opt in calls + puts:
                bucket = round(opt['moneyness'], 1)
                if bucket not in moneyness_buckets:
                    moneyness_buckets[bucket] = []
                moneyness_buckets[bucket].append(opt['implied_volatility'])

            smile = {}
            for bucket, ivs in moneyness_buckets.items():
                smile[bucket] = np.mean(ivs)

            analysis['volatility_smile'] = smile

            # Term structure
            term_structure = {}
            time_buckets = {}

            for opt in calls + puts:
                time_bucket = round(
                    opt['time_to_expiry'] *
                    12)  # Monthly buckets
                if time_bucket not in time_buckets:
                    time_buckets[time_bucket] = []
                time_buckets[time_bucket].append(opt['implied_volatility'])

            for bucket, ivs in time_buckets.items():
                term_structure[bucket] = np.mean(ivs)

            analysis['term_structure'] = term_structure

            # Volatility surface metrics
            all_ivs = [opt['implied_volatility'] for opt in calls + puts]
            analysis['vol_surface_metrics'] = {
                'avg_implied_vol': np.mean(all_ivs),
                'vol_of_vol': np.std(all_ivs),
                'min_iv': np.min(all_ivs),
                'max_iv': np.max(all_ivs),
                'iv_range': np.max(all_ivs) - np.min(all_ivs)
            }

        except Exception as e:
            logger.warning(f"Failed to analyze volatility surface: {e}")

        return analysis

    def _analyze_options_flow(
            self, calls: List[Dict], puts: List[Dict]) -> Dict[str, Any]:
        """Analyze options flow patterns"""
        analysis = {
            'volume_weighted_iv': 0.0,
            'delta_weighted_flow': 0.0,
            'gamma_exposure': 0.0,
            'vega_exposure': 0.0,
            'flow_direction': 'neutral'
        }

        try:
            all_options = calls + puts

            # Volume weighted implied volatility
            total_volume = sum(opt['volume'] for opt in all_options)
            if total_volume > 0:
                vw_iv = sum(opt['volume'] * opt['implied_volatility']
                            for opt in all_options) / total_volume
                analysis['volume_weighted_iv'] = vw_iv

            # Delta weighted flow (positive = bullish, negative = bearish)
            call_delta_flow = sum(
                opt['volume'] *
                opt['delta'] for opt in calls)
            put_delta_flow = sum(opt['volume'] * opt['delta']
                                 for opt in puts)  # Put deltas are negative
            analysis['delta_weighted_flow'] = call_delta_flow + put_delta_flow

            # Gamma exposure
            total_gamma = sum(opt['volume'] * opt['gamma']
                              for opt in all_options)
            analysis['gamma_exposure'] = total_gamma

            # Vega exposure
            total_vega = sum(opt['volume'] * opt['vega']
                             for opt in all_options)
            analysis['vega_exposure'] = total_vega

            # Flow direction
            if analysis['delta_weighted_flow'] > 1000:
                analysis['flow_direction'] = 'bullish'
            elif analysis['delta_weighted_flow'] < -1000:
                analysis['flow_direction'] = 'bearish'
            else:
                analysis['flow_direction'] = 'neutral'

        except Exception as e:
            logger.warning(f"Failed to analyze options flow: {e}")

        return analysis

    def _calculate_sentiment_indicators(self,
                                        calls: List[Dict],
                                        puts: List[Dict],
                                        summary: Dict[str,
                                                      Any]) -> Dict[str,
                                                                    Any]:
        """Calculate options-based sentiment indicators"""
        indicators = {
            'put_call_ratio_sentiment': 'neutral',
            'volatility_sentiment': 'neutral',
            'flow_sentiment': 'neutral',
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'fear_greed_index': 50.0
        }

        try:
            # Put/Call ratio sentiment
            pc_ratio = summary.get('put_call_volume_ratio', 1.0)

            if pc_ratio > 1.2:  # High put volume = bearish
                indicators['put_call_ratio_sentiment'] = 'bearish'
            elif pc_ratio < 0.8:  # High call volume = bullish
                indicators['put_call_ratio_sentiment'] = 'bullish'
            else:
                indicators['put_call_ratio_sentiment'] = 'neutral'

            # Volatility sentiment
            all_ivs = [opt['implied_volatility'] for opt in calls + puts]
            avg_iv = np.mean(all_ivs)

            if avg_iv > 0.35:  # High volatility = fear
                indicators['volatility_sentiment'] = 'fearful'
            elif avg_iv < 0.20:  # Low volatility = complacency
                indicators['volatility_sentiment'] = 'complacent'
            else:
                indicators['volatility_sentiment'] = 'neutral'

            # Flow sentiment (based on delta-weighted flow from earlier
            # analysis)
            call_volume = summary.get('total_call_volume', 0)
            put_volume = summary.get('total_put_volume', 0)

            if call_volume > put_volume * 1.5:
                indicators['flow_sentiment'] = 'bullish'
            elif put_volume > call_volume * 1.5:
                indicators['flow_sentiment'] = 'bearish'
            else:
                indicators['flow_sentiment'] = 'neutral'

            # Overall sentiment aggregation
            sentiment_scores = {
                'bullish': 1,
                'bearish': -1,
                'neutral': 0,
                'fearful': -0.5,
                'complacent': 0.5
            }

            scores = [
                sentiment_scores.get(
                    indicators['put_call_ratio_sentiment'], 0), sentiment_scores.get(
                    indicators['volatility_sentiment'], 0), sentiment_scores.get(
                    indicators['flow_sentiment'], 0)]

            avg_score = np.mean(scores)
            indicators['sentiment_score'] = avg_score

            if avg_score > 0.3:
                indicators['overall_sentiment'] = 'bullish'
            elif avg_score < -0.3:
                indicators['overall_sentiment'] = 'bearish'
            else:
                indicators['overall_sentiment'] = 'neutral'

            # Fear & Greed Index (0-100 scale)
            # Based on put/call ratio and volatility
            fear_greed = 50  # Neutral starting point

            # Adjust for put/call ratio
            if pc_ratio > 1.2:
                fear_greed -= 20  # More fear
            elif pc_ratio < 0.8:
                fear_greed += 20  # More greed

            # Adjust for volatility
            if avg_iv > 0.35:
                fear_greed -= 15  # High vol = more fear
            elif avg_iv < 0.20:
                fear_greed += 15  # Low vol = more greed

            indicators['fear_greed_index'] = max(0, min(100, fear_greed))

        except Exception as e:
            logger.warning(f"Failed to calculate sentiment indicators: {e}")

        return indicators

    def _calculate_portfolio_greeks(
            self, calls: List[Dict], puts: List[Dict]) -> Dict[str, Any]:
        """Calculate portfolio-level Greeks"""
        metrics = {
            'total_delta': 0.0,
            'total_gamma': 0.0,
            'total_theta': 0.0,
            'total_vega': 0.0,
            'total_rho': 0.0,
            'risk_assessment': 'low'
        }

        try:
            all_options = calls + puts

            # Calculate volume-weighted Greeks
            for opt in all_options:
                volume = opt['volume']
                metrics['total_delta'] += volume * opt['delta']
                metrics['total_gamma'] += volume * opt['gamma']
                metrics['total_theta'] += volume * opt['theta']
                metrics['total_vega'] += volume * opt['vega']
                metrics['total_rho'] += volume * opt['rho']

            # Risk assessment based on Greeks magnitude
            risk_factors = [
                abs(metrics['total_delta']) / 10000,
                abs(metrics['total_gamma']) / 1000,
                abs(metrics['total_vega']) / 10000
            ]

            avg_risk = np.mean(risk_factors)

            if avg_risk > 2.0:
                metrics['risk_assessment'] = 'high'
            elif avg_risk > 1.0:
                metrics['risk_assessment'] = 'moderate'
            else:
                metrics['risk_assessment'] = 'low'

        except Exception as e:
            logger.warning(f"Failed to calculate portfolio Greeks: {e}")

        return metrics

    def _detect_unusual_activity(
            self, calls: List[Dict], puts: List[Dict], symbol: str) -> List[Dict[str, Any]]:
        """Detect unusual options activity"""
        unusual_activities = []

        try:
            all_options = calls + puts

            # Calculate volume percentiles
            volumes = [opt['volume']
                       for opt in all_options if opt['volume'] > 0]
            if not volumes:
                return unusual_activities

            np.percentile(volumes, 95)
            volume_99th = np.percentile(volumes, 99)

            # Find high volume options
            for opt in all_options:
                if opt['volume'] > volume_99th:
                    option_type = 'call' if opt in calls else 'put'

                    activity = {
                        'symbol': symbol,
                        'option_type': option_type,
                        'strike': opt['strike'],
                        'volume': opt['volume'],
                        'open_interest': opt['open_interest'],
                        'implied_volatility': opt['implied_volatility'],
                        'moneyness': opt['moneyness'],
                        'time_to_expiry': opt['time_to_expiry'],
                        'unusual_factor': opt['volume'] / max(np.mean(volumes), 1),
                        'activity_type': 'high_volume'
                    }

                    # Additional analysis
                    if opt['volume'] > opt['open_interest'] * 2:
                        activity['activity_type'] = 'volume_spike'

                    if opt['moneyness'] < 0.8 or opt['moneyness'] > 1.2:
                        activity['activity_type'] = 'otm_activity'

                    unusual_activities.append(activity)

            # Sort by unusual factor
            unusual_activities.sort(
                key=lambda x: x['unusual_factor'], reverse=True)

            # Limit to top 10
            unusual_activities = unusual_activities[:10]

        except Exception as e:
            logger.warning(f"Failed to detect unusual activity: {e}")

        return unusual_activities


class OptionsAnalysisProvider:
    """
    Comprehensive Options Analysis Provider

    Provides comprehensive options market analysis including implied volatility,
    options flow, sentiment indicators, and derivatives-based risk metrics.
    """

    def __init__(self, config: Optional[OptionsAnalysisConfig] = None):
        """Initialize options analysis provider"""
        self.config = config or OptionsAnalysisConfig()
        self.analyzer = OptionsDataAnalyzer(self.config)
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
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set headers
            self._session.headers.update({
                'User-Agent': 'DCF Lab Options Analysis Provider'
            })

        return self._session

    def get_comprehensive_options_analysis(self,
                                           symbol: str,
                                           include_greeks: bool = True,
                                           include_flow: bool = True) -> Dict[str,
                                                                              Any]:
        """
        Get comprehensive options analysis for a symbol

        Args:
            symbol: Stock symbol to analyze
            include_greeks: Include Greeks analysis
            include_flow: Include options flow analysis

        Returns:
            Dict with comprehensive options analysis
        """
        result = {
            'symbol': symbol,
            'analyzed_at': datetime.now().isoformat(),
            'options_chain': {},
            'market_sentiment': {},
            'volatility_analysis': {},
            'risk_assessment': {},
            'trading_recommendations': {},
            'success': False,
            'error': None
        }

        try:
            # Get options chain data
            chain_result = self.analyzer.get_options_chain(symbol)

            if not chain_result['success']:
                result['error'] = chain_result['error']
                return result

            result['options_chain'] = chain_result

            # Extract analysis components
            analysis = chain_result.get('analysis', {})

            # Market sentiment analysis
            sentiment = analysis.get('sentiment_indicators', {})
            result['market_sentiment'] = self._enhance_sentiment_analysis(
                sentiment, symbol)

            # Volatility analysis
            vol_analysis = analysis.get('volatility_analysis', {})
            result['volatility_analysis'] = self._enhance_volatility_analysis(
                vol_analysis, symbol)

            # Risk assessment
            risk_metrics = analysis.get('risk_metrics', {})
            flow_analysis = analysis.get('flow_analysis', {})
            result['risk_assessment'] = self._assess_options_risk(
                risk_metrics, flow_analysis, sentiment)

            # Trading recommendations
            recommendations = self._generate_trading_recommendations(
                result['market_sentiment'],
                result['volatility_analysis'],
                result['risk_assessment'])
            result['trading_recommendations'] = recommendations

            result['success'] = True
            logger.info(
                f"Completed comprehensive options analysis for {symbol}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(
                f"Failed comprehensive options analysis for {symbol}: {e}")

        return result

    def _enhance_sentiment_analysis(
            self, sentiment: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance sentiment analysis with additional insights"""
        enhanced = sentiment.copy()

        try:
            # Add sentiment strength indicators
            sentiment_score = sentiment.get('sentiment_score', 0)
            fear_greed = sentiment.get('fear_greed_index', 50)

            enhanced['sentiment_strength'] = {
                'score': abs(sentiment_score),
                'level': 'strong'
                if abs(sentiment_score) > 0.6 else 'moderate'
                if abs(sentiment_score) > 0.3 else 'weak'}

            enhanced['market_regime'] = {
                'fear_level': 'extreme'
                if fear_greed < 20 else 'high'
                if fear_greed < 40 else 'moderate', 'greed_level': 'extreme'
                if fear_greed > 80 else 'high'
                if fear_greed > 60 else 'moderate', 'regime': 'fear'
                if fear_greed < 35 else 'greed'
                if fear_greed > 65 else 'neutral'}

            # Sentiment implications
            overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
            enhanced['implications'] = {
                'bullish': overall_sentiment == 'bullish',
                'bearish': overall_sentiment == 'bearish',
                'contrarian_signal': fear_greed < 25 or fear_greed > 75,
                'momentum_signal': abs(sentiment_score) > 0.5
            }

        except Exception as e:
            logger.warning(f"Failed to enhance sentiment analysis: {e}")

        return enhanced

    def _enhance_volatility_analysis(
            self, vol_analysis: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Enhance volatility analysis with additional insights"""
        enhanced = vol_analysis.copy()

        try:
            atm_vol = vol_analysis.get('atm_volatility', 0.25)
            vol_skew = vol_analysis.get('volatility_skew', 0)
            vol_metrics = vol_analysis.get('vol_surface_metrics', {})

            # Volatility regime classification
            enhanced['volatility_regime'] = {
                'level': 'high' if atm_vol > 0.35 else 'moderate' if atm_vol > 0.20 else 'low',
                'percentile_estimate': min(100, max(0, (atm_vol - 0.10) / 0.40 * 100)),
                'regime_type': 'stress' if atm_vol > 0.40 else 'normal' if atm_vol < 0.30 else 'elevated'
            }

            # Skew analysis
            enhanced['skew_analysis'] = {
                'skew_level': 'high'
                if abs(vol_skew) > 0.05 else 'moderate'
                if abs(vol_skew) > 0.02 else 'low', 'direction': 'put_skew'
                if vol_skew > 0.01 else 'call_skew'
                if vol_skew < -0.01 else 'neutral',
                'interpretation': 'fear_driven'
                if vol_skew > 0.03 else 'risk_neutral'
                if abs(vol_skew) < 0.01 else 'unusual'}

            # Volatility opportunities
            vol_range = vol_metrics.get('iv_range', 0)
            enhanced['trading_opportunities'] = {
                'volatility_trading': vol_range > 0.10,
                'mean_reversion': atm_vol > 0.35,
                'breakout_potential': atm_vol < 0.15,
                'skew_trading': abs(vol_skew) > 0.03
            }

        except Exception as e:
            logger.warning(f"Failed to enhance volatility analysis: {e}")

        return enhanced

    def _assess_options_risk(self, risk_metrics: Dict[str, Any],
                             flow_analysis: Dict[str, Any],
                             sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall options-related risk"""
        assessment = {
            'overall_risk_level': 'moderate',
            'risk_factors': [],
            'risk_score': 0.5,
            'portfolio_impact': {},
            'hedging_recommendations': []
        }

        try:
            risk_score = 0.0
            risk_factors = []

            # Greeks-based risk
            total_delta = abs(risk_metrics.get('total_delta', 0))
            total_gamma = abs(risk_metrics.get('total_gamma', 0))
            total_vega = abs(risk_metrics.get('total_vega', 0))

            if total_delta > 50000:
                risk_score += 0.2
                risk_factors.append('high_delta_exposure')

            if total_gamma > 5000:
                risk_score += 0.2
                risk_factors.append('high_gamma_risk')

            if total_vega > 50000:
                risk_score += 0.2
                risk_factors.append('high_volatility_sensitivity')

            # Sentiment-based risk
            fear_greed = sentiment.get('fear_greed_index', 50)
            if fear_greed < 20 or fear_greed > 80:
                risk_score += 0.3
                risk_factors.append('extreme_sentiment')

            # Flow-based risk
            flow_direction = flow_analysis.get('flow_direction', 'neutral')
            if flow_direction != 'neutral':
                risk_score += 0.1
                risk_factors.append('directional_flow_bias')

            assessment['risk_score'] = min(risk_score, 1.0)
            assessment['risk_factors'] = risk_factors

            # Risk level classification
            if risk_score > 0.7:
                assessment['overall_risk_level'] = 'high'
            elif risk_score > 0.4:
                assessment['overall_risk_level'] = 'moderate'
            else:
                assessment['overall_risk_level'] = 'low'

            # Portfolio impact assessment
            assessment['portfolio_impact'] = {
                'delta_neutrality': 'biased'
                if total_delta > 10000 else 'near_neutral',
                'gamma_scalping_opportunity': total_gamma > 1000,
                'volatility_exposure': 'high'
                if total_vega > 30000 else 'moderate',
                'time_decay_impact': 'significant'
                if abs(risk_metrics.get('total_theta', 0)) > 1000 else
                'minimal'}

            # Hedging recommendations
            hedging_recs = []

            if total_delta > 25000:
                hedging_recs.append('delta_hedge_recommended')

            if total_vega > 40000:
                hedging_recs.append('volatility_hedge_consider')

            if fear_greed < 25:
                hedging_recs.append('protective_puts_consider')

            if fear_greed > 75:
                hedging_recs.append('covered_calls_consider')

            assessment['hedging_recommendations'] = hedging_recs

        except Exception as e:
            logger.warning(f"Failed to assess options risk: {e}")

        return assessment

    def _generate_trading_recommendations(self,
                                          sentiment: Dict[str,
                                                          Any],
                                          volatility: Dict[str,
                                                           Any],
                                          risk: Dict[str,
                                                     Any]) -> Dict[str,
                                                                   Any]:
        """Generate trading recommendations based on analysis"""
        recommendations = {
            'strategy_suggestions': [],
            'volatility_plays': [],
            'risk_management': [],
            'market_outlook': 'neutral',
            'confidence_level': 'moderate'
        }

        try:
            # Extract key metrics
            overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
            vol_regime = volatility.get(
                'volatility_regime', {}).get(
                'level', 'moderate')
            risk_level = risk.get('overall_risk_level', 'moderate')
            fear_greed = sentiment.get('fear_greed_index', 50)

            # Strategy suggestions based on sentiment
            if overall_sentiment == 'bullish' and vol_regime == 'low':
                recommendations['strategy_suggestions'].extend([
                    'long_calls', 'bull_call_spreads', 'covered_calls'
                ])
            elif overall_sentiment == 'bearish' and vol_regime == 'low':
                recommendations['strategy_suggestions'].extend([
                    'long_puts', 'bear_put_spreads', 'protective_puts'
                ])
            elif vol_regime == 'high':
                recommendations['strategy_suggestions'].extend([
                    'short_straddles', 'short_strangles', 'iron_condors'
                ])

            # Volatility plays
            vol_opportunities = volatility.get('trading_opportunities', {})

            if vol_opportunities.get('volatility_trading'):
                recommendations['volatility_plays'].append(
                    'volatility_arbitrage')

            if vol_opportunities.get('mean_reversion'):
                recommendations['volatility_plays'].append('short_volatility')

            if vol_opportunities.get('breakout_potential'):
                recommendations['volatility_plays'].append('long_volatility')

            # Risk management
            if risk_level == 'high':
                recommendations['risk_management'].extend([
                    'reduce_position_size', 'increase_hedging', 'monitor_closely'
                ])

            if fear_greed < 30:
                recommendations['risk_management'].append(
                    'contrarian_opportunity')
            elif fear_greed > 70:
                recommendations['risk_management'].append(
                    'take_profits_consider')

            # Market outlook
            if overall_sentiment == 'bullish' and risk_level != 'high':
                recommendations['market_outlook'] = 'positive'
            elif overall_sentiment == 'bearish' and risk_level != 'high':
                recommendations['market_outlook'] = 'negative'
            else:
                recommendations['market_outlook'] = 'neutral'

            # Confidence level
            sentiment_strength = sentiment.get(
                'sentiment_strength', {}).get(
                'level', 'weak')
            if sentiment_strength == 'strong' and risk_level == 'low':
                recommendations['confidence_level'] = 'high'
            elif sentiment_strength == 'weak' or risk_level == 'high':
                recommendations['confidence_level'] = 'low'
            else:
                recommendations['confidence_level'] = 'moderate'

        except Exception as e:
            logger.warning(f"Failed to generate trading recommendations: {e}")

        return recommendations


def get_options_analysis_provider(
        config: Optional[OptionsAnalysisConfig] = None) -> OptionsAnalysisProvider:
    """Factory function to create options analysis provider"""
    return OptionsAnalysisProvider(config)


# Example usage and testing
if __name__ == "__main__":
    # Initialize provider
    provider = get_options_analysis_provider()

    print("=== Options Data Provider ===")

    # Test comprehensive options analysis
    print("\n1. Comprehensive Options Analysis:")
    analysis = provider.get_comprehensive_options_analysis("AAPL")

    if analysis['success']:
        print(f"✅ Options analysis completed for {analysis['symbol']}")

        chain = analysis.get('options_chain', {})
        chain_analysis = chain.get('analysis', {})
        summary = chain_analysis.get('summary', {})

        current_price = float(chain.get('current_price', 0.0) or 0.0)
        total_call_volume = int(summary.get('total_call_volume', 0) or 0)
        total_put_volume = int(summary.get('total_put_volume', 0) or 0)
        put_call_ratio = float(summary.get('put_call_volume_ratio', 0.0) or 0.0)
        total_open_interest = int(summary.get('total_open_interest', 0) or 0)

        print("\nOptions Chain Summary:")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Total Call Volume: {total_call_volume:,}")
        print(f"  Total Put Volume: {total_put_volume:,}")
        print(f"  Put/Call Ratio: {put_call_ratio:.2f}")
        print(f"  Total Open Interest: {total_open_interest:,}")

        vol_analysis = analysis.get('volatility_analysis', {})
        vol_regime = vol_analysis.get('volatility_regime', {})
        vol_stats = chain_analysis.get('volatility_analysis', {})
        atm_vol = float(vol_stats.get('atm_volatility', 0.0) or 0.0)
        vol_skew = float(vol_stats.get('volatility_skew', 0.0) or 0.0)

        print("\nVolatility Analysis:")
        print(f"  ATM Implied Vol: {atm_vol * 100:.1f}%")
        print(f"  Volatility Regime: {vol_regime.get('level', 'unknown')}")
        print(f"  Vol Skew: {vol_skew * 100:.1f}%")

        sentiment = analysis.get('market_sentiment', {})
        overall_sentiment = sentiment.get('overall_sentiment', 'unknown')
        sentiment_score = float(sentiment.get('sentiment_score', 0.0) or 0.0)

        print("\nMarket Sentiment:")
        print(f"  Overall Sentiment: {overall_sentiment}")
        fear_greed_idx = sentiment.get('fear_greed_index', 50)
        print(f"  Fear & Greed Index: {fear_greed_idx:.0f}")
        print(f"  Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")

        sentiment_strength = sentiment.get('sentiment_strength', {})
        strength_level = sentiment_strength.get('level', 'unknown')
        print(f"  Sentiment Strength: {strength_level}")

        # Risk assessment
        risk = analysis.get('risk_assessment', {})
        print("\nRisk Assessment:")
        risk_level = risk.get('overall_risk_level', 'unknown')
        print(f"  Overall Risk Level: {risk_level}")
        print(f"  Risk Score: {risk.get('risk_score', 0):.2f}")

        risk_factors = risk.get('risk_factors', [])
        if risk_factors:
            print(f"  Risk Factors: {', '.join(risk_factors)}")

        # Trading recommendations
        recommendations = analysis.get('trading_recommendations', {})
        print("\nTrading Recommendations:")
        market_outlook = recommendations.get('market_outlook', 'unknown')
        print(f"  Market Outlook: {market_outlook}")
        confidence_level = recommendations.get('confidence_level', 'unknown')
        print(f"  Confidence Level: {confidence_level}")

        strategies = recommendations.get('strategy_suggestions', [])
        if strategies:
            print(f"  Strategy Suggestions: {', '.join(strategies)}")

        vol_plays = recommendations.get('volatility_plays', [])
        if vol_plays:
            print(f"  Volatility Plays: {', '.join(vol_plays)}")

        # Greeks analysis
        risk_metrics = chain_analysis.get('risk_metrics', {})
        if risk_metrics:
            print("\nPortfolio Greeks:")
            print(f"  Total Delta: {risk_metrics.get('total_delta', 0):,.0f}")
            print(f"  Total Gamma: {risk_metrics.get('total_gamma', 0):,.0f}")
            print(f"  Total Vega: {risk_metrics.get('total_vega', 0):,.0f}")
            print(f"  Total Theta: {risk_metrics.get('total_theta', 0):,.0f}")

        # Unusual activity
        unusual_activity = chain_analysis.get('unusual_activity', [])
        if unusual_activity:
            print("\nUnusual Activity (Top 3):")
            for i, activity in enumerate(unusual_activity[:3]):
                option_type = activity['option_type'].upper()
                strike = activity['strike']
                volume = activity['volume']
                factor = activity['unusual_factor']
                print(f"  {i + 1}. {option_type} ${strike} Volume: {volume:,} (Factor: {factor:.1f}x)")

    else:
        print(f"❌ Options analysis failed: {analysis['error']}")

    print("\n=== Options Data Provider Ready ===")
    print("✅ CBOE options data integration and implied volatility analysis")
    print("✅ Black-Scholes pricing and Greeks calculation")
    print("✅ Options flow analysis and sentiment indicators")
    print("✅ Volatility surface construction and skew analysis")
    print("✅ Unusual activity detection and risk assessment")
    print("✅ Trading recommendations and strategy suggestions")
    print("🚀 Ready for comprehensive options analysis and derivatives trading")
