"""
Market Regime Detection

This module implements methods to detect market regimes based on
financial indicators, volatility, and other metrics.
"""

from typing import Any, Dict, Optional

import pandas as pd


class RegimeDetector:
    """
    Detect market regimes based on various indicators

    This class analyzes market data to identify different regimes,
    such as bull market, bear market, high volatility, etc.
    """

    def __init__(self,
                 look_back_period: int = 60,
                 volatility_window: int = 20):
        """
        Initialize the regime detector

        Args:
            look_back_period: Number of days to look back for regime detection
            volatility_window: Window size for volatility calculation
        """
        self.look_back_period = look_back_period
        self.volatility_window = volatility_window

        # Define regime thresholds
        self.volatility_threshold_high = 0.25
        self.volatility_threshold_low = 0.15
        self.momentum_threshold_positive = 0.05
        self.momentum_threshold_negative = -0.05
        self.credit_spread_threshold_wide = 150  # basis points
        self.credit_spread_threshold_narrow = 80  # basis points

    def detect_regime(
            self, market_data: pd.DataFrame,
            macro_indicators: Optional[Dict[str, Any]] = None) -> str:
        """
        Detect the current market regime

        Args:
            market_data: DataFrame with market data columns
                Required columns: 'market_price', 'volatility', 'volume'
            macro_indicators: Dictionary with macroeconomic indicators (optional)
                Optional keys: 'gdp_growth', 'inflation_rate', 'unemployment_rate',
                'fed_funds_rate', 'fed_funds_change'

        Returns:
            String indicating the detected market regime
        """
        # Process market data
        market_regime_data = self._analyze_market_data(market_data)

        # Process macro indicators if available
        if macro_indicators:
            macro_regime_data = self._analyze_macro_indicators(
                macro_indicators)
            # Combine market and macro regimes
            combined_regime_data = self._combine_regimes(
                market_regime_data, macro_regime_data)
        else:
            combined_regime_data = market_regime_data

        # Return the overall market regime string
        return combined_regime_data['market_regime']

    def _analyze_market_data(
            self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data to detect regime

        Args:
            market_data: DataFrame with market data columns

        Returns:
            Dictionary with market regime analysis
        """
        results = {}

        # Check if the dataframe has enough data
        if len(market_data) < self.look_back_period:
            return self._get_default_market_results()

        # Analyze different components
        self._analyze_volatility(market_data, results)
        self._analyze_momentum(market_data, results)
        results['credit_regime'] = 'normal'  # Default for credit spreads

        # Determine overall market regime
        self._determine_market_regime(results)

        return results

    def _get_default_market_results(self) -> Dict[str, Any]:
        """Return default market results when there's not enough data"""
        return {
            'volatility_regime': 'normal',
            'momentum_regime': 'neutral',
            'credit_regime': 'normal',
            'market_regime': 'neutral'
        }

    def _analyze_volatility(self, market_data: pd.DataFrame,
                            results: Dict[str, Any]) -> None:
        """Analyze volatility regime and update results dictionary"""
        if 'volatility' in market_data.columns:
            # Get the mean volatility from the last window
            current_vol = market_data['volatility'].iloc[-self.volatility_window:].mean()

            if current_vol > self.volatility_threshold_high:
                results['volatility_regime'] = 'high'
            elif current_vol < self.volatility_threshold_low:
                results['volatility_regime'] = 'low'
            else:
                results['volatility_regime'] = 'normal'
        else:
            results['volatility_regime'] = 'normal'  # Default

    def _analyze_momentum(self, market_data: pd.DataFrame,
                          results: Dict[str, Any]) -> None:
        """Analyze price momentum and update results dictionary"""
        if 'market_price' in market_data.columns:
            # Calculate return over look_back_period
            start_price = market_data['market_price'].iloc[-self.look_back_period]
            end_price = market_data['market_price'].iloc[-1]
            price_change = (end_price / start_price) - 1

            if price_change > self.momentum_threshold_positive:
                results['momentum_regime'] = 'positive'
            elif price_change < self.momentum_threshold_negative:
                results['momentum_regime'] = 'negative'
            else:
                results['momentum_regime'] = 'neutral'

            results['price_change'] = price_change
        else:
            results['momentum_regime'] = 'neutral'  # Default

    def _determine_market_regime(self, results: Dict[str, Any]) -> None:
        """Determine overall market regime based on component regimes"""
        vol_regime = results['volatility_regime']
        mom_regime = results['momentum_regime']
        credit_regime = results['credit_regime']

        # Simple rule-based classification
        if vol_regime == 'high' and mom_regime == 'negative':
            overall_regime = 'crisis'
        elif vol_regime == 'high' and mom_regime != 'negative':
            overall_regime = 'high_uncertainty'
        elif vol_regime == 'low' and mom_regime == 'positive':
            overall_regime = 'bull_market'
        elif mom_regime == 'positive' and credit_regime != 'stressed':
            overall_regime = 'growth'
        elif mom_regime == 'negative' and credit_regime == 'stressed':
            overall_regime = 'bear_market'
        elif credit_regime == 'accommodative':
            overall_regime = 'recovery'
        else:
            overall_regime = 'neutral'

        results['market_regime'] = overall_regime

    def _analyze_macro_indicators(
            self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze macroeconomic indicators

        Args:
            indicators: Dictionary with macroeconomic indicators

        Returns:
            Dictionary with macro regime analysis
        """
        results = {}

        # Analyze individual components
        self._analyze_gdp_growth(indicators, results)
        self._analyze_inflation(indicators, results)
        self._analyze_monetary_policy(indicators, results)

        # Determine overall macro regime
        self._determine_macro_regime(results)

        return results

    def _analyze_gdp_growth(
            self, indicators: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Analyze GDP growth and update results dictionary"""
        if 'gdp_growth' in indicators:
            gdp_growth = indicators['gdp_growth']
            if gdp_growth > 0.03:  # 3% growth is strong
                results['growth_regime'] = 'expansion'
            elif gdp_growth > 0:
                results['growth_regime'] = 'slow_growth'
            else:
                results['growth_regime'] = 'contraction'
        else:
            results['growth_regime'] = 'unknown'

    def _analyze_inflation(
            self, indicators: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Analyze inflation and update results dictionary"""
        if 'inflation_rate' in indicators:
            inflation = indicators['inflation_rate']
            if inflation > 0.04:  # 4% inflation is high
                results['inflation_regime'] = 'high'
            elif inflation < 0.01:  # 1% inflation is low
                results['inflation_regime'] = 'low'
            else:
                results['inflation_regime'] = 'target'
        else:
            results['inflation_regime'] = 'unknown'

    def _analyze_monetary_policy(
            self, indicators: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Analyze monetary policy and update results dictionary"""
        if 'fed_funds_rate' in indicators and 'fed_funds_change' in indicators:
            rate = indicators['fed_funds_rate']
            change = indicators['fed_funds_change']

            if change > 0:
                results['monetary_policy'] = 'tightening'
            elif change < 0:
                results['monetary_policy'] = 'easing'
            elif rate > 0.03:  # 3% is restrictive
                results['monetary_policy'] = 'restrictive'
            elif rate < 0.01:  # 1% is accommodative
                results['monetary_policy'] = 'accommodative'
            else:
                results['monetary_policy'] = 'neutral'
        else:
            results['monetary_policy'] = 'unknown'

    def _determine_macro_regime(self, results: Dict[str, Any]) -> None:
        """Determine overall macro regime based on component regimes"""
        growth = results.get('growth_regime', 'unknown')
        inflation = results.get('inflation_regime', 'unknown')
        policy = results.get('monetary_policy', 'unknown')

        if growth == 'expansion' and inflation != 'high':
            overall_macro = 'healthy_growth'
        elif growth == 'contraction':
            overall_macro = 'recession'
        elif inflation == 'high' and policy == 'tightening':
            overall_macro = 'stagflation'
        elif policy == 'easing' and growth != 'expansion':
            overall_macro = 'stimulus'
        else:
            overall_macro = 'mixed'

        results['macro_regime'] = overall_macro

    def _combine_regimes(
            self, market: Dict[str, Any], macro: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine market and macro regimes into a single assessment

        Args:
            market: Market regime analysis
            macro: Macro regime analysis

        Returns:
            Combined regime analysis
        """
        results = {**market, **macro}  # Merge dictionaries

        # Determine overall regime
        overall = self._determine_overall_regime(market, macro)
        results['overall_regime'] = overall

        # Add adjustments based on regime
        self._apply_risk_premium_adjustment(results, overall)
        self._apply_growth_rate_adjustment(results, overall)

        return results

    def _determine_overall_regime(
            self, market: Dict[str, Any], macro: Dict[str, Any]) -> str:
        """Determine overall regime based on market and macro regimes"""
        market_regime = market.get('market_regime', 'neutral')
        macro_regime = macro.get('macro_regime', 'mixed')

        # Crisis conditions take precedence
        if market_regime == 'crisis' or macro_regime == 'recession':
            return 'crisis'

        # Severe negative conditions
        if market_regime == 'bear_market' and macro_regime in [
                'recession', 'stagflation']:
            return 'severe_downturn'

        # Strong positive conditions
        if market_regime == 'bull_market' and macro_regime == 'healthy_growth':
            return 'strong_expansion'

        # Stable growth conditions
        if market_regime == 'growth' and macro_regime in [
                'healthy_growth', 'stimulus']:
            return 'stable_growth'

        # Other specific conditions
        if market_regime == 'high_uncertainty':
            return 'high_uncertainty'

        if macro_regime == 'stimulus':
            return 'recovery'

        # Default case
        return 'mixed'

    def _apply_risk_premium_adjustment(
            self, results: Dict[str, Any], overall: str) -> None:
        """Apply risk premium adjustment based on regime"""
        # Risk premium map to avoid long if-else chain
        risk_premium_map = {
            'crisis': 0.02,           # +200 bps
            'severe_downturn': 0.02,  # +200 bps
            'high_uncertainty': 0.01,  # +100 bps
            'recovery': 0.0,          # No change
            'mixed': 0.0,             # No change
            'stable_growth': -0.005,  # -50 bps
            'strong_expansion': -0.01  # -100 bps
        }

        # Get adjustment or default to 0.0 if regime not found
        results['risk_premium_adj'] = risk_premium_map.get(overall, 0.0)

    def _apply_growth_rate_adjustment(
            self, results: Dict[str, Any], overall: str) -> None:
        """Apply growth rate adjustment based on regime"""
        # Growth adjustment map to avoid long if-else chain
        growth_adj_map = {
            'crisis': -0.02,          # -2 percentage points
            'severe_downturn': -0.02,  # -2 percentage points
            'high_uncertainty': -0.01,  # -1 percentage point
            'recovery': 0.01,         # +1 percentage point
            'stable_growth': 0.015,   # +1.5 percentage points
            'strong_expansion': 0.02  # +2 percentage points
        }

        # Get adjustment or default to 0.0 if regime not found
        results['growth_adj'] = growth_adj_map.get(overall, 0.0)


def get_regime_adjusted_assumptions(
    base_assumptions: Dict[str, Any],
    regime: str,
    news_impact: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Adjust valuation assumptions based on market regime and news impact

    Args:
        base_assumptions: Dictionary with base case assumptions
        regime: String indicating the market regime (e.g., 'bull_market', 'bear_market')
        news_impact: Optional dictionary with news impact analysis

    Returns:
        Dictionary with adjusted assumptions
    """
    adjusted = base_assumptions.copy()

    # Define regime adjustments based on regime type
    regime_adjustments = {
        'bull_market': {'risk_premium_adj': -0.01, 'growth_adj': 0.02},
        'bear_market': {'risk_premium_adj': 0.02, 'growth_adj': -0.03},
        'crisis': {'risk_premium_adj': 0.04, 'growth_adj': -0.05},
        'recovery': {'risk_premium_adj': -0.005, 'growth_adj': 0.01},
        'high_uncertainty': {'risk_premium_adj': 0.015, 'growth_adj': -0.01},
        'growth': {'risk_premium_adj': -0.005, 'growth_adj': 0.015},
        'neutral': {'risk_premium_adj': 0.0, 'growth_adj': 0.0}
    }

    # Get regime adjustments (default to neutral if regime not recognized)
    regime_risk_adj = regime_adjustments.get(
        regime, {'risk_premium_adj': 0.0, 'growth_adj': 0.0}
    ).get('risk_premium_adj', 0.0)

    regime_growth_adj = regime_adjustments.get(
        regime, {'risk_premium_adj': 0.0, 'growth_adj': 0.0}
    ).get('growth_adj', 0.0)

    # Get news impact adjustments
    news_risk_adj = 0.0
    news_growth_adj = 0.0
    news_margin_adj = 0.0

    if news_impact is not None:
        news_risk_adj = news_impact.get('risk_premium_adj', 0.0)
        news_growth_adj = news_impact.get('revenue_growth_adj', 0.0)
        news_margin_adj = news_impact.get('margin_adj', 0.0)

    # Apply adjustments
    if 'wacc' in adjusted:
        adjusted['wacc'] += regime_risk_adj + news_risk_adj

    if 'revenue_growth' in adjusted:
        adjusted['revenue_growth'] += regime_growth_adj + news_growth_adj

    if 'ebitda_margin' in adjusted:
        adjusted['ebitda_margin'] += news_margin_adj

    # Add metadata
    adjusted['regime'] = regime
    adjusted['adjustments'] = {
        'risk_premium': regime_risk_adj + news_risk_adj,
        'growth': regime_growth_adj + news_growth_adj,
        'margin': news_margin_adj
    }

    return adjusted
