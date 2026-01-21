"""
Market Regime Detection and Decision Rules

This module implements advanced market regime detection using machine learning 
techniques and economic indicators to classify the current market environment. 
It also provides decision rules for generating trading signals based on 
valuation and regime.
"""

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

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


@dataclass
class MarketRegime:
    """Market regime classification with probabilities"""
    name: str
    probability: float
    characteristics: Dict[str, Any]
    start_date: Optional[datetime] = None

    def __repr__(self):
        return f"MarketRegime(name='{self.name}', probability={self.probability:.2f})"


class RegimeDetector:
    """
    Advanced market regime detector using multiple models and indicators

    This class implements several regime detection algorithms and ensembles them
    for more robust classification.
    """

    REGIME_TYPES = {
        'bull': {
            'description': 'Strong growth, risk-on sentiment, expanding multiples',
            'valuation_bias': 'growth-focused',
            'risk_appetite': 'high',
            'typical_duration': '1-3 years'
        },
        'bear': {
            'description': 'Declining prices, risk-off sentiment, contracting multiples',
            'valuation_bias': 'defensive',
            'risk_appetite': 'low',
            'typical_duration': '1-2 years'
        },
        'recovery': {
            'description': ('Early economic recovery, steepening yield curve, '
                           'improving sentiment'),
            'valuation_bias': 'cyclicals',
            'risk_appetite': 'moderate-high',
            'typical_duration': '6-18 months'
        },
        'late_cycle': {
            'description': 'Peak growth, inflation concerns, flattening yield curve',
            'valuation_bias': 'value/quality',
            'risk_appetite': 'moderate',
            'typical_duration': '1-2 years'
        },
        'recession': {
            'description': 'Economic contraction, high volatility, flight to safety',
            'valuation_bias': 'FCF, defensive',
            'risk_appetite': 'very low',
            'typical_duration': '6-18 months'
        }
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the regime detector

        Args:
            model_path: Path to pre-trained model, if available
        """
        self.models = {}
        self.current_regime = None
        self.regime_history = []
        self.indicators = {}

        # Try to load pre-trained model if available
        if model_path is not None and os.path.exists(model_path):
            try:
                self.models['rf'] = joblib.load(model_path)
                print(f"Loaded regime detection model from {model_path}")
            except Exception as e:
                warnings.warn(f"Could not load model from {model_path}: {e}")

    def _extract_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime detection from market data

        Args:
            market_data: DataFrame with market indicators

        Returns:
            DataFrame with extracted features
        """
        if len(market_data) < 252:
            warnings.warn(
                "Less than 252 days of market data available. Features may be unreliable.")

        # Make copy to avoid modifying original data
        data = market_data.copy()

        # Ensure we have essential columns
        required_cols = ['vix', 'yield_10y', 'yield_2y']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Required column {col} not found in market_data")

        features = pd.DataFrame(index=data.index)

        # Volatility features
        features['vix'] = data['vix']
        features['vix_ma20'] = data['vix'].rolling(20).mean()
        features['vix_ma20_ratio'] = features['vix'] / features['vix_ma20']

        # Yield curve features
        features['yield_curve'] = data['yield_10y'] - data['yield_2y']
        features['yield_curve_ma20'] = features['yield_curve'].rolling(
            20).mean()
        features['yield_10y_ma50_ratio'] = data['yield_10y'] / \
            data['yield_10y'].rolling(50).mean()

        # Momentum features
        if 'excess_return' in data.columns:
            features['excess_return_20d'] = data['excess_return'].rolling(
                20).sum()
            features['excess_return_60d'] = data['excess_return'].rolling(
                60).sum()
            features['excess_return_252d'] = data['excess_return'].rolling(
                252).sum()

        # Trend features
        if 'market_price' in data.columns:
            features['price_200d_ratio'] = data['market_price'] / \
                data['market_price'].rolling(200).mean()
            features['price_50d_ratio'] = data['market_price'] / \
                data['market_price'].rolling(50).mean()
            features['price_50d_200d_ratio'] = (
                data['market_price'].rolling(50).mean() /
                data['market_price'].rolling(200).mean())

        # Market breadth features (if available)
        if 'advance_decline_ratio' in data.columns:
            features['adv_dec_ma10'] = data['advance_decline_ratio'].rolling(
                10).mean()

        # Economic indicators (if available)
        if 'pmi' in data.columns:
            features['pmi'] = data['pmi']
            features['pmi_change_3m'] = data['pmi'] - data['pmi'].shift(63)

        # Replace NaNs with 0 (from rolling calculations)
        features = features.fillna(0)

        # Store indicators for later analysis
        self.indicators = features.iloc[-1].to_dict()

        return features

    def _apply_vix_rules(self, latest, regime_probs):
        """Apply rules based on VIX indicators."""
        if latest['vix'] > 30:
            regime_probs['bear'] += 0.15
            regime_probs['recession'] += 0.15
            regime_probs['bull'] -= 0.1
        elif latest['vix'] < 15:
            regime_probs['bull'] += 0.1
            regime_probs['late_cycle'] += 0.05
        return regime_probs

    def _apply_yield_curve_rules(self, latest, regime_probs):
        """Apply rules based on yield curve indicators."""
        if latest['yield_curve'] < -0.1:  # Inverted yield curve
            regime_probs['late_cycle'] += 0.15
            regime_probs['recession'] += 0.1
            regime_probs['bull'] -= 0.1
        elif latest['yield_curve'] > 1.5:  # Steep yield curve
            regime_probs['recovery'] += 0.15
            regime_probs['bull'] += 0.05
        return regime_probs

    def _apply_momentum_rules(self, latest, regime_probs):
        """Apply rules based on momentum indicators."""
        if 'excess_return_252d' in latest:
            if latest['excess_return_252d'] > 0.25:  # Strong positive momentum
                regime_probs['bull'] += 0.2
                regime_probs['bear'] -= 0.1
                regime_probs['recession'] -= 0.1
            elif latest['excess_return_252d'] < -0.15:  # Strong negative momentum
                regime_probs['bear'] += 0.2
                regime_probs['recession'] += 0.1
                regime_probs['bull'] -= 0.1
        return regime_probs

    def _apply_trend_rules(self, latest, regime_probs):
        """Apply rules based on trend indicators."""
        if 'price_200d_ratio' in latest:
            if latest['price_200d_ratio'] > 1.2:  # Strong uptrend
                regime_probs['bull'] += 0.1
                regime_probs['late_cycle'] += 0.1
            elif latest['price_200d_ratio'] < 0.9:  # Strong downtrend
                regime_probs['bear'] += 0.15
                regime_probs['recession'] += 0.05
        return regime_probs

    def _apply_economic_rules(self, latest, regime_probs):
        """Apply rules based on economic indicators."""
        if 'pmi' in latest:
            if latest['pmi'] > 55:  # Strong expansion
                regime_probs['bull'] += 0.1
                regime_probs['late_cycle'] += 0.05
            elif latest['pmi'] < 45:  # Contraction
                regime_probs['recession'] += 0.2
                regime_probs['bear'] += 0.05
            # Improving from contraction
            elif latest['pmi'] < 50 and latest.get('pmi_change_3m', 0) > 2:
                regime_probs['recovery'] += 0.15
        return regime_probs

    def _rule_based_classification(
            self, features: pd.DataFrame) -> MarketRegime:
        """
        Rule-based regime classification

        Args:
            features: DataFrame with extracted features

        Returns:
            MarketRegime object
        """
        # Get the latest feature values
        latest = features.iloc[-1]

        # Default regime probabilities
        regime_probs = {
            'bull': 0.2,
            'bear': 0.2,
            'recovery': 0.2,
            'late_cycle': 0.2,
            'recession': 0.2
        }

        # Apply all rule sets
        regime_probs = self._apply_vix_rules(latest, regime_probs)
        regime_probs = self._apply_yield_curve_rules(latest, regime_probs)
        regime_probs = self._apply_momentum_rules(latest, regime_probs)
        regime_probs = self._apply_trend_rules(latest, regime_probs)
        regime_probs = self._apply_economic_rules(latest, regime_probs)

        # Normalize probabilities
        total_prob = sum(regime_probs.values())
        regime_probs = {k: v/total_prob for k, v in regime_probs.items()}

        # Find most likely regime
        most_likely_regime = max(regime_probs.items(), key=lambda x: x[1])

        return MarketRegime(
            name=most_likely_regime[0],
            probability=most_likely_regime[1],
            characteristics=self.REGIME_TYPES[most_likely_regime[0]]
        )

    def _model_based_classification(
            self, features: pd.DataFrame) -> Optional[MarketRegime]:
        """
        Model-based regime classification using pre-trained models

        Args:
            features: DataFrame with extracted features

        Returns:
            MarketRegime object or None if no model is available
        """
        if 'r' not in self.models:
            return None

        try:
            # Select relevant features used during training
            model_features = [
                'vix', 'vix_ma20', 'vix_ma20_ratio',
                'yield_curve', 'yield_curve_ma20'
            ]
            available_features = [
                f for f in model_features if f in features.columns]

            # If we don't have enough features, return None
            if len(available_features) < 3:
                return None

            # Prepare data
            X = features[available_features].iloc[[-1]]

            # Get prediction and probabilities
            regime_idx = self.models['r'].predict(X)[0]
            probas = self.models['rf'].predict_proba(X)[0]

            # Map index to regime name
            regime_map = {
                0: 'bull',
                1: 'bear',
                2: 'recovery',
                3: 'late_cycle',
                4: 'recession'
            }

            regime_name = regime_map.get(regime_idx, 'unknown')
            probability = probas[regime_idx]

            return MarketRegime(
                name=regime_name,
                probability=probability,
                characteristics=self.REGIME_TYPES[regime_name]
            )

        except Exception as e:
            warnings.warn(f"Error in model-based classification: {e}")
            return None

    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime using multiple methods

        Args:
            market_data: DataFrame with market indicators

        Returns:
            MarketRegime object
        """
        # Extract features
        features = self._extract_features(market_data)

        # Get rule-based regime
        rule_regime = self._rule_based_classification(features)

        # Get model-based regime if available
        model_regime = self._model_based_classification(features)

        # If model regime is available, blend results
        # Otherwise just use rule-based regime
        if model_regime is not None:
            # Simple ensemble: Average probabilities (could be more
            # sophisticated)
            if rule_regime.name == model_regime.name:
                final_regime = MarketRegime(
                    name=rule_regime.name,
                    probability=(
                        rule_regime.probability +
                        model_regime.probability) /
                    2,
                    characteristics=rule_regime.characteristics)
            else:
                # Different regimes detected - use the one with higher
                # probability
                if rule_regime.probability > model_regime.probability:
                    final_regime = rule_regime
                else:
                    final_regime = model_regime
        else:
            final_regime = rule_regime

        # Update current regime
        self.current_regime = final_regime

        # Add to history
        final_regime.start_date = datetime.now()
        self.regime_history.append(final_regime)

        return final_regime

    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get recommended valuation method weights for current regime

        Returns:
            Dictionary of method weights
        """
        if self.current_regime is None:
            # Default weights
            return {
                'dc': 0.5,
                'residual_income': 0.2,
                'relative': 0.3
            }

        regime = self.current_regime.name

        # Regime-specific weights
        weights = {
            'bull': {
                'dc': 0.35,
                'residual_income': 0.15,
                'relative': 0.50  # More weight on relative valuation in bull markets
            },
            'bear': {
                'dc': 0.60,
                'residual_income': 0.30,
                'relative': 0.10  # Less weight on relative valuation in bear markets
            },
            'recovery': {
                'dc': 0.45,
                'residual_income': 0.15,
                'relative': 0.40  # More weight on relative valuation in recovery
            },
            'late_cycle': {
                'dc': 0.50,
                'residual_income': 0.30,
                'relative': 0.20  # Balance in late cycle
            },
            'recession': {
                'dc': 0.65,
                'residual_income': 0.25,
                'relative': 0.10  # Focus on fundamentals in recession
            }
        }

        return weights.get(regime, weights.get('late_cycle')
                           )  # Default to late_cycle weights


class PositionSizer:
    """
    Position sizing based on valuation metrics and risk parameters

    This class implements various position sizing methods based on
    valuation confidence, risk tolerance, and other factors.
    """

    def __init__(self,
                 max_position_size: float = 0.1,
                 min_position_size: float = 0.01,
                 base_confidence: float = 0.8):
        """
        Initialize the position sizer

        Args:
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
            base_confidence: Base confidence level for sizing
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.base_confidence = base_confidence

    def kelly_criterion(self,
                        win_probability: float,
                        win_loss_ratio: float,
                        fraction: float = 0.5) -> float:
        """
        Calculate position size using the Kelly criterion

        Args:
            win_probability: Probability of a winning trade (0-1)
            win_loss_ratio: Ratio of average win to average loss
            fraction: Fraction of full Kelly to use (typically 0.5 for Half Kelly)

        Returns:
            Recommended position size as fraction of portfolio
        """
        # Full Kelly formula: f* = p - (1-p)/r
        # where p is probability of win, r is win/loss ratio
        full_kelly = win_probability - (1 - win_probability) / win_loss_ratio

        # Apply fraction and bounds
        position_size = full_kelly * fraction
        position_size = max(
            self.min_position_size, min(
                self.max_position_size, position_size))

        return position_size

    def size_from_edge(self,
                       expected_return: float,
                       downside_risk: float,
                       confidence: float) -> float:
        """
        Calculate position size based on edge and confidence

        Args:
            expected_return: Expected return (e.g., 0.15 for 15%)
            downside_risk: Downside risk (e.g., 0.08 for 8%)
            confidence: Confidence in the forecast (0-1)

        Returns:
            Recommended position size as fraction of portfolio
        """
        if downside_risk <= 0:
            # Avoid division by zero
            downside_risk = 0.01

        # Calculate win/loss ratio
        win_loss_ratio = expected_return / downside_risk

        # Use confidence as win probability
        win_probability = confidence

        # Apply Kelly with a conservative fraction
        position_size = self.kelly_criterion(
            win_probability=win_probability,
            win_loss_ratio=win_loss_ratio,
            fraction=0.3  # Conservative fraction
        )

        return position_size

    def size_from_valuation(self,
                            valuation_results: Dict[str, Any],
                            market_price: float,
                            regime: str = 'late_cycle') -> Dict[str, Any]:
        """
        Calculate position size from valuation results

        Args:
            valuation_results: Dictionary with valuation outputs
            market_price: Current market price
            regime: Current market regime

        Returns:
            Dictionary with position sizing recommendations
        """
        # Extract key metrics from valuation results
        mean_value = valuation_results.get(
            'mean', valuation_results.get('mean_value'))
        if mean_value is None:
            raise ValueError(
                "Valuation results missing 'mean' or 'mean_value'")

        # Calculate expected return
        expected_return = (mean_value / market_price) - 1

        # Extract or calculate downside risk
        if 'confidence_interval' in valuation_results:
            lower_bound = valuation_results['confidence_interval'][0]
            downside_risk = max(0, (market_price - lower_bound) / market_price)
        else:
            # Estimate downside risk as a fraction of expected return
            downside_risk = abs(expected_return) * 0.7

        # Extract or calculate confidence
        confidence = valuation_results.get('confidence', 0.8)

        # Adjust confidence based on regime
        regime_confidence_adj = {
            'bull': 0.9,
            'bear': 0.7,
            'recovery': 0.85,
            'late_cycle': 0.75,
            'recession': 0.6
        }

        adj_confidence = confidence * regime_confidence_adj.get(regime, 0.8)

        # Calculate position size
        if expected_return > 0:
            position_size = self.size_from_edge(
                expected_return=expected_return,
                downside_risk=downside_risk,
                confidence=adj_confidence
            )
        else:
            # Negative expected return - no position
            position_size = 0

        # Prepare result
        result = {
            'position_size': position_size,
            'expected_return': expected_return,
            'downside_risk': downside_risk,
            'confidence': adj_confidence,
            'regime': regime,
            'max_position': self.max_position_size
        }

        return result


class DecisionRules:
    """
    Decision rules for generating trading signals based on valuation and regime

    This class implements various decision rules for translating valuation
    results into actionable trading signals.
    """

    def __init__(self,
                 risk_preference: float = 0.5,
                 min_upside: float = 0.15,
                 max_downside: float = 0.1):
        """
        Initialize decision rules

        Args:
            risk_preference: Risk preference from 0 (risk-averse) to 1 (risk-seeking)
            min_upside: Minimum upside required for a buy signal
            max_downside: Maximum acceptable downside risk
        """
        self.risk_preference = risk_preference
        self.min_upside = min_upside
        self.max_downside = max_downside
        self.position_sizer = PositionSizer()
        self.regime_detector = None

    def set_regime_detector(self, detector: RegimeDetector) -> None:
        """
        Set the regime detector

        Args:
            detector: RegimeDetector instance
        """
        self.regime_detector = detector

    def adjust_thresholds_for_regime(self, regime: str) -> Tuple[float, float]:
        """
        Adjust decision thresholds based on market regime

        Args:
            regime: Market regime name

        Returns:
            Tuple of (adjusted_min_upside, adjusted_max_downside)
        """
        # Regime-specific adjustments
        adjustments = {
            'bull': (self.min_upside * 1.2, self.max_downside * 1.1),
            'bear': (self.min_upside * 0.8, self.max_downside * 0.8),
            'recovery': (self.min_upside * 0.9, self.max_downside * 1.0),
            'late_cycle': (self.min_upside * 1.1, self.max_downside * 0.9),
            'recession': (self.min_upside * 0.7, self.max_downside * 0.7)
        }

        return adjustments.get(regime, (self.min_upside, self.max_downside))

    def _detect_market_regime(self,
                              valuation_results: Dict[str,
                                                      Any],
                              market_data: Optional[pd.DataFrame]) -> str:
        """
        Detect market regime from available data

        Args:
            valuation_results: Dictionary with valuation results
            market_data: Optional market data for regime detection

        Returns:
            String with regime name
        """
        if self.regime_detector is not None and market_data is not None:
            regime_result = self.regime_detector.detect_regime(market_data)
            return regime_result.name
        elif 'regime' in valuation_results:
            return valuation_results['regime']

        return 'normal'

    def _calculate_expected_return(
            self,
            mean_value: float,
            market_price: float) -> float:
        """
        Calculate expected return from valuation

        Args:
            mean_value: Mean valuation estimate
            market_price: Current market price

        Returns:
            Expected return as a decimal
        """
        return (mean_value / market_price) - 1

    def _estimate_upside_probability(
            self, valuation_results: Dict[str, Any],
            expected_return: float) -> float:
        """
        Estimate probability of upside from valuation results

        Args:
            valuation_results: Dictionary with valuation results
            expected_return: Expected return from valuation

        Returns:
            Probability of upside as a decimal
        """
        prob_upside = valuation_results.get('prob_upside', None)
        if prob_upside is not None:
            return prob_upside

        # Try to calculate from distribution if available
        distribution = valuation_results.get('distribution', None)
        if distribution is not None:
            return np.mean(
                distribution > valuation_results.get(
                    'market_price', 0))

        # Default assumption based on expected return
        return 0.6 if expected_return > 0 else 0.4

    def _calculate_downside_risk(self,
                                 valuation_results: Dict[str,
                                                         Any],
                                 market_price: float,
                                 expected_return: float) -> float:
        """
        Calculate downside risk from valuation results

        Args:
            valuation_results: Dictionary with valuation results
            market_price: Current market price
            expected_return: Expected return from valuation

        Returns:
            Downside risk as a decimal
        """
        if 'confidence_interval' in valuation_results:
            lower_bound = valuation_results['confidence_interval'][0]
            return max(0, (market_price - lower_bound) / market_price)

        # Estimate downside risk as a fraction of expected return
        return abs(expected_return) * 0.7

    def _determine_trading_decision(
            self,
            expected_return: float,
            downside_risk: float,
            adj_min_upside: float,
            adj_max_downside: float) -> str:
        """
        Determine trading decision based on risk/return metrics

        Args:
            expected_return: Expected return from valuation
            downside_risk: Downside risk estimate
            adj_min_upside: Adjusted minimum upside threshold
            adj_max_downside: Adjusted maximum downside threshold

        Returns:
            Decision string: 'buy', 'sell', or 'hold'
        """
        if expected_return > adj_min_upside and downside_risk < adj_max_downside:
            return 'buy'
        elif expected_return < -adj_min_upside or downside_risk > adj_max_downside * 1.5:
            return 'sell'

        return 'hold'

    def generate_decision(self,
                          valuation_results: Dict[str, Any],
                          market_price: float,
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate a trading decision based on valuation results

        Args:
            valuation_results: Dictionary with valuation outputs
            market_price: Current market price
            market_data: Optional DataFrame with market indicators for regime detection

        Returns:
            Dictionary with decision recommendations
        """
        # Detect regime and adjust thresholds
        regime = self._detect_market_regime(valuation_results, market_data)
        adj_min_upside, adj_max_downside = self.adjust_thresholds_for_regime(
            regime)

        # Extract key metrics from valuation results
        mean_value = valuation_results.get(
            'mean', valuation_results.get('mean_value'))
        if mean_value is None:
            raise ValueError(
                "Valuation results missing 'mean' or 'mean_value'")

        # Calculate metrics
        expected_return = self._calculate_expected_return(
            mean_value, market_price)
        prob_upside = self._estimate_upside_probability(
            valuation_results, expected_return)
        downside_risk = self._calculate_downside_risk(
            valuation_results, market_price, expected_return)

        # Calculate risk-adjusted score
        risk_score = (self.risk_preference * expected_return * \
                      prob_upside) - ((1 - self.risk_preference) * downside_risk)

        # Determine trading decision
        decision = self._determine_trading_decision(
            expected_return, downside_risk, adj_min_upside, adj_max_downside)

        # Calculate position size recommendation
        position_info = self.position_sizer.size_from_valuation(
            valuation_results=valuation_results,
            market_price=market_price,
            regime=regime
        )

        # Combine results
        return {
            'decision': decision,
            'expected_return': expected_return,
            'downside_risk': downside_risk,
            'prob_upside': prob_upside,
            'risk_score': risk_score,
            'regime': regime,
            'position_size': position_info['position_size'],
            'confidence': position_info['confidence']
        }


def create_decision_engine(
        risk_preference: float = 0.5) -> Tuple[RegimeDetector, DecisionRules]:
    """
    Factory function to create a regime detector and decision rules

    Args:
        risk_preference: Risk preference from 0 (risk-averse) to 1 (risk-seeking)

    Returns:
        Tuple of (RegimeDetector, DecisionRules)
    """
    # Try to find pre-trained model
    model_path = os.path.join(
        get_data_path(),
        "models",
        "regime_detector_model.pkl")

    # Create regime detector
    detector = RegimeDetector(
        model_path=model_path if os.path.exists(model_path) else None)

    # Create decision rules with regime detector
    rules = DecisionRules(risk_preference=risk_preference)
    rules.set_regime_detector(detector)

    return detector, rules
