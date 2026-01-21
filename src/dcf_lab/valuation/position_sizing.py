"""
Position Sizing Module

This module provides tools for generating position sizing recommendations
based on valuation, conviction, and risk parameters.
"""

from typing import Any, Dict, List, Optional


class PositionSizer:
    """
    Generates position size recommendations based on conviction,
    valuation, and risk parameters.
    """

    def __init__(self,
                 max_position_size: float = 0.05,
                 risk_per_trade: float = 0.01):
        """
        Initialize PositionSizer.

        Args:
            max_position_size: Maximum position size as fraction of portfolio
            risk_per_trade: Maximum risk per trade as fraction of portfolio
        """
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade

    def calculate_position(self,
                           decision: Dict[str, Any],
                           portfolio_value: float,
                           current_price: float,
                           value_at_risk: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate recommended position size based on decision.

        Args:
            decision: Decision dictionary from DecisionRules
            portfolio_value: Total portfolio value
            current_price: Current price of the security
            value_at_risk: Optional [low, high] range for value

        Returns:
            Dictionary with position recommendations
        """
        action = decision['action']
        conviction = decision['conviction']

        # If action is HOLD, no position change
        if action == "HOLD":
            return {
                'position_type': "HOLD",
                'position_size': 0.0,
                'shares': 0,
                'investment': 0.0,
                'stop_loss': None,
                'target_price': None
            }

        # Base position size on conviction
        conviction_weights = {
            "STRONG": 1.0,
            "MODERATE": 0.7,
            "MILD": 0.4,
            "NEUTRAL": 0.0
        }

        conviction_weight = conviction_weights.get(conviction, 0.5)
        base_position_size = self.max_position_size * conviction_weight

        # Adjust for margin of safety
        margin_of_safety = decision['margin_of_safety']
        mos_factor = min(1.5, max(0.5, 1.0 + margin_of_safety))
        position_size = base_position_size * mos_factor

        # Cap at maximum
        position_size = min(position_size, self.max_position_size)

        # Calculate investment amount
        investment = portfolio_value * position_size
        shares = int(investment / current_price) if current_price > 0 else 0

        # Calculate stop loss and target prices
        stop_loss = self._calculate_stop_loss(
            action, current_price, value_at_risk
        )

        target_price = self._calculate_target_price(
            action, decision['value'], value_at_risk
        )

        return {
            'position_type': action,
            'position_size': position_size,
            'shares': shares,
            'investment': investment,
            'stop_loss': stop_loss,
            'target_price': target_price
        }

    def _calculate_stop_loss(
            self, action: str, current_price: float,
            value_at_risk: Optional[List[float]] = None) -> Optional[float]:
        """Calculate stop loss price"""
        if action == "BUY":
            # If value at risk provided, use the lower bound
            if value_at_risk and len(value_at_risk) == 2:
                # Use either 90% of lower bound or 90% of current price,
                # whichever is higher
                return max(value_at_risk[0] * 0.9, current_price * 0.9)
            else:
                # Default to 10% below current price
                return current_price * 0.9
        elif action == "SELL":
            # For short positions
            if value_at_risk and len(value_at_risk) == 2:
                # Use either 110% of upper bound or 110% of current price,
                # whichever is lower
                return min(value_at_risk[1] * 1.1, current_price * 1.1)
            else:
                # Default to 10% above current price
                return current_price * 1.1

        return None

    def _calculate_target_price(
            self, action: str, intrinsic_value: float,
            value_at_risk: Optional[List[float]] = None) -> Optional[float]:
        """Calculate target price"""
        if action == "BUY":
            # If value at risk provided, use the upper bound
            if value_at_risk and len(value_at_risk) == 2:
                return value_at_risk[1]
            else:
                # Default to intrinsic value + 20%
                return intrinsic_value * 1.2
        elif action == "SELL":
            # For short positions
            if value_at_risk and len(value_at_risk) == 2:
                return value_at_risk[0]
            else:
                # Default to intrinsic value - 20%
                return intrinsic_value * 0.8

        return None
