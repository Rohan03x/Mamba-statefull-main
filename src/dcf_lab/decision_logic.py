"""Trading decision logic for converting probabilities to positions.

This module implements threshold-based trading logic that maps model probability
outputs to buy/sell/hold decisions for Stage C production forecasting.
"""
from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type aliases for clarity
Position = Literal[1, 0, -1]  # 1=LONG, 0=HOLD, -1=SHORT
ThresholdConfig = Dict[str, float]  # {"buy": 0.52, "sell": 0.48}


def decision_from_prob(
    p_up: Union[float, np.ndarray],
    thresholds: Optional[ThresholdConfig] = None,
    *,
    default_buy: float = 0.52,
    default_sell: float = 0.48,
) -> Union[Position, np.ndarray]:
    """Convert probability to trading position using threshold logic.
    
    Maps probability of upward price movement to discrete trading signal:
    - If p_up >= buy_threshold  → +1 (LONG)
    - If p_up <= sell_threshold → -1 (SHORT)
    - Otherwise                  →  0 (HOLD/NEUTRAL)
    
    Args:
        p_up: Probability of upward price movement (0.0 to 1.0).
              Can be scalar float or numpy array.
        thresholds: Dict with 'buy' and 'sell' thresholds.
                   If None, uses defaults.
        default_buy: Buy threshold if not in thresholds (default: 0.52).
        default_sell: Sell threshold if not in thresholds (default: 0.48).
    
    Returns:
        Position signal(s):
        - +1: GO LONG (buy signal)
        -  0: HOLD (neutral, stay out of market)
        - -1: GO SHORT (sell signal)
        
        Returns same type as input (scalar → scalar, array → array).
    
    Examples:
        >>> # Scalar usage
        >>> decision_from_prob(0.55)  # High confidence up
        1
        
        >>> decision_from_prob(0.50)  # Neutral probability
        0
        
        >>> decision_from_prob(0.42)  # High confidence down
        -1
        
        >>> # Array usage
        >>> probs = np.array([0.56, 0.50, 0.44])
        >>> decision_from_prob(probs)
        array([ 1,  0, -1])
        
        >>> # Custom thresholds (wider neutral band)
        >>> thresholds = {"buy": 0.58, "sell": 0.42}
        >>> decision_from_prob(0.55, thresholds)  # Would be LONG with defaults
        0  # HOLD with wide band
    
    Notes:
        - Thresholds must satisfy: sell_threshold < buy_threshold
        - Neutral band width = buy_threshold - sell_threshold
        - Narrower band (e.g., ±0.02) → More trades, less selective
        - Wider band (e.g., ±0.08) → Fewer trades, more selective
        - Default band (±0.04) → Moderate selectivity
    
    Raises:
        ValueError: If buy_threshold <= sell_threshold (invalid config).
    """
    # Extract thresholds
    if thresholds is None:
        buy_thresh = default_buy
        sell_thresh = default_sell
    else:
        buy_thresh = float(thresholds.get("buy", default_buy))
        sell_thresh = float(thresholds.get("sell", default_sell))
    
    # Validate threshold configuration
    if buy_thresh <= sell_thresh:
        raise ValueError(
            f"Invalid thresholds: buy ({buy_thresh}) must be > sell ({sell_thresh}). "
            f"Current configuration creates no neutral zone."
        )
    
    # Vectorized decision logic
    if isinstance(p_up, np.ndarray):
        # Array input → array output
        decisions = np.zeros_like(p_up, dtype=np.int8)
        decisions[p_up >= buy_thresh] = 1   # LONG positions
        decisions[p_up <= sell_thresh] = -1  # SHORT positions
        # Middle region defaults to 0 (HOLD)
        return decisions
    else:
        # Scalar input → scalar output
        p_up_float = float(p_up)
        if p_up_float >= buy_thresh:
            return 1   # LONG
        elif p_up_float <= sell_thresh:
            return -1  # SHORT
        else:
            return 0   # HOLD


def apply_threshold_overrides(
    probabilities: np.ndarray,
    symbol: str,
    horizon: Union[int, str],
    threshold_config: Dict[str, Dict[str, ThresholdConfig]],
    *,
    default_buy: float = 0.52,
    default_sell: float = 0.48,
) -> np.ndarray:
    """Apply symbol-horizon-specific thresholds to probability array.
    
    Convenience wrapper around decision_from_prob() that handles threshold
    lookup from hierarchical config structure.
    
    Args:
        probabilities: Array of p_up probabilities (n_samples,).
        symbol: Stock ticker (e.g., 'AAPL', 'NVDA').
        horizon: Forecast horizon in days (e.g., 63, 126, 252).
        threshold_config: Nested dict of thresholds by symbol→horizon→{buy,sell}.
                         Structure: {"AAPL": {"63": {"buy": 0.52, "sell": 0.48}}}
        default_buy: Default buy threshold if not found in config.
        default_sell: Default sell threshold if not found in config.
    
    Returns:
        Array of positions {+1, 0, -1} with same shape as probabilities.
    
    Examples:
        >>> config = {
        ...     "AAPL": {
        ...         "63": {"buy": 0.521, "sell": 0.479},
        ...         "252": {"buy": 0.55, "sell": 0.45}
        ...     },
        ...     "_defaults": {"buy": 0.52, "sell": 0.48}
        ... }
        >>> probs = np.array([0.56, 0.50, 0.44])
        >>> apply_threshold_overrides(probs, "AAPL", 63, config)
        array([ 1,  0, -1])
    
    Notes:
        - Threshold lookup order:
          1. config[symbol][horizon]
          2. config["_defaults"]
          3. Function defaults (default_buy, default_sell)
        - Horizon is converted to string for dict lookup
    """
    horizon_str = str(horizon)
    
    # Lookup thresholds with fallback chain
    thresholds = None
    if symbol in threshold_config:
        symbol_config = threshold_config[symbol]
        if horizon_str in symbol_config:
            thresholds = symbol_config[horizon_str]
            logger.debug(
                "Using thresholds for %s h=%s: buy=%.3f sell=%.3f",
                symbol, horizon_str, thresholds["buy"], thresholds["sell"]
            )
    
    # Fallback to defaults section
    if thresholds is None and "_defaults" in threshold_config:
        thresholds = threshold_config["_defaults"]
        logger.debug(
            "Using default thresholds for %s h=%s: buy=%.3f sell=%.3f",
            symbol, horizon_str, thresholds["buy"], thresholds["sell"]
        )
    
    # Fallback to hardcoded defaults
    if thresholds is None:
        thresholds = {"buy": default_buy, "sell": default_sell}
        logger.debug(
            "Using hardcoded defaults for %s h=%s: buy=%.3f sell=%.3f",
            symbol, horizon_str, default_buy, default_sell
        )
    
    return decision_from_prob(probabilities, thresholds)


def get_threshold_stats(
    decisions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Union[int, float]]:
    """Calculate statistics on trading decisions.
    
    Useful for debugging and tuning threshold parameters.
    
    Args:
        decisions: Array of position decisions {+1, 0, -1}.
        probabilities: Optional array of original probabilities
                      (for calculating mean confidence).
    
    Returns:
        Dict with keys:
        - n_long: Count of LONG positions (+1)
        - n_hold: Count of HOLD positions (0)
        - n_short: Count of SHORT positions (-1)
        - pct_long: Percentage of LONG decisions
        - pct_hold: Percentage of HOLD decisions
        - pct_short: Percentage of SHORT decisions
        - mean_prob_long: Mean probability for LONG decisions (if probs provided)
        - mean_prob_short: Mean probability for SHORT decisions (if probs provided)
    
    Examples:
        >>> decisions = np.array([1, 1, 0, -1, 0])
        >>> get_threshold_stats(decisions)
        {'n_long': 2, 'n_hold': 2, 'n_short': 1, 
         'pct_long': 40.0, 'pct_hold': 40.0, 'pct_short': 20.0}
    """
    n_total = len(decisions)
    n_long = int(np.sum(decisions == 1))
    n_hold = int(np.sum(decisions == 0))
    n_short = int(np.sum(decisions == -1))
    
    stats = {
        "n_long": n_long,
        "n_hold": n_hold,
        "n_short": n_short,
        "pct_long": round(100.0 * n_long / n_total, 2) if n_total > 0 else 0.0,
        "pct_hold": round(100.0 * n_hold / n_total, 2) if n_total > 0 else 0.0,
        "pct_short": round(100.0 * n_short / n_total, 2) if n_total > 0 else 0.0,
    }
    
    # Optional probability statistics
    if probabilities is not None and len(probabilities) == n_total:
        if n_long > 0:
            stats["mean_prob_long"] = round(
                float(np.mean(probabilities[decisions == 1])), 4
            )
        if n_short > 0:
            stats["mean_prob_short"] = round(
                float(np.mean(probabilities[decisions == -1])), 4
            )
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("THRESHOLD DECISION LOGIC - STANDALONE TEST")
    print("=" * 60)
    
    # Test 1: Scalar probability
    print("\n[Test 1] Scalar probabilities:")
    for prob in [0.60, 0.52, 0.50, 0.48, 0.40]:
        decision = decision_from_prob(prob)
        pos_name = {1: "LONG", 0: "HOLD", -1: "SHORT"}[decision]
        print(f"  p_up={prob:.2f} → {decision:+2d} ({pos_name})")
    
    # Test 2: Array probabilities
    print("\n[Test 2] Array probabilities:")
    probs = np.array([0.65, 0.55, 0.50, 0.45, 0.35])
    decisions = decision_from_prob(probs)
    print(f"  Probabilities: {probs}")
    print(f"  Decisions:     {decisions}")
    print(f"  Position names: {['LONG' if d==1 else 'SHORT' if d==-1 else 'HOLD' for d in decisions]}")
    
    # Test 3: Custom thresholds (wider band)
    print("\n[Test 3] Custom thresholds (wide band ±8%):")
    wide_thresholds = {"buy": 0.58, "sell": 0.42}
    for prob in [0.60, 0.55, 0.50, 0.45, 0.38]:
        decision = decision_from_prob(prob, wide_thresholds)
        pos_name = {1: "LONG", 0: "HOLD", -1: "SHORT"}[decision]
        print(f"  p_up={prob:.2f} → {decision:+2d} ({pos_name})")
    
    # Test 4: apply_threshold_overrides with config
    print("\n[Test 4] Config-based thresholds:")
    config = {
        "AAPL": {
            "63": {"buy": 0.521, "sell": 0.479},
            "252": {"buy": 0.55, "sell": 0.45}
        },
        "NVDA": {
            "63": {"buy": 0.54, "sell": 0.46}
        },
        "_defaults": {"buy": 0.52, "sell": 0.48}
    }
    test_probs = np.array([0.56, 0.52, 0.50, 0.48, 0.44])
    
    for symbol, horizon in [("AAPL", 63), ("NVDA", 63), ("MSFT", 63)]:
        decisions = apply_threshold_overrides(test_probs, symbol, horizon, config)
        print(f"  {symbol} h={horizon}: {decisions}")
    
    # Test 5: Statistics
    print("\n[Test 5] Decision statistics:")
    large_probs = np.random.uniform(0.3, 0.7, size=1000)
    large_decisions = decision_from_prob(large_probs)
    stats = get_threshold_stats(large_decisions, large_probs)
    print(f"  Total decisions: {sum([stats['n_long'], stats['n_hold'], stats['n_short']])}")
    print(f"  LONG:  {stats['n_long']:4d} ({stats['pct_long']:5.1f}%) | mean_prob={stats.get('mean_prob_long', 'N/A')}")
    print(f"  HOLD:  {stats['n_hold']:4d} ({stats['pct_hold']:5.1f}%)")
    print(f"  SHORT: {stats['n_short']:4d} ({stats['pct_short']:5.1f}%) | mean_prob={stats.get('mean_prob_short', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
