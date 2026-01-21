"""Regime detection utilities with hysteresis to avoid rapid flapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeRouter:
    """Detect high/low volatility and bull/bear trend regimes with hysteresis."""

    hysteresis_days: int = 10
    state: Optional[str] = None
    counter: int = 0

    def detect(self, prices: pd.Series) -> str:
        """Classify the current regime based on recent prices.

        Args:
            prices: Price series indexed by date.

        Returns:
            Regime label among ``{"high_bull", "high_bear", "low_bull", "low_bear"}``.
        """

        clean_prices = prices.dropna()
        # Require at least 20 days for moving averages (matches learn_module_weights.py threshold)
        min_days_for_regime = 20
        if clean_prices.size < min_days_for_regime:
            # Not enough data for regime detection; default to neutral low_bull.
            return "low_bull"

        returns = clean_prices.pct_change()
        
        # Volatility detection: Use 21-day rolling std vs historical average
        realized_vol = returns.rolling(21, min_periods=10).std()
        if realized_vol.dropna().empty:
            vol_label = "low"
        else:
            # Compare recent vol (last 5 days avg) vs full window average
            recent_vol_series = realized_vol.iloc[-5:].mean()
            baseline_vol_series = realized_vol.mean()
            
            recent_vol = recent_vol_series.item() if hasattr(recent_vol_series, 'item') else float(recent_vol_series)
            baseline_vol = baseline_vol_series.item() if hasattr(baseline_vol_series, 'item') else float(baseline_vol_series)
            
            if np.isnan(recent_vol) or np.isnan(baseline_vol):
                vol_label = "low"
            else:
                # High volatility if recent vol is 1.2x above average
                vol_label = "high" if recent_vol > baseline_vol * 1.2 else "low"

        # Trend detection: Use adaptive period based on data length
        # Longer windows = longer trend analysis for better regime classification
        if len(clean_prices) >= 252:  # >= 1 year of data
            trend_period = min(63, len(clean_prices) - 5)  # Use 63-day (3 months) trend
        elif len(clean_prices) >= 126:  # >= 6 months  
            trend_period = min(42, len(clean_prices) - 5)  # Use 42-day (2 months) trend
        else:
            trend_period = min(21, len(clean_prices) - 5)  # Use 21-day (1 month) trend
            
        trend = clean_prices.pct_change(trend_period)
        trend_smoothed = trend.rolling(5, min_periods=1).mean()
        latest_trend_series = trend_smoothed.iloc[-1]
        latest_trend = latest_trend_series.item() if hasattr(latest_trend_series, 'item') else float(latest_trend_series)
        
        # Adaptive thresholds based on trend period (annualized basis)
        # For 63-day: ±3% threshold (~23% annualized)
        # For 42-day: ±2.5% threshold (~29% annualized)  
        # For 21-day: ±1.5% threshold (~35% annualized)
        if trend_period >= 60:
            bull_threshold = 0.03   # 3% over ~3 months
            bear_threshold = -0.03
        elif trend_period >= 40:
            bull_threshold = 0.025  # 2.5% over ~2 months
            bear_threshold = -0.025
        else:
            bull_threshold = 0.015  # 1.5% over ~1 month  
            bear_threshold = -0.015
        
        # More nuanced trend classification with sideways regime
        if np.isnan(latest_trend):
            bull_label = "sideways"
        elif latest_trend > bull_threshold:
            bull_label = "bull"
        elif latest_trend < bear_threshold:
            bull_label = "bear"
        else:  # Small moves = sideways
            bull_label = "sideways"

        return f"{vol_label}_{bull_label}"

    def step(self, new_state: str) -> str:
        """Update router state with hysteresis and return the active regime."""

        if self.state is None or self.state == new_state:
            self.state = new_state
            self.counter = 0
            return self.state

        self.counter += 1
        if self.counter >= self.hysteresis_days:
            self.state = new_state
            self.counter = 0
        return self.state
