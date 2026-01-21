"""
Backtesting module using backtesting.py library.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from .metrics import calculate_trading_metrics

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """Strategy using ML model predictions for trading signals."""

    def init(self):
        """Initialize strategy with predictions."""
        # Get predictions from data
        self.predictions = self.data.predictions
        self.probabilities = getattr(self.data, "probabilities", None)

        # Strategy parameters
        self.threshold = self.threshold
        self.position_limit = self.position_limit
        self.stop_loss = self.stop_loss
        self.take_profit = self.take_profit
        self.kelly_fraction = self.kelly_fraction
        self.use_proba = self.use_proba

        # Initialize position size
        self.position_size = 1.0

    def next(self):
        """Define trading logic for each step."""
        # Get current prediction
        current_pred = self.predictions[-1]

        # Determine position
        if self.use_proba and self.probabilities is not None:
            # Use classification probabilities
            prob_up = self.probabilities[-1]

            if prob_up > (0.5 + self.threshold):
                signal = 1
            elif prob_up < (0.5 - self.threshold):
                signal = -1
            else:
                signal = 0
        else:
            # Use regression predictions
            if current_pred > self.threshold:
                signal = 1
            elif current_pred < -self.threshold:
                signal = -1
            else:
                signal = 0

        # Calculate position size using Kelly criterion
        if self.kelly_fraction and signal != 0:
            win_prob = self.probabilities[-1] if self.probabilities is not None else 0.5
            if signal < 0:
                win_prob = 1 - win_prob

            # Estimated gain/loss ratio (can be improved)
            gain_loss_ratio = 1.0

            # Kelly fraction
            f = win_prob - (1 - win_prob) / gain_loss_ratio
            f = max(0, f * self.kelly_fraction)  # Fractional Kelly

            self.position_size = min(f, self.position_limit)
        else:
            self.position_size = self.position_limit

        # Set position
        self.position.size = signal * self.position_size

        # Set stop-loss and take-profit
        if self.position.size != 0:
            if self.stop_loss:
                self.position.sl = self.position.entry_price * \
                    (1 - self.stop_loss * signal)
            if self.take_profit:
                self.position.tp = self.position.entry_price * \
                    (1 + self.take_profit * signal)


class BacktestRunner:
    """Class for running and analyzing backtests."""

    def __init__(
        self,
        data: pd.DataFrame,
        predictions: Union[pd.Series, np.ndarray],
        probabilities: Optional[Union[pd.Series, np.ndarray]] = None,
        transaction_cost_bps: float = 3.0,
        position_limit: float = 1.0,
        threshold: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
        use_proba: bool = False,
        cash: float = 10000,
        commission: Optional[float] = None,
        margin: float = 1.0
    ):
        """
        Initialize backtester.

        Args:
            data: OHLCV DataFrame
            predictions: Model predictions
            probabilities: Classification probabilities (optional)
            transaction_cost_bps: Transaction costs in basis points
            position_limit: Maximum absolute position size
            threshold: Threshold for taking positions
            stop_loss: Stop-loss percentage (if None, no stop-loss)
            take_profit: Take-profit percentage (if None, no take-profit)
            kelly_fraction: Kelly fraction for position sizing (if None, use fixed size)
            use_proba: Whether to use probabilities for signals
            cash: Initial cash
            commission: Commission per trade (if None, use transaction_cost_bps)
            margin: Required margin
        """
        self.data = data.copy()

        # Add predictions to data
        self.data["predictions"] = predictions
        if probabilities is not None:
            self.data["probabilities"] = probabilities

        # Strategy parameters
        self.transaction_cost_bps = transaction_cost_bps
        self.position_limit = position_limit
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.kelly_fraction = kelly_fraction
        self.use_proba = use_proba

        # Backtest parameters
        self.cash = cash
        self.commission = commission or transaction_cost_bps / 10000
        self.margin = margin

        # Initialize backtest
        self.bt = Backtest(
            self.data,
            MLStrategy,
            cash=cash,
            commission=self.commission,
            margin=margin,
            exclusive_orders=True
        )

    def run(self) -> Dict:
        """
        Run backtest.

        Returns:
            Dictionary with backtest results
        """
        # Run backtest with strategy parameters
        results = self.bt.run(
            threshold=self.threshold,
            position_limit=self.position_limit,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            kelly_fraction=self.kelly_fraction,
            use_proba=self.use_proba
        )

        # Get trades
        trades = self.bt.trades

        # Calculate additional metrics
        returns = self.data["Close"].pct_change()
        positions = pd.Series(index=returns.index, data=0.0)

        for trade in trades:
            positions.loc[trade.entry_time:trade.exit_time] = trade.size

        metrics = calculate_trading_metrics(
            returns=returns,
            positions=positions,
            transaction_costs_bps=self.transaction_cost_bps
        )

        # Combine metrics
        combined_metrics = {
            **metrics,
            "exposure_time": results["Exposure Time [%]"],
            "equity_final": results["Equity Final [$]"],
            "equity_peak": results["Equity Peak [$]"],
            "return_pct": results["Return [%]"],
            "buy_hold_return_pct": results["Buy & Hold Return [%]"],
            "max_drawdown": results["Max. Drawdown [%]"] / 100,
            "avg_drawdown": results["Avg. Drawdown [%]"] / 100,
            "max_drawdown_duration": results["Max. Drawdown Duration"],
            "trades": len(trades)
        }

        return combined_metrics

    def save_results(
        self,
        results: Dict,
        save_dir: Path,
        prefix: str = ""
    ) -> None:
        """
        Save backtest results to files.

        Args:
            results: Dictionary of results
            save_dir: Directory to save results
            prefix: Prefix for filenames
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = save_dir / f"{prefix}metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=4)

        # Save equity curve
        equity_file = save_dir / f"{prefix}equity.csv"
        self.bt._equity_curve.to_csv(equity_file)

        # Save trades
        trades_file = save_dir / f"{prefix}trades.csv"
        pd.DataFrame(self.bt.trades).to_csv(trades_file)
