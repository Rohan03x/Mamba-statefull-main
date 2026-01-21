"""
Stage C: Policy Module
======================

Transforms LSTM prediction tapes into trading signals with:
- Threshold/coverage sweeps
- Static policy backtesting  
- Risk controls (volatility targeting, drawdown cooling)

This module operates entirely on the prediction tape exported by Stage B,
so you never need to re-run the LSTM for coverage/threshold experiments.
"""

from src.stage_c.stage_c_policy import (
    load_prediction_tape,
    sweep_thresholds,
    evaluate_threshold,
    find_optimal_threshold,
    run_policy_backtest,
    apply_static_policy_with_risk,
    plot_coverage_frontier,
    generate_sweep_report,
    annualised_sharpe,
    max_drawdown,
    hit_rate,
    calmar_ratio,
)

__all__ = [
    "load_prediction_tape",
    "sweep_thresholds",
    "evaluate_threshold",
    "find_optimal_threshold",
    "run_policy_backtest",
    "apply_static_policy_with_risk",
    "plot_coverage_frontier",
    "generate_sweep_report",
    "annualised_sharpe",
    "max_drawdown",
    "hit_rate",
    "calmar_ratio",
]
