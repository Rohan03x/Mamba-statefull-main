"""Backtesting utilities for Stage-B model outputs.

This module provides a vectorized backtest engine that supports multiple
signal construction rules, transaction cost modeling, and rich performance
metrics. It is intentionally symbol- and horizon-agnostic so the Stage-B
pipeline can plug predictions directly into realistic P&L simulations.

🔧 STEP 7 FIX: Now includes regime-aware threshold filtering to match Optuna.
When regime_threshold params are provided, backtest filters weak signals
the same way Optuna does during optimization.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SignalInput = Tuple[pd.Series, pd.Series, pd.Series, pd.Series, float]

# =============================================================================
# STEP 7: Regime Detection & Threshold Filtering (matches Optuna)
# =============================================================================

def detect_regime(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Detect market regime from returns.
    
    Regime is an INPUT FEATURE (not optimization target).
    Uses rolling statistics to classify each timestamp:
    - 0: Bull (positive returns, low volatility)
    - 1: Bear (negative returns, moderate volatility)  
    - 2: Crisis (high volatility, regardless of direction)
    
    Args:
        returns: Series of returns (daily or period)
        window: Rolling window for statistics
        
    Returns:
        Series of regime labels (0=bull, 1=bear, 2=crisis)
    """
    if len(returns) < window:
        return pd.Series(0, index=returns.index, dtype=int)
    
    roll_mean = returns.rolling(window=window, min_periods=1).mean()
    roll_vol = returns.rolling(window=window, min_periods=1).std()
    
    vol_median = roll_vol.median()
    vol_high = roll_vol.quantile(0.85)
    
    regime = pd.Series(0, index=returns.index, dtype=int)
    
    for i in range(len(returns)):
        vol = roll_vol.iloc[i] if not np.isnan(roll_vol.iloc[i]) else vol_median
        ret = roll_mean.iloc[i] if not np.isnan(roll_mean.iloc[i]) else 0
        
        if vol > vol_high:
            regime.iloc[i] = 2  # Crisis
        elif ret < 0:
            regime.iloc[i] = 1  # Bear
        else:
            regime.iloc[i] = 0  # Bull
    
    return regime


def apply_regime_thresholds(
    predictions: pd.Series,
    regime: pd.Series,
    threshold: float,
    bull_mult: float = 0.8,
    bear_mult: float = 1.5,
    crisis_mult: float = 3.0,
    *,
    bull_short_mult: Optional[float] = None,
    bear_short_mult: Optional[float] = None,
    crisis_short_mult: Optional[float] = None,
    conf: Optional[pd.Series] = None,
    conf_threshold: Optional[float] = None,
    vol_scaler: Optional[float] = None,
    vol_ratio: Optional[pd.Series] = None,
) -> pd.Series:
    """Apply regime-aware thresholds to filter weak signals.

    Backwards-compatible implementation:
    - If only (bull_mult, bear_mult, crisis_mult) are provided, applies a symmetric
      threshold for long/short.
    - If *_short_mult are provided, uses separate long/short thresholds.
    - If conf/conf_threshold are provided, gates signals (conf < conf_threshold => 0).
    - If vol_scaler and vol_ratio are provided, normalizes predictions by volatility:
        pred_norm = pred / (vol_ratio ** vol_scaler)
    """

    pred = predictions.astype(float)
    reg = regime.reindex(pred.index).fillna(0).astype(int)

    # Optional volatility normalization (matches Stage-B optimizer semantics).
    if vol_scaler is not None and vol_ratio is not None:
        vr = vol_ratio.reindex(pred.index).astype(float)
        vr = vr.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
        scale = np.power(vr.to_numpy(dtype=float), float(vol_scaler))
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        pred = pd.Series(pred.to_numpy(dtype=float) / scale, index=pred.index)

    # Long threshold per regime.
    T_long = np.where(
        reg.to_numpy() == 0,
        float(threshold) * float(bull_mult),
        np.where(
            reg.to_numpy() == 1,
            float(threshold) * float(bear_mult),
            float(threshold) * float(crisis_mult),
        ),
    )

    # Short threshold per regime.
    bs = float(bull_short_mult) if bull_short_mult is not None else float(bull_mult)
    bes = float(bear_short_mult) if bear_short_mult is not None else float(bear_mult)
    cs = float(crisis_short_mult) if crisis_short_mult is not None else float(crisis_mult)
    T_short = np.where(
        reg.to_numpy() == 0,
        float(threshold) * bs,
        np.where(
            reg.to_numpy() == 1,
            float(threshold) * bes,
            float(threshold) * cs,
        ),
    )

    pred_arr = pred.to_numpy(dtype=float)
    direction = np.where(pred_arr > T_long, 1.0, np.where(pred_arr < -T_short, -1.0, 0.0))

    # Optional confidence gating.
    if conf is not None and conf_threshold is not None:
        c = conf.reindex(pred.index).astype(float).clip(0.0, 1.0)
        direction = np.where(c.to_numpy(dtype=float) < float(conf_threshold), 0.0, direction)

    return pd.Series(direction, index=pred.index, dtype=float)


def load_optuna_thresholds(symbol: str, horizon: int) -> Optional[Dict[str, float]]:
    """Load Optuna-optimized threshold parameters if available.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        horizon: Forecast horizon (e.g., 63)

    Returns:
        Dict with threshold params or None if not found
    """
    optuna_path = Path("artifacts/optuna") / f"{symbol.upper()}_h{horizon}_optuna.json"
    if not optuna_path.exists():
        return None
    
    try:
        with open(optuna_path) as f:
            params = json.load(f)
        
        # Extract threshold params
        threshold = params.get("threshold")
        bull_mult = params.get("bull_mult")
        bear_mult = params.get("bear_mult")
        crisis_mult = params.get("crisis_mult")
        
        if threshold is not None:
            return {
                "threshold": threshold,
                "bull_mult": bull_mult or 0.8,
                "bear_mult": bear_mult or 1.5,
                "crisis_mult": crisis_mult or 3.0,
            }
    except Exception:
        pass
    
    return None


def _sig_zscore(mu: pd.Series, sigma: pd.Series, rho: pd.Series, p_up: pd.Series, k: float) -> pd.Series:
    denom = sigma.abs().replace(0.0, np.nan).fillna(sigma.median() or 0.02)
    return np.tanh(k * (mu / denom))


def _sig_prob(mu: pd.Series, sigma: pd.Series, rho: pd.Series, p_up: pd.Series, k: float) -> pd.Series:
    return 2.0 * (p_up - 0.5)


def _sig_hybrid(mu: pd.Series, sigma: pd.Series, rho: pd.Series, p_up: pd.Series, k: float) -> pd.Series:
    z_component = _sig_zscore(mu, sigma, rho, p_up, k)
    prob_component = _sig_prob(mu, sigma, rho, p_up, k)
    return 0.5 * z_component + 0.5 * prob_component


SIGNAL_RULES = {
    "zscore": _sig_zscore,
    "prob": _sig_prob,
    "hybrid": _sig_hybrid,
}


class BacktestEngine:
    """Simulate portfolio P&L from Stage-B forecasts."""

    def __init__(self, price_data: pd.DataFrame, fee: float = 0.0, slippage_bp: float = 0.0) -> None:
        if price_data is None or price_data.empty:
            raise ValueError("price_data must be a non-empty DataFrame")
        self.price_history = price_data.sort_index()
        self.close_series = self._resolve_close_column(self.price_history)
        self.daily_returns = self.close_series.pct_change().fillna(0.0)
        self.fee = fee
        self.slippage_bp = slippage_bp
        self.trading_days = 252

    def run(
        self,
        preds_df: pd.DataFrame,
        horizon: int,
        strategy_cfg: Mapping[str, Any],
        symbol: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if preds_df is None or preds_df.empty:
            raise ValueError("preds_df must be populated with model outputs")
        df = preds_df.sort_index().copy()
        rule = str(strategy_cfg.get("signal_rule", "hybrid")).lower()
        signal_fn = SIGNAL_RULES.get(rule, _sig_hybrid)
        k = float(strategy_cfg.get("k", 1.0))
        leverage = float(strategy_cfg.get("leverage", 1.0))
        max_exposure = float(strategy_cfg.get("max_exposure", leverage))
        conf_weighting = bool(strategy_cfg.get("confidence_weighting", True))
        holding_period = int(strategy_cfg.get("holding_period_days", horizon))
        overlap = bool(strategy_cfg.get("overlap", True))
        sigma_floor = float(strategy_cfg.get("sigma_floor", 0.02))

        # ═══════════════════════════════════════════════════════════════════════════════
        # VOL_SCALER → POSITION CAP: Minimal wiring for risk-aware sizing
        # vol_scaler ∈ [0.0, 1.0] acts as a multiplicative cap on max_exposure.
        # When vol_scaler is low (high vol regime), positions are scaled down.
        # This is a bridge until Workstream 7 (robust optimization) is complete.
        # ═══════════════════════════════════════════════════════════════════════════════
        vol_scaler_cap = strategy_cfg.get("vol_scaler_position_cap")
        if vol_scaler_cap is not None:
            # Clamp to [0.1, 1.0] to prevent division by zero / too small positions
            effective_cap = float(np.clip(vol_scaler_cap, 0.1, 1.0))
            max_exposure = max_exposure * effective_cap
            leverage = min(leverage, max_exposure)

        mu = df.get("mu_hat", pd.Series(index=df.index, data=0.0)).astype(float)
        sigma = df.get("sigma_hat", pd.Series(index=df.index, data=sigma_floor)).astype(float)
        sigma = sigma.replace(0.0, np.nan).fillna(sigma_floor)
        p_up = df.get("p_up", pd.Series(index=df.index, data=0.5)).astype(float).clip(0.0, 1.0)
        rho = df.get("rho", pd.Series(index=df.index, data=1.0)).astype(float).clip(0.0, 1.0)

        # =====================================================================
        # 🔧 STEP 7 FIX: Apply Optuna regime-aware thresholds
        # =====================================================================
        # Check for Optuna threshold params in strategy_cfg or load from file
        regime_threshold_enabled = strategy_cfg.get("regime_threshold_enabled", True)
        threshold_params = strategy_cfg.get("regime_threshold_params")
        
        if threshold_params is None and symbol is not None and regime_threshold_enabled:
            # Try to load from Optuna cache
            threshold_params = load_optuna_thresholds(symbol, horizon)
        
        if threshold_params is not None and regime_threshold_enabled:
            # Use regime-aware threshold filtering (matches Optuna objective)
            actual_returns = df.get("actual_return", self.daily_returns.reindex(df.index))
            if actual_returns is None or actual_returns.empty:
                actual_returns = self.daily_returns.reindex(df.index).fillna(0.0)
            
            regime = detect_regime(actual_returns.fillna(0.0))
            vol_ratio = None
            try:
                vol20 = actual_returns.rolling(window=20, min_periods=5).std()
                vol252 = actual_returns.rolling(window=252, min_periods=20).std()
                vol_ratio = (vol20 / vol252.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            except Exception:
                vol_ratio = None

            direction = apply_regime_thresholds(
                predictions=mu,
                regime=regime,
                threshold=float(threshold_params["threshold"]),
                bull_mult=float(threshold_params.get("bull_mult", threshold_params.get("bull_long_mult", 0.8))),
                bear_mult=float(threshold_params.get("bear_mult", threshold_params.get("bear_long_mult", 1.5))),
                crisis_mult=float(threshold_params.get("crisis_mult", threshold_params.get("crisis_long_mult", 3.0))),
                bull_short_mult=threshold_params.get("bull_short_mult"),
                bear_short_mult=threshold_params.get("bear_short_mult"),
                crisis_short_mult=threshold_params.get("crisis_short_mult"),
                conf=rho,
                conf_threshold=threshold_params.get("conf_threshold"),
                vol_scaler=threshold_params.get("vol_scaler"),
                vol_ratio=vol_ratio,
            )
            # Use direction directly as weights (scaled by leverage)
            raw_signal = direction.astype(float)
            if conf_weighting:
                raw_signal = raw_signal * rho
            weights = np.clip(raw_signal * leverage, -max_exposure, max_exposure).fillna(0.0)
            
            # Store regime info for diagnostics
            df["regime"] = regime
            # Keep this diagnostic as the LONG threshold (short threshold can differ).
            _bull_long = threshold_params.get("bull_mult", threshold_params.get("bull_long_mult", 0.8))
            _bear_long = threshold_params.get("bear_mult", threshold_params.get("bear_long_mult", 1.5))
            _crisis_long = threshold_params.get("crisis_mult", threshold_params.get("crisis_long_mult", 3.0))
            df["regime_threshold"] = np.where(
                regime == 0, threshold_params["threshold"] * _bull_long,
                np.where(
                    regime == 1, threshold_params["threshold"] * _bear_long,
                    threshold_params["threshold"] * _crisis_long,
                ),
            )
        else:
            # Legacy mode: use continuous signal without threshold filtering
            raw_signal = signal_fn(mu, sigma, rho, p_up, k)
            if conf_weighting:
                raw_signal = raw_signal * rho
            weights = np.clip(raw_signal * leverage, -max_exposure, max_exposure).fillna(0.0)

        turnover = weights.diff().abs().fillna(weights.abs())
        daily_returns = self.daily_returns.reindex(weights.index).fillna(0.0)
        if overlap and holding_period > 1:
            lagged = [weights.shift(i) for i in range(holding_period)]
            exposure = pd.concat(lagged, axis=1).mean(axis=1).fillna(0.0)
        else:
            exposure = weights
        gross_returns = exposure * daily_returns
        fee_cost = self.fee * (turnover > 0).astype(float)
        slip_cost = self.slippage_bp * turnover
        costs = fee_cost + slip_cost
        net_returns = gross_returns - costs

        equity = pd.DataFrame(
            {
                "target_weight": weights,
                "avg_weight": exposure,
                "daily_return": daily_returns,
                "gross_return": gross_returns,
                "costs": costs,
                "net_return": net_returns,
                "turnover": turnover,
            }
        )
        equity["equity_curve"] = (1.0 + equity["net_return"]).cumprod()
        equity["cum_pnl"] = equity["net_return"].cumsum()
        if "actual_return" in df:
            equity["actual_return"] = df["actual_return"].reindex(equity.index)
        if "hf_agg_score" in df:
            equity["hf_agg_score"] = df["hf_agg_score"].reindex(equity.index)
        metrics = self._compute_metrics(net_returns, weights, df, horizon)
        return equity, metrics

    def _compute_metrics(self, net_returns: pd.Series, weights: pd.Series, preds: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """Compute performance metrics.
        
        🔧 FIX: For horizons > 1, compute Sharpe on horizon-aligned strategy returns
        (direction * forward_return) rather than daily returns. This correctly measures
        whether the model's directional prediction over the forecast horizon was profitable.
        
        The daily-based net_returns are still used for equity curve and sortino (which
        makes sense for actual P&L simulation), but the Sharpe for model evaluation
        uses the forward returns when available.
        """
        # For model evaluation: use horizon-aligned returns if available
        if horizon > 1 and "actual_return" in preds:
            forward_returns = preds["actual_return"].reindex(weights.index).fillna(0.0)
            # Strategy return = direction * forward_return
            # direction = sign of position (weights)
            direction = np.sign(weights)
            strategy_returns = direction * forward_returns
            # Only count non-neutral positions for Sharpe
            active_mask = direction != 0
            if active_mask.sum() > 0:
                active_strategy = strategy_returns[active_mask]
                sharpe = self._sharpe_horizon(active_strategy, horizon)
                # 🔧 FIX: Use non-overlapping samples for hit rate too
                non_overlapping_active = active_strategy.iloc[::horizon]
                hit_rate = float((non_overlapping_active > 0).mean()) if len(non_overlapping_active) > 0 else 0.0
            else:
                sharpe = 0.0
                hit_rate = 0.0
        else:
            # Daily mode (horizon=1 or no forward returns available)
            sharpe = self._sharpe(net_returns)
            hit_rate = float((net_returns > 0).mean())
        
        sortino = self._sortino(net_returns)
        coverage = float(1.0 - preds.get("actual_return", pd.Series(index=net_returns.index)).isna().mean())
        turnover = float(weights.diff().abs().fillna(weights.abs()).mean())
        rwa = self._return_weighted_accuracy(weights, preds.get("actual_return"))
        ece, brier = self._calibration_metrics(preds)
        stability = self._stability(weights)
        max_dd = self._max_drawdown((1.0 + net_returns).cumprod())
        drift_on, drift_off = self._drift_stats(net_returns, preds.get("drift_flag"))
        metrics = {
            "sharpe": sharpe,
            "sortino": sortino,
            "hit_rate": hit_rate,
            "turnover": turnover,
            "rwa": rwa,
            "ece": ece,
            "brier": brier,
            "stability": stability,
            "max_drawdown": max_dd,
            "drift_on": drift_on,
            "drift_off": drift_off,
            "coverage": coverage,
            "sample_size": float(net_returns.notna().sum()),
        }
        metrics["composite"] = 0.6 * sharpe + 0.15 * rwa + 0.15 * (1.0 - ece) + 0.1 * stability
        return metrics

    def _resolve_close_column(self, price_data: pd.DataFrame) -> pd.Series:
        for candidate in ("close", "adj_close", "adjclose", "price"):
            if candidate in price_data.columns:
                return price_data[candidate].astype(float)
        return price_data.iloc[:, 0].astype(float)

    def _sharpe(self, returns: pd.Series) -> float:
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        scale = np.sqrt(self.trading_days)
        return float(returns.mean() / (std + 1e-9) * scale)

    def _sharpe_horizon(self, strategy_returns: pd.Series, horizon: int) -> float:
        """Compute Sharpe ratio for horizon-aligned strategy returns.
        
        🔧 FIX: For multi-day horizons, we MUST sample only non-overlapping periods.
        Using all overlapping returns artificially deflates std (due to ~98% autocorrelation)
        which inflates Sharpe by ~sqrt(horizon).
        
        Args:
            strategy_returns: direction * forward_return for each prediction
            horizon: forecast horizon in days
            
        Returns:
            Annualized Sharpe ratio properly scaled for the horizon (capped at ±5)
        """
        MAX_SHARPE = 5.0  # Cap to prevent extreme values from noisy low-sample estimates
        
        if strategy_returns.empty or len(strategy_returns) < horizon:
            return 0.0
        
        # 🔧 CRITICAL FIX: Sample only non-overlapping returns every H days
        # This avoids the autocorrelation problem that inflates Sharpe by ~sqrt(H)
        non_overlapping = strategy_returns.iloc[::horizon]
        
        if len(non_overlapping) < 2:
            return 0.0
            
        std = non_overlapping.std()
        if std == 0 or np.isnan(std):
            return 0.0
        
        # Number of non-overlapping periods per year
        periods_per_year = max(1, self.trading_days // horizon)
        scale = np.sqrt(periods_per_year)
        raw_sharpe = float(non_overlapping.mean() / (std + 1e-9) * scale)
        return float(np.clip(raw_sharpe, -MAX_SHARPE, MAX_SHARPE))

    def _sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < 0]
        denom = downside.std()
        if denom == 0 or np.isnan(denom):
            return 0.0
        scale = np.sqrt(self.trading_days)
        return float(returns.mean() / (denom + 1e-9) * scale)

    def _return_weighted_accuracy(self, weights: pd.Series, actual_return: Optional[pd.Series]) -> float:
        if actual_return is None:
            return 0.0
        aligned = actual_return.reindex(weights.index).fillna(0.0)
        direction_match = (np.sign(weights) == np.sign(aligned)).astype(float)
        weights_abs = aligned.abs() + 1e-6
        return float(np.average(direction_match, weights=weights_abs))

    def _calibration_metrics(self, preds: pd.DataFrame) -> Tuple[float, float]:
        if "p_up" not in preds or "actual_return" not in preds:
            return 0.5, 0.25
        probs = preds["p_up"].clip(0.0, 1.0)
        actual_binary = (preds["actual_return"] > 0).astype(float)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        total = len(probs)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            bucket = probs[mask]
            if bucket.empty:
                continue
            acc = actual_binary[mask].mean()
            ece += (bucket.size / total) * abs(bucket.mean() - acc)
        brier = float(np.mean((probs - actual_binary) ** 2))
        return float(ece), brier

    def _stability(self, weights: pd.Series) -> float:
        if weights.empty:
            return 0.0
        mean = weights.mean()
        std = weights.std()
        if std == 0:
            return 1.0
        return float(1.0 - np.clip(std / (abs(mean) + 1e-6), 0.0, 1.0))

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        if equity_curve.empty:
            return 0.0
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak.replace(0.0, np.nan)
        return float(drawdown.min())

    def _drift_stats(self, net_returns: pd.Series, drift_flag: Optional[pd.Series]) -> Tuple[float, float]:
        if drift_flag is None:
            return 0.0, 0.0
        flags = drift_flag.reindex(net_returns.index).fillna(0.0)
        on = net_returns[flags >= 1]
        off = net_returns[flags < 1]
        return float(on.mean()) if not on.empty else 0.0, float(off.mean()) if not off.empty else 0.0

    @staticmethod
    def compute_return_metrics(returns: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {}
        mean = returns.mean()
        std = returns.std()
        sharpe = float(mean / (std + 1e-9) * np.sqrt(252)) if std else 0.0
        downside = returns[returns < 0]
        sortino = float(mean / (downside.std() + 1e-9) * np.sqrt(252)) if not downside.empty else 0.0
        hit_rate = float((returns > 0).mean())
        equity = (1.0 + returns).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity - peak) / peak.replace(0.0, np.nan)).min())
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "hit_rate": hit_rate,
            "max_drawdown": max_dd,
            "sample_size": float(len(returns)),
        }


def run_backtest(
    price_data: pd.DataFrame,
    preds_df: pd.DataFrame,
    horizon: int,
    strategy_cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    fee = float(strategy_cfg.get("fee_bp", 0.0)) / 10000.0
    slippage = float(strategy_cfg.get("slippage_bp", 0.0)) / 10000.0
    engine = BacktestEngine(price_data, fee=fee, slippage_bp=slippage)
    return engine.run(preds_df, horizon, strategy_cfg)


def aggregate_backtests(
    records: Sequence[Mapping[str, Any]],
    weight_mode: str = "equal",
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    return_cols: Dict[str, pd.Series] = {}
    for entry in records:
        symbol = entry.get("symbol", "NA")
        horizon = entry.get("horizon", 0)
        metrics = entry.get("metrics", {}) or {}
        equity = entry.get("equity")
        if isinstance(equity, pd.DataFrame) and "net_return" in equity:
            key = f"{symbol}_h{horizon}"
            return_cols[key] = equity["net_return"].copy()
        row = {"symbol": symbol, "horizon": horizon}
        row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    per_symbol = pd.DataFrame(rows)

    portfolio_metrics: Dict[str, float] = {}
    if return_cols:
        returns = pd.concat(return_cols, axis=1).fillna(0.0)
        if weight_mode == "vol":
            vol = returns.std().replace(0.0, np.nan)
            inv_vol = 1.0 / vol
            weights = inv_vol / inv_vol.sum()
        else:
            weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
        portfolio_metrics = BacktestEngine.compute_return_metrics(portfolio_returns)
        portfolio_metrics["weight_mode"] = weight_mode
    return {"per_symbol": per_symbol, "portfolio": portfolio_metrics}


__all__ = [
    "BacktestEngine",
    "SIGNAL_RULES",
    "aggregate_backtests",
    "run_backtest",
]
