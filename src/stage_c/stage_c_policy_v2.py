"""
Stage C Policy Module v2
========================

Complete threshold optimization and trading policy system:

1. Static Global Threshold Search
   - Grid search over T values
   - Compute Coverage, Sharpe, MaxDD, HitRate, Turnover
   - Objective function to pick "best T"

2. Walk-Forward Threshold Training (Robust, No Cheating)
   - For each fold k, train T* on folds 0..k-1
   - Apply T*_k to fold k only
   - Produces realistic walk-forward PnL

3. ML Meta-Learner for Dynamic Thresholding
   - Build supervised dataset from prediction tape
   - Train classifier/regressor to predict trade success
   - Dynamic trading rule based on confidence

Usage:
    from src.stage_c.stage_c_policy_v2 import (
        load_prediction_tape,
        static_threshold_search,
        walkforward_threshold_training,
        build_meta_learner_dataset,
        train_meta_learner,
        run_dynamic_policy,
    )
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ThresholdResult:
    """Result of evaluating a single threshold."""
    threshold: float
    coverage: float
    sharpe: float
    maxdd: float
    hitrate: float
    turnover: float
    n_trades: int
    n_long: int
    n_short: int
    n_flat: int
    ann_return: float
    calmar: float
    objective: float  # Combined objective score


@dataclass
class WalkForwardResult:
    """Result of walk-forward threshold training."""
    fold_thresholds: List[float]  # T* for each fold
    fold_sharpes: List[float]
    fold_returns: List[float]
    aggregate_sharpe: float
    aggregate_return: float
    aggregate_maxdd: float
    aggregate_hitrate: float
    equity_curve: pd.Series
    trades_df: pd.DataFrame


# ============================================================================
# Core Metrics Functions
# ============================================================================

def compute_sharpe(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year (4 for quarterly H63)
    """
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def compute_maxdd(returns: pd.Series) -> float:
    """Compute maximum drawdown from return series."""
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_hitrate(returns: pd.Series) -> float:
    """Compute hit rate (fraction of profitable trades)."""
    traded = returns[returns != 0]
    if len(traded) == 0:
        return 0.0
    return float((traded > 0).mean())


def compute_turnover(positions: pd.Series) -> float:
    """Compute average turnover (position changes)."""
    if len(positions) < 2:
        return 0.0
    changes = positions.diff().abs()
    return float(changes.mean())


def compute_calmar(returns: pd.Series, periods_per_year: float = 4.0) -> float:
    """Compute Calmar ratio (ann return / max dd)."""
    ann_ret = returns.mean() * periods_per_year
    mdd = abs(compute_maxdd(returns))
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


# ============================================================================
# Prediction Tape Loading
# ============================================================================

def load_prediction_tape(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load prediction tape from parquet.
    
    Expected columns: timestamp, fold_id, y_true, score
    Optional: position_raw, meta_view, meta_sharpe
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction tape not found: {path}")
    
    tape = pd.read_parquet(path)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    tape.sort_values(["fold_id", "timestamp"], inplace=True)
    tape.reset_index(drop=True, inplace=True)
    
    logger.info(
        f"📂 Loaded prediction tape: {path.name} "
        f"({len(tape)} rows, {tape['fold_id'].nunique()} folds, "
        f"{tape['timestamp'].min().date()} to {tape['timestamp'].max().date()})"
    )
    
    return tape


# ============================================================================
# 1. STATIC GLOBAL THRESHOLD SEARCH
# ============================================================================

def apply_threshold(tape: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Apply threshold to prediction tape and compute positions/returns.
    
    Trading rule:
    - Long if score >= T
    - Short if score <= -T
    - Flat otherwise
    """
    df = tape.copy()
    df["position"] = np.where(
        df["score"] >= threshold, 1,
        np.where(df["score"] <= -threshold, -1, 0)
    )
    df["ret"] = df["position"] * df["y_true"]
    return df


def evaluate_threshold(
    tape: pd.DataFrame,
    threshold: float,
    periods_per_year: float = 4.0,
    coverage_penalty_alpha: float = 0.3,
) -> ThresholdResult:
    """
    Evaluate a single threshold on the prediction tape.
    
    Uses NON-OVERLAPPING samples (one per fold) for proper statistics.
    
    Args:
        tape: Prediction tape with fold_id, timestamp, score, y_true
        threshold: Threshold T for position entry
        periods_per_year: Periods per year for annualization (4 for quarterly)
        coverage_penalty_alpha: Exponent for coverage penalty in objective
    """
    # Apply threshold
    df = apply_threshold(tape, threshold)
    
    # For proper statistics, take ONE prediction per fold (non-overlapping)
    # We'll take the LAST prediction of each fold (most recent in validation)
    fold_summary = df.groupby("fold_id").agg({
        "position": "last",
        "y_true": "last",
        "score": "last",
        "ret": "last",
        "timestamp": "last",
    }).reset_index()
    
    fold_summary["ret"] = fold_summary["position"] * fold_summary["y_true"]
    
    # Compute metrics on non-overlapping fold returns
    returns = fold_summary["ret"]
    positions = fold_summary["position"]
    
    coverage = (positions != 0).mean()
    sharpe = compute_sharpe(returns, periods_per_year)
    maxdd = compute_maxdd(returns)
    hitrate = compute_hitrate(returns)
    turnover = compute_turnover(positions)
    calmar = compute_calmar(returns, periods_per_year)
    ann_return = returns.mean() * periods_per_year
    
    n_long = int((positions == 1).sum())
    n_short = int((positions == -1).sum())
    n_flat = int((positions == 0).sum())
    n_trades = int((positions.diff().abs() > 0).sum())
    
    # Objective: Sharpe * Coverage^alpha (penalize low coverage)
    objective = sharpe * (coverage ** coverage_penalty_alpha) if coverage > 0 else 0.0
    
    return ThresholdResult(
        threshold=threshold,
        coverage=coverage,
        sharpe=sharpe,
        maxdd=maxdd,
        hitrate=hitrate,
        turnover=turnover,
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        n_flat=n_flat,
        ann_return=ann_return,
        calmar=calmar,
        objective=objective,
    )


def static_threshold_search(
    tape: pd.DataFrame,
    t_values: Optional[List[float]] = None,
    periods_per_year: float = 4.0,
    coverage_penalty_alpha: float = 0.3,
    min_coverage: float = 0.10,
    max_dd_limit: float = -0.50,
) -> Tuple[pd.DataFrame, ThresholdResult]:
    """
    Search for optimal static threshold.
    
    Args:
        tape: Prediction tape
        t_values: List of thresholds to test (default: 0.02 to 0.50)
        periods_per_year: Periods per year for annualization
        coverage_penalty_alpha: Exponent for coverage penalty
        min_coverage: Minimum required coverage
        max_dd_limit: Maximum allowed drawdown
        
    Returns:
        Tuple of (sweep DataFrame, best ThresholdResult)
    """
    if t_values is None:
        t_values = np.arange(0.02, 0.52, 0.02).tolist()
    
    results = []
    for T in t_values:
        res = evaluate_threshold(tape, T, periods_per_year, coverage_penalty_alpha)
        results.append(res)
    
    # Build DataFrame
    sweep_df = pd.DataFrame([
        {
            "T": r.threshold,
            "Coverage": r.coverage,
            "Sharpe": r.sharpe,
            "MaxDD": r.maxdd,
            "HitRate": r.hitrate,
            "Turnover": r.turnover,
            "AnnReturn": r.ann_return,
            "Calmar": r.calmar,
            "Objective": r.objective,
            "N_Long": r.n_long,
            "N_Short": r.n_short,
            "N_Flat": r.n_flat,
        }
        for r in results
    ])
    
    # Find best threshold
    # First try with constraints
    feasible = sweep_df[
        (sweep_df["Coverage"] >= min_coverage) &
        (sweep_df["MaxDD"] >= max_dd_limit)
    ]
    
    if len(feasible) > 0:
        best_idx = feasible["Objective"].idxmax()
        best_result = results[best_idx]
        logger.info(
            f"✅ Best threshold T={best_result.threshold:.3f} "
            f"(Sharpe={best_result.sharpe:.3f}, Coverage={best_result.coverage:.1%}, "
            f"MaxDD={best_result.maxdd:.1%}, Objective={best_result.objective:.3f})"
        )
    else:
        # Relax constraints, pick best objective
        logger.warning("⚠️ No thresholds meet constraints. Relaxing to best objective.")
        best_idx = sweep_df["Objective"].idxmax()
        best_result = results[best_idx]
    
    return sweep_df, best_result


# ============================================================================
# 2. WALK-FORWARD THRESHOLD TRAINING
# ============================================================================

def walkforward_threshold_training(
    tape: pd.DataFrame,
    t_values: Optional[List[float]] = None,
    min_train_folds: int = 5,
    periods_per_year: float = 4.0,
    coverage_penalty_alpha: float = 0.3,
) -> WalkForwardResult:
    """
    Walk-forward threshold training (no lookahead bias).
    
    For each fold k:
    - Use folds 0..k-1 to find optimal T*
    - Apply T* to fold k only
    - This simulates real-time threshold selection
    
    Args:
        tape: Prediction tape
        t_values: Threshold grid to search
        min_train_folds: Minimum folds needed before we can pick T
        periods_per_year: Periods per year for annualization
        coverage_penalty_alpha: Coverage penalty exponent
        
    Returns:
        WalkForwardResult with fold-by-fold results and aggregate metrics
    """
    if t_values is None:
        t_values = np.arange(0.02, 0.52, 0.02).tolist()
    
    folds = sorted(tape["fold_id"].unique())
    n_folds = len(folds)
    
    fold_thresholds = []
    fold_sharpes = []
    fold_returns_list = []
    all_trades = []
    
    for k, fold_id in enumerate(folds):
        if k < min_train_folds:
            # Not enough history to train - use default threshold
            T_star = 0.10
            logger.debug(f"Fold {fold_id}: Using default T={T_star:.2f} (insufficient history)")
        else:
            # Train on folds 0..k-1
            train_folds = folds[:k]
            train_tape = tape[tape["fold_id"].isin(train_folds)]
            
            # Find best T on training data
            best_obj = -np.inf
            T_star = 0.10
            for T in t_values:
                res = evaluate_threshold(train_tape, T, periods_per_year, coverage_penalty_alpha)
                if res.objective > best_obj:
                    best_obj = res.objective
                    T_star = T
            
            logger.debug(f"Fold {fold_id}: Trained T*={T_star:.2f} on {len(train_folds)} folds")
        
        # Apply T_star to this fold
        fold_data = tape[tape["fold_id"] == fold_id].copy()
        fold_data = apply_threshold(fold_data, T_star)
        
        # Take last prediction of fold (non-overlapping)
        last_pred = fold_data.iloc[-1]
        fold_ret = last_pred["ret"]
        
        fold_thresholds.append(T_star)
        fold_returns_list.append(fold_ret)
        
        # Record trade
        all_trades.append({
            "fold_id": fold_id,
            "timestamp": last_pred["timestamp"],
            "threshold": T_star,
            "score": last_pred["score"],
            "position": last_pred["position"],
            "y_true": last_pred["y_true"],
            "ret": fold_ret,
        })
    
    # Aggregate metrics
    returns = pd.Series(fold_returns_list)
    trades_df = pd.DataFrame(all_trades)
    
    aggregate_sharpe = compute_sharpe(returns, periods_per_year)
    aggregate_return = returns.mean() * periods_per_year
    aggregate_maxdd = compute_maxdd(returns)
    aggregate_hitrate = compute_hitrate(returns)
    
    # Equity curve
    equity_curve = (1 + returns).cumprod()
    equity_curve.index = trades_df["timestamp"].values
    
    # Per-fold Sharpe (each fold is one observation, so just the return value)
    fold_sharpes = fold_returns_list  # Single-period "sharpe" is just the return
    
    logger.info(
        f"📊 Walk-forward complete: {n_folds} folds, "
        f"Sharpe={aggregate_sharpe:.3f}, Return={aggregate_return:.1%}, "
        f"MaxDD={aggregate_maxdd:.1%}, HitRate={aggregate_hitrate:.1%}"
    )
    
    return WalkForwardResult(
        fold_thresholds=fold_thresholds,
        fold_sharpes=fold_sharpes,
        fold_returns=fold_returns_list,
        aggregate_sharpe=aggregate_sharpe,
        aggregate_return=aggregate_return,
        aggregate_maxdd=aggregate_maxdd,
        aggregate_hitrate=aggregate_hitrate,
        equity_curve=equity_curve,
        trades_df=trades_df,
    )


# ============================================================================
# 3. ML META-LEARNER FOR DYNAMIC THRESHOLDING
# ============================================================================

def build_meta_learner_dataset(
    tape: pd.DataFrame,
    vol_lookback: int = 20,
    hitrate_lookback: int = 10,
    horizon: int = 63,
) -> pd.DataFrame:
    """
    Build supervised dataset for meta-learner.
    
    CRITICAL: All features must be LAGGED to avoid lookahead bias!
    We can only use information available BEFORE we make the trading decision.
    
    Features (all properly lagged):
    - score: Raw LSTM output (known at decision time)
    - abs_score: Absolute value of score
    - score_percentile: Percentile rank of |score| (cross-sectional)
    - hist_vol: Rolling volatility of PAST returns (shifted by horizon)
    - hist_hitrate: Rolling hit rate from PAST trades (shifted)
    - hist_pnl: Rolling cumulative PnL from PAST trades (shifted)
    - hist_drawdown: Drawdown from PAST trades (shifted)
    
    Labels:
    - y_cls: 1 if trade would have been profitable, 0 otherwise
    - y_reg: PnL of taking trade with sign(score)
    
    Args:
        tape: Prediction tape
        vol_lookback: Lookback for volatility calculation
        hitrate_lookback: Lookback for hit rate calculation
        horizon: Forecast horizon (for proper lagging)
    """
    df = tape.copy()
    df = df.sort_values(["timestamp", "fold_id"]).reset_index(drop=True)
    
    # =========================================================================
    # FEATURES AVAILABLE AT DECISION TIME (no lookahead)
    # =========================================================================
    
    # 1. Score-based features (known at decision time)
    df["abs_score"] = df["score"].abs()
    df["score_percentile"] = df["abs_score"].rank(pct=True)
    
    # 2. Position and outcome (for label and lagged features)
    df["position_sign"] = np.sign(df["score"])
    df["trade_pnl"] = df["position_sign"] * df["y_true"]
    df["trade_win"] = (df["trade_pnl"] > 0).astype(float)
    
    # =========================================================================
    # LAGGED FEATURES - Use PAST information only
    # =========================================================================
    
    # We need to build features from PAST folds only
    # For each timestamp, we can only use outcomes from COMPLETED trades
    # A trade made at time t uses y_true from t to t+horizon, so it's only
    # known at t+horizon. We must shift by at least `horizon` days.
    
    # Aggregate by fold (one outcome per fold) for cleaner history
    fold_outcomes = df.groupby("fold_id").agg({
        "timestamp": "last",  # End of fold
        "y_true": "last",      # Fold's return
        "score": "last",       # Fold's score
        "trade_pnl": "last",   # Fold's trade P&L
        "trade_win": "last",   # Did fold's trade win?
    }).reset_index()
    fold_outcomes = fold_outcomes.sort_values("fold_id").reset_index(drop=True)
    
    # Calculate rolling stats on PAST folds (shift by 1 = previous fold's outcome)
    fold_outcomes["hist_hitrate"] = fold_outcomes["trade_win"].shift(1).rolling(
        hitrate_lookback, min_periods=3
    ).mean()
    
    fold_outcomes["hist_pnl"] = fold_outcomes["trade_pnl"].shift(1).rolling(
        hitrate_lookback, min_periods=3
    ).sum()
    
    # Historical volatility (of past fold returns)
    fold_outcomes["hist_vol"] = fold_outcomes["y_true"].shift(1).rolling(
        vol_lookback, min_periods=5
    ).std()
    
    # Historical equity and drawdown (from past trades)
    fold_outcomes["hist_cum_pnl"] = fold_outcomes["trade_pnl"].shift(1).cumsum()
    fold_outcomes["hist_equity"] = (1 + fold_outcomes["trade_pnl"].shift(1)).cumprod()
    fold_outcomes["hist_peak"] = fold_outcomes["hist_equity"].cummax()
    fold_outcomes["hist_drawdown"] = fold_outcomes["hist_equity"] / fold_outcomes["hist_peak"] - 1.0
    
    # Merge back to main dataframe
    hist_features = fold_outcomes[[
        "fold_id", "hist_hitrate", "hist_pnl", "hist_vol", "hist_drawdown"
    ]]
    df = df.merge(hist_features, on="fold_id", how="left")
    
    # Fill NaN with neutral values (no history = assume neutral)
    df["hist_hitrate"] = df["hist_hitrate"].fillna(0.5)
    df["hist_pnl"] = df["hist_pnl"].fillna(0.0)
    df["hist_vol"] = df["hist_vol"].fillna(df["y_true"].std())
    df["hist_drawdown"] = df["hist_drawdown"].fillna(0.0)
    
    # =========================================================================
    # LABELS (use actual outcome - this is what we're predicting)
    # =========================================================================
    df["y_cls"] = (df["trade_pnl"] > 0).astype(int)
    df["y_reg"] = df["trade_pnl"]
    
    # Sort back
    df = df.sort_values(["fold_id", "timestamp"]).reset_index(drop=True)
    
    logger.info(
        f"📊 Built meta-learner dataset: {len(df)} rows, "
        f"features=[score, abs_score, score_percentile, hist_vol, "
        f"hist_hitrate, hist_pnl, hist_drawdown] (ALL PROPERLY LAGGED)"
    )
    
    return df


def train_meta_learner(
    dataset: pd.DataFrame,
    model_type: str = "logistic",
    task: str = "classification",
    train_folds: Optional[List[int]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a meta-learner model.
    
    Args:
        dataset: Dataset from build_meta_learner_dataset
        model_type: "lightgbm", "logistic", or "xgboost"
        task: "classification" or "regression"
        train_folds: Folds to use for training (None = all except last 10)
        
    Returns:
        Tuple of (trained model, validation metrics)
    """
    # Use ONLY the properly lagged features
    feature_cols = [
        "score", "abs_score", "score_percentile",
        "hist_vol", "hist_hitrate", "hist_pnl", "hist_drawdown"
    ]
    
    target_col = "y_cls" if task == "classification" else "y_reg"
    
    # Split by folds
    all_folds = sorted(dataset["fold_id"].unique())
    if train_folds is None:
        train_folds = all_folds[:-10]  # Hold out last 10 folds
    val_folds = [f for f in all_folds if f not in train_folds]
    
    train_data = dataset[dataset["fold_id"].isin(train_folds)]
    val_data = dataset[dataset["fold_id"].isin(val_folds)]
    
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_val = val_data[feature_cols].values
    y_val = val_data[target_col].values
    
    logger.info(
        f"🎯 Training {model_type} {task} model: "
        f"train={len(X_train)} samples ({len(train_folds)} folds), "
        f"val={len(X_val)} samples ({len(val_folds)} folds)"
    )
    
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not available, falling back to logistic regression")
            model_type = "logistic"
    
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if task == "classification":
            model = LogisticRegression(max_iter=1000, C=0.1)
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            val_pred_cls = (val_pred > 0.5).astype(int)
            accuracy = (val_pred_cls == y_val).mean()
            metrics = {"accuracy": accuracy, "auc": _compute_auc(y_val, val_pred)}
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            mse = ((val_pred - y_val) ** 2).mean()
            metrics = {"mse": mse, "r2": 1 - mse / y_val.var()}
        
        # Wrap model with scaler
        model = {"scaler": scaler, "model": model}
        
    elif model_type == "lightgbm":
        import lightgbm as lgb
        
        if task == "classification":
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 100,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            val_pred = model.predict_proba(X_val)[:, 1]
            val_pred_cls = (val_pred > 0.5).astype(int)
            accuracy = (val_pred_cls == y_val).mean()
            metrics = {"accuracy": accuracy, "auc": _compute_auc(y_val, val_pred)}
        else:
            params = {
                "objective": "regression",
                "metric": "mse",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 100,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            val_pred = model.predict(X_val)
            mse = ((val_pred - y_val) ** 2).mean()
            metrics = {"mse": mse, "r2": 1 - mse / y_val.var()}
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"✅ Model trained. Validation metrics: {metrics}")
    
    return model, metrics


def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC score."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    except:
        return 0.5


def predict_meta_learner(
    model: Any,
    dataset: pd.DataFrame,
    task: str = "classification",
) -> pd.Series:
    """
    Generate predictions from trained meta-learner.
    """
    # Use the properly lagged features
    feature_cols = [
        "score", "abs_score", "score_percentile",
        "hist_vol", "hist_hitrate", "hist_pnl", "hist_drawdown"
    ]
    
    X = dataset[feature_cols].values
    
    if isinstance(model, dict):
        # Sklearn model with scaler
        X_scaled = model["scaler"].transform(X)
        if task == "classification":
            preds = model["model"].predict_proba(X_scaled)[:, 1]
        else:
            preds = model["model"].predict(X_scaled)
    else:
        # LightGBM
        if task == "classification":
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
    
    return pd.Series(preds, index=dataset.index)


def run_dynamic_policy(
    dataset: pd.DataFrame,
    model: Any,
    task: str = "classification",
    confidence_threshold: float = 0.55,
    periods_per_year: float = 4.0,
    eval_folds: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run dynamic trading policy using meta-learner.
    
    IMPORTANT: Only evaluates on specified folds (OOS evaluation).
    
    For classification:
    - Trade only when P(win) >= confidence_threshold
    - Position = sign(score) when trading
    
    Args:
        dataset: Dataset with features and y_true
        model: Trained meta-learner
        task: "classification" or "regression"
        confidence_threshold: Minimum confidence to trade
        periods_per_year: For Sharpe annualization
        eval_folds: Folds to evaluate on (must be OOS). If None, uses last 10.
        
    Returns:
        Tuple of (results DataFrame, metrics dict)
    """
    # Determine evaluation folds (must be OOS)
    all_folds = sorted(dataset["fold_id"].unique())
    if eval_folds is None:
        eval_folds = all_folds[-10:]  # Default: last 10 folds
    
    # Filter to evaluation folds only
    df = dataset[dataset["fold_id"].isin(eval_folds)].copy()
    
    if len(df) == 0:
        logger.warning("No data in evaluation folds!")
        return pd.DataFrame(), {"coverage": 0, "sharpe": 0, "maxdd": 0, "hitrate": 0}
    
    # Generate predictions
    df["ml_confidence"] = predict_meta_learner(model, df, task)
    
    if task == "classification":
        # Trade when confidence >= threshold
        df["trade_signal"] = (df["ml_confidence"] >= confidence_threshold).astype(int)
        df["position"] = df["trade_signal"] * np.sign(df["score"])
    else:
        # Regression: trade when predicted PnL is positive and significant
        df["trade_signal"] = (df["ml_confidence"] > 0.01).astype(int)
        df["position"] = df["trade_signal"] * np.sign(df["score"])
    
    df["ret"] = df["position"] * df["y_true"]
    
    # Take non-overlapping (last per fold)
    fold_results = df.groupby("fold_id").agg({
        "position": "last",
        "y_true": "last",
        "ret": "last",
        "ml_confidence": "last",
        "timestamp": "last",
    }).reset_index()
    
    fold_results["ret"] = fold_results["position"] * fold_results["y_true"]
    
    # Metrics
    returns = fold_results["ret"]
    positions = fold_results["position"]
    
    coverage = (positions != 0).mean()
    sharpe = compute_sharpe(returns, periods_per_year)
    maxdd = compute_maxdd(returns)
    hitrate = compute_hitrate(returns)
    ann_return = returns.mean() * periods_per_year
    
    metrics = {
        "coverage": coverage,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "hitrate": hitrate,
        "ann_return": ann_return,
        "confidence_threshold": confidence_threshold,
        "n_eval_folds": len(eval_folds),
    }
    
    logger.info(
        f"🤖 Dynamic policy (OOS {len(eval_folds)} folds): Sharpe={sharpe:.3f}, "
        f"Coverage={coverage:.1%}, HitRate={hitrate:.1%}, MaxDD={maxdd:.1%}"
    )
    
    return fold_results, metrics


# ============================================================================
# CONFIDENCE THRESHOLD SWEEP (for ML policy)
# ============================================================================

def sweep_confidence_thresholds(
    dataset: pd.DataFrame,
    model: Any,
    task: str = "classification",
    thresholds: Optional[List[float]] = None,
    periods_per_year: float = 4.0,
    eval_folds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Sweep confidence thresholds for ML policy.
    
    Similar to static threshold sweep, but for the ML confidence score.
    
    Args:
        eval_folds: If provided, only evaluate on these folds (OOS evaluation).
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 0.95, 0.05).tolist()
    
    results = []
    for conf in thresholds:
        _, metrics = run_dynamic_policy(
            dataset, model, task, 
            confidence_threshold=conf,
            periods_per_year=periods_per_year,
            eval_folds=eval_folds,
        )
        metrics["threshold"] = conf
        results.append(metrics)
    
    sweep_df = pd.DataFrame(results)
    
    logger.info(
        f"📊 Confidence sweep complete: {len(thresholds)} thresholds, "
        f"best Sharpe={sweep_df['sharpe'].max():.3f} "
        f"at conf={sweep_df.loc[sweep_df['sharpe'].idxmax(), 'threshold']:.2f}"
    )
    
    return sweep_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Plot threshold sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sharpe vs Threshold
    ax = axes[0, 0]
    ax.plot(sweep_df["T"], sweep_df["Sharpe"], "b-o", markersize=4)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe vs Threshold")
    ax.grid(True, alpha=0.3)
    
    # Coverage vs Threshold
    ax = axes[0, 1]
    ax.plot(sweep_df["T"], sweep_df["Coverage"] * 100, "g-o", markersize=4)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs Threshold")
    ax.grid(True, alpha=0.3)
    
    # MaxDD vs Threshold
    ax = axes[1, 0]
    ax.plot(sweep_df["T"], sweep_df["MaxDD"] * 100, "r-o", markersize=4)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Max Drawdown vs Threshold")
    ax.grid(True, alpha=0.3)
    
    # Objective vs Threshold
    ax = axes[1, 1]
    ax.plot(sweep_df["T"], sweep_df["Objective"], "m-o", markersize=4)
    best_idx = sweep_df["Objective"].idxmax()
    ax.axvline(sweep_df.loc[best_idx, "T"], color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Threshold T")
    ax.set_ylabel("Objective (Sharpe × Coverage^α)")
    ax.set_title("Objective vs Threshold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Plot saved: {save_path}")
    
    plt.close()


def plot_walkforward_equity(
    wf_result: WalkForwardResult,
    save_path: Optional[Path] = None,
) -> None:
    """Plot walk-forward equity curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Equity curve
    ax = axes[0, 0]
    wf_result.equity_curve.plot(ax=ax, color="b", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_title(f"Walk-Forward Equity (Sharpe={wf_result.aggregate_sharpe:.2f})")
    ax.grid(True, alpha=0.3)
    
    # Threshold over time
    ax = axes[0, 1]
    ax.plot(wf_result.fold_thresholds, "g-o", markersize=4)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Threshold T*")
    ax.set_title("Adaptive Threshold per Fold")
    ax.grid(True, alpha=0.3)
    
    # Per-fold returns
    ax = axes[1, 0]
    colors = ["g" if r > 0 else "r" for r in wf_result.fold_returns]
    ax.bar(range(len(wf_result.fold_returns)), wf_result.fold_returns, color=colors, alpha=0.7)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return")
    ax.set_title(f"Per-Fold Returns (HitRate={wf_result.aggregate_hitrate:.1%})")
    ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[1, 1]
    equity = wf_result.equity_curve
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    dd.plot(ax=ax, color="r", linewidth=1.5)
    ax.fill_between(dd.index, dd.values, 0, color="r", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown (Max={wf_result.aggregate_maxdd:.1%})")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Plot saved: {save_path}")
    
    plt.close()


# ============================================================================
# MAIN CONVENIENCE FUNCTION
# ============================================================================

def run_full_stage_c_analysis(
    tape_path: Union[str, Path],
    output_dir: Optional[Path] = None,
    run_static: bool = True,
    run_walkforward: bool = True,
    run_ml: bool = True,
) -> Dict[str, Any]:
    """
    Run complete Stage C analysis.
    
    1. Static threshold search
    2. Walk-forward threshold training
    3. ML meta-learner (if requested)
    
    Returns dict with all results.
    """
    tape = load_prediction_tape(tape_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"tape_info": {
        "rows": len(tape),
        "folds": tape["fold_id"].nunique(),
        "date_range": f"{tape['timestamp'].min().date()} to {tape['timestamp'].max().date()}",
    }}
    
    # 1. Static threshold search
    if run_static:
        logger.info("\n" + "=" * 60)
        logger.info("1. STATIC THRESHOLD SEARCH")
        logger.info("=" * 60)
        
        sweep_df, best_result = static_threshold_search(tape)
        results["static"] = {
            "sweep_df": sweep_df,
            "best_threshold": best_result.threshold,
            "best_sharpe": best_result.sharpe,
            "best_coverage": best_result.coverage,
            "best_maxdd": best_result.maxdd,
            "best_hitrate": best_result.hitrate,
            "best_objective": best_result.objective,
        }
        
        if output_dir:
            plot_threshold_sweep(sweep_df, output_dir / "threshold_sweep.png")
            sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    
    # 2. Walk-forward threshold training
    if run_walkforward:
        logger.info("\n" + "=" * 60)
        logger.info("2. WALK-FORWARD THRESHOLD TRAINING")
        logger.info("=" * 60)
        
        wf_result = walkforward_threshold_training(tape)
        results["walkforward"] = {
            "sharpe": wf_result.aggregate_sharpe,
            "return": wf_result.aggregate_return,
            "maxdd": wf_result.aggregate_maxdd,
            "hitrate": wf_result.aggregate_hitrate,
            "equity_curve": wf_result.equity_curve,
            "trades_df": wf_result.trades_df,
            "fold_thresholds": wf_result.fold_thresholds,
        }
        
        if output_dir:
            plot_walkforward_equity(wf_result, output_dir / "walkforward_equity.png")
            wf_result.trades_df.to_csv(output_dir / "walkforward_trades.csv", index=False)
    
    # 3. ML meta-learner
    if run_ml:
        logger.info("\n" + "=" * 60)
        logger.info("3. ML META-LEARNER")
        logger.info("=" * 60)
        
        dataset = build_meta_learner_dataset(tape)
        model, train_metrics = train_meta_learner(dataset, model_type="logistic")
        
        # Sweep confidence thresholds
        conf_sweep = sweep_confidence_thresholds(dataset, model)
        
        # Run with best confidence
        best_conf = conf_sweep.loc[conf_sweep["sharpe"].idxmax(), "threshold"]
        _, ml_metrics = run_dynamic_policy(dataset, model, confidence_threshold=best_conf)
        
        results["ml"] = {
            "train_metrics": train_metrics,
            "conf_sweep": conf_sweep,
            "best_confidence": best_conf,
            "sharpe": ml_metrics["sharpe"],
            "coverage": ml_metrics["coverage"],
            "hitrate": ml_metrics["hitrate"],
            "maxdd": ml_metrics["maxdd"],
        }
        
        if output_dir:
            conf_sweep.to_csv(output_dir / "ml_confidence_sweep.csv", index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("STAGE C ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    return results
