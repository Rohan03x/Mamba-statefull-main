"""
Stage B Export Module
=====================

Handles building the "Prediction Tape" from LSTM walk-forward folds.

The prediction tape is a unified dataframe containing all predictions
across all folds, which can then be used by Stage C for threshold
sweeps and policy optimization without re-running the LSTM.

Structure (per row):
- timestamp: date of prediction
- fold_id: which walk-forward fold
- y_true: realised future return over horizon
- score: raw LSTM output (signed)
- position_raw: sign of prediction before thresholds
- regime (optional): market regime if available
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default output directory for prediction tapes
DEFAULT_TAPE_DIR = Path("artifacts/prediction_tapes")


def collect_fold_predictions(
    fold_id: int,
    timestamps: Union[pd.DatetimeIndex, np.ndarray, List],
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    regime: Optional[Union[pd.Series, np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Collect predictions from a single walk-forward fold.
    
    Args:
        fold_id: Identifier for the fold (0-indexed)
        timestamps: Timestamps for each prediction
        y_true: Realised future returns (ground truth)
        y_pred: Raw LSTM predictions/scores (signed)
        regime: Optional regime labels for each timestamp
        metadata: Optional additional metadata dict
        
    Returns:
        DataFrame with columns: timestamp, fold_id, y_true, score, position_raw, [regime]
    """
    # Convert to numpy arrays for consistent handling
    timestamps = pd.to_datetime(timestamps)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Validate lengths match
    n = len(timestamps)
    if len(y_true) != n or len(y_pred) != n:
        raise ValueError(
            f"Length mismatch: timestamps={n}, y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    # Build base dataframe
    df = pd.DataFrame({
        "timestamp": timestamps,
        "fold_id": fold_id,
        "y_true": y_true,
        "score": y_pred,
    })
    
    # Raw position: sign of prediction before any thresholds
    df["position_raw"] = np.sign(df["score"]).astype(int)
    
    # Add regime if provided
    if regime is not None:
        regime = np.asarray(regime).flatten()
        if len(regime) == n:
            df["regime"] = regime
        else:
            logger.warning(
                f"Regime length mismatch ({len(regime)} vs {n}), skipping"
            )
    
    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (int, float, str)):
                df[f"meta_{key}"] = value
    
    logger.debug(
        f"Collected fold {fold_id}: {len(df)} predictions, "
        f"y_true range=[{y_true.min():.4f}, {y_true.max():.4f}], "
        f"score range=[{y_pred.min():.4f}, {y_pred.max():.4f}]"
    )
    
    return df


def save_prediction_tape(
    fold_dfs: List[pd.DataFrame],
    symbol: str,
    horizon: int,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Concatenate fold predictions and save as a unified prediction tape.
    
    Args:
        fold_dfs: List of DataFrames from collect_fold_predictions
        symbol: Stock symbol (e.g., "AAPL")
        horizon: Forecast horizon in days
        out_dir: Output directory (default: artifacts/prediction_tapes)
        filename: Custom filename (default: {symbol}_h{horizon}_predictions.parquet)
        
    Returns:
        Path to saved prediction tape
    """
    if not fold_dfs:
        raise ValueError("No fold predictions to save")
    
    # Concatenate all folds
    tape = pd.concat(fold_dfs, ignore_index=True)
    
    # Sort by timestamp, then fold_id
    tape.sort_values(["timestamp", "fold_id"], inplace=True)
    tape.reset_index(drop=True, inplace=True)
    
    # Sanity checks
    _validate_prediction_tape(tape)
    
    # Determine output path
    out_dir = Path(out_dir) if out_dir else DEFAULT_TAPE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"{symbol.upper()}_h{horizon}_predictions.parquet"
    
    out_path = out_dir / filename
    
    # Save to parquet
    tape.to_parquet(out_path, index=False)
    
    logger.info(
        f"💾 Saved prediction tape: {out_path} "
        f"({len(tape)} rows, {tape['fold_id'].nunique()} folds, "
        f"{tape['timestamp'].min().date()} to {tape['timestamp'].max().date()})"
    )
    
    return out_path


def _validate_prediction_tape(tape: pd.DataFrame) -> None:
    """
    Run sanity checks on prediction tape.
    
    Checks:
    - No overlapping timestamps within the same fold
    - No NaN in y_pred / y_true
    - Basic data integrity
    """
    # Check required columns
    required_cols = ["timestamp", "fold_id", "y_true", "score", "position_raw"]
    missing = [c for c in required_cols if c not in tape.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for NaN in critical columns
    for col in ["y_true", "score"]:
        nan_count = tape[col].isna().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in '{col}' column")
    
    # Check for duplicate timestamps within folds
    duplicates = tape.groupby("fold_id")["timestamp"].apply(
        lambda x: x.duplicated().sum()
    )
    total_dups = duplicates.sum()
    if total_dups > 0:
        logger.warning(
            f"⚠️ Found {total_dups} duplicate timestamps within folds. "
            f"Folds with duplicates: {duplicates[duplicates > 0].to_dict()}"
        )
    
    # Log statistics
    logger.info(
        f"✅ Prediction tape validation passed: "
        f"{len(tape)} rows, {tape['fold_id'].nunique()} folds, "
        f"score μ={tape['score'].mean():.4f} σ={tape['score'].std():.4f}, "
        f"y_true μ={tape['y_true'].mean():.4f} σ={tape['y_true'].std():.4f}"
    )


def load_prediction_tape(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a saved prediction tape.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame with prediction tape
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction tape not found: {path}")
    
    tape = pd.read_parquet(path)
    tape["timestamp"] = pd.to_datetime(tape["timestamp"])
    
    logger.info(
        f"📂 Loaded prediction tape: {path} "
        f"({len(tape)} rows, {tape['fold_id'].nunique()} folds)"
    )
    
    return tape


def get_tape_summary(tape: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a prediction tape.
    
    Args:
        tape: Prediction tape DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        "n_rows": len(tape),
        "n_folds": tape["fold_id"].nunique(),
        "date_start": tape["timestamp"].min().isoformat(),
        "date_end": tape["timestamp"].max().isoformat(),
        "score_mean": float(tape["score"].mean()),
        "score_std": float(tape["score"].std()),
        "score_min": float(tape["score"].min()),
        "score_max": float(tape["score"].max()),
        "y_true_mean": float(tape["y_true"].mean()),
        "y_true_std": float(tape["y_true"].std()),
        "position_raw_dist": tape["position_raw"].value_counts().to_dict(),
        "has_regime": "regime" in tape.columns,
    }
