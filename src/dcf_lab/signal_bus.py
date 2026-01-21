"""Signal bus utilities for normalizing module outputs.

Provides a lightweight dataclass for transporting directional signals and a
standardization helper to enforce consistent scaling across modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pandas import DataFrame


@dataclass
class ModuleSignal:
    """Container for a module's directional signal output.

    Attributes:
        name: Identifier for the producing module.
        horizon: Forecast horizon (in trading days) for the signal.
        df: Pandas DataFrame indexed by date with at least a ``score`` column
            in ``[-1, 1]`` and an optional ``conf`` column in ``[0, 1]``.
    """

    name: str
    horizon: int
    df: "DataFrame"
    symbol: Optional[str] = None


def standardize_module(df: pd.DataFrame, rolling: int = 252) -> pd.DataFrame:
    """Normalize a module signal to the canonical score range.

    The procedure applies a causal z-score using a trailing ``rolling`` window
    (default ~1 trading year), clips extreme values, and squashes the result to
    ``[-1, 1]``. Confidence is ensured to exist and bounded within ``[0, 1]``.

    Args:
        df: Signal dataframe with a ``score`` column and optional ``conf`` column.
        rolling: Window size for the rolling statistics used in the z-score.

    Returns:
        A two-column dataframe (``score``, ``conf``) aligned with the input
        index, suitable for downstream aggregation.
    """

    if "score" not in df.columns:
        raise KeyError("signal dataframe must include a 'score' column")

    standardized = df.copy()
    scores = standardized["score"].astype(float)

    # Use causal statistics (only look backwards).
    rolling_mean = scores.shift(1).rolling(rolling, min_periods=50).mean()
    rolling_std = scores.shift(1).rolling(rolling, min_periods=50).std()

    # Protect against division by zero or NaNs from insufficient history.
    rolling_std = rolling_std.replace(0.0, 1.0).fillna(1.0)
    rolling_mean = rolling_mean.fillna(0.0)

    z_scores = (scores - rolling_mean) / rolling_std
    standardized_scores = (z_scores.clip(-3.0, 3.0) / 3.0).clip(-1.0, 1.0)
    standardized["score"] = standardized_scores

    if "conf" not in standardized.columns:
        standardized["conf"] = 1.0
    else:
        standardized["conf"] = (
            standardized["conf"].astype(float).fillna(1.0).clip(0.0, 1.0)
        )

    return standardized[["score", "conf"]]
