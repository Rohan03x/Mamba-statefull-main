import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timezone


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    MOCK DATA DISABLED - Real narrative novelty requires sentence-transformers embeddings.
    Returns empty DataFrame to prevent synthetic data contamination.
    """
    # Return empty DataFrame - no mock data allowed
    return pd.DataFrame()
