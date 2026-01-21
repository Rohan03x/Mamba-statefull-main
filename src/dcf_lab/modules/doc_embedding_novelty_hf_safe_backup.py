"""Document novelty module using count-based heuristics (torch-free fallback)."""
from __future__ import annotations

import logging
import math
from datetime import time as dt_time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..signal_bus import ModuleSignal
from ..utils import to_market_session

LOGGER = logging.getLogger(__name__)

_DATA_CACHE = Path(__file__).resolve().parents[3] / "data_cache"


def _merge_text(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if not title and not body:
        return ""
    if body and not body.endswith("."):
        body = body + "."
    if title:
        return f"{title}. {body}".strip()
    return body


def _confidence_from_counts(count: int, *, scale: float = 60.0) -> float:
    if count <= 0:
        return 0.0
    return float(min(1.0, math.log1p(count) / math.log1p(scale)))


def load_news_for_symbol(symbol: str, max_items: int = 320) -> pd.DataFrame:
    """Load news from cache (simplified version)."""
    cache_path = _DATA_CACHE / "news_cache" / f"news_{symbol.upper()}.parquet"
    if not cache_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return df
        
        # Ensure timestamp column
        if "timestamp" not in df.columns and df.index.name == "timestamp":
            df = df.reset_index()
        
        df = df.tail(max_items).copy()
        return df
    except Exception as exc:
        LOGGER.debug("Failed to load news cache: %s", exc)
        return pd.DataFrame()


class DocumentEmbeddingNoveltyHF:
    """Estimate narrative novelty using count-based heuristics (torch-free)."""

    NAME = "doc_embedding_novelty_hf"
    metadata = {
        "group": "hf",
        "group_cap": 0.5,
        "group_penalty": 1.0,
        "w_min": 0.0,
        "w_max": 0.4,
    }

    def __init__(
        self,
        *,
        lookback_days: int = 60,
        max_items: int = 320,
        z_window: int = 20,
        cutoff: str = "16:00:00",
        tz: str = "America/New_York",
        polarity: float = -1.0,
    ) -> None:
        self.lookback_days = int(max(5, lookback_days))
        self.max_items = int(max(25, max_items))
        self.z_window = int(max(5, z_window))
        self.cutoff_time = self._parse_cutoff(cutoff)
        self.tz = tz
        self.polarity = float(np.clip(polarity, -1.0, 1.0)) or -1.0

    @staticmethod
    def _parse_cutoff(value: str) -> dt_time:
        if not value:
            return dt_time(16, 0, 0)
        parts = [int(part) for part in value.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return dt_time(parts[0], parts[1], parts[2])

    def _load_documents(self, symbol: str) -> pd.DataFrame:
        df = load_news_for_symbol(symbol, max_items=self.max_items)
        if df.empty:
            return df
        
        df = df.tail(self.max_items).copy()
        
        # Merge title and body
        if "title" in df.columns and "body" in df.columns:
            df["text"] = [
                _merge_text(str(title), str(body))
                for title, body in zip(df["title"], df["body"])
            ]
            df = df[df["text"].str.len() > 0]
        
        if df.empty:
            return df
        
        # Convert to market sessions
        if "timestamp" in df.columns:
            df["session_date"] = to_market_session(
                pd.to_datetime(df["timestamp"], utc=True, errors="coerce"),
                cutoff=self.cutoff_time,
                tz=self.tz
            )
        else:
            df["session_date"] = pd.to_datetime(df.index).normalize()
        
        df = df.dropna(subset=["session_date"])
        df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
        df = df.sort_values("session_date")
        return df.reset_index(drop=True)

    def emit_signal(self, *, symbol: str, horizon: int) -> ModuleSignal:
        """Generate novelty signal from news article counts (torch-free fallback)."""
        docs = self._load_documents(symbol)
        if docs.empty:
            return ModuleSignal(
                name=self.NAME,
                horizon=int(horizon),
                df=pd.DataFrame(columns=["score", "conf"]),
                symbol=symbol
            )

        # Group by session date and count articles
        grouped = docs.groupby("session_date").size()
        if len(grouped) == 0:
            return ModuleSignal(
                name=self.NAME,
                horizon=int(horizon),
                df=pd.DataFrame(columns=["score", "conf"]),
                symbol=symbol
            )

        # Calculate novelty as deviation from baseline
        series = grouped.sort_index()
        baseline = series.shift(1).rolling(self.z_window, min_periods=1).mean()
        delta = (series - baseline).fillna(0.0)
        
        # Normalize
        denom = float(delta.abs().max() or 1.0)
        scaled = delta / denom
        
        # Build signal
        rows: List[Tuple[pd.Timestamp, float, float]] = []
        for when, value, count in zip(series.index, scaled.values, series.values):
            score = float(np.clip(value * self.polarity, -1.0, 1.0))
            conf = _confidence_from_counts(int(count), scale=40.0)
            rows.append((pd.Timestamp(when), score, conf))
        
        df = pd.DataFrame(rows, columns=["date", "score", "conf"]).set_index("date")
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=df, symbol=symbol)


__all__ = ["DocumentEmbeddingNoveltyHF"]
