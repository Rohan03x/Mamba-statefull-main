"""Daily news sentiment module powered by Hugging Face FinBERT."""
from __future__ import annotations

import json
import logging
from datetime import time as dt_time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..signal_bus import ModuleSignal
from ..utils import to_market_session
from ..hf_brains.base import HFBrain, HFSpec
from ..news import NewsProvider

LOGGER = logging.getLogger(__name__)

_DATA_CACHE = Path(__file__).resolve().parents[3] / "data_cache"


def _coerce_timestamp(value: object) -> Optional[pd.Timestamp]:
    # Handle numeric timestamps (Unix epoch in seconds or milliseconds)
    if isinstance(value, (int, float)):
        # If value is very small (< 2e9), it's likely milliseconds stored as seconds
        # Unix epoch for 2033 is ~2e9, so anything smaller is likely in wrong unit
        if value < 2000000000:
            # This is likely milliseconds stored as fractional seconds (1743750000 ms = 1.74375 s)
            # Convert to milliseconds
            value = value * 1000
        
        # If value is larger than year 3000 (32503680000 seconds), it's milliseconds
        if value > 32503680000:
            value = value / 1000
    
    ts = pd.to_datetime(value, utc=True, errors="coerce", unit='s' if isinstance(value, (int, float)) else None)
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _iter_cache_paths(symbol: str) -> Iterable[Path]:
    pattern = f"company-news_news_{symbol.upper()}_*.json"
    if not _DATA_CACHE.exists():
        return []
    return sorted(_DATA_CACHE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_cached_news(symbol: str, *, max_items: int = 750) -> List[dict]:
    records: List[dict] = []
    for path in _iter_cache_paths(symbol):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            LOGGER.debug("Failed to read cached news from %s: %s", path, exc)
            continue
        for item in payload:
            ts = _coerce_timestamp(
                item.get("timestamp")
                or item.get("date")
                or item.get("datetime")
                or item.get("providerPublishTime")
                or item.get("publishedAt")
                or item.get("time")
            )
            if ts is None:
                continue
            title = (item.get("title") or item.get("headline") or "").strip()
            body = (
                item.get("body")
                or item.get("summary")
                or item.get("description")
                or item.get("content")
                or ""
            )
            if not title and not body:
                continue
            records.append({"timestamp": ts, "title": title, "body": body})
            if len(records) >= max_items:
                return records
    return records


def _fetch_live_news(symbol: str, *, days: int = 14, limit: int = 250) -> List[dict]:
    provider = NewsProvider()
    try:
        items = provider.get_recent_news(symbol, days=days, limit=limit)
    except Exception as exc:
        LOGGER.debug("News provider lookup failed for %s: %s", symbol, exc)
        return []
    records: List[dict] = []
    for item in items:
        ts = _coerce_timestamp(item.get("date") or item.get("timestamp") or item.get("published"))
        if ts is None:
            continue
        title = (item.get("headline") or item.get("title") or "").strip()
        body = (
            item.get("body")
            or item.get("summary")
            or item.get("content")
            or ""
        )
        if not title and not body:
            continue
        records.append({"timestamp": ts, "title": title, "body": body})
    return records


def load_news_for_symbol(symbol: str, *, max_items: int = 750) -> pd.DataFrame:
    """Load raw news records for ``symbol`` with timestamp/title/body columns."""

    records = _load_cached_news(symbol, max_items=max_items)
    if len(records) < max_items // 4:
        # Fall back to live provider when cache is sparse
        live_records = _fetch_live_news(symbol, limit=max_items - len(records))
        records.extend(live_records)

    if not records:
        return pd.DataFrame(columns=["timestamp", "title", "body"])

    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    return df.reset_index(drop=True)


class NewsSentimentHF:
    NAME = "news_sentiment_hf"
    metadata = {
        "group": "hf",
        "group_cap": 0.5,
        "group_penalty": 1.0,
        "w_min": 0.0,
        "w_max": 0.4,
    }

    def __init__(
        self,
        model_id: str = "ProsusAI/finbert",
        revision: Optional[str] = None,
        *,
        max_length: int = 256,
        batch_size: int = 16,
        cutoff: str = "16:00:00",
        tz: str = "America/New_York",
    ) -> None:
        self.brain = HFBrain(
            HFSpec(
                model_id=model_id,
                revision=revision,
                max_length=max_length,
                batch_size=batch_size,
            )
        )
        hours, minutes, seconds = [int(part) for part in cutoff.split(":")]
        self.cutoff = dt_time(hours, minutes, seconds)
        self.tz = tz

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["score", "conf"])

        frame = df.copy()
        frame["text"] = (frame["title"].fillna("") + ". " + frame["body"].fillna(""))
        frame["session_date"] = to_market_session(frame["timestamp"], cutoff=self.cutoff, tz=self.tz)
        probabilities = self.brain.infer_probs(frame["text"].tolist())
        if len(probabilities) != len(frame):
            LOGGER.warning(
                "HFBrain returned %s probabilities for %s rows; truncating to min length",
                len(probabilities),
                len(frame),
            )
        frame = frame.iloc[: len(probabilities)].copy()
        frame["p"] = probabilities[: len(frame)]

        grouped = frame.groupby("session_date")["p"].agg(["mean", "count"]).rename(
            columns={"mean": "p", "count": "n"}
        )
        if grouped.empty:
            return pd.DataFrame(columns=["score", "conf"])

        score = (2.0 * grouped["p"] - 1.0).clip(-1.0, 1.0)
        intensity = np.clip(np.abs(grouped["p"] - 0.5) * 2.0, 0.0, 1.0)
        volume = np.clip(np.log1p(grouped["n"]) / np.log(10.0), 0.2, 1.0)
        conf = (intensity * volume).clip(0.0, 1.0)

        daily = pd.DataFrame({"score": score, "conf": conf})
        return daily.sort_index()

    def emit_signal(self, symbol: str, horizon: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> ModuleSignal:
        news = load_news_for_symbol(symbol)
        daily = self._aggregate_daily(news)
        
        # Filter daily aggregated data to date range if provided
        if start_date and end_date and not daily.empty:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            daily = daily[(daily.index >= start_ts) & (daily.index <= end_ts)]
        
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=daily, symbol=symbol)


__all__ = ["NewsSentimentHF", "load_news_for_symbol"]
