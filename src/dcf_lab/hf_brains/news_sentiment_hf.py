"""Daily news sentiment signal powered by Hugging Face models."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..news import NewsProvider, score_sentiment
from ..signal_bus import ModuleSignal
from .base import (
    batched_text_classification,
    build_signal,
    coerce_date,
    confidence_from_count,
    get_pipeline,
    label_score,
)


class NewsSentimentHF:
    """Generate a ModuleSignal from recent news headlines using FinBERT-style HF models."""

    NAME = "news_sentiment_hf"
    DEFAULT_MODEL = "yiyanghkust/finbert-tone"
    DEFAULT_REVISION = "main"

    def __init__(
        self,
        *,
        lookback_days: int = 21,
        max_items: int = 120,
        batch_size: int = 16,
        model_name: Optional[str] = None,
        model_revision: Optional[str] = None,
    ) -> None:
        self.lookback_days = int(max(1, lookback_days))
        self.max_items = int(max(10, max_items))
        self.batch_size = int(max(4, batch_size))
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model_revision = model_revision or self.DEFAULT_REVISION
        self._pipeline = None
        self._provider = NewsProvider()
        if "finbert" in self.model_name.lower():
            self.batch_size = int(max(8, min(self.batch_size, 16)))

    def _resolve_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        self._pipeline = get_pipeline(
            "text-classification",
            self.model_name,
            tokenizer=self.model_name,
            pipeline_kwargs={"return_all_scores": False},
            revision=self.model_revision,
        )
        return self._pipeline

    def _retrieve_news(self, symbol: str) -> Sequence[Dict]:
        try:
            return self._provider.get_recent_news(
                symbol,
                days=self.lookback_days,
                limit=self.max_items,
            )
        except Exception:
            return []

    def _score_with_pipeline(self, texts: Sequence[str]) -> Sequence[float]:
        classifier = self._resolve_pipeline()
        if classifier is None:
            # Fallback to classic sentiment scoring
            scores: List[float] = []
            for text in texts:
                try:
                    scores.append(float(score_sentiment(text)))
                except Exception:
                    scores.append(0.0)
            return scores
        raw_results = batched_text_classification(
            classifier,
            texts,
            batch_size=self.batch_size,
            max_length=256,
        )
        if not raw_results:
            return [0.0 for _ in texts]
        return [float(np.clip(label_score(result), -1.0, 1.0)) for result in raw_results]

    def emit_signal(self, *, symbol: str, horizon: int) -> Optional[ModuleSignal]:
        items = self._retrieve_news(symbol)
        if not items:
            return None

        texts: List[str] = []
        meta: List[pd.Timestamp] = []
        for item in items:
            headline = (
                item.get("headline")
                or item.get("title")
                or item.get("summary")
                or item.get("description")
                or ""
            ).strip()
            if not headline:
                continue
            published = coerce_date(
                item.get("date")
                or item.get("published")
                or item.get("timestamp")
                or item.get("datetime")
            )
            if published is None:
                continue
            texts.append(headline)
            meta.append(published)

        if not texts:
            return None

        scores = self._score_with_pipeline(texts)
        buckets: Dict[pd.Timestamp, List[float]] = defaultdict(list)
        for when, score in zip(meta, scores):
            buckets[when].append(float(np.clip(score, -1.0, 1.0)))

        rows: List[Tuple[pd.Timestamp, float, float]] = []
        for when, values in sorted(buckets.items()):
            if not values:
                continue
            mean_score = float(np.mean(values))
            confidence = confidence_from_count(len(values)) * float(np.mean(np.abs(values)))
            confidence = float(np.clip(confidence, 0.0, 1.0))
            rows.append((when, float(np.clip(mean_score, -1.0, 1.0)), confidence))

        return build_signal(self.NAME, horizon, rows)


__all__ = ["NewsSentimentHF"]
