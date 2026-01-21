"""Hugging Face powered signal modules ("brains").

This package exposes a lightweight manifest describing each available HF-based
signal emitter and convenience helpers to load or register them with the
universal aggregator.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HFBrainManifestEntry:
    name: str
    module_path: str
    object_name: str
    description: str
    tags: Sequence[str]
    default_horizons: Sequence[int]
    refresh: str
    brain: Optional[Mapping[str, object]] = None
    data: Optional[Mapping[str, object]] = None
    license: Optional[str] = None

    def load(self) -> object:
        module = import_module(f".{self.module_path}", package=__name__)
        brain_cls = getattr(module, self.object_name)
        return brain_cls()


HF_BRAIN_MANIFEST: Dict[str, HFBrainManifestEntry] = {
    "news_sentiment_hf": HFBrainManifestEntry(
        name="news_sentiment_hf",
        module_path="news_sentiment_hf",
        object_name="NewsSentimentHF",
        description="Daily news sentiment scored with FinBERT via Hugging Face.",
        tags=("sentiment", "news", "huggingface"),
        default_horizons=(1, 5, 30),
        refresh="daily",
        brain={
            "model_id": "ProsusAI/finbert",
            "revision": "main",
            "max_length": 256,
            "batch_size": 16,
            "dtype": "fp16",
            "device": "auto",
            "repo_url": "https://huggingface.co/ProsusAI/finbert",
        },
        data={
            "source": "artifacts/news/{symbol}.parquet",
            "cache_glob": "data_cache/company-news_news_{symbol}_*.json",
            "cutoff": "16:00:00",
            "timezone": "America/New_York",
        },
        license="apache-2.0",
    ),
    "earnings_transcript_hf": HFBrainManifestEntry(
        name="earnings_transcript_hf",
        module_path="earnings_transcript_hf",
        object_name="EarningsTranscriptHF",
        description="Quarterly earnings call tone using transformer sentiment stacks.",
        tags=("sentiment", "earnings", "huggingface"),
        default_horizons=(5, 30, 63),
        refresh="quarterly",
        brain={
            "model_id": "ProsusAI/finbert",
            "revision": "main",
            "max_length": 1024,
            "stride": 128,
            "batch_size": 8,
            "dtype": "fp16",
            "device": "auto",
            "repo_url": "https://huggingface.co/ProsusAI/finbert",
        },
        data={
            "source": "data_cache/earnings_transcripts/{symbol}",
            "fallback": "analyzer.get_earnings_transcript",
            "cutoff": "21:00:00",
            "timezone": "America/New_York",
        },
        license="apache-2.0",
    ),
    "doc_embedding_novelty_hf": HFBrainManifestEntry(
        name="doc_embedding_novelty_hf",
        module_path="doc_embedding_novelty_hf",
        object_name="DocumentEmbeddingNoveltyHF",
        description="Detects narrative novelty from document embeddings.",
        tags=("novelty", "embeddings", "news", "huggingface"),
        default_horizons=(1, 5, 30),
        refresh="daily",
        brain={
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "main",
            "normalize_embeddings": True,
            "batch_size": 32,
            "device": "cpu",
            "repo_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        },
        data={
            "source": "artifacts/news/{symbol}.parquet",
            "cache_glob": "data_cache/company-news_news_{symbol}_*.json",
            "cutoff": "16:00:00",
            "timezone": "America/New_York",
        },
        license="apache-2.0",
    ),
    "macro_tst_hf": HFBrainManifestEntry(
        name="macro_tst_hf",
        module_path="macro_tst_hf",
        object_name="MacroTSTHF",
        description="Macro panel directional probability via TimeSeriesTransformer.",
        tags=("macro", "timeseries", "huggingface"),
        default_horizons=(5, 21, 63),
        refresh="weekly",
        brain={
            "model_id": "",
            "context_length": 252,
            "prediction_length": 1,
            "encoder_layers": 2,
            "dropout": 0.1,
            "device": "auto",
            "repo_url": None,
        },
        data={
            "source": "data_cache/macro_panel/macro_features_{symbol}.parquet",
            "timezone": "America/New_York",
            "frequency": "daily",
        },
        license="unknown",
    ),
}


def iter_brains() -> Iterable[HFBrainManifestEntry]:
    return HF_BRAIN_MANIFEST.values()


def manifest_summary() -> Mapping[str, Mapping[str, object]]:
    return {
        name: {
            "description": entry.description,
            "tags": list(entry.tags),
            "default_horizons": list(entry.default_horizons),
            "refresh": entry.refresh,
            "brain": dict(entry.brain) if entry.brain else None,
            "data": dict(entry.data) if entry.data else None,
        }
        for name, entry in HF_BRAIN_MANIFEST.items()
    }


def load_brain(name: str) -> object:
    entry = HF_BRAIN_MANIFEST[name]
    return entry.load()


def register_with_aggregator(names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    from ..universal_aggregator import register_module

    selected = names or tuple(HF_BRAIN_MANIFEST.keys())
    registered: list[str] = []
    for name in selected:
        entry = HF_BRAIN_MANIFEST.get(name)
        if entry is None:
            continue
        module = entry.load()
        register_module(name, module)
        registered.append(name)
    return tuple(registered)


__all__ = [
    "HFBrainManifestEntry",
    "HF_BRAIN_MANIFEST",
    "iter_brains",
    "manifest_summary",
    "load_brain",
    "register_with_aggregator",
]
