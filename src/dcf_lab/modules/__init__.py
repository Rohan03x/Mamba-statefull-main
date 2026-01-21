"""Hugging Face-enabled signal modules.

Keep this package import lightweight.

Several optional modules pull in heavy third-party dependencies (and can emit
stdout at import time). Importing them eagerly would make unrelated tools (like
Stage A selector) noisy and slow.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


__all__ = [
	"NewsSentimentHF",
	"EarningsTranscriptHF",
	"DocumentEmbeddingNoveltyHF",
	"MacroTSTHF",
]

_EXPORTS = {
	"NewsSentimentHF": ".news_sentiment_hf",
	"EarningsTranscriptHF": ".earnings_transcript_hf",
	"DocumentEmbeddingNoveltyHF": ".doc_embedding_novelty_hf",
	"MacroTSTHF": ".macro_tst_hf",
}


def __getattr__(name: str):  # noqa: ANN001
	module_path = _EXPORTS.get(name)
	if module_path is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module = importlib.import_module(module_path, __name__)
	return getattr(module, name)


if TYPE_CHECKING:  # pragma: no cover
	from .earnings_transcript_hf import EarningsTranscriptHF
	from .doc_embedding_novelty_hf import DocumentEmbeddingNoveltyHF
	from .macro_tst_hf import MacroTSTHF
	from .news_sentiment_hf import NewsSentimentHF
