from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from typing import Dict, List

# Reduce transformers logging noise if available
try:  # pragma: no cover
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*Xet Storage is enabled.*")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover - optional dep
    SentimentIntensityAnalyzer = None  # type: ignore

# Optional FinBERT sentiment (if transformers installed)
try:  # pragma: no cover
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _FINBERT_AVAIL = True
except Exception:
    _FINBERT_AVAIL = False

# Cache FinBERT (heavy) so we don't reload per call
_FINBERT_CACHE: dict | None = None


def _load_finbert():  # pragma: no cover
    global _FINBERT_CACHE
    if _FINBERT_CACHE is not None:
        return _FINBERT_CACHE
    if not _FINBERT_AVAIL or os.environ.get("FINBERT_DISABLE"):
        return None
    
    # Updated FinBERT loader with PyTorch 2.6+ compatibility
    # Using safetensors format for secure model loading
    model_name = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")
    fallback_name = os.environ.get("FINBERT_MODEL_FALLBACK", "ProsusAI/finbert")

    # Default to CPU unless FINBERT_USE_GPU=1 is explicitly set
    use_gpu = os.environ.get("FINBERT_USE_GPU", "0") in ("1", "true", "True")
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    def _try_load(name: str):
        print(f"🔄 Loading FinBERT model: {name} with safetensors")
        
        try:
            # Modern transformers approach with safetensors
            tok = AutoTokenizer.from_pretrained(
                name, 
                use_fast=True,
                trust_remote_code=False,  # Security: disable remote code
                local_files_only=False,
                use_auth_token=None
            )
            
            # Load model with safetensors format for PyTorch 2.6+ compatibility
            mdl = AutoModelForSequenceClassification.from_pretrained(
                name,
                use_safetensors=True,  # Force safetensors for security
                trust_remote_code=False,  # Security: disable remote code
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False,
                device_map=None,  # Manual device assignment
                offload_folder=None,
                revision="main"
            )
            print(f"✅ Successfully loaded {name} with safetensors (PyTorch 2.6+ compatible)")
                
            mdl.to(device)
            mdl.eval()
            
            # Test the model with a sample input to ensure it works
            test_input = "The market is performing well today."
            with torch.no_grad():
                inputs = tok(
                    test_input,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = mdl(**inputs)
            
            print(f"🎯 FinBERT model verified and ready on {device}")
            return {"tok": tok, "mdl": mdl, "device": device}
            
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            print("💡 Try: pip install --upgrade transformers safetensors torch")
            return None

    try:
        # Try primary model first
        print("🚀 Initializing FinBERT sentiment analysis...")
        _FINBERT_CACHE = _try_load(model_name)
        if _FINBERT_CACHE:
            return _FINBERT_CACHE
    except Exception as e:
        print(f"⚠️ Primary FinBERT model failed: {e}")
        
    # Try fallback model if primary fails
    try:
        print("🔄 Trying fallback FinBERT model...")
        _FINBERT_CACHE = _try_load(fallback_name)
        if _FINBERT_CACHE:
            return _FINBERT_CACHE
    except Exception as e:
        print(f"❌ Fallback FinBERT model also failed: {e}")
        
    print("💔 FinBERT unavailable - sentiment features will be disabled")
    return None


def _to_ts(x) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
        if isinstance(x, datetime):
            return x.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        pass
    return 0.0


def score_sentiment(headline: str) -> float | None:
    text = (headline or "").strip()
    if not text:
        return 0.0
    # Prefer FinBERT if available, cached, and not disabled
    fb = _load_finbert()
    if fb:
        try:
            import torch
            tok = fb["tok"]
            mdl = fb["mdl"]
            device = fb["device"]
            with torch.no_grad():
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=int(
                    os.environ.get("FINBERT_MAX_LEN", 128)), padding="max_length").to(device)
                logits = mdl(**inputs).logits.squeeze()
                probs = torch.softmax(logits, dim=-1).tolist()
                return float(probs[2] - probs[0])  # positive - negative
        except Exception:
            pass
    if SentimentIntensityAnalyzer:
        try:
            sia = SentimentIntensityAnalyzer()
            return float(sia.polarity_scores(text).get("compound"))
        except Exception:
            return 0.0
    return 0.0


def rank_news(items: List[Dict], days: int = 7) -> List[Dict]:
    """Rank news items by recency and sentiment strength."""
    scored = []
    for it in items:
        h = it.get("headline") or it.get("title") or ""
        if not h.strip():
            continue
        s = score_sentiment(h)
        ts = _to_ts(it.get("date"))
        it2 = dict(it)
        if not it2.get("headline"):
            it2["headline"] = h
        if not it2.get("source") and it2.get("publisher"):
            it2["source"] = it2.get("publisher")
        it2["sentiment"] = s
        it2["timestamp"] = ts
        scored.append(it2)
    scored.sort(
        key=lambda d: (
            d.get(
                "timestamp", 0) or 0, abs(
                d.get("sentiment") or 0)), reverse=True)
    return scored


class NewsProvider:
    """Lightweight news provider using yfinance when available."""
    def __init__(self):
        try:
            import yfinance as yf  # local import to avoid hard dep at import time
            self._yf = yf
        except Exception:
            self._yf = None

    def get_recent_news(self, symbol: str, days: int = 7, limit: int = 50) -> List[Dict]:
        items: List[Dict] = []
        if not self._yf:
            return items
        try:
            tk = self._yf.Ticker(symbol)
            raw = getattr(tk, "news", []) or []
            for it in raw[:limit]:
                items.append({
                    "headline": it.get("title") or it.get("headline") or "",
                    "date": it.get("providerPublishTime") or it.get("publishedAt"),
                    "source": it.get("publisher") or it.get("source"),
                    "link": it.get("link") or it.get("url"),
                })
        except Exception:
            return []
        return [i for i in items if (i.get("headline") or "").strip()]

    def get_headlines(self, symbol: str, days: int = 7, limit: int = 50) -> List[str]:
        return [n.get("headline", "") for n in self.get_recent_news(symbol, days, limit)]
