"""
FinBERT sentiment analysis for financial news headlines.
Uses the FinBERT model trained specifically on financial text.
"""

import os
import warnings
from datetime import timezone
from typing import Dict, List, Optional, Union

import pandas as pd

# Lazy import torch and transformers to avoid potential module load issues
torch = None
TRANSFORMERS_AVAILABLE = None  # Will be checked lazily
transformers = None
AutoModelForSequenceClassification = None
AutoTokenizer = None
pipeline = None

def _lazy_import_transformers():
    """Lazy import transformers and torch to avoid loading at module level."""
    global TRANSFORMERS_AVAILABLE, transformers, AutoModelForSequenceClassification, AutoTokenizer, pipeline, torch
    if TRANSFORMERS_AVAILABLE is not None:
        return TRANSFORMERS_AVAILABLE
    try:
        # Import torch first for GPU support
        import torch as _torch  # type: ignore
        torch = _torch
        
        import transformers as _transformers  # type: ignore
        transformers = _transformers
        AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
        AutoTokenizer = transformers.AutoTokenizer
        pipeline = transformers.pipeline
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        warnings.warn(
            "Transformers library not installed. FinBERT will not be available. To enable, install with: pip install transformers")
    return TRANSFORMERS_AVAILABLE

# Cache for model and tokenizer to avoid reloading
_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None
_FINBERT_DEVICE_INDEX: Optional[int] = None

MAX_FINBERT_SEQ_LENGTH = 512
DEFAULT_FINBERT_BATCH_SIZE = int(os.environ.get("FINBERT_BATCH_SIZE", "128"))


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or .env file"""
    # Use provided token if environment variables not set
    fallback_token = "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"
    
    # Check environment variable first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    
    # Check .env file in project root
    try:
        from pathlib import Path
        env_file = Path(__file__).parent.parent.parent / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        return line.split('=', 1)[1].strip().strip('"\'')
    except Exception:
        pass
    
    # Fall back to hardcoded token as last resort
    return fallback_token


def _should_use_gpu() -> bool:
    flag = os.environ.get("FINBERT_USE_GPU")
    if flag is None:
        return True  # default to GPU when available unless explicitly disabled
    return str(flag).lower() in {"1", "true", "yes", "y"}


def _resolve_finbert_device() -> int:
    """Return huggingface pipeline device index (-1 CPU, >=0 GPU)."""
    global _FINBERT_DEVICE_INDEX, torch

    if _FINBERT_DEVICE_INDEX is not None:
        return _FINBERT_DEVICE_INDEX

    if not _should_use_gpu():
        _FINBERT_DEVICE_INDEX = -1
        return _FINBERT_DEVICE_INDEX

    # Ensure torch is imported before checking GPU availability
    if torch is None:
        try:
            import torch as _torch  # type: ignore
            torch = _torch
        except ImportError:
            warnings.warn("FINBERT GPU requested but torch not installed; using CPU instead")
            _FINBERT_DEVICE_INDEX = -1
            return _FINBERT_DEVICE_INDEX

    if not torch.cuda.is_available():
        warnings.warn("FINBERT GPU requested but no CUDA device detected; using CPU instead")
        _FINBERT_DEVICE_INDEX = -1
        return _FINBERT_DEVICE_INDEX

    _FINBERT_DEVICE_INDEX = 0
    os.environ.setdefault("FINBERT_USE_GPU", "1")
    return _FINBERT_DEVICE_INDEX


def _maybe_move_model_to_device(model) -> None:
    device_index = _resolve_finbert_device()
    if device_index < 0:
        return
    if torch is None:
        return
    try:
        model.to(torch.device(f"cuda:{device_index}"))  # type: ignore[attr-defined]
        model.eval()
    except Exception as exc:  # pragma: no cover - only when GPU move fails
        warnings.warn(f"Failed to place FinBERT on GPU: {exc}")


def get_finbert_model():
    """Load and cache FinBERT model with PyTorch 2.6+ safetensors compatibility and HF authentication"""
    global _FINBERT_MODEL, _FINBERT_TOKENIZER

    # If transformers is not available, return None immediately
    if not _lazy_import_transformers():
        print("⚠️ Transformers library not available - FinBERT disabled")
        return None, None

    if _FINBERT_MODEL is None:
        try:
            # Set up HuggingFace authentication
            hf_token = _get_hf_token()
            if hf_token:
                print("🔑 Using HuggingFace authentication token")
                # Set token for current session
                os.environ.setdefault("HF_TOKEN", hf_token)
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            else:
                print("⚠️ No HuggingFace token found - may hit rate limits")
            
            print("🚀 Loading FinBERT model with safetensors...")
            # Primary model - optimized for financial sentiment
            model_name = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")
            fallback_name = "ProsusAI/finbert"

            # Check for offline model path first
            offline_path = os.environ.get("FINBERT_MODEL_PATH")
            if offline_path and os.path.exists(offline_path):
                print(f"📁 Using local FinBERT model from {offline_path}")
                _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
                    offline_path,
                    use_safetensors=True,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True
                )
                _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(
                    offline_path,
                    use_fast=True,
                    trust_remote_code=False
                )
                try:
                    _FINBERT_TOKENIZER.model_max_length = MAX_FINBERT_SEQ_LENGTH
                except Exception:
                    pass
            else:
                # Prepare authentication kwargs
                auth_kwargs = {}
                if hf_token:
                    auth_kwargs['token'] = hf_token
                
                # Try primary model with safetensors - force safetensors to avoid torch.load
                try:
                    print(f"🔄 Loading {model_name} with safetensors...")
                    _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        use_safetensors=True,
                        trust_remote_code=False,  # Security: disable remote code
                        low_cpu_mem_usage=True,
                        local_files_only=False,
                        revision="main",
                        force_download=False,  # Use cache when available
                        **auth_kwargs
                    )
                    _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=True,
                        trust_remote_code=False,
                        local_files_only=False,
                        **auth_kwargs
                    )
                    try:
                        _FINBERT_TOKENIZER.model_max_length = MAX_FINBERT_SEQ_LENGTH
                    except Exception:
                        pass
                    print(f"✅ Successfully loaded {model_name}")
                    
                except Exception as e:
                    print(f"⚠️ Primary model failed: {e}")
                    print(f"🔄 Trying fallback model {fallback_name}...")
                    
                    try:
                        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
                            fallback_name,
                            use_safetensors=True,
                            trust_remote_code=False,
                            low_cpu_mem_usage=True,
                            local_files_only=False,
                            force_download=False,
                            **auth_kwargs
                        )
                    except Exception as fallback_error:
                        print(f"❌ Fallback model also failed: {fallback_error}")
                        print("🔄 Attempting CPU-only fallback model without strict safetensors...")
                        
                        # Last resort: try loading without strict requirements
                        try:
                            _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
                                fallback_name,
                                low_cpu_mem_usage=True,
                                **auth_kwargs
                            )
                            print(f"✅ CPU fallback model {fallback_name} loaded successfully")
                        except Exception as final_error:
                            print(f"❌ All FinBERT loading attempts failed: {final_error}")
                            print("💡 FinBERT disabled - using simple sentiment fallback")
                            return None, None
                    _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(
                        fallback_name,
                        use_fast=True,
                        trust_remote_code=False,
                        local_files_only=False,
                        **auth_kwargs
                    )
                    try:
                        _FINBERT_TOKENIZER.model_max_length = MAX_FINBERT_SEQ_LENGTH
                    except Exception:
                        pass
                    print(f"✅ Successfully loaded fallback {fallback_name}")

            # Test model functionality
            if torch and _FINBERT_MODEL and _FINBERT_TOKENIZER:
                test_text = "The market outlook is positive."
                with torch.no_grad():
                    inputs = _FINBERT_TOKENIZER(
                        test_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding="max_length",
                    )
                    _ = _FINBERT_MODEL(**inputs)
                
            print("🎯 FinBERT model loaded and verified successfully")
            _maybe_move_model_to_device(_FINBERT_MODEL)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error loading FinBERT model: {error_msg}")
            
            # Check if it's the PyTorch version issue
            if "torch.load" in error_msg and "vulnerability" in error_msg:
                print("🔧 PyTorch version incompatibility detected")
                print("💡 FinBERT disabled due to torch.load security restrictions")
                print("📝 Solution: Upgrade PyTorch to 2.6+ or use safetensors models")
            else:
                print("💡 Try: pip install --upgrade transformers safetensors torch")
            
            print("🔄 Using simple sentiment analysis fallback")
            # Fallback to None - will use simple sentiment analysis
            return None, None

    if _FINBERT_MODEL is not None:
        _maybe_move_model_to_device(_FINBERT_MODEL)

    return _FINBERT_MODEL, _FINBERT_TOKENIZER


def _fallback_sentiment_analysis(
        text: Union[str, List[str]]) -> List[Dict[str, float]]:
    """
    Fallback sentiment analysis using a simple approach when FinBERT is not available.

    Args:
        text: Single text string or list of text strings

    Returns:
        List of dictionaries with sentiment scores (positive, negative, neutral)
    """
    # Import here to avoid circular imports
    from .news import score_sentiment

    if isinstance(text, list):
        return [{"positive": max(0, score_sentiment(t)),
                "negative": max(0, -score_sentiment(t)),
                 "neutral": 0.5} for t in text]
    else:
        score = score_sentiment(text)
        return [{"positive": max(0, score),
                "negative": max(0, -score),
                 "neutral": 0.5}]


def _process_batch_with_finbert(
    batch: List[str], nlp) -> List[Dict[str, float]]:
    """
    Process a batch of texts with FinBERT model.

    Args:
        batch: List of text strings
        nlp: HuggingFace pipeline for sentiment analysis

    Returns:
        List of dictionaries with sentiment scores
    """
    try:
        normalized_batch = [
            text if isinstance(text, str) and text.strip() else ""
            for text in batch
        ]
        results = nlp(
            normalized_batch,
            batch_size=len(normalized_batch),
            truncation=True,
            padding="max_length",
            max_length=MAX_FINBERT_SEQ_LENGTH,
        )
        return results
    except Exception as e:
        print(f"Error in batch sentiment analysis: {e}")
        # Fall back to simple scoring for this batch
        return _fallback_sentiment_analysis(batch)


def _process_sentiment_results(results: List[Dict]) -> List[Dict[str, float]]:
    """Process FinBERT pipeline results into standardized format"""
    processed_results = []

    for result in results:
        label = result['label']
        score = result['score']

        if label == 'positive':
            sentiment_scores = {
                'positive': score,
                'negative': 0.0,
                'neutral': 1.0 - score}
        elif label == 'negative':
            sentiment_scores = {
                'positive': 0.0,
                'negative': score,
                'neutral': 1.0 - score}
        else:  # neutral
            sentiment_scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': score}

        processed_results.append(sentiment_scores)

    return processed_results


def _run_finbert_pipeline(texts: List[str], nlp) -> List[Dict[str, float]]:
    """Run FinBERT sentiment analysis pipeline on texts"""
    if not texts:
        return []

    processed_results: List[Dict[str, float]] = []

    batch_size = max(1, DEFAULT_FINBERT_BATCH_SIZE)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_results = _process_batch_with_finbert(batch, nlp)

        for text, result in zip(batch, batch_results):
            if not isinstance(result, dict) or 'label' not in result:
                processed_results.append({
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'confidence': 0.0  # Add confidence score
                })
                continue

            label = result.get('label', 'neutral').lower()  # ✅ FIX: Convert to lowercase
            score = float(result.get('score', 0.0))

            if label == 'positive':
                sentiment_scores = {
                    'positive': score,
                    'negative': 0.0,
                    'neutral': 1.0 - score,
                    'confidence': score  # Model confidence for this prediction
                }
            elif label == 'negative':
                sentiment_scores = {
                    'positive': 0.0,
                    'negative': score,
                    'neutral': 1.0 - score,
                    'confidence': score  # Model confidence for this prediction
                }
            else:  # neutral or other labels
                sentiment_scores = {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': score,
                    'confidence': score  # Model confidence for this prediction
                }

            processed_results.append(sentiment_scores)

    # Handle any length mismatch between inputs and results
    while len(processed_results) < len(texts):
        processed_results.append({'positive': 0.0, 'negative': 0.0, 'neutral': 1.0})

    if len(processed_results) > len(texts):
        processed_results = processed_results[:len(texts)]

    return processed_results


def analyze_finbert_sentiment(
        text: Union[str, List[str]]) -> List[Dict[str, float]]:
    """
    Analyze sentiment of text using FinBERT.

    Args:
        text: Single text string or list of text strings

    Returns:
        List of dictionaries with sentiment scores (positive, negative, neutral)
    """
    model, tokenizer = get_finbert_model()

    # If model is not available, use fallback
    if model is None or tokenizer is None:
        return _fallback_sentiment_analysis(text)

    try:
        # Create sentiment analysis pipeline
        device_index = _resolve_finbert_device()
        nlp = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
        )

        # Process the text(s)
        texts = [text] if isinstance(text, str) else text

        return _run_finbert_pipeline(texts, nlp)

    except Exception as e:
        print(f"Error in FinBERT sentiment analysis: {e}")
        return _fallback_sentiment_analysis(text)


def _ultimate_fallback_sentiment_analysis(
        text: Union[str, List[str]]) -> List[Dict[str, float]]:
    """Ultimate fallback when all other methods fail"""
    try:
        # Very basic fallback
        if isinstance(text, list):
            return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                    for _ in text]
        else:
            return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}]
    except Exception:
        # Ultimate fallback if even this fails
        if isinstance(text, list):
            return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                    for _ in range(len(text))]
        else:
            return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}]


def get_finbert_sentiment_score(text: str) -> float:
    """
    Get a single sentiment score from FinBERT (-1 to 1 range).

    Args:
        text: Text to analyze

    Returns:
        Score between -1 (negative) and 1 (positive)
    """
    try:
        result = analyze_finbert_sentiment(text)[0]
        # Calculate a weighted score: 1*positive + (-1)*negative + 0*neutral
        score = result.get("positive", 0.0) - result.get("negative", 0.0)
        return max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
    except Exception as e:
        print(f"Error getting FinBERT sentiment score: {str(e)}")
        # Fallback to simple sentiment scoring
        from dcf_lab.news import score_sentiment
        return score_sentiment(text)


def build_finbert_sentiment_features(
        news_items: List[Dict],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Build sentiment features from news items using FinBERT.

    Args:
        news_items: List of news items with 'headline' and 'date' fields
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with date index and sentiment features
    """
    from datetime import datetime

    import pandas as pd

    if not news_items:
        return pd.DataFrame(
            columns=[
                "date",
                "finbert_sent_7d",
                "finbert_sent_30d"]).set_index("date")

    # Set default date range if not provided
    if end_date is None:
        end_date = pd.Timestamp.now()
    if start_date is None:
        start_date = end_date - pd.Timedelta(days=90)

    # Extract headlines for batch processing
    headlines = [item.get("headline", "") for item in news_items]
    # Get sentiment scores in batch
    sentiments = analyze_finbert_sentiment(headlines)

    rows = []
    for item, sentiment in zip(news_items, sentiments):
        ts = item.get("date")
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            elif isinstance(ts, str):
                dt = pd.to_datetime(ts)
            else:
                dt = None
        except Exception:
            dt = None

        if dt and start_date <= pd.Timestamp(dt) <= end_date:
            # Calculate a single score: positive - negative
            score = sentiment.get("positive", 0.0) - \
                sentiment.get("negative", 0.0)
            rows.append({"date": dt.date(), "score": score})

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "finbert_sent_7d",
                "finbert_sent_30d"]).set_index("date")

    # Aggregate by date
    df = pd.DataFrame(rows)
    daily = df.groupby("date")["score"].mean().to_frame("score")
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").fillna(method="ffill")

    # Calculate rolling averages
    daily["finbert_sent_7d"] = daily["score"].rolling(7, min_periods=1).mean()
    daily["finbert_sent_30d"] = daily["score"].rolling(
        30, min_periods=1).mean()

    return daily[["finbert_sent_7d", "finbert_sent_30d"]]


class FinBERTSentimentAnalyzer:
    """Simple wrapper class providing a consistent analyzer interface.
    Loads ProsusAI/finbert via transformers and exposes analyze() helpers.
    """
    def __init__(self):
        self.model, self.tokenizer = get_finbert_model()
        self.available = self.model is not None and self.tokenizer is not None

    def is_available(self) -> bool:
        return self.available

    def analyze(self, texts):
        """Analyze a string or list of strings and return list of dicts with positive/negative/neutral.
        Falls back automatically if transformers/model are unavailable.
        """
        return analyze_finbert_sentiment(texts)

    def score(self, text: str) -> float:
        """Return compound score in [-1, 1]."""
        return get_finbert_sentiment_score(text)

    def build_features(self, news_items, start_date=None, end_date=None):
        """Return a dataframe with rolling FinBERT features for news stream."""
        return build_finbert_sentiment_features(news_items, start_date, end_date)

    # Adapter for UniversalFeatureAggregator
    def get_finbert_sentiment_features(self, symbol: str) -> dict:
        """Adapter: fetch recent news via existing news loader if available and return last feature snapshot.
        Falls back to empty dict if upstream loaders are unavailable.
        """
        try:
            from src.dcf_lab.earnings_transcript_analyzer import get_recent_news
        except Exception:
            get_recent_news = None
        try:
            if get_recent_news is None:
                return {}
            news = get_recent_news(symbol, days=30) or []
            if not news:
                return {}
            df = self.build_features(news)
            if df is None or df.empty:
                return {}
            row = df.iloc[-1]
            out = {}
            for k, v in row.dropna().items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out
        except Exception:
            return {}
