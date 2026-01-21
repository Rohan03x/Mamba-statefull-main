"""Shared utilities for Hugging Face powered signal modules ("brains").

This module centralises lazy loading of Hugging Face pipelines, device/cache
resolution and lightweight helpers for constructing :class:`ModuleSignal`
instances. Every helper here is intentionally defensive so that optional
Hugging Face dependencies can fail gracefully without breaking the wider
pipeline.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..signal_bus import ModuleSignal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:  # pragma: no cover - optional dependency hints
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow CPU-only environments
    torch = None  # type: ignore

if torch is not None:  # pragma: no cover - optional dependency hints
    try:
        import torch.nn.functional as F  # type: ignore
    except Exception:  # pragma: no cover - optional dependency hints
        F = None  # type: ignore
else:  # pragma: no cover - optional dependency hints
    F = None  # type: ignore


class TransformersUnavailable(RuntimeError):
    """Raised when :mod:`transformers` can't be imported."""


_PIPELINE_CACHE: Dict[Tuple[Any, ...], Any] = {}
_WARNED_KEYS: Dict[str, bool] = {}


@dataclass
class HFSpec:
    """Configuration bundle for ``HFBrain`` text classification runners."""

    model_id: str
    tokenizer_id: Optional[str] = None
    revision: Optional[str] = None
    max_length: int = 512
    stride: int = 128
    batch_size: int = 16
    device: Optional[str] = None
    dtype: str = "fp16"


class HFBrain:
    """Thin wrapper around AutoModelForSequenceClassification for reuse."""

    def __init__(self, spec: HFSpec):
        if torch is None:
            raise RuntimeError("PyTorch is required for HFBrain but is not installed")
        if F is None:
            raise RuntimeError("torch.nn.functional could not be imported; check PyTorch installation")

        transformers = _import_transformers()
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")

        self.spec = spec
        if spec.device:
            device = spec.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        prefers_fp16 = (str(spec.dtype).lower() == "fp16") and device and device.startswith("cuda")
        self.dtype = torch.float16 if prefers_fp16 else torch.float32

        tokenizer_kwargs: Dict[str, Any] = {"use_fast": True}
        if spec.revision:
            tokenizer_kwargs["revision"] = spec.revision
        tokenizer_id = spec.tokenizer_id or spec.model_id
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)

        model_kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if spec.revision:
            model_kwargs["revision"] = spec.revision
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(spec.model_id, **model_kwargs)
            .to(self.device)
            .eval()
        )

    def _batches(self, items: Iterable[str], batch_size: int) -> Iterable[List[str]]:
        buffer: List[str] = []
        for item in items:
            buffer.append(item)
            if len(buffer) == batch_size:
                yield list(buffer)
                buffer.clear()
        if buffer:
            yield list(buffer)

    def infer_probs(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        outputs: List[float] = []
        with torch.no_grad():
            for chunk in self._batches(texts, self.spec.batch_size):
                encoded = self.tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probs = F.softmax(logits, dim=1)[:, 1]
                outputs.extend(probs.detach().float().cpu().tolist())
        return outputs

    def infer_probs_from_token_ids(self, token_sequences: Sequence[Sequence[int]]) -> List[float]:
        """Run inference directly on pre-tokenised sequences."""

        if not token_sequences:
            return []

        prepared_batches: List[Mapping[str, Any]] = []
        for seq in token_sequences:
            if not seq:
                continue
            try:
                token_list = [int(token) for token in seq]
            except Exception:
                continue
            prepared = self.tok.prepare_for_model(
                token_list,
                add_special_tokens=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            prepared_batches.append(prepared)

        if not prepared_batches:
            return []

        encoded = self.tok.pad(
            prepared_batches,
            padding=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=1)[:, 1]
            return probs.detach().float().cpu().tolist()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    _WARNED_KEYS[key] = True


def _import_transformers() -> Any:
    try:
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TransformersUnavailable(
            "transformers is required for Hugging Face brain modules; install it via 'pip install transformers'."
        ) from exc
    return transformers


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(v) for v in value)
    return value


def _normalise_cache_dir(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(path_value)))
    try:
        cache_path = Path(expanded).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        return str(cache_path)
    except Exception:  # pragma: no cover - filesystem guard
        _warn_once("hf_cache", f"Unable to create Hugging Face cache directory at {expanded}")
        return None


def resolve_cache_dir(explicit: Optional[str] = None) -> Optional[str]:
    candidates = [
        explicit,
        os.environ.get("HF_BRAINS_CACHE_DIR"),
        os.environ.get("TRANSFORMERS_CACHE"),
        os.environ.get("HF_HOME"),
    ]
    for candidate in candidates:
        resolved = _normalise_cache_dir(candidate)
        if resolved:
            return resolved
    # Default under ~/.cache/hf_brains
    home_default = Path.home() / ".cache" / "hf_brains"
    return _normalise_cache_dir(str(home_default))


def resolve_auth_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip() or None
    return None


def hf_offline_enabled() -> bool:
    flag = os.environ.get("HF_BRAINS_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")
    if flag is None:
        return False
    return str(flag).strip().lower() in {"1", "true", "yes", "y"}


def resolve_device_preference(preferred: Optional[str] = None) -> Optional[Union[int, str]]:
    device_pref = preferred or os.environ.get("HF_BRAINS_DEVICE") or "auto"
    device_pref = str(device_pref).strip().lower()
    if device_pref in {"", "auto"}:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            return 0
        return -1
    if device_pref in {"cpu", "-1"}:
        return -1
    if device_pref in {"cuda", "gpu", "0"}:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return 0
        return -1
    if device_pref.startswith("cuda:"):
        try:
            index = int(device_pref.split(":", 1)[1])
        except Exception:
            index = 0
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            device_count = torch.cuda.device_count() if torch is not None else 0
            if device_count and 0 <= index < device_count:
                return index
        return -1
    if device_pref in {"mps"}:
        return "mps"
    try:
        index = int(device_pref)
    except ValueError:
        return device_pref
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device_count = torch.cuda.device_count() if torch is not None else 0
        if device_count and 0 <= index < device_count:
            return index
    return -1


def get_pipeline(
    task: str,
    model: str,
    *,
    tokenizer: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    tokenizer_kwargs: Optional[Mapping[str, Any]] = None,
    pipeline_kwargs: Optional[Mapping[str, Any]] = None,
) -> Optional[Any]:
    """Create (or reuse) a Hugging Face pipeline with defensive defaults."""

    cache_key = (
        task,
        model,
        tokenizer,
        resolve_device_preference(device),
        cache_dir,
        revision,
        trust_remote_code,
        _freeze(model_kwargs),
        _freeze(tokenizer_kwargs),
        _freeze(pipeline_kwargs),
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    try:
        transformers = _import_transformers()
    except TransformersUnavailable as exc:
        _warn_once("transformers", str(exc))
        return None

    runtime_kwargs: Dict[str, Any] = dict(pipeline_kwargs or {})
    model_kwargs_resolved: Dict[str, Any] = dict(model_kwargs or {})
    tokenizer_kwargs_resolved: Dict[str, Any] = dict(tokenizer_kwargs or {})
    cache_dir_resolved = resolve_cache_dir(cache_dir)
    if cache_dir_resolved and "cache_dir" not in runtime_kwargs:
        runtime_kwargs["cache_dir"] = cache_dir_resolved
        runtime_kwargs.setdefault("hf_cache_folder", cache_dir_resolved)
    if revision and "revision" not in runtime_kwargs:
        runtime_kwargs["revision"] = revision
    if trust_remote_code:
        runtime_kwargs["trust_remote_code"] = True
    if hf_offline_enabled():
        runtime_kwargs.setdefault("local_files_only", True)
    token = resolve_auth_token()
    if token and "token" not in runtime_kwargs and "use_auth_token" not in runtime_kwargs:
        runtime_kwargs["token"] = token
    device_spec = resolve_device_preference(device)
    if device_spec is not None and "device" not in runtime_kwargs:
        runtime_kwargs["device"] = device_spec

    prefers_gpu = False
    if torch is not None:
        if isinstance(device_spec, int) and device_spec >= 0 and torch.cuda.is_available():  # type: ignore[attr-defined]
            prefers_gpu = True
        elif isinstance(device_spec, str) and device_spec.startswith("cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            prefers_gpu = True
    if prefers_gpu:
        model_kwargs_resolved.setdefault("torch_dtype", torch.float16)

    tokenizer_kwargs_resolved.setdefault("use_fast", True)

    factory_kwargs: Dict[str, Any] = {
        "task": task,
        "model": model,
    }
    if tokenizer:
        factory_kwargs["tokenizer"] = tokenizer
    if model_kwargs_resolved:
        factory_kwargs["model_kwargs"] = model_kwargs_resolved
    if tokenizer_kwargs_resolved:
        factory_kwargs["tokenizer_kwargs"] = tokenizer_kwargs_resolved
    factory_kwargs.update(runtime_kwargs)

    try:
        pipeline = transformers.pipeline(**factory_kwargs)
    except Exception as exc:  # pragma: no cover - external dependency failures
        _warn_once("hf_pipeline", f"Failed to initialise Hugging Face pipeline for {model}: {exc}")
        return None

    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def batched_text_classification(
    classifier: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 16,
    max_length: int = 256,
) -> Sequence[Mapping[str, Any]]:
    if not texts:
        return []
    texts_normalized = [text or "" for text in texts]
    order = list(range(len(texts_normalized)))
    order.sort(key=lambda idx: len(texts_normalized[idx]))
    batched_results: Dict[int, Mapping[str, Any]] = {}

    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        chunk = [texts_normalized[idx] for idx in batch_indices]
        try:
            chunk_results = classifier(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            _warn_once("hf_batch", f"HF classifier batch failed: {exc}")
            for idx in batch_indices:
                batched_results.setdefault(idx, {})
            continue
        if isinstance(chunk_results, Mapping):
            chunk_results = [chunk_results]
        for idx, result in zip(batch_indices, chunk_results):
            batched_results[idx] = result

    ordered_results: List[Mapping[str, Any]] = []
    for idx in range(len(texts_normalized)):
        ordered_results.append(batched_results.get(idx, {}))
    return ordered_results


def label_score(result: Mapping[str, Any]) -> float:
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))
    if "positive" in label or label == "bullish":
        return max(-1.0, min(1.0, score))
    if "negative" in label or label == "bearish":
        return max(-1.0, min(1.0, -score))
    if "neutral" in label:
        return 0.0
    # Unknown labels default to centred score
    return max(-1.0, min(1.0, score * 0.0))


def confidence_from_count(count: int, *, scale: float = 12.0) -> float:
    if count <= 0:
        return 0.0
    return float(min(1.0, math.log1p(count) / math.log1p(scale)))


def coerce_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)
    return ts.normalize()


def build_signal(
    name: str,
    horizon: Union[int, str],
    rows: Iterable[Tuple[Any, float, float]],
    *,
    min_points: int = 1,
) -> Optional[ModuleSignal]:
    df = pd.DataFrame(list(rows), columns=["date", "score", "conf"])
    if df.empty:
        return None
    df["date"] = df["date"].apply(coerce_date)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    if df.empty or len(df) < min_points:
        return None
    df["score"] = df["score"].astype(float).clip(-1.0, 1.0)
    if "conf" in df:
        df["conf"] = df["conf"].astype(float).fillna(0.0).clip(0.0, 1.0)
    else:
        df["conf"] = 1.0
    try:
        horizon_int = int(float(horizon))
    except Exception:
        horizon_int = int(horizon) if isinstance(horizon, int) else 1
    return ModuleSignal(name=name, horizon=horizon_int, df=df[["score", "conf"]])


def embed_texts(
    extractor: Any,
    texts: Sequence[str],
    *,
    normalize: bool = True,
    max_length: int = 256,
) -> Optional[np.ndarray]:
    if not texts:
        return None
    try:
        outputs = extractor(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
    except Exception as exc:  # pragma: no cover - transformer runtime failures
        _warn_once("hf_embed", f"HF feature extractor failed: {exc}")
        return None
    if isinstance(outputs, Mapping):
        outputs = [outputs]
    vectors: list[np.ndarray] = []
    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype=np.float32)
        if normalize:
            norm = float(np.linalg.norm(arr))
            if norm > 1e-8:
                arr = arr / norm
        vectors.append(arr)
    if not vectors:
        return None
    return np.vstack(vectors)


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return (1.0 - cosine) / 2.0


__all__ = [
    "TransformersUnavailable",
    "HFSpec",
    "HFBrain",
    "get_pipeline",
    "batched_text_classification",
    "label_score",
    "confidence_from_count",
    "coerce_date",
    "build_signal",
    "embed_texts",
    "cosine_distance",
    "resolve_cache_dir",
    "resolve_auth_token",
    "resolve_device_preference",
    "hf_offline_enabled",
]
