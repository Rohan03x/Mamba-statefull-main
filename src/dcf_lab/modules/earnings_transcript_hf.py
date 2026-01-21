"""HEDGE-FUND GRADE: Quarterly Earnings Transcript Sentiment (10 features)

PHILOSOPHY: Confidence-weighted narrative changes, NOT raw sentiment levels

A. Overall Sentiment (2): CORE
   - score: Normalized sentiment (-1 to 1), cross-sectionally comparable
   - conf: CRITICAL - intensity × volume × dispersion penalty
     Funds care less about "is sentiment positive?" and more about "how confident?"

B. Sectional Sentiment (2): KEEP BOTH
   - score_prepared: Management narrative (often biased upward)
   - score_qa: Reveals stress/defensiveness (contradicts prepared remarks)
   Funds explicitly look for divergence between sections

C. Tone Dimensions (2): VOL AMPLIFIERS
   - uncertainty_score: Information asymmetry (NOT bearish, volatility amplifier)
   - risk_score: Explicit downside language (pairs with cboe_term + credit spreads)

D. Time-Series Changes (2): HIGH ALPHA
   - score_delta_qoq: Short-term trajectory, narrative inflection
   - score_delta_yoy: Removes seasonality, long-term credibility trend
   Deltas far more predictive than raw sentiment levels

E. Interaction Features (2): NEW
   - sentiment_divergence: score_prepared - score_qa (management defensiveness)
   - sentiment_shock: score_delta_qoq × conf (confidence-weighted change)
     Distinguishes noise from real narrative change

DECAY DESIGN (DO NOT CHANGE):
- Strongest on earnings day, rapid decay over 3-4 sessions
- Confidence decays slower than score
- Matches market pricing: fast reaction, quick saturation, lingering confidence

GOVERNANCE:
- Stop exactly here: More NLP causes overfitting (transcripts are events, not streams)
- Confidence-weighted deltas already capture signal
- Do NOT add: topic modeling, LLM summaries, long-window averages

Data Flow:
1. Loads transcripts from data_cache/earnings_transcripts/ (JSON/TXT/MD/HTML)
2. Splits prepared remarks vs Q&A sections
3. Scores each section with FinBERT sentiment + tone analysis
4. Computes QoQ/YoY deltas from quarterly history
5. Adds interaction features (divergence, shock)
6. Aggregates to daily time series with decay weights

Total: 10 features (optimal size for event-driven sentiment)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from ..hf_brains.base import HFBrain, HFSpec
from ..signal_bus import ModuleSignal
from ..utils import to_market_session

try:  # pragma: no cover - optional fallback if analyzer is available
    from ..earnings_transcript_analyzer import EarningsTranscriptAnalyzer  # type: ignore
except Exception:  # pragma: no cover - analyzer is optional
    EarningsTranscriptAnalyzer = None  # type: ignore

LOGGER = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_TRANSCRIPT_CACHE = _ROOT / "data_cache" / "earnings_transcripts"
_HF_INFERENCE_CACHE = _ROOT / "data_cache" / "hf_transcript_inference"
_FILENAME_DATE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_QUARTER_PATTERN = re.compile(r"[Qq]([1-4])[^0-9]*(20\d{2})")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fetch_defeatbeta_transcripts(symbol: str, *, max_records: int) -> List[Dict[str, object]]:
    """Fetch real earnings transcripts via defeatbeta-api (best effort).

    Returns records matching the on-disk cache schema consumed by `_load_cached_transcripts`.
    """
    try:
        from defeatbeta_api.data.ticker import Ticker  # type: ignore
    except Exception:
        return []

    try:
        ticker = Ticker(symbol)
        transcripts_obj = ticker.earning_call_transcripts()
        transcripts_df = transcripts_obj.get_transcripts_list()
        if not isinstance(transcripts_df, pd.DataFrame) or transcripts_df.empty:
            return []

        df = transcripts_df.copy()
        if "report_date" not in df.columns:
            return []

        df["timestamp"] = pd.to_datetime(df["report_date"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return []

        # Some payloads store paragraphs under `transcripts` as list/np.ndarray of dicts.
        texts: List[str] = []
        for _, row in df.iterrows():
            paras = row.get("transcripts")
            text = ""
            if isinstance(paras, (list, tuple, np.ndarray)):
                parts: List[str] = []
                for para in paras:
                    if isinstance(para, Mapping):
                        content = str(para.get("content") or "").strip()
                        if content:
                            parts.append(content)
                text = " ".join(parts)
            else:
                text = str(paras or "")
            texts.append(text)

        df["text"] = texts
        df = df[df["text"].astype(str).str.len() > 100]
        if df.empty:
            return []

        df = df.sort_values("timestamp", ascending=False)

        records: List[Dict[str, object]] = []
        for _, row in df.head(max(1, int(max_records))).iterrows():
            ts: pd.Timestamp = row["timestamp"]
            try:
                period = ts.to_period("Q")
                quarter = f"{period.year}Q{period.quarter}"
            except Exception:
                quarter = None

            records.append(
                {
                    "timestamp": ts,
                    "text": str(row.get("text") or ""),
                    "quarter": quarter,
                    "symbol": str(symbol or "").upper(),
                    "source": "defeatbeta_api",
                }
            )

        return records
    except Exception as exc:
        LOGGER.debug("defeatbeta transcript fetch failed for %s: %s", symbol, exc)
        return []


def _persist_transcript_cache(symbol: str, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    try:
        symbol_upper = str(symbol or "").upper() or "UNKNOWN"
        out_dir = _TRANSCRIPT_CACHE / symbol_upper
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d")
        out_path = out_dir / f"defeatbeta_api_{stamp}.json"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        payload = {"source": "defeatbeta_api", "symbol": symbol_upper, "records": list(records)}
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
        tmp_path.replace(out_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Failed to persist transcript cache for %s: %s", symbol, exc)


def _slugify_segment(value: str, *, fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    slug = slug.strip("._-") or fallback
    return slug.replace("/", "__")


def _model_cache_key(spec: HFSpec) -> str:
    revision = spec.revision or "latest"
    return _slugify_segment(f"{spec.model_id}@{revision}")


def _symbol_cache_key(symbol: str) -> str:
    return _slugify_segment(symbol or "unknown", fallback="unknown")


def _document_cache_key(
    symbol: str,
    timestamp: Optional[pd.Timestamp],
    quarter: Optional[str],
    text_hash: str,
) -> str:
    parts: List[str] = []
    if quarter:
        parts.append(str(quarter).upper())
    if isinstance(timestamp, pd.Timestamp):
        ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
        parts.append(ts.strftime("%Y%m%d"))
    if not parts:
        parts.append(text_hash[:16])
    parts.append(_slugify_segment(symbol or "unknown"))
    return "__".join(parts)


def _inference_cache_path(spec: HFSpec, symbol: str, doc_key: str) -> Path:
    model_dir = _HF_INFERENCE_CACHE / _model_cache_key(spec)
    symbol_dir = model_dir / _symbol_cache_key(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    slug_key = _slugify_segment(doc_key, fallback="document")
    return symbol_dir / f"{slug_key}.json"


def _load_inference_payload(path: Path) -> Optional[Mapping[str, object]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - cache resilience
        LOGGER.debug("Failed to read HF inference cache %s: %s", path, exc)
        return None


def _store_inference_payload(path: Path, payload: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        tmp_path.replace(path)
    except Exception as exc:  # pragma: no cover - cache resilience
        LOGGER.debug("Failed to persist HF inference cache %s: %s", path, exc)


def _coerce_token_chunks(raw: object, *, max_chunks: int) -> List[List[int]]:
    if not isinstance(raw, list):
        return []
    token_chunks: List[List[int]] = []
    for chunk in raw[:max_chunks]:
        if isinstance(chunk, list):
            try:
                token_chunks.append([int(token) for token in chunk])
            except Exception:
                continue
    return token_chunks


def _coerce_probabilities(raw: object, *, limit: int) -> List[float]:
    if not isinstance(raw, list):
        return []
    probs: List[float] = []
    for value in raw[:limit]:
        try:
            probs.append(float(value))
        except Exception:
            continue
    return probs


def _coerce_timestamp(value: object) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return ts


def _canonical_quarter(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = _QUARTER_PATTERN.search(str(value))
    if not match:
        return None
    quarter, year = match.groups()
    return f"{year}Q{quarter}"


def _quarter_to_timestamp(label: Optional[str]) -> Optional[pd.Timestamp]:
    canonical = _canonical_quarter(label)
    if not canonical:
        return None
    try:
        period = pd.Period(canonical, freq="Q")
    except Exception:
        return None
    ts = period.to_timestamp(how="end") + pd.Timedelta(hours=21)
    return ts.tz_localize("UTC")


def _timestamp_from_path(path: Path) -> pd.Timestamp:
    match = _FILENAME_DATE.search(path.stem)
    if match:
        year, month, day = map(int, match.groups())
        try:
            ts = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
        except Exception:
            ts = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        return ts + pd.Timedelta(hours=21)
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def _split_transcript_sections(text: str) -> Tuple[str, str]:
    """Split transcript into prepared remarks and Q&A sections."""
    text_lower = text.lower()
    
    # Common markers for Q&A section start
    qa_markers = [
        "question-and-answer", "question and answer", "q&a session",
        "operator\n", "operator:", "first question", "our first question",
        "questions and answers", "qa session", "now open for questions"
    ]
    
    split_idx = -1
    for marker in qa_markers:
        idx = text_lower.find(marker)
        if idx != -1:
            if split_idx == -1 or idx < split_idx:
                split_idx = idx
    
    if split_idx == -1:
        # No clear Q&A section found - treat all as prepared
        return text, ""
    
    prepared = text[:split_idx].strip()
    qa = text[split_idx:].strip()
    return prepared, qa


def _extract_record(data: Mapping[str, object], *, default_ts: Optional[pd.Timestamp]) -> Optional[Dict[str, object]]:
    text = (
        data.get("transcript")
        or data.get("text")
        or data.get("content")
        or data.get("body")
        or data.get("raw")
        or data.get("summary")
    )
    if isinstance(text, list):
        text = "\n".join(str(part) for part in text if part)
    text = (str(text or "").strip())
    if not text:
        return None

    timestamp_value = (
        data.get("timestamp")
        or data.get("date")
        or data.get("datetime")
        or data.get("published")
        or data.get("release_time")
    )
    ts = _coerce_timestamp(timestamp_value)
    if ts is None:
        ts = _quarter_to_timestamp(data.get("quarter") or data.get("period"))
    if ts is None:
        ts = default_ts

    quarter = (
        _canonical_quarter(data.get("quarter") or data.get("period") or data.get("fiscal_quarter"))
        or _canonical_quarter(data.get("label"))
    )

    symbol = (data.get("symbol") or data.get("ticker") or "").strip().upper()

    return {
        "timestamp": ts,
        "text": text,
        "quarter": quarter,
        "symbol": symbol or None,
        "source": data.get("source") or "cache",
    }


def _iter_structure(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        yield payload
        for key in ("records", "items", "transcripts", "data", "results"):
            inner = payload.get(key)
            if isinstance(inner, (list, tuple)):
                for item in inner:
                    if isinstance(item, Mapping):
                        yield from _iter_structure(item)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            if isinstance(item, Mapping):
                yield from _iter_structure(item)


def _load_cached_transcripts(symbol: str, *, max_records: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    # Auto-populate the transcript cache from defeatbeta-api when missing.
    if not _TRANSCRIPT_CACHE.exists() and _env_flag("EARNINGS_TRANSCRIPT_AUTOFETCH", default=True):
        fetched = _fetch_defeatbeta_transcripts(symbol, max_records=max_records)
        if fetched:
            _persist_transcript_cache(symbol, fetched)
        # Continue into the normal loader.

    symbol_upper = symbol.upper()
    candidates = sorted(
        (path for path in _TRANSCRIPT_CACHE.rglob("*") if path.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # If cache exists but is empty, attempt to populate once.
    if not candidates and _env_flag("EARNINGS_TRANSCRIPT_AUTOFETCH", default=True):
        fetched = _fetch_defeatbeta_transcripts(symbol, max_records=max_records)
        if fetched:
            _persist_transcript_cache(symbol, fetched)
            candidates = sorted(
                (path for path in _TRANSCRIPT_CACHE.rglob("*") if path.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

    for path in candidates:
        try:
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                default_ts = _timestamp_from_path(path)
                for item in _iter_structure(payload):
                    record = _extract_record(item, default_ts=default_ts)
                    if not record:
                        continue
                    if record.get("timestamp") is None:
                        continue
                    record_symbol = str(record.get("symbol") or "").upper()
                    if record_symbol and record_symbol != symbol_upper:
                        continue
                    record.setdefault("symbol", symbol_upper)
                    records.append(record)
            elif path.suffix.lower() in {".txt", ".md", ".html"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if symbol_upper not in path.name.upper() and symbol_upper not in content.upper():
                    continue
                records.append(
                    {
                        "timestamp": _timestamp_from_path(path),
                        "text": content,
                        "quarter": _canonical_quarter(path.stem) or _canonical_quarter(content),
                        "symbol": symbol_upper,
                        "source": "cache",
                    }
                )
            else:
                continue
        except Exception as exc:  # pragma: no cover - robust to cache failures
            LOGGER.debug("Failed to read transcript cache %s: %s", path, exc)
            continue
        if len(records) >= max_records:
            break

    return records[:max_records]


def _fallback_transcripts(
    symbol: str,
    *,
    limit: int,
    analyzer: Optional[Any],
    existing_quarters: Sequence[str],
) -> List[Dict[str, object]]:
    if analyzer is None or limit <= 0:
        return []

    existing = {q for q in existing_quarters if q}
    records: List[Dict[str, object]] = []
    current = pd.Timestamp.utcnow().to_period("Q")

    for offset in range(limit * 2):
        period = current - offset
        canonical = f"{period.year}Q{period.quarter}"
        if canonical in existing:
            continue
        quarter_label = f"Q{period.quarter} {period.year}"
        try:
            transcript = analyzer.get_earnings_transcript(symbol, quarter_label)
        except Exception as exc:  # pragma: no cover - analyzer fallback is best effort
            LOGGER.debug("Analyzer fallback failed for %s %s: %s", symbol, quarter_label, exc)
            continue
        if not transcript:
            continue
        ts = period.to_timestamp(how="end") + pd.Timedelta(hours=21)
        records.append(
            {
                "timestamp": ts.tz_localize("UTC"),
                "text": str(transcript),
                "quarter": canonical,
                "symbol": symbol.upper(),
                "source": "analyzer_fallback",
            }
        )
        existing.add(canonical)
        if len(records) >= limit:
            break

    return records


def _prepare_transcript_frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "text", "quarter", "symbol", "source"])

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = df["timestamp"].apply(_coerce_timestamp)
    df["text"] = df["text"].astype(str)
    df = df.dropna(subset=["timestamp", "text"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "text", "quarter", "symbol", "source"])

    df["quarter"] = df["quarter"].apply(_canonical_quarter)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["quarter", "timestamp"], keep="last")
    return df.reset_index(drop=True)


class EarningsTranscriptHF:
    """Summarize earnings call transcripts into a decay-weighted daily signal."""

    NAME = "earnings_transcript_hf"
    metadata = {
        "group": "hf",
        "group_cap": 0.5,
        "group_penalty": 1.0,
        "w_min": 0.0,
        "w_max": 0.4,
    }

    def __init__(
        self,
        model_id: str = "fergusq/finbert-finnsentiment",  # Uses safetensors (torch 2.6 not required)
        revision: Optional[str] = None,
        *,
        max_length: int = 1024,
        stride: int = 256,
        batch_size: int = 8,
        max_chunks: int = 48,
        horizon_decay: Sequence[float] = (0.5, 0.25, 0.15, 0.1),
        lookback_quarters: int = 12,
        cutoff: str = "21:00:00",
        tz: str = "America/New_York",
        analyzer_fallback: bool = False,  # DISABLED: Use only real DefeatBeta/EODHD transcripts
    ) -> None:
        model_lower = model_id.lower()
        if "longformer" in model_lower:
            max_length = int(min(2048, max(1024, max_length)))
            stride = int(min(max_length // 2, max(256, stride)))
            batch_size = int(max(1, min(batch_size, 4)))
        elif "finbert" in model_lower or "bert" in model_lower:
            # BERT-family models have max_position_embeddings=512.
            max_length = int(min(512, max(128, max_length)))
            batch_size = int(max(8, min(batch_size, 16)))
            stride = int(max(128, min(stride, max_length // 2)))

        self.brain = HFBrain(
            HFSpec(
                model_id=model_id,
                revision=revision,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
            )
        )
        self.max_chunks = int(max(1, max_chunks))
        self.horizon_decay = self._normalise_decay(horizon_decay)
        self.conf_decay = self._confidence_decay(self.horizon_decay)
        self.lookback_quarters = int(max(1, lookback_quarters))
        self.cutoff = self._parse_cutoff(cutoff)
        self.tz = tz
        self._analyzer = EarningsTranscriptAnalyzer() if (analyzer_fallback and EarningsTranscriptAnalyzer) else None

    @staticmethod
    def _parse_cutoff(value: str) -> dt_time:
        parts = [int(part) for part in value.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return dt_time(parts[0], parts[1], parts[2])

    @staticmethod
    def _normalise_decay(weights: Sequence[float]) -> np.ndarray:
        arr = np.asarray([w for w in weights if w is not None], dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            arr = np.asarray([1.0], dtype=float)
        total = float(arr.sum())
        if total <= 0:
            arr = np.asarray([1.0], dtype=float)
            total = 1.0
        return arr / total

    @staticmethod
    def _confidence_decay(weights: np.ndarray) -> np.ndarray:
        if weights.size == 0:
            return np.asarray([1.0])
        base = weights / weights.max()
        return np.clip(base, 0.25, 1.0)

    def _load_transcripts(self, symbol: str) -> pd.DataFrame:
        cached = _load_cached_transcripts(symbol, max_records=self.lookback_quarters * 3)
        cached_frame = _prepare_transcript_frame(cached)

        if self._analyzer is not None:
            existing_quarters = cached_frame["quarter"].dropna().tolist()
            fallback = _fallback_transcripts(
                symbol,
                limit=self.lookback_quarters,
                analyzer=self._analyzer,
                existing_quarters=existing_quarters,
            )
            if fallback:
                fallback_frame = _prepare_transcript_frame(fallback)
                cached_frame = pd.concat([cached_frame, fallback_frame], ignore_index=True)
                cached_frame = cached_frame.sort_values("timestamp")
                cached_frame = cached_frame.drop_duplicates(subset=["quarter", "timestamp"], keep="last")

        if cached_frame.empty:
            return cached_frame

        trimmed = cached_frame.tail(self.lookback_quarters).copy()
        trimmed["timestamp"] = pd.to_datetime(trimmed["timestamp"], utc=True)
        return trimmed.reset_index(drop=True)

    def _chunk_text(self, text: str) -> List[List[int]]:
        tokens = self.brain.tok.encode(text, add_special_tokens=False, truncation=False)
        if not tokens:
            return []
        requested_max_len = int(max(32, self.brain.spec.max_length))
        tokenizer_limit = getattr(self.brain.tok, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
            max_len = int(min(requested_max_len, tokenizer_limit))
        else:
            max_len = requested_max_len
        stride_requested = int(max(0, self.brain.spec.stride))
        if max_len >= 1024:
            stride_floor = max_len // 2
            stride = int(max(256, min(stride_requested, min(512, stride_floor))))
        else:
            stride = int(max(64, min(stride_requested or max_len // 4, max_len - 1)))
        stride = min(stride, max_len - 1) if max_len > 1 else 0
        step = max_len - stride if stride < max_len else max_len
        chunks: List[List[int]] = []
        for start in range(0, len(tokens), step):
            end = start + max_len
            token_slice = tokens[start:end]
            if not token_slice:
                continue
            chunks.append([int(token) for token in token_slice])
            if len(chunks) >= self.max_chunks or end >= len(tokens):
                break
        return chunks

    def _score_transcript(
        self,
        text: str,
        *,
        symbol: str,
        timestamp: Optional[pd.Timestamp],
        quarter: Optional[str],
    ) -> Dict[str, float]:
        """Score transcript with sentiment split, uncertainty, and risk dimensions.
        
        Returns:
            Dictionary with keys:
            - score: Overall sentiment (-1 to 1)
            - conf: Overall confidence (0 to 1)
            - score_prepared: Sentiment in prepared remarks section
            - score_qa: Sentiment in Q&A section
            - uncertainty_score: Hedging/uncertain language intensity (0 to 1)
            - risk_score: Downside/risk language intensity (0 to 1)
        """
        text_clean = (text or "").strip()
        if not text_clean:
            return {
                "score": 0.0, "conf": 0.0, "score_prepared": 0.0, 
                "score_qa": 0.0, "uncertainty_score": 0.0, "risk_score": 0.0
            }

        # Split into prepared vs Q&A sections
        prepared_text, qa_text = _split_transcript_sections(text_clean)
        
        text_hash = hashlib.sha1(text_clean.encode("utf-8")).hexdigest()
        doc_key = _document_cache_key(symbol, timestamp, quarter, text_hash)
        cache_path = _inference_cache_path(self.brain.spec, symbol, doc_key)
        cached_payload = _load_inference_payload(cache_path)

        # Try to load from cache
        probabilities: List[float] = []
        prepared_probs: List[float] = []
        qa_probs: List[float] = []

        if cached_payload:
            cache_valid = (
                cached_payload.get("text_hash") == text_hash
                and cached_payload.get("model_id") == self.brain.spec.model_id
                and cached_payload.get("revision") == self.brain.spec.revision
                and int(cached_payload.get("max_length", self.brain.spec.max_length)) == self.brain.spec.max_length
                and int(cached_payload.get("stride", self.brain.spec.stride)) == self.brain.spec.stride
                and int(cached_payload.get("max_chunks", self.max_chunks)) == self.max_chunks
            )
            if cache_valid:
                token_chunks = _coerce_token_chunks(cached_payload.get("token_ids"), max_chunks=self.max_chunks)
                probabilities = _coerce_probabilities(cached_payload.get("probabilities"), limit=len(token_chunks))
                prepared_probs = _coerce_probabilities(cached_payload.get("prepared_probs", []), limit=self.max_chunks)
                qa_probs = _coerce_probabilities(cached_payload.get("qa_probs", []), limit=self.max_chunks)
                
                if not probabilities and token_chunks:
                    probabilities = self.brain.infer_probs_from_token_ids(token_chunks)
                    cached_payload = dict(cached_payload)
                    cached_payload["probabilities"] = probabilities
                    cached_payload["updated"] = datetime.now(timezone.utc).isoformat()
                    _store_inference_payload(cache_path, cached_payload)

        # Compute fresh if cache miss
        if not probabilities:
            token_chunks = self._chunk_text(text_clean)
            if not token_chunks:
                return {
                    "score": 0.0, "conf": 0.0, "score_prepared": 0.0,
                    "score_qa": 0.0, "uncertainty_score": 0.0, "risk_score": 0.0
                }
            probabilities = self.brain.infer_probs_from_token_ids(token_chunks)
            if not probabilities:
                return {
                    "score": 0.0, "conf": 0.0, "score_prepared": 0.0,
                    "score_qa": 0.0, "uncertainty_score": 0.0, "risk_score": 0.0
                }
            
            # Score prepared and Q&A sections separately
            if prepared_text:
                prepared_chunks = self._chunk_text(prepared_text)
                if prepared_chunks:
                    prepared_probs = self.brain.infer_probs_from_token_ids(prepared_chunks)
            
            if qa_text:
                qa_chunks = self._chunk_text(qa_text)
                if qa_chunks:
                    qa_probs = self.brain.infer_probs_from_token_ids(qa_chunks)
            
            # Cache everything
            payload = {
                "model_id": self.brain.spec.model_id,
                "revision": self.brain.spec.revision,
                "symbol": symbol.upper(),
                "document_id": doc_key,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else None,
                "quarter": quarter,
                "text_hash": text_hash,
                "max_length": self.brain.spec.max_length,
                "stride": self.brain.spec.stride,
                "max_chunks": self.max_chunks,
                "token_ids": token_chunks,
                "probabilities": probabilities,
                "prepared_probs": prepared_probs,
                "qa_probs": qa_probs,
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            _store_inference_payload(cache_path, payload)

        # Compute overall score and confidence
        probs = np.asarray(probabilities, dtype=float)
        mean_prob = float(np.mean(probs))
        score = float(np.clip(2.0 * mean_prob - 1.0, -1.0, 1.0))
        intensity = float(np.clip(np.abs(mean_prob - 0.5) * 2.0, 0.0, 1.0))
        dispersion = float(np.std(probs))
        dispersion_penalty = float(np.clip(1.0 - dispersion * 3.0, 0.3, 1.0))
        volume = float(np.clip(np.log1p(len(probs)) / np.log(10.0), 0.2, 1.0))
        confidence = float(np.clip(intensity * volume * dispersion_penalty, 0.0, 1.0))
        
        # Prepared remarks score
        score_prepared = 0.0
        if prepared_probs:
            prep_probs = np.asarray(prepared_probs, dtype=float)
            prep_mean = float(np.mean(prep_probs))
            score_prepared = float(np.clip(2.0 * prep_mean - 1.0, -1.0, 1.0))
        
        # Q&A score
        score_qa = 0.0
        if qa_probs:
            qa_probs_arr = np.asarray(qa_probs, dtype=float)
            qa_mean = float(np.mean(qa_probs_arr))
            score_qa = float(np.clip(2.0 * qa_mean - 1.0, -1.0, 1.0))
        
        # Uncertainty score: measure dispersion and middle-ground probabilities
        # High uncertainty = more chunks near 0.5, higher dispersion
        uncertainty_score = 0.0
        if len(probs) > 0:
            # How many chunks are in the uncertain zone (0.3-0.7)?
            uncertain_mask = (probs > 0.3) & (probs < 0.7)
            uncertain_pct = float(np.mean(uncertain_mask))
            # Combine with dispersion
            uncertainty_score = float(np.clip((dispersion * 2.0 + uncertain_pct) / 2.0, 0.0, 1.0))
        
        # Risk score: measure negative sentiment intensity and keywords
        # Higher negative scores = higher risk
        risk_score = 0.0
        if len(probs) > 0:
            # Count chunks with strong negative sentiment (prob < 0.3)
            negative_mask = probs < 0.3
            negative_pct = float(np.mean(negative_mask))
            # Measure how negative the negative chunks are
            if negative_pct > 0:
                negative_chunks = probs[negative_mask]
                negative_intensity = float(1.0 - np.mean(negative_chunks))  # Lower prob = higher risk
                risk_score = float(np.clip((negative_pct + negative_intensity) / 2.0, 0.0, 1.0))
            
            # Additional keyword-based risk scoring
            text_lower = text_clean.lower()
            risk_keywords = [
                'risk', 'concern', 'uncertainty', 'challenge', 'difficult', 'pressure',
                'headwind', 'decline', 'weakness', 'cautious', 'warning', 'threat',
                'downturn', 'adverse', 'negative impact', 'volatility'
            ]
            risk_count = sum(text_lower.count(kw) for kw in risk_keywords)
            risk_density = float(np.clip(risk_count / max(1, len(text_clean.split()) / 100), 0.0, 1.0))
            risk_score = float(np.clip((risk_score * 0.7 + risk_density * 0.3), 0.0, 1.0))
        
        return {
            "score": score,
            "conf": confidence,
            "score_prepared": score_prepared,
            "score_qa": score_qa,
            "uncertainty_score": uncertainty_score,
            "risk_score": risk_score,
        }

    def _aggregate_daily(self, transcripts: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
        """Aggregate transcript scores into daily time series with QoQ/YoY changes + interaction features."""
        if transcripts.empty:
            return pd.DataFrame(columns=[
                "score", "conf", "score_prepared", "score_qa",
                "uncertainty_score", "risk_score", "score_delta_qoq", "score_delta_yoy",
                "sentiment_divergence", "sentiment_shock"
            ])

        contributions: Dict[pd.Timestamp, Dict[str, float]] = {}
        quarterly_scores: Dict[str, float] = {}  # quarter -> score for QoQ/YoY
        decay = self.horizon_decay
        conf_decay = self.conf_decay

        for row in transcripts.itertuples(index=False):
            text = getattr(row, "text", "")
            timestamp = getattr(row, "timestamp", None)
            quarter = getattr(row, "quarter", None)
            
            if not isinstance(timestamp, pd.Timestamp):
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            
            result = self._score_transcript(
                text,
                symbol=getattr(row, "symbol", symbol),
                timestamp=timestamp,
                quarter=quarter,
            )
            
            score = result["score"]
            confidence = result["conf"]
            if confidence <= 0.0:
                continue
            
            # Store quarterly score for delta calculations
            if quarter:
                quarterly_scores[quarter] = score

            # Aggregate across decay horizons
            for idx, weight in enumerate(decay):
                offset = BDay(idx)
                session_ts = timestamp + offset
                session_date = to_market_session(session_ts, cutoff=self.cutoff, tz=self.tz)
                if session_date is None or pd.isna(session_date):
                    continue
                session_date = pd.Timestamp(session_date).normalize()
                bucket = contributions.setdefault(session_date, {
                    "score": 0.0, "weight": 0.0, "conf": 0.0,
                    "score_prepared": 0.0, "score_qa": 0.0,
                    "uncertainty_score": 0.0, "risk_score": 0.0,
                    "quarter": quarter,  # Track quarter for deltas
                })
                
                decay_factor = conf_decay[min(idx, len(conf_decay) - 1)]
                bucket["score"] += score * float(weight)
                bucket["weight"] += float(weight)
                bucket["conf"] += confidence * float(weight) * float(decay_factor)
                bucket["score_prepared"] += result["score_prepared"] * float(weight)
                bucket["score_qa"] += result["score_qa"] * float(weight)
                bucket["uncertainty_score"] += result["uncertainty_score"] * float(weight)
                bucket["risk_score"] += result["risk_score"] * float(weight)

        rows: List[Tuple[pd.Timestamp, float, float, float, float, float, float, float, float, float, float]] = []
        for date, payload in sorted(contributions.items()):
            total_weight = float(payload.get("weight", 0.0))
            if total_weight <= 0.0:
                continue
            
            mean_score = float(np.clip(payload["score"] / total_weight, -1.0, 1.0))
            mean_conf = float(np.clip(payload["conf"] / total_weight, 0.0, 1.0))
            mean_prepared = float(np.clip(payload["score_prepared"] / total_weight, -1.0, 1.0))
            mean_qa = float(np.clip(payload["score_qa"] / total_weight, -1.0, 1.0))
            mean_uncertainty = float(np.clip(payload["uncertainty_score"] / total_weight, 0.0, 1.0))
            mean_risk = float(np.clip(payload["risk_score"] / total_weight, 0.0, 1.0))
            
            # Calculate QoQ and YoY deltas
            score_delta_qoq = 0.0
            score_delta_yoy = 0.0
            quarter = payload.get("quarter")
            
            if quarter and quarter in quarterly_scores:
                current_score = quarterly_scores[quarter]
                
                # QoQ: Compare to previous quarter
                try:
                    period = pd.Period(quarter, freq="Q")
                    prev_quarter = str(period - 1)
                    if prev_quarter in quarterly_scores:
                        score_delta_qoq = current_score - quarterly_scores[prev_quarter]
                except Exception:
                    pass
                
                # YoY: Compare to same quarter last year
                try:
                    period = pd.Period(quarter, freq="Q")
                    prev_year_quarter = str(period - 4)
                    if prev_year_quarter in quarterly_scores:
                        score_delta_yoy = current_score - quarterly_scores[prev_year_quarter]
                except Exception:
                    pass
            
            # HEDGE-FUND: Add interaction features
            # 9. sentiment_divergence: Management defensiveness (prepared - Q&A)
            sentiment_divergence = mean_prepared - mean_qa
            
            # 10. sentiment_shock: Confidence-weighted change (distinguishes noise from real change)
            sentiment_shock = score_delta_qoq * mean_conf
            
            rows.append((
                date, mean_score, mean_conf, mean_prepared, mean_qa,
                mean_uncertainty, mean_risk, score_delta_qoq, score_delta_yoy,
                sentiment_divergence, sentiment_shock
            ))

        if not rows:
            return pd.DataFrame(columns=[
                "score", "conf", "score_prepared", "score_qa",
                "uncertainty_score", "risk_score", "score_delta_qoq", "score_delta_yoy",
                "sentiment_divergence", "sentiment_shock"
            ])

        daily = pd.DataFrame(rows, columns=[
            "date", "score", "conf", "score_prepared", "score_qa",
            "uncertainty_score", "risk_score", "score_delta_qoq", "score_delta_yoy",
            "sentiment_divergence", "sentiment_shock"
        ]).set_index("date").sort_index()
        daily.index = pd.to_datetime(daily.index).tz_localize(None)
        return daily

    def emit_signal(self, *, symbol: str, horizon: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> ModuleSignal:
        transcripts = self._load_transcripts(symbol)
        daily = self._aggregate_daily(transcripts, symbol=symbol)
        
        # Filter to date range if provided
        if start_date and end_date and not daily.empty:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            daily = daily[(daily.index >= start_ts) & (daily.index <= end_ts)]

        if not daily.empty:
            daily = daily.copy()
            daily["has_data"] = 1.0
        
        return ModuleSignal(name=self.NAME, horizon=int(horizon), df=daily, symbol=symbol)


__all__ = ["EarningsTranscriptHF"]
