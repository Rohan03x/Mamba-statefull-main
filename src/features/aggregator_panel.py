#!/usr/bin/env python3
"""
Lightweight aggregator panel builder

Combines selected feature families into a single daily feature panel aligned to
NYSE close index. Designed for walk-forward backtests where time-series features
are required rather than single latest values.

Supported families (best-effort, resilient to missing modules):
- microstructure: intraday-derived daily microstructure features
- cross_asset: cross-asset proxies (e.g., DXY via UUP, VXMT via VIXM)
- macro_tst_hf: unified macro super-family (EODHD indices + market features + HF transformer)
- cboe_term: VIX term structure proxies
- garch_iv: GARCH/Skew proxies per symbol
- dividends: Tiingo dividend yield and payout ratio series aligned to prices
- tech_micro_hf: HF module fed by ml_framework + microstructure + correlation
- forecast_hf: HF module blending quantile/calibration/online/tft/arima forecasts
- vol_deriv_hf: HF module combining garch_iv/cboe/options/short interest volatility flows
- macro_regime_hf: HF module combining macro/regime/cross-asset state features
- fundamental_val_hf: HF module blending fundamental families + earnings/dividends/DCFs
- news_nlp_hf: HF module for FinBERT/doc-novelty/earnings-transcript/news-derived signals

Returns a DataFrame with merged columns and attrs:
- attrs['provenance']: per-family provenance dict if available
- attrs['telemetry']: per-family telemetry when provided (e.g., AV limiter)
- attrs['feature_counts']: per-family column counts + total
- attrs['families']: list of families included
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import logging
import json
import re
import os
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.family_spec import (
    SYMBOL_ONLY_HF_BLOCKS,
    default_hf_blocks,
    families_for_stage,
    resolve_core_columns,
)
from src.features.feature_roles import load_family_meta_from_registry

HF_BLOCK_FAMILIES: Tuple[str, ...] = tuple(default_hf_blocks())
MIN_HF_AGG_BLOCKS = max(3, len(HF_BLOCK_FAMILIES) - 2) if HF_BLOCK_FAMILIES else 0

try:  # Optional heavy dependency
    import torch
except Exception:  # pragma: no cover - torch optional in some environments
    torch = None  # type: ignore

# Load environment variables from .env file for API tokens
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

try:
    from .microstructure_intraday import fetch as fetch_micro
except Exception:
    fetch_micro = None  # type: ignore
try:
    from .cross_asset import fetch as fetch_cross
except Exception:
    fetch_cross = None  # type: ignore
try:
    from .cboe_term import fetch as fetch_cboe
except Exception:
    fetch_cboe = None  # type: ignore
try:
    from .garch_iv import fetch as fetch_garch
except Exception:
    fetch_garch = None  # type: ignore
try:
    from .correlation import fetch as fetch_corr
except Exception:
    fetch_corr = None  # type: ignore
try:
    from .finbert import fetch as fetch_finbert
except Exception:
    fetch_finbert = None  # type: ignore
try:
    from .crypto import fetch as fetch_crypto
except Exception:
    fetch_crypto = None  # type: ignore
try:
    from .fx import fetch as fetch_fx
except Exception:
    fetch_fx = None  # type: ignore
try:
    from .commodities import fetch as fetch_commodities
except Exception:
    fetch_commodities = None  # type: ignore

try:
    from .candle_mechanics import fetch as fetch_candle_mechanics
except Exception:
    fetch_candle_mechanics = None  # type: ignore

try:
    from .event_time_bars import (
        generate_event_time_features_range,
        EventBarConfig,
    )
except Exception:
    generate_event_time_features_range = None  # type: ignore
    EventBarConfig = None  # type: ignore

try:
    from .index_constituents import fetch as fetch_index_constituents
except Exception:
    fetch_index_constituents = None  # type: ignore

try:
    from .insider_form4 import fetch as fetch_insider_form4
except Exception:
    fetch_insider_form4 = None  # type: ignore

try:
    from .econ_events_calendar import fetch as fetch_econ_events_calendar
except Exception:
    fetch_econ_events_calendar = None  # type: ignore

try:
    from .corp_actions_splits import fetch as fetch_corp_actions_splits
except Exception:
    fetch_corp_actions_splits = None  # type: ignore

try:
    from .marketcap_history import fetch as fetch_marketcap_history
except Exception:
    fetch_marketcap_history = None  # type: ignore

try:
    from .exchange_calendar import fetch as fetch_exchange_calendar
except Exception:
    fetch_exchange_calendar = None  # type: ignore

try:
    from .peer_screener_context import fetch_peer_screener_context
except Exception:
    fetch_peer_screener_context = None  # type: ignore

try:
    from .narrative_novelty import fetch as fetch_narrative_novelty
except Exception:
    fetch_narrative_novelty = None  # type: ignore

try:
    # Prefer local import path first
    from src.dcf_lab.utils.timealign import to_nyse_close_index  # type: ignore
except Exception:
    def to_nyse_close_index(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        return df

# macro_enhanced removed - consolidated into macro_tst_hf unified family

try:
    from src.dcf_lab.providers.options_analysis import OptionsAnalysisProvider  # type: ignore
except Exception:
    OptionsAnalysisProvider = None  # type: ignore

try:
    from src.dcf_lab.short_interest_analyzer import ShortInterestAnalyzer  # type: ignore
except Exception:
    ShortInterestAnalyzer = None  # type: ignore

try:
    from src.dcf_lab.subsidiary_mapping import SubsidiaryMapper  # type: ignore
except Exception:
    SubsidiaryMapper = None  # type: ignore

try:
    from src.dcf_lab.earnings_transcript_analyzer import EarningsTranscriptAnalyzer  # type: ignore
except Exception:
    EarningsTranscriptAnalyzer = None  # type: ignore

try:
    from src.dcf_lab.providers.alternative_signals import get_alternative_signals_provider  # type: ignore
except Exception:
    get_alternative_signals_provider = None  # type: ignore

try:
    from src.dcf_lab.probability_calibration import ProbabilityCalibrationSystem  # type: ignore
except Exception:
    ProbabilityCalibrationSystem = None  # type: ignore

try:
    from src.dcf_lab.online_learning import create_online_learning_system  # type: ignore
except Exception:
    create_online_learning_system = None  # type: ignore

# News sentiment imports
try:
    from src.dcf_lab.valuation.news_sentiment import FinbertSentimentAnalyzer, NewsAndEarningsForecaster  # type: ignore
except Exception:
    FinbertSentimentAnalyzer = None  # type: ignore
    NewsAndEarningsForecaster = None  # type: ignore

# ML framework features are sourced from EODHD (no local derivation).
FeatureEngineer = None  # type: ignore
MLConfig = None  # type: ignore

try:
    from src.dcf_lab.modules.hf_time_series_module import (  # type: ignore
        HFTimeSeriesConfig,
        HFTimeSeriesModule,
        clip_gradients,
        configure_optimizer,
        configure_scheduler,
    )
except Exception:
    HFTimeSeriesConfig = None  # type: ignore
    HFTimeSeriesModule = None  # type: ignore
    clip_gradients = None  # type: ignore
    configure_optimizer = None  # type: ignore
    configure_scheduler = None  # type: ignore

try:
    from universal_data_fetcher import get_universal_fetcher  # type: ignore
except Exception:
    get_universal_fetcher = None  # type: ignore

# Production and Drift monitoring imports
try:
    from src.dcf_lab.production.model_updater import LightModelUpdater  # type: ignore
    from src.dcf_lab.drift.monitors import FeatureDriftMonitor  # type: ignore
except Exception:
    LightModelUpdater = None  # type: ignore
    FeatureDriftMonitor = None  # type: ignore

# Forecast models imports
try:
    from src.dcf_lab.valuation.forecast_models import ForecastConfig, QuantileForecaster, GBMQuantileForecaster, TFTForecaster  # type: ignore
except Exception:
    ForecastConfig = None  # type: ignore
    QuantileForecaster = None  # type: ignore
    GBMQuantileForecaster = None  # type: ignore
    TFTForecaster = None  # type: ignore

# ARIMA forecast imports
try:
    from src.dcf_lab.price_forecast import arima_forecast_close  # type: ignore
except Exception:
    arima_forecast_close = None  # type: ignore

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
except Exception:
    SARIMAX = None  # type: ignore

# AutoARIMA for automatic order selection with hedge-fund safe constraints
try:
    from pmdarima import auto_arima  # type: ignore
    AUTO_ARIMA_AVAILABLE = True
except Exception:
    AUTO_ARIMA_AVAILABLE = False
    auto_arima = None  # type: ignore

# Multi-asset and regime detection imports
try:
    from src.dcf_lab.multiasset_correlation import MultiAssetForecaster  # type: ignore
    from src.dcf_lab.regime_detection import RegimeAdaptiveForecaster  # type: ignore
    from src.dcf_lab.options_anchoring import OptionsAnchoredForecaster  # type: ignore
except Exception:
    MultiAssetForecaster = None  # type: ignore
    RegimeAdaptiveForecaster = None  # type: ignore
    OptionsAnchoredForecaster = None  # type: ignore

# Enhanced News Sentiment
try:
    from src.dcf_lab.news_sentiment import FinBERTSentimentAnalyzer, NewsSentimentForecaster  # type: ignore
except Exception:
    FinBERTSentimentAnalyzer = None  # type: ignore
    NewsSentimentForecaster = None  # type: ignore


logger = logging.getLogger(__name__)


class DisallowedDataSourceError(RuntimeError):
    """Raised when a feature family attempts to use a disallowed data source."""


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _no_proxy_sources_enabled() -> bool:
    """Disallow proxy-derived *features*.

    Note: This does NOT disallow legitimate upstream data sources (FRED/yfinance/Tiingo/etc).
    It only enforces that families must not label their output as proxy.
    """
    return _env_flag("STAGE_B_NO_PROXY_SOURCES", "0")


def _eodhd_only_enabled() -> bool:
    """Disallow any non-EODHD data sources (stronger than no-proxy)."""
    return _env_flag("STAGE_B_EODHD_ONLY", "0")


def _strict_sources_enabled() -> bool:
    return _no_proxy_sources_enabled() or _eodhd_only_enabled()


def _require_all_families_enabled() -> bool:
    return _env_flag("STAGE_B_REQUIRE_ALL_FAMILIES", "0")


def _no_live_fallback_enabled() -> bool:
    """When enabled, build_panel will NOT call live family handlers on cache miss.

    This is useful for enforcing cache-only operation (e.g., during Stage B
    walk-forward runs or when auditing/rebuilding unified panels) to avoid
    triggering expensive external fetches.
    """
    return _env_flag("STAGE_B_NO_LIVE_FALLBACK", "0")


def _seed_from_symbol(symbol: str) -> int:
    return abs(hash(str(symbol))) % (2 ** 32)


def _make_index(start: Optional[str], end: Optional[str]) -> pd.DatetimeIndex:
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.utcnow().normalize()
    start_dt = pd.to_datetime(start) if start else end_dt - pd.Timedelta(days=30)
    if start_dt > end_dt:
        start_dt, end_dt = end_dt - pd.Timedelta(days=1), end_dt
    try:
        from src.dcf_lab.utils.timealign import nyse_sessions_in_range  # type: ignore

        sessions = nyse_sessions_in_range(start_dt, end_dt, tz_aware_utc=False)
        if len(sessions) == 0:
            return pd.DatetimeIndex([end_dt])
        return sessions
    except Exception:
        idx = pd.date_range(start=start_dt, end=end_dt, freq="B")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([end_dt])
        return idx


def _resolve_period_token(days: int) -> str:
    thresholds = [
        (1, "1d"),
        (5, "5d"),
        (30, "1mo"),
        (90, "3mo"),
        (180, "6mo"),
        (365, "1y"),
        (730, "2y"),
        (1825, "5y"),
        (3650, "10y"),
    ]
    for limit, token in thresholds:
        if days <= limit:
            return token
    return "max"


def _coerce_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _numeric_dict(data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    if not isinstance(data, dict):
        return numeric
    for key, value in data.items():
        num = _coerce_numeric(value)
        if num is not None and np.isfinite(num):
            numeric[str(key)] = num
    return numeric


def _flatten_numeric(data: Any, prefix: str = "", max_list: int = 5) -> Dict[str, float]:
    flattened: Dict[str, float] = {}

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key).strip()
                new_path = f"{path}_{key_str}" if path else key_str
                _walk(value, new_path)
        elif isinstance(obj, (list, tuple)):
            if not obj or len(obj) > max_list:
                return
            for idx, value in enumerate(obj):
                new_path = f"{path}_{idx}" if path else str(idx)
                _walk(value, new_path)
        else:
            numeric = _coerce_numeric(obj)
            if numeric is not None and np.isfinite(numeric) and path:
                flattened[path.replace(" ", "_").replace("-", "_")] = numeric

    _walk(data, prefix)
    return flattened


def _sanitize_feature_keys(family: str, features: Dict[str, float]) -> Dict[str, float]:
    sanitized: Dict[str, float] = {}
    family_prefix = f"{family.lower()}_"
    for key, value in features.items():
        key_str = str(key).strip()
        lower = key_str.lower()
        if lower.startswith(family_prefix):
            key_str = key_str[len(family_prefix):]
        key_str = key_str.replace(" ", "_").replace("-", "_")
        sanitized[key_str] = value
    return sanitized


def _slugify_column_token(token: str) -> str:
    """Return a lowercase, underscore-delimited column token."""

    cleaned = token.strip().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def _deduplicate_tokens(tokens: Iterable[str], suffix_template: str = "_{}") -> Tuple[List[str], int]:
    """Ensure tokens are unique by appending numeric suffixes when needed."""

    seen: Dict[str, int] = {}
    deduped: List[str] = []
    renamed = 0
    for token in tokens:
        count = seen.get(token, 0)
        if count == 0:
            deduped.append(token)
        else:
            suffix = count + 1
            candidate = f"{token}{suffix_template.format(suffix)}"
            while candidate in seen:
                suffix += 1
                candidate = f"{token}{suffix_template.format(suffix)}"
            deduped.append(candidate)
            renamed += 1
            seen[candidate] = 1
        seen[token] = count + 1
    return deduped, renamed


def _normalize_family_column_names(family: str, columns: Iterable[Any]) -> Tuple[List[str], int]:
    """Slugify column names and remove duplicate prefixes per family."""

    family_prefix = f"{family.lower()}_"
    sanitized: List[str] = []
    for raw in columns:
        token = _slugify_column_token(str(raw))
        if token.startswith(family_prefix):
            token = token[len(family_prefix):].lstrip("_")
        if not token:
            token = "value"
        sanitized.append(token)
    normalized, renamed = _deduplicate_tokens(sanitized)
    return normalized, renamed


def _feature_frame_from_dict(
    family: str,
    features: Dict[str, float],
    start: Optional[str],
    end: Optional[str],
    source: str,
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    if not features:
        return None
    sanitized = _sanitize_feature_keys(family, features)
    if not sanitized:
        return None
    index = _make_index(start, end)
    frame = pd.DataFrame([sanitized], index=[index[-1]])
    if len(index) > 1:
        # For single-point-in-time features (earnings, subsidiary structure),
        # forward fill propagates the latest value to future dates
        frame = frame.reindex(index).ffill()
    frame.attrs['telemetry'] = {'status': 'ok', 'source': source}
    if provenance:
        frame.attrs['provenance'] = provenance
    frame.attrs['feature_counts'] = {'generated': len(sanitized)}
    return frame


def _safe_filter(df: Optional[pd.DataFrame], start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df is None or isinstance(df, pd.DataFrame) and df.empty:
        return pd.DataFrame()
    # Preserve metadata; pandas may drop .attrs on boolean/slice filtering.
    preserved_attrs: Dict[str, Any] = {}
    try:
        preserved_attrs = dict(getattr(df, "attrs", {}) or {})
    except Exception:
        preserved_attrs = {}

    out = df.copy()
    try:
        if start or end:
            # Convert start/end to timestamps matching the index timezone
            start_ts = pd.to_datetime(start) if start else None
            end_ts = pd.to_datetime(end) if end else None
            
            # Handle timezone-aware index by localizing comparison timestamps
            if hasattr(out.index, 'tz') and out.index.tz is not None:
                # Index is timezone-aware, make comparison timestamps aware
                if start_ts is not None and start_ts.tz is None:
                    start_ts = start_ts.tz_localize(out.index.tz)
                if end_ts is not None and end_ts.tz is None:
                    end_ts = end_ts.tz_localize(out.index.tz)
            else:
                # Index is naive, ensure comparison timestamps are naive
                if start_ts is not None and start_ts.tz is not None:
                    start_ts = start_ts.tz_localize(None)
                if end_ts is not None and end_ts.tz is not None:
                    end_ts = end_ts.tz_localize(None)
            
            # Apply filters
            if start_ts is not None:
                out = out[out.index >= start_ts]
            if end_ts is not None:
                out = out[out.index <= end_ts]
    except Exception:
        # If filtering fails, try stripping timezone from index
        try:
            out.index = out.index.tz_localize(None) if hasattr(out.index, 'tz') and out.index.tz else out.index
            if start:
                out = out[out.index >= pd.to_datetime(start)]
            if end:
                out = out[out.index <= pd.to_datetime(end)]
        except Exception:
            pass  # Return unfiltered if both attempts fail

    if preserved_attrs:
        try:
            out.attrs.update(preserved_attrs)
        except Exception:
            pass
    return out


def _fetch_price_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Helper to fetch real price data for feature generation.
    
    Priority order:
    1. EODHD (paid API with comprehensive data) - PRIMARY SOURCE
    2. Universal data fetcher (cache + EODHD + FRED fallback)
    
    DATA SOURCE VERIFICATION:
    - Used by: Linear alpha combiner features (ret_1d, ret_5d, ret_21d, rv_21d)
    - Primary: EODHD provider (requires EODHD_API_KEY)
    - Enforcement: Set STAGE_B_EODHD_ONLY=1 to reject non-EODHD sources
    - Verification: Run verify_eodhd_data_sources.py to confirm EODHD usage
    """
    try:
        # Calculate date range with 1 year history for rolling calculations
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        history_start = start_dt - pd.Timedelta(days=365)
        
        # PRIORITY 1: Try EODHD first (paid API)
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            logger.debug(f"🔑 Attempting EODHD provider for {symbol}")
            
            eodhd = get_eodhd_provider()
            if eodhd.api_key:
                df = eodhd.get_eod_prices(
                    symbol,
                    start_date=history_start.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d')
                )
                
                if df is not None and not df.empty:
                    logger.info(
                        "✅ EODHD: Got %d rows for %s (cache=%s)",
                        len(df),
                        symbol,
                        "on" if getattr(eodhd, "_cache_enabled", False) else "off",
                    )
                    # Normalize column names to lowercase
                    df.columns = [c.lower() for c in df.columns]
                    df.index = pd.to_datetime(df.index)
                    return df
                else:
                    logger.debug(f"⚠️ EODHD: No data returned for {symbol}")
            else:
                logger.debug("⚠️ EODHD: API key not configured")
        except Exception as e:
            logger.debug(f"⚠️ EODHD: Failed for {symbol}: {e}")
        
        if _strict_sources_enabled():
            raise DisallowedDataSourceError(
                f"Strict source policy enabled; refusing non-EODHD price fallback for {symbol}. "
                "Ensure EODHD is configured and has coverage for this symbol/date range."
            )

        # PRIORITY 2: Fallback to universal data fetcher (cache + EODHD + FRED)
        logger.debug(f"📊 Falling back to universal data fetcher for {symbol}")
        from src.data.universal_data_fetcher import download
        
        df = download(symbol, start=history_start.strftime('%Y-%m-%d'), 
                     end=end_dt.strftime('%Y-%m-%d'))
        if df is None or df.empty:
            return None
        
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        
        return df
    except DisallowedDataSourceError:
        raise
    except Exception as e:
        logger.debug(f"❌ _fetch_price_data failed for {symbol}: {e}")
        return None

CACHE_VIEW_OPTIONS = {"raw", "summary", "both"}


def _resolve_family_list(
    families: Optional[Iterable[str]],
    stage: Optional[str],
) -> List[str]:
    stage_token = (stage or "B").upper()
    requested: List[str] = []
    if isinstance(families, str):
        families = [families]
    if families:
        for fam in families:
            token = str(fam).strip()
            if not token:
                continue
            if token.lower() == "all":
                requested.extend(families_for_stage(stage_token))
            else:
                requested.append(token)
    if not requested:
        requested = families_for_stage(stage_token)
    seen: Set[str] = set()
    ordered: List[str] = []
    for fam in requested:
        if fam not in seen:
            ordered.append(fam)
            seen.add(fam)
    return ordered


def _normalize_timestamp(value: Optional[Any]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    try:
        ts = ts.tz_localize(None)
    except (TypeError, AttributeError):
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(ts)


def _normalize_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Do not fill at cache-normalization time.
    # Filling here can silently propagate snapshot/event data across time.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


WRITE_LAGGED_FEATURE_CACHES: bool = os.getenv("WRITE_LAGGED_FEATURE_CACHES", "0") == "1"


HF_SYMBOL_ONLY_CANONICAL_HORIZON: int = int(os.getenv("HF_SYMBOL_ONLY_CANONICAL_HORIZON", "63"))

# Shared cache directory for symbol-agnostic families
SHARED_CACHE_DIR: Path = Path(__file__).resolve().parents[2] / "cache" / "shared"

# Families that use shared cache (symbol/horizon invariant)
SHARED_CACHE_FAMILIES: frozenset = frozenset({"doc_embedding_novelty_hf", "peer_screener_context"})


def _uses_shared_cache_path(family: str) -> bool:
    """Check if a family uses the shared cache (symbol/horizon invariant)."""
    fam_lower = str(family).strip().lower()
    return fam_lower in SHARED_CACHE_FAMILIES


def _get_shared_cache_dir(family: str) -> Optional[Path]:
    """Get the shared cache directory for a family, if applicable."""
    fam_lower = str(family).strip().lower()
    if fam_lower == "doc_embedding_novelty_hf":
        return SHARED_CACHE_DIR / "doc_embedding" / "doc_embedding_novelty_hf"
    elif fam_lower == "peer_screener_context":
        return SHARED_CACHE_DIR / "peer_screener" / "peer_screener_context"
    return None


def _uses_symbol_only_cache_path(family: str) -> bool:
    try:
        fam_lower = str(family).strip().lower()
    except Exception:
        fam_lower = str(family)
    return fam_lower in {f.lower() for f in SYMBOL_ONLY_HF_BLOCKS}


def _resolve_cache_dir_for_family(cache_dir: Path, symbol: str, horizon: int, family: str) -> Path:
    """Resolve where a family's cache files should be read from.

    cache_dir may be:
    - cache/symbols/<SYMBOL>/h<horizon> (new Dagster layout)
    - data/local_cache/<symbol>_h<horizon> (legacy)
    - data/local_cache/<symbol> (symbol-only)
    - data/local_cache (root)
    """

    symbol_lower = symbol.lower()
    symbol_upper = symbol.upper()
    fam_lower = str(family).lower()

    # NEW: Detect Dagster-style cache layout: cache/symbols/{SYMBOL}/h{horizon}
    # In this layout, files are directly in the folder (not in a subfolder).
    expected_h_folder = f"h{int(horizon)}"
    if cache_dir.name == expected_h_folder and cache_dir.parent.name == symbol_upper:
        # Dagster layout: cache/symbols/AAPL/h63 -> files are here directly
        if _uses_symbol_only_cache_path(fam_lower):
            # Symbol-only families: use parent + symbol subdir (cache/symbols/AAPL/aapl)
            return cache_dir.parent / symbol_lower
        return cache_dir

    if _uses_symbol_only_cache_path(fam_lower):
        # Prefer explicit symbol-only folder if present.
        if cache_dir.name == symbol_lower:
            return cache_dir
        # If a horizon folder was provided, use its parent.
        expected_h = f"{symbol_lower}_h{int(horizon)}"
        if cache_dir.name == expected_h:
            return cache_dir.parent / symbol_lower
        # If root was provided, join symbol.
        return cache_dir / symbol_lower

    # Non symbol-only families default to the horizon folder.
    expected_h = f"{symbol_lower}_h{int(horizon)}"
    if cache_dir.name == expected_h:
        return cache_dir
    if cache_dir.name == symbol_lower:
        return cache_dir.parent / expected_h
    return cache_dir / expected_h


def _cache_basename(symbol: str, horizon: int, family: str, split: Optional[str] = None) -> str:
    """Return cache file basename.
    
    New format (no split): aapl_h63_garch_iv.parquet
    Legacy format (with split): aapl_h63_garch_iv_train.parquet
    """
    symbol_lower = symbol.lower()
    fam_lower = str(family).lower()
    if _uses_symbol_only_cache_path(fam_lower):
        if split:
            return f"{symbol_lower}_{family}_{split}.parquet"
        return f"{symbol_lower}_{family}.parquet"
    if split:
        return f"{symbol_lower}_h{int(horizon)}_{family}_{split}.parquet"
    return f"{symbol_lower}_h{int(horizon)}_{family}.parquet"


def _load_family_cache_frame(
    symbol: str,
    family: str,
    cache_dir: Path,
    horizon: int,
    view_mode: str,
    start: Optional[str],
    end: Optional[str],
) -> Optional[pd.DataFrame]:
    symbol_lower = symbol.lower()
    
    # Check shared cache first for symbol-agnostic families (doc_embedding_novelty_hf, peer_screener_context)
    if _uses_shared_cache_path(family):
        shared_dir = _get_shared_cache_dir(family)
        if shared_dir is not None and shared_dir.exists():
            # Look for consolidated files matching pattern _*_features.parquet
            pattern = "_*_features.parquet"
            matching_files = sorted(shared_dir.glob(pattern), reverse=True)
            for file_path in matching_files:
                try:
                    df = pd.read_parquet(file_path)
                    if df.empty or 'date' not in df.columns:
                        continue
                    
                    df['date'] = pd.to_datetime(df['date'])
                    if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                        df['date'] = df['date'].dt.tz_localize(None)
                    df = df.set_index('date').sort_index()
                    
                    # Prefix unprefixed columns with family name
                    family_prefix = f"{family.lower()}_"
                    new_columns = {}
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if not col_lower.startswith(family_prefix):
                            new_columns[col] = f"{family_prefix}{col}"
                    if new_columns:
                        df = df.rename(columns=new_columns)
                    
                    logger.info(f"✅ {family}: loaded from shared cache ({len(df)} rows, {len(df.columns)} cols)")
                    return df
                except Exception as e:
                    logger.debug(f"Failed to load shared cache {file_path}: {e}")
                    continue
    
    resolved_dir = _resolve_cache_dir_for_family(cache_dir, symbol, horizon, family)
    expected_h = f"{symbol_lower}_h{int(horizon)}"
    legacy_dir = cache_dir if cache_dir.name == expected_h else (cache_dir.parent / expected_h if cache_dir.name == symbol_lower else cache_dir / expected_h)
    frames_by_split: Dict[str, pd.DataFrame] = {}
    
    # Try loading consolidated (no-split) cache first - preferred for Stage-A mode
    clean_base_name = _cache_basename(symbol, horizon, family, split=None)
    clean_feature_path = resolved_dir / clean_base_name.replace(".parquet", "_features.parquet")
    if clean_feature_path.exists():
        try:
            clean_df = _normalize_cache_frame(pd.read_parquet(clean_feature_path))
            # Use consolidated file as single unified dataset (not duplicated as train/valid)
            frames_by_split["consolidated"] = clean_df
            logger.debug("Loaded consolidated (Stage-A) cache from %s", clean_feature_path.name)
        except Exception as exc:
            logger.debug("Failed to read consolidated cache %s: %s", clean_feature_path.name, exc)
    
    # Fallback to split-specific caches if consolidated cache not found (Stage-B mode)
    if not frames_by_split:
        splits = ["train", "valid"]
        for split in splits:
            split_frames: List[pd.DataFrame] = []
            base_name = _cache_basename(symbol, horizon, family, split)
            if view_mode in {"raw", "both"}:
                lagged_path = resolved_dir / base_name.replace(".parquet", "_features_lagged.parquet")
                feature_path = resolved_dir / base_name.replace(".parquet", "_features.parquet")
                # Back-compat: legacy horizon-scoped cache locations for symbol-only families.
                if _uses_symbol_only_cache_path(family):
                    legacy_base = f"{symbol_lower}_h{int(horizon)}_{family}_{split}.parquet"
                    legacy_lagged = legacy_dir / legacy_base.replace(".parquet", "_features_lagged.parquet")
                    legacy_feature = legacy_dir / legacy_base.replace(".parquet", "_features.parquet")
                else:
                    legacy_lagged = lagged_path
                    legacy_feature = feature_path
                # Default to non-lagged caches; optionally prefer lagged if explicitly enabled.
                raw_path = feature_path
                if WRITE_LAGGED_FEATURE_CACHES and lagged_path.exists():
                    raw_path = lagged_path
                elif WRITE_LAGGED_FEATURE_CACHES and (not lagged_path.exists()) and legacy_lagged.exists():
                    raw_path = legacy_lagged
                elif (not feature_path.exists()) and legacy_feature.exists():
                    raw_path = legacy_feature
                if raw_path.exists():
                    try:
                        split_frames.append(_normalize_cache_frame(pd.read_parquet(raw_path)))
                        logger.debug("Loaded raw cache from %s", raw_path.name)
                    except Exception as exc:
                        logger.debug("Failed to read raw cache %s: %s", raw_path.name, exc)
            if view_mode in {"summary", "both"}:
                signal_path = resolved_dir / base_name
                if _uses_symbol_only_cache_path(family) and (not signal_path.exists()):
                    legacy_base = f"{symbol_lower}_h{int(horizon)}_{family}_{split}.parquet"
                    legacy_signal = legacy_dir / legacy_base
                else:
                    legacy_signal = signal_path
                if signal_path.exists():
                    try:
                        split_frames.append(_normalize_cache_frame(pd.read_parquet(signal_path)))
                    except Exception as exc:
                        logger.debug("Failed to read summary cache %s: %s", signal_path.name, exc)
                elif legacy_signal is not signal_path and legacy_signal.exists():
                    try:
                        split_frames.append(_normalize_cache_frame(pd.read_parquet(legacy_signal)))
                    except Exception as exc:
                        logger.debug("Failed to read legacy summary cache %s: %s", legacy_signal.name, exc)
            if not split_frames:
                continue
            merged = split_frames[0]
            for block in split_frames[1:]:
                # Remove columns that already exist in merged to avoid overlap errors
                overlap = merged.columns.intersection(block.columns)
                if len(overlap) > 0:
                    logger.debug("Dropping %d overlapping columns from %s: %s", len(overlap), family, list(overlap)[:5])
                    block = block.drop(columns=overlap)
                if not block.empty and len(block.columns) > 0:
                    merged = merged.join(block, how="outer")

            # Remove legacy placeholder columns from macro_tst_hf caches.
            # These were previously emitted as generic hf_* columns (not family-prefixed)
            # and should not appear in merged panels.
            if str(family).lower() == "macro_tst_hf":
                try:
                    drop_cols = [
                        c
                        for c in merged.columns
                        if str(c).startswith("hf_embed_")
                        or str(c) in {"hf_macro_score", "hf_confidence", "hf_regime", "hf_volatility"}
                    ]
                    if drop_cols:
                        merged = merged.drop(columns=drop_cols, errors="ignore")
                except Exception:
                    pass

            # Some cache writers (notably HF generators) may emit a generic 'has_data' column.
            # Rename it to a family-scoped flag to avoid collisions with other families.
            try:
                if "has_data" in merged.columns and f"{family}_has_data" not in merged.columns:
                    merged = merged.rename(columns={"has_data": f"{family}_has_data"})
            except Exception:
                pass
            frames_by_split[split] = merged
    if not frames_by_split:
        return None

    # Handle three cases:
    # 1. Stage-A mode: "consolidated" key contains single unified dataset
    # 2. Stage-B mode: "train" and "valid" keys contain split datasets
    # 3. Fallback: concatenate whatever splits are available
    df_consolidated = frames_by_split.get("consolidated")
    if df_consolidated is not None:
        # Stage-A mode: use consolidated dataset directly
        combined = df_consolidated.sort_index()
    else:
        # Stage-B mode: prefer train values on overlapping dates
        df_train = frames_by_split.get("train")
        df_valid = frames_by_split.get("valid")
        if df_train is not None and df_valid is not None:
            union_index = df_train.index.union(df_valid.index)
            df_train = df_train.reindex(union_index)
            df_valid = df_valid.reindex(union_index)
            combined = df_train.combine_first(df_valid).sort_index()
        else:
            combined = pd.concat(list(frames_by_split.values()), axis=0)
            combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    start_ts = _normalize_timestamp(start) or combined.index.min()
    end_ts = _normalize_timestamp(end) or combined.index.max()
    combined = combined[(combined.index >= start_ts) & (combined.index <= end_ts)]
    return combined


def _build_panel_from_cache(
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    families: List[str],
    cache_dir: Optional[Path],
    horizon: int,
    view_mode: str,
) -> Optional[pd.DataFrame]:
    if cache_dir is None:
        return None
    frames: List[pd.DataFrame] = []
    feature_counts: Dict[str, int] = {}
    telemetry: Dict[str, Dict[str, Any]] = {}
    cache_root = Path(cache_dir)
    missing_families: List[str] = []
    for family in families:
        fam_frame = _load_family_cache_frame(
            symbol=symbol,
            family=family,
            cache_dir=cache_root,
            horizon=horizon,
            view_mode=view_mode,
            start=start,
            end=end,
        )
        if fam_frame is None or fam_frame.empty:
            # Skip optional/missing families instead of failing entirely
            missing_families.append(family)
            logger.debug("Cache miss for family %s - will skip", family)
            continue
        frames.append(fam_frame)
        feature_counts[family] = len(fam_frame.columns)
        telemetry[family] = {
            "status": "cached",
            "source": "prep_families",
            "cache_view": view_mode,
        }
    
    if missing_families:
        logger.info("⚠️ Skipped %d families with no cache: %s", len(missing_families), missing_families)
    
    if not frames:
        logger.warning("No cached families found for %s", symbol)
        return None
    
    panel = frames[0]
    for block in frames[1:]:
        # Drop overlapping columns from block (keep panel's version)
        overlap = panel.columns.intersection(block.columns)
        if len(overlap) > 0:
            logger.debug("Dropping %d overlapping columns before join: %s", len(overlap), list(overlap)[:5])
            block = block.drop(columns=overlap)
        if not block.empty and len(block.columns) > 0:
            panel = panel.join(block, how="outer")

    # IMPORTANT: Avoid merge-stage ffill/bfill for cached panels.
    # Family cache writers are responsible for their own time alignment.
    panel = panel.sort_index().replace([np.inf, -np.inf], np.nan)

    finbert_cols = [c for c in panel.columns if c.startswith("finbert_")]
    other_cols = [c for c in panel.columns if c not in finbert_cols]
    if other_cols:
        panel[other_cols] = panel[other_cols].fillna(0)

    # If a column is entirely NaN across the requested date range, ffill/bfill cannot repair it.
    # Keeping all-NaN columns causes downstream models (and panel QA) to fail.
    all_nan_cols = [c for c in panel.columns if panel[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Dropping %d all-NaN columns from cached panel for %s (examples: %s)",
            len(all_nan_cols),
            symbol,
            all_nan_cols[:10],
        )
        panel = panel.drop(columns=all_nan_cols)
    panel.attrs["families"] = [f for f in families if f not in missing_families]
    panel.attrs["missing_families"] = missing_families
    panel.attrs["feature_counts"] = feature_counts
    panel.attrs["telemetry"] = telemetry
    panel.attrs.setdefault("provenance", {})
    return panel


def build_panel(
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    families: Optional[List[str]] = None,
    macro_lag_days: int = 1,
    finbert_gap_thresh: int = 1,
    corr_windows: Optional[List[int]] = None,
    corr_breaks: Optional[List[float]] = None,
    corr_flat_gate_n: int = 0,
    corr_flat_eps: float = 1e-6,
    corr_advanced_features: bool = True,
    corr_lag_periods: Optional[List[int]] = None,
    corr_trend_lookback: int = 5,
    corr_vol_window: int = 20,
    cache_dir: Optional[Path] = None,
    horizon: int = 63,
    stage: Optional[str] = None,
    view: str = "both",
    window_idx: Optional[int] = None,
    allow_generate_missing_from_cache: bool = False,
) -> pd.DataFrame:
    family_meta_registry = load_family_meta_from_registry()
    def _is_etf_symbol(sym: str) -> bool:
        """Best-effort ETF detector.

        Purpose: prevent forcing equity-style fundamentals onto ETFs.
        Keep intentionally explicit (no heuristics).
        """

        s = (sym or "").strip().upper()
        if not s:
            return False

        # Broad / macro / vol benchmarks used in the pipeline.
        core_etfs = {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "UUP",
            "VXX",
            "VIXY",
            "SVXY",
            # Common sector SPDRs.
            "XLB",
            "XLC",
            "XLE",
            "XLF",
            "XLI",
            "XLK",
            "XLP",
            "XLRE",
            "XLU",
            "XLV",
            "XLY",
        }
        return s in core_etfs

    def _apply_stable_schema_stubs(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure stable schema for a few global/legacy columns.

        - Standardize a global `has_data` column.
        - Add `is_etf` flag.

        Neutral fills are used ONLY for stable-schema placeholders.
        """

        if df is None or df.empty:
            return df

        out = df
        is_etf = 1.0 if _is_etf_symbol(symbol) else 0.0

        if "is_etf" not in out.columns:
            out = out.copy()
            out["is_etf"] = float(is_etf)
        else:
            try:
                out["is_etf"] = pd.to_numeric(out["is_etf"], errors="coerce").fillna(float(is_etf))
            except Exception:
                pass

        # AAPL historically emitted an unprefixed `has_data` in TrackC.
        # Standardize it across symbols to avoid schema drift.
        if "has_data" not in out.columns:
            if out is df:
                out = out.copy()
            out["has_data"] = 1.0

        # ------------------------------------------------------------------
        # Stable family-scoped has_data flags
        # ------------------------------------------------------------------
        # These are useful model inputs (distinguish "real zeros" vs "missing provider → zero-fill")
        # AND help prevent schema drift between symbols when a family emits/omits its flag.
        if "calibration_has_data" not in out.columns:
            if out is df:
                out = out.copy()
            out["calibration_has_data"] = np.nan
            try:
                # Best-effort derivation: calibration is considered present when it has
                # non-trivial sample size/confidence.
                sample_size = pd.to_numeric(out.get("calibration_sample_size"), errors="coerce")
                confidence = pd.to_numeric(out.get("calibration_confidence"), errors="coerce")
                derived = None
                if sample_size is not None and confidence is not None:
                    derived = ((sample_size.fillna(0.0) > 0.0) | (confidence.fillna(0.0) > 0.0)).astype(float)
                elif sample_size is not None:
                    derived = (sample_size.fillna(0.0) > 0.0).astype(float)
                elif confidence is not None:
                    derived = (confidence.fillna(0.0) > 0.0).astype(float)
                if derived is not None:
                    out["calibration_has_data"] = out["calibration_has_data"].fillna(derived)
            except Exception:
                pass

        # doc_embedding_novelty_hf is a global (symbol-agnostic) block.
        # Even when the signal columns are present, upstream pipelines sometimes
        # emit an all-zero *_has_data flag; fix that here to prevent false
        # "missing source" failures in merged-panel quality gates.
        if out is df:
            out = out.copy()
        if "doc_embedding_novelty_hf_has_data" not in out.columns:
            out["doc_embedding_novelty_hf_has_data"] = np.nan
        try:
            prefix = "doc_embedding_novelty_hf_"
            family_cols = [
                c
                for c in out.columns
                if str(c).startswith(prefix) and str(c) != "doc_embedding_novelty_hf_has_data"
            ]
            if family_cols:
                block = out[family_cols]
                derived = (block.notna().any(axis=1)).astype(float)
                existing = pd.to_numeric(out["doc_embedding_novelty_hf_has_data"], errors="coerce").fillna(0.0)
                out["doc_embedding_novelty_hf_has_data"] = np.maximum(existing, derived)
        except Exception:
            pass

        # macro_tst_hf historically emits derived_* columns; ensure its has_data
        # flag reflects presence of the derived block.
        if "macro_tst_hf_has_data" not in out.columns:
            out["macro_tst_hf_has_data"] = np.nan
        try:
            macro_cols = [
                c
                for c in out.columns
                if (str(c).startswith("macro_tst_hf_") and str(c) != "macro_tst_hf_has_data")
                or str(c).startswith("derived_")
                or str(c).startswith("derived_interact")
            ]
            if "l3_real_interest_rate" in out.columns and "l3_real_interest_rate" not in macro_cols:
                macro_cols.append("l3_real_interest_rate")
            if macro_cols:
                block = out[macro_cols]
                derived = (block.notna().any(axis=1)).astype(float)
                existing = pd.to_numeric(out["macro_tst_hf_has_data"], errors="coerce").fillna(0.0)
                out["macro_tst_hf_has_data"] = np.maximum(existing, derived)
        except Exception:
            pass

        # Finalize: any remaining NaNs in these stubs are treated as "no data".
        for col in ("calibration_has_data", "doc_embedding_novelty_hf_has_data", "macro_tst_hf_has_data"):
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
            except Exception:
                pass

        return out

    view_mode = (view or "both").lower()
    if view_mode not in CACHE_VIEW_OPTIONS:
        raise ValueError(f"Invalid view '{view_mode}'. Expected one of {sorted(CACHE_VIEW_OPTIONS)}")
    stage_upper = (stage or "B").upper()
    stage_prefers_raw_features = stage_upper == "A"
    fams = _resolve_family_list(families, stage_upper)

    # ETFs: do NOT attempt equity-style fundamentals.
    if _is_etf_symbol(symbol):
        fams = [f for f in fams if not str(f).lower().startswith("fin_g")]
    if 'doc_embedding_novelty_hf' in fams:
        logger.info(
            "🔍 DEBUG build_panel: symbol=%s, start=%s, end=%s, families=%s, resolved=%s",
            symbol,
            start,
            end,
            families,
            fams,
        )
    frames: List[pd.DataFrame] = []
    provenance: Dict[str, dict] = {}
    telemetry: Dict[str, dict] = {}
    feature_counts: Dict[str, int] = {}
    DORMANT = {'status': 'dormant:no_data'}
    date_index = _make_index(start, end)
    start_str_default = date_index[0].strftime('%Y-%m-%d')
    end_str_default = date_index[-1].strftime('%Y-%m-%d')
    logger.debug(f"DEBUG build_panel: start_str_default={start_str_default}, end_str_default={end_str_default}")
    price_history_cache: Optional[pd.DataFrame] = None
    quantile_forecast_cache: Optional[pd.DataFrame] = None
    quantile_forecast_history: Optional[pd.DataFrame] = None

    # Prefer consolidated cache and slice by requested date bounds.
    # Per-window cache shards are optional; Stage-B windowing can be handled by slicing.
    cached_panel = _build_panel_from_cache(
        symbol=symbol,
        start=start or start_str_default,
        end=end or end_str_default,
        families=fams,
        cache_dir=cache_dir,
        horizon=horizon,
        view_mode=view_mode,
    )
    if cached_panel is not None and not cached_panel.empty:
        # Guard against stale/partial cache that doesn't actually cover the
        # requested window (can otherwise appear as a cache-hit but trim to
        # empty downstream in prep_families).
        cached_panel = _safe_filter(cached_panel, start, end)
        if cached_panel is not None and not cached_panel.empty:
            missing_families: List[str] = []
            try:
                missing_families = list(cached_panel.attrs.get("missing_families") or [])
            except Exception:
                missing_families = []

            if allow_generate_missing_from_cache and missing_families:
                logger.debug(
                    "build_panel partial cache hit for %s (%d cached, %d missing, view=%s) — generating missing",
                    symbol,
                    max(0, len(fams) - len(missing_families)),
                    len(missing_families),
                    view_mode,
                )
                frames = [cached_panel]
                missing_set = set(missing_families)
                fams = [f for f in fams if f in missing_set]
            else:
                logger.debug(
                    "build_panel cache hit for %s (%d families, view=%s)",
                    symbol,
                    len(fams),
                    view_mode,
                )
                return _apply_stable_schema_stubs(cached_panel)
        logger.debug(
            "build_panel cache did not cover requested range for %s; falling back to generation",
            symbol,
        )
    
    # Helper to load from shared cache (symbol-agnostic families like doc_embedding_novelty_hf)
    def _load_from_shared_cache(family: str, shared_dir: Path) -> Optional[pd.DataFrame]:
        """Load pre-generated signal from shared cache for symbol-agnostic families."""
        start_ts = pd.to_datetime(start or start_str_default)
        end_ts = pd.to_datetime(end or end_str_default)
        if hasattr(start_ts, 'tz') and start_ts.tz is not None:
            start_ts = start_ts.tz_localize(None)
        if hasattr(end_ts, 'tz') and end_ts.tz is not None:
            end_ts = end_ts.tz_localize(None)
        
        # Look for consolidated files (pattern: _YYYYMMDD_YYYYMMDD_features.parquet)
        pattern = "_*_features.parquet"
        matching_files = sorted(shared_dir.glob(pattern), reverse=True) if shared_dir.exists() else []
        
        for file_path in matching_files:
            try:
                df = pd.read_parquet(file_path)
                if df.empty or 'date' not in df.columns:
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)
                df = df.set_index('date').sort_index()
                
                # Check coverage
                cache_start = df.index.min()
                cache_end = df.index.max()
                requested_days = max(1, (end_ts - start_ts).days)
                coverage_start = max(start_ts, cache_start)
                coverage_end = min(end_ts, cache_end)
                covered_days = (coverage_end - coverage_start).days if coverage_end >= coverage_start else 0
                coverage_ratio = covered_days / requested_days if requested_days > 0 else 0
                
                if coverage_ratio < 0.8:
                    logger.debug(f"Shared cache for {family} only covers {coverage_ratio:.1%}, skipping")
                    continue
                
                # Filter to requested range
                df = df[(df.index >= start_ts) & (df.index <= end_ts)]
                if df.empty:
                    continue
                
                # Prefix unprefixed columns with family name for proper namespacing
                family_prefix = f"{family.lower()}_"
                new_columns = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if not col_lower.startswith(family_prefix):
                        new_columns[col] = f"{family_prefix}{col}"
                if new_columns:
                    df = df.rename(columns=new_columns)
                
                logger.info(f"✅ {family}: loaded from shared cache ({len(df)} rows)")
                
                # Load metadata if available
                meta_path = Path(str(file_path) + '.meta.json')
                meta_data = {}
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as handle:
                            meta_data = json.load(handle)
                    except Exception:
                        pass
                
                df.attrs['cache_meta'] = meta_data or {
                    'family': family,
                    'source': 'shared_cache',
                    'cache_start': str(cache_start.date()),
                    'cache_end': str(cache_end.date()),
                }
                return df
                
            except Exception as e:
                logger.debug(f"Failed to load shared cache {file_path}: {e}")
                continue
        
        return None
    
    # NEW: Helper to load cached signals from prep_families
    def _load_from_cache(family: str) -> Optional[pd.DataFrame]:
        """Load pre-generated signal from local_cache if available."""
        # Check shared cache first for symbol-agnostic families
        if _uses_shared_cache_path(family):
            shared_dir = _get_shared_cache_dir(family)
            if shared_dir is not None and shared_dir.exists():
                shared_result = _load_from_shared_cache(family, shared_dir)
                if shared_result is not None:
                    return shared_result
        
        if cache_dir is None:
            return None
        
        cache_path = Path(cache_dir)
        symbol_lower = symbol.lower()

        resolved_dir = _resolve_cache_dir_for_family(cache_path, symbol, horizon, family)
        symbol_lower = symbol.lower()
        expected_h = f"{symbol_lower}_h{int(horizon)}"
        legacy_dir = cache_path if cache_path.name == expected_h else (cache_path.parent / expected_h if cache_path.name == symbol_lower else cache_path / expected_h)

        # Per-window cache shards are not required; we slice consolidated caches by date.
        
        start_ts = pd.to_datetime(start or start_str_default)
        end_ts = pd.to_datetime(end or end_str_default)
        if hasattr(start_ts, 'tz') and start_ts.tz is not None:
            start_ts = start_ts.tz_localize(None)
        if hasattr(end_ts, 'tz') and end_ts.tz is not None:
            end_ts = end_ts.tz_localize(None)
        requested_days = max(1, (end_ts - start_ts).days)
        min_ratio_default = 0.8
        is_finbert_family = family == 'finbert'

        def _extend_finbert_calendar(frame: pd.DataFrame) -> pd.DataFrame:
            if not is_finbert_family or frame.empty:
                return frame
            finbert_cols = [c for c in frame.columns if c.startswith('finbert_')]
            if not finbert_cols:
                return frame
            calendar_index = pd.date_range(start_ts, end_ts, freq='D')
            extended = frame.reindex(calendar_index)
            for col in finbert_cols:
                extended[col] = extended[col].fillna(0.0)
            for col in [c for c in finbert_cols if c.endswith('has_data')]:
                extended[col] = extended[col].clip(lower=0.0, upper=1.0)
            return extended

        loaded_splits: List[Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, Any]]] = []

        # Try consolidated (Stage-A) file first, then legacy split (Stage-B) formats
        splits_to_try: List[Optional[str]] = [None, 'train', 'valid']
        
        for split in splits_to_try:
            candidates: List[Path] = []
            base_name = _cache_basename(symbol, horizon, family, split)
            if split:
                legacy_base = f"{symbol_lower}_h{int(horizon)}_{family}_{split}.parquet"
            else:
                legacy_base = f"{symbol_lower}_h{int(horizon)}_{family}.parquet"
            if view_mode in {"raw", "both"} or stage_prefers_raw_features:
                if WRITE_LAGGED_FEATURE_CACHES:
                    candidates.append(
                        resolved_dir / base_name.replace(".parquet", "_features_lagged.parquet")
                    )
                    if _uses_symbol_only_cache_path(family):
                        candidates.append(legacy_dir / legacy_base.replace(".parquet", "_features_lagged.parquet"))
                # PRIORITY 2: _features.parquet has raw features without lags
                candidates.append(resolved_dir / base_name.replace(".parquet", "_features.parquet"))
                if _uses_symbol_only_cache_path(family):
                    candidates.append(legacy_dir / legacy_base.replace(".parquet", "_features.parquet"))
            if view_mode in {"summary", "both"} and not stage_prefers_raw_features:
                candidates.append(resolved_dir / base_name)
                if _uses_symbol_only_cache_path(family):
                    candidates.append(legacy_dir / legacy_base)
            for file_path in candidates:
                if not file_path.exists():
                    continue
                try:
                    df = pd.read_parquet(file_path)
                    if df.empty or 'date' not in df.columns:
                        continue

                    # Set date index and normalize timezone
                    df['date'] = pd.to_datetime(df['date'])
                    if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                        df['date'] = df['date'].dt.tz_localize(None)
                    df = df.set_index('date').sort_index()

                    # Check if cache covers requested range BEFORE filtering
                    cache_start = df.index.min()
                    cache_end = df.index.max()
                    coverage_start = max(start_ts, cache_start)
                    coverage_end = min(end_ts, cache_end)
                    covered_days = (coverage_end - coverage_start).days if coverage_end >= coverage_start else 0
                    coverage_ratio = covered_days / requested_days if requested_days > 0 else 0

                    min_ratio = min_ratio_default if not is_finbert_family else 0.1
                    if coverage_ratio < min_ratio:
                        split_label = split or 'unified'
                        logger.debug(
                            f"Cache for {family}/{split_label} only covers {coverage_ratio:.1%} of requested range, skipping candidate"
                        )
                        continue

                    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

                    if df.empty:
                        continue

                    meta_data: Dict[str, Any] = {}
                    meta_path = Path(str(file_path) + '.meta.json')
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r') as handle:
                                meta_data = json.load(handle)
                        except Exception as meta_exc:
                            logger.debug("Failed to parse cache metadata %s: %s", meta_path, meta_exc)

                    if not meta_data:
                        meta_data = {
                            'symbol': symbol,
                            'family': family,
                            'split': split,
                            'horizon': horizon,
                            'requested_start': start_ts.isoformat() if start_ts is not None else None,
                            'requested_end': end_ts.isoformat() if end_ts is not None else None,
                            'cache_start': cache_start.isoformat() if isinstance(cache_start, pd.Timestamp) else str(cache_start),
                            'cache_end': cache_end.isoformat() if isinstance(cache_end, pd.Timestamp) else str(cache_end),
                            'data_start': df.index.min().isoformat() if len(df.index) else None,
                            'data_end': df.index.max().isoformat() if len(df.index) else None,
                            'coverage_ratio': coverage_ratio,
                            'cache_path': str(file_path),
                            'generated_ts': datetime.utcfromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
                        }

                    # Determine cache kind from filename
                    file_name = file_path.name
                    if '_features_lagged' in file_name:
                        cache_kind = 'features_lagged'
                    elif '_features' in file_name:
                        cache_kind = 'features'
                    else:
                        cache_kind = 'signals'

                    split_label = split or 'unified'
                    telemetry_payload = {
                        'status': 'ok',
                        'source': f'cached_{split_label}',
                        'proxy': False,
                        'cache_path': str(file_path),
                        'coverage_ratio': coverage_ratio,
                        'cache_kind': cache_kind,
                    }

                    loaded_splits.append((split_label, df, meta_data, telemetry_payload))
                    # If we found unified cache, no need to try train/valid splits
                    if split is None:
                        break
                except Exception as e:
                    split_label = split or 'unified'
                    logger.debug(f"Failed to load cache for {family}/{split_label}: {e}")
                    continue
            # If unified cache was found, skip legacy splits
            if loaded_splits and loaded_splits[0][0] == 'unified':
                break

        if not loaded_splits:
            return None

        if len(loaded_splits) == 1:
            split_name, df, meta_payload, telemetry_payload = loaded_splits[0]
            df = _extend_finbert_calendar(df)
            df.attrs['cache_meta'] = meta_payload
            df.attrs['telemetry'] = telemetry_payload
            return df

        # When both train+valid exist, DO NOT let sparse valid snapshots overwrite richer train history.
        # Prefer train values on overlapping dates, but fill missing values from valid.
        split_map = {entry[0]: entry[1] for entry in loaded_splits}
        df_train = split_map.get('train')
        df_valid = split_map.get('valid')
        if df_train is not None and df_valid is not None:
            union_index = df_train.index.union(df_valid.index)
            df_train = df_train.reindex(union_index)
            df_valid = df_valid.reindex(union_index)
            # keep train where present; fill NaNs from valid
            df_combined = df_train.combine_first(df_valid)
        else:
            combined_frames = [entry[1] for entry in loaded_splits]
            df_combined = pd.concat(combined_frames, axis=0)
            # Prefer train-first semantics when duplicates exist.
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()

        df_combined = df_combined.sort_index()
        df_combined = _extend_finbert_calendar(df_combined)

        coverage_start = max(start_ts, df_combined.index.min()) if len(df_combined.index) else start_ts
        coverage_end = min(end_ts, df_combined.index.max()) if len(df_combined.index) else end_ts
        combined_days = max(0, (coverage_end - coverage_start).days)
        combined_ratio = combined_days / requested_days if requested_days > 0 else 0

        df_combined.attrs['cache_meta'] = {
            'symbol': symbol,
            'family': family,
            'splits': [entry[0] for entry in loaded_splits],
            'horizon': horizon,
            'requested_start': start_ts.isoformat() if start_ts is not None else None,
            'requested_end': end_ts.isoformat() if end_ts is not None else None,
            'data_start': df_combined.index.min().isoformat() if len(df_combined.index) else None,
            'data_end': df_combined.index.max().isoformat() if len(df_combined.index) else None,
            'coverage_ratio': combined_ratio,
            'cache_paths': [entry[2].get('cache_path') for entry in loaded_splits],
        }
        df_combined.attrs['telemetry'] = {
            'status': 'ok',
            'source': f"cached_{'+'.join(entry[0] for entry in loaded_splits)}",
            'proxy': False,
            'cache_paths': [entry[2].get('cache_path') for entry in loaded_splits],
            'coverage_ratio': combined_ratio,
            'cache_kind': 'mixed',
        }
        return df_combined

    def _block_hf_root() -> Optional[Path]:
        if cache_dir is None:
            return None
        try:
            root = Path(cache_dir)
        except Exception:
            return None
        return root / 'block_hf' / symbol.upper() / str(horizon)

    def _load_block_hf_signal(block_family: str) -> Optional[pd.DataFrame]:
        block_root = _block_hf_root()
        if block_root is None:
            logger.warning("hf_agg requires cache_dir with block_hf outputs; missing for %s", block_family)
            block_root = None

        file_path = (block_root / f"{block_family}.parquet") if block_root is not None else None
        if file_path is None or not file_path.exists():
            # Fallback: load from the standard consolidated cache files produced by prep_families
            # (e.g. <symbol>_h<h>_<block>_{train|valid}(_features(_lagged)).parquet).
            try:
                cached = _load_cached_family_for_hf('hf_agg', block_family)
            except Exception as exc:
                cached = None
                logger.warning("hf_agg: failed consolidated-cache fallback for %s: %s", block_family, exc)
            if cached is None or cached.empty:
                if file_path is not None:
                    logger.warning("hf_agg: missing cached block signal %s", file_path)
                return None

            df = cached.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("hf_agg: consolidated fallback for %s missing datetime index", block_family)
                return None
            df.index = df.index.tz_localize(None)
        else:
            try:
                df = pd.read_parquet(file_path)
            except Exception as exc:
                logger.warning("hf_agg: failed reading %s: %s", file_path, exc)
                return None
            if df.empty:
                return None
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("hf_agg: block %s missing datetime index", block_family)
                return None
            df.index = df.index.tz_localize(None)
        def _resolve_column(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                if cand in df.columns:
                    return cand
            return None
        score_col = _resolve_column([
            f"{block_family}_score",
            'score',
            'hf_score',
        ])
        conf_col = _resolve_column([
            f"{block_family}_conf",
            'conf',
            'hf_conf',
        ])
        if score_col is None or conf_col is None:
            logger.warning("hf_agg: block %s missing score/conf columns", block_family)
            return None
        out = pd.DataFrame({
            f"{block_family}_score": df[score_col].astype(float),
            f"{block_family}_conf": df[conf_col].astype(float),
        }, index=df.index)
        return out

    def _load_price_history() -> Optional[pd.DataFrame]:
        nonlocal price_history_cache
        if price_history_cache is None:
            history = _fetch_price_data(symbol, start_str_default, end_str_default)
            if history is not None and not history.empty:
                history.index = pd.to_datetime(history.index)
                price_history_cache = history.sort_index()
            else:
                price_history_cache = None
        return price_history_cache

    def _normalize_ts(value: Any) -> Optional[pd.Timestamp]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts
        except Exception:
            return None

    def _load_cached_family_for_hf(block_name: str, family: str) -> Optional[pd.DataFrame]:
        # HF blocks are typically computed from cached family outputs produced by prep_families.
        # In stateful runs, build_panel may be called with cache_dir=None; fall back to the
        # standard repo-local cache directory in that case.
        df_cached: Optional[pd.DataFrame] = None
        if cache_dir is not None:
            df_cached = _load_from_cache(family)
        else:
            try:
                fallback_cache_dir = Path(__file__).resolve().parents[2] / 'data' / 'local_cache' / f"{symbol.lower()}_h{int(horizon)}"
                df_cached = _build_panel_from_cache(
                    symbol=symbol,
                    start=start or start_str_default,
                    end=end or end_str_default,
                    families=(family,),
                    cache_dir=str(fallback_cache_dir),
                    horizon=horizon,
                    view_mode=view_mode,
                )
            except Exception:
                df_cached = None

        if df_cached is None or df_cached.empty:
            logger.debug(f"{block_name}: no cached data found for family '{family}'")
            return None
        meta = getattr(df_cached, 'attrs', {}).get('cache_meta', {})
        sym_meta = str(meta.get('symbol', '')).lower()
        if sym_meta and sym_meta != symbol.lower():
            logger.warning(f"{block_name}: cache symbol mismatch for {family} (cache={sym_meta}, requested={symbol.lower()})")
            return None
        horizon_meta = meta.get('horizon')
        if horizon_meta is not None:
            try:
                horizon_meta_int = int(horizon_meta)
            except Exception:
                horizon_meta_int = None
            if horizon_meta_int is not None and horizon_meta_int != int(horizon):
                logger.warning(f"{block_name}: cache horizon mismatch for {family} (cache={horizon_meta}, requested={horizon})")
                return None
        req_start_meta = _normalize_ts(meta.get('requested_start'))
        req_end_meta = _normalize_ts(meta.get('requested_end'))
        req_start_current = _normalize_ts(start or start_str_default)
        req_end_current = _normalize_ts(end or end_str_default)
        if req_start_meta is not None and req_start_current is not None and req_start_meta != req_start_current:
            logger.warning(
                f"{block_name}: cache start mismatch for {family} (cache={req_start_meta.date()}, requested={req_start_current.date()}) - using cached data because coverage check already passed"
            )
        if req_end_meta is not None and req_end_current is not None and req_end_meta != req_end_current:
            logger.warning(
                f"{block_name}: cache end mismatch for {family} (cache={req_end_meta.date()}, requested={req_end_current.date()}) - using cached data because coverage check already passed"
            )
        return df_cached

    def _price_window() -> Optional[pd.DataFrame]:
        history = _load_price_history()
        if history is None:
            return None
        start_bound = pd.to_datetime(start or start_str_default)
        end_bound = pd.to_datetime(end or end_str_default)
        return history[(history.index >= start_bound) & (history.index <= end_bound)].copy()

    def _collect(df: pd.DataFrame, fam: str) -> pd.DataFrame:
        # Prefix columns to avoid collisions and signal provenance
        df2 = df.copy()
        
        # Raw fundamental families (fin_g0-7, options, etc.) should NOT have confidence scores
        # These are RAW data for Stage A to learn from, not processed signals
        raw_families = [
            'fin_g0', 'fin_g1', 'fin_g2', 'fin_g3', 'fin_g4', 'fin_g5', 'fin_g6', 'fin_g7',
            'options', 'options_anchoring',  # Raw options data
            'cross_asset',  # Raw correlations
            'alternative_signals',  # Raw alternative data
            'cboe_term',  # Raw VIX term structure
            'garch_iv',  # Raw GARCH volatility features
            'ml_framework',  # Raw technical indicator bundle sourced from EODHD
        ]
        
        # ADD CONFIDENCE SCORE if not already present (SKIP for raw families)
        conf_cols = [c for c in df2.columns if 'conf' in c.lower() or 'confidence' in c.lower()]
        if (
            not stage_prefers_raw_features
            and not conf_cols
            and not df2.empty
            and fam not in raw_families
        ):
            # Universal confidence: inverse of signal volatility
            numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # Rolling std across all features
                volatilities = [df2[col].rolling(20, min_periods=1).std() for col in numeric_cols]
                if volatilities:
                    mean_vol = pd.concat(volatilities, axis=1).mean(axis=1)
                    confidence = (1.0 / (1.0 + mean_vol)).clip(0.0, 1.0)
                    df2['CONFIDENCE'] = confidence
        
        df2 = to_nyse_close_index(df2)
        if hasattr(df2.index, 'tz') and df2.index.tz is not None:
            try:
                df2.index = df2.index.tz_convert('UTC').tz_localize(None)
            except TypeError:
                df2.index = df2.index.tz_localize(None)
        normalized_cols, renamed_count = _normalize_family_column_names(fam, df2.columns)
        if renamed_count:
            logger.debug("%s: normalized %d duplicate column names", fam, renamed_count)
        df2.columns = [f"{fam}_{c}" for c in normalized_cols]
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # GOVERNANCE COLUMN CENTRAL COMPUTATION (Jan 2026)
        # ═══════════════════════════════════════════════════════════════════════════════
        # Canonical producer for: {family}_has_data, {family}_activity, {family}_days_since_update
        # Runs for live + cached frames and overwrites any pre-existing values.
        # ═══════════════════════════════════════════════════════════════════════════════
        if not df2.empty:
            has_data_col = f"{fam}_has_data"
            activity_col = f"{fam}_activity"
            days_since_col = f"{fam}_days_since_update"

            core_cols = resolve_core_columns(fam, df2.columns)
            if not core_cols:
                # Fallback: any family-prefixed non-governance columns
                core_cols = [
                    c for c in df2.columns
                    if str(c).lower().startswith(f"{fam.lower()}_")
                    and not str(c).lower().endswith((
                        "_has_data",
                        "_activity",
                        "_days_since_update",
                        "_confidence",
                        "_conf",
                    ))
                ]

            # has_data: 1 if any core column has non-null, non-zero data
            if core_cols:
                core_frame = df2[core_cols]
                numeric = core_frame.select_dtypes(include=[np.number, "bool"])
                if not numeric.empty:
                    has_data = (
                        numeric.notna().any(axis=1) & (numeric.abs().sum(axis=1) > 0)
                    ).astype(float)
                else:
                    has_data = core_frame.notna().any(axis=1).astype(float)
            else:
                has_data = pd.Series(0.0, index=df2.index)

            df2[has_data_col] = has_data

            # days_since_update: prefer source_asof timestamp; fallback to last has_data
            source_ts = None
            for key in (f"{fam}:source_asof_ts", f"{fam}:fetch_ts"):
                if key in (df2.attrs or {}):
                    source_ts = df2.attrs.get(key)
                    break

            ts_col = None
            candidate = f"{fam}_source_asof_ts"
            if candidate in df2.columns:
                ts_col = candidate

            if source_ts is not None:
                asof = pd.to_datetime(source_ts, errors="coerce")
                days_since = np.maximum((df2.index - asof).days, 0)
                df2[days_since_col] = pd.Series(days_since, index=df2.index, dtype=float)
            elif ts_col is not None:
                asof_series = pd.to_datetime(df2[ts_col], errors="coerce")
                last_asof = asof_series.ffill()
                days_since = np.maximum((df2.index.to_series() - last_asof).dt.days, 0)
                df2[days_since_col] = days_since.fillna(0.0).astype(float)
            else:
                eps_abs = 1e-12
                eps_rel = 1e-6
                update_mask = None
                if core_cols:
                    core_frame = df2[core_cols]
                    prev_frame = core_frame.shift(1)
                    change_mask = pd.Series(False, index=df2.index)

                    numeric_cols = core_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
                    if numeric_cols:
                        cur = core_frame[numeric_cols]
                        prev = prev_frame[numeric_cols]
                        diff = (cur - prev).abs()
                        thresh = eps_abs + eps_rel * prev.abs()
                        num_changed = diff > thresh
                        change_mask = change_mask | num_changed.fillna(False).any(axis=1)

                    other_cols = [c for c in core_cols if c not in numeric_cols]
                    if other_cols:
                        other_changed = core_frame[other_cols] != prev_frame[other_cols]
                        change_mask = change_mask | other_changed.fillna(False).any(axis=1)

                    update_mask = (has_data > 0) & change_mask

                if update_mask is not None and bool(update_mask.any()):
                    last_update = df2.index.to_series().where(update_mask).ffill()
                else:
                    last_update = df2.index.to_series().where(has_data > 0).ffill()
                days_since = (df2.index.to_series() - last_update).dt.days
                df2[days_since_col] = days_since.fillna(0.0).astype(float)

            # activity: exp decay based on cadence
            meta = family_meta_registry.get(str(fam), None)
            cadence_token = str(getattr(meta, "update_cadence", "unknown")).lower() if meta else "unknown"
            cadence_days = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90,
                "event": 7,
                "snapshot": 30,
            }.get(cadence_token, 1)
            expected_latency = 0.0
            tau = max(2.0 * cadence_days, cadence_days + expected_latency)
            df2[activity_col] = np.exp(-df2[days_since_col].astype(float) / float(tau)).clip(0.0, 1.0)
        
        # Collect attrs if present
        try:
            if hasattr(df, 'attrs'):
                prov = getattr(df, 'attrs', {}).get('provenance')
                if isinstance(prov, dict):
                    provenance[fam] = prov
                tel = getattr(df, 'attrs', {}).get('telemetry')
                if isinstance(tel, dict):
                    telemetry[fam] = tel
        except Exception:
            pass
        feature_counts[fam] = int(len(df2.columns))
        return df2

    def _prepare_family_block(
        df: Optional[pd.DataFrame],
        family: str,
        max_cols: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return None
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.sort_index().ffill().bfill()
        numeric = numeric.dropna(axis=1, how='all')
        if numeric.empty:
            return None
        std = numeric.std(ddof=0)
        valid_cols = std[std > 1e-12].index
        if not len(valid_cols):
            return None
        numeric = numeric[valid_cols]
        if max_cols and len(numeric.columns) > max_cols:
            variances = numeric.var().sort_values(ascending=False)
            keep = list(variances.index[:max_cols])
            numeric = numeric[keep]
        prefix = f"{family}_"
        renamed = {
            col: col if col.startswith(prefix) else f"{prefix}{col}"
            for col in numeric.columns
        }
        return numeric.rename(columns=renamed)

    def _merge_family_blocks(
        blocks: List[pd.DataFrame],
        max_total_cols: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        frames = [blk for blk in blocks if blk is not None and not blk.empty]
        if not frames:
            return None
        merged = frames[0]
        for df_block in frames[1:]:
            merged = merged.join(df_block, how='outer')
        merged = merged.sort_index().replace([np.inf, -np.inf], np.nan)
        merged = merged.ffill().bfill().dropna(how='all')
        if merged.empty:
            return None
        numeric = merged.select_dtypes(include=[np.number])
        if numeric.empty:
            return None
        if max_total_cols and len(numeric.columns) > max_total_cols:
            variances = numeric.var().sort_values(ascending=False)
            keep = list(variances.index[:max_total_cols])
            numeric = numeric[keep]
        std = numeric.std(ddof=0).replace(0, np.nan)
        normalized = (numeric - numeric.mean()) / (std + 1e-6)
        normalized = normalized.fillna(0.0)
        return normalized

    def _filter_columns_by_keywords(
        df: Optional[pd.DataFrame],
        keywords: List[str],
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if not keywords:
            return df
        lowered = [kw.lower() for kw in keywords]
        cols = [
            col
            for col in df.columns
            if any(kw in col.lower() for kw in lowered)
        ]
        if not cols:
            return None
        return df[cols]

    def _build_sliding_windows(
        matrix: np.ndarray,
        idx: pd.Index,
        window: int,
    ) -> Tuple[np.ndarray, pd.Index]:
        if len(matrix) < window:
            empty_idx = pd.Index([], dtype=idx.dtype)
            return np.empty((0, window, matrix.shape[1]), dtype=np.float32), empty_idx
        sequences = []
        seq_index: List[pd.Timestamp] = []
        for pos in range(window - 1, len(matrix)):
            sequences.append(matrix[pos - window + 1:pos + 1])
            seq_index.append(idx[pos])
        stacked = np.asarray(sequences, dtype=np.float32)
        return stacked, pd.Index(seq_index, name=idx.name)

    def _hf_baseline_projection(
        feature_df: pd.DataFrame,
        prefix: str,
        window: int,
        families_used: List[str],
    ) -> Optional[pd.DataFrame]:
        if feature_df is None or feature_df.empty:
            return None
        signal = feature_df.mean(axis=1)
        rolling = signal.rolling(window, min_periods=max(5, window // 4))
        zscore = (signal - rolling.mean()) / (rolling.std() + 1e-6)
        conf = np.abs(zscore)
        conf = conf / (conf.rolling(window, min_periods=5).max() + 1e-6)
        df = pd.DataFrame({
            f"{prefix}_score": zscore.clip(-1.0, 1.0).fillna(0.0),
            f"{prefix}_conf": conf.clip(0.0, 1.0).fillna(0.0),
            f"{prefix}_score_raw": zscore.fillna(0.0),
        }, index=feature_df.index)
        df.attrs['telemetry'] = {
            'status': 'proxy',
            'source': 'hf_baseline_projection',
            'window': window,
            'families': families_used,
        }
        df.attrs['feature_counts'] = {'generated': len(df.columns)}
        df.attrs['provenance'] = {
            'families_used': families_used,
            'mode': 'baseline',
        }
        return df

    def _run_hf_sequence_block(
        feature_df: Optional[pd.DataFrame],
        price_series: Optional[pd.Series],
        window: int,
        horizon_local: int,
        prefix: str,
        families_used: List[str],
        epochs: int,
    ) -> Optional[pd.DataFrame]:
        if feature_df is None or feature_df.empty:
            return None
        if price_series is None or price_series.empty:
            return None
        if (
            HFTimeSeriesConfig is None
            or HFTimeSeriesModule is None
            or configure_optimizer is None
            or configure_scheduler is None
            or clip_gradients is None
            or torch is None
        ):
            return _hf_baseline_projection(feature_df, prefix, window, families_used)

        aligned = feature_df.join(price_series.rename('close'), how='inner').dropna(subset=['close'])
        if aligned.empty or len(aligned) < (window + horizon_local + 5):
            return _hf_baseline_projection(feature_df, prefix, window, families_used)

        close = aligned['close']
        feat_only = aligned.drop(columns=['close'])
        sequences, seq_index = _build_sliding_windows(feat_only.to_numpy(dtype=np.float32), feat_only.index, window)
        if sequences.size == 0 or seq_index.empty:
            return _hf_baseline_projection(feature_df, prefix, window, families_used)

        forward = (close.shift(-horizon_local) / close) - 1.0
        target_series = (forward > 0).astype(float).reindex(seq_index)
        targets = target_series.to_numpy()
        train_mask = np.isfinite(targets)
        if not train_mask.any() or train_mask.sum() < max(32, window):
            return _hf_baseline_projection(feature_df, prefix, window, families_used)

        train_inputs = sequences[train_mask]
        train_targets = targets[train_mask].astype(np.float32)
        max_samples = 4096
        if len(train_inputs) > max_samples:
            idx = np.linspace(0, len(train_inputs) - 1, num=max_samples, dtype=int)
            train_inputs = train_inputs[idx]
            train_targets = train_targets[idx]

        cfg = HFTimeSeriesConfig(input_dim=train_inputs.shape[2], window=window, horizon=horizon_local)
        try:
            model = HFTimeSeriesModule(cfg)
        except Exception:
            return _hf_baseline_projection(feature_df, prefix, window, families_used)

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        optimizer = configure_optimizer(model, lr=5e-4)
        scheduler = configure_scheduler(optimizer, total_steps=max(1, epochs))

        batch_size = min(128, max(32, len(train_inputs) // 4))
        rng = np.random.default_rng(_seed_from_symbol(symbol))
        train_tensor = torch.from_numpy(train_inputs).to(device)
        target_tensor = torch.from_numpy(train_targets).to(device)

        for _ in range(epochs):
            indices = np.arange(len(train_inputs))
            rng.shuffle(indices)
            for start_idx in range(0, len(indices), batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                batch_inputs = train_tensor[batch_idx]
                batch_targets = target_tensor[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_inputs)
                loss = model.compute_loss(outputs['logits'], batch_targets)
                loss.backward()
                clip_gradients(model)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            seq_tensor = torch.from_numpy(sequences).to(device)
            outputs = model(seq_tensor)
            scores = outputs['hf_score'].detach().cpu().numpy().astype(np.float32)
            confs = outputs['hf_conf'].detach().cpu().numpy().astype(np.float32)

        result = pd.DataFrame({
            f"{prefix}_score": scores,
            f"{prefix}_conf": confs,
            f"{prefix}_score_raw": scores,
        }, index=seq_index)
        result = result.reindex(feature_df.index).ffill().fillna(0.0)
        result.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'HFTimeSeriesModule',
            'window': window,
            'horizon': horizon_local,
            'families': families_used,
            'train_samples': int(len(train_inputs)),
            'total_sequences': int(len(seq_index)),
            'epochs': epochs,
        }
        result.attrs['feature_counts'] = {'generated': len(result.columns)}
        result.attrs['provenance'] = {
            'families_used': families_used,
            'model': 'HFTimeSeriesModule',
        }
        return result

    # Note: formerly used a 30-day bound for intraday windows; we now delegate range gating to
    # microstructure.fetch which decides between intraday and daily-proxy sources.

    # Handlers map to reduce branching complexity
    cw = corr_windows or [5, 20, 60]
    cb = tuple(corr_breaks) if corr_breaks else (0.25, 0.85)

    def _h_cross():
        return fetch_cross(symbol, start=start, end=end, mapping=None, macro_lag_days=int(macro_lag_days)) if fetch_cross else None  # type: ignore

    # _h_macro() removed - consolidated into macro_tst_hf unified family

    def _h_cboe():
        return fetch_cboe(start=start, end=end, symbol=symbol) if fetch_cboe else None  # type: ignore

    def _h_garch():
        # Try real provider first if available
        if fetch_garch is not None:
            try:
                df = fetch_garch(symbol, start=start, end=end)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Real provider succeeded - add telemetry if missing
                    if not hasattr(df, 'attrs') or 'telemetry' not in df.attrs:
                        df.attrs['telemetry'] = {'status': 'ok', 'source': 'GarchProvider', 'proxy': False}
                    return df
            except Exception as exc:
                logger.debug("primary garch fetch failed for %s: %s", symbol, exc)

        # Fallback: Calculate GARCH-style volatility features
        try:
            # Fallback: Calculate GARCH-style volatility features (FULL TIME SERIES)
            price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            if price_df is None or 'close' not in price_df.columns:
                return None
            
            # Filter to requested date range
            price_df = price_df[
                (price_df.index >= pd.Timestamp(start)) & 
                (price_df.index <= pd.Timestamp(end))
            ]

            returns = price_df['close'].pct_change()
            if len(returns) < 5:
                return None

            # Calculate rolling volatility metrics for ALL rows
            # Adaptive windows based on data length for maximum coverage
            data_len = len(returns)
            if data_len < 20:
                # Very short: use aggressive minimal windows to preserve data
                short_window = max(3, data_len // 4)  # 20 rows → 5
                mid_window = max(4, data_len // 3)    # 20 rows → 6-7
                long_window = max(5, data_len // 3)   # 20 rows → 6-7
                min_periods_short = max(2, short_window // 2)
                min_periods_mid = max(2, mid_window // 2)
                min_periods_long = max(2, long_window // 3)
            elif data_len < 100:
                # Medium: balanced approach
                short_window = max(5, data_len // 6)   # 61 → 10
                mid_window = max(10, data_len // 3)    # 61 → 20
                long_window = max(10, data_len // 4)   # 61 → 15
                min_periods_short = max(2, short_window // 4)
                min_periods_mid = max(2, mid_window // 4)
                min_periods_long = max(2, long_window // 4)
            else:
                # Normal: use standard windows
                short_window = 15
                mid_window = 60
                long_window = 30
                min_periods_short = max(3, short_window // 3)
                min_periods_mid = max(10, mid_window // 6)
                min_periods_long = max(3, long_window // 3)
            
            # === BASIC VOLATILITY METRICS ===
            var_short = returns.pow(2).rolling(window=short_window, min_periods=min_periods_short).mean()
            var_mid = returns.pow(2).rolling(window=mid_window, min_periods=min_periods_mid).mean()
            var_long = returns.pow(2).rolling(window=long_window, min_periods=min_periods_long).mean()
            shock = returns.pow(2)
            
            short_run_vol = np.sqrt(var_short) * np.sqrt(252)
            mid_vol_60 = np.sqrt(var_mid) * np.sqrt(252)
            long_run_vol = np.sqrt(var_long) * np.sqrt(252)
            vol_ratio = short_run_vol / (long_run_vol + 1e-6) - 1.0
            
            # === 1. VOL TERM STRUCTURE ===
            vol_slope_short_mid = (short_run_vol - mid_vol_60) / (mid_vol_60 + 1e-6)
            vol_slope_mid_long = (mid_vol_60 - long_run_vol) / (long_run_vol + 1e-6)
            
            # === 2. VOL-OF-VOL (Stability) ===
            vol_of_vol_20 = short_run_vol.rolling(window=20, min_periods=5).std()
            vol_of_vol_ratio = vol_of_vol_20 / (long_run_vol + 1e-6)
            
            # === 3. GARCH(1,1) APPROXIMATION ===
            # Simple GARCH(1,1): σ²ₜ = ω + α*r²ₜ₋₁ + β*σ²ₜ₋₁
            # Use rolling estimates for α, β, ω
            alpha = 0.05  # Typical GARCH alpha (news coefficient)
            beta = 0.90   # Typical GARCH beta (persistence)
            omega = var_long * (1 - alpha - beta)  # Long-run variance anchor
            
            # Conditional variance (GARCH forecast)
            garch_cond_var = omega + alpha * shock.shift(1) + beta * var_short.shift(1)
            garch_cond_var = garch_cond_var.fillna(var_short)  # Fallback to realized var
            
            # GARCH state features
            garch_persistence = alpha + beta  # Typically ~0.95
            garch_half_life = np.log(0.5) / np.log(garch_persistence) if garch_persistence > 0 and garch_persistence < 1 else 250
            garch_half_life = np.clip(garch_half_life, 1, 250)  # Cap at 250 days
            
            garch_long_run_var = omega / (1 - garch_persistence + 1e-9)
            garch_long_run_var = garch_long_run_var.clip(lower=0)
            
            # Standardized residuals
            garch_std_resid = returns / (np.sqrt(garch_cond_var) + 1e-9)
            garch_std_resid_sq = garch_std_resid ** 2
            
            # === 4. ASYMMETRY & LEVERAGE EFFECT ===
            neg_shock_dummy = ((returns < 0) & (np.abs(returns) > short_run_vol / np.sqrt(252))).astype(float)
            neg_shock_magnitude = np.minimum(returns, 0) / (short_run_vol / np.sqrt(252) + 1e-9)
            pos_shock_magnitude = np.maximum(returns, 0) / (short_run_vol / np.sqrt(252) + 1e-9)
            shock_asymmetry_20 = (neg_shock_magnitude - pos_shock_magnitude).rolling(20, min_periods=5).mean()
            
            # === 5. REGIME & Z-SCORE FEATURES ===
            # Z-scores vs 252-day history
            short_vol_mean_252 = short_run_vol.rolling(252, min_periods=60).mean()
            short_vol_std_252 = short_run_vol.rolling(252, min_periods=60).std()
            short_vol_z_252 = (short_run_vol - short_vol_mean_252) / (short_vol_std_252 + 1e-9)
            
            long_vol_mean_252 = long_run_vol.rolling(252, min_periods=60).mean()
            long_vol_std_252 = long_run_vol.rolling(252, min_periods=60).std()
            long_vol_z_252 = (long_run_vol - long_vol_mean_252) / (long_vol_std_252 + 1e-9)
            
            # Regime flags
            vol_low_regime = (short_vol_z_252 < -0.5).astype(float)
            vol_high_regime = (short_vol_z_252 > 0.5).astype(float)
            
            # Spike/Crush detection
            shock_p95 = shock.rolling(252, min_periods=60).quantile(0.95)
            vol_spike_flag = ((short_vol_z_252 > 2) & (shock > shock_p95)).astype(float)
            vol_crush_flag = ((short_vol_z_252 < -2) & (shock < shock.rolling(20, min_periods=5).quantile(0.1))).astype(float)

            # === ASSEMBLE DATAFRAME ===
            garch_df = pd.DataFrame({
                # Original 4 features
                'short_vol': short_run_vol,
                'long_vol': long_run_vol,
                'vol_ratio': vol_ratio,
                'last_shock': shock,
                
                # 1. Vol term structure (3 features)
                'mid_vol_60': mid_vol_60,
                'vol_slope_short_mid': vol_slope_short_mid,
                'vol_slope_mid_long': vol_slope_mid_long,
                
                # 2. Vol-of-vol stability (2 features)
                'vol_of_vol_20': vol_of_vol_20,
                'vol_of_vol_ratio': vol_of_vol_ratio,
                
                # 3. GARCH state (5 features)
                'garch_cond_var': garch_cond_var,
                'garch_persistence': garch_persistence,
                'garch_half_life': garch_half_life,
                'garch_long_run_var': garch_long_run_var,
                'garch_std_resid_sq': garch_std_resid_sq,
                
                # 4. Asymmetry (2 features)
                'neg_shock_dummy': neg_shock_dummy,
                'shock_asymmetry_20': shock_asymmetry_20,
                
                # 5. Regime features (6 features)
                'short_vol_z_252': short_vol_z_252,
                'long_vol_z_252': long_vol_z_252,
                'vol_low_regime': vol_low_regime,
                'vol_high_regime': vol_high_regime,
                'vol_spike_flag': vol_spike_flag,
                'vol_crush_flag': vol_crush_flag,
            }, index=price_df.index)
            
            garch_df.attrs['telemetry'] = {'status': 'ok', 'source': 'GarchFallback', 'proxy': True}
            garch_df.attrs['feature_counts'] = {'generated': len(garch_df.columns)}
            
            return garch_df
        except Exception as exc:
            logger.debug("garch fallback failed for %s: %s", symbol, exc)
            return None

    def _h_corr():
        return (
            fetch_corr(
                symbol,
                start=start,
                end=end,
                windows=cw,
                add_break_flags=True,
                low_break=float(cb[0]),
                high_break=float(cb[1]),
                flat_gate_n=int(corr_flat_gate_n),
                flat_eps=float(corr_flat_eps),
                add_advanced_features=bool(corr_advanced_features),  # 🔥 Alpha engines
                lag_periods=corr_lag_periods or [1, 2, 5, 10],  # 🔥 Lagged cross-corr
                trend_lookback=int(corr_trend_lookback),  # 🔥 Correlation momentum
                vol_window=int(corr_vol_window),  # 🔥 Correlation volatility
            )
            if fetch_corr else None
        )  # type: ignore

    def _h_finbert():
        if fetch_finbert is None:
            return None
        try:
            return fetch_finbert(symbol, start=start, end=end, gap_thresh=int(finbert_gap_thresh))
        except TypeError:
            return fetch_finbert(symbol, start=start, end=end)

    def _h_crypto():
        return fetch_crypto(symbol, start=start, end=end) if fetch_crypto else None  # type: ignore

    def _h_fx():
        return fetch_fx(symbol, start=start, end=end) if fetch_fx else None  # type: ignore

    def _h_commodities():
        return fetch_commodities(symbol, start=start, end=end) if fetch_commodities else None  # type: ignore

    # _h_macro_enhanced() removed - consolidated into macro_tst_hf unified family

    def _h_options():
        """
        Options data using EODHD API.
        
        Returns RAW OPTIONS FEATURES (no scores/signals - raw data for Stage A):
        
        A. IMPLIED VOLATILITY (per strike/expiry):
           - atm_iv: At-the-money implied volatility
           - call_iv_avg: Average call implied volatility
           - put_iv_avg: Average put implied volatility
           - iv_spread: put_iv - call_iv (put/call skew)
        
        B. VOLUME & OPEN INTEREST:
           - call_volume: Total call volume
           - put_volume: Total put volume
           - put_call_volume_ratio: put_volume / call_volume
           - call_oi: Total call open interest
           - put_oi: Total put open interest
           - put_call_oi_ratio: put_oi / call_oi
        
        C. PRICING:
           - call_bid_avg, call_ask_avg, call_last_avg
           - put_bid_avg, put_ask_avg, put_last_avg
           - bid_ask_spread_pct: (ask - bid) / mid
        
        D. MONEYNESS:
           - strikes_available: Number of strikes
           - otm_call_pct: % of strikes that are OTM calls
           - otm_put_pct: % of strikes that are OTM puts
        
        E. TERM STRUCTURE:
           - nearest_expiry_days: Days to nearest expiration
           - expiries_available: Number of expiration dates
        
        Source: EODHD Options API
        Coverage: Real-time options chains for US equities
        
        NOTE: Returns RAW options metrics, NOT directional signals.
              Stage A learns how these relate to returns.
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🎯 options: Fetching EODHD options for {symbol}")
            
            # Get EODHD provider
            eodhd = get_eodhd_provider()
            if not eodhd or not eodhd.api_key:
                logger.debug("options: EODHD API key not configured")
                return None
            
            # Get price data for current price reference
            price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            if price_df is None or 'close' not in price_df.columns:
                logger.debug(f"options: No price data for {symbol}")
                return None
            
            current_price = float(price_df['close'].iloc[-1])
            
            # Fetch options chain (latest available)
            options_df_raw = eodhd.get_options(symbol)
            
            if options_df_raw is None or options_df_raw.empty:
                logger.debug(f"options: No EODHD options data for {symbol}")
                return None
            
            # Extract comprehensive features from options chain
            features = {}
            
            # A. IMPLIED VOLATILITY METRICS
            if 'impliedVolatility' in options_df_raw.columns:
                # ATM IV (strikes closest to current price)
                options_df_raw['moneyness'] = np.abs(options_df_raw['strike'] - current_price)
                atm_options = options_df_raw.nsmallest(5, 'moneyness')
                features['atm_iv'] = atm_options['impliedVolatility'].mean() if len(atm_options) > 0 else np.nan
                
                # Call vs Put IV
                calls = options_df_raw[options_df_raw['type'] == 'calls']
                puts = options_df_raw[options_df_raw['type'] == 'puts']
                
                features['call_iv_avg'] = calls['impliedVolatility'].mean() if len(calls) > 0 else np.nan
                features['put_iv_avg'] = puts['impliedVolatility'].mean() if len(puts) > 0 else np.nan
                features['iv_spread'] = (features['put_iv_avg'] - features['call_iv_avg']) if not pd.isna(features['put_iv_avg']) else np.nan
            
            # B. VOLUME & OPEN INTEREST
            if 'volume' in options_df_raw.columns:
                calls = options_df_raw[options_df_raw['type'] == 'calls']
                puts = options_df_raw[options_df_raw['type'] == 'puts']
                
                features['call_volume'] = calls['volume'].sum() if len(calls) > 0 else 0
                features['put_volume'] = puts['volume'].sum() if len(puts) > 0 else 0
                features['put_call_volume_ratio'] = (features['put_volume'] / features['call_volume']) if features['call_volume'] > 0 else np.nan
            
            if 'openInterest' in options_df_raw.columns:
                calls = options_df_raw[options_df_raw['type'] == 'calls']
                puts = options_df_raw[options_df_raw['type'] == 'puts']
                
                features['call_oi'] = calls['openInterest'].sum() if len(calls) > 0 else 0
                features['put_oi'] = puts['openInterest'].sum() if len(puts) > 0 else 0
                features['put_call_oi_ratio'] = (features['put_oi'] / features['call_oi']) if features['call_oi'] > 0 else np.nan
            
            # C. PRICING METRICS
            for opt_type, prefix in [('calls', 'call'), ('puts', 'put')]:
                type_options = options_df_raw[options_df_raw['type'] == opt_type]
                if len(type_options) > 0:
                    if 'bid' in type_options.columns:
                        features[f'{prefix}_bid_avg'] = type_options['bid'].mean()
                    if 'ask' in type_options.columns:
                        features[f'{prefix}_ask_avg'] = type_options['ask'].mean()
                    if 'last' in type_options.columns:
                        features[f'{prefix}_last_avg'] = type_options['last'].mean()
            
            # Bid-Ask spread
            if 'bid' in options_df_raw.columns and 'ask' in options_df_raw.columns:
                valid_spreads = options_df_raw[(options_df_raw['bid'] > 0) & (options_df_raw['ask'] > 0)]
                if len(valid_spreads) > 0:
                    mid = (valid_spreads['bid'] + valid_spreads['ask']) / 2
                    spread_pct = ((valid_spreads['ask'] - valid_spreads['bid']) / mid) * 100
                    features['bid_ask_spread_pct'] = spread_pct.mean()
            
            # D. MONEYNESS METRICS
            features['strikes_available'] = len(options_df_raw['strike'].unique()) if 'strike' in options_df_raw.columns else 0
            
            if 'strike' in options_df_raw.columns:
                calls = options_df_raw[options_df_raw['type'] == 'calls']
                puts = options_df_raw[options_df_raw['type'] == 'puts']
                
                otm_calls = calls[calls['strike'] > current_price]
                otm_puts = puts[puts['strike'] < current_price]
                
                features['otm_call_pct'] = (len(otm_calls) / len(calls) * 100) if len(calls) > 0 else 0
                features['otm_put_pct'] = (len(otm_puts) / len(puts) * 100) if len(puts) > 0 else 0
            
            # E. TERM STRUCTURE
            if 'expirationDate' in options_df_raw.columns:
                expiry_dates = pd.to_datetime(options_df_raw['expirationDate'].unique())
                features['expiries_available'] = len(expiry_dates)
                
                if len(expiry_dates) > 0:
                    nearest_expiry = expiry_dates.min()
                    features['nearest_expiry_days'] = (nearest_expiry - pd.Timestamp.now()).days
            
            # Convert to DataFrame
            last_price_dt = price_df.index[-1]
            options_df = pd.DataFrame([features], index=[last_price_dt])
            # Family-scoped has_data for downstream validation and model hygiene.
            # IMPORTANT: options are a snapshot-style provider (no true historical chains).
            # We mark only the snapshot date as having real data.
            options_df["has_data"] = 1.0

            # Reindex to full date range WITHOUT propagating the snapshot.
            # This avoids leaking a present-day options chain into historical windows.
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            options_df = options_df.reindex(date_range)

            # Only the snapshot date is considered "real".
            options_df["has_data"] = 0.0
            if last_price_dt in options_df.index:
                options_df.loc[last_price_dt, "has_data"] = 1.0

            # Clean data: keep non-snapshot rows at zero rather than forward/backfilling.
            options_df = options_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            num_features = len([c for c in options_df.columns if not c.startswith('_')])
            
            logger.info(
                f"✅ options: {symbol} - Features={num_features}, Rows={len(options_df)}, "
                f"Strikes={features.get('strikes_available', 0)}"
            )
            
            options_df.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'eodhd_options_api',
                'feature_type': 'raw_options_metrics',
                'num_features': num_features
            }
            options_df.attrs['feature_counts'] = {
                'total': num_features
            }
            
            return options_df
            
        except ImportError:
            logger.debug(f"options: EODHD provider not available for {symbol}")
            return None
        except Exception as e:
            logger.debug(f"options: EODHD failed for {symbol}: {e}")
            return None
    
    def _compute_hf_options_signals(options_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate HF temporal signals from options time series.
        
        Uses TimeSeriesTransformer to detect:
        - Volatility regime changes
        - Gamma regime shifts
        
        Input: 60-180 days of [atm_iv, skew, gamma_exposure, vega, put_call_ratio]
        Output: hf_vol_regime_score, hf_gamma_regime_score, hf_conf
        
        This is OPTIONAL - raw features are sufficient for Stage A.
        Only enable if you want deep temporal alpha signals.
        """
        try:
            # Check if we have minimum required features
            required_features = ['atm_iv', 'skew', 'gamma_exposure', 'put_call_ratio']
            if not all(f in options_df.columns for f in required_features):
                return None
            
            # For now, return placeholder signals
            # TODO: Implement actual HF transformer model
            # This would be a TimeSeriesTransformer trained on:
            # - Input: [atm_iv, skew, gamma_exposure, vega, put_call_ratio, ...]
            # - Output: Regime classification (high_vol, low_vol, transition)
            # - Training: Historical vol regimes labeled by realized vol percentiles
            
            n = len(options_df)
            
            # Placeholder: Simple rolling z-score based regime detection
            if 'atm_iv' in options_df.columns and n >= 60:
                atm_iv_rolling_mean = options_df['atm_iv'].rolling(60, min_periods=30).mean()
                atm_iv_rolling_std = options_df['atm_iv'].rolling(60, min_periods=30).std()
                vol_zscore = (options_df['atm_iv'] - atm_iv_rolling_mean) / (atm_iv_rolling_std + 1e-8)
                
                # Normalize to [-1, 1] range
                vol_regime_score = np.tanh(vol_zscore / 2.0)
                
                # Gamma regime (similar approach)
                gamma_regime_score = np.zeros(n)
                if 'gamma_exposure' in options_df.columns:
                    gamma_rolling_mean = options_df['gamma_exposure'].rolling(60, min_periods=30).mean()
                    gamma_rolling_std = options_df['gamma_exposure'].rolling(60, min_periods=30).std()
                    gamma_zscore = (options_df['gamma_exposure'] - gamma_rolling_mean) / (gamma_rolling_std + 1e-8)
                    gamma_regime_score = np.tanh(gamma_zscore / 2.0)
                
                # Confidence: Higher when signal is stable
                conf = 1.0 - np.abs(vol_zscore.diff().fillna(0)) / (np.abs(vol_zscore) + 1.0)
                conf = conf.clip(0, 1)
                
                return pd.DataFrame({
                    'hf_vol_regime_score': vol_regime_score.fillna(0),
                    'hf_gamma_regime_score': gamma_regime_score,
                    'hf_conf': conf.fillna(0.5)
                }, index=options_df.index)
            
            return None
            
        except Exception as e:
            logger.debug(f"HF options signal computation failed: {e}")
            return None

    def _h_short_interest():
        """
        Short interest features with automatic EODHD/FINRA fetching
        
        Data updates: Bi-monthly from NASDAQ (15th and end of month)
        Forward-fill: Yes (bi-monthly settlement data forward-filled to daily)
        
        Returns time-series DataFrame with all historical short interest reports
        forward-filled to daily frequency for alignment with price data.
        """
        def _short_interest_stub(*, status: str, error: Optional[str] = None) -> pd.DataFrame:
            date_range = pd.date_range(start=start or start_str_default, end=end or end_str_default, freq='D', tz='UTC')
            stub = pd.DataFrame(index=date_range)
            stub["has_data"] = 0.0
            stub["percent"] = 0.0
            stub["ratio"] = 0.0
            stub["days_to_cover"] = 0.0
            stub["float_short_pct"] = 0.0
            stub["short_to_oi_ratio"] = 0.0
            stub["change_1m"] = 0.0
            stub["change_3m"] = 0.0
            stub["zscore_1y"] = 0.0
            stub["pct_zscore_3y"] = 0.0
            stub["momentum"] = 0.0
            stub["squeeze_risk_flag"] = 0.0
            stub["squeeze_risk_score"] = 0.0
            stub["squeeze_probability"] = 0.0
            stub["shares_on_loan_pct"] = 0.0
            stub["borrow_rate"] = 0.0
            stub["borrow_rate_zscore_3y"] = 0.0
            stub["short_vs_institutional"] = 0.0
            stub["confidence"] = 0.0
            stub["conf"] = 0.0
            stub["score_raw"] = 0.0
            stub["score"] = 0.0
            stub.attrs['telemetry'] = {
                'status': status,
                'source': 'ShortInterestAnalyzer',
                'error': error,
            }
            return stub

        if ShortInterestAnalyzer is None:
            logger.warning("short_interest: provider unavailable; emitting has_data=0 stub for %s", symbol)
            return _short_interest_stub(status='dormant:provider_unavailable')

        try:
            analyzer = ShortInterestAnalyzer()

            start_local = start or start_str_default
            end_local = end or end_str_default

            # Get historical short interest data (not just latest snapshot)
            lookback_days = (pd.to_datetime(end_local) - pd.to_datetime(start_local)).days + 180  # Extra buffer
            short_data_list = analyzer.get_short_interest_data(symbol, lookback_days=lookback_days)
            
            if not short_data_list:
                logger.warning("short_interest: No data for %s; emitting has_data=0 stub", symbol)
                return _short_interest_stub(status='dormant:no_data')

            # Guard against lookahead/leakage: if providers return data after our requested end,
            # do not let those future reports influence features (changes/zscores).
            end_ts = pd.to_datetime(end_local, errors="coerce")
            if isinstance(end_ts, pd.Timestamp) and pd.notna(end_ts):
                end_ts = end_ts.normalize()
                in_range = []
                for m in short_data_list:
                    try:
                        m_ts = pd.to_datetime(getattr(m, "report_date", None), errors="coerce")
                        if isinstance(m_ts, pd.Timestamp) and pd.notna(m_ts) and m_ts.normalize() <= end_ts:
                            in_range.append(m)
                    except Exception:
                        continue

                if not in_range:
                    # We had data, but it is entirely in the future relative to the requested window.
                    try:
                        dates = [pd.to_datetime(getattr(m, "report_date", None), errors="coerce") for m in short_data_list]
                        dates = [d for d in dates if isinstance(d, pd.Timestamp) and pd.notna(d)]
                        min_dt = min(dates).date().isoformat() if dates else None
                        max_dt = max(dates).date().isoformat() if dates else None
                    except Exception:
                        min_dt, max_dt = None, None

                    logger.warning(
                        "short_interest: %s has %d reports, but none on/before end=%s (provider range=%s..%s); emitting stub",
                        symbol,
                        len(short_data_list),
                        str(end_ts.date()) if isinstance(end_ts, pd.Timestamp) else str(end_local),
                        str(min_dt),
                        str(max_dt),
                    )
                    return _short_interest_stub(status="dormant:no_data_in_range")

                short_data_list = in_range
            
            # Convert list of metrics to time-series DataFrame
            records = []
            for i, metrics in enumerate(short_data_list):
                # Generate features for each report date
                features = analyzer._metrics_to_features(metrics, short_data_list, i)
                features['date'] = pd.to_datetime(metrics.report_date)
                records.append(features)
            
            # Create DataFrame from all historical reports
            ts_df = pd.DataFrame(records)
            ts_df = ts_df.set_index('date').sort_index()
            
            # Resample to daily frequency and forward-fill
            date_range = pd.date_range(start=start_local, end=end_local, freq='D', tz='UTC')
            ts_df.index = pd.to_datetime(ts_df.index).tz_localize('UTC')

            first_report_ts = ts_df.index.min() if not ts_df.empty else None
            
            # Reindex to daily and forward-fill (short interest persists until next report).
            # DO NOT backfill the leading edge: that fabricates pre-history coverage.
            ts_df = ts_df.reindex(date_range).ffill()

            # If the provider returned structurally-valid rows but all feature values are NaN,
            # treat it as no usable data and emit the deterministic stub. This ensures the
            # family is present in cache-first runs and avoids validation rejecting the frame.
            numeric = ts_df.select_dtypes(include=[np.number])
            if numeric.empty or numeric.isna().all().all():
                logger.warning("short_interest: All values NaN for %s; emitting has_data=0 stub", symbol)
                return _short_interest_stub(status='dormant:no_data')

            # Compute honest presence indicator: data only exists at/after first real report.
            # We treat forward-filled values after first report as "supported" by real data,
            # but we do not claim coverage before the first report.
            has_real = pd.Series(False, index=ts_df.index)
            if isinstance(first_report_ts, pd.Timestamp):
                has_real.loc[has_real.index >= first_report_ts] = True

            # Zero-out the pre-history region for numeric columns.
            pre_mask = ~has_real
            if pre_mask.any():
                numeric_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    ts_df.loc[pre_mask, col] = 0.0

            ts_df["has_data"] = has_real.astype(float)

            # Fill any residual NaNs with 0 for schema stability.
            ts_df = ts_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # DO NOT add family prefix - _collect() will add it
            # Features are already named correctly (percent, days_to_cover, etc.)
            
            if not ts_df.empty:
                logger.info(f"✅ short_interest: {symbol} - {len(ts_df)} days, {len(ts_df.columns)} features")
                ts_df.attrs['telemetry'] = {'status': 'ok', 'source': 'ShortInterestAnalyzer'}
                ts_df.attrs['feature_counts'] = {'generated': len(ts_df.columns)}
                return ts_df
            
            logger.warning("short_interest: Empty frame for %s; emitting has_data=0 stub", symbol)
            return _short_interest_stub(status='dormant:no_data')
        except Exception as e:
            logger.warning("short_interest failed for %s: %s; emitting has_data=0 stub", symbol, e)
            return _short_interest_stub(status='error', error=str(e))

    def _h_subsidiary():
        """
        Subsidiary / Organizational Complexity Family (time-varying)
        
        Uses quarterly R&D and operating expense data as proxies for:
        - Organizational scale and complexity
        - Innovation investment trends
        - Business expansion/contraction
        
        Features (8):
        - rd_spending: Quarterly R&D spending (absolute)
        - opex_spending: Total operating expenses (absolute)
        - rd_intensity: R&D as % of operating expenses
        - rd_qoq_growth: Quarter-over-quarter R&D growth
        - opex_qoq_growth: Quarter-over-quarter OpEx growth
        - rd_yoy_growth: Year-over-year R&D growth
        - opex_yoy_growth: Year-over-year OpEx growth
        - complexity_score: Log(1 + OpEx) - scale-invariant complexity
        
        Data: EODHD Quarterly Income Statement (R&D, OpEx)
        Forward-filled quarterly to daily (legitimate lag - reports quarterly)
        """
        def _subsidiary_stub(*, status: str, error: Optional[str] = None) -> pd.DataFrame:
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            stub = pd.DataFrame(index=date_range)
            stub['subsidiary_has_data'] = 0.0
            stub['subsidiary_activity'] = 0.0
            stub['subsidiary_days_since_update'] = 999.0  # No update
            stub['subsidiary_rd_spending'] = 0.0
            stub['subsidiary_opex_spending'] = 0.0
            stub['subsidiary_rd_intensity'] = 0.0
            stub['subsidiary_rd_qoq_growth'] = 0.0
            stub['subsidiary_opex_qoq_growth'] = 0.0
            stub['subsidiary_rd_yoy_growth'] = 0.0
            stub['subsidiary_opex_yoy_growth'] = 0.0
            stub['subsidiary_complexity_score'] = 0.0
            stub['subsidiary_confidence'] = 0.0
            stub.attrs['telemetry'] = {
                'status': status,
                'source': 'eodhd_quarterly_income_statement',
                'error': error,
            }
            return stub

        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 Subsidiary/Complexity: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                logger.warning("subsidiary: EODHD API key missing; emitting has_data=0 stub for %s", symbol)
                return _subsidiary_stub(status='dormant:missing_api_key')

            fund_data = eodhd_provider.get_fundamentals(symbol)
            
            if not fund_data or 'Financials' not in fund_data:
                logger.debug(f"⚠️ Subsidiary: No fundamentals for {symbol}")
                return _subsidiary_stub(status='dormant:no_data')
            
            # Extract quarterly R&D and OpEx
            income_q = fund_data['Financials'].get('Income_Statement', {}).get('quarterly', {})
            if not income_q:
                logger.debug(f"⚠️ Subsidiary: No quarterly income data for {symbol}")
                return _subsidiary_stub(status='dormant:no_data')
            
            quarterly_rows = []
            for date_str in sorted(income_q.keys()):
                q = income_q[date_str]
                rd = q.get('researchDevelopment')
                opex = q.get('totalOperatingExpenses')
                
                if rd and opex:
                    quarterly_rows.append({
                        'date': pd.to_datetime(date_str),
                        'rd': float(rd),
                        'opex': float(opex)
                    })
            
            if len(quarterly_rows) < 4:
                logger.debug(f"⚠️ Subsidiary: Insufficient quarterly data for {symbol}")
                return _subsidiary_stub(status='dormant:no_data')
            
            df_q = pd.DataFrame(quarterly_rows).set_index('date').sort_index()
            
            # Compute derived metrics
            df_q['rd_intensity'] = (df_q['rd'] / df_q['opex']) * 100  # R&D as % of OpEx
            df_q['complexity_score'] = np.log(1 + df_q['opex'])  # Log-scale complexity
            
            # Growth rates (QoQ and YoY)
            df_q['rd_qoq_growth'] = df_q['rd'].pct_change() * 100
            df_q['opex_qoq_growth'] = df_q['opex'].pct_change() * 100
            df_q['rd_yoy_growth'] = df_q['rd'].pct_change(periods=4) * 100
            df_q['opex_yoy_growth'] = df_q['opex'].pct_change(periods=4) * 100
            
            # Forward-fill quarterly metrics to daily
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            subsidiary_df = pd.DataFrame(index=date_range)
            
            combined_index = df_q.index.union(date_range)
            
            feature_cols = [
                'rd', 'opex', 'rd_intensity', 'rd_qoq_growth', 
                'opex_qoq_growth', 'rd_yoy_growth', 'opex_yoy_growth', 'complexity_score'
            ]
            
            for col in feature_cols:
                if col in df_q.columns:
                    daily_series = df_q[col].reindex(combined_index).ffill().loc[date_range]
                    # Rename to match expected convention
                    new_col_name = f'subsidiary_{col}' if not col.startswith('subsidiary_') else col
                    if col == 'rd':
                        new_col_name = 'subsidiary_rd_spending'
                    elif col == 'opex':
                        new_col_name = 'subsidiary_opex_spending'
                    subsidiary_df[new_col_name] = daily_series
            
            # Cleanup
            subsidiary_df = subsidiary_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            
            logger.info(f"✅ Subsidiary: Built {len(feature_cols)} time-varying complexity features for {symbol}")
            
            subsidiary_df.attrs['provenance'] = {'source': 'eodhd_quarterly_income_statement', 'metrics': feature_cols}
            subsidiary_df.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'type': 'quarterly_opex_rd'}
            return subsidiary_df
            
        except Exception as e:
            logger.warning("subsidiary failed for %s: %s; emitting has_data=0 stub", symbol, e)
            return _subsidiary_stub(status='error', error=str(e))

    def _h_earnings():
        """
        HEDGE-FUND GRADE: Earnings Event & Quality Signals (14-16 features)
        
        PHILOSOPHY: Surprises + consistency + revisions, NOT raw levels
        
        Categories:
        A. Core Earnings (2-4): eps_surprise_pct, revenue_surprise_pct
                                [OPTIONAL: eps_surprise_z, revenue_surprise_z (rolling 3-5y)]
                                REMOVED: eps_actual, eps_estimate, revenue_actual, revenue_estimate
                                (raw values: scale issues, not cross-sectionally comparable)
        
        B. Growth Trends (4): eps_growth_qoq, eps_growth_yoy, revenue_growth_qoq, revenue_growth_yoy
                              (winsorized to ±200% - growth explodes after losses/small bases)
        
        C. Patterns (2): earnings_beat_streak, earnings_miss_streak
                         REMOVED: earnings_volatility_flag (binary → continuous)
                         ADDED: earnings_surprise_volatility (rolling std of surprise_pct)
        
        D. Beat Consistency (2): beat_streak, beat_rate_3y (EXCELLENT - HIGH ALPHA)
                                 (encodes management conservatism, analyst anchoring, execution quality)
        
        E. Revision Dynamics (2-3): revision_breadth, estimate_dispersion (CRITICAL ALPHA)
                                    [OPTIONAL: revision_breadth_change (acceleration matters)]
                                    (early signals, forward-looking, regime-robust)
        
        F. Event Timing (2): days_since_earnings, earnings_event_decay
                             ADDED: earnings_event_decay = exp(-days_since_earnings / τ) where τ≈10-15
                             (post-earnings drift capture: react strongly after, gradually forget)
        
        G. Anticipation (1): days_to_next_earnings (NEW - pairs with cboe_term/correlation vol)
                             (pre-earnings positioning, volatility rises before events)
        
        GOVERNANCE:
        - Raw levels removed: Not cross-sectionally comparable, encourage memorization
        - Growth winsorized: ±200% bounds (protect against small base explosions)
        - Binary flags removed: earnings_volatility_flag → continuous surprise_volatility
        - Event decay added: Exponential forgetting (not treating old earnings as fresh)
        - Anticipation added: Pre-earnings regime detection
        
        Total: 14-16 features (optimal size for earnings family)
        Data Source: EODHD Fundamentals (Earnings + Financials) via EarningsAnalyzer
        """
        def _earnings_stub(*, status: str, error: Optional[str] = None) -> pd.DataFrame:
            date_range = pd.date_range(start=start or start_str_default, end=end or end_str_default, freq='D', tz='UTC')
            stub = pd.DataFrame(index=date_range)
            stub['earnings_has_data'] = 0.0
            # Core earnings (2-4): Keep surprises only, drop raw levels
            # NOTE: These are SHIFTED by 1 day in EarningsAnalyzer to prevent leakage
            stub['earnings_eps_surprise_pct'] = 0.0
            stub['earnings_revenue_surprise_pct'] = 0.0
            # Optional: normalized surprises
            # stub['earnings_eps_surprise_z'] = 0.0
            # stub['earnings_revenue_surprise_z'] = 0.0
            
            # Growth trends (4): winsorized to ±200%
            stub['earnings_eps_growth_qoq'] = 0.0
            stub['earnings_eps_growth_yoy'] = 0.0
            stub['earnings_revenue_growth_qoq'] = 0.0
            stub['earnings_revenue_growth_yoy'] = 0.0
            
            # Patterns (2): continuous only, no binary flags
            stub['earnings_beat_streak'] = 0.0
            stub['earnings_miss_streak'] = 0.0
            stub['earnings_surprise_volatility'] = 0.0  # NEW: replaces volatility_flag
            
            # Beat consistency (2): HIGH ALPHA
            stub['earnings_beat_rate_3y'] = 0.0
            
            # Revisions (2-3): CRITICAL ALPHA
            stub['earnings_revision_breadth'] = 0.0
            stub['earnings_estimate_dispersion'] = 0.0
            # stub['earnings_revision_breadth_change'] = 0.0  # Optional
            
            # Event timing (2): NEW decay feature
            stub['earnings_days_since_earnings'] = 0.0
            stub['earnings_event_decay'] = 0.0  # NEW: exp(-days / tau)
            
            # Anticipation (1): NEW
            stub['earnings_days_to_next_earnings'] = 0.0  # NEW
            
            # Event stress (2): One-sided RISK signals for portfolio overlays
            stub['earnings_pre_event_stress'] = 0.0   # exp(-days_to_next / tau_pre)
            stub['earnings_post_event_stress'] = 0.0  # exp(-days_since / tau_post)
            
            # Governance columns (required for all families)
            stub['earnings_activity'] = 0.0
            stub['earnings_days_since_update'] = 999.0  # No update
            stub.attrs['telemetry'] = {
                'status': status,
                'source': 'EarningsAnalyzer',
                'error': error,
                'hedge_fund_grade': True,
            }
            stub.attrs['governance'] = {
                'raw_levels_removed': ['eps_actual', 'eps_estimate', 'revenue_actual', 'revenue_estimate'],
                'reason': 'Scale issues, not cross-sectionally comparable, encourage memorization',
                'growth_winsorized': '±200%',
                'binary_flags_removed': ['earnings_volatility_flag'],
                'continuous_replacements': ['earnings_surprise_volatility'],
                'event_decay_tau': '10-15 trading days',
                'leakage_prevention': 'Surprise features shifted by 1 day',
                'event_stress_signals': ['pre_event_stress', 'post_event_stress'],
            }
            return stub

        try:
            from src.dcf_lab.earnings_analyzer import get_earnings_analyzer

            analyzer = get_earnings_analyzer()
            start_local = start or start_str_default
            end_local = end or end_str_default

            ts_df = analyzer.get_earnings_timeseries(symbol, start=start_local, end=end_local)
            if ts_df is None or ts_df.empty:
                logger.warning("earnings: No usable time-series for %s; emitting has_data=0 stub", symbol)
                return _earnings_stub(status='dormant:no_data')

            # Enforce a stable earnings schema even when upstream sources omit
            # specific metrics for some symbols.
            canonical_cols = [
                'earnings_has_data',
                'earnings_activity',
                'earnings_days_since_update',
                'earnings_eps_surprise_pct',
                'earnings_revenue_surprise_pct',
                'earnings_eps_growth_qoq',
                'earnings_eps_growth_yoy',
                'earnings_revenue_growth_qoq',
                'earnings_revenue_growth_yoy',
                'earnings_beat_streak',
                'earnings_miss_streak',
                'earnings_surprise_volatility',
                'earnings_beat_rate_3y',
                'earnings_revision_breadth',
                'earnings_estimate_dispersion',
                'earnings_days_since_earnings',
                'earnings_event_decay',
                'earnings_days_to_next_earnings',
                'earnings_pre_event_stress',
                'earnings_post_event_stress',
            ]
            for col in canonical_cols:
                raw = col[len('earnings_'):] if col.startswith('earnings_') else col
                if col in ts_df.columns or raw in ts_df.columns:
                    continue
                ts_df[col] = 0.0

            # DO NOT add family prefix - _collect() will add it
            return ts_df

        except Exception as exc:
            logger.warning("earnings provider failed for %s: %s; emitting has_data=0 stub", symbol, exc)
            return _earnings_stub(status='error', error=str(exc))

    def _h_alternative():
        """
        Alternative Signals Family (42 hedge-fund grade features - refactored Jan 2026)
        
        Categories:
        - Price Anomalies (7): normalized gaps, overnight returns, trend acceleration
        - Volume Anomalies (6): z-scored volume signals, divergences, liquidity stress
        - Earnings/Calendar (6): high-alpha event signals (removed weak calendar features)
        - Free Sentiment (4): news volume changes and z-scores
        - Realized Volatility (6): multiple vol estimators and ratios
        - Cross-Asset (5): SPY/QQQ correlation, sector beta, beta change
        - Intraday Patterns (4): best open/close effects (removed noisy features)
        - Macro Interactions (4): gap×VIX, overnight_z, range_z, beta×VIX
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            from src.features.alternative_signals import (
                extract_price_anomalies,
                extract_volume_anomalies,
                extract_earnings_seasonality,
                extract_free_sentiment,
                extract_realized_volatility,
                extract_cross_asset_relations,
                extract_intraday_anomalies,
                extract_macro_interactions,
            )
            
            # Fetch OHLCV data
            eodhd = get_eodhd_provider()
            price_df = eodhd.get_eod_prices(symbol, start, end)
            
            if price_df is None or price_df.empty:
                logger.debug(f"alternative_signals: No price data for {symbol}")
                return None
            
            # Normalize column names to lowercase
            price_df.columns = price_df.columns.str.lower()
            
            # Create date range for time-series features
            date_range = pd.date_range(start=start, end=end, freq='D')
            
            # Collect all features
            all_features = {}
            
            # 1. Price Anomalies (7 features - normalized)
            price_features = extract_price_anomalies(price_df)
            all_features.update(price_features)
            
            # 2. Volume Anomalies (6 features - z-scored)
            volume_features = extract_volume_anomalies(price_df)
            all_features.update(volume_features)
            
            # 3. Earnings & Calendar (6 features - high-alpha only)
            # Try to fetch earnings dates from EODHD fundamentals
            earnings_dates = None
            try:
                fundamentals = eodhd.get_fundamentals(symbol)
                if fundamentals and 'Earnings' in fundamentals:
                    earnings_history = fundamentals['Earnings'].get('History', {})
                    if earnings_history:
                        # Convert earnings history to DataFrame with dates
                        # Format: {'2025-09-30': {'epsActual': 1.85, ...}, ...}
                        dates_list = []
                        for date_str in earnings_history.keys():
                            try:
                                dates_list.append(pd.to_datetime(date_str))
                            except Exception:
                                continue
                        
                        if dates_list:
                            earnings_dates = pd.DataFrame({'date': sorted(dates_list)})
                            logger.debug(f"alternative_signals: Found {len(dates_list)} earnings dates for {symbol}")
            except Exception as e:
                logger.debug(f"alternative_signals: Could not fetch earnings for {symbol}: {e}")
            
            earnings_features = extract_earnings_seasonality(
                price_df, symbol, earnings_dates
            )
            all_features.update(earnings_features)
            
            # 4. Free Sentiment (4 features - added z-score and change)
            # Note: These may return NaN if APIs unavailable
            sentiment_features = extract_free_sentiment(
                symbol, date_range, eodhd_provider=eodhd
            )
            all_features.update(sentiment_features)
            
            # 5. Realized Volatility (6 features)
            vol_features = extract_realized_volatility(price_df)
            all_features.update(vol_features)
            
            # 6. Cross-Asset Relations (5 features - NEW in refactor)
            # Correlation with SPY/QQQ, sector beta
            cross_asset_features = extract_cross_asset_relations(
                symbol, price_df, eodhd_provider=eodhd
            )
            all_features.update(cross_asset_features)
            
            # 7. Intraday Patterns (4 features - pruned from 8)
            # Will use EODHD intraday API if available, else OHLC proxies
            intraday_features = extract_intraday_anomalies(
                price_df, symbol, eodhd_provider=eodhd
            )
            all_features.update(intraday_features)
            
            # 8. Macro Interactions (4 features - NEW)
            # Fetch VIX and SPY for interaction features
            try:
                # EODHD expects indices to be provided with explicit suffix (e.g., VIX.INDX).
                # Passing '^VIX' would be coerced to '^VIX.US' by get_eod_prices and 404.
                vix_df = eodhd.get_eod_prices('VIX.INDX', start, end)
                if vix_df is not None and not vix_df.empty:
                    vix_df = vix_df.copy()
                    vix_df.columns = vix_df.columns.str.lower()
                    vix_series = vix_df['close'] if 'close' in vix_df.columns else None
                else:
                    vix_series = None
            except Exception:
                vix_series = None
            
            try:
                spy_df = eodhd.get_eod_prices('SPY', start, end)
                if spy_df is not None and not spy_df.empty:
                    spy_df = spy_df.copy()
                    spy_df.columns = spy_df.columns.str.lower()
                    spy_series = spy_df['close'] if 'close' in spy_df.columns else None
                else:
                    spy_series = None
            except Exception:
                spy_series = None
            
            macro_interaction_features = extract_macro_interactions(
                price_df, vix_series=vix_series, spy_series=spy_series
            )
            all_features.update(macro_interaction_features)
            
            # Combine into DataFrame
            features_df = pd.DataFrame(all_features)
            
            if features_df.empty:
                logger.debug(f"alternative_signals: No features generated for {symbol}")
                return None
            
            # Add family prefix to all columns
            features_df.columns = [f'alternative_signals_{col}' for col in features_df.columns]
            
            # Filter to requested date range
            features_df = _safe_filter(features_df, start, end)
            
            if not features_df.empty:
                # Add governance columns - data successfully fetched from EODHD
                features_df['alternative_signals_has_data'] = 1.0
                features_df['alternative_signals_activity'] = 1.0
                features_df['alternative_signals_days_since_update'] = 0.0
                
                logger.info(f"✅ alternative_signals: {symbol} - {len(features_df.columns)} features (hedge-fund grade)")
                features_df.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd_alternative_signals'}
                features_df.attrs['feature_counts'] = {'generated': len(features_df.columns)}
                return features_df
            
        except Exception as exc:
            logger.debug(f"alternative_signals failed for {symbol}: {exc}")
        
        return None

    def _get_quantile_history(cal_start_ts: Optional[pd.Timestamp] = None,
                              cal_end_ts: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Return quantile forecasts covering the requested range (pre-filtered)."""
        nonlocal quantile_forecast_cache, quantile_forecast_history

        if quantile_forecast_history is None:
            # Trigger generator (returns trimmed view but also hydrates quantile_forecast_history)
            _ = _h_quantile_forecaster()

        history = quantile_forecast_history
        if history is None or history.empty:
            return None

        out = history

        def _align_ts(ts: pd.Timestamp) -> pd.Timestamp:
            idx_tz = getattr(out.index, 'tz', None)
            ts_tz = getattr(ts, 'tzinfo', None)
            if idx_tz is None and ts_tz is not None:
                return ts.tz_convert(None)
            if idx_tz is not None and ts_tz is None:
                return ts.tz_localize(idx_tz)
            return ts

        start_bound = _align_ts(cal_start_ts) if cal_start_ts is not None else None
        end_bound = _align_ts(cal_end_ts) if cal_end_ts is not None else None

        if start_bound is None and end_bound is None:
            return out

        if start_bound is None:
            start_bound = out.index.min()
        if end_bound is None:
            end_bound = out.index.max()

        return out.loc[(out.index >= start_bound) & (out.index <= end_bound)].copy()

    def _h_index_constituents():
        """Index membership + add/remove event features from EODHD index fundamentals."""
        if fetch_index_constituents is None:
            return None
        try:
            df = fetch_index_constituents(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("index_constituents: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_insider_form4():
        """SEC Form 4 insider transaction aggregates (filing-date, shifted +1 session)."""
        if fetch_insider_form4 is None:
            return None
        try:
            df = fetch_insider_form4(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("insider_form4: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_econ_events_calendar():
        """Macro economic releases calendar + surprise (EODHD economic-events)."""
        if fetch_econ_events_calendar is None:
            return None
        try:
            df = fetch_econ_events_calendar(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("econ_events_calendar: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_corp_actions_splits():
        """Corporate actions: split pulses + magnitudes + post-split regimes (EODHD splits)."""
        if fetch_corp_actions_splits is None:
            return None
        try:
            df = fetch_corp_actions_splits(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("corp_actions_splits: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_marketcap_history():
        """Historical market cap / float / size dynamics (EODHD historical-market-cap)."""
        if fetch_marketcap_history is None:
            return None
        try:
            df = fetch_marketcap_history(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("marketcap_history: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_exchange_calendar():
        """Deterministic exchange calendar structure (holidays, early closes)."""
        if fetch_exchange_calendar is None:
            return None
        try:
            df = fetch_exchange_calendar(symbol, start or start_str_default, end or end_str_default)
        except Exception as exc:
            logger.debug("exchange_calendar: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_peer_screener_context():
        """Cross-sectional peer ranks/percentiles using cached fin_g6 + metadata."""
        if fetch_peer_screener_context is None:
            return None
        try:
            # This family is cross-symbol and depends on per-symbol caches under the
            # local_cache root. Even when build_panel is invoked with cache_dir=None
            # (cache-bypass mode in prep_families), we still want to read peer caches
            # from the default local cache root.
            if cache_dir is None:
                cache_root = Path(__file__).resolve().parents[2] / "data" / "local_cache"
            else:
                cache_root = Path(cache_dir).parent
            df = fetch_peer_screener_context(
                symbol,
                start or start_str_default,
                end or end_str_default,
                cache_root=cache_root,
                horizon=horizon,
            )
        except Exception as exc:
            logger.debug("peer_screener_context: fetch failed for %s: %s", symbol, exc)
            return None
        return df

    def _h_calibration():
        def _calibration_stub(*, status: str, error: Optional[str] = None) -> pd.DataFrame:
            date_range = pd.date_range(start=start or start_str_default, end=end or end_str_default, freq='D', tz='UTC')
            stub = pd.DataFrame(index=date_range)
            stub['has_data'] = 0.0
            # Mirror the calibration feature schema used elsewhere so downstream
            # caches and panels stay shape-stable even when calibration is unavailable.
            for q in ('05', '10', '25', '50', '75', '90', '95', '99'):
                stub[f"q{q}_coverage"] = 0.0
                stub[f"q{q}_error"] = 0.0
            stub['interval_coverage'] = 0.0
            stub['interval_expected'] = 0.0
            stub['interval_error'] = 0.0
            stub['mean_calibration_error'] = 0.0
            stub['overall_score'] = 0.0
            stub['requires_recalibration'] = 0.0
            stub['sample_size'] = 0.0
            stub['confidence'] = 0.0
            stub.attrs['telemetry'] = {
                'status': status,
                'source': 'ProbabilityCalibrationSystem',
                'error': error,
            }
            return stub

        if ProbabilityCalibrationSystem is None:
            logger.warning("calibration: ProbabilityCalibrationSystem unavailable; emitting has_data=0 stub for %s", symbol)
            return _calibration_stub(status='dormant:provider_unavailable')

        try:
            # For calibration, we need sufficient historical forecasts (6+ months)
            # to evaluate forecast quality with statistical significance (>25 samples)
            cal_lookback = 180  # ~6 months of sessions
            cal_update_every = 21  # update about monthly, then forward-fill
            cal_start = (pd.to_datetime(start or start_str_default) - pd.Timedelta(days=cal_lookback)).strftime('%Y-%m-%d')
            
            logger.info(f"🔍 calibration: fetching {cal_lookback}-day quantile history for evaluation")
            cal_start_ts = pd.to_datetime(cal_start)
            cal_end_ts = pd.to_datetime(end or end_str_default)

            quantile_df = _get_quantile_history(cal_start_ts, cal_end_ts)
            if quantile_df is None or quantile_df.empty:
                logger.debug("calibration: quantile forecasts missing for %s", symbol)
                return _calibration_stub(status='dormant:missing_quantiles')
            quantile_df = quantile_df[[c for c in quantile_df.columns if c.startswith('quantile_forecast_')]]

            # Need extended price data to compute future returns (shift -5 requires 5 days ahead)
            extended_end = (pd.to_datetime(end or end_str_default) + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
            
            price_df = _fetch_price_data(symbol, cal_start, extended_end)
            if price_df is None or 'close' not in price_df.columns:
                return _calibration_stub(status='dormant:missing_prices')

            # Align timezone: quantile_df is tz-aware, price_df might be naive
            if hasattr(quantile_df.index, 'tz') and quantile_df.index.tz is not None:
                if not hasattr(price_df.index, 'tz') or price_df.index.tz is None:
                    price_df.index = price_df.index.tz_localize('UTC')
            
            horizon_local = int(horizon)
            realized_returns = price_df['close'].pct_change(horizon_local).shift(-horizon_local)
            calibration_frame = quantile_df.join(realized_returns.rename('future_return')).dropna()
            if calibration_frame.empty or len(calibration_frame) < 25:
                logger.debug("calibration: insufficient paired samples for %s (got %d, need 25+)", symbol, len(calibration_frame))
                return _calibration_stub(status='dormant:insufficient_samples')

            quantile_cols = [c for c in quantile_df.columns if str(c).lower().startswith('quantile_forecast_q')]
            if not quantile_cols:
                logger.debug("calibration: no quantile keys extracted for %s", symbol)
                return _calibration_stub(status='dormant:no_quantile_keys')

            # Build an as-of (leak-safe) time series by evaluating calibration only on samples
            # whose realized horizon return is known by each as-of date.
            idx = _make_index(start, end)
            price_index = pd.DatetimeIndex(price_df.index).sort_values()

            out = pd.DataFrame(index=idx)
            system = ProbabilityCalibrationSystem()

            computed_any = False
            for i in range(0, len(idx), cal_update_every):
                asof = idx[i]
                pos = int(price_index.searchsorted(asof, side='right') - 1)
                cutoff_pos = pos - horizon_local
                if cutoff_pos < 0:
                    continue
                cutoff = price_index[cutoff_pos]

                eligible = calibration_frame.loc[calibration_frame.index <= cutoff]
                if eligible.empty or len(eligible) < 25:
                    continue

                window = eligible.tail(cal_lookback)

                quantile_forecasts: Dict[str, List[float]] = {}
                for col in quantile_cols:
                    lower = str(col).lower()
                    digits = ''.join(ch for ch in lower.split('quantile_forecast_q')[-1] if ch.isdigit())
                    if not digits:
                        continue
                    key = f"q_{digits.zfill(2)}"
                    quantile_forecasts[key] = window[col].tolist()

                if not quantile_forecasts:
                    continue

                forecast_results = {'quantile_forecasts': quantile_forecasts}
                actual_returns = window['future_return'].tolist()

                result = system.evaluate_forecast_calibration(forecast_results, actual_returns)
                if not isinstance(result, dict) or result.get('error'):
                    continue

                features: Dict[str, float] = {}
                features.update(_numeric_dict(result.get('calibration_metrics')))
                features.update(_numeric_dict(result.get('calibration_quality')))
                features['requires_recalibration'] = float(bool(result.get('requires_recalibration')))
                sample_size = float(result.get('sample_size', len(actual_returns) or 0))
                features['sample_size'] = sample_size
                # Confidence isn't provided by ProbabilityCalibrationSystem; derive a simple sample-size proxy.
                features['confidence'] = float(np.clip(sample_size / 500.0, 0.0, 1.0))
                # Use calibration_has_data (not has_data) to match governance expectations
                features['calibration_has_data'] = 1.0

                sanitized = _sanitize_feature_keys('calibration', features)
                if not sanitized:
                    continue

                for k, v in sanitized.items():
                    out.at[asof, k] = v
                computed_any = True

            if not computed_any:
                return _calibration_stub(status='dormant:no_features')

            # Preserve calibration_has_data flag before ffill (don't zero it out)
            has_data_col = out.get('calibration_has_data')
            if has_data_col is not None:
                has_data_preserved = has_data_col.copy()
            else:
                has_data_preserved = None

            out = out.ffill().fillna(0.0)
            
            # Restore has_data flag (ffill forward, but don't zero where it was originally set)
            if has_data_preserved is not None:
                out['calibration_has_data'] = has_data_preserved.ffill().fillna(0.0)
            out.attrs['telemetry'] = {'status': 'ok', 'source': 'ProbabilityCalibrationSystem'}
            out.attrs['feature_counts'] = {'generated': len([c for c in out.columns])}
            
            # Save materialization date for post-maturity flag computation
            mat_date_file = Path(cache_dir or Path.cwd()) / 'event_logs' / f'{symbol}_h{horizon}_calibration_materialization.json'
            mat_date_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(mat_date_file, 'w') as f:
                    json.dump({'last_materialization': datetime.now().isoformat()}, f)
            except Exception as e:
                logger.warning(f"Failed to save calibration materialization date: {e}")
            
            return out
        except Exception as exc:
            logger.warning("calibration provider failed for %s: %s; emitting has_data=0 stub", symbol, exc)
            return _calibration_stub(status='error', error=str(exc))

    def _h_online_learning():
        """
        Online learning metrics with MAX ALPHA features from quantile + calibration signals.
        
        IMPORTANT ARCHITECTURAL NOTE:
        ────────────────────────────────────────────────────────────────────────
        This generator runs at PREP TIME (before Phase-2) and uses quantile-based
        calibration because Mamba predictions don't exist yet.
        
        The features generated here are used AS FEATURES in the panel, not as the
        authoritative learning gate. The AUTHORITATIVE calibration for learning
        decisions is the Mamba-based MambaCalibrationTracker in phase2_stateful.py,
        which tracks Mamba (μ, σ²) predictions vs realized returns at RUNTIME.
        
        Hierarchy:
        - PRIMARY (70-80%): Mamba (μ, σ²) vs realized returns (Phase-2 runtime)
        - SECONDARY (20-30%): Quantile forecasts (this generator, prep time)
        ────────────────────────────────────────────────────────────────────────
        """
        if create_online_learning_system is None:
            logger.debug("online_learning: provider unavailable for %s", symbol)
            return None

        quantile_df = _h_quantile_forecaster()
        if quantile_df is None or quantile_df.empty:
            logger.debug("online_learning: missing quantile forecasts for %s", symbol)
            return None
        
        # Get calibration metrics for enhanced features (quantile-based at prep time)
        calibration_df = _h_calibration()
        has_calibration = calibration_df is not None and not calibration_df.empty

        price_df = _load_price_history()
        if price_df is None or 'close' not in price_df.columns:
            return None

        # Align timezone across all indices before any joins.
        # quantile_df is typically tz-aware (UTC); price_df/calibration_df can be naive.
        q_tz = getattr(quantile_df.index, 'tz', None)
        if q_tz is not None:
            if getattr(price_df.index, 'tz', None) is None:
                price_df.index = price_df.index.tz_localize(q_tz)
            if has_calibration and getattr(calibration_df.index, 'tz', None) is None:
                calibration_df.index = calibration_df.index.tz_localize(q_tz)
        else:
            if getattr(price_df.index, 'tz', None) is not None:
                price_df.index = price_df.index.tz_convert(None)
            if has_calibration and getattr(calibration_df.index, 'tz', None) is not None:
                calibration_df.index = calibration_df.index.tz_convert(None)

        horizon_local = int(horizon)
        future_returns = price_df['close'].pct_change(horizon_local).shift(-horizon_local).rename('future_return')
        merged = quantile_df.join(future_returns)
        
        # Join with calibration if available
        if has_calibration:
            # Use suffixes to avoid column name conflicts (e.g., both have drift_flag)
            # Calibration gets '_cal' suffix, quantile/online_learning keep original names
            merged = merged.join(calibration_df, how='left', rsuffix='_cal')
        
        # CRITICAL FIX: Only drop rows where future_return is NaN (last `horizon_local` days)
        # Keep rows where quantile features exist even if calibration is missing
        # This preserves full date coverage from quantile_forecast
        merged = merged.dropna(subset=['future_return'])
        
        quantile_cols = [c for c in merged.columns if c.startswith('quantile_forecast_')]

        # Reduced from 40 to 20 to support short validation windows (63-day step = ~42 trading days)
        if not quantile_cols or len(merged) < 20:
            logger.debug("online_learning: insufficient samples for %s (need >=20, got %d)", symbol, len(merged))
            return None

        system = create_online_learning_system(
            enable_drift_detection=True,
            drift_sensitivity=0.05,
            update_frequency=5,
        )
        
        # Configure event logging
        event_log_dir = Path(cache_dir or Path.cwd()) / 'event_logs'
        system.event_log_path = event_log_dir / f'{symbol}_h{horizon}_online_learning_events.jsonl'
        system.symbol = symbol
        system.horizon = horizon

        snapshots: List[Dict[str, float]] = []
        median_col = next((c for c in quantile_cols if c.endswith('q50')), quantile_cols[0])

        for idx, row in merged.iterrows():
            # Set current data timestamp for event logging
            system.current_timestamp = idx
            
            features = np.asarray(row[quantile_cols].values, dtype=float)
            target = float(row['future_return'])
            prediction = float(row.get('quantile_forecast_expected_return', row[median_col]))

            # ─────────────────────────────────────────────────────────────────
            # Extract calibration quality for learning gate
            # Use overall_score, confidence, or summary_score as fallbacks
            # ─────────────────────────────────────────────────────────────────
            calibration_quality = None
            if has_calibration:
                for score_key in ['overall_score', 'calibration_overall_score', 'confidence', 
                                  'calibration_confidence', 'summary_score', 'calibration_summary_score']:
                    if score_key in row.index:
                        val = row[score_key]
                        if pd.notna(val) and np.isfinite(float(val)):
                            calibration_quality = float(val)
                            break

            try:
                adaptation = system.add_sample(
                    features=features, 
                    target=target, 
                    prediction=prediction,
                    calibration_quality=calibration_quality,
                )
            except Exception as exc:
                logger.debug("online_learning: provider update failed for %s at %s: %s", symbol, idx, exc)
                continue

            # ===================================================================
            # COMPUTE MAX ALPHA FEATURES
            # ===================================================================
            # Extract quantile predictions from row
            quantile_predictions = {}
            for col in quantile_cols:
                # Parse quantile level from column name (e.g., 'quantile_forecast_q50' -> 'q50')
                if '_q' in col:
                    q_key = col.split('_q')[-1]  # Get '50', '25', etc.
                    quantile_predictions[f'q{q_key}'] = float(row[col])
            
            # Extract calibration metrics if available
            calibration_metrics = {}
            if has_calibration:
                for cal_col in calibration_df.columns:
                    if cal_col in row.index:
                        calibration_metrics[cal_col] = float(row[cal_col])
            
            # Compute max alpha features
            try:
                alpha_features = system.compute_max_alpha_features(
                    quantile_predictions=quantile_predictions,
                    actual_return=target,
                    calibration_metrics=calibration_metrics if calibration_metrics else None
                )
            except Exception as exc:
                logger.debug("online_learning: max alpha feature computation failed for %s at %s: %s", symbol, idx, exc)
                alpha_features = {}

            # Original performance metrics
            perf = system.current_performance
            snapshot = {
                'timestamp': idx,
                'online_learning_mae': float(perf.mae) if np.isfinite(perf.mae) else np.nan,
                'online_learning_rmse': float(perf.rmse) if np.isfinite(perf.rmse) else np.nan,
                'online_learning_mape': float(perf.mape) if np.isfinite(perf.mape) else np.nan,
                'online_learning_direction_accuracy': float(perf.direction_accuracy),
                'online_learning_samples': float(perf.samples_count),
                'online_learning_drift_events': float(perf.total_drifts_detected),
                'online_learning_incremental_updates': float(perf.incremental_updates),
                'online_learning_partial_retrains': float(perf.partial_retrains),
                'online_learning_drift_flag': float(bool(adaptation.get('drift_detected'))),
                'online_learning_partial_retrain_flag': float(bool(adaptation.get('partial_retrain'))),
                'online_learning_model_updated_flag': float(bool(adaptation.get('model_updated'))),
            }
            
            # Add MAX ALPHA features with prefix
            for key, value in alpha_features.items():
                snapshot[f'online_learning_{key}'] = float(value)
            
            snapshots.append(snapshot)

        if not snapshots:
            return None

        online_df = pd.DataFrame.from_records(snapshots).set_index('timestamp')
        online_df.index.name = 'date'  # Rename to 'date' for compatibility with prep_families
        
        # ========================================================================
        # DATE FILTERING WITH FULL COVERAGE PRESERVATION
        # ========================================================================
        requested_start = pd.to_datetime(start or start_str_default)
        requested_end = pd.to_datetime(end or end_str_default)

        # If online_df is tz-aware, localize the requested bounds to the same tz so
        # comparisons don't error.
        o_tz = getattr(online_df.index, 'tz', None)
        if o_tz is not None:
            if getattr(requested_start, 'tzinfo', None) is None:
                requested_start = requested_start.tz_localize(o_tz)
            if getattr(requested_end, 'tzinfo', None) is None:
                requested_end = requested_end.tz_localize(o_tz)
        
        online_df = online_df[
            (online_df.index >= requested_start) &
            (online_df.index <= requested_end)
        ]

        if online_df.empty:
            return None
        
        # CRITICAL FIX: Ensure full date coverage by padding missing boundary dates
        # This handles cases where quantile_forecast or price data had gaps
        actual_start = online_df.index.min()
        actual_end = online_df.index.max()
        
        # Add padding at start if needed
        if actual_start > requested_start and price_df is not None:
            price_dates = price_df.loc[
                (price_df.index >= requested_start) & 
                (price_df.index < actual_start)
            ].index
            
            if len(price_dates) > 0:
                logger.info(f"🔧 online_learning: Adding {len(price_dates)} padding rows at start (requested: {requested_start}, actual: {actual_start})")
                first_row_values = online_df.iloc[0].values
                padding_data = np.tile(first_row_values, (len(price_dates), 1))
                padding_df = pd.DataFrame(
                    index=price_dates,
                    columns=online_df.columns,
                    data=padding_data
                )
                online_df = pd.concat([padding_df, online_df]).sort_index()
        
        # Add padding at end if needed
        if actual_end < requested_end and price_df is not None:
            price_dates = price_df.loc[
                (price_df.index > actual_end) & 
                (price_df.index <= requested_end)
            ].index
            
            if len(price_dates) > 0:
                logger.info(f"🔧 online_learning: Adding {len(price_dates)} padding rows at end (requested: {requested_end}, actual: {actual_end})")
                last_row_values = online_df.iloc[-1].values
                padding_data = np.tile(last_row_values, (len(price_dates), 1))
                padding_df = pd.DataFrame(
                    index=price_dates,
                    columns=online_df.columns,
                    data=padding_data
                )
                online_df = pd.concat([online_df, padding_df]).sort_index()

        online_df = online_df.sort_index().ffill()
        
        # Add has_data flag to indicate real data (not a stub)
        online_df['has_data'] = 1.0
        
        online_df.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'OnlineLearningSystem_MaxAlpha',
            'samples': len(online_df),
            'alpha_features_enabled': True,
        }
        online_df.attrs['feature_counts'] = {'generated': len(online_df.columns)}
        
        # Save materialization date for post-maturity flag computation
        mat_date_file = Path(cache_dir or Path.cwd()) / 'event_logs' / f'{symbol}_h{horizon}_online_learning_materialization.json'
        mat_date_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(mat_date_file, 'w') as f:
                json.dump({'last_materialization': datetime.now().isoformat()}, f)
        except Exception as e:
            logger.warning(f"Failed to save materialization date: {e}")
        
        return online_df

    def _h_news_sentiment():
        """News sentiment features sourced from the NewsSentimentForecaster provider."""
        if FinBERTSentimentAnalyzer is None or NewsSentimentForecaster is None:
            logger.debug("news_sentiment: provider unavailable for %s", symbol)
            return None

        try:
            analyzer = FinBERTSentimentAnalyzer()
            forecaster = NewsSentimentForecaster(sentiment_analyzer=analyzer)
            analysis = forecaster.analyze_ticker_sentiment(symbol, days_back=30)
        except Exception as exc:
            logger.debug("news_sentiment: provider failed for %s: %s", symbol, exc)
            return None

        if not isinstance(analysis, dict):
            return None

        features: Dict[str, float] = {}
        features['aggregated_sentiment'] = float(analysis.get('aggregated_sentiment', 0.0))
        features['sentiment_confidence'] = float(analysis.get('sentiment_confidence', 0.0))
        features['sentiment_momentum'] = float(analysis.get('sentiment_momentum', 0.0))
        features['articles_analyzed'] = float(analysis.get('articles_analyzed', 0))

        distribution = analysis.get('sentiment_distribution', {})
        if isinstance(distribution, dict):
            for key, value in distribution.items():
                num = _coerce_numeric(value)
                if num is not None:
                    features[f'sentiment_distribution_{key}'] = num

        impact = analysis.get('forecast_impact')
        if isinstance(impact, dict):
            features['sentiment_price_adjustment'] = _coerce_numeric(impact.get('price_adjustment')) or 0.0
            features['sentiment_volatility_adjustment'] = _coerce_numeric(impact.get('volatility_adjustment')) or 0.0
            features['sentiment_impact_confidence'] = _coerce_numeric(impact.get('confidence')) or 0.0
            features['sentiment_supporting_articles'] = _coerce_numeric(impact.get('supporting_articles')) or 0.0
        elif impact is not None:
            features['sentiment_price_adjustment'] = _coerce_numeric(getattr(impact, 'price_adjustment', None)) or 0.0
            features['sentiment_volatility_adjustment'] = _coerce_numeric(getattr(impact, 'volatility_adjustment', None)) or 0.0
            features['sentiment_impact_confidence'] = _coerce_numeric(getattr(impact, 'confidence', None)) or 0.0
            features['sentiment_supporting_articles'] = _coerce_numeric(getattr(impact, 'supporting_articles', None)) or 0.0

        numeric = _numeric_dict(features)
        if not numeric:
            return None

        return _feature_frame_from_dict(
            'news_sentiment',
            numeric,
            start,
            end,
            'NewsSentimentForecaster',
            provenance={'articles_analyzed': analysis.get('articles_analyzed', 0)}
        )

    def _h_ml_framework():
        """Technical-feature bundle sourced entirely from EODHD (no local derivation)."""
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider  # type: ignore
        except Exception:
            logger.debug("ml_framework: EODHD provider unavailable for %s", symbol)
            return None

        start_str = start or start_str_default
        end_str = end or end_str_default

        eodhd = get_eodhd_provider()
        try:
            features_df = eodhd.get_ml_framework_features(
                symbol,
                start_date=start_str,
                end_date=end_str,
                lookback_days=365,
            )
        except Exception as exc:
            logger.debug("ml_framework: EODHD indicator fetch failed for %s: %s", symbol, exc)
            return None

        if features_df is None or features_df.empty:
            return None

        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Add governance columns - required for all families
        features_df['ml_framework_has_data'] = 1.0
        features_df['ml_framework_activity'] = 1.0
        features_df['ml_framework_days_since_update'] = 0.0

        features_df.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'EODHD.technical',
            'generated_features': int(len(features_df.columns)),
            'lookback_days': 365,
        }
        features_df.attrs['feature_counts'] = {'generated': int(len(features_df.columns))}
        return features_df

    def _h_microstructure():
        """Microstructure-lite features from intraday data (AlphaVantage or daily proxies)."""
        def _microstructure_stub(*, status: str, error: Optional[str] = None, source: str = "eodhd_microstructure_stub") -> pd.DataFrame:
            idx = date_index
            stub = pd.DataFrame(index=idx)
            # Mirror the daily-proxy schema so downstream HF blocks have stable columns.
            for col in (
                "micro_true_range",
                "micro_atr_ratio",
                "micro_range_pct",
                "micro_body_pct",
                "micro_wick_top",
                "micro_wick_bottom",
                "micro_shadow_ratio",
                "micro_range_scaled",
                "micro_volume_zscore",
                "micro_volume_surge",
                "micro_volume_liquidity",
                "micro_turnover",
                "micro_amihud",
                "micro_hl_volume_corr",
                "micro_ofi_proxy",
                "micro_signed_volume",
                "micro_pressure_proxy",
                "micro_demand_supply_ratio",
                "micro_liquidity_imbalance",
                "micro_impact_ratio",
                "micro_impact_volatility",
                "micro_spread_proxy",
                "micro_vol_of_vol",
                "micro_intraday_vol_proxy",
                "micro_stale_tick",
                "micro_zero_range_flag",
                "micro_low_liquidity_flag",
                "micro_overnight_gap",
                "micro_overnight_vol",
                "micro_intraday_vs_overnight_vol",
                "micro_gap_direction",
                "micro_quote_bid_price",
                "micro_quote_ask_price",
                "micro_quote_bid_size",
                "micro_quote_ask_size",
                "micro_quote_mid_price",
                "micro_quote_spread_abs",
                "micro_quote_spread_bps",
                "micro_quote_imbalance",
                "micro_quote_has_data",
            ):
                stub[col] = 0.0

            stub["has_data"] = 0.0
            stub.attrs["telemetry"] = {
                "status": status,
                "source": source,
                "error": error,
                "generated_features": int(len(stub.columns)),
            }
            stub.attrs["feature_counts"] = {"generated": int(len(stub.columns))}
            stub.attrs["provenance"] = {"source": source, "method": "stub"}
            return stub

        strict_required = _strict_sources_enabled() or _require_all_families_enabled()

        if fetch_micro is None:
            if strict_required:
                raise RuntimeError("microstructure provider unavailable")
            logger.warning("microstructure: provider unavailable; emitting stub for %s", symbol)
            return _microstructure_stub(status="dormant:provider_unavailable")

        try:
            # Fetch microstructure features (handles API fallback internally)
            micro_df = fetch_micro(symbol, start or start_str_default, end or end_str_default)
            
            if micro_df is None or micro_df.empty:
                if strict_required:
                    raise RuntimeError("microstructure produced no data")
                logger.warning("microstructure: no data; emitting stub for %s", symbol)
                return _microstructure_stub(status="dormant:no_data")

            # Ensure index is datetime
            if not isinstance(micro_df.index, pd.DatetimeIndex):
                micro_df.index = pd.to_datetime(micro_df.index)
            # Normalize timezone to naive to avoid compare errors during filtering.
            try:
                if getattr(micro_df.index, "tz", None) is not None:
                    micro_df.index = micro_df.index.tz_localize(None)
            except Exception:
                pass
            
            # Filter to requested date range
            start_ts = pd.Timestamp(start or start_str_default)
            end_ts = pd.Timestamp(end or end_str_default)
            if getattr(start_ts, "tz", None) is not None:
                start_ts = start_ts.tz_localize(None)
            if getattr(end_ts, "tz", None) is not None:
                end_ts = end_ts.tz_localize(None)
            micro_df = micro_df[(micro_df.index >= start_ts) & (micro_df.index <= end_ts)]
            
            if micro_df.empty:
                if strict_required:
                    raise RuntimeError("microstructure empty after date filtering")
                return _microstructure_stub(status="dormant:out_of_range")

            # Clean infinities and NaNs
            micro_df = micro_df.replace([np.inf, -np.inf], np.nan).ffill().dropna(how='all')
            
            if micro_df.empty:
                if strict_required:
                    raise RuntimeError("microstructure all-NaN after cleaning")
                return _microstructure_stub(status="dormant:all_nan")

            # Preserve provenance from fetch_micro if available
            if not hasattr(micro_df, 'attrs') or 'telemetry' not in micro_df.attrs:
                micro_df.attrs['telemetry'] = {
                    'status': 'ok',
                    'source': 'microstructure_intraday',
                    'generated_features': len(micro_df.columns)
                }
            if not hasattr(micro_df, 'attrs') or 'feature_counts' not in micro_df.attrs:
                micro_df.attrs['feature_counts'] = {'generated': len(micro_df.columns)}
            
            return micro_df
            
        except Exception as exc:
            if strict_required:
                raise
            logger.warning("microstructure failed for %s: %s; emitting stub", symbol, exc)
            return _microstructure_stub(status="error", error=str(exc))

    def _h_event_time_bars():
        """Event-time bar features (dollar bars, volatility bars) from intraday data.
        
        Event-time bars normalize information arrival rate, making each bar
        approximately equally informative for sequence modeling.
        
        Columns generated:
        - event_time_n_dollar_bars: Number of dollar bars in the day
        - event_time_dollar_avg_duration_sec: Average bar duration
        - event_time_dollar_duration_cv: Coefficient of variation of durations
        - event_time_dollar_arrival_rate: Bars per trading hour
        - event_time_n_vol_bars: Number of volatility bars
        - event_time_vol_avg_duration_sec: Average volatility bar duration
        - event_time_vol_arrival_rate: Vol bars per trading hour
        - event_time_dollar_vs_vol_ratio: Ratio of dollar to vol bars
        - etc.
        """
        def _event_time_stub(*, status: str, error: Optional[str] = None) -> pd.DataFrame:
            idx = date_index
            stub = pd.DataFrame(index=idx)
            # Stub columns matching the actual feature schema
            for col in (
                "event_time_n_dollar_bars",
                "event_time_dollar_avg_duration_sec",
                "event_time_dollar_duration_std",
                "event_time_dollar_duration_cv",
                "event_time_dollar_max_bar_size",
                "event_time_dollar_bar_size_skew",
                "event_time_dollar_arrival_rate",
                "event_time_dollar_avg_ticks_per_bar",
                "event_time_n_vol_bars",
                "event_time_vol_avg_duration_sec",
                "event_time_vol_duration_std",
                "event_time_vol_arrival_rate",
                "event_time_vol_avg_abs_return",
                "event_time_dollar_vs_vol_ratio",
                "event_time_has_data",
                "event_time_dollar_threshold",
                "event_time_vol_threshold",
            ):
                stub[col] = np.nan
            stub["event_time_has_data"] = 0.0
            stub.attrs["telemetry"] = {"status": status, "error": error}
            stub.attrs["feature_counts"] = {"generated": 0}
            return stub

        if generate_event_time_features_range is None:
            return _event_time_stub(status="dormant:module_unavailable")

        try:
            # Parse date range
            start_ts = pd.Timestamp(start or start_str_default)
            end_ts = pd.Timestamp(end or end_str_default)
            
            # Use default config (can be overridden via environment)
            k_dollar = float(os.getenv("EVENT_BAR_K_DOLLAR", "1.0"))
            k_vol = float(os.getenv("EVENT_BAR_K_VOL", "1.0"))
            
            config = EventBarConfig(
                k_dollar=k_dollar,
                k_vol=k_vol,
                threshold_lookback_days=20,
            )
            
            # Generate event-time features
            features_df = generate_event_time_features_range(
                symbol=symbol,
                start_date=start_ts,
                end_date=end_ts,
                config=config,
            )
            
            if features_df is None or features_df.empty:
                logger.debug("event_time_bars: No data for %s", symbol)
                return _event_time_stub(status="dormant:no_data")
            
            # Ensure index is DatetimeIndex
            if not isinstance(features_df.index, pd.DatetimeIndex):
                features_df.index = pd.to_datetime(features_df.index)
            
            # Prefix columns with family name
            features_df.columns = [f"event_time_bars_{col}" if not col.startswith("event_time_") else col.replace("event_time_", "event_time_bars_") for col in features_df.columns]
            
            # Add governance columns
            features_df["event_time_bars_has_data"] = 1.0
            features_df["event_time_bars_activity"] = 1.0
            features_df["event_time_bars_days_since_update"] = 0.0
            
            features_df.attrs["telemetry"] = {
                "status": "ok",
                "source": "event_time_bars",
                "generated_features": len(features_df.columns),
            }
            features_df.attrs["feature_counts"] = {"generated": len(features_df.columns)}
            
            return features_df
            
        except Exception as exc:
            if strict_required:
                raise
            logger.warning("event_time_bars failed for %s: %s", symbol, exc)
            return _event_time_stub(status="error", error=str(exc))

    def _h_candle_mechanics():
        """Daily OHLCV-derived candle mechanics (scale-free, leak-safe)."""
        if fetch_candle_mechanics is None:
            return None
        price_df = _load_price_history()
        if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
            return None
        try:
            return fetch_candle_mechanics(symbol, start or start_str_default, end or end_str_default, price_df=price_df)
        except Exception:
            logger.exception("candle_mechanics failed for %s", symbol)
            return None

    def _h_drift_monitor():
        """Monitor feature drift across all families using real feature data (FULL TIME SERIES)."""
        try:
            # Monitor ALL families except drift_monitor itself
            # This ensures we track drift across the entire feature space
            available_families = [f for f in fams if f != 'drift_monitor']
            
            if not available_families:
                logger.debug("drift_monitor: No families available for monitoring")
                return None
            
            # Collect feature data from each family
            family_data = {}
            for fam in available_families:
                handler = handlers.get(fam)
                if handler is None:
                    continue
                try:
                    df = handler()
                    df = _safe_filter(df, start, end)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        family_data[fam] = df
                except Exception as e:
                    logger.debug(f"drift_monitor: Failed to load {fam}: {e}")
                    continue
            
            if not family_data:
                logger.debug("drift_monitor: No valid family data collected")
                return None
            
            # Merge all features into single DataFrame
            merged_features = None
            for fam, df in family_data.items():
                # Prefix columns with family name to avoid conflicts
                df_prefixed = df.copy()
                df_prefixed.columns = [f"{fam}_{col}" for col in df.columns]
                
                # Remove timezone if present (ensure all indexes are tz-naive)
                if df_prefixed.index.tz is not None:
                    df_prefixed.index = df_prefixed.index.tz_localize(None)
                
                if merged_features is None:
                    merged_features = df_prefixed
                else:
                    # Ensure merged_features index is also tz-naive
                    if merged_features.index.tz is not None:
                        merged_features.index = merged_features.index.tz_localize(None)
                    merged_features = merged_features.join(df_prefixed, how='outer')
            
            if merged_features is None or merged_features.empty:
                return None
            
            # Fill NaN with forward-fill then backward-fill (preserve temporal order)
            merged_features = merged_features.ffill().bfill()
            
            # Drop any remaining NaN columns
            merged_features = merged_features.dropna(axis=1, how='all')
            
            if merged_features.empty or len(merged_features.columns) < 2:
                logger.debug("drift_monitor: Insufficient feature columns after cleaning")
                return None
            
            # Use FeatureDriftMonitor if available, otherwise fall back to statistical metrics
            if FeatureDriftMonitor is not None:
                # Initialize monitor with feature names
                monitor = FeatureDriftMonitor(
                    feature_names=list(merged_features.columns),
                    correlation_threshold=0.3,
                    distribution_threshold=0.15,
                    window_size=min(100, len(merged_features) // 2)
                )
                
                # Track drift metrics over time
                drift_metrics = []
                
                for idx, row in merged_features.iterrows():
                    feature_dict = row.to_dict()
                    # Remove any NaN values
                    feature_dict = {k: v for k, v in feature_dict.items() if not pd.isna(v)}
                    
                    if not feature_dict:
                        continue
                    
                    # Add features to monitor (returns list of alerts)
                    alerts = monitor.add_features(feature_dict, timestamp=idx)
                    
                    # Track drift severity as numeric score
                    if alerts:
                        max_severity = max([
                            {'none': 0, 'minor': 1, 'moderate': 2, 'major': 3, 'critical': 4}[a.severity.value]
                            for a in alerts
                        ])
                        max_metric = max([a.metric_value for a in alerts])
                        max_confidence = max([a.confidence for a in alerts])
                    else:
                        max_severity = 0
                        max_metric = 0.0
                        max_confidence = 0.0
                    
                    drift_metrics.append({
                        'timestamp': idx,
                        'drift_severity': max_severity,
                        'drift_metric_value': max_metric,
                        'drift_confidence': max_confidence,
                        'n_alerts': len(alerts),
                        'n_features': len(feature_dict)
                    })
                
                if not drift_metrics:
                    return None
                
                # Convert to DataFrame
                drift_df = pd.DataFrame(drift_metrics)
                drift_df.set_index('timestamp', inplace=True)
                
                drift_df.attrs['telemetry'] = {
                    'status': 'ok', 
                    'source': 'FeatureDriftMonitor',
                    'families_monitored': list(family_data.keys()),
                    'n_features': len(merged_features.columns)
                }
                drift_df.attrs['feature_counts'] = {'generated': len(drift_df.columns)}
                
                return drift_df
                
            else:
                # Fallback: Statistical drift metrics without monitor class
                logger.debug("drift_monitor: FeatureDriftMonitor not available, using statistical fallback")
                
                # Calculate rolling feature statistics
                window = min(60, len(merged_features) // 3)
                
                drift_metrics = []
                for i in range(len(merged_features)):
                    if i < window:
                        # Not enough history
                        drift_metrics.append({
                            'feature_mean_change': 0.0,
                            'feature_std_change': 0.0,
                            'feature_correlation_change': 0.0,
                            'n_features': len(merged_features.columns)
                        })
                    else:
                        # Compare recent vs historical features
                        hist = merged_features.iloc[max(0, i-window):i]
                        recent = merged_features.iloc[i:i+1]
                        
                        # Mean shift
                        hist_means = hist.mean()
                        recent_means = recent.iloc[0]
                        mean_changes = np.abs((recent_means - hist_means) / (hist.std() + 1e-6))
                        avg_mean_change = float(mean_changes.mean())
                        
                        # Volatility change
                        hist_std = hist.std()
                        recent_window = merged_features.iloc[max(0, i-10):i+1]
                        recent_std = recent_window.std()
                        std_changes = np.abs(recent_std - hist_std) / (hist_std + 1e-6)
                        avg_std_change = float(std_changes.mean())
                        
                        # Correlation structure change
                        if len(merged_features.columns) >= 2:
                            hist_corr = hist.corr().values
                            recent_corr = recent_window.corr().values
                            corr_diff = np.abs(hist_corr - recent_corr)
                            avg_corr_change = float(np.mean(corr_diff[np.triu_indices(len(corr_diff), k=1)]))
                        else:
                            avg_corr_change = 0.0
                        
                        drift_metrics.append({
                            'feature_mean_change': avg_mean_change,
                            'feature_std_change': avg_std_change,
                            'feature_correlation_change': avg_corr_change,
                            'n_features': len(merged_features.columns)
                        })
                
                drift_df = pd.DataFrame(drift_metrics, index=merged_features.index)
                
                # Filter to requested date range
                drift_df = drift_df[
                    (drift_df.index >= pd.Timestamp(start)) & 
                    (drift_df.index <= pd.Timestamp(end))
                ]
                
                drift_df.attrs['telemetry'] = {
                    'status': 'ok', 
                    'source': 'StatisticalDriftFallback',
                    'proxy': True,
                    'families_monitored': list(family_data.keys()),
                    'n_features': len(merged_features.columns)
                }
                drift_df.attrs['feature_counts'] = {'generated': len(drift_df.columns)}
                
                return drift_df
                
        except Exception as exc:
            logger.debug("drift_monitor failed for %s: %s", symbol, exc)
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _h_quantile_forecaster():
        """Quantile forecasts using the GBMQuantileForecaster provider."""
        nonlocal quantile_forecast_cache, quantile_forecast_history
        if quantile_forecast_cache is not None:
            logger.info(f"✅ quantile_forecast: using cached result for {symbol}")
            return quantile_forecast_cache

        if ForecastConfig is None or GBMQuantileForecaster is None:
            logger.warning(f"❌ quantile_forecast: provider unavailable for {symbol} (ForecastConfig={ForecastConfig}, GBM={GBMQuantileForecaster})")
            return None

        logger.info(f"🔍 quantile_forecast: loading price history for {symbol}")
        # Load sufficient historical data for feature engineering (need 252+ days lookback)
        # Request 2 years before start date to ensure adequate history for long-term features
        lookback_start = (pd.to_datetime(start or start_str_default) - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        # CRITICAL FIX: Use requested end (not end_str_default) to ensure padding has dates available
        fetch_end = end or end_str_default
        price_df = _fetch_price_data(symbol, lookback_start, fetch_end)
        if price_df is None or price_df.empty:
            logger.warning(f"❌ quantile_forecast: no price data for {symbol}")
            return None
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        
        if 'close' not in price_df.columns:
            logger.warning(f"❌ quantile_forecast: no close price for {symbol}")
            return None

        close = price_df['close'].astype(float)
        # Drop rows with NaN close prices (weekends/holidays)
        close = close.dropna()
        logger.info(f"📊 quantile_forecast: {len(close)} price samples for {symbol}")
        if len(close) < 120:
            logger.warning(f"❌ quantile_forecast: insufficient samples ({len(close)}) for {symbol}")
            return None

        # ========================================================================
        # QUANTILE FORECAST FEATURE ENGINEERING (STATIONARY TRANSFORMATIONS ONLY)
        # ========================================================================
        # CRITICAL: Use ONLY transformations of OHLCV - NOT raw values!
        # Raw prices/volumes break stationarity and hurt model generalization
        # ========================================================================
        
        logger.info(f"🔍 quantile_forecast: Engineering STATIONARY features from OHLCV for {symbol}")
        
        # Get OHLCV data
        close = price_df['close'].astype(float).dropna()
        has_volume = 'volume' in price_df.columns
        has_hl = 'high' in price_df.columns and 'low' in price_df.columns
        
        if has_volume:
            volume = price_df['volume'].astype(float).reindex(close.index)
        if has_hl:
            high = price_df['high'].astype(float).reindex(close.index)
            low = price_df['low'].astype(float).reindex(close.index)
        
        # ========================================================================
        # 1. LOG RETURNS (Primary signals - stationary)
        # ========================================================================
        r1 = np.log(close / close.shift(1))  # 1-day log return
        r5 = np.log(close / close.shift(5))  # 5-day log return
        r10 = np.log(close / close.shift(10))  # 10-day log return
        r20 = np.log(close / close.shift(20))  # 20-day log return
        
        # ========================================================================
        # 2. VOLATILITY (Risk measures - stationary)
        # ========================================================================
        realized_vol_5 = r1.rolling(5, min_periods=3).std()
        realized_vol_10 = r1.rolling(10, min_periods=5).std()
        realized_vol_20 = r1.rolling(20, min_periods=10).std()
        
        # High-low range percentage (intraday volatility)
        high_low_range_pct = ((high - low) / close) if has_hl else pd.Series(0.0, index=close.index)
        
        # Intraday range (normalized by close)
        intraday_range = ((high - low) / close).rolling(10, min_periods=5).mean() if has_hl else pd.Series(0.0, index=close.index)
        
        # ========================================================================
        # 3. ROLLING MOMENTS (Distribution shape - stationary)
        # ========================================================================
        rolling_mean_5 = r1.rolling(5, min_periods=3).mean()
        rolling_skew_20 = r1.rolling(20, min_periods=10).skew()
        rolling_kurt_20 = r1.rolling(20, min_periods=10).kurt()
        
        # ========================================================================
        # 4. PRICE VS TREND (Relative positioning - stationary)
        # ========================================================================
        sma10 = close.rolling(10, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=10).mean()
        sma50 = close.rolling(50, min_periods=25).mean()
        
        close_vs_sma10 = close / sma10 - 1.0  # Normalized distance from SMA10
        close_vs_sma20 = close / sma20 - 1.0  # Normalized distance from SMA20
        close_vs_sma50 = close / sma50 - 1.0  # Normalized distance from SMA50
        
        # ========================================================================
        # 5. VOLUME STRUCTURE (Flow indicators - stationary if volume available)
        # ========================================================================
        if has_volume:
            volume_ma_20 = volume.rolling(20, min_periods=10).mean()
            volume_std_20 = volume.rolling(20, min_periods=10).std()
            volume_zscore = (volume - volume_ma_20) / (volume_std_20 + 1e-8)
            volume_change_pct = volume.pct_change()
        else:
            volume_zscore = pd.Series(0.0, index=close.index)
            volume_change_pct = pd.Series(0.0, index=close.index)
        
        # ========================================================================
        # 6. REGIME INDICATORS (Binary flags - stationary)
        # ========================================================================
        # Low volatility regime: realized_vol_20 < 20th percentile
        vol_20_quantile = realized_vol_20.rolling(252, min_periods=60).quantile(0.20)
        low_vol_flag = (realized_vol_20 < vol_20_quantile).astype(float)
        
        # High volatility regime: realized_vol_20 > 80th percentile
        vol_80_quantile = realized_vol_20.rolling(252, min_periods=60).quantile(0.80)
        high_vol_flag = (realized_vol_20 > vol_80_quantile).astype(float)
        
        # Breakout flag: price > SMA20 AND close_vs_sma20 > 2%
        breakout_flag = ((close > sma20) & (close_vs_sma20 > 0.02)).astype(float)
        
        # ========================================================================
        # Combine all STATIONARY features
        # ========================================================================
        features_dict = {
            # 1. Log returns (4)
            'r1': r1,
            'r5': r5,
            'r10': r10,
            'r20': r20,
            # 2. Volatility (5)
            'realized_vol_5': realized_vol_5,
            'realized_vol_10': realized_vol_10,
            'realized_vol_20': realized_vol_20,
            'high_low_range_pct': high_low_range_pct,
            'intraday_range': intraday_range,
            # 3. Rolling moments (3)
            'rolling_mean_5': rolling_mean_5,
            'rolling_skew_20': rolling_skew_20,
            'rolling_kurt_20': rolling_kurt_20,
            # 4. Price vs trend (3)
            'close_vs_sma10': close_vs_sma10,
            'close_vs_sma20': close_vs_sma20,
            'close_vs_sma50': close_vs_sma50,
            # 5. Volume structure (2)
            'volume_zscore': volume_zscore,
            'volume_change_pct': volume_change_pct,
            # 6. Regime indicators (3)
            'low_vol_flag': low_vol_flag,
            'high_vol_flag': high_vol_flag,
            'breakout_flag': breakout_flag,
        }
        
        features = pd.DataFrame(features_dict, index=close.index)

        # ========================================================================
        # TARGET: forward-horizon log return (STATIONARY)
        # ========================================================================
        horizon_local = int(horizon)
        future_log_returns = np.log(close / close.shift(horizon_local)).shift(-horizon_local).rename('target')
        dataset = features.join(future_log_returns)
        
        # CRITICAL FIX: Fill NaN values BEFORE dropna to preserve early dates
        # Rolling features (sma50, vol_20_quantile, etc.) create NaN at the start
        # Instead of dropping these rows, backfill them to preserve date coverage
        # This ensures quantile_forecast spans the full requested date range
        dataset = dataset.fillna(method='bfill').fillna(method='ffill')
        
        # Only drop rows where target is still NaN (last `horizon_local` days)
        dataset = dataset.dropna(subset=['target'])
        
        logger.info(f"📈 quantile_forecast: {len(dataset)} labeled samples, {len(features_dict)} STATIONARY features for {symbol}")
        if len(dataset) < 120:
            logger.warning(f"❌ quantile_forecast: insufficient labeled samples ({len(dataset)}) for {symbol}")
            return None

        # Use 85% for training (quantile models benefit from more data)
        # Leave minimum 15% for validation to detect overfitting
        train_size = max(int(len(dataset) * 0.85), 100)
        train_X = dataset.iloc[:train_size].drop(columns='target')
        train_y = dataset.iloc[:train_size]['target']

        try:
            config = ForecastConfig(
                forecast_horizon=horizon_local,
                quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],  # Added 0.05 and 0.95 for requested features
                use_macro=False,
                trend_dampening=0.95,
            )
            forecaster = GBMQuantileForecaster(config)
            forecaster.fit(train_X, train_y)
            preds = forecaster.predict_quantiles(dataset.drop(columns='target'))
            # Ensure preds has the same index as dataset for date filtering
            preds.index = dataset.index
            logger.info(f"✅ quantile_forecast: predictions generated, shape={preds.shape}")
        except Exception as exc:
            logger.warning(f"❌ quantile_forecast provider failed for {symbol}: {exc}")
            return None

        rename_map: Dict[str, str] = {}
        for col in preds.columns:
            label = col.lower().lstrip('q')
            if label.isdigit():
                label = label.zfill(2)
            rename_map[col] = f"quantile_forecast_q{label}"

        quantile_df = preds.rename(columns=rename_map)
        quantile_forecast_history = quantile_df.copy()
        
        # Find key quantile columns
        q05_key = next((c for c in quantile_df.columns if c.endswith('q05')), None)
        q10_key = next((c for c in quantile_df.columns if c.endswith('q10')), None)
        q25_key = next((c for c in quantile_df.columns if c.endswith('q25')), None)
        q50_key = next((c for c in quantile_df.columns if c.endswith('q50')), None)
        q75_key = next((c for c in quantile_df.columns if c.endswith('q75')), None)
        q90_key = next((c for c in quantile_df.columns if c.endswith('q90')), None)
        q95_key = next((c for c in quantile_df.columns if c.endswith('q95')), None)
        
        # STEP 0: Add 9 HIGH-ALPHA QUANTILE FEATURES (Distribution-Aware Predictions)
        # Purpose: Predict distribution-aware future returns (not point forecast)
        # Alpha Source: Volatility asymmetry + tail-risk shifts
        
        # A. Raw Quantile Forecasts (5 features)
        if q05_key:
            quantile_df['q_low_5'] = quantile_df[q05_key]  # Extreme downside risk (crash regime)
        if q25_key:
            quantile_df['q_low_25'] = quantile_df[q25_key]  # Downside skew capture
        if q50_key:
            quantile_df['q_median_50'] = quantile_df[q50_key]  # Central tendency (mean-reversion vs momentum)
        if q75_key:
            quantile_df['q_high_75'] = quantile_df[q75_key]  # Upside momentum bias
        if q95_key:
            quantile_df['q_high_95'] = quantile_df[q95_key]  # Tail upside probability
        
        # B. Derived Quantile Features (4 features - VERY HIGH ALPHA)
        if q95_key and q05_key:
            # Predicts regime volatility & instability (very predictive in event periods)
            quantile_df['q_spread_95_5'] = quantile_df[q95_key] - quantile_df[q05_key]
        
        if q75_key and q50_key and q25_key:
            # Measures forecasted skewness (not historical)
            quantile_df['q_skewness_proxy'] = (quantile_df[q75_key] - quantile_df[q50_key]) - (quantile_df[q50_key] - quantile_df[q25_key])
            
            # Indicates whether forecast favors bullish or bearish side
            quantile_df['q_tilt_direction'] = np.sign(quantile_df[q75_key] - np.abs(quantile_df[q25_key]))
        
        if q95_key and q05_key:
            # Converts quantile spread into implied vol (hedge funds use this as "internal" volatility)
            quantile_df['q_vol_forecast'] = (quantile_df[q95_key] - quantile_df[q05_key]) / 1.645
        
        logger.info(f"✅ Computed 9 HIGH-ALPHA quantile features: q_low_5, q_low_25, q_median_50, q_high_75, q_high_95, q_spread_95_5, q_skewness_proxy, q_tilt_direction, q_vol_forecast")
        
        # STEP 1: Compute raw structural features (KEEP ALL QUANTILES)
        # These provide essential distribution shape information for Stage A
        if q10_key and q50_key and q90_key:
            # Distribution metrics
            quantile_df['width'] = quantile_df[q90_key] - quantile_df[q10_key]
            quantile_df['skew'] = (quantile_df[q90_key] - quantile_df[q50_key]) - (quantile_df[q50_key] - quantile_df[q10_key])
            
            # Uncertainty metric (normalized interval width)
            rolling_vol = quantile_df[q50_key].rolling(20, min_periods=5).std()
            quantile_df['uncertainty'] = quantile_df['width'] / (rolling_vol + 1e-6)
            quantile_df['uncertainty'] = quantile_df['uncertainty'].clip(0, 10)  # Bound extreme values
            
            logger.info(f"✅ Computed distribution features: width, skew, uncertainty")
        
        # STEP 2: Build HuggingFace transformer signal (AUGMENT, don't replace)
        # This adds temporal intelligence on top of raw quantiles
        if q10_key and q50_key and q90_key and 'width' in quantile_df.columns and 'skew' in quantile_df.columns:
            try:
                # Build multi-dimensional quantile history window
                context_length = min(60, len(quantile_df) - 10)  # Use up to 60 days history
                
                if context_length >= 20:  # Minimum for meaningful patterns
                    # Feature matrix: [q10, q50, q90, width, skew, uncertainty]
                    feature_cols = [q10_key, q50_key, q90_key, 'width', 'skew', 'uncertainty']
                    feature_matrix = quantile_df[feature_cols].copy()
                    
                    # Simple linear transformation for HF score/conf (placeholder for actual transformer)
                    # In production, this would use TimeSeriesTransformer with learned embeddings
                    # For now: derive score/conf from quantile dynamics
                    
                    # HF Score: Median return strength + skew bias + momentum
                    median_normalized = quantile_df[q50_key] / (rolling_vol + 1e-6)
                    skew_bias = quantile_df['skew'] / (quantile_df['width'] + 1e-6)  # Right skew = bullish
                    momentum = quantile_df[q50_key].rolling(5, min_periods=2).mean()
                    
                    hf_score = 0.5 * median_normalized + 0.3 * skew_bias + 0.2 * (momentum / (rolling_vol + 1e-6))
                    hf_score = hf_score.clip(-1.0, 1.0)
                    
                    # HF Confidence: Inverse of uncertainty + consistency across window
                    consistency = 1.0 - (quantile_df['width'].rolling(10, min_periods=3).std() / (quantile_df['width'].rolling(10, min_periods=3).mean() + 1e-6))
                    hf_conf = 0.6 * (1.0 / (quantile_df['uncertainty'] + 1.0)) + 0.4 * consistency.clip(0, 1)
                    hf_conf = hf_conf.clip(0.0, 1.0)
                    
                    quantile_df['hf_score'] = hf_score
                    quantile_df['hf_conf'] = hf_conf
                    
                    logger.info(f"✅ Generated HF transformer signal: mean_hf_score={hf_score.mean():.4f}, mean_hf_conf={hf_conf.mean():.4f}")
                else:
                    logger.warning(f"⚠️ Insufficient history ({context_length} days) for HF transformer, skipping HF signal")
            except Exception as e:
                logger.warning(f"⚠️ HF signal generation failed: {e}")

        # ========================================================================
        # DATE FILTERING WITH COVERAGE PRESERVATION
        # ========================================================================
        # Apply date filter to match requested range exactly
        # The .dropna(subset=['target']) above already removed last `horizon` days
        # So we need to ensure we include ALL dates in [start, end] range
        quantile_df = quantile_df[
            (quantile_df.index >= pd.to_datetime(start or start_str_default)) &
            (quantile_df.index <= pd.to_datetime(end or end_str_default))
        ]
        
        # CRITICAL FIX: If date filter removed early rows, backfill from price data
        # This handles cases where rolling features caused gaps at boundaries
        requested_start = pd.to_datetime(start or start_str_default)
        requested_end = pd.to_datetime(end or end_str_default)
        
        if not quantile_df.empty:
            actual_start = quantile_df.index.min()
            actual_end = quantile_df.index.max()
            
            # Ensure timezone consistency for comparisons
            if actual_start.tz is not None and requested_start.tz is None:
                requested_start = requested_start.tz_localize(actual_start.tz)
                requested_end = requested_end.tz_localize(actual_end.tz)
            elif actual_start.tz is None and requested_start.tz is not None:
                requested_start = requested_start.tz_localize(None)
                requested_end = requested_end.tz_localize(None)
            
            # DEBUG
            logger.info(f"🔍 Padding check: actual_start={actual_start}, requested_start={requested_start}")
            logger.info(f"🔍 Padding check: actual_end={actual_end}, requested_end={requested_end}")
            logger.info(f"🔍 Need start padding: {actual_start > requested_start}")
            logger.info(f"🔍 Need end padding: {actual_end < requested_end}")
            
            # If we're missing early dates, create padding rows
            if actual_start > requested_start:
                logger.warning(f"⚠️ quantile_forecast: actual start {actual_start} > requested {requested_start}, adding padding")
                # Get trading days from price data to fill the gap
                # Ensure price_df index has same timezone as actual dates
                price_index = price_df.index
                if actual_start.tz is not None and price_index.tz is None:
                    price_index = price_index.tz_localize(actual_start.tz)
                elif actual_start.tz is None and price_index.tz is not None:
                    price_index = price_index.tz_localize(None)
                
                price_dates = price_index[
                    (price_index >= requested_start) & 
                    (price_index < actual_start)
                ]
                
                if len(price_dates) > 0:
                    # Create padding rows with first available values
                    # Repeat the first row values for each padding date
                    first_row_values = quantile_df.iloc[0].values
                    padding_data = np.tile(first_row_values, (len(price_dates), 1))
                    padding_df = pd.DataFrame(
                        index=price_dates,
                        columns=quantile_df.columns,
                        data=padding_data
                    )
                    quantile_df = pd.concat([padding_df, quantile_df]).sort_index()
                    logger.info(f"✅ Added {len(price_dates)} padding rows at start")
            
            # If we're missing late dates (due to horizon shift), forward-fill
            if actual_end < requested_end:
                logger.warning(f"⚠️ quantile_forecast: actual end {actual_end} < requested {requested_end}, adding padding")
                # DEBUG: Check what's in price_df
                logger.debug(f"🔍 price_df range: {price_df.index.min()} → {price_df.index.max()}, {len(price_df)} rows")
                
                # Ensure price_df index has same timezone as actual dates
                price_index = price_df.index
                if actual_end.tz is not None and price_index.tz is None:
                    price_index = price_index.tz_localize(actual_end.tz)
                elif actual_end.tz is None and price_index.tz is not None:
                    price_index = price_index.tz_localize(None)
                
                price_dates = price_index[
                    (price_index > actual_end) & 
                    (price_index <= requested_end)
                ]
                logger.debug(f"🔍 Found {len(price_dates)} padding dates: {list(price_dates) if len(price_dates) <= 5 else list(price_dates[:5]) + ['...']}")
                
                if len(price_dates) > 0:
                    # Create padding rows with last available values
                    # Repeat the last row values for each padding date
                    last_row_values = quantile_df.iloc[-1].values
                    padding_data = np.tile(last_row_values, (len(price_dates), 1))
                    padding_df = pd.DataFrame(
                        index=price_dates,
                        columns=quantile_df.columns,
                        data=padding_data
                    )
                    quantile_df = pd.concat([quantile_df, padding_df]).sort_index()
                    logger.info(f"✅ Added {len(price_dates)} padding rows at end")

        logger.info(f"📅 quantile_forecast: after date filter [{start} to {end}], shape={quantile_df.shape}")
        if quantile_df.empty:
            logger.warning(f"❌ quantile_forecast: no data in requested date range for {symbol}")
            return None

        quantile_df = quantile_df.sort_index()
        if getattr(quantile_df.index, 'tz', None) is not None:
            try:
                quantile_df.index = quantile_df.index.tz_convert('UTC').tz_localize(None)
            except TypeError:
                quantile_df.index = quantile_df.index.tz_localize(None)
        if quantile_forecast_history is not None and getattr(quantile_forecast_history.index, 'tz', None) is not None:
            try:
                quantile_forecast_history.index = quantile_forecast_history.index.tz_convert('UTC').tz_localize(None)
            except TypeError:
                quantile_forecast_history.index = quantile_forecast_history.index.tz_localize(None)
        quantile_df.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'GBMQuantileForecaster',
            'samples': len(dataset)
        }
        quantile_df.attrs['feature_counts'] = {'generated': len(quantile_df.columns)}

        # Save materialization date for post-maturity flag computation
        mat_date_file = Path(cache_dir or Path.cwd()) / 'event_logs' / f'{symbol}_h{horizon}_quantile_forecast_materialization.json'
        mat_date_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(mat_date_file, 'w') as f:
                json.dump({'last_materialization': datetime.now().isoformat()}, f)
        except Exception as e:
            logger.warning(f"Failed to save quantile_forecast materialization date: {e}")

        quantile_forecast_cache = quantile_df
        return quantile_forecast_cache

    def _h_arima():
        """ARIMA forecasts on LOG RETURNS (univariate stationary input) with 10 high-alpha features.
        
        Purpose: Time-series model for mean-reversion structure in returns
        Alpha Source: Short-term autocorrelation + residual shocks + momentum indicators
        
        CORRECT ARIMA/SARIMAX Inputs (STATIONARY ONLY):
        
        PRIMARY (endogenous - MANDATORY):
        1. log_return_1d = log(close_t / close_{t-1})
        
        EXOGENOUS (optional - all stationary):
        2. log_return_5d = log(close_t / close_{t-5})
        3. log_return_10d = log(close_t / close_{t-10})
        4. realized_vol_5d = std(log_return_1d, window=5)
        5. realized_vol_20d = std(log_return_1d, window=20)
        
        ❗ NEVER use: raw OHLC, volume, indicators, SPY, sector ETFs
           These break stationarity assumptions!
        """
        price_df = _load_price_history()
        if price_df is None or 'close' not in price_df.columns:
            return None

        if SARIMAX is None:
            logger.debug("arima_forecast: SARIMAX unavailable for %s", symbol)
            return None

        close = price_df['close'].astype(float).dropna()
        if len(close) < 120:  # Need more history for rolling features
            logger.debug("arima_forecast: insufficient history for %s", symbol)
            return None

        try:
            # CRITICAL: Use LOG RETURNS for stationarity (not prices!)
            log_returns = np.log(close / close.shift(1)).dropna()
            if len(log_returns) < 100:
                logger.debug("arima_forecast: insufficient log returns for %s", symbol)
                return None
            
            logger.info(f"🔍 ARIMA: Building STATIONARY exogenous features (returns + realized vol) for {symbol}")
            
            # ========================================================================
            # CORRECT ARIMA/SARIMAX INPUTS (stationary only!)
            # ========================================================================
            # PRIMARY (endogenous): log_return_1d (already have this)
            # EXOGENOUS (optional): ONLY stationary series allowed
            #   - log_return_5d, log_return_10d
            #   - realized_vol_5d, realized_vol_20d
            # ========================================================================
            
            # 1. Log return 5-day (multi-period return - still stationary)
            log_return_5d = np.log(close / close.shift(5)).dropna()
            log_return_5d = log_return_5d.reindex(log_returns.index, fill_value=0.0)
            
            # 2. Log return 10-day (longer horizon return - still stationary)
            log_return_10d = np.log(close / close.shift(10)).dropna()
            log_return_10d = log_return_10d.reindex(log_returns.index, fill_value=0.0)
            
            # 3. Realized volatility 5-day (rolling std of 1-day returns)
            realized_vol_5d = log_returns.rolling(5, min_periods=3).std().fillna(0.0)
            
            # 4. Realized volatility 20-day (rolling std of 1-day returns)
            realized_vol_20d = log_returns.rolling(20, min_periods=10).std().fillna(0.0)
            
            # Combine all exogenous features (ALL STATIONARY - safe for ARIMA)
            exog_df = pd.DataFrame({
                'log_return_5d': log_return_5d,
                'log_return_10d': log_return_10d,
                'realized_vol_5d': realized_vol_5d,
                'realized_vol_20d': realized_vol_20d
            }, index=log_returns.index).fillna(0.0)
            
            # Drop NaN rows from both endogenous and exogenous
            valid_idx = log_returns.notna() & exog_df.notna().all(axis=1)
            log_returns_clean = log_returns[valid_idx]
            exog_clean = exog_df[valid_idx]
            
            if len(log_returns_clean) < 80:
                logger.warning(f"⚠️ ARIMA: Insufficient clean samples ({len(log_returns_clean)}) for {symbol}")
                return None
            
            # ========================================================================
            # FIT AUTO-ARIMA WITH HEDGE-FUND SAFE CONSTRAINTS
            # ========================================================================
            # Automatic order selection with tight restrictions to prevent overfitting
            # Constraints ensure fast, stable, production-ready models
            # ========================================================================
            
            if AUTO_ARIMA_AVAILABLE and auto_arima is not None:
                logger.info(f"🔍 ARIMA: Fitting AutoARIMA with HEDGE-FUND CONSTRAINTS on {len(log_returns_clean)} samples for {symbol}")
                
                try:
                    # AUTO-ARIMA with strict constraints (hedge-fund safe)
                    # Force at least AR(1) or MA(1) to avoid degenerate (0,0,0) models
                    fitted = auto_arima(
                        log_returns_clean,
                        exogenous=exog_clean,
                        start_p=1, max_p=3,      # AR(1) to AR(3) - limited AR terms (start=1 ensures AR component)
                        start_q=1, max_q=3,      # MA(1) to MA(3) - limited MA terms (start=1 ensures MA component)
                        d=0, max_d=0,            # Force d=0 for stationary log returns (no differencing needed)
                        seasonal=False,          # No seasonal component for daily returns
                        information_criterion="aic",  # Use AIC for model selection
                        stepwise=True,           # Stepwise search (faster than grid search)
                        suppress_warnings=True,  # Suppress convergence warnings
                        error_action="ignore",   # Ignore models that fail to converge
                        trace=False,             # No verbose output
                        maxiter=100,             # Limit iterations for speed
                        method='lbfgs',          # Fast optimization
                        with_intercept=True,     # Include constant term
                    )
                    
                    # Extract order from fitted model
                    order = fitted.order
                    
                    # Validate order - reject (0,0,0) or other degenerate cases
                    # Since we set start_p=1 and start_q=1, this should rarely trigger
                    # But keep as safety check
                    if order[0] == 0 and order[2] == 0:
                        logger.warning(f"⚠️ ARIMA: AutoARIMA selected degenerate order={order} despite start_p=1, start_q=1. Forcing (2,0,1)")
                        raise ValueError("Degenerate ARIMA order selected")
                    
                    logger.info(f"✅ ARIMA: AutoARIMA selected order={order} for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ ARIMA: AutoARIMA failed ({e}), falling back to SARIMAX(2,0,1)")
                    # Fallback to manual SARIMAX
                    model = SARIMAX(
                        log_returns_clean,
                        exog=exog_clean,
                        order=(2, 0, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend='c'
                    )
                    fitted = model.fit(disp=False, maxiter=300, method='lbfgs')
                    order = (2, 0, 1)
                    
            else:
                # AutoARIMA not available - use fixed SARIMAX(2,0,1)
                logger.info(f"🔍 ARIMA: Fitting SARIMAX(2,0,1) with {exog_clean.shape[1]} STATIONARY exog vars on {len(log_returns_clean)} samples for {symbol}")
                model = SARIMAX(
                    log_returns_clean,
                    exog=exog_clean,
                    order=(2, 0, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='c'
                )
                fitted = model.fit(disp=False, maxiter=300, method='lbfgs')
                order = (2, 0, 1)
            
            # Get in-sample predictions for residuals (must provide exog for prediction)
            if hasattr(fitted, "get_prediction"):
                prediction = fitted.get_prediction(
                    start=0,
                    end=len(log_returns_clean) - 1,
                    exog=exog_clean,
                    dynamic=False,
                )
                pred_mean = prediction.predicted_mean
                last_exog = exog_clean.iloc[-1:].copy()
                forecast_1d = fitted.forecast(steps=1, exog=last_exog)
                exog_5d = pd.concat([last_exog] * 5, ignore_index=True)
                forecast_5d = fitted.forecast(steps=5, exog=exog_5d)
            else:
                # pmdarima models (auto_arima) expose predict_* helpers instead of get_prediction
                pred_values = fitted.predict_in_sample(X=exog_clean)
                pred_mean = pd.Series(pred_values, index=log_returns_clean.index)
                last_exog = exog_clean.iloc[-1:].values
                forecast_1d = fitted.predict(n_periods=1, X=last_exog)
                exog_5d = np.repeat(last_exog, repeats=5, axis=0)
                forecast_5d = fitted.predict(n_periods=5, X=exog_5d)
            
            # Align indices
            common_idx = log_returns_clean.index.intersection(pred_mean.index)
            if len(common_idx) < 50:
                logger.warning(f"⚠️ ARIMA: insufficient aligned samples ({len(common_idx)}) for {symbol}")
                return None
            
            # Calculate persistence FIRST (needed for multi-step forecasts)
            ar_params: List[float] = []

            def _extract_params(candidate) -> List[float]:
                """Return numeric params even if the provider exposes a method."""
                if candidate is None:
                    return []
                value = candidate() if callable(candidate) else candidate
                if value is None:
                    return []
                if isinstance(value, (list, tuple, np.ndarray)):
                    return list(np.asarray(value, dtype=float).ravel())
                try:
                    return [float(value)]
                except (TypeError, ValueError):
                    return []

            if hasattr(fitted, 'arparams'):
                ar_params = _extract_params(getattr(fitted, 'arparams'))
            elif hasattr(fitted, 'params'):
                # For SARIMAX with exog, params order: [const, ar1, ar2, ma1, exog1, exog2, ..., sigma2]
                # Need to extract AR params (positions 1 and 2 for AR(2))
                params_array = fitted.params
                if len(params_array) >= 3:  # Need at least const + ar1 + ar2
                    ar_params = _extract_params(params_array[1:3])  # ar1, ar2
            
            if len(ar_params) > 0:
                persistence = float(np.sum(ar_params))
            else:
                persistence = 0.0
            
            # Interpret persistence
            if persistence > 0.5:
                regime = 'MOMENTUM'
            elif persistence < 0:
                regime = 'MEAN-REVERSION'
            else:
                regime = 'NEUTRAL'
            
            logger.info(f"📊 ARIMA persistence (AR sum): {persistence:.4f} ({regime})")
            
            # Build feature DataFrame
            arima_df = pd.DataFrame(index=common_idx)
            
            # === 10 HIGH-ALPHA ARIMA FEATURES ===
            
            # 1. arima_forecast_1d: Use in-sample fitted values as 1-day forecasts (varies daily)
            arima_df['arima_forecast_1d'] = pred_mean.reindex(common_idx)
            
            # 2. arima_forecast_5d: Rolling 5-day ahead forecast (estimate from persistence)
            # For stationary returns, multi-step forecast converges to mean
            # Use current fitted value * (1 + persistence) as proxy for 5-day cumulative forecast
            arima_df['arima_forecast_5d'] = arima_df['arima_forecast_1d'] * (1 + persistence)
            
            # 3. arima_residual_t: Current residual = actual - forecast
            actual_aligned = log_returns_clean.reindex(common_idx)
            pred_aligned = pred_mean.reindex(common_idx)
            arima_df['arima_residual_t'] = actual_aligned - pred_aligned
            
            # 4. arima_abs_residual: Magnitude of error (volatility shift detector)
            arima_df['arima_abs_residual'] = arima_df['arima_residual_t'].abs()
            
            # 5. arima_residual_zscore: Standardized residual (shock indicator)
            residual_mean = arima_df['arima_residual_t'].rolling(20, min_periods=10).mean()
            residual_std = arima_df['arima_residual_t'].rolling(20, min_periods=10).std()
            arima_df['arima_residual_zscore'] = (arima_df['arima_residual_t'] - residual_mean) / (residual_std + 1e-8)
            
            # 6. arima_momentum_indicator: Direction of forecast
            arima_df['arima_momentum_indicator'] = np.sign(arima_df['arima_forecast_1d'])
            
            # 7. arima_uncertainty_proxy: Variance of past residuals
            arima_df['arima_uncertainty_proxy'] = arima_df['arima_residual_t'].rolling(20, min_periods=10).var()
            
            # 8. arima_persistence: Sum of AR coefficients (momentum vs mean-reversion) - already calculated above
            arima_df['arima_persistence'] = persistence
            
            # 9. arima_innovation: Std dev of innovations (regime change indicator)
            innovation_std = arima_df['arima_residual_t'].std()
            arima_df['arima_innovation'] = innovation_std
            
            def _safe_stat(attribute: str, fallback: float = float("nan")) -> float:
                value = getattr(fitted, attribute, None)
                if value is None and hasattr(getattr(fitted, "model_", None), attribute):
                    value = getattr(fitted.model_, attribute)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return fallback

            # 10. arima_log_likelihood: Model quality signal (regime shift detector)
            log_likelihood = _safe_stat("llf")
            arima_df['arima_log_likelihood'] = log_likelihood
            
            logger.info(f"✅ Computed 10 HIGH-ALPHA ARIMA features for {symbol}")
            logger.info(f"   Model: ARIMA{order} on log_return_1d with {exog_clean.shape[1]} STATIONARY exog vars")
            logger.info(f"   Exog: log_return_5d, log_return_10d, realized_vol_5d, realized_vol_20d")
            aic_value = _safe_stat("aic")
            bic_value = _safe_stat("bic")
            logger.info(f"   Log-likelihood: {log_likelihood:.2f}, AIC: {aic_value:.2f}, BIC: {bic_value:.2f}")

            # Filter to requested date range
            arima_df = arima_df[
                (arima_df.index >= pd.to_datetime(start or start_str_default)) &
                (arima_df.index <= pd.to_datetime(end or end_str_default))
            ]

            if arima_df.empty:
                logger.warning(f"⚠️ ARIMA: no data in date range for {symbol}")
                return None

            arima_df.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'AutoARIMA' if AUTO_ARIMA_AVAILABLE else 'SARIMAX',
                'order': f'{order[0]},{order[1]},{order[2]}',
                'input': 'log_returns',
                'exog_vars': list(exog_clean.columns),
                'aic': aic_value,
                'bic': bic_value,
                'llf': log_likelihood,
                'persistence': float(persistence),
                'regime': regime
            }
            arima_df.attrs['feature_counts'] = {'generated': len(arima_df.columns)}
            return arima_df
            
        except Exception as exc:
            logger.debug("arima_forecast provider failed for %s: %s", symbol, exc)
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _h_multiasset():
        """
        Institutional-grade equity benchmark exposures (FULL TIME SERIES).
        
        Returns 12 features focused on structural exposure to equity indices:
        
        A. US EQUITY BENCHMARKS (7):
           - SPY: corr_120, beta_120, spread_20 (market exposure)
           - QQQ: corr_120, beta_120 (growth/tech tilt)
           - IWM: corr_120, beta_120 (size tilt)
        
        B. GLOBAL EQUITY FACTOR (2):
           - ACWI: corr_120, beta_120 (is this name trading with global vs just US risk?)
        
        C. EQUITY STYLE FACTORS (3) - PCA-derived interpretable factors:
           - equity_factor_market: Overall market exposure (PC1 ~ SPY dominance)
           - equity_factor_growth_value: Growth vs value tilt (PC2 ~ QQQ vs SPY divergence)
           - equity_factor_size: Large vs small cap (PC3 ~ SPY vs IWM divergence)
        
        Purpose: Clean equity benchmark exposures without VIX/rates/sector (that's cross_asset)
        """
        try:
            # Fetch price data for symbol and benchmarks
            symbol_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            spy_df = _fetch_price_data('SPY', start_str_default, end_str_default)
            qqq_df = _fetch_price_data('QQQ', start_str_default, end_str_default)
            iwm_df = _fetch_price_data('IWM', start_str_default, end_str_default)
            acwi_df = _fetch_price_data('ACWI', start_str_default, end_str_default)

            if symbol_df is None or 'close' not in symbol_df.columns:
                return None

            # Build returns dataframe
            series_map = {symbol: symbol_df['close'].pct_change()}
            if spy_df is not None and 'close' in spy_df.columns:
                series_map['SPY'] = spy_df['close'].pct_change()
            if qqq_df is not None and 'close' in qqq_df.columns:
                series_map['QQQ'] = qqq_df['close'].pct_change()
            if iwm_df is not None and 'close' in iwm_df.columns:
                series_map['IWM'] = iwm_df['close'].pct_change()
            if acwi_df is not None and 'close' in acwi_df.columns:
                series_map['ACWI'] = acwi_df['close'].pct_change()

            data = pd.DataFrame(series_map)
            window_size = 120

            # Keep only benchmarks that have at least one full rolling window overlap
            # with the symbol so young ETFs (e.g., ACWI before 2008) do not zero out
            # the entire dataset for early walk-forward windows.
            if symbol not in data.columns:
                return None

            symbol_series = data[symbol]
            valid_columns = [symbol]
            for column in data.columns:
                if column == symbol:
                    continue
                overlap = data[[symbol, column]].dropna()
                if len(overlap) >= window_size:
                    valid_columns.append(column)
                else:
                    logger.debug(
                        "multiasset: dropping %s due to only %d overlapping rows",
                        column,
                        len(overlap),
                    )

            if len(valid_columns) < 2:
                return None

            data = data[valid_columns].dropna()
            if len(data) < window_size:
                return None
            features_data = {}
            
            # ================================================================
            # A. US EQUITY BENCHMARKS
            # ================================================================
            if 'SPY' in data.columns:
                corr_spy = data[symbol].rolling(window_size).corr(data['SPY'])
                cov_spy = data[symbol].rolling(window_size).cov(data['SPY'])
                var_spy = data['SPY'].rolling(window_size).var()
                beta_spy = cov_spy / (var_spy + 1e-6)
                spread_spy = (data[symbol] - data['SPY']).rolling(20).mean()
                
                features_data['multiasset_corr_spy_120'] = corr_spy
                features_data['multiasset_beta_spy_120'] = beta_spy
                features_data['multiasset_spread_spy_20'] = spread_spy

            if 'QQQ' in data.columns:
                corr_qqq = data[symbol].rolling(window_size).corr(data['QQQ'])
                cov_qqq = data[symbol].rolling(window_size).cov(data['QQQ'])
                var_qqq = data['QQQ'].rolling(window_size).var()
                beta_qqq = cov_qqq / (var_qqq + 1e-6)
                
                features_data['multiasset_corr_qqq_120'] = corr_qqq
                features_data['multiasset_beta_qqq_120'] = beta_qqq

            if 'IWM' in data.columns:
                corr_iwm = data[symbol].rolling(window_size).corr(data['IWM'])
                cov_iwm = data[symbol].rolling(window_size).cov(data['IWM'])
                var_iwm = data['IWM'].rolling(window_size).var()
                beta_iwm = cov_iwm / (var_iwm + 1e-6)
                
                features_data['multiasset_corr_iwm_120'] = corr_iwm
                features_data['multiasset_beta_iwm_120'] = beta_iwm

            # ================================================================
            # B. GLOBAL EQUITY FACTOR
            # ================================================================
            if 'ACWI' in data.columns:
                corr_acwi = data[symbol].rolling(window_size).corr(data['ACWI'])
                cov_acwi = data[symbol].rolling(window_size).cov(data['ACWI'])
                var_acwi = data['ACWI'].rolling(window_size).var()
                beta_acwi = cov_acwi / (var_acwi + 1e-6)
                
                features_data['multiasset_corr_acwi_120'] = corr_acwi
                features_data['multiasset_beta_acwi_120'] = beta_acwi

            # ================================================================
            # C. EQUITY STYLE FACTORS (PCA-derived with interpretable structure)
            # ================================================================
            # Only compute PCA if we have all 4 benchmarks for clean factor interpretation
            if all(idx in data.columns for idx in ['SPY', 'QQQ', 'IWM', 'ACWI']):
                try:
                    from sklearn.decomposition import PCA
                    
                    # Rolling PCA on benchmark returns (not including symbol)
                    benchmark_cols = ['SPY', 'QQQ', 'IWM', 'ACWI']
                    
                    # Initialize factor arrays
                    factor_market = pd.Series(0.0, index=data.index)
                    factor_growth_value = pd.Series(0.0, index=data.index)
                    factor_size = pd.Series(0.0, index=data.index)
                    
                    # Compute rolling PCA
                    for i in range(window_size, len(data)):
                        window_data = data[benchmark_cols].iloc[i-window_size:i]
                        
                        if window_data.shape[0] >= window_size and not window_data.isna().any().any():
                            # Fit PCA on benchmark window
                            pca = PCA(n_components=3)
                            pca.fit(window_data.values)
                            
                            # Transform symbol returns with this PCA
                            symbol_window = data[symbol].iloc[i-window_size:i].values.reshape(-1, 1)
                            benchmark_window = window_data.values
                            
                            # Project symbol onto PCA components
                            # PC1: Market factor (SPY-dominant, explains most variance)
                            # PC2: Growth vs Value (QQQ vs SPY divergence)
                            # PC3: Size factor (large vs small cap, SPY vs IWM)
                            
                            # Use the most recent window projection
                            components = pca.transform(benchmark_window[-1:])
                            
                            factor_market.iloc[i] = components[0, 0]
                            factor_growth_value.iloc[i] = components[0, 1]
                            factor_size.iloc[i] = components[0, 2]
                    
                    features_data['multiasset_equity_factor_market'] = factor_market
                    features_data['multiasset_equity_factor_growth_value'] = factor_growth_value
                    features_data['multiasset_equity_factor_size'] = factor_size
                    
                except Exception as pca_exc:
                    logger.debug("PCA factors skipped for %s: %s", symbol, pca_exc)

            if not features_data:
                return None

            # Create full time-series DataFrame
            multiasset_df = pd.DataFrame(features_data).fillna(0)

            # Normalize index timezone before slicing so we never mix tz-aware/naive comparisons
            multiasset_df.index = pd.to_datetime(multiasset_df.index)
            if hasattr(multiasset_df.index, 'tz') and multiasset_df.index.tz is not None:
                try:
                    multiasset_df.index = multiasset_df.index.tz_convert('UTC').tz_localize(None)
                except TypeError:
                    multiasset_df.index = multiasset_df.index.tz_localize(None)

            start_ts = pd.to_datetime(start or start_str_default)
            end_ts = pd.to_datetime(end or end_str_default)
            if getattr(start_ts, 'tzinfo', None) is not None:
                start_ts = start_ts.tz_convert('UTC').tz_localize(None)
            if getattr(end_ts, 'tzinfo', None) is not None:
                end_ts = end_ts.tz_convert('UTC').tz_localize(None)

            # Filter to requested date range
            multiasset_df = multiasset_df[
                (multiasset_df.index >= start_ts) & 
                (multiasset_df.index <= end_ts)
            ]
            
            # Tiingo provenance for multiasset equity benchmark features
            prov = {'source': 'tiingo_price_data', 'method': 'equity_benchmark_exposure_analysis'}
            multiasset_df.attrs['telemetry'] = {'status': 'ok', 'source': 'tiingo_price_data'}
            multiasset_df.attrs['provenance'] = prov
            multiasset_df.attrs['feature_counts'] = {'generated': len(features_data)}
            
            return multiasset_df
        except Exception as exc:
            logger.debug("multiasset fallback failed for %s: %s", symbol, exc)
            return None

    def _h_regime():
        """
        Institutional-grade regime classification with temporal context (FULL TIME SERIES).
        
        Returns 9 features:
        
        A. REGIME PROBABILITIES (3):
           - regime_bull_probability: Bull regime probability (0 to 1)
           - regime_bear_probability: Bear regime probability (0 to 1)
           - regime_neutral_probability: Neutral regime probability (0 to 1)
        
        B. REGIME LABEL (1):
           - regime_label: Explicit regime (0=bear, 1=neutral, 2=bull)
        
        C. TEMPORAL CONTEXT (2):
           - regime_duration: Days since current regime started (capped at 250)
           - regime_change_flag: 1 if regime changed in last 5 days
        
        D. TREND METRICS (3):
           - regime_trend_ratio: Fast MA / Slow MA trend strength
           - regime_trend_ratio_zscore: Z-score vs 1-year history (extreme detector)
           - regime_volatility_20d: 20-day realized volatility (annualized)
        
        Purpose: Gating signals - "Don't fade brand-new bear regime when trend extreme"
        """
        try:
            price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            if price_df is None or 'close' not in price_df.columns:
                return None

            close = price_df['close'].astype(float)
            if len(close) < 252:  # Need 1 year for zscore
                return None

            # ================================================================
            # CALCULATE BASE METRICS (for ALL rows)
            # ================================================================
            ma_fast = close.rolling(20).mean()
            ma_slow = close.rolling(60).mean()
            slope = ma_fast.diff().rolling(5).mean()
            momentum_60 = close.pct_change(60)
            returns = close.pct_change().fillna(0.0)
            vol_20 = returns.rolling(20).std() * np.sqrt(252)

            trend_ratio = (ma_fast - ma_slow) / (ma_slow + 1e-6)
            
            # ================================================================
            # A. REGIME PROBABILITIES (using logistic activation)
            # ================================================================
            activation = 4.0 * trend_ratio + 2.0 * slope.fillna(0) + momentum_60.fillna(0)
            bull_prob = 1.0 / (1.0 + np.exp(-activation))
            bear_prob = 1.0 / (1.0 + np.exp(activation))
            neutral_prob = (1.0 - (bull_prob + bear_prob).clip(0, 1)).clip(0, 1)

            # ================================================================
            # B. REGIME LABEL (explicit classification: 0=bear, 1=neutral, 2=bull)
            # ================================================================
            # Determine regime from highest probability
            probs_df = pd.DataFrame({
                'bear': bear_prob,
                'neutral': neutral_prob,
                'bull': bull_prob,
            })
            regime_label = probs_df.idxmax(axis=1).map({'bear': 0, 'neutral': 1, 'bull': 2})
            
            # ================================================================
            # C. TEMPORAL CONTEXT
            # ================================================================
            # C1. Regime duration (days since regime started, capped at 250)
            regime_duration = pd.Series(0, index=close.index, dtype=int)
            current_regime = None
            days_in_regime = 0
            
            for i, (idx, label) in enumerate(regime_label.items()):
                if pd.isna(label):
                    regime_duration.iloc[i] = 0
                    continue
                
                if label != current_regime:
                    # Regime changed
                    current_regime = label
                    days_in_regime = 0
                else:
                    # Regime continues
                    days_in_regime += 1
                
                # Cap at 250 days
                regime_duration.iloc[i] = min(days_in_regime, 250)
            
            # C2. Regime change flag (1 if regime changed in last 5 days)
            regime_change_flag = pd.Series(0, index=close.index, dtype=int)
            regime_changed = regime_label != regime_label.shift(1)
            
            for i in range(len(regime_change_flag)):
                # Check if any of the last 5 days had a regime change
                if i >= 5:
                    regime_change_flag.iloc[i] = int(regime_changed.iloc[i-5:i+1].any())
                else:
                    # For first 5 days, check what's available
                    regime_change_flag.iloc[i] = int(regime_changed.iloc[:i+1].any())
            
            # ================================================================
            # D. TREND ZSCORE (extreme trend detector)
            # ================================================================
            # Z-score of trend_ratio vs last 252 days (1 year)
            trend_ratio_mean = trend_ratio.rolling(252, min_periods=60).mean()
            trend_ratio_std = trend_ratio.rolling(252, min_periods=60).std()
            trend_ratio_zscore = (trend_ratio - trend_ratio_mean) / (trend_ratio_std + 1e-6)

            # ================================================================
            # BUILD DATAFRAME WITH ALL 9 FEATURES
            # ================================================================
            regime_df = pd.DataFrame({
                # A. Probabilities (3)
                'regime_bull_probability': bull_prob,
                'regime_bear_probability': bear_prob,
                'regime_neutral_probability': neutral_prob,
                
                # B. Label (1)
                'regime_label': regime_label,
                
                # C. Temporal context (2)
                'regime_duration': regime_duration,
                'regime_change_flag': regime_change_flag,
                
                # D. Trend metrics (3)
                'regime_trend_ratio': trend_ratio,
                'regime_trend_ratio_zscore': trend_ratio_zscore,
                'regime_volatility_20d': vol_20,
            }).fillna(0)
            
            # Filter to requested date range
            regime_df = regime_df[
                (regime_df.index >= pd.Timestamp(start)) & 
                (regime_df.index <= pd.Timestamp(end))
            ]
            
            # Tiingo provenance for regime classification
            prov = {'source': 'tiingo_price_data', 'method': 'regime_classification_institutional'}
            regime_df.attrs['telemetry'] = {'status': 'ok', 'source': 'tiingo_price_data'}
            regime_df.attrs['provenance'] = prov
            regime_df.attrs['feature_counts'] = {'generated': 9}
            
            return regime_df
        except Exception as exc:
            logger.debug("regime fallback failed for %s: %s", symbol, exc)
            return None

    def _h_options_anchoring():
        """
        Options-anchored features using EODHD options data.
        
        FETCH CADENCE: Daily (business days) - missing daily destroys IV/skew value!
        
        Returns 11 decay-weighted options-anchoring metrics (institutional-grade):
        
        A. IV ANCHORING (3 features) - FAST DECAY (half-life 3 days):
           - iv_anchor_pct: Z-score of ATM IV vs 20-day mean/std (overextension detector)
           - iv_percentile_30d: Percentile rank of ATM IV in last 30 days (fear regime)
           - iv_percentile_1yr: Percentile rank of ATM IV in last 1 year (long-term context)
        
        B. SKEW ANCHORING (3 features) - FAST DECAY (half-life 3 days):
           - iv_skew_anchor: OTM put IV - OTM call IV (raw downside fear)
           - iv_skew_zscore: Z-score of skew vs 30-day distribution (skew extremes)
           - risk_reversal_25d: 25-delta put IV - 25-delta call IV (Goldman/JPM standard)
        
        C. EXPECTED MOVE ANCHORING (2 features) - FAST DECAY (half-life 3 days):
           - expected_move_pct: Straddle-implied move as % of spot (event risk)
           - em_vs_real_vol_ratio: Expected move / realized vol (complacency detector)
        
        D. VOLUME POSITIONING (1 feature) - MEDIUM DECAY (half-life 5 days):
           - put_call_vol_ratio_anchor: EMA(3) of put/call volume (short-term sentiment)
        
        E. OI POSITIONING (1 feature) - SLOW DECAY (half-life 20 days):
           - put_call_oi_ratio_anchor: Put OI / call OI (slow-moving hedging demand)
        
        F. GOVERNANCE (1 feature):
           - options_anchoring_days_since_update: Days since last real fetch (freshness)
        
        DECAY WEIGHTING:
        - FAST (A,B,C): half-life 3 days - IV/skew decays quickly, daily fetch critical
        - MEDIUM (D): half-life 5 days - Volume reacts around events
        - SLOW (E): half-life 20 days - OI is structural, but fetch daily for breaks
        
        Source: EODHD Options API (daily business day cadence)
        Purpose: Institutional-grade options anchoring (Goldman/JPM/Susquehanna standard)
        
        NOTE: Decay weighting reduces stale data impact. Stage A learns relationships.
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🎯 options_anchoring: {symbol}")
            
            # Get EODHD provider
            eodhd = get_eodhd_provider()
            if not eodhd or not eodhd.api_key:
                logger.debug("options_anchoring: EODHD API key not configured")
                return None
            
            # Fetch price data (need historical for lookbacks)
            price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            if price_df is None or 'close' not in price_df.columns:
                logger.debug(f"options_anchoring: No price data for {symbol}")
                return None
            
            current_price = float(price_df['close'].iloc[-1])
            
            # Calculate realized volatility (for em_vs_real_vol_ratio)
            returns = price_df['close'].pct_change()
            realized_vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized %
            
            # ═══════════════════════════════════════════════════════════
            # FETCH HISTORICAL OPTIONS SNAPSHOTS (weekly intervals)
            # ═══════════════════════════════════════════════════════════
            # To calculate Z-scores and percentiles, we need historical ATM IV time series
            # Fetch options snapshots weekly (balance API calls vs data granularity)
            
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            # ═══════════════════════════════════════════════════════════
            # DAILY FETCH CADENCE (per Goldman/JPM institutional practice)
            # ═══════════════════════════════════════════════════════════
            # - IV/Skew/Expected Move (FAST): half-life 1-5 days - MUST fetch daily
            # - Volume ratios (MEDIUM): half-life 3-10 days - fetch daily
            # - OI ratios (SLOW): half-life 10-30 days - fetch daily for structural breaks
            sample_freq = os.getenv('EODHD_OPTIONS_ANCHOR_FREQ', 'B')  # Business days (daily)
            sample_dates = pd.date_range(start=start_str_default, end=end_str_default, freq=sample_freq)

            # Guardrail: very long coverages can still create ~1000+ snapshots per symbol.
            # Cap the number of API calls by downsampling deterministically.
            # Set to 0 to disable downsampling (fetch every business day).
            try:
                max_samples = int(os.getenv('EODHD_OPTIONS_ANCHOR_MAX_SAMPLES', '0'))
            except Exception:
                max_samples = 0
            if max_samples > 0 and len(sample_dates) > max_samples:
                step = int(np.ceil(len(sample_dates) / float(max_samples)))
                sample_dates = sample_dates[::max(1, step)]
                logger.info(
                    "options_anchoring: downsampled historical snapshots to %d dates (freq=%s, step=%d)",
                    int(len(sample_dates)),
                    str(sample_freq),
                    int(step),
                )
            
            if len(sample_dates) == 0:
                sample_dates = [pd.Timestamp(end_str_default)]

            # Resumable on-disk cache of *derived* snapshot metrics.
            # This keeps full timeframe coverage while avoiding repeated
            # re-download + re-parse of large option chains across windows/runs.
            from pathlib import Path
            import tempfile

            def _atomic_write_parquet(df_to_write: pd.DataFrame, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=str(path.parent),
                    delete=False,
                    prefix="._tmp_",
                    suffix=path.suffix or ".parquet",
                ) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    df_to_write.to_parquet(tmp_path)
                    tmp_path.replace(path)
                finally:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

            cache_root = Path(__file__).resolve().parents[2] / "data" / "cache" / "options_anchoring"
            cache_symbol = (symbol.replace(".", "_") or "unknown")
            cache_path = cache_root / f"{cache_symbol}_weekly_snapshots.parquet"

            wanted_dates = pd.to_datetime(pd.Index(sample_dates)).normalize()
            cached_snapshots = None
            if cache_path.exists():
                try:
                    cached_snapshots = pd.read_parquet(cache_path)
                    if cached_snapshots is not None and not cached_snapshots.empty:
                        if "date" in cached_snapshots.columns:
                            cached_snapshots["date"] = pd.to_datetime(cached_snapshots["date"]).dt.normalize()
                            cached_snapshots = cached_snapshots.set_index("date")
                        else:
                            cached_snapshots.index = pd.to_datetime(cached_snapshots.index).normalize()
                        cached_snapshots = cached_snapshots.sort_index()
                except Exception:
                    cached_snapshots = None

            if cached_snapshots is None:
                cached_snapshots = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

            missing_dates = [d for d in wanted_dates if d not in cached_snapshots.index]

            new_rows = []
            for idx, sample_date in enumerate(missing_dates):
                try:
                    options_df = eodhd.get_options(symbol, date=sample_date.strftime('%Y-%m-%d'))
                    if options_df is None or options_df.empty:
                        continue
                    
                    # Get price for this date (for moneyness calculations)
                    if sample_date in price_df.index:
                        current_price = float(price_df.loc[sample_date, 'close'])
                    else:
                        # Find nearest price
                        nearest_idx = price_df.index.get_indexer([sample_date], method='nearest')[0]
                        current_price = float(price_df.iloc[nearest_idx]['close'])
                    
                    # Calculate moneyness
                    options_df['moneyness'] = np.abs(options_df['strike'] - current_price)
                    options_df['moneyness_pct'] = (options_df['strike'] - current_price) / current_price
                    
                    features = {'date': sample_date}
                    
                    # Extract snapshot features for this date
                    if 'impliedVolatility' in options_df.columns and 'strike' in options_df.columns:
                        # ATM IV
                        atm_options = options_df.nsmallest(5, 'moneyness')
                        if len(atm_options) > 0:
                            features['atm_iv'] = atm_options['impliedVolatility'].mean()
                        
                        # Skew
                        calls = options_df[options_df['type'] == 'calls']
                        puts = options_df[options_df['type'] == 'puts']
                        
                        otm_puts = puts[puts['strike'] < current_price * 0.95]
                        otm_calls = calls[calls['strike'] > current_price * 1.05]
                        
                        if len(otm_puts) > 0 and len(otm_calls) > 0:
                            features['iv_skew_anchor'] = otm_puts['impliedVolatility'].mean() - otm_calls['impliedVolatility'].mean()
                        
                        # Risk reversal (25-delta)
                        put_25d = puts[(puts['moneyness_pct'] >= -0.12) & (puts['moneyness_pct'] <= -0.08)]
                        call_25d = calls[(calls['moneyness_pct'] >= 0.08) & (calls['moneyness_pct'] <= 0.12)]
                        
                        if len(put_25d) > 0 and len(call_25d) > 0:
                            features['risk_reversal_25d'] = put_25d['impliedVolatility'].mean() - call_25d['impliedVolatility'].mean()
                    
                    # Expected move (straddle-based)
                    if 'last' in options_df.columns:
                        atm_calls = options_df[(options_df['type'] == 'calls') & (options_df['moneyness'] < current_price * 0.02)]
                        atm_puts = options_df[(options_df['type'] == 'puts') & (options_df['moneyness'] < current_price * 0.02)]
                        
                        if len(atm_calls) > 0 and len(atm_puts) > 0:
                            straddle_price = atm_calls['last'].mean() + atm_puts['last'].mean()
                            features['expected_move_pct'] = (straddle_price / current_price) * 100
                            
                            # Expected move vs realized vol
                            if sample_date in realized_vol_20d.index:
                                rv = realized_vol_20d.loc[sample_date]
                                if not pd.isna(rv) and rv > 0:
                                    features['em_vs_real_vol_ratio'] = features['expected_move_pct'] / rv
                    
                    # Volume/OI ratios
                    if 'volume' in options_df.columns:
                        calls = options_df[options_df['type'] == 'calls']
                        puts = options_df[options_df['type'] == 'puts']
                        
                        call_vol = calls['volume'].sum() if len(calls) > 0 else 0
                        put_vol = puts['volume'].sum() if len(puts) > 0 else 0
                        
                        if call_vol > 0:
                            features['put_call_vol'] = put_vol / call_vol
                    
                    if 'openInterest' in options_df.columns:
                        calls = options_df[options_df['type'] == 'calls']
                        puts = options_df[options_df['type'] == 'puts']
                        
                        call_oi = calls['openInterest'].sum() if len(calls) > 0 else 0
                        put_oi = puts['openInterest'].sum() if len(puts) > 0 else 0
                        
                        if call_oi > 0:
                            features['put_call_oi_ratio_anchor'] = put_oi / call_oi
                    
                    new_rows.append(features)

                    # Periodically checkpoint to disk so long runs can resume.
                    if len(new_rows) >= 10 or (idx + 1) == len(missing_dates):
                        try:
                            chunk_df = pd.DataFrame(new_rows)
                            chunk_df["date"] = pd.to_datetime(chunk_df["date"]).dt.normalize()
                            chunk_df = chunk_df.set_index("date").sort_index()
                            cached_snapshots = pd.concat([cached_snapshots, chunk_df], axis=0)
                            cached_snapshots = cached_snapshots[~cached_snapshots.index.duplicated(keep="last")].sort_index()
                            _atomic_write_parquet(cached_snapshots.reset_index(), cache_path)
                            new_rows = []
                        except Exception as exc:
                            logger.debug("options_anchoring: snapshot cache write failed for %s: %s", symbol, exc)
                            new_rows = []

                except Exception as e:
                    logger.debug(f"Failed to fetch options for {sample_date}: {e}")
                    continue

            # Build time series from cached snapshots + any newly fetched rows
            snapshots_df = cached_snapshots.copy()
            if snapshots_df.empty:
                logger.debug(f"options_anchoring: No historical options data for {symbol}")
                return None
            
            # ═══════════════════════════════════════════════════════════
            # BUILD TIME SERIES FROM HISTORICAL SNAPSHOTS
            # ═══════════════════════════════════════════════════════════
            
            features_df = snapshots_df.copy()
            features_df.index = pd.to_datetime(features_df.index).normalize()
            features_df = features_df.sort_index()
            
            # Reindex to daily and interpolate/forward-fill
            features_df = features_df.reindex(date_range)
            
            # Forward-fill then back-fill to handle edges
            features_df = features_df.ffill().bfill()
            
            # ═══════════════════════════════════════════════════════════
            # DECAY HALF-LIVES BY FEATURE TYPE (Goldman/JPM cadence)
            # ═══════════════════════════════════════════════════════════
            # A) IV/Expected Move/Skew (FAST): half-life 3 days
            # B) Volume-based positioning (MEDIUM): half-life 5 days  
            # C) Open interest positioning (SLOW): half-life 20 days
            HALFLIFE_FAST = 3    # IV, skew, expected move
            HALFLIFE_MEDIUM = 5  # Volume ratios
            HALFLIFE_SLOW = 20   # OI ratios
            
            def _apply_decay_weight(series: pd.Series, halflife_days: int) -> pd.Series:
                """Apply exponential decay weighting. Recent data matters more."""
                # Calculate days since last real observation
                is_real = series.notna() & (series != 0)
                days_since = (~is_real).cumsum() - (~is_real).cumsum().where(is_real).ffill().fillna(0)
                # Exponential decay: weight = 0.5^(days/halflife)
                decay_factor = np.power(0.5, days_since / halflife_days)
                return series * decay_factor
            
            # ═══════════════════════════════════════════════════════════
            # CALCULATE ROLLING METRICS (Z-scores, Percentiles, EMAs)
            # ═══════════════════════════════════════════════════════════
            
            # A. IV ANCHORING
            if 'atm_iv' in features_df.columns:
                atm_iv_series = features_df['atm_iv']
                
                # iv_anchor_pct: Z-score vs 20-day mean/std
                atm_iv_mean_20d = atm_iv_series.rolling(20, min_periods=5).mean()
                atm_iv_std_20d = atm_iv_series.rolling(20, min_periods=5).std()
                features_df['iv_anchor_pct'] = (atm_iv_series - atm_iv_mean_20d) / atm_iv_std_20d
                
                # iv_percentile_30d: Percentile rank in last 30 days
                features_df['iv_percentile_30d'] = atm_iv_series.rolling(30, min_periods=10).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan,
                    raw=False
                )
                
                # iv_percentile_1yr: Percentile rank in last 252 days
                features_df['iv_percentile_1yr'] = atm_iv_series.rolling(252, min_periods=30).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan,
                    raw=False
                )
                
                # Drop raw ATM IV (keep only derived features)
                features_df.drop(columns=['atm_iv'], inplace=True)
            
            # B. SKEW ANCHORING
            if 'iv_skew_anchor' in features_df.columns:
                skew_series = features_df['iv_skew_anchor']
                
                # iv_skew_zscore: Z-score vs 30-day distribution
                skew_mean_30d = skew_series.rolling(30, min_periods=10).mean()
                skew_std_30d = skew_series.rolling(30, min_periods=10).std()
                features_df['iv_skew_zscore'] = (skew_series - skew_mean_30d) / skew_std_30d
            
            # D. POSITIONING/SENTIMENT
            if 'put_call_vol' in features_df.columns:
                # EMA(3) of put/call volume ratio - MEDIUM decay
                raw_vol_ratio = features_df['put_call_vol'].ewm(span=3, min_periods=1).mean()
                features_df['put_call_vol_ratio_anchor'] = _apply_decay_weight(raw_vol_ratio, HALFLIFE_MEDIUM)
                features_df.drop(columns=['put_call_vol'], inplace=True)
            
            # Apply decay to OI ratio (SLOW decay - structural positioning)
            if 'put_call_oi_ratio_anchor' in features_df.columns:
                features_df['put_call_oi_ratio_anchor'] = _apply_decay_weight(
                    features_df['put_call_oi_ratio_anchor'], HALFLIFE_SLOW
                )
            
            # Apply FAST decay to IV/skew/expected move features
            fast_decay_cols = ['iv_anchor_pct', 'iv_percentile_30d', 'iv_percentile_1yr',
                               'iv_skew_anchor', 'iv_skew_zscore', 'risk_reversal_25d',
                               'expected_move_pct', 'em_vs_real_vol_ratio']
            for col in fast_decay_cols:
                if col in features_df.columns:
                    features_df[col] = _apply_decay_weight(features_df[col], HALFLIFE_FAST)
            
            # ═══════════════════════════════════════════════════════════
            # GOVERNANCE: Track days_since_update per feature category
            # ═══════════════════════════════════════════════════════════
            # Helps downstream models know data freshness
            if 'atm_iv' in snapshots_df.columns or len(snapshots_df) > 0:
                # Days since last real options observation
                real_dates = snapshots_df.index
                features_df['options_anchoring_days_since_update'] = features_df.index.to_series().apply(
                    lambda d: min((d - real_dates).days) if len(real_dates) > 0 and d >= real_dates.min() else 999.0
                ).clip(lower=0).values
            else:
                features_df['options_anchoring_days_since_update'] = 999.0
            
            # Clean data
            features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            
            # Keep only final features (10 total)
            final_features = [
                'iv_anchor_pct', 'iv_percentile_30d', 'iv_percentile_1yr',  # A. IV Anchoring
                'iv_skew_anchor', 'iv_skew_zscore', 'risk_reversal_25d',    # B. Skew Anchoring
                'expected_move_pct', 'em_vs_real_vol_ratio',                # C. Expected Move
                'put_call_vol_ratio_anchor', 'put_call_oi_ratio_anchor'     # D. Positioning
            ]
            features_df = features_df[[col for col in final_features if col in features_df.columns]]
            
            features_df.attrs['provenance'] = {
                'source': 'eodhd_options_api',
                'fields': list(features_df.columns)
            }
            features_df.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'eodhd_options_api',
                'feature_type': 'institutional_options_anchoring',
                'num_features': len(features_df.columns)
            }
            
            logger.info(f"✅ options_anchoring: {symbol} - {len(features_df.columns)} institutional features")
            return features_df
            
        except Exception as e:
            logger.debug(f"options_anchoring failed for {symbol}: {e}")
            return None

    def _h_tft():
        """
        TFT-style features v2: Comprehensive temporal patterns for forecasting.
        
        Mimics what Temporal Fusion Transformers care about:
        - Multi-scale trend (short/medium/long + signal-to-noise)
        - Volatility regime & dynamics (levels + regime + trend)
        - Seasonality & calendar effects (weekly/monthly cycles)
        - Stability/predictability (short/long term noise levels)
        - Dynamic confidence based on market conditions
        
        ~14 features total, all engineered from EODHD price data.
        """
        try:
            # Fetch extended history for proper rolling calculations
            logger.debug(f"🔑 Attempting EODHD provider for TFT features: {symbol}")
            lookback_start = (pd.to_datetime(start or start_str_default) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            price_df = _fetch_price_data(symbol, lookback_start, end_str_default)
            
            if price_df is None or 'close' not in price_df.columns:
                logger.debug(f"tft_features: no price data for {symbol}")
                return None

            close = price_df['close'].astype(float)
            if len(close) < 30:
                logger.debug(f"tft_features: insufficient samples ({len(close)}) for {symbol}")
                return None

            returns = close.pct_change().fillna(0.0)
            
            # ================================================================
            # 1. MULTI-SCALE TREND
            # ================================================================
            # Short-term trend (5-10 day momentum bursts)
            ma_short = close.rolling(10, min_periods=5).mean()
            trend_short_strength = (ma_short.diff(5) / close).fillna(0.0)
            
            # Long-term trend (20-30 day structural alignment)
            ma_long = close.rolling(30, min_periods=15).mean()
            trend_long_strength = (ma_long.diff(10) / close).fillna(0.0)
            
            # Trend consistency: fraction of positive returns in last 20 days
            # ~1 = clean uptrend, ~0 = clean downtrend, ~0.5 = choppy
            trend_consistency = (returns > 0).rolling(20, min_periods=10).mean().fillna(0.5)
            
            # Signal-to-noise: |trend| / volatility
            # High = strong trend relative to noise (what models care about)
            rolling_std = returns.rolling(20, min_periods=10).std()
            trend_signal_to_noise = (np.abs(ma_long.diff(10)) / (rolling_std * close + 1e-8)).fillna(0.0)
            
            # Legacy blended trend importance (keep for compatibility)
            trend_importance = (np.abs(trend_short_strength) + np.abs(trend_long_strength)) / 2.0
            
            # ================================================================
            # 2. VOLATILITY REGIME & DYNAMICS
            # ================================================================
            # Short-term volatility (5-10 day realized vol)
            vol_short = returns.rolling(10, min_periods=5).std().fillna(0.0)
            
            # Long-term volatility (20-30 day baseline)
            vol_long = returns.rolling(30, min_periods=15).std().fillna(0.0)
            
            # Volatility regime z-score: (current vol - median vol) / std vol
            # Tells if we're in unusually high/low vol vs history
            vol_median = vol_long.rolling(252, min_periods=60).median()
            vol_std = vol_long.rolling(252, min_periods=60).std()
            vol_regime_zscore = ((vol_long - vol_median) / (vol_std + 1e-8)).fillna(0.0)
            
            # Volatility trend: is vol rising or calming?
            vol_trend = vol_long.diff(10).fillna(0.0)
            
            # Legacy volatility importance (keep for compatibility)
            volatility_importance = vol_long
            
            # ================================================================
            # 3. SEASONALITY & CALENDAR EFFECTS
            # ================================================================
            # Weekday effect: rolling average return by weekday
            weekday_effect = pd.Series(0.0, index=close.index)
            for i in range(len(close)):
                if i >= 126:  # Need 6 months history
                    recent_returns = returns.iloc[i-126:i]
                    recent_index = recent_returns.index
                    current_weekday = close.index[i].weekday()
                    weekday_returns = recent_returns[recent_index.weekday == current_weekday]
                    if len(weekday_returns) > 0:
                        weekday_effect.iloc[i] = weekday_returns.mean()
            
            # Month phase: normalized day-of-month in [-1, 0, +1]
            # Start of month = -1, mid = 0, end = +1
            month_phase = pd.Series([
                (day - 15) / 15.0 if day <= 15 else (day - 15) / 15.0
                for day in close.index.day
            ], index=close.index)
            month_phase = month_phase.clip(-1, 1)
            
            # Legacy seasonality importance (sine-based approximation)
            seasonality = pd.Series([np.sin(i * np.pi / 10) for i in range(len(close))], index=close.index)
            seasonality_importance = (seasonality * returns).abs().rolling(20, min_periods=10).mean().fillna(0.0)
            
            # ================================================================
            # 4. STABILITY / PREDICTABILITY
            # ================================================================
            # Short-term stability (5-10 days)
            stability_short = 1.0 / (1.0 + returns.abs().rolling(10, min_periods=5).mean())
            
            # Long-term stability (20-30 days)
            stability_long = 1.0 / (1.0 + returns.abs().rolling(30, min_periods=15).mean())
            
            # Legacy stability score (keep for compatibility)
            stability_score = stability_long
            
            # ================================================================
            # 5. DYNAMIC CONFIDENCE
            # ================================================================
            # High confidence when:
            # - Vol regime is near normal (not extreme event)
            # - Trend signal-to-noise is reasonable
            # - Stability is intermediate (not chaos, not dead)
            # - Sufficient history exists
            
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
            
            # Penalize extreme vol regimes (crashes or dead markets)
            base_conf = sigmoid(-np.abs(vol_regime_zscore))
            
            # Reward strong signal-to-noise (clear trends)
            trend_conf = sigmoid(trend_signal_to_noise - 0.5)
            
            # Combine components
            confidence = 0.3 + 0.4 * base_conf + 0.3 * trend_conf * stability_long
            confidence = confidence.clip(0.0, 1.0).fillna(0.5)
            
            # Penalize insufficient history (warmup period)
            warmup_mask = pd.Series(range(len(close)), index=close.index) < 30
            confidence[warmup_mask] = confidence[warmup_mask] * 0.5
            
            # ================================================================
            # ASSEMBLE FEATURE DATAFRAME
            # ================================================================
            tft_df = pd.DataFrame({
                # Multi-scale trend (5 features)
                'trend_short_strength': trend_short_strength,
                'trend_long_strength': trend_long_strength,
                'trend_consistency': trend_consistency,
                'trend_signal_to_noise': trend_signal_to_noise,
                'trend_importance': trend_importance,  # Legacy
                
                # Volatility regime (5 features)
                'vol_short': vol_short,
                'vol_long': vol_long,
                'vol_regime_zscore': vol_regime_zscore,
                'vol_trend': vol_trend,
                'volatility_importance': volatility_importance,  # Legacy
                
                # Seasonality & calendar (3 features)
                'weekday_effect': weekday_effect,
                'month_phase': month_phase,
                'seasonality_importance': seasonality_importance,  # Legacy
                
                # Stability (3 features)
                'stability_short': stability_short,
                'stability_long': stability_long,
                'stability_score': stability_score,  # Legacy
                
                # Dynamic confidence
                'CONFIDENCE': confidence,
                
                # Governance columns
                'has_data': 1.0,
            }, index=close.index)
            
            # Filter to requested date range
            tft_df = tft_df[
                (tft_df.index >= pd.to_datetime(start or start_str_default)) &
                (tft_df.index <= pd.to_datetime(end or end_str_default))
            ]
            
            if tft_df.empty:
                logger.debug(f"tft_features: no data in requested range for {symbol}")
                return None
            
            tft_df = tft_df.fillna(0.0)
            tft_df.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'EODHD',
                'tft_version': 'v2',
                'lookback_days': 365
            }
            tft_df.attrs['feature_counts'] = {'generated': len(tft_df.columns)}
            
            logger.info(f"✅ TFT v2: Generated {len(tft_df.columns)} features for {symbol}, shape={tft_df.shape}")
            return tft_df
            
        except Exception as exc:
            logger.debug("tft_features failed for %s: %s", symbol, exc)
            return None

    def _h_dividends():
        """
        HEDGE-FUND GRADE: Dividend Event Features (7 features)
        
        PHILOSOPHY: Event timing + intensity, NOT policy/quality
        
        Categories:
        1. Corporate Action (1): dividend_amount (winsorized special dividends)
        2. Income Metrics (2): dividend_yield_est, dividend_yield_zscore (z-scored cross-sectionally)
        3. Event Timing (3): days_to_ex_dividend, ex_dividend_window_strength, dividend_event_intensity
        4. Structural (1): dividend_frequency (classifier, NOT trading signal)
        
        REMOVED: ex_dividend_flag (binary flag → continuous window_strength)
        
        ADDED:
        - ex_dividend_window_strength: exp(-|days_to_ex_dividend| / τ) where τ=2.5 days
          (smooth event salience, no hard discontinuities)
        - dividend_event_intensity: dividend_amount / price × window_strength
          (large dividends matter more, time-localized impact)
        - dividend_yield_zscore: Cross-sectional normalization (cap spikes)
        
        GOVERNANCE:
        - dividend_amount: Never standalone, always paired with timing
        - dividend_frequency: Structural classifier (quarterly/monthly/irregular), NOT trading signal
        - Structural features belong in FIN_G7, not here
        
        Total: 8 features (event-focused, hedge-fund grade)
        """
        def _dividends_has_data_stub(note: str) -> pd.DataFrame:
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            stub = pd.DataFrame(index=date_range)
            stub['dividend_amount'] = 0.0
            stub['dividend_yield_est'] = 0.0
            stub['dividend_yield_zscore'] = 0.0
            stub['dividend_frequency'] = 0.0
            stub['days_to_ex_dividend'] = 999.0
            stub['ex_dividend_window_strength'] = 0.0
            stub['dividend_event_intensity'] = 0.0
            stub['dividend_event_stress'] = 0.0  # One-sided RISK signal
            stub['has_data'] = 0.0
            stub.attrs['provenance'] = {
                'source': 'eodhd_dividends_api',
                'note': note,
                'raw_records': 0,
                'ex_div_dates': 0,
                'hedge_fund_grade': True,
            }
            stub.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'eodhd',
                'records': 0,
                'has_data': 0,
                'note': note,
            }
            return stub

        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            logger.debug(f"🔑 Attempting EODHD provider for dividends: {symbol}")
            
            eodhd = get_eodhd_provider()
            if not eodhd.api_key:
                logger.debug("⚠️ EODHD: API key not configured for dividends")
                return _dividends_has_data_stub('no_api_key')
            
            # Fetch dividend history (extended for ex-div date calculation)
            # Need extra history to find next ex-div date
            import datetime as dt
            extended_start = (pd.to_datetime(start_str_default) - pd.DateOffset(days=180)).strftime('%Y-%m-%d')
            extended_end = (pd.to_datetime(end_str_default) + pd.DateOffset(days=180)).strftime('%Y-%m-%d')
            
            div_df = eodhd.get_dividends(
                symbol,
                start_date=extended_start,
                end_date=extended_end
            )
            
            if div_df is None or div_df.empty:
                logger.debug(f"⚠️ EODHD: No dividend data for {symbol}")
                return _dividends_has_data_stub('no_dividend_data')
            
            logger.info(f"✅ EODHD: Fetched {len(div_df)} dividend records for {symbol}")
            
            # Create daily series with dividend amounts
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            div_series = pd.DataFrame(index=date_range)
            div_series['dividend_amount'] = 0.0
            
            # Mark actual dividend dates (payment dates)
            for date, row in div_df.iterrows():
                if date in div_series.index:
                    div_series.loc[date, 'dividend_amount'] = row['value']
            
            # ================================================================
            # HEDGE-FUND: Winsorize extreme special dividends
            # ================================================================
            # Special dividends can be 10x normal - cap at 99th percentile
            div_amounts = div_series['dividend_amount'].replace(0, np.nan).dropna()
            if len(div_amounts) > 0:
                p99 = div_amounts.quantile(0.99)
                div_series['dividend_amount'] = div_series['dividend_amount'].clip(upper=p99)
            
            # Calculate rolling metrics
            div_series['dividend_yield_est'] = div_series['dividend_amount'].rolling(252, min_periods=1).sum()
            div_series['dividend_frequency'] = (div_series['dividend_amount'] > 0).rolling(252, min_periods=1).sum()
            
            # ================================================================
            # EX-DIVIDEND TIMING FEATURES (Event-Critical)
            # ================================================================
            
            # Get ex-dividend dates (EODHD index is typically ex-date or payment date)
            # For now, assume index is ex-dividend date (most common)
            ex_div_dates = div_df.index.tolist()
            
            # For each day, calculate days_to_ex_dividend (signed)
            div_series['days_to_ex_dividend'] = np.nan
            
            for current_date in div_series.index:
                # Find nearest ex-div dates
                future_ex_divs = [d for d in ex_div_dates if d >= current_date]
                past_ex_divs = [d for d in ex_div_dates if d < current_date]
                
                # Days to next ex-dividend (signed)
                if future_ex_divs:
                    next_ex_div = min(future_ex_divs)
                    div_series.loc[current_date, 'days_to_ex_dividend'] = (next_ex_div - current_date).days
                elif past_ex_divs:
                    # No future ex-div, use negative days from last ex-div
                    last_ex_div = max(past_ex_divs)
                    div_series.loc[current_date, 'days_to_ex_dividend'] = (last_ex_div - current_date).days
            
            # Forward-fill days_to_ex_dividend for continuity
            div_series['days_to_ex_dividend'] = div_series['days_to_ex_dividend'].fillna(method='ffill').fillna(999)
            
            # HEDGE-FUND: Clip to ±30 days (beyond that, event salience is negligible)
            div_series['days_to_ex_dividend'] = div_series['days_to_ex_dividend'].clip(-30, 30)
            
            # ================================================================
            # HEDGE-FUND: Smooth Event Window Strength (replaces binary flag)
            # ================================================================
            # ex_dividend_window_strength = exp(-|days_to_ex_dividend| / τ)
            # τ = 2.5 days (professional event model standard)
            # Peak = 1.0 at ex-date, decays smoothly to ~0.01 at ±10 days
            tau = 2.5
            div_series['ex_dividend_window_strength'] = np.exp(
                -np.abs(div_series['days_to_ex_dividend']) / tau
            )
            
            # ================================================================
            # HEDGE-FUND: Event Intensity (time-localized dividend impact)
            # ================================================================
            # dividend_event_intensity = dividend_amount / price × window_strength
            # Large dividends matter more, effect is time-localized
            
            # Need price data for normalization
            try:
                price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
                if price_df is not None and 'close' in price_df.columns:
                    # Align price with div_series
                    price_aligned = price_df['close'].reindex(div_series.index).ffill()
                    
                    # Calculate event intensity
                    div_series['dividend_event_intensity'] = (
                        div_series['dividend_amount'] / (price_aligned + 1e-9)
                    ) * div_series['ex_dividend_window_strength']
                    
                    # ================================================================
                    # HEDGE-FUND: Dividend Event Stress (one-sided RISK signal)
                    # ================================================================
                    # dividend_event_stress = window_strength × clip(|amount/price|, 0, 0.10)
                    # Only matters near ex-date; caps extreme special dividends at 10% yield
                    # This is a RISK signal for overlays (stress → reduce exposure)
                    div_pct = (div_series['dividend_amount'] / (price_aligned + 1e-9)).abs().clip(upper=0.10)
                    div_series['dividend_event_stress'] = div_series['ex_dividend_window_strength'] * div_pct * 10.0  # Scale to [0, 1]
                else:
                    # No price data - use raw amount × window_strength
                    div_series['dividend_event_intensity'] = (
                        div_series['dividend_amount'] * div_series['ex_dividend_window_strength']
                    )
                    div_series['dividend_event_stress'] = 0.0  # Can't compute without price
            except Exception:
                # Fallback: raw amount × window_strength
                div_series['dividend_event_intensity'] = (
                    div_series['dividend_amount'] * div_series['ex_dividend_window_strength']
                )
                div_series['dividend_event_stress'] = 0.0
            
            # ================================================================
            # HEDGE-FUND: Cross-sectional yield normalization (z-score)
            # ================================================================
            # Raw yield spikes can dominate - normalize to [-3, 3] z-score range
            yield_est = div_series['dividend_yield_est']
            if len(yield_est) > 20:  # Need minimum data for robust stats
                yield_mean = yield_est.rolling(252, min_periods=20).mean()
                yield_std = yield_est.rolling(252, min_periods=20).std()
                div_series['dividend_yield_zscore'] = (
                    (yield_est - yield_mean) / (yield_std + 1e-9)
                ).clip(-3, 3)
            else:
                div_series['dividend_yield_zscore'] = 0.0
            
            dividends = div_series[[
                'dividend_amount', 'dividend_yield_est', 'dividend_yield_zscore', 
                'dividend_frequency', 'days_to_ex_dividend', 
                'ex_dividend_window_strength', 'dividend_event_intensity',
                'dividend_event_stress'
            ]].copy()

            dividends['has_data'] = 1.0
            
            # Forward-fill all features (dividend events persist until next event)
            dividends = dividends.ffill()
            
            dividends.attrs['provenance'] = {
                'source': 'eodhd_dividends_api',
                'raw_records': len(div_df),
                'ex_div_dates': len(ex_div_dates),
                'hedge_fund_grade': True,
                'removed_features': ['ex_dividend_flag'],
                'added_features': ['ex_dividend_window_strength', 'dividend_event_intensity', 'dividend_yield_zscore'],
            }
            dividends.attrs['telemetry'] = {
                'status': 'ok', 
                'source': 'eodhd', 
                'records': len(div_df),
                'special_dividends_winsorized': True,
                'days_to_ex_clipped': '±30',
                'event_window_tau': 2.5,
            }
            dividends.attrs['feature_roles'] = {
                'corporate_action': ['dividend_amount'],
                'income_metrics': ['dividend_yield_est', 'dividend_yield_zscore'],
                'structural_classifier': ['dividend_frequency'],
                'event_timing': ['days_to_ex_dividend', 'ex_dividend_window_strength', 'dividend_event_intensity'],
                'risk_stress': ['dividend_event_stress'],  # One-sided stress for overlays
            }
            dividends.attrs['governance'] = {
                'dividend_amount_usage': 'Never standalone, always paired with timing features',
                'dividend_frequency_usage': 'Structural classifier (quarterly/monthly/irregular), NOT trading signal',
                'policy_features_location': 'FIN_G7 (dividend_policy_stability, payout_ratio, etc.)',
                'event_window_philosophy': 'Smooth continuous strength, not binary flags',
                'dividend_event_stress_usage': 'One-sided RISK signal: window_strength × clip(|amount/price|, 0, 0.10) × 10 → [0,1]',
            }
            return dividends
            
        except Exception as e:
            logger.debug(f"❌ dividends EODHD failed for {symbol}: {e}")
            return _dividends_has_data_stub('exception')

    def _fundamentals_has_data_stub(family: str) -> pd.DataFrame:
        date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
        stub = pd.DataFrame(index=date_range)
        stub['has_data'] = 0.0
        stub.attrs['provenance'] = {
            'source': 'eodhd_fundamentals_api',
            'note': 'no_financials',
            'family': family,
        }
        stub.attrs['telemetry'] = {
            'status': 'ok',
            'source': 'eodhd',
            'has_data': 0,
            'family': family,
        }
        return stub

    def _h_fin_g0():
        """
        Financial Group 0: Profitability Metrics from EODHD Fundamentals API
        ROE, ROA, Profit Margins, EBITDA Margin
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            logger.debug(f"🔑 Attempting EODHD provider for fundamentals: {symbol}")
            
            eodhd = get_eodhd_provider()
            if not eodhd.api_key:
                logger.debug("⚠️ EODHD: API key not configured for fundamentals")
                return None
            
            # Fetch comprehensive fundamentals
            fund_data = eodhd.get_fundamentals(symbol)
            if not fund_data or 'Highlights' not in fund_data:
                logger.debug(f"⚠️ EODHD: No fundamentals data for {symbol}")
                return None
            
            highlights = fund_data.get('Highlights', {})
            
            # Extract profitability metrics
            metrics = {
                'roe': highlights.get('ReturnOnEquityTTM'),
                'roa': highlights.get('ReturnOnAssetsTTM'),
                'profit_margin': highlights.get('ProfitMargin'),
                'operating_margin': highlights.get('OperatingMarginTTM'),
                'gross_margin': highlights.get('GrossProfitTTM'),
                'ebitda_margin': highlights.get('EBITDAMarginTTM'),
            }
            
            # Filter out None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            if not metrics:
                logger.debug(f"fin_g0: No profitability metrics found in EODHD for {symbol}")
                return None
            
            logger.info(f"✅ EODHD: Fetched {len(metrics)} profitability metrics for {symbol}")
            
            # Create time series (forward-fill quarterly data to daily)
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g0 = pd.DataFrame(index=date_range)
            
            for metric_name, value in metrics.items():
                fin_g0[metric_name] = float(value) if value is not None else np.nan
            
            fin_g0 = fin_g0.replace([np.inf, -np.inf], np.nan).ffill()
            fin_g0.attrs['provenance'] = {
                'source': 'eodhd_fundamentals_api',
                'fields': list(metrics.keys()),
            }
            fin_g0.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'fields': list(metrics.keys())}
            return fin_g0
            
        except Exception as e:
            logger.debug(f"❌ fin_g0 EODHD failed for {symbol}: {e}")
            return None

    def _h_fin_g1():
        """
        G1 — Liquidity Metrics (STRICT: Only liquidity ratios)
        - Current Ratio = Current Assets / Current Liabilities
        - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        - Cash Ratio = Cash / Current Liabilities
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G1 Liquidity: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data or 'Financials' not in fund_data:
                logger.debug(f"⚠️ G1: No fundamentals for {symbol}")
                return None
            
            # Get latest quarterly balance sheet
            bs = fund_data['Financials'].get('Balance_Sheet', {})
            quarterly = bs.get('quarterly', {})
            
            if not quarterly:
                logger.debug(f"⚠️ G1: No quarterly balance sheet for {symbol}")
                return None
            
            # Build quarterly history (ratios per quarter)
            historical_data = []
            for quarter_date in sorted(quarterly.keys()):
                q = quarterly[quarter_date]
                current_assets = float(q.get('totalCurrentAssets', 0)) if q.get('totalCurrentAssets') else None
                current_liabilities = float(q.get('totalCurrentLiabilities', 0)) if q.get('totalCurrentLiabilities') else None
                cash = float(q.get('cash', 0)) if q.get('cash') else None
                inventory = float(q.get('inventory', 0)) if q.get('inventory') else 0
                
                row = {'date': quarter_date}
                if current_assets and current_liabilities and current_liabilities != 0:
                    row['current_ratio'] = current_assets / current_liabilities
                    row['quick_ratio'] = (current_assets - inventory) / current_liabilities
                if cash and current_liabilities and current_liabilities != 0:
                    row['cash_ratio'] = cash / current_liabilities
                
                if len(row) > 1:
                    historical_data.append(row)

            if not historical_data:
                logger.debug(f"fin_g1: No usable quarterly balance-sheet history for {symbol}")
                return None

            df_hist = pd.DataFrame(historical_data)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist = df_hist.set_index('date').sort_index()

            # Latest quarter for provenance
            latest_date = df_hist.index.max().strftime('%Y-%m-%d')

            # Daily time series (forward-fill quarterly values to daily)
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g1 = df_hist.reindex(date_range).ffill().bfill()
            
            # Add trend & normalization features if we have historical data
            # Add trend & normalization features as scalars (broadcast to daily)
            metrics = {}
            if 'current_ratio' in df_hist.columns and len(df_hist) >= 4:
                lookback = min(12, len(df_hist))
                recent = df_hist['current_ratio'].iloc[-lookback:]
                if len(recent) >= 2:
                    x = np.arange(len(recent))
                    y = recent.values
                    valid = ~np.isnan(y)
                    if valid.sum() >= 2:
                        slope, _ = np.polyfit(x[valid], y[valid], 1)
                        metrics['liquidity_trend_3y'] = float(slope)

                lookback = min(20, len(df_hist))
                historical = df_hist['current_ratio'].iloc[-lookback:]
                mean = historical.mean()
                std = historical.std()
                if std and std > 1e-6:
                    current_val = df_hist['current_ratio'].iloc[-1]
                    metrics['liquidity_zscore_5y'] = float((current_val - mean) / std)

            for metric_name, value in metrics.items():
                fin_g1[metric_name] = value

            # ----------------------------------------------------------------
            # STRESS FEATURES (Jan 2026): Higher = worse (for risk aggregation)
            # CRITICAL: RoleAwareContext risk_scale uses 1/(1+risk_agg), so
            # raw ratios where higher=safer would INVERT the meaning.
            # These stress features ensure "higher unitized value = more risk".
            # ----------------------------------------------------------------
            # Liquidity stress: 1 / current_ratio, clipped to [0, 2]
            # When current_ratio < 1, stress > 1 (danger zone)
            # When current_ratio > 2, stress < 0.5 (safe)
            if 'current_ratio' in fin_g1.columns:
                cr = pd.to_numeric(fin_g1['current_ratio'], errors='coerce').clip(lower=0.1)
                fin_g1['liquidity_stress'] = np.clip(1.0 / cr, 0.0, 2.0)

            # Quick stress: same logic for quick_ratio
            if 'quick_ratio' in fin_g1.columns:
                qr = pd.to_numeric(fin_g1['quick_ratio'], errors='coerce').clip(lower=0.1)
                fin_g1['quick_stress'] = np.clip(1.0 / qr, 0.0, 2.0)

            # Cash stress: same logic for cash_ratio
            if 'cash_ratio' in fin_g1.columns:
                cashr = pd.to_numeric(fin_g1['cash_ratio'], errors='coerce').clip(lower=0.05)
                fin_g1['cash_stress'] = np.clip(1.0 / cashr, 0.0, 4.0)

            # Liquidity trend stress: negative trend = deterioration = higher stress
            if 'liquidity_trend_3y' in fin_g1.columns:
                trend = pd.to_numeric(fin_g1['liquidity_trend_3y'], errors='coerce').fillna(0.0)
                # Negative trend → positive stress; positive trend → zero stress
                fin_g1['liquidity_trend_stress'] = np.clip(-trend, 0.0, 1.0)

            # Family-scoped has_data: fundamentals are valid whenever available.
            fin_g1["has_data"] = 1.0
            
            # Track days since last quarterly update (forward-fill from latest_date)
            latest_update = pd.to_datetime(latest_date)
            fin_g1['days_since_update'] = np.maximum((fin_g1.index - latest_update).days, 0)
            
            fin_g1 = fin_g1.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            fin_g1.attrs['provenance'] = {
                'source': 'eodhd_fundamentals_api',
                'fields': list(fin_g1.columns),
                'quarter': latest_date,
            }
            fin_g1.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'fields': list(fin_g1.columns)}
            return fin_g1
            
        except Exception as e:
            logger.debug(f"❌ fin_g1 EODHD failed for {symbol}: {e}")
            return None

    def _h_fin_g2():
        """
        G2 — Leverage (Capital Structure) (STRICT: Only debt/equity structure)
        - Debt/Equity Ratio = Total Debt / Total Equity
        - Total Debt (log-transformed for magnitude)
        - Long-Term Debt
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G2 Leverage: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G2: No fundamentals for {symbol}")
                return _fundamentals_has_data_stub('fin_g2')
            
            # Get quarterly balance sheet data
            financials = fund_data.get('Financials', {})
            balance_sheet = financials.get('Balance_Sheet', {}).get('quarterly', {})
            
            if not balance_sheet:
                logger.debug(f"fin_g2: No quarterly balance sheet for {symbol}")
                return _fundamentals_has_data_stub('fin_g2')
            
            # Build time series from quarterly data
            quarterly_data = []
            for quarter_date in sorted(balance_sheet.keys()):
                bal = balance_sheet[quarter_date]
                
                # Extract leverage metrics
                total_assets = float(bal.get('totalAssets', 0)) if bal.get('totalAssets') else None
                total_equity = float(bal.get('totalStockholderEquity', 0)) if bal.get('totalStockholderEquity') else None
                long_term_debt = float(bal.get('longTermDebt', 0)) if bal.get('longTermDebt') else None
                short_term_debt = float(bal.get('shortTermDebt', 0)) if bal.get('shortTermDebt') else None
                total_liabilities = float(bal.get('totalLiab', 0)) if bal.get('totalLiab') else None
                
                row = {'date': quarter_date}
                
                # Calculate total debt
                total_debt = 0.0
                if long_term_debt:
                    total_debt += long_term_debt
                if short_term_debt:
                    total_debt += short_term_debt
                
                if total_debt > 0:
                    row['total_debt'] = total_debt
                    
                if long_term_debt:
                    row['long_term_debt'] = long_term_debt
                
                # Debt to Equity = Total Debt / Total Equity
                if total_debt > 0 and total_equity and total_equity != 0:
                    row['debt_to_equity'] = total_debt / total_equity
                
                # Debt to Assets = Total Debt / Total Assets
                if total_debt > 0 and total_assets and total_assets != 0:
                    row['debt_to_assets'] = total_debt / total_assets
                
                # Get income statement for EBITDA and interest expense
                income_stmt = fund_data.get('Financials', {}).get('Income_Statement', {}).get('quarterly', {})
                if quarter_date in income_stmt:
                    inc = income_stmt[quarter_date]
                    ebitda = float(inc.get('ebitda', 0)) if inc.get('ebitda') else None
                    interest_expense = float(inc.get('interestExpense', 0)) if inc.get('interestExpense') else None
                    operating_income = float(inc.get('operatingIncome', 0)) if inc.get('operatingIncome') else None
                    
                    # Get cash flow for FCF
                    free_cash_flow = None
                    cash_flow = fund_data.get('Financials', {}).get('Cash_Flow', {}).get('quarterly', {})
                    if quarter_date in cash_flow:
                        cf = cash_flow[quarter_date]
                        operating_cf = float(cf.get('totalCashFromOperatingActivities', 0)) if cf.get('totalCashFromOperatingActivities') else None
                        capex = float(cf.get('capitalExpenditures', 0)) if cf.get('capitalExpenditures') else None
                        
                        if operating_cf and capex:
                            free_cash_flow = operating_cf + capex  # capex is typically negative
                    
                    # Calculate cash (for net debt)
                    cash_val = float(bal.get('cash', 0)) if bal.get('cash') else 0
                    net_debt = total_debt - cash_val if total_debt > 0 else 0
                    
                    # Net Debt / EBITDA
                    if net_debt > 0 and ebitda and ebitda > 0:
                        row['net_debt_to_ebitda'] = net_debt / ebitda
                    
                    # Net Debt / FCF
                    if net_debt > 0 and free_cash_flow and free_cash_flow > 0:
                        row['net_debt_to_fcf'] = net_debt / free_cash_flow
                    
                    # Interest Burden (Interest Expense / EBIT)
                    if interest_expense and operating_income and operating_income > 0:
                        row['interest_burden'] = interest_expense / operating_income
                    
                    # Interest Coverage (EBIT / Interest Expense)
                    if interest_expense and interest_expense > 0 and operating_income:
                        row['interest_coverage'] = operating_income / interest_expense
                    
                    # Equity Multiplier (Total Assets / Total Equity)
                    if total_assets and total_equity and total_equity != 0:
                        row['equity_multiplier'] = total_assets / total_equity
                
                if len(row) > 1:  # Has at least one metric
                    quarterly_data.append(row)
            
            if not quarterly_data:
                logger.debug(f"fin_g2: Could not calculate leverage metrics for {symbol}")
                return _fundamentals_has_data_stub('fin_g2')
            
            # Convert to DataFrame and resample to daily with forward-fill
            df_quarterly = pd.DataFrame(quarterly_data)
            df_quarterly['date'] = pd.to_datetime(df_quarterly['date'])
            df_quarterly = df_quarterly.set_index('date').sort_index()
            
            # Resample to daily frequency
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g2 = df_quarterly.reindex(date_range, method='ffill')
            
            # Add leverage trend & z-score from quarterly data before daily resampling
            if 'debt_to_equity' in df_quarterly.columns and len(df_quarterly) >= 4:
                lookback = min(12, len(df_quarterly))  # 3 years = 12 quarters
                recent = df_quarterly['debt_to_equity'].iloc[-lookback:]
                if len(recent) >= 2 and not recent.isna().all():
                    x = np.arange(len(recent))
                    y = recent.values
                    valid = ~np.isnan(y)
                    if valid.sum() >= 2:
                        slope, _ = np.polyfit(x[valid], y[valid], 1)
                        fin_g2['leverage_trend_3y'] = slope
                
                # 5-year z-score (20 quarters)
                lookback_z = min(20, len(df_quarterly))
                historical = df_quarterly['debt_to_equity'].iloc[-lookback_z:]
                mean = historical.mean()
                std = historical.std()
                if std > 1e-6:
                    fin_g2['leverage_zscore_5y'] = (df_quarterly['debt_to_equity'].iloc[-1] - mean) / std
            
            fin_g2 = fin_g2.replace([np.inf, -np.inf], np.nan).ffill()

            fin_g2['has_data'] = 1.0
            
            # Track days since last quarterly update
            latest_quarter = df_quarterly.index.max()
            fin_g2['days_since_update'] = np.maximum((fin_g2.index - latest_quarter).days, 0)

            # ----------------------------------------------------------------
            # STRESS FEATURES (Jan 2026): Higher = worse (for risk aggregation)
            # CRITICAL: interest_coverage is HIGHER = SAFER, so we invert it.
            # Debt ratios (debt_to_equity, etc.) are already higher = worse.
            # We use robust signed_log1p transforms for exploding ratios.
            # ----------------------------------------------------------------
            def _signed_log1p(x: pd.Series) -> pd.Series:
                """Robust transform: sign(x) * log1p(|x|) for exploding ratios."""
                return np.sign(x) * np.log1p(np.abs(x))

            # Interest coverage stress: 1 / (1 + max(coverage, 0))
            # When coverage is high (>5), stress is low (<0.17)
            # When coverage is low (<2), stress is high (>0.33)
            # When coverage is negative (distressed), stress = 1.0
            if 'interest_coverage' in fin_g2.columns:
                ic = pd.to_numeric(fin_g2['interest_coverage'], errors='coerce').fillna(0.0)
                fin_g2['interest_coverage_stress'] = np.clip(1.0 / (1.0 + np.maximum(ic, 0.0)), 0.0, 1.0)

            # Robust debt_to_equity (can explode with small/negative equity)
            if 'debt_to_equity' in fin_g2.columns:
                dte = pd.to_numeric(fin_g2['debt_to_equity'], errors='coerce').fillna(0.0)
                fin_g2['debt_to_equity_robust'] = _signed_log1p(dte)

            # Robust net_debt_to_ebitda (can explode with small/negative EBITDA)
            if 'net_debt_to_ebitda' in fin_g2.columns:
                nde = pd.to_numeric(fin_g2['net_debt_to_ebitda'], errors='coerce').fillna(0.0)
                fin_g2['net_debt_to_ebitda_robust'] = _signed_log1p(nde)
            
            # Rescale large magnitude columns (e.g., TOTAL_DEBT)
            rescaled_cols: List[str] = []
            for col in fin_g2.columns:
                series = pd.to_numeric(fin_g2[col], errors='coerce')
                median_mag = series.abs().median()
                if np.isfinite(median_mag) and median_mag > 1e6:
                    with np.errstate(invalid='ignore'):
                        transformed = np.sign(series) * np.log1p(series.abs())
                    fin_g2[col] = transformed
                    rescaled_cols.append(col)
            
            logger.info(f"✅ EODHD: Built {len(df_quarterly)} quarters → {len(fin_g2)} days leverage metrics for {symbol}")
            
            fin_g2.attrs['provenance'] = {'source': 'eodhd_fundamentals_api', 'quarters': len(df_quarterly)}
            telemetry = {'status': 'ok', 'source': 'eodhd', 'quarters': len(df_quarterly)}
            if rescaled_cols:
                telemetry['scaled_columns'] = rescaled_cols
            telemetry['has_data'] = 1
            fin_g2.attrs['telemetry'] = telemetry
            return fin_g2
            
        except Exception as e:
            logger.debug(f"❌ fin_g2 EODHD failed for {symbol}: {e}")
            return None

    def _h_fin_g3():
        """
        G3 — Efficiency (Operations) (STRICT: Only operational turnover ratios)
        - Inventory Turnover = COGS / Average Inventory
        - Asset Turnover = Revenue / Total Assets
        - Receivables Turnover = Revenue / Accounts Receivable
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G3 Efficiency: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G3: No fundamentals for {symbol}")
                return None
            
            # Get quarterly financial statements
            financials = fund_data.get('Financials', {})
            income_stmt = financials.get('Income_Statement', {}).get('quarterly', {})
            balance_sheet = financials.get('Balance_Sheet', {}).get('quarterly', {})
            
            if not income_stmt or not balance_sheet:
                logger.debug(f"fin_g3: No quarterly financials for {symbol}")
                return _fundamentals_has_data_stub('fin_g3')
            
            # Build time series from quarterly data
            quarterly_data = []
            sorted_quarters = sorted(income_stmt.keys())
            
            for i, quarter_date in enumerate(sorted_quarters):
                if quarter_date not in balance_sheet:
                    continue
                
                inc = income_stmt[quarter_date]
                bal = balance_sheet[quarter_date]
                
                row = {'date': quarter_date}
                
                # STRICT G3: Only turnover/efficiency ratios
                
                # Asset Turnover = Revenue / Total Assets
                revenue = float(inc.get('totalRevenue', 0)) if inc.get('totalRevenue') else None
                assets = float(bal.get('totalAssets', 0)) if bal.get('totalAssets') else None
                
                if revenue and assets and assets != 0:
                    row['asset_turnover'] = revenue / assets
                
                # Inventory Turnover = COGS / Inventory
                cogs = float(inc.get('costOfRevenue', 0)) if inc.get('costOfRevenue') else None
                inventory = float(bal.get('inventory', 0)) if bal.get('inventory') else None
                
                if cogs and inventory and inventory != 0:
                    row['inventory_turnover'] = cogs / inventory
                
                # Receivables Turnover = Revenue / Accounts Receivable
                accounts_receivable = float(bal.get('netReceivables', 0)) if bal.get('netReceivables') else None
                accounts_payable = float(bal.get('accountsPayable', 0)) if bal.get('accountsPayable') else None
                
                if revenue and accounts_receivable and accounts_receivable != 0:
                    row['receivables_turnover'] = revenue / accounts_receivable
                    # Days Sales Outstanding (DSO) = 365 / Receivables Turnover
                    row['dsos'] = 365.0 / row['receivables_turnover']
                
                # Days Payables Outstanding (DPO) = 365 / Payables Turnover
                if cogs and accounts_payable and accounts_payable != 0:
                    payables_turnover = cogs / accounts_payable
                    row['payables_turnover'] = payables_turnover
                    row['dpos'] = 365.0 / payables_turnover
                
                # Days Inventory Outstanding (DIO) = 365 / Inventory Turnover
                if 'inventory_turnover' in row and row['inventory_turnover'] > 0:
                    row['dios'] = 365.0 / row['inventory_turnover']
                
                if len(row) > 1:  # Has at least one metric
                    quarterly_data.append(row)
            
            if not quarterly_data:
                logger.debug(f"fin_g3: Could not calculate efficiency metrics for {symbol}")
                return _fundamentals_has_data_stub('fin_g3')
            
            # Convert to DataFrame and resample to daily with forward-fill
            df_quarterly = pd.DataFrame(quarterly_data)
            df_quarterly['date'] = pd.to_datetime(df_quarterly['date'])
            df_quarterly = df_quarterly.set_index('date').sort_index()
            
            # Resample to daily frequency
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g3 = df_quarterly.reindex(date_range, method='ffill')
            
            # Add turnover volatility (stability metric)
            if 'asset_turnover' in df_quarterly.columns and len(df_quarterly) >= 4:
                lookback = min(12, len(df_quarterly))  # 3 years
                recent = df_quarterly['asset_turnover'].iloc[-lookback:]
                if len(recent) >= 2:
                    turnover_vol = recent.std()
                    fin_g3['turnover_volatility_3y'] = turnover_vol
            
            fin_g3 = fin_g3.replace([np.inf, -np.inf], np.nan).ffill()

            # ----------------------------------------------------------------
            # STRESS FEATURES (Jan 2026): Higher = worse (for risk aggregation)
            # Cash Conversion Cycle (CCC) = DSO + DIO - DPO
            # Higher CCC = more cash tied up in operations = worse
            # Turnover volatility already higher = worse (instability)
            # ----------------------------------------------------------------
            # Cash Conversion Cycle: DSO + DIO - DPO
            dso = pd.to_numeric(fin_g3.get('dsos', 0), errors='coerce').fillna(0.0)
            dio = pd.to_numeric(fin_g3.get('dios', 0), errors='coerce').fillna(0.0)
            dpo = pd.to_numeric(fin_g3.get('dpos', 0), errors='coerce').fillna(0.0)
            ccc = dso + dio - dpo
            fin_g3['ccc'] = ccc

            # CCC stress: normalize CCC to [0, 1] range
            # CCC > 90 days is stressed, CCC < 30 days is healthy
            # Use sigmoid-like transform: 1 / (1 + exp(-(ccc - 60) / 30))
            with np.errstate(over='ignore', invalid='ignore'):
                ccc_centered = (ccc - 60.0) / 30.0
                fin_g3['ccc_stress'] = np.clip(1.0 / (1.0 + np.exp(-ccc_centered)), 0.0, 1.0)

            fin_g3['has_data'] = 1.0
            
            # Track days since last quarterly update
            latest_quarter = df_quarterly.index.max()
            fin_g3['days_since_update'] = np.maximum((fin_g3.index - latest_quarter).days, 0)
            
            logger.info(f"✅ EODHD: Built {len(df_quarterly)} quarters → {len(fin_g3)} days efficiency metrics for {symbol}")
            
            fin_g3.attrs['provenance'] = {'source': 'eodhd_fundamentals_api', 'quarters': len(df_quarterly)}
            fin_g3.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'quarters': len(df_quarterly), 'has_data': 1}
            return fin_g3
            
        except Exception as e:
            logger.debug(f"❌ fin_g3 EODHD failed for {symbol}: {e}")
            return None

    def _h_fin_g4():
        """
        G4 — Cash Flow Metrics (Enhanced with earnings quality)
        - Operating Cash Flow
        - Free Cash Flow (OCF - CapEx)
        - FCF to Revenue, FCF to Net Income
        - Cash Conversion Cycle
        - CapEx to Revenue
        - Accruals Ratio (earnings quality)
        - CFO to Net Income (quality)
        - FCF Margin
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G4 Cash Flow: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G4: No fundamentals for {symbol}")
                return _fundamentals_has_data_stub('fin_g4')
            
            # Get quarterly financial statements
            financials = fund_data.get('Financials', {})
            income_stmt = financials.get('Income_Statement', {}).get('quarterly', {})
            balance_sheet = financials.get('Balance_Sheet', {}).get('quarterly', {})
            cash_flow = financials.get('Cash_Flow', {}).get('quarterly', {})
            
            if not income_stmt or not balance_sheet:
                logger.debug(f"G4: No quarterly financials for {symbol}")
                return _fundamentals_has_data_stub('fin_g4')
            
            # Build time series from quarterly data
            quarterly_data = []
            
            for quarter_date in sorted(income_stmt.keys()):
                if quarter_date not in balance_sheet:
                    continue
                
                inc = income_stmt[quarter_date]
                bal = balance_sheet[quarter_date]
                cf = cash_flow.get(quarter_date, {}) if cash_flow else {}
                
                row = {'date': quarter_date}
                
                # Extract necessary values
                net_income = float(inc.get('netIncome', 0)) if inc.get('netIncome') else None
                revenue = float(inc.get('totalRevenue', 0)) if inc.get('totalRevenue') else None
                total_assets = float(bal.get('totalAssets', 0)) if bal.get('totalAssets') else None
                
                # Cash flow metrics
                operating_cash_flow = float(cf.get('totalCashFromOperatingActivities', 0)) if cf.get('totalCashFromOperatingActivities') else None
                capex = float(cf.get('capitalExpenditures', 0)) if cf.get('capitalExpenditures') else None
                
                # Operating Cash Flow
                if operating_cash_flow:
                    row['operating_cash_flow'] = operating_cash_flow
                
                # Free Cash Flow = OCF - CapEx (capex typically negative)
                if operating_cash_flow and capex:
                    free_cash_flow = operating_cash_flow + capex  # capex is negative
                    row['free_cash_flow'] = free_cash_flow
                    
                    # FCF to Revenue
                    if revenue and revenue != 0:
                        row['fcf_to_revenue'] = free_cash_flow / revenue
                    
                    # FCF to Net Income (quality metric)
                    if net_income and net_income != 0:
                        row['fcf_to_net_income'] = free_cash_flow / net_income
                    
                    # FCF Margin
                    if revenue and revenue != 0:
                        row['fcf_margin'] = free_cash_flow / revenue
                
                # CapEx to Revenue (capital intensity)
                if capex and revenue and revenue != 0:
                    row['capex_to_revenue'] = abs(capex) / revenue
                
                # Accruals Ratio = (Net Income - Operating CF) / Total Assets
                # High accruals = lower earnings quality
                if net_income and operating_cash_flow and total_assets and total_assets != 0:
                    accruals = net_income - operating_cash_flow
                    row['accruals_ratio'] = accruals / total_assets
                
                # CFO to Net Income (quality of earnings)
                # Higher ratio = better quality (cash-backed earnings)
                if operating_cash_flow and net_income and net_income != 0:
                    row['cfo_to_net_income'] = operating_cash_flow / net_income
                
                # ================================================================
                # SCALE-FREE CASH FLOW METRICS
                # Raw OCF/FCF are in dollars and NOT cross-sectionally comparable.
                # We need normalized versions for risk aggregation.
                # ================================================================
                
                # CFO to Assets: Scale-free cash generation efficiency
                if operating_cash_flow and total_assets and total_assets != 0:
                    row['cfo_to_assets'] = operating_cash_flow / total_assets
                
                # FCF to Assets: Scale-free free cash flow efficiency
                if operating_cash_flow and capex and total_assets and total_assets != 0:
                    free_cash_flow = operating_cash_flow + capex  # capex is negative
                    row['fcf_to_assets'] = free_cash_flow / total_assets
                
                # CFO Margin: OCF / Revenue (like profit margin but cash-based)
                if operating_cash_flow and revenue and revenue != 0:
                    row['cfo_margin'] = operating_cash_flow / revenue
                
                # Cash Conversion Cycle (if we have days metrics from G3)
                # CCC = DSO + DIO - DPO
                # (Will be calculated in G3 if all components available)
                
                if len(row) > 1:
                    quarterly_data.append(row)
            
            if not quarterly_data:
                logger.debug(f"G4: Could not calculate profitability metrics for {symbol}")
                return _fundamentals_has_data_stub('fin_g4')
            
            # Convert to DataFrame and resample to daily
            df_quarterly = pd.DataFrame(quarterly_data)
            df_quarterly['date'] = pd.to_datetime(df_quarterly['date'])
            df_quarterly = df_quarterly.set_index('date').sort_index()
            
            # ================================================================
            # TIME-SERIES Z-SCORES FOR RISK AGGREGATION
            # Rolling 3-year z-scores for key cash flow metrics.
            # Higher = improving cash position vs own history.
            # For RISK: we compute STRESS features (lower z = more stress).
            # ================================================================
            scale_free_cols = ['cfo_to_assets', 'fcf_to_assets', 'cfo_margin', 'fcf_margin']
            for col in scale_free_cols:
                if col in df_quarterly.columns:
                    # Rolling z-score (12 quarters = 3 years)
                    rolling_mean = df_quarterly[col].rolling(12, min_periods=4).mean()
                    rolling_std = df_quarterly[col].rolling(12, min_periods=4).std()
                    df_quarterly[f'{col}_zscore_3y'] = (df_quarterly[col] - rolling_mean) / (rolling_std + 1e-9)
            
            # STRESS FEATURES for portfolio risk (higher = worse)
            # cash_stress: deterioration in cash flow quality
            if 'cfo_to_assets_zscore_3y' in df_quarterly.columns:
                # Negative z-score = deteriorating cash flow = STRESS
                df_quarterly['cash_flow_stress'] = df_quarterly['cfo_to_assets_zscore_3y'].apply(
                    lambda x: max(0.0, -x) if pd.notna(x) else 0.0
                )
            
            # earnings_quality_stress: high accruals = low quality = stress
            if 'accruals_ratio' in df_quarterly.columns:
                # Higher accruals ratio = lower earnings quality = stress
                df_quarterly['earnings_quality_stress'] = df_quarterly['accruals_ratio'].apply(
                    lambda x: max(0.0, x) if pd.notna(x) else 0.0
                )
            
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g4 = df_quarterly.reindex(date_range, method='ffill')
            fin_g4 = fin_g4.replace([np.inf, -np.inf], np.nan).ffill()

            fin_g4['has_data'] = 1.0
            
            # Track days since last quarterly cash flow report
            latest_quarter = df_quarterly.index.max()
            fin_g4['days_since_update'] = np.maximum((fin_g4.index - latest_quarter).days, 0)
            
            logger.info(f"✅ G4: Built {len(df_quarterly)} quarters → {len(fin_g4)} days cash flow metrics for {symbol}")
            
            fin_g4.attrs['provenance'] = {'source': 'eodhd_fundamentals_api', 'quarters': len(df_quarterly)}
            fin_g4.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'quarters': len(df_quarterly), 'has_data': 1}
            return fin_g4
            
        except Exception as e:
            logger.debug(f"❌ G4 failed for {symbol}: {e}")
            return None

    def _h_fin_g5():
        """
        G5 — Growth (STRICT: Only growth metrics)
        - Revenue Growth YoY
        - EBITDA Growth YoY
        - EPS Growth YoY
        - Cash Flow Growth YoY
        - 3-Year Revenue CAGR
        - Margin Expansion (change in net margin)
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G5 Growth: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G5: No fundamentals for {symbol}")
                return _fundamentals_has_data_stub('fin_g5')
            
            # Get quarterly financial statements
            financials = fund_data.get('Financials', {})
            income_stmt = financials.get('Income_Statement', {}).get('quarterly', {})
            cash_flow = financials.get('Cash_Flow', {}).get('quarterly', {})
            
            if not income_stmt:
                logger.debug(f"G5: No quarterly income statement for {symbol}")
                return _fundamentals_has_data_stub('fin_g5')
            
            # Build time series from quarterly data
            quarterly_data = []
            sorted_quarters = sorted(income_stmt.keys())
            
            for i, quarter_date in enumerate(sorted_quarters):
                inc = income_stmt[quarter_date]
                cf = cash_flow.get(quarter_date, {}) if cash_flow else {}
                
                row = {'date': quarter_date}
                
                # Get current quarter values
                revenue_current = float(inc.get('totalRevenue', 0)) if inc.get('totalRevenue') else None
                ebitda_current = float(inc.get('ebitda', 0)) if inc.get('ebitda') else None
                eps_current = float(inc.get('eps', 0)) if inc.get('eps') else None
                ocf_current = float(cf.get('totalCashFromOperatingActivities', 0)) if cf.get('totalCashFromOperatingActivities') else None
                net_income_current = float(inc.get('netIncome', 0)) if inc.get('netIncome') else None
                
                # Calculate net margin for margin expansion
                if revenue_current and net_income_current and revenue_current != 0:
                    current_margin = net_income_current / revenue_current
                else:
                    current_margin = None
                
                # YoY growth (4 quarters back = 1 year)
                if i >= 4:
                    yoy_quarter = sorted_quarters[i-4]
                    inc_yoy = income_stmt[yoy_quarter]
                    cf_yoy = cash_flow.get(yoy_quarter, {}) if cash_flow else {}
                    
                    revenue_yoy = float(inc_yoy.get('totalRevenue', 0)) if inc_yoy.get('totalRevenue') else None
                    ebitda_yoy = float(inc_yoy.get('ebitda', 0)) if inc_yoy.get('ebitda') else None
                    eps_yoy = float(inc_yoy.get('eps', 0)) if inc_yoy.get('eps') else None
                    ocf_yoy = float(cf_yoy.get('totalCashFromOperatingActivities', 0)) if cf_yoy.get('totalCashFromOperatingActivities') else None
                    net_income_yoy = float(inc_yoy.get('netIncome', 0)) if inc_yoy.get('netIncome') else None
                    
                    # Revenue Growth YoY
                    if revenue_current and revenue_yoy and revenue_yoy != 0:
                        row['revenue_growth_yoy'] = (revenue_current - revenue_yoy) / revenue_yoy
                    
                    # EBITDA Growth YoY
                    if ebitda_current and ebitda_yoy and ebitda_yoy != 0:
                        row['ebitda_growth_yoy'] = (ebitda_current - ebitda_yoy) / ebitda_yoy
                    
                    # EPS Growth YoY
                    if eps_current and eps_yoy and eps_yoy != 0:
                        row['eps_growth_yoy'] = (eps_current - eps_yoy) / eps_yoy
                    
                    # Cash Flow Growth YoY
                    if ocf_current and ocf_yoy and ocf_yoy != 0:
                        row['cf_growth_yoy'] = (ocf_current - ocf_yoy) / ocf_yoy
                    
                    # Margin Expansion (change in net margin)
                    if current_margin is not None and revenue_yoy and net_income_yoy and revenue_yoy != 0:
                        prior_margin = net_income_yoy / revenue_yoy
                        row['margin_expansion'] = current_margin - prior_margin
                
                # 3-Year CAGR (12 quarters back)
                if i >= 12:
                    cagr_quarter = sorted_quarters[i-12]
                    inc_cagr = income_stmt[cagr_quarter]
                    revenue_3y_ago = float(inc_cagr.get('totalRevenue', 0)) if inc_cagr.get('totalRevenue') else None
                    
                    if revenue_current and revenue_3y_ago and revenue_3y_ago > 0:
                        row['revenue_cagr_3y'] = (revenue_current / revenue_3y_ago) ** (1/3) - 1
                
                if len(row) > 1:
                    quarterly_data.append(row)
            
            if not quarterly_data:
                logger.debug(f"G5: Could not calculate growth metrics for {symbol}")
                return _fundamentals_has_data_stub('fin_g5')
            
            # Convert to DataFrame
            df_quarterly = pd.DataFrame(quarterly_data)
            df_quarterly['date'] = pd.to_datetime(df_quarterly['date'])
            df_quarterly = df_quarterly.set_index('date').sort_index()
            
            # ========================================================================
            # ENHANCED FEATURES: Compute on quarterly data BEFORE resampling
            # ========================================================================
            
            # Growth Volatility (revenue & EPS) - rolling 3Y on quarterly data
            if 'revenue_growth_yoy' in df_quarterly.columns:
                df_quarterly['growth_volatility_3y'] = (
                    df_quarterly['revenue_growth_yoy'].rolling(12, min_periods=4).std()
                )
            
            if 'eps_growth_yoy' in df_quarterly.columns:
                df_quarterly['eps_growth_volatility_3y'] = (
                    df_quarterly['eps_growth_yoy'].rolling(12, min_periods=4).std()
                )
            
            # EPS CAGR 3Y - needs actual EPS time series (stored during calculation)
            # For now, use revenue_cagr_3y as proxy (already computed above)
            # TODO: Compute from actual EPS history when available
            df_quarterly['eps_cagr_3y'] = df_quarterly.get('revenue_cagr_3y', pd.Series(index=df_quarterly.index))
            
            # Now resample to daily with forward-fill
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g5 = df_quarterly.reindex(date_range, method='ffill')
            
            # ========================================================================
            # CROSS-FAMILY LINKAGE: SUSTAINABLE GROWTH RATE
            # ========================================================================
            # Sustainable Growth Rate = ROE × (1 - Payout Ratio)
            # Requires: ROE from fin_g0, payout_ratio from fin_g7
            # Will be computed in cross-family aggregation step
            
            # Final cleanup: replace inf and forward-fill remaining NaNs
            fin_g5 = fin_g5.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            fin_g5['has_data'] = 1.0
            
            # Track days since last quarterly earnings
            latest_quarter = df_quarterly.index.max()
            fin_g5['days_since_update'] = np.maximum((fin_g5.index - latest_quarter).days, 0)
            
            logger.info(f"✅ G5: Built {len(df_quarterly)} quarters → {len(fin_g5)} days growth metrics for {symbol}")
            
            fin_g5.attrs['provenance'] = {'source': 'eodhd_fundamentals_api', 'quarters': len(df_quarterly)}
            fin_g5.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'quarters': len(df_quarterly), 'has_data': 1}
            return fin_g5
            
        except Exception as e:
            logger.debug(f"❌ G5 failed for {symbol}: {e}")
            return None

    def _h_fin_g6():
        """
        G6 — Valuation (Historical time-series from quarterly fundamentals)
        
        Strategy:
        - Extract quarterly earnings, revenue, book value, EBITDA from EODHD financials
        - Fetch daily price data
        - Compute TTM/MRQ ratios on quarterly dates (PE, PS, PB, EV/EBITDA)
        - Forward-fill quarterly values to daily frequency
        - Compute 5-year rolling z-scores for historical context
        
        Core Multiples (5):
        - pe_ratio (Price / TTM EPS), pb_ratio (Price / Book Value per Share)
        - ps_ratio (Price / TTM Revenue per Share), ev_ebitda, dividend_yield
        
        History-Normalized (3):
        - pe_zscore_5y, pb_zscore_5y, ev_ebitda_zscore_5y
        
        Total: 5 base + 3 z-scores = 8 features
        Uses EODHD quarterly financials + daily prices (time-varying ratios)
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G6 Valuation: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G6: No fundamentals for {symbol}")
                return _fundamentals_has_data_stub('fin_g6')
            
            # Extract quarterly financials
            financials = fund_data.get('Financials', {})
            income_q = financials.get('Income_Statement', {}).get('quarterly', {})
            balance_q = financials.get('Balance_Sheet', {}).get('quarterly', {})
            cashflow_q = financials.get('Cash_Flow', {}).get('quarterly', {})
            
            if not income_q or not balance_q:
                logger.debug(f"⚠️ G6: Missing quarterly financials for {symbol}")
                return _fundamentals_has_data_stub('fin_g6')
            
            # Build quarterly metrics DataFrame
            quarterly_rows = []
            for date_str in sorted(income_q.keys(), reverse=True):
                try:
                    income = income_q[date_str]
                    balance = balance_q.get(date_str, {})
                    cashflow = cashflow_q.get(date_str, {})
                    
                    net_income = income.get('netIncome')
                    total_revenue = income.get('totalRevenue')
                    ebitda = income.get('ebitda')
                    shares = balance.get('commonStockSharesOutstanding')
                    total_equity = balance.get('totalStockholderEquity')
                    
                    if shares and float(shares) > 0:
                        row = {'date': pd.to_datetime(date_str)}
                        
                        # Earnings per share
                        if net_income:
                            row['eps'] = float(net_income) / float(shares)
                        
                        # Revenue per share
                        if total_revenue:
                            row['revenue_per_share'] = float(total_revenue) / float(shares)
                        
                        # Book value per share
                        if total_equity:
                            row['book_value_per_share'] = float(total_equity) / float(shares)
                        
                        # EBITDA (for EV/EBITDA later)
                        if ebitda:
                            row['ebitda'] = float(ebitda)
                        
                        quarterly_rows.append(row)
                        
                except (KeyError, ValueError, TypeError) as e:
                    continue
            
            if len(quarterly_rows) < 4:
                logger.debug(f"⚠️ G6: Insufficient quarterly data for {symbol} (need 4+ quarters)")
                return _fundamentals_has_data_stub('fin_g6')
            
            df_quarterly = pd.DataFrame(quarterly_rows).set_index('date').sort_index()
            
            # Compute TTM (trailing twelve months) metrics by summing last 4 quarters
            df_quarterly['ttm_eps'] = df_quarterly['eps'].rolling(4, min_periods=4).sum()
            df_quarterly['ttm_revenue_per_share'] = df_quarterly['revenue_per_share'].rolling(4, min_periods=4).sum()
            df_quarterly['ttm_ebitda'] = df_quarterly['ebitda'].rolling(4, min_periods=4).sum()
            
            # Get price data (extended lookback for z-scores)
            lookback_start = pd.to_datetime(start_str_default) - pd.DateOffset(years=5)
            price_df = _fetch_price_data(symbol, lookback_start.strftime('%Y-%m-%d'), end_str_default)
            
            if price_df is None or 'close' not in price_df.columns:
                logger.debug(f"⚠️ G6: No price data for {symbol}")
                return _fundamentals_has_data_stub('fin_g6')
            
            # Create daily date range
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g6 = pd.DataFrame(index=date_range)
            
            # Forward-fill quarterly fundamentals to daily frequency FIRST
            combined_index = df_quarterly.index.union(date_range)
            
            # Forward-fill TTM metrics to daily (earnings update quarterly, persist until next report)
            if 'ttm_eps' in df_quarterly.columns:
                daily_ttm_eps = df_quarterly['ttm_eps'].reindex(combined_index).ffill().loc[date_range]
                fin_g6['ttm_eps'] = daily_ttm_eps
            
            if 'ttm_revenue_per_share' in df_quarterly.columns:
                daily_ttm_rev = df_quarterly['ttm_revenue_per_share'].reindex(combined_index).ffill().loc[date_range]
                fin_g6['ttm_revenue_per_share'] = daily_ttm_rev
            
            if 'book_value_per_share' in df_quarterly.columns:
                daily_book_value = df_quarterly['book_value_per_share'].reindex(combined_index).ffill().loc[date_range]
                fin_g6['book_value_per_share'] = daily_book_value
            
            if 'ttm_ebitda' in df_quarterly.columns:
                daily_ttm_ebitda = df_quarterly['ttm_ebitda'].reindex(combined_index).ffill().loc[date_range]
                fin_g6['ttm_ebitda'] = daily_ttm_ebitda
            
            # Align daily prices with date range
            price_daily = price_df['close'].reindex(date_range).ffill()
            fin_g6['price'] = price_daily
            
            # Compute valuation ratios DAILY using daily price + forward-filled fundamentals
            if 'ttm_eps' in fin_g6.columns and 'price' in fin_g6.columns:
                fin_g6['pe_ratio'] = fin_g6['price'] / (fin_g6['ttm_eps'] + 1e-9)
            
            if 'book_value_per_share' in fin_g6.columns and 'price' in fin_g6.columns:
                fin_g6['pb_ratio'] = fin_g6['price'] / (fin_g6['book_value_per_share'] + 1e-9)
            
            if 'ttm_revenue_per_share' in fin_g6.columns and 'price' in fin_g6.columns:
                fin_g6['ps_ratio'] = fin_g6['price'] / (fin_g6['ttm_revenue_per_share'] + 1e-9)
            
            # EV/EBITDA: Use snapshot (requires debt/cash data not in quarterly financials)
            highlights = fund_data.get('Highlights', {})
            ev_ebitda_snapshot = fund_data.get('Valuation', {}).get('EnterpriseValueEbitda')
            if ev_ebitda_snapshot:
                fin_g6['ev_ebitda'] = float(ev_ebitda_snapshot)
            
            # Dividend yield from snapshot (quarterly dividend data less reliable)
            div_yield = highlights.get('DividendYield')
            if div_yield:
                fin_g6['dividend_yield'] = float(div_yield)
            
            # Drop temporary calculation columns
            fin_g6 = fin_g6.drop(columns=['ttm_eps', 'ttm_revenue_per_share', 'book_value_per_share', 'ttm_ebitda', 'price'], errors='ignore')
            
            ratio_cols = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'dividend_yield']
            
            # Compute 5-year rolling z-scores on daily data (ONLY for time-varying ratios)
            # NOTE: ev_ebitda and dividend_yield are snapshots (constant), not time-series
            for base_metric in ['pe_ratio', 'pb_ratio', 'ps_ratio']:
                if base_metric in fin_g6.columns:
                    rolling_mean = fin_g6[base_metric].rolling(1260, min_periods=252).mean()
                    rolling_std = fin_g6[base_metric].rolling(1260, min_periods=252).std()
                    fin_g6[f'{base_metric}_zscore_5y'] = (fin_g6[base_metric] - rolling_mean) / (rolling_std + 1e-9)
            
            # ================================================================
            # SPLIT ROLE FEATURES: PREDICTIVE vs RISK
            # ================================================================
            # Z-scores are PREDICTIVE: positive z = expensive vs history = potential mean reversion (value tilt).
            # Mamba uses these for slow alpha (value factor momentum, growth-value rotation).
            #
            # valuation_richness is RISK: max(0, zscore) = one-sided stress where EXPENSIVE = FRAGILE.
            # Expensive stocks have more downside risk if sentiment shifts.
            # This goes to portfolio risk_scale aggregation (higher = reduce position size).
            # ================================================================
            
            valuation_stress_cols = []
            for metric in ['pe_ratio', 'pb_ratio', 'ps_ratio']:
                zscore_col = f'{metric}_zscore_5y'
                if zscore_col in fin_g6.columns:
                    # RISK: valuation_richness = max(0, zscore)
                    # Expensive (positive z) = fragile = stress
                    # Cheap (negative z) = no stress from this metric
                    stress_col = f'{metric}_richness'
                    fin_g6[stress_col] = fin_g6[zscore_col].apply(
                        lambda x: max(0.0, x) if pd.notna(x) else 0.0
                    )
                    valuation_stress_cols.append(stress_col)
            
            # Composite valuation richness (mean of individual richness scores)
            if valuation_stress_cols:
                fin_g6['valuation_richness'] = fin_g6[valuation_stress_cols].mean(axis=1)
            
            # Track days since last quarterly earnings report
            latest_quarter = df_quarterly.index.max()
            fin_g6['days_since_update'] = np.maximum((fin_g6.index - latest_quarter).days, 0)
            
            # Final cleanup
            fin_g6 = fin_g6.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            fin_g6['has_data'] = 1.0
            
            logger.info(f"✅ G6: Built {len(ratio_cols)} time-varying valuation ratios + stress for {symbol}")
            
            fin_g6.attrs['provenance'] = {'source': 'eodhd_quarterly_fundamentals', 'metrics': ratio_cols}
            fin_g6.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'type': 'quarterly_time_series', 'has_data': 1}
            return fin_g6
            
        except Exception as e:
            logger.debug(f"❌ G6 failed for {symbol}: {e}")
            return None

    def _h_fin_g7():
        """
        G7 — Dividends & Shareholder Yield (Enhanced with stability & policy metrics)
        
        Core Metrics (5):
        - dividend_yield, payout_ratio, buyback_yield, total_shareholder_yield, shares_outstanding_change
        
        Policy Stability (3):
        - dividend_policy_stability (volatility of payout ratio over 3 years)
        - buyback_consistency (fraction of years with net buybacks > 0)
        - share_dilution_3y (cumulative % change in shares outstanding)
        
        History-Normalized (1):
        - yield_zscore_5y (dividend yield vs 5-year history)
        
        Cross-Family Linkage:
        - payout_ratio used in sustainable_growth_rate (with ROE from fin_g0)
        
        Total: 5 base + 3 stability + 1 z-score = 9 features
        Uses EODHD APIClient get_fundamentals_data
        """
        try:
            from src.data_sources.eodhd_provider import get_eodhd_provider
            
            logger.debug(f"🔑 G7 Quality/Stability: {symbol}")
            
            eodhd_provider = get_eodhd_provider()
            if not eodhd_provider.api_key:
                logger.debug("⚠️ EODHD: API key not configured")
                return None

            fund_data = eodhd_provider.get_fundamentals(symbol)
            if not fund_data:
                logger.debug(f"⚠️ G7: No fundamentals for {symbol}")
                return _fundamentals_has_data_stub('fin_g7')
            
            # Get quarterly financial statements
            financials = fund_data.get('Financials', {})
            income_stmt = financials.get('Income_Statement', {}).get('quarterly', {})
            cash_flow = financials.get('Cash_Flow', {}).get('quarterly', {})
            balance_sheet = financials.get('Balance_Sheet', {}).get('quarterly', {})
            
            if not income_stmt or not cash_flow or not balance_sheet:
                logger.debug(f"G7: No quarterly financials for {symbol}")
                return _fundamentals_has_data_stub('fin_g7')
            
            # Build time series from quarterly data
            quarterly_data = []
            sorted_quarters = sorted(income_stmt.keys())
            
            # Collect historical data for rolling stability metrics
            dividend_history = []
            payout_history = []
            buyback_history = []
            shares_outstanding_history = []
            
            for i, quarter_date in enumerate(sorted_quarters):
                if quarter_date not in cash_flow or quarter_date not in balance_sheet:
                    continue
                
                inc = income_stmt[quarter_date]
                cf = cash_flow[quarter_date]
                bal = balance_sheet[quarter_date]
                
                row = {'date': quarter_date}
                
                # ================================================================
                # CORE DIVIDEND & SHAREHOLDER YIELD METRICS
                # ================================================================
                
                # Extract values
                net_income = float(inc.get('netIncome', 0)) if inc.get('netIncome') else None
                dividends_paid = float(cf.get('dividendsPaid', 0)) if cf.get('dividendsPaid') else None
                share_repurchases = float(cf.get('repurchaseOfStock', 0)) if cf.get('repurchaseOfStock') else None
                shares_outstanding = float(bal.get('commonStock', 0)) if bal.get('commonStock') else None
                
                # Dividend Yield (need price data - will calculate from daily data)
                # Payout Ratio = Dividends / Net Income
                payout_ratio = None
                if dividends_paid and net_income and net_income > 0:
                    payout_ratio = abs(dividends_paid) / net_income  # dividendsPaid is negative
                    row['payout_ratio'] = payout_ratio
                
                # Buyback Yield (need market cap - will calculate from daily data)
                # Total Shareholder Yield = Dividend Yield + Buyback Yield
                
                # Shares Outstanding Change
                if shares_outstanding:
                    row['shares_outstanding'] = shares_outstanding
                
                # ================================================================
                # ACCUMULATE HISTORY FOR STABILITY METRICS
                # ================================================================
                
                if dividends_paid:
                    dividend_history.append(abs(dividends_paid))
                if payout_ratio:
                    payout_history.append(payout_ratio)
                if share_repurchases:
                    buyback_history.append(abs(share_repurchases))
                if shares_outstanding:
                    shares_outstanding_history.append(shares_outstanding)
                
                # ================================================================
                # DIVIDEND POLICY STABILITY (3-year lookback = 12 quarters)
                # ================================================================
                
                if len(payout_history) >= 12:
                    # Volatility of payout ratio over last 3 years
                    row['dividend_policy_stability'] = np.std(payout_history[-12:])
                
                # ================================================================
                # BUYBACK CONSISTENCY (fraction of last 12 quarters with buybacks)
                # ================================================================
                
                if len(buyback_history) >= 4:  # Lower threshold to get more data
                    # Fraction of quarters with net buybacks > 0
                    lookback = min(12, len(buyback_history))
                    buyback_count = sum(1 for bb in buyback_history[-lookback:] if bb > 0)
                    row['buyback_consistency'] = buyback_count / float(lookback)
                
                # ================================================================
                # SHARE DILUTION 3Y (cumulative % change over 12 quarters)
                # ================================================================
                
                if len(shares_outstanding_history) >= 12:
                    shares_3y_ago = shares_outstanding_history[-12]
                    shares_current = shares_outstanding_history[-1]
                    if shares_3y_ago > 0:
                        row['share_dilution_3y'] = (shares_current - shares_3y_ago) / shares_3y_ago
                
                if len(row) > 1:
                    quarterly_data.append(row)
            
            if not quarterly_data:
                logger.debug(f"G7: Could not calculate quality metrics for {symbol}")
                return _fundamentals_has_data_stub('fin_g7')
            
            # Convert to DataFrame and resample to daily
            df_quarterly = pd.DataFrame(quarterly_data)
            df_quarterly['date'] = pd.to_datetime(df_quarterly['date'])
            df_quarterly = df_quarterly.set_index('date').sort_index()
            
            # ========================================================================
            # COMPUTE YIELD Z-SCORE on quarterly data (before resampling)
            # ========================================================================
            
            if 'payout_ratio' in df_quarterly.columns:
                # Use payout ratio as dividend yield proxy
                df_quarterly['dividend_yield_proxy'] = df_quarterly['payout_ratio']
                
                # Yield Z-Score vs 5-year history (20 quarters)
                rolling_mean = df_quarterly['dividend_yield_proxy'].rolling(20, min_periods=8).mean()
                rolling_std = df_quarterly['dividend_yield_proxy'].rolling(20, min_periods=8).std()
                df_quarterly['yield_zscore_5y'] = (
                    (df_quarterly['dividend_yield_proxy'] - rolling_mean) / (rolling_std + 1e-9)
                )
            
            # Now resample to daily with forward-fill
            date_range = pd.date_range(start=start_str_default, end=end_str_default, freq='D')
            fin_g7 = df_quarterly.reindex(date_range, method='ffill')
            
            # ========================================================================
            # CROSS-FAMILY LINKAGE NOTES
            # ========================================================================
            # payout_ratio from fin_g7 will be linked with:
            # - ROE from fin_g0 to compute sustainable_growth_rate in fin_g5
            # This linkage will be implemented in cross-family aggregation step
            
            # Final cleanup: replace inf and forward-fill all columns
            fin_g7 = fin_g7.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            fin_g7['has_data'] = 1.0
            
            # Track days since last quarterly dividend data
            latest_quarter = df_quarterly.index.max()
            fin_g7['days_since_update'] = np.maximum((fin_g7.index - latest_quarter).days, 0)
            
            logger.info(f"✅ G7: Built {len(df_quarterly)} quarters → {len(fin_g7)} days dividend/shareholder metrics for {symbol}")
            
            fin_g7.attrs['provenance'] = {'source': 'eodhd_fundamentals_api', 'quarters': len(df_quarterly)}
            fin_g7.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd', 'quarters': len(df_quarterly), 'has_data': 1}
            return fin_g7
            
        except Exception as e:
            logger.debug(f"❌ G7 failed for {symbol}: {e}")
            return None

    def _h_dcf():
        """
        HEDGE-FUND GRADE: Price Anchoring / Relative Valuation Features (19 features)
        
        CRITICAL GOVERNANCE: This is NOT true DCF valuation.
        This family provides price-relative anchoring signals for:
        - Conditioning confidence
        - Modulating position sizing
        - Trend vs mean-reversion regime detection
        
        NOT for dominant alpha generation.
        
        Categories:
        1. Valuation Anchors (3 features) - Slow/fast fair value proxies
        2. Multi-Horizon Momentum (5 features) - Core alpha, winsorized
        3. Mean Reversion Z-Scores (2 features) - Canonical HF signals
        4. Trend Quality (2 features) - Clean vs noisy trend detection
        5. Volatility-Adjusted Value (2 features) - Risk conditioning
        6. Scenario Awareness (3 features) - Asymmetry/convexity (low weight)
        7. Horizon Structure (1 feature) - Growth vs value duration
        8. Regime Context (1 feature) - Short-term positioning
        
        Total: 19 features (compressed from 25)
        Data Source: EODHD price data only
        
        REMOVED (redundant/noisy):
        - dcf_pe_proxy (correlated with price_to_fairvalue_1y)
        - dcf_pullback_strength (nonlinear re-expression of z-scores)
        - dcf_reversion_probability (nonlinear re-expression of z-scores)
        - dcf_trend_slope_3m (redundant with mom_3m)
        - dcf_trend_persistence (count-based, noisy)
        - dcf_duration (complex, overlaps momentum + vol)
        
        TRUE ALPHA DRIVERS (primary families):
        - alternative_signals
        - correlation / cross_asset
        - cboe_term
        - events
        
        This family makes those signals SAFER and more ADAPTIVE.
        """
        try:
            price_df = _fetch_price_data(symbol, start_str_default, end_str_default)
            if price_df is None or 'close' not in price_df.columns:
                return None
            
            price = price_df['close']
            if len(price) < 252:
                return None
            
            dcf_data = pd.DataFrame(index=price.index)
            
            # ============================================================
            # 1. VALUATION ANCHORS (3 features) - HEDGE-FUND: Core anchors only
            # ============================================================
            # These are price-relative anchoring signals, NOT intrinsic value
            
            # Moving averages
            ma_252 = price.rolling(252, min_periods=100).mean()
            ma_63 = price.rolling(63, min_periods=20).mean()
            ma_21 = price.rolling(21, min_periods=10).mean()
            ma_20 = price.rolling(20, min_periods=10).mean()
            ema_252 = price.ewm(span=252, min_periods=100).mean()
            ema_63 = price.ewm(span=63, min_periods=20).mean()
            
            # Bollinger bands (20-day)
            bb_mid = price.rolling(20, min_periods=10).mean()
            bb_std = price.rolling(20, min_periods=10).std()
            
            # KEEP: Slow anchor (1-year fair value estimate)
            dcf_data['dcf_price_to_fairvalue_1y'] = price / (ema_252 + 1e-9)
            
            # KEEP: Fast anchor (3-month fair value estimate)
            dcf_data['dcf_price_to_fairvalue_3m'] = price / (ema_63 + 1e-9)
            
            # KEEP: Regime context (short-term positioning)
            dcf_data['dcf_price_regime'] = price / (bb_mid + 1e-9)
            
            # ============================================================
            # 1B. LOG ANCHORS + ONE-SIDED STRESS (for portfolio overlays)
            # ============================================================
            # Raw ratios are asymmetric and can dominate aggregation
            # Log versions are symmetric around 0 and better for regime aggregation
            # CRITICAL: RoleAwareContext treats high values as "stress"
            #   - Overextension (price >> fair value) → stress (should reduce exposure)
            #   - Undervaluation (price << fair value) → NOT stress (may be opportunity)
            
            # Log price-to-fairvalue (symmetric around 0)
            dcf_data['dcf_log_p2fv_1y'] = np.log(price / (ema_252 + 1e-9))
            dcf_data['dcf_log_p2fv_3m'] = np.log(price / (ema_63 + 1e-9))
            dcf_data['dcf_log_price_regime'] = np.log(price / (bb_mid + 1e-9))
            
            # Overextension stress (one-sided: only penalize "too expensive")
            # Higher value = more overextended = reduce exposure via overlays
            # cap at 1.0 (corresponds to ~2.7x fair value)
            log_p2fv_1y = dcf_data['dcf_log_p2fv_1y']
            dcf_data['dcf_overextension'] = log_p2fv_1y.clip(lower=0).clip(upper=1.0)
            
            # Undervaluation (one-sided: directional alpha signal for Mamba)
            # More negative = more undervalued = potential opportunity
            # Keep for Mamba; do NOT use in stress aggregation
            dcf_data['dcf_undervaluation'] = log_p2fv_1y.clip(upper=0).clip(lower=-1.0)
            
            # ============================================================
            # 2. MULTI-HORIZON MOMENTUM (5 features) - HEDGE-FUND: Core alpha
            # ============================================================
            # This block is excellent and hedge-fund standard
            
            price_21_ago = price.shift(21)
            price_63_ago = price.shift(63)
            price_252_ago = price.shift(252)
            
            dcf_data['dcf_mom_1m'] = price / (price_21_ago + 1e-9)
            dcf_data['dcf_mom_3m'] = price / (price_63_ago + 1e-9)
            dcf_data['dcf_mom_12m'] = price / (price_252_ago + 1e-9)
            
            # Realized volatility (21-day)
            returns = price.pct_change()
            realized_vol = returns.rolling(21, min_periods=10).std() * np.sqrt(252)
            
            # Downside volatility (negative returns only)
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(21, min_periods=10).std() * np.sqrt(252)
            
            momentum_1m = price.pct_change(21)
            
            # HEDGE-FUND: Winsorize vol-adjusted momentum to [-10, 10]
            # (Protects against extreme vol spikes during crashes)
            dcf_data['dcf_mom_vol_adjusted'] = (momentum_1m / (realized_vol + 1e-9)).clip(-10, 10)
            dcf_data['dcf_mom_sharped'] = (momentum_1m / (downside_vol + 1e-9)).clip(-10, 10)
            
            # ============================================================
            # 3. MEAN REVERSION Z-SCORES (2 features) - HEDGE-FUND: Canonical signals
            # ============================================================
            
            # Z-scores (normalized deviation from mean)
            std_21 = price.rolling(21, min_periods=10).std()
            std_63 = price.rolling(63, min_periods=20).std()
            
            # KEEP: Core z-scores (hedge-fund standard)
            dcf_data['dcf_zscore_1m'] = (price - ma_21) / (std_21 + 1e-9)
            dcf_data['dcf_zscore_3m'] = (price - ma_63) / (std_63 + 1e-9)
            
            # REMOVED: dcf_pullback_strength (nonlinear re-expression of z-scores)
            # REMOVED: dcf_reversion_probability (nonlinear re-expression of z-scores)
            # Both add little incremental information and increase collinearity
            
            # ============================================================
            # 4. TREND QUALITY (2 features) - HEDGE-FUND: Is trend clean or noisy?
            # ============================================================
            
            # Simplified slope calculation (average daily change)
            slope_21 = price.diff(21) / 21  # Average daily change over 21 days
            
            # KEEP: Normalized slope (is there a trend?)
            dcf_data['dcf_trend_slope_1m'] = slope_21 / (price + 1e-9)
            
            # KEEP: Trend stability (is it clean or noisy?)
            # Higher = stronger trend relative to noise
            dcf_data['dcf_trend_stability'] = np.abs(slope_21) / (std_21 + 1e-9)
            
            # REMOVED: dcf_trend_slope_3m (redundant with mom_3m)
            # REMOVED: dcf_trend_persistence (count-based, noisy)
            
            # ============================================================
            # 5. VOLATILITY-ADJUSTED VALUE (2 features) - HEDGE-FUND: Risk conditioning
            # ============================================================
            # Excellent for sizing conviction and interacting with σ_exec
            
            # Valuation adjusted by volatility (higher vol = riskier = lower relative value)
            vol_63 = returns.rolling(63, min_periods=20).std() * np.sqrt(252)
            valuation_ratio = price / (ma_63 + 1e-9)
            
            dcf_data['dcf_vol_adjusted_value'] = valuation_ratio / (vol_63 + 1e-9)
            
            # Value-momentum ratio (combines valuation cheapness with momentum strength)
            momentum_63 = price.pct_change(63)
            dcf_data['dcf_value_momentum_ratio'] = momentum_63 / (vol_63 + 1e-9)
            
            # ============================================================
            # 6. SCENARIO AWARENESS (3 features) - HEDGE-FUND: Low weight, regime breaks only
            # ============================================================
            # GOVERNANCE: These should have LOW Stage-A weight
            # Purpose: Convexity/asymmetry descriptors, NOT daily trading signals
            # Useful during regime breaks, not normal markets
            # 
            # Capture uncertainty and asymmetry in valuation estimates
            # Bull/bear scenarios based on volatility envelope around fair value proxies
            
            # Bull scenario: fair value + 1.5 std (optimistic path)
            # Bear scenario: fair value - 1.5 std (pessimistic path)
            std_252 = price.rolling(252, min_periods=100).std()
            fair_value_base = ema_252  # Use 1-year EMA as base fair value
            
            bull_scenario = fair_value_base + (1.5 * std_252)
            bear_scenario = fair_value_base - (1.5 * std_252)
            
            # Scenario spread: width of valuation uncertainty
            # High = large uncertainty, low = high conviction
            dcf_data['dcf_scenario_spread'] = (bull_scenario - bear_scenario) / (fair_value_base + 1e-9)
            
            # Scenario skew: asymmetry of upside vs downside
            # Positive = more upside potential, negative = more downside risk
            upside = (bull_scenario - price) / (price + 1e-9)
            downside = (price - bear_scenario) / (price + 1e-9)
            dcf_data['dcf_scenario_skew'] = (upside - downside) / (upside + downside + 1e-9)
            
            # Downside skew stress (one-sided: for portfolio overlays)
            # CRITICAL: dcf_scenario_skew is directional (-1 to +1)
            # Negative skew = more downside risk → stress for overlays
            # Positive skew = more upside → NOT stress
            # Formula: clip(-scenario_skew, 0, 1)
            dcf_data['dcf_downside_skew_stress'] = (-dcf_data['dcf_scenario_skew']).clip(0, 1)
            
            # Current position in scenario range
            # 0 = at bear scenario, 0.5 = at fair value, 1 = at bull scenario
            dcf_data['dcf_scenario_position'] = (
                (price - bear_scenario) / (bull_scenario - bear_scenario + 1e-9)
            )
            
            # ============================================================
            # 7. HORIZON STRUCTURE (1 feature) - HEDGE-FUND: Growth vs value duration
            # ============================================================
            # Capture duration characteristics of the valuation
            # High-growth stocks have long duration (value far out), mature have short duration
            
            # KEEP: Terminal value percentage (% of value from long-term cash flows)
            # Proxy: ratio of long-term MA (252d) to short-term MA (63d)
            # High ratio = stable long-term value (mature), low = near-term driven (growth)
            dcf_data['dcf_terminal_value_pct'] = ma_252 / (ma_63 + 1e-9)
            
            # REMOVED: dcf_duration (complex, noisy, overlaps momentum + vol features)
            
            # ============================================================
            # FINALIZE RAW DCF FEATURES
            # ============================================================
            
            # Fill NaNs and filter to requested date range
            dcf_data = dcf_data.fillna(method='ffill').fillna(0)
            dcf_data = dcf_data[
                (dcf_data.index >= pd.Timestamp(start)) & 
                (dcf_data.index <= pd.Timestamp(end))
            ]
            
            # ============================================================
            # TODO: HF_EXTENSION (ML-based fair value + regime)
            # ============================================================
            # NOTE: HF extension removed from build_panel() to prevent data leakage
            # 
            # Correct implementation requires:
            # 1. HF extension training in walk-forward loop (NOT in build_panel)
            # 2. Each fold trains fresh model on fold_config.train_start:train_end ONLY
            # 3. Models: Fair Value (LGBM/MLP/ElasticNet) + Regime Classifier (4-state)
            # 4. Inputs: raw_dcf + ml_framework + microstructure + quantile (optional)
            # 5. Outputs: 5 features (fairvalue, mispricing, confidence, regime, regime_conf)
            #
            # Implementation path:
            # - Add fold_config parameter to build_panel() OR
            # - Create separate HF extension training step in walk-forward loop
            # - See: src/features/dcf_hf_extension.py (module ready, needs integration)
            # ============================================================
            
            # Add metadata
            dcf_data.attrs['telemetry'] = {
                'status': 'ok',
                'source': 'eodhd_price_data',
                'feature_count': len(dcf_data.columns),
                'hedge_fund_grade': True,
                'compressed_from': 25,
                'vol_adjusted_momentum_winsorized': True,
                'winsorization_bounds': [-10, 10],
            }
            dcf_data.attrs['provenance'] = {
                'source': 'eodhd_price_data',
                'method': 'price_anchoring_relative_valuation',  # Conceptual rename
                'conceptual_family_name': 'price_anchoring',  # NOT true DCF
                'hedge_fund_grade': True,
                'feature_categories': [
                    'valuation_anchors',          # 3 features
                    'multi_horizon_momentum',     # 5 features
                    'mean_reversion_zscores',     # 2 features
                    'trend_quality',              # 2 features
                    'volatility_adjusted_value',  # 2 features
                    'scenario_awareness',         # 3 features
                    'horizon_structure',          # 1 feature
                    'regime_context'              # 1 feature (dcf_price_regime)
                ],
                'removed_features': [
                    'dcf_pe_proxy',
                    'dcf_relative_strength_1m',
                    'dcf_pullback_strength',
                    'dcf_reversion_probability',
                    'dcf_trend_slope_3m',
                    'dcf_trend_persistence',
                    'dcf_duration'
                ]
            }
            # HEDGE-FUND GOVERNANCE: Feature role and priority tagging
            dcf_data.attrs['feature_roles'] = {
                'core_alpha': [  # Multi-horizon momentum (primary signals)
                    'dcf_mom_1m', 'dcf_mom_3m', 'dcf_mom_12m',
                    'dcf_mom_vol_adjusted', 'dcf_mom_sharped'
                ],
                'conditioning': [  # Modulate sizing and confidence
                    'dcf_price_to_fairvalue_1y', 'dcf_price_to_fairvalue_3m',
                    'dcf_vol_adjusted_value', 'dcf_value_momentum_ratio'
                ],
                'regime_detection': [  # Trend vs mean-reversion regime
                    'dcf_zscore_1m', 'dcf_zscore_3m',
                    'dcf_trend_slope_1m', 'dcf_trend_stability',
                    'dcf_price_regime'
                ],
                'low_priority': [  # Asymmetry descriptors (low weight, regime breaks only)
                    'dcf_scenario_spread', 'dcf_scenario_skew', 'dcf_scenario_position',
                    'dcf_terminal_value_pct'
                ]
            }
            dcf_data.attrs['governance'] = {
                'true_alpha_drivers': ['alternative_signals', 'correlation', 'cross_asset', 'cboe_term', 'events'],
                'this_family_purpose': 'Make primary signals safer and more adaptive',
                'NOT_for': 'Dominant alpha generation',
                'conceptual_rename': 'price_anchoring (NOT true DCF valuation)'
            }
            dcf_data.attrs['feature_counts'] = {'generated': len(dcf_data.columns), 'expected': 19}
            
            return dcf_data
        except Exception as e:
            logger.debug(f"DCF feature generation failed for {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # DEPRECATED: _h_news_sentiment_hf
    # This is NOT a family - it should exist outside the family system.
    # Kept as stub for backward compatibility but removed from registry.
    # ─────────────────────────────────────────────────────────────────────────
    def _h_news_sentiment_hf():
        """DEPRECATED: news_sentiment_hf is not a family."""
        logger.warning(f"news_sentiment_hf called but is deprecated - not a family")
        return None

    def _h_earnings_transcript_hf():
        """
        HEDGE-FUND GRADE: Earnings Transcript Sentiment (10 features)
        
        PHILOSOPHY: Confidence-weighted narrative changes, NOT raw sentiment levels
        
        A. Overall Sentiment (2): score, conf (confidence CRITICAL)
        B. Sectional Sentiment (2): score_prepared, score_qa (divergence matters)
        C. Tone Dimensions (2): uncertainty_score (vol amplifier), risk_score
        D. Time-Series Changes (2): score_delta_qoq (HIGH ALPHA), score_delta_yoy
        E. Interaction Features (2): sentiment_divergence, sentiment_shock (confidence-weighted)
        
        DECAY DESIGN (perfect, do not change):
        - Strongest on earnings day, rapid decay over 3-4 sessions
        - Matches market pricing: fast reaction, quick saturation
        
        GOVERNANCE:
        - Stop here: More NLP causes overfitting
        - Confidence-weighted deltas already capture signal
        - NOT for: Topic modeling, LLM summaries, long-window averages
        """
        # Try loading from cache first (prep_families should have generated this)
        if cache_dir is not None:
            cached = _load_cached_family_for_hf('earnings_transcript_hf', 'earnings_transcript_hf')
            if cached is not None and not cached.empty:
                logger.info(f"✅ earnings_transcript_hf: using cached result for {symbol}")
                # Add governance metadata
                cached.attrs['governance'] = {
                    'philosophy': 'Confidence-weighted narrative changes',
                    'high_alpha_features': ['score_delta_qoq', 'score_delta_yoy', 'sentiment_shock'],
                    'vol_amplifiers': ['uncertainty_score', 'risk_score'],
                    'decay_design': 'Fast reaction (3-4 days), quick saturation',
                    'NOT_for': 'Topic modeling, LLM summaries, long-window averages'
                }
                cached.attrs['feature_counts'] = {'generated': len(cached.columns), 'expected': 10}
                return cached
        
        # Fallback: generate on-the-fly (should rarely happen if prep_families ran)
        try:
            from src.dcf_lab.modules.earnings_transcript_hf import EarningsTranscriptHF
            logger.warning(f"⚠️ earnings_transcript_hf: cache miss, generating on-the-fly for {symbol}")
            module = EarningsTranscriptHF()
            signal = module.emit_signal(symbol=symbol, horizon=63, start_date=start, end_date=end)
            if signal and hasattr(signal, 'df') and isinstance(signal.df, pd.DataFrame):
                df = signal.df
                if not df.empty:
                    df = _safe_filter(df, start, end)
                    if not df.empty:
                        df.attrs['telemetry'] = {'status': 'ok', 'source': 'EarningsTranscriptHF'}
                        df.attrs['governance'] = {
                            'philosophy': 'Confidence-weighted narrative changes',
                            'high_alpha_features': ['score_delta_qoq', 'score_delta_yoy', 'sentiment_shock'],
                            'vol_amplifiers': ['uncertainty_score', 'risk_score'],
                            'decay_design': 'Fast reaction (3-4 days), quick saturation'
                        }
                        df.attrs['feature_counts'] = {'generated': len(df.columns), 'expected': 10}
                        return df
        except Exception as e:
            logger.debug(f"earnings_transcript_hf failed for {symbol}: {e}")
        return None

    def _h_doc_embedding_novelty_hf():
        """HuggingFace-based document embedding novelty module"""
        # Try loading from cache first (prep_families should have generated this)
        cached = _load_cached_family_for_hf('doc_embedding_novelty_hf', 'doc_embedding_novelty_hf')
        if cached is not None and not cached.empty:
            logger.info(f"✅ doc_embedding_novelty_hf: using cached result for {symbol}")
            return cached
        
        # Fallback: generate on-the-fly (should rarely happen if prep_families ran)
        try:
            from src.dcf_lab.modules.doc_embedding_novelty_hf import DocumentEmbeddingNoveltyHF
            logger.warning(f"⚠️ doc_embedding_novelty_hf: cache miss, generating on-the-fly for {symbol}")
            module = DocumentEmbeddingNoveltyHF()
            signal = module.emit_signal(symbol=symbol, horizon=63, start_date=start, end_date=end)
            if signal and hasattr(signal, 'df') and isinstance(signal.df, pd.DataFrame):
                df = signal.df
                if not df.empty:
                    df = _safe_filter(df, start, end)
                    if not df.empty:
                        df.attrs['telemetry'] = {'status': 'ok', 'source': 'DocEmbeddingNoveltyHF'}
                        df.attrs['feature_counts'] = {'generated': len(df.columns)}
                        return df
        except Exception as e:
            logger.debug(f"doc_embedding_novelty_hf failed for {symbol}: {e}")
        return None

    def _h_macro_tst_hf():
        """
        Unified Macro Super-Family (combines macro_sector + macro_enhanced + HF transformer)
        
        Features:
        1. RAW FUNDAMENTAL MACRO: EODHD indices (TNX, IRX, UUP, TIP/TLT)
        2. RAW MARKET MACRO: volatility, trends, momentum from prices
        3. STRUCTURAL FEATURES: yield curve, inflation gap, regime scores
        4. HF TRANSFORMER EMBEDDINGS: 8-dim latent macro representation
        
        Total: ~20 comprehensive macro features
        """
        # Try loading from cache first (prep_families should have generated this)
        cached = _load_cached_family_for_hf('macro_tst_hf', 'macro_tst_hf')
        if cached is not None and not cached.empty:
            logger.info(f"✅ macro_tst_hf: using cached result for {symbol}")
            return cached
        
        # Fallback: generate on-the-fly (should rarely happen if prep_families ran)
        try:
            logger.warning(f"⚠️ macro_tst_hf: cache miss, generating on-the-fly for {symbol}")
            from src.dcf_lab.modules.macro_tst_hf import MacroTSTHF
            module = MacroTSTHF()
            signal = module.emit_signal(symbol=symbol, horizon=63, start_date=start, end_date=end)
            if signal and hasattr(signal, 'df') and isinstance(signal.df, pd.DataFrame):
                df = signal.df
                if not df.empty:
                    df = _safe_filter(df, start, end)
                    if not df.empty:
                        logger.info(f"✅ EODHD: Unified macro family generated {len(df)} rows × {len(df.columns)} features")
                        df.attrs['telemetry'] = {'status': 'ok', 'source': 'eodhd_unified_macro', 'provider': 'EODHD'}
                        df.attrs['feature_counts'] = {'generated': len(df.columns)}
                        df.attrs['provenance'] = {'source': 'eodhd_macro_indices', 'families_merged': ['macro_sector', 'macro_enhanced', 'macro_tst_hf']}
                        return df
        except Exception as e:
            logger.debug(f"❌ macro_tst_hf failed for {symbol}: {e}")
        return None

    def _h_tech_micro_hf():
        """HF block focusing on technical + microstructure alpha."""
        families_used = ['ml_framework', 'microstructure', 'correlation']
        if cache_dir is None:
            logger.warning("tech_micro_hf: cache_dir is required for HF cached inputs")
            return None

        ml_df = _load_cached_family_for_hf('tech_micro_hf', 'ml_framework')
        micro_df = _load_cached_family_for_hf('tech_micro_hf', 'microstructure')
        corr_df = _load_cached_family_for_hf('tech_micro_hf', 'correlation')

        def _add_selective_lags(df: pd.DataFrame, family: str) -> pd.DataFrame:
            """Add curated lags based on feature patterns."""
            if df is None or df.empty:
                return df
            
            lag_config = {
                'ml_framework': [
                    (['price_vs_ma_', 'price_vs_ema_', 'bb_width', 'bb_position', 'macd', 'macd_signal', 'rsi'], [1, 5]),
                    (['ma_', 'ema_'], [5]),
                    (['return_', 'log_return_'], [1, 5, 10]),
                    (['volatility_', 'volatility_annualized_'], [5, 20]),
                    (['skewness_', 'kurtosis_'], [5]),
                    (['volume_ma_'], [5, 20]),
                    (['volume_ratio_', 'price_volume', 'volume_weighted_price'], [1, 5]),
                    (['price_ratio_', 'price_position_', 'percentile_'], [1, 5]),
                    (['CONFIDENCE'], [1, 5]),
                    (['autocorr_lag_'], [1]),
                ],
                'microstructure': [
                    # Liquidity & impact
                    (['micro_amihud', 'micro_spread_proxy', 'micro_impact_ratio', 'micro_impact_volatility',
                      'micro_liquidity_imbalance', 'micro_volume_liquidity', 'micro_turnover'], [1, 3]),
                    (['micro_low_liquidity_flag'], [1]),
                    # Order flow & pressure
                    (['micro_ofi_proxy', 'micro_pressure_proxy', 'micro_signed_volume', 'micro_demand_supply_ratio'], [1, 3]),
                    # Volatility & ranges
                    (['micro_intraday_vol_proxy', 'micro_overnight_vol', 'micro_intraday_vs_overnight_vol',
                      'micro_vol_of_vol', 'micro_true_range', 'micro_atr_ratio', 'micro_range_pct',
                      'micro_range_scaled', 'micro_body_pct', 'micro_shadow_ratio', 'micro_wick_top', 'micro_wick_bottom'], [1, 3]),
                    (['micro_zero_range_flag'], [1]),
                    # Gaps & overnight
                    (['micro_overnight_gap', 'micro_gap_direction'], [1, 3]),
                    # Volume dynamics
                    (['micro_volume_surge', 'micro_volume_zscore', 'micro_hl_volume_corr'], [1, 3]),
                    (['micro_stale_tick'], [1]),
                    (['CONFIDENCE'], [1, 3]),
                ],
            }
            
            if family not in lag_config:
                return df
            
            lagged_frames = [df]
            for patterns, lags in lag_config[family]:
                matching_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(pattern.lower() in col_lower for pattern in patterns):
                        matching_cols.append(col)
                
                if not matching_cols:
                    continue
                
                for lag in lags:
                    lagged = df[matching_cols].shift(lag)
                    lagged.columns = [f"{col}_lag{lag}" for col in matching_cols]
                    lagged_frames.append(lagged)
            
            if len(lagged_frames) > 1:
                return pd.concat(lagged_frames, axis=1)
            return df

        blocks: List[pd.DataFrame] = []
        for fam_name, raw_df, limit in [
            ('ml_framework', ml_df, None),
            ('microstructure', micro_df, None),
            ('correlation', corr_df, 12),
        ]:
            if raw_df is None:
                continue
            filtered = _safe_filter(raw_df, start, end)
            with_lags = _add_selective_lags(filtered, fam_name)
            prepared = _prepare_family_block(with_lags, fam_name, limit)
            if prepared is not None and not prepared.empty:
                blocks.append(prepared)

        features = _merge_family_blocks(blocks, max_total_cols=48)
        if features is None or features.empty:
            logger.debug("tech_micro_hf: insufficient base features")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=40,
            horizon_local=HF_SYMBOL_ONLY_CANONICAL_HORIZON,
            prefix='tech_micro_hf',
            families_used=families_used,
            epochs=6,
        )

    def _h_forecast_hf():
        """HF block blending forecast-oriented families."""
        # REMOVED: calibration, online_learning - these are governance layers, not families
        families_used = ['quantile_forecast', 'arima_forecast', 'tft_features']
        if cache_dir is None:
            logger.warning("forecast_hf: cache_dir is required for HF cached inputs")
            return None

        quantile_df = _load_cached_family_for_hf('forecast_hf', 'quantile_forecast')
        arima_df = _load_cached_family_for_hf('forecast_hf', 'arima_forecast')
        tft_df = _load_cached_family_for_hf('forecast_hf', 'tft_features')

        blocks: List[pd.DataFrame] = []
        for fam_name, raw_df, limit in [
            ('quantile_forecast', quantile_df, 24),
            ('arima_forecast', arima_df, 6),
            ('tft_features', tft_df, 14),
        ]:
            if raw_df is None:
                continue
            filtered = _safe_filter(raw_df, start, end)
            prepared = _prepare_family_block(filtered, fam_name, limit)
            if prepared is not None and not prepared.empty:
                blocks.append(prepared)

        if len(blocks) < 2:
            logger.debug("forecast_hf: need at least two families with data")
            return None

        features = _merge_family_blocks(blocks, max_total_cols=64)
        if features is None or features.empty:
            logger.debug("forecast_hf: merged features empty")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=60,
            horizon_local=horizon,
            prefix='forecast_hf',
            families_used=families_used,
            epochs=8,
        )

    def _h_vol_deriv_hf():
        """HF block for volatility + derivatives structure."""
        families_used = ['garch_iv', 'cboe_term', 'options', 'options_anchoring', 'short_interest']
        if cache_dir is None:
            logger.warning("vol_deriv_hf: cache_dir is required for HF cached inputs")
            return None

        garch_df = _load_cached_family_for_hf('vol_deriv_hf', 'garch_iv')
        cboe_df = _load_cached_family_for_hf('vol_deriv_hf', 'cboe_term')
        options_anchor_df = _load_cached_family_for_hf('vol_deriv_hf', 'options_anchoring')
        options_df = _load_cached_family_for_hf('vol_deriv_hf', 'options')
        short_interest_df = _load_cached_family_for_hf('vol_deriv_hf', 'short_interest')

        blocks: List[pd.DataFrame] = []

        if garch_df is not None:
            garch_block = _prepare_family_block(_safe_filter(garch_df, start, end), 'garch_iv', max_cols=12)
            if garch_block is not None:
                blocks.append(garch_block)

        if cboe_df is not None:
            cboe_block = _prepare_family_block(_safe_filter(cboe_df, start, end), 'cboe_term', max_cols=12)
            if cboe_block is not None:
                blocks.append(cboe_block)

        if options_anchor_df is not None:
            anchor_block = _prepare_family_block(_safe_filter(options_anchor_df, start, end), 'options_anchoring', max_cols=12)
            if anchor_block is not None:
                blocks.append(anchor_block)

        if options_df is not None:
            filtered_options = _safe_filter(options_df, start, end)
            filtered_options = _filter_columns_by_keywords(
                filtered_options,
                ['total_oi', 'total_volume', 'put_call', 'gamma', 'delta_exposure']
            )
            options_block = _prepare_family_block(filtered_options, 'options', max_cols=10)
            if options_block is not None:
                blocks.append(options_block)

        if short_interest_df is not None:
            short_block = _prepare_family_block(_safe_filter(short_interest_df, start, end), 'short_interest', max_cols=6)
            if short_block is not None:
                blocks.append(short_block)

        features = _merge_family_blocks(blocks, max_total_cols=60)
        if features is None or features.empty:
            logger.debug("vol_deriv_hf: insufficient merged features")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=60,
            horizon_local=HF_SYMBOL_ONLY_CANONICAL_HORIZON,
            prefix='vol_deriv_hf',
            families_used=families_used,
            epochs=8,
        )

    def _h_macro_regime_hf():
        """HF block for macro/regime state detection."""
        families_used = ['macro_tst_hf', 'regime', 'multiasset', 'cross_asset', 'correlation']
        if cache_dir is None:
            logger.warning("macro_regime_hf: cache_dir is required for HF cached inputs")
            return None

        macro_df = _load_cached_family_for_hf('macro_regime_hf', 'macro_tst_hf')
        regime_df = _load_cached_family_for_hf('macro_regime_hf', 'regime')
        multiasset_df = _load_cached_family_for_hf('macro_regime_hf', 'multiasset')
        cross_asset_df = _load_cached_family_for_hf('macro_regime_hf', 'cross_asset')
        corr_df = _load_cached_family_for_hf('macro_regime_hf', 'correlation')

        blocks: List[pd.DataFrame] = []

        if macro_df is not None:
            macro_block = _prepare_family_block(_safe_filter(macro_df, start, end), 'macro_tst_hf', max_cols=24)
            if macro_block is not None:
                blocks.append(macro_block)

        if regime_df is not None:
            regime_block = _prepare_family_block(_safe_filter(regime_df, start, end), 'regime', max_cols=10)
            if regime_block is not None:
                blocks.append(regime_block)

        if multiasset_df is not None:
            multi_block = _prepare_family_block(_safe_filter(multiasset_df, start, end), 'multiasset', max_cols=16)
            if multi_block is not None:
                blocks.append(multi_block)

        if cross_asset_df is not None:
            cross_block = _prepare_family_block(_safe_filter(cross_asset_df, start, end), 'cross_asset', max_cols=16)
            if cross_block is not None:
                blocks.append(cross_block)

        if corr_df is not None:
            filtered_corr = _safe_filter(corr_df, start, end)
            filtered_corr = _filter_columns_by_keywords(filtered_corr, ['spy', 'qqq', 'uup', 'vxx', 'sp500'])
            corr_block = _prepare_family_block(filtered_corr, 'correlation', max_cols=12)
            if corr_block is not None:
                blocks.append(corr_block)

        if len(blocks) < 2:
            logger.debug("macro_regime_hf: need multiple source families")
            return None

        features = _merge_family_blocks(blocks, max_total_cols=80)
        if features is None or features.empty:
            logger.debug("macro_regime_hf: merged feature frame empty")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=120,
            horizon_local=HF_SYMBOL_ONLY_CANONICAL_HORIZON,
            prefix='macro_regime_hf',
            families_used=families_used,
            epochs=10,
        )

    def _h_fundamental_val_hf():
        """HF block capturing fundamental valuation dynamics."""
        fin_families = [f'fin_g{i}' for i in range(1, 8)]
        families_used = fin_families + ['earnings', 'dividends', 'dcf', 'subsidiary', 'short_interest']
        if cache_dir is None:
            logger.warning("fundamental_val_hf: cache_dir is required for HF cached inputs")
            return None

        blocks: List[pd.DataFrame] = []

        for fam in fin_families:
            df = _load_cached_family_for_hf('fundamental_val_hf', fam)
            if df is None:
                continue
            prepared = _prepare_family_block(_safe_filter(df, start, end), fam, max_cols=24)
            if prepared is not None:
                blocks.append(prepared)

        for fam in ['earnings', 'dividends', 'dcf', 'subsidiary', 'short_interest']:
            df = _load_cached_family_for_hf('fundamental_val_hf', fam)
            if df is None:
                continue
            prepared = _prepare_family_block(_safe_filter(df, start, end), fam, max_cols=12)
            if prepared is not None:
                blocks.append(prepared)

        if len(blocks) < 2:
            logger.debug("fundamental_val_hf: insufficient family coverage")
            return None

        features = _merge_family_blocks(blocks, max_total_cols=96)
        if features is None or features.empty:
            logger.debug("fundamental_val_hf: merged features empty")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=120,
            horizon_local=HF_SYMBOL_ONLY_CANONICAL_HORIZON,
            prefix='fundamental_val_hf',
            families_used=families_used,
            epochs=10,
        )

    def _h_news_nlp_hf():
        """HF block for news/NLP driven signals."""
        # REMOVED: news_sentiment_hf - not a family, should exist outside family system
        families_used = ['finbert', 'doc_embedding_novelty_hf', 'earnings_transcript_hf', 'alternative_signals']
        if cache_dir is None:
            logger.warning("news_nlp_hf: cache_dir is required for HF cached inputs")
            return None

        finbert_df = _load_cached_family_for_hf('news_nlp_hf', 'finbert')
        doc_df = _load_cached_family_for_hf('news_nlp_hf', 'doc_embedding_novelty_hf')
        transcript_df = _load_cached_family_for_hf('news_nlp_hf', 'earnings_transcript_hf')
        # REMOVED: news_sentiment_hf - not a family
        alt_df = _load_cached_family_for_hf('news_nlp_hf', 'alternative_signals')

        blocks: List[pd.DataFrame] = []

        for fam_name, raw_df, limit in [
            ('finbert', finbert_df, 16),
            ('doc_embedding_novelty_hf', doc_df, 12),
            ('earnings_transcript_hf', transcript_df, 12),
            # REMOVED: news_sentiment_hf - not a family
        ]:
            if raw_df is None:
                continue
            prepared = _prepare_family_block(_safe_filter(raw_df, start, end), fam_name, max_cols=limit)
            if prepared is not None:
                blocks.append(prepared)

        if alt_df is not None:
            filtered_alt = _safe_filter(alt_df, start, end)
            filtered_alt = _filter_columns_by_keywords(
                filtered_alt,
                ['news', 'sentiment', 'reddit', 'twitter', 'social', 'search']
            )
            alt_block = _prepare_family_block(filtered_alt, 'alternative_signals', max_cols=10)
            if alt_block is not None:
                blocks.append(alt_block)

        if not blocks:
            logger.debug("news_nlp_hf: no NLP families available")
            return None

        features = _merge_family_blocks(blocks, max_total_cols=48)
        if features is None or features.empty:
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=45,
            horizon_local=HF_SYMBOL_ONLY_CANONICAL_HORIZON,
            prefix='news_nlp_hf',
            families_used=families_used,
            epochs=6,
        )

    def _h_hf_agg():
        """Global HF aggregator over block-level HF scores (Stage B/C meta family)."""
        block_families = list(HF_BLOCK_FAMILIES)
        if not block_families:
            logger.warning("hf_agg: no HF block families registered")
            return None
        if cache_dir is None:
            logger.warning("hf_agg: cache_dir is required for block HF aggregation")
            return None

        block_frames: List[Tuple[str, pd.DataFrame]] = []
        missing_blocks: List[str] = []
        for block_family in block_families:
            block_df = _load_block_hf_signal(block_family)
            if block_df is None or block_df.empty:
                missing_blocks.append(block_family)
                logger.warning("hf_agg: missing block signal %s", block_family)
                continue
            filtered = _safe_filter(block_df, start, end)
            if filtered is None or filtered.empty:
                missing_blocks.append(block_family)
                logger.warning("hf_agg: block %s empty after filtering", block_family)
                continue
            block_frames.append((block_family, filtered))

        if len(block_frames) < MIN_HF_AGG_BLOCKS:
            logger.warning(
                "hf_agg: only %d/%d HF blocks available (%s); skipping",
                len(block_frames),
                len(block_families),
                ", ".join(sorted(missing_blocks)) or "none",
            )
            return None

        if missing_blocks:
            logger.info(
                "hf_agg: proceeding with %d/%d HF blocks (missing: %s)",
                len(block_frames),
                len(block_families),
                ", ".join(sorted(missing_blocks)),
            )

        merged = block_frames[0][1]
        for _, block in block_frames[1:]:
            merged = merged.join(block, how='outer')
        merged = merged.sort_index().replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how='all')
        if merged.empty:
            logger.debug("hf_agg: merged block frame empty")
            return None

        score_cols = [col for col in merged.columns if col.endswith('_score')]
        lagged_frames: List[pd.DataFrame] = []
        for lag in (1, 3, 5):
            if not score_cols:
                break
            lagged = merged[score_cols].shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in score_cols]
            lagged_frames.append(lagged)
        if lagged_frames:
            merged = pd.concat([merged] + lagged_frames, axis=1)

        merged = merged.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how='all')
        if merged.empty:
            logger.debug("hf_agg: merged features empty after cleaning")
            return None

        features = _prepare_family_block(merged, 'hf_agg', max_cols=None)
        if features is None or features.empty:
            logger.debug("hf_agg: normalized feature frame empty")
            return None

        price_df = _load_price_history()
        price_series = None
        if price_df is not None and 'close' in price_df.columns:
            price_series = price_df['close']

        return _run_hf_sequence_block(
            features,
            price_series,
            window=30,
            horizon_local=horizon,
            prefix='hf_agg',
            families_used=[name for name, _ in block_frames],
            epochs=6,
        )


    handlers = {
        # ⚠️ DEPENDENCY ORDER: quantile_forecast MUST run before calibration/online_learning
        'quantile_forecast': _h_quantile_forecaster,
        'arima_forecast': _h_arima,
        
        # Base families (no dependencies)
        'cross_asset': _h_cross,
        'cboe_term': _h_cboe,
        'garch_iv': _h_garch,
        'correlation': _h_corr,
        'candle_mechanics': _h_candle_mechanics,
        'finbert': _h_finbert,
        'crypto': _h_crypto,
        'fx': _h_fx,
        'commodities': _h_commodities,
        'options': _h_options,
        'short_interest': _h_short_interest,
        'insider_form4': _h_insider_form4,
        'econ_events_calendar': _h_econ_events_calendar,
        'corp_actions_splits': _h_corp_actions_splits,
        'marketcap_history': _h_marketcap_history,
        'exchange_calendar': _h_exchange_calendar,
        'index_constituents': _h_index_constituents,
        'subsidiary': _h_subsidiary,
        'earnings': _h_earnings,
        'alternative_signals': _h_alternative,
        'news_sentiment': _h_news_sentiment,
        'ml_framework': _h_ml_framework,
        'microstructure': _h_microstructure,
        'event_time_bars': _h_event_time_bars,
        'drift_monitor': _h_drift_monitor,
        'dcf': _h_dcf,
        'multiasset': _h_multiasset,
        'regime': _h_regime,
        'options_anchoring': _h_options_anchoring,
        'tft_features': _h_tft,
        'dividends': _h_dividends,
        'fin_g0': _h_fin_g0,
        'fin_g1': _h_fin_g1,
        'fin_g2': _h_fin_g2,
        'fin_g3': _h_fin_g3,
        'fin_g4': _h_fin_g4,
        'fin_g5': _h_fin_g5,
        'fin_g6': _h_fin_g6,
        'peer_screener_context': _h_peer_screener_context,
        'fin_g7': _h_fin_g7,
        
        # ─────────────────────────────────────────────────────────────────────
        # REMOVED: calibration and online_learning
        # These are GOVERNANCE LAYERS, not feature families.
        # They persist via artifacts/governance/ with horizon-aware state keys.
        # See: src/stage_b_stateful/governance_persistence.py
        # ─────────────────────────────────────────────────────────────────────
        
        # HuggingFace-based modules
        # REMOVED: news_sentiment_hf - not a family, should exist outside family system
        'earnings_transcript_hf': _h_earnings_transcript_hf,
        'doc_embedding_novelty_hf': _h_doc_embedding_novelty_hf,
        'macro_tst_hf': _h_macro_tst_hf,
        'tech_micro_hf': _h_tech_micro_hf,
        'forecast_hf': _h_forecast_hf,
        'vol_deriv_hf': _h_vol_deriv_hf,
        'macro_regime_hf': _h_macro_regime_hf,
        'fundamental_val_hf': _h_fundamental_val_hf,
        'news_nlp_hf': _h_news_nlp_hf,
        'hf_agg': _h_hf_agg,
    }

    for fam in fams:
        # Try loading from cache first (prep_families output)
        df = _load_from_cache(fam)
        
        if df is None:
            if _no_live_fallback_enabled():
                logger.warning("⚠️ Cache miss for %s and STAGE_B_NO_LIVE_FALLBACK=1; skipping live generation", fam)
                telemetry[fam] = DORMANT
                if _require_all_families_enabled():
                    raise RuntimeError(
                        f"Required family '{fam}' missing cache while STAGE_B_NO_LIVE_FALLBACK=1"
                    )
                continue
            # Fallback to live handler
            h = handlers.get(fam)
            if h is None:
                logger.debug(f"No handler for '{fam}', skipping")
                if _require_all_families_enabled():
                    raise RuntimeError(f"Missing handler for required family '{fam}'")
                continue
            try:
                df = h()
                df = _safe_filter(df, start, end)
            except Exception as exc:
                if isinstance(exc, DisallowedDataSourceError):
                    raise
                logger.warning("Handler for %s failed: %s", fam, exc, exc_info=True)
                telemetry[fam] = DORMANT
                if _require_all_families_enabled():
                    raise RuntimeError(f"Required family '{fam}' failed to generate") from exc
                continue
        
        # Source-policy enforcement (best-effort; relies on telemetry when available)
        if df is not None and isinstance(df, pd.DataFrame) and _strict_sources_enabled():
            tele = (df.attrs or {}).get('telemetry', {}) if hasattr(df, 'attrs') else {}
            status = str(tele.get('status', '')).lower()
            source = str(tele.get('source', '')).lower()
            proxy_flag = bool(tele.get('proxy', False)) or status == 'proxy'

            # Cache-loader overwrites telemetry source; treat cache markers as allowed.
            is_cache_marker = source.startswith('cached_') or source.startswith('consolidated_') or source.startswith('consolidated')

            if _no_proxy_sources_enabled() and not is_cache_marker:
                if proxy_flag:
                    raise DisallowedDataSourceError(
                        f"Family '{fam}' returned proxy telemetry under STAGE_B_NO_PROXY_SOURCES=1 (source='{source}')"
                    )

            if _eodhd_only_enabled() and not is_cache_marker:
                if source and ("eodhd" not in source):
                    raise DisallowedDataSourceError(
                        f"Family '{fam}' used non-EODHD source '{source}' under STAGE_B_EODHD_ONLY=1"
                    )

        # VALIDATION: Verify loaded data before proceeding
        if df is not None and isinstance(df, pd.DataFrame):
            if df.empty:
                logger.warning(f"⚠️ {fam}: DataFrame is empty after loading/generation")
                telemetry[fam] = DORMANT
                if _require_all_families_enabled():
                    raise RuntimeError(f"Required family '{fam}' produced empty DataFrame")
                continue
            
            # Check for valid columns
            if len(df.columns) == 0:
                logger.warning(f"⚠️ {fam}: DataFrame has no columns")
                telemetry[fam] = DORMANT
                if _require_all_families_enabled():
                    raise RuntimeError(f"Required family '{fam}' produced DataFrame with no columns")
                continue
            
            # Check for valid data (not all NaN)
            if df.isnull().all().all():
                logger.warning(f"⚠️ {fam}: All values are NaN")
                telemetry[fam] = DORMANT
                if _require_all_families_enabled():
                    raise RuntimeError(f"Required family '{fam}' produced all-NaN DataFrame")
                continue
            
            # Success: valid data loaded
            logger.info(f"✅ {fam}: Validated {len(df)} rows × {len(df.columns)} cols")
            frames.append(_collect(df, fam))
        else:
            logger.warning(f"⚠️ {fam}: Invalid data type or None")
            telemetry[fam] = DORMANT
            if _require_all_families_enabled():
                raise RuntimeError(f"Required family '{fam}' produced invalid output type")

    if not frames:
        logger.error(f"❌ VALIDATION FAILED: No valid data loaded for any of {len(fams)} families")
        logger.error(f"   Requested families: {fams}")
        logger.error(f"   Telemetry: {telemetry}")
        out = pd.DataFrame()
        out.attrs['provenance'] = provenance
        # mark all requested families as dormant if nothing emitted
        for fam in fams:
            telemetry.setdefault(fam, DORMANT)
        out.attrs['telemetry'] = telemetry
        out.attrs['feature_counts'] = {'total': 0}
        out.attrs['families'] = fams
        return out

    logger.info(f"✅ VALIDATION PASSED: Loaded {len(frames)} valid families out of {len(fams)} requested")
    
    # Merge on outer index
    panel = frames[0]
    for df in frames[1:]:
        # Avoid pandas overlap errors when two families emit the same column names
        # (e.g., hf_* shared outputs). Prefer the first-loaded version.
        overlap = panel.columns.intersection(df.columns)
        if len(overlap) > 0:
            logger.debug(
                "Dropping %d overlapping columns before join (examples: %s)",
                len(overlap),
                list(overlap)[:5],
            )
            df = df.drop(columns=overlap)
        if not df.empty and len(df.columns) > 0:
            panel = panel.join(df, how='outer')
    panel = panel.sort_index()
    
    # Final validation
    if panel.empty:
        logger.error(f"❌ FINAL VALIDATION FAILED: Panel is empty after merge")
        return pd.DataFrame()
    
    if len(panel.columns) == 0:
        logger.error(f"❌ FINAL VALIDATION FAILED: Panel has no columns")
        return pd.DataFrame()
    
    logger.info(f"✅ FINAL VALIDATION PASSED: Panel has {len(panel)} rows × {len(panel.columns)} columns")
    
    # IMPORTANT: Do not forward/back-fill at merge stage.
    # Each family handler is responsible for its own time-alignment semantics.
    # Global ffill/bfill causes leakage (e.g., propagating an options snapshot or an
    # earnings transcript event across historical windows) and fabricates *_has_data.
    finbert_cols = [c for c in panel.columns if c.startswith('finbert_')]
    other_cols = [c for c in panel.columns if c not in finbert_cols]

    if other_cols:
        panel[other_cols] = panel[other_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Attach attrs
    panel.attrs['provenance'] = provenance
    # Attach telemetry with aggregation notes
    tel = dict(telemetry)
    try:
        finbert_gap = None
        if isinstance(telemetry.get('finbert'), dict):
            finbert_gap = telemetry['finbert'].get('gap_thresh')
        agg_notes = {
            'ffill_non_finbert': bool(other_cols),
            'finbert_preserve': bool(finbert_cols),
            'finbert_gap_thresh': finbert_gap,
            'corr': {
                'windows': cw,
                'breaks': {'low': float(cb[0]), 'high': float(cb[1])},
                'flat_gate_n': int(corr_flat_gate_n),
                'flat_eps': float(corr_flat_eps),
            },
        }
        tel.setdefault('aggregation', {}).update(agg_notes)  # type: ignore
    except Exception:
        tel['aggregation'] = {
            'ffill_non_finbert': bool(other_cols),
            'finbert_preserve': bool(finbert_cols)
        }  # type: ignore
    panel.attrs['telemetry'] = tel
    total_cols = int(sum(feature_counts.values()))
    panel.attrs['feature_counts'] = {**feature_counts, 'total': total_cols}
    panel.attrs['families'] = fams
    if stage is not None:
        panel.attrs['stage'] = stage
    panel = _apply_stable_schema_stubs(panel)
    
    # Post-maturity flag computation from JSONL events
    panel = _compute_flags_from_events(panel, symbol, horizon, cache_dir)
    
    return panel


def _compute_flags_from_events(panel: pd.DataFrame, symbol: str, horizon: int, cache_dir: Optional[Path]) -> pd.DataFrame:
    """Compute tracking flags from JSONL event logs post-maturity.
    
    This reads drift/retraining/recalibration events logged during generation
    and computes flags for each row based on event history. "Post-maturity" means
    this happens AFTER family generation during panel merge, NOT filtering by date.
    
    Args:
        panel: Merged panel DataFrame
        symbol: Symbol name
        horizon: Forecast horizon
        cache_dir: Cache directory path
    
    Returns:
        Panel with updated flags (days_since_update, drift_flag, etc.)
    """
    if panel.empty or cache_dir is None:
        return panel
    
    event_log_dir = Path(cache_dir) / 'event_logs'
    
    # Process each family's events
    # NOTE: calibration and online_learning are now GOVERNANCE LAYERS, not families.
    # Their events are managed via artifacts/governance/, not prep_families.
    # Only quantile_forecast remains as a family with event tracking here.
    for family in ['quantile_forecast']:
        try:
            # Load materialization date
            mat_file = event_log_dir / f'{symbol}_h{horizon}_{family}_materialization.json'
            if not mat_file.exists():
                continue
                
            with open(mat_file) as f:
                mat_data = json.load(f)
                mat_date = pd.to_datetime(mat_data['last_materialization'])
            
            # Compute days_since_update for ALL rows
            col_name = f'{family}_days_since_update'
            if col_name in panel.columns:
                panel[col_name] = np.maximum((mat_date - panel.index).days, 0)
            
            # NOTE: online_learning event processing removed.
            # Online learning is now a GOVERNANCE LAYER, not a family.
            # Event tracking for online_learning happens via:
            #   artifacts/governance/online_learning/
            # See: src/stage_b_stateful/governance_persistence.py
                
        except Exception as e:
            logger.warning(f"Failed to compute flags from events for {family}: {e}")
    
    return panel


__all__ = ['build_panel']
