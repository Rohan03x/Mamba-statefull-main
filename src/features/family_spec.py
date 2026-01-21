"""Central registry for feature families used across prep_families and Stage B.

Provides a single source of truth for:
- Stage (A/B/HF/META) membership
- Dependency ordering between families
- Canonical governance columns (has_data/activity/days_since_update)
- Optional summary and reliability metadata hooks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FamilySpec:
    """Declarative description of a feature family."""

    name: str
    stage: str  # "A", "B", "HF_MODULE", "HF_BLOCK", "META"
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    reliability_cols: List[str] = field(default_factory=list)
    drift_cols: List[str] = field(default_factory=list)
    summary_fn: Optional[str] = None


SummaryFn = Callable[[str, pd.DataFrame], Dict[str, pd.Series]]
SUMMARY_FN_REGISTRY: Dict[str, SummaryFn] = {}


def register_summary_fn(name: str) -> Callable[[SummaryFn], SummaryFn]:
    def _decorator(func: SummaryFn) -> SummaryFn:
        SUMMARY_FN_REGISTRY[name] = func
        return func

    return _decorator


@register_summary_fn("mean_std")
def _mean_std_summary(family: str, frame: pd.DataFrame) -> Dict[str, pd.Series]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    return {
        f"{family}_mean": numeric.mean(axis=1),
        f"{family}_std": numeric.std(axis=1).fillna(0.0),
    }


@register_summary_fn("quantile_summary")
def _quantile_summary(family: str, frame: pd.DataFrame) -> Dict[str, pd.Series]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    q_loc = numeric.median(axis=1)
    q_spread = numeric.quantile(0.95, axis=1) - numeric.quantile(0.05, axis=1)
    tail = numeric.quantile(0.99, axis=1) - numeric.quantile(0.01, axis=1)
    vol = numeric.std(axis=1).fillna(0.0)
    skew_proxy = (numeric - numeric.mean(axis=1).values.reshape(-1, 1)).pow(3).mean(axis=1)
    return {
        f"{family}_q_loc": q_loc,
        f"{family}_q_spread_95_5": q_spread,
        f"{family}_q_tail_99_01": tail,
        f"{family}_q_vol_forecast": vol,
        f"{family}_q_skew_proxy": skew_proxy.fillna(0.0),
    }


@register_summary_fn("arima_summary")
def _arima_summary(family: str, frame: pd.DataFrame) -> Dict[str, pd.Series]:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    level = numeric.mean(axis=1)
    trend = level.diff().fillna(0.0)
    resid = numeric.sub(level, axis=0)
    resid_vol = resid.pow(2).mean(axis=1).pow(0.5).fillna(0.0)
    accel = trend.diff().fillna(0.0)
    return {
        f"{family}_ar_level": level,
        f"{family}_ar_trend": trend,
        f"{family}_ar_resid_vol": resid_vol,
        f"{family}_ar_accel": accel,
    }


def compute_family_summary(family: str, frame: pd.DataFrame) -> Dict[str, pd.Series]:
    spec = FAMILY_SPECS.get(family)
    if spec is None or not spec.summary_fn:
        return {}
    fn = SUMMARY_FN_REGISTRY.get(spec.summary_fn)
    if fn is None:
        logger.warning("Summary function '%s' missing for %s", spec.summary_fn, family)
        return {}
    try:
        return fn(family, frame)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Summary function '%s' failed for %s: %s", spec.summary_fn, family, exc)
        return {}


def canonical_metric_columns(family: str) -> Tuple[str, str, str]:
    """Return the 3 required governance columns for a family.
    
    Returns:
        Tuple of (has_data_col, activity_col, days_since_update_col)
    """
    has_data = f"{family}_has_data"
    activity = f"{family}_activity"
    days_since = f"{family}_days_since_update"
    return has_data, activity, days_since


LEGACY_BASE_FAMILY_ORDER: List[str] = [
    "quantile_forecast",
    "arima_forecast",
    "cross_asset",
    "exchange_calendar",
    "econ_events_calendar",
    "corp_actions_splits",
    "marketcap_history",
    "garch_iv",
    "cboe_term",
    "correlation",
    "candle_mechanics",
    "microstructure",
    "options",
    "short_interest",
    "insider_form4",
    "index_constituents",
    "subsidiary",
    "earnings",
    "dividends",
    "alternative_signals",
    "ml_framework",
    "multiasset",
    "regime",
    "options_anchoring",
    "tft_features",
    "fin_g2",
    "fin_g3",
    "fin_g4",
    "fin_g5",
    "fin_g6",
    "peer_screener_context",
    "fin_g7",
    "finbert",
    "dcf",
    "calibration",
    "online_learning",
]

LEGACY_HF_MODULE_ORDER = [
    "earnings_transcript_hf",
    "doc_embedding_novelty_hf",
    "macro_tst_hf",
]

HF_BLOCK_ORDER = [
    "tech_micro_hf",
    "forecast_hf",
    "vol_deriv_hf",
    "macro_regime_hf",
    "fundamental_val_hf",
    "news_nlp_hf",
]

# Cache partitioning policy for HF blocks.
#
# The requested architecture is that most HF blocks are *symbol-only* caches
# (compute once per symbol, shared across horizons), while only blocks that
# depend on horizon-bound families remain horizon-scoped.
HORIZON_BOUND_HF_BLOCKS: Set[str] = {"forecast_hf"}
SYMBOL_ONLY_HF_BLOCKS: Set[str] = set(HF_BLOCK_ORDER) - set(HORIZON_BOUND_HF_BLOCKS)

META_FAMILIES = ["hf_agg"]


def _ordered_subset(candidates: Iterable[str], preferred: Sequence[str]) -> List[str]:
    candidate_set = [c for c in candidates]
    seen = set()
    ordered: List[str] = []
    for token in preferred:
        if token in candidate_set and token not in seen:
            ordered.append(token)
            seen.add(token)
    for token in sorted(candidate_set):
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def default_base_families() -> List[str]:
    candidates = [
        name
        for name, spec in FAMILY_SPECS.items()
        if spec.stage in {"A", "B"} and "hf" not in spec.tags and "meta" not in spec.tags
    ]
    return _ordered_subset(candidates, LEGACY_BASE_FAMILY_ORDER)


def default_hf_modules() -> List[str]:
    candidates = [name for name, spec in FAMILY_SPECS.items() if spec.stage == "HF_MODULE"]
    return _ordered_subset(candidates, LEGACY_HF_MODULE_ORDER)


def default_hf_blocks() -> List[str]:
    candidates = [name for name, spec in FAMILY_SPECS.items() if spec.stage == "HF_BLOCK"]
    return _ordered_subset(candidates, HF_BLOCK_ORDER)


def families_for_stage(stage: str) -> List[str]:
    stage_upper = (stage or "B").upper()
    if stage_upper == "A":
        return _ordered_subset(
            [name for name, spec in FAMILY_SPECS.items() if spec.stage == "A"],
            LEGACY_BASE_FAMILY_ORDER,
        )
    if stage_upper == "B":
        combined: List[str] = []
        combined.extend(default_base_families())
        combined.extend(default_hf_modules())
        combined.extend(default_hf_blocks())
        combined.extend(META_FAMILIES)
        return combined
    return _ordered_subset(FAMILY_SPECS.keys(), LEGACY_BASE_FAMILY_ORDER)


def get_family_spec(name: str) -> Optional[FamilySpec]:
    return FAMILY_SPECS.get(name)


def family_dependencies(name: str) -> List[str]:
    spec = FAMILY_SPECS.get(name)
    return list(spec.dependencies) if spec else []


_BASE_SPECS: List[Tuple[str, Dict[str, object]]] = [
    (
        "quantile_forecast",
        {
            "stage": "B",
            "tags": {"base", "probability"},
            "summary_fn": "quantile_summary",
        },
    ),
    (
        "arima_forecast",
        {
            "stage": "B",
            "tags": {"ts_forecast"},
            "summary_fn": "arima_summary",
        },
    ),
    (
        "calibration",
        {
            "stage": "B",
            "tags": {"calibration"},
            "dependencies": ["quantile_forecast"],
            "reliability_cols": ["cal_ECE", "cal_ECE_max", "cal_quality"],
        },
    ),
    (
        "online_learning",
        {
            "stage": "B",
            "tags": {"drift", "online"},
            "dependencies": ["quantile_forecast", "calibration"],
            "reliability_cols": ["alpha_weight"],
            "drift_cols": ["drift_gap", "drift_eps", "drift_flag", "drift_severity"],
        },
    ),
    ("cross_asset", {"stage": "A", "tags": {"macro"}}),
    ("exchange_calendar", {"stage": "A", "tags": {"calendar", "structure"}}),
    ("econ_events_calendar", {"stage": "A", "tags": {"macro", "event"}}),
    ("corp_actions_splits", {"stage": "A", "tags": {"fundamental", "event"}}),
    ("marketcap_history", {"stage": "A", "tags": {"fundamental", "size"}}),
    ("garch_iv", {"stage": "A", "tags": {"vol"}}),
    ("cboe_term", {"stage": "A", "tags": {"vol"}}),
    ("correlation", {"stage": "A", "tags": {"correlation"}}),
    ("candle_mechanics", {"stage": "A", "tags": {"price_action", "ohlcv"}}),
    ("microstructure", {"stage": "A", "tags": {"microstructure"}}),
    ("options", {"stage": "A", "tags": {"derivatives"}}),
    ("short_interest", {"stage": "A", "tags": {"positioning"}}),
    ("insider_form4", {"stage": "A", "tags": {"fundamental", "insider", "event"}}),
    ("index_constituents", {"stage": "A", "tags": {"fundamental", "index"}}),
    ("subsidiary", {"stage": "A", "tags": {"fundamental"}}),
    ("earnings", {"stage": "A", "tags": {"fundamental"}}),
    ("dividends", {"stage": "A", "tags": {"fundamental"}}),
    ("alternative_signals", {"stage": "A", "tags": {"alternative"}}),
    ("ml_framework", {"stage": "A", "tags": {"ml"}}),
    ("multiasset", {"stage": "A", "tags": {"macro"}}),
    ("regime", {"stage": "A", "tags": {"regime"}}),
    ("options_anchoring", {"stage": "A", "tags": {"derivatives"}}),
    ("tft_features", {"stage": "A", "tags": {"sequence"}}),
    ("finbert", {"stage": "A", "tags": {"nlp"}}),
    ("dcf", {"stage": "A", "tags": {"valuation"}}),
]

HF_MODULE_SPECS: List[Tuple[str, Dict[str, object]]] = [
    (
        "news_sentiment_hf",
        {"stage": "HF_MODULE", "tags": {"hf", "nlp"}},
    ),
    (
        "earnings_transcript_hf",
        {"stage": "HF_MODULE", "tags": {"hf", "nlp"}},
    ),
    (
        "doc_embedding_novelty_hf",
        {"stage": "HF_MODULE", "tags": {"hf", "nlp"}},
    ),
    (
        "macro_tst_hf",
        {"stage": "HF_MODULE", "tags": {"hf", "macro"}},
    ),
]

HF_BLOCK_SPECS: List[Tuple[str, Dict[str, object]]] = [
    (
        "tech_micro_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "microstructure"},
            "dependencies": ["ml_framework", "microstructure", "correlation"],
        },
    ),
    (
        "forecast_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "ensemble"},
            "dependencies": [
                "quantile_forecast",
                "calibration",
                "online_learning",
                "arima_forecast",
                "tft_features",
            ],
        },
    ),
    (
        "vol_deriv_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "vol"},
            "dependencies": [
                "garch_iv",
                "cboe_term",
                "options_anchoring",
                "options",
                "short_interest",
            ],
        },
    ),
    (
        "macro_regime_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "macro"},
            "dependencies": [
                "macro_tst_hf",
                "regime",
                "multiasset",
                "cross_asset",
                "correlation",
            ],
        },
    ),
    (
        "fundamental_val_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "fundamental"},
            "dependencies": [
                "fin_g1",
                "fin_g2",
                "fin_g3",
                "fin_g4",
                "fin_g5",
                "fin_g6",
                "fin_g7",
                "earnings",
                "dividends",
                "dcf",
                "subsidiary",
                "short_interest",
            ],
        },
    ),
    (
        "news_nlp_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "nlp"},
            "dependencies": [
                "finbert",
                "doc_embedding_novelty_hf",
                "earnings_transcript_hf",
                "news_sentiment_hf",
                "alternative_signals",
            ],
        },
    ),
]

META_SPEC: List[Tuple[str, Dict[str, object]]] = [
    (
        "hf_agg",
        {
            "stage": "META",
            "tags": {"hf", "meta"},
            "dependencies": HF_BLOCK_ORDER,
        },
    )
]

FAMILY_SPECS: Dict[str, FamilySpec] = {}
for name, payload in _BASE_SPECS + HF_MODULE_SPECS + HF_BLOCK_SPECS + META_SPEC:
    FAMILY_SPECS[name] = FamilySpec(name=name, **payload)

for idx in range(1, 8):
    token = f"fin_g{idx}"
    FAMILY_SPECS[token] = FamilySpec(name=token, stage="A", tags={"fundamental", "fin_g"})

FAMILY_SPECS.setdefault(
    "peer_screener_context",
    FamilySpec(
        name="peer_screener_context",
        stage="A",
        tags={"cross_sectional", "peer", "screener"},
        dependencies=["fin_g6"],
        summary_fn="mean_std",
    ),
)

# Additional optional/API dependent families that are not always enabled
for optional_name in ("commodities", "crypto", "fx"):
    FAMILY_SPECS.setdefault(
        optional_name,
        FamilySpec(name=optional_name, stage="A", tags={"optional"}),
    )

__all__ = [
    "FamilySpec",
    "FAMILY_SPECS",
    "canonical_metric_columns",
    "compute_family_summary",
    "default_base_families",
    "default_hf_blocks",
    "default_hf_modules",
    "families_for_stage",
    "family_dependencies",
    "get_family_spec",
]
