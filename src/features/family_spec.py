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


GOVERNANCE_SUFFIXES: Set[str] = {
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
    "conf",
    "coverage_pct",
    "missing_ratio",
    "source_lag_days",
    "is_valid",
    "is_stale",
}

# Event core column lists (canonical) for update-point detection.
EVENT_CORE_COLS: Dict[str, List[str]] = {
    "corp_actions_splits": [
        "corp_actions_splits_flag",
        "corp_actions_splits_split_ratio",
        "corp_actions_splits_split_ratio_log",
        "corp_actions_splits_split_count_5y",
        "corp_actions_splits_reverse_split_flag",
    ],
    "dividends": [
        "dividends_dividend_amount",
        "dividends_dividend_frequency",
        "dividends_payout_ratio_proxy",
        "dividends_dividend_change_1y",
        "dividends_dividend_cut_flag",
        "dividends_dividend_increase_flag",
    ],
    "earnings": [
        "earnings_eps_surprise_pct",
        "earnings_revenue_surprise_pct",
        "earnings_eps_surprise_z_126",
        "earnings_revenue_surprise_z_126",
        "earnings_guidance_surprise_proxy",
        "earnings_analyst_revision_breadth_30d",
        "earnings_estimate_dispersion_score",
        "earnings_calendar_confirmed",
    ],
    "insider_form4": [
        "insider_form4_num_trades_1d",
        "insider_form4_num_unique_insiders_1d",
        "insider_form4_net_shares_1d",
        "insider_form4_net_value_1d",
        "insider_form4_buy_value_1d",
        "insider_form4_sell_value_1d",
        "insider_form4_exec_net_value_1d",
        "insider_form4_cluster_buy_value_1d",
    ],
    "index_constituents": [
        "index_constituents_member_gspc",
        "index_constituents_added_gspc",
        "index_constituents_removed_gspc",
        "index_constituents_weight_gspc",
        "index_constituents_member_dji",
        "index_constituents_added_dji",
        "index_constituents_removed_dji",
        "index_constituents_weight_dji",
        "index_constituents_num_indices",
    ],
    "doc_embedding_novelty_hf": [
        "doc_embedding_novelty_hf_n_events",
        "doc_embedding_novelty_hf_n_articles",
        "doc_embedding_novelty_hf_top_theme",
        "doc_embedding_novelty_hf_theme_weight",
        "doc_embedding_novelty_hf_novelty_score_macro",
        "doc_embedding_novelty_hf_novelty_score_micro",
        "doc_embedding_novelty_hf_novelty_cluster_entropy",
        "doc_embedding_novelty_hf_novelty_spike_flag",
        "doc_embedding_novelty_hf_conf",
    ],
    "news_sentiment_hf": [
        "news_sentiment_hf_score",
        "news_sentiment_hf_conf",
    ],
}

ECON_EVENT_CORE_INCLUDE_PREFIXES: Tuple[str, ...] = (
    "econ_events_calendar_event_occurrence_",
    "econ_events_calendar_pulse_occurrence_",
    "econ_events_calendar_pulse_surprise_",
    "econ_events_calendar_pulse_strength_",
    "econ_events_calendar_surprise_",
    "econ_events_calendar_surprise_has_forecast_",
)

ECON_EVENT_CORE_EXCLUDE_PREFIXES: Tuple[str, ...] = (
    "econ_events_calendar_prox_next_",
    "econ_events_calendar_recency_last_",
    "econ_events_calendar_window_pre_",
    "econ_events_calendar_window_post_",
    "econ_events_calendar_macro_upcoming_",
)

ECON_EVENT_CORE_INCLUDE_EXACT: Tuple[str, ...] = (
    "econ_events_calendar_macro_shock_major",
    "econ_events_calendar_macro_surprise_signed",
)


def resolve_core_columns(family: str, columns: Sequence[str]) -> List[str]:
    """Return core columns for a family, excluding governance/diagnostics.

    For EVENT families, uses explicit lists/patterns. Otherwise defaults to
    family-prefixed numeric columns excluding governance/diagnostic suffixes.
    """
    fam = str(family).strip().lower()
    cols = [str(c) for c in columns]

    def _is_governance(col: str) -> bool:
        for suffix in GOVERNANCE_SUFFIXES:
            if col.lower().endswith(f"_{suffix}"):
                return True
        return False

    if fam == "econ_events_calendar":
        included: List[str] = []
        for c in cols:
            cl = c.lower()
            if cl in ECON_EVENT_CORE_INCLUDE_EXACT:
                included.append(c)
                continue
            if any(cl.startswith(pfx) for pfx in ECON_EVENT_CORE_INCLUDE_PREFIXES):
                if not any(cl.startswith(pfx) for pfx in ECON_EVENT_CORE_EXCLUDE_PREFIXES):
                    included.append(c)
        return included

    if fam in EVENT_CORE_COLS:
        wanted = set(EVENT_CORE_COLS[fam])
        return [c for c in cols if c in wanted]

    # Default: family-prefixed columns excluding governance/diagnostics.
    prefix = f"{fam}_"
    core = [c for c in cols if c.lower().startswith(prefix) and not _is_governance(c)]
    return core


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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HF Block Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# HF blocks are intermediate inference steps that run AFTER Mamba and BEFORE
# portfolio allocation.  They annotate/modulate the Mamba signal but cannot
# flip direction.  Only blocks with proven edge should remain active.
#
# ACTIVE blocks (proven signal-to-noise):
#   - tech_micro_hf   : Technical + microstructure (high-frequency, low noise)
#   - forecast_hf     : Forecast ensemble (calibrated, horizon-bound)
#
# DISABLED blocks (noise/redundancy/sparsity):
#   - vol_deriv_hf        : Duplicates portfolio risk module
#   - macro_regime_hf     : Macro data is low-frequency, ≠ sequence learning
#   - fundamental_val_hf  : Low frequency, sparse updates (quarterly)
#   - news_nlp_hf         : Noise amplification, sentiment already in base families
#
# This restraint is what real hedge funds do: fewer signals, higher conviction.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF_BLOCK_ORDER = [
    "tech_micro_hf",
    "forecast_hf",
    "event_risk_hf",  # Portfolio-only event risk management
]

# Disabled HF blocks - kept for documentation and potential future reactivation.
DISABLED_HF_BLOCKS: Set[str] = {
    "vol_deriv_hf",        # Duplicates portfolio risk
    "macro_regime_hf",     # Macro ≠ sequence learning
    "fundamental_val_hf",  # Low frequency, sparse
    "news_nlp_hf",         # Noise amplification
}

# Cache partitioning policy for HF blocks.
#
# The requested architecture is that most HF blocks are *symbol-only* caches
# (compute once per symbol, shared across horizons), while only blocks that
# depend on horizon-bound families remain horizon-scoped.
HORIZON_BOUND_HF_BLOCKS: Set[str] = {"forecast_hf"}
SYMBOL_ONLY_HF_BLOCKS: Set[str] = set(HF_BLOCK_ORDER) - set(HORIZON_BOUND_HF_BLOCKS)

# hf_agg is DISABLED - HF block outputs are consumed directly by Stage B/C
# without an intermediate aggregation step.
META_FAMILIES: List[str] = []  # was ["hf_agg"]


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
    # NOTE: calibration and online_learning are DISABLED - not part of current pipeline
    # They were: calibration -> depends on quantile_forecast
    #            online_learning -> depends on quantile_forecast, calibration
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
    ("event_time_bars", {"stage": "A", "tags": {"microstructure", "intraday"}}),
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HF Block Input Eligibility (Non-Negotiable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# A base family may be consumed by an HF block ONLY if ALL THREE are true:
#
#   1. DENSE IN TIME      → Meaningful values on ≥80% of trading days
#   2. STATIONARY         → Changes, spreads, ratios; NOT raw levels or counts
#   3. SEQUENCE-NATIVE    → Information is in patterns across days, not impulses
#
# If any one fails → DO NOT FEED HF.
# This rule separates hedge-fund ML from Kaggle ML.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF_BLOCK_SPECS: List[Tuple[str, Dict[str, object]]] = [
    (
        "tech_micro_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "microstructure"},
            # ─────────────────────────────────────────────────────────────────
            # Eligibility constraints (enforced at generator level):
            #   ml_framework     : Full (already normalized technicals)
            #   microstructure   : Full (price mechanics are sequence-native)
            #   correlation      : Full (rolling windows, stationary)
            #   candle_mechanics : Normalized anatomy only, no raw returns
            #   garch_iv         : Cap ~6 features (changes, ratios, residuals)
            #   options          : Flows, skews, changes only (no OI/levels)
            #   options_anchoring: Distance-to-strike, pinning strength only
            # ─────────────────────────────────────────────────────────────────
            "dependencies": [
                "ml_framework",
                "microstructure",
                "correlation",
                "candle_mechanics",
                "garch_iv",
                "options",
                "options_anchoring",
            ],
        },
    ),
    (
        "forecast_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "ensemble"},
            # ─────────────────────────────────────────────────────────────────
            # Eligibility constraints (enforced at generator level):
            #   quantile_forecast : Full (probability distributions)
            #   arima_forecast    : Full (forecast residuals, changes)
            #   tft_features      : Full (temporal patterns, normalized)
            #   regime            : Full (regime states are sequence-native)
            #   cboe_term         : Slope & curvature changes only (no levels)
            #   short_interest    : Smoothed deltas, z-scores only (no raw SI%)
            # NOTE: calibration and online_learning removed from pipeline
            # ─────────────────────────────────────────────────────────────────
            "dependencies": [
                "quantile_forecast",
                "arima_forecast",
                "tft_features",
                "regime",
                "cboe_term",
                "short_interest",
            ],
        },
    ),
    (
        "twitter_twikit_hf",
        {
            "stage": "DISABLED",
            "tags": {"hf", "sentiment", "social"},
            # ─────────────────────────────────────────────────────────────────
            # DISABLED: Reddit sentiment analysis (via YARS - no auth required)
            # Reason: Requires ~1,560 API calls for 15-year range (30+ minutes)
            #         Weekly batching needed for historical coverage
            # Three-channel architecture:
            #   - Channel A: Symbol cashtag search (e.g., "AAPL")
            #   - Channel B: Company name search (e.g., "Apple Inc stock")
            #   - Channel C: Curated accounts (future)
            #
            # Features (24 total):
            #   - Governance: has_data, activity, sample_n, coverage
            #   - Attention: tweet_count, engagement metrics
            #   - Sentiment: FinBERT scores, neg/pos/neu fractions
            #   - Novelty: sentence-transformers, topic extraction
            #   - Portfolio: event_risk, panic, sent_regime
            #
            # Data source: Reddit (r/wallstreetbets, r/stocks, r/investing)
            # ML models: FinBERT + sentence-transformers
            # No dependencies - direct data source
            # ─────────────────────────────────────────────────────────────────
            "dependencies": [],
        },
    ),
    (
        "event_risk_hf",
        {
            "stage": "HF_BLOCK",
            "tags": {"hf", "risk", "event", "portfolio_only"},
            # ─────────────────────────────────────────────────────────────────
            # Event Risk Management for Portfolio Parquet (NOT Mamba)
            #
            # This block provides explicit, controllable event risk features:
            #   - event_risk_hf_earnings_next_1d (0/1)
            #   - event_risk_hf_earnings_next_3d (0/1)
            #   - event_risk_hf_macro_next_1d (0/1)
            #   - event_risk_hf_score (0..1)
            #   - event_risk_hf_earnings_imminent (continuous 0..1)
            #   - event_risk_hf_macro_imminent (continuous 0..1)
            #
            # Routing: ALL columns → RISK role → Portfolio parquet ONLY
            # Data sources: earnings family (EODHD), econ_events_calendar
            # ─────────────────────────────────────────────────────────────────
            "dependencies": [
                "earnings",
                "econ_events_calendar",
            ],
        },
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # DISABLED HF BLOCKS (kept for reference, not loaded into FAMILY_SPECS)
    # ─────────────────────────────────────────────────────────────────────────
    # These blocks are disabled due to noise/redundancy/sparsity concerns.
    # See DISABLED_HF_BLOCKS constant for rationale.
    #
    # (
    #     "vol_deriv_hf",
    #     {
    #         "stage": "HF_BLOCK",
    #         "tags": {"hf", "vol"},
    #         "dependencies": [
    #             "garch_iv", "cboe_term", "options_anchoring", "options", "short_interest",
    #         ],
    #     },
    # ),
    # (
    #     "macro_regime_hf",
    #     {
    #         "stage": "HF_BLOCK",
    #         "tags": {"hf", "macro"},
    #         "dependencies": [
    #             "macro_tst_hf", "regime", "multiasset", "cross_asset", "correlation",
    #         ],
    #     },
    # ),
    # (
    #     "fundamental_val_hf",
    #     {
    #         "stage": "HF_BLOCK",
    #         "tags": {"hf", "fundamental"},
    #         "dependencies": [
    #             "fin_g1", "fin_g2", "fin_g3", "fin_g4", "fin_g5", "fin_g6", "fin_g7",
    #             "earnings", "dividends", "dcf", "subsidiary", "short_interest",
    #         ],
    #     },
    # ),
    # (
    #     "news_nlp_hf",
    #     {
    #         "stage": "HF_BLOCK",
    #         "tags": {"hf", "nlp"},
    #         "dependencies": [
    #             "finbert", "doc_embedding_novelty_hf", "earnings_transcript_hf",
    #             "news_sentiment_hf", "alternative_signals",
    #         ],
    #     },
    # ),
    # ─────────────────────────────────────────────────────────────────────────
]

# hf_agg DISABLED - see META_FAMILIES comment above.
# HF block signals are consumed directly by build_panel without aggregation.
META_SPEC: List[Tuple[str, Dict[str, object]]] = [
    # (
    #     "hf_agg",
    #     {
    #         "stage": "META",
    #         "tags": {"hf", "meta"},
    #         "dependencies": HF_BLOCK_ORDER,
    #     },
    # )
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
    "DISABLED_HF_BLOCKS",
    "HF_BLOCK_ORDER",
    "HORIZON_BOUND_HF_BLOCKS",
    "SYMBOL_ONLY_HF_BLOCKS",
    "canonical_metric_columns",
    "compute_family_summary",
    "default_base_families",
    "default_hf_blocks",
    "default_hf_modules",
    "families_for_stage",
    "family_dependencies",
    "get_family_spec",
]
