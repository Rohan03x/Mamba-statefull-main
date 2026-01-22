# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

"""Deterministic feature-role classification + normalization utilities.

Implements a practical, scalable workflow:
- Declare a primary intent once per family (fallback role).
- Classify columns by ordered naming/domain rules.
- Apply role-specific transforms programmatically.

This module is intentionally lightweight and has no heavy deps (no scipy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd


class FeatureRole(str, Enum):
    HYGIENE = "hygiene"
    REGIME = "regime"
    RISK = "risk"
    PREDICTIVE = "predictive"


@dataclass(frozen=True)
class FamilyMeta:
    primary_intent: FeatureRole
    update_cadence: str = "unknown"  # e.g. "daily", "weekly", "snapshot"
    decay: str = "unknown"  # e.g. "slow", "fast", "event"
    # Alpha usability policy: "true" | "false" | "conditional".
    # This is enforced AFTER feature-level role classification.
    allow_direct_alpha: str = "false"

    # Allowed-usage defaults (family-level). These should be conservative.
    risk_scale_ok: bool = False
    gating_ok: bool = False
    veto_ok: bool = False

    # Point-in-time + latency hints.
    requires_point_in_time: bool = False
    update_latency_class: str = "unknown"  # e.g. t_plus_1, same_day_close

    # Decay/sparsity hints.
    decay_half_life_days: float = 0.0
    expected_sparsity: str = "unknown"  # e.g. dense, sparse, event


# -----------------------------------------------------------------------------
# Family-level primary intents (fallback).
# Keep this small and stable; rules do most of the work.
# -----------------------------------------------------------------------------
DEFAULT_FAMILY_META: Dict[str, FamilyMeta] = {
    # Predictive / alpha-oriented
    "dcf": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="quarterly", decay="slow", allow_direct_alpha="false"),
    "ml_framework": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="fast", allow_direct_alpha="true"),
    "finbert": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="fast", allow_direct_alpha="true"),
    # Registry intent is "mixed"; roles don't have "mixed", so keep predictive fallback but gate executability.
    "alternative_signals": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="irregular", decay="fast", allow_direct_alpha="conditional"),
    "quantile_forecast": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="fast", allow_direct_alpha="true"),

    # Risk / uncertainty oriented
    "garch_iv": FamilyMeta(FeatureRole.RISK, update_cadence="daily", decay="medium", allow_direct_alpha="false"),
    "cboe_term": FamilyMeta(FeatureRole.RISK, update_cadence="daily", decay="medium", allow_direct_alpha="false"),
    "microstructure": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="fast", allow_direct_alpha="true"),
    "correlation": FamilyMeta(FeatureRole.RISK, update_cadence="daily", decay="none", allow_direct_alpha="false"),

    # Cross-asset / multi-asset structural stat-arb
    "cross_asset": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="slow", allow_direct_alpha="true"),
    "multiasset": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="slow", allow_direct_alpha="true"),

    # Mixed families (fallback still needed; rules should classify most columns)
    "options": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="daily", decay="fast", allow_direct_alpha="conditional"),
    "short_interest": FamilyMeta(FeatureRole.RISK, update_cadence="snapshot", decay="slow", allow_direct_alpha="true"),

    # Context/regime
    "regime": FamilyMeta(FeatureRole.REGIME, update_cadence="daily", decay="medium", allow_direct_alpha="false"),
    # Registry intent is "mixed"; roles don't have "mixed", so keep predictive fallback but gate executability.
    "macro_tst_hf": FamilyMeta(FeatureRole.PREDICTIVE, update_cadence="event", decay="slow", allow_direct_alpha="conditional"),
    "econ_events_calendar": FamilyMeta(FeatureRole.REGIME, update_cadence="event", decay="none", allow_direct_alpha="false"),

    # Macro regime summaries
    "macro_regime_hf": FamilyMeta(FeatureRole.REGIME, update_cadence="daily", decay="none", allow_direct_alpha="false"),

    # Hygiene / gating
    "listing_status": FamilyMeta(FeatureRole.HYGIENE, update_cadence="daily", decay="slow", allow_direct_alpha="false"),
    "exchange_calendar": FamilyMeta(FeatureRole.REGIME, update_cadence="daily", decay="none", allow_direct_alpha="false"),
}


def _repo_root() -> Path:
    # src/features/feature_roles.py -> repo root is two parents up from src/
    return Path(__file__).resolve().parents[2]


def _coerce_allow_policy(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    token = str(v or "").strip().lower()
    if token in {"true", "false", "conditional"}:
        return token
    if token in {"1", "yes", "y", "t"}:
        return "true"
    if token in {"0", "no", "n", "f"}:
        return "false"
    return "false"


def _intent_to_feature_role(token: str) -> FeatureRole:
    t = str(token or "").strip().lower()
    if t in {"predictive", "risk", "regime", "hygiene"}:
        return FeatureRole(t)
    # Canonical registry supports "mixed"; roles do not. Treat as predictive fallback.
    if t == "mixed":
        return FeatureRole.PREDICTIVE
    return FeatureRole.PREDICTIVE


def load_family_meta_from_registry(path: Optional[Path] = None) -> Dict[str, FamilyMeta]:
    """Load family fallback intents + alpha usability policy from the canonical registry.

    This is the STEP 0 anchor: registry is authoritative; DEFAULT_FAMILY_META is only a fallback.
    """

    reg_path = path or (_repo_root() / "docs" / "family_metadata_registry.json")
    try:
        from src.features.family_metadata import load_family_metadata  # type: ignore
    except Exception:
        return dict(DEFAULT_FAMILY_META)

    registry = load_family_metadata(reg_path if reg_path.exists() else None)
    if not registry:
        return dict(DEFAULT_FAMILY_META)

    def _default_halflife_from_decay(decay_token: str) -> float:
        p = str(decay_token or "").strip().lower()
        if p == "fast":
            return 3.0
        if p == "medium":
            return 10.0
        if p == "slow":
            return 30.0
        if p == "two_phase":
            return 6.0
        return 0.0

    out: Dict[str, FamilyMeta] = {}
    for fam, meta in registry.items():
        intent_obj = getattr(meta, "primary_intent", "mixed")
        intent_token = getattr(intent_obj, "value", intent_obj)

        freq_obj = getattr(meta, "update_frequency", "unknown")
        freq_token = getattr(freq_obj, "value", freq_obj)

        decay_obj = getattr(meta, "decay_profile", "unknown")
        decay_token = getattr(decay_obj, "value", decay_obj)

        # Optional governance fields (new schema). Missing fields are safe defaults.
        risk_scale_ok = bool(getattr(meta, "risk_scale_ok", False))
        gating_ok = bool(getattr(meta, "gating_ok", False))
        veto_ok = bool(getattr(meta, "veto_ok", False))
        requires_pit = bool(getattr(meta, "requires_point_in_time", False))
        latency_obj = getattr(meta, "update_latency_class", "unknown")
        latency_token = getattr(latency_obj, "value", latency_obj)
        half_life = float(getattr(meta, "decay_half_life_days", 0.0) or 0.0)
        if half_life <= 0.0:
            half_life = _default_halflife_from_decay(str(decay_token))
        sparsity_obj = getattr(meta, "expected_sparsity", "unknown")
        sparsity_token = getattr(sparsity_obj, "value", sparsity_obj)

        out[str(fam)] = FamilyMeta(
            primary_intent=_intent_to_feature_role(str(intent_token)),
            update_cadence=str(freq_token),
            decay=str(decay_token),
            allow_direct_alpha=_coerce_allow_policy(getattr(meta, "allow_direct_alpha", "false")),

            risk_scale_ok=risk_scale_ok,
            gating_ok=gating_ok,
            veto_ok=veto_ok,
            requires_point_in_time=requires_pit,
            update_latency_class=str(latency_token),
            decay_half_life_days=float(half_life),
            expected_sparsity=str(sparsity_token),
        )
    return out


def infer_allowed_usages_with_reasons(
    *,
    feature_role: FeatureRole,
    family_meta: FamilyMeta,
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """Infer per-column usage permissions.

    These are intended for the portfolio engine and should be conservative.
    """

    usages: Dict[str, bool] = {
        "risk_scale_ok": False,
        "gating_ok": False,
        "veto_ok": False,
    }
    reasons: Dict[str, str] = {
        "risk_scale_ok": "U0:default",
        "gating_ok": "U0:default",
        "veto_ok": "U0:default",
    }

    # Risk scaling is primarily for risk features; allow family-level opt-in.
    if feature_role == FeatureRole.RISK and bool(getattr(family_meta, "risk_scale_ok", False)):
        usages["risk_scale_ok"] = True
        reasons["risk_scale_ok"] = "U1:role=risk,fam_ok"
    elif feature_role == FeatureRole.RISK:
        reasons["risk_scale_ok"] = "U1:role=risk,fam_blocked"
    else:
        reasons["risk_scale_ok"] = f"U1:role={feature_role.value}"

    # Gating/veto should generally be non-alpha channels.
    if feature_role in {FeatureRole.HYGIENE, FeatureRole.REGIME, FeatureRole.RISK} and bool(
        getattr(family_meta, "gating_ok", False)
    ):
        usages["gating_ok"] = True
        reasons["gating_ok"] = f"U2:role={feature_role.value},fam_ok"
    else:
        reasons["gating_ok"] = f"U2:role={feature_role.value},fam_blocked"

    if feature_role in {FeatureRole.HYGIENE, FeatureRole.REGIME, FeatureRole.RISK} and bool(
        getattr(family_meta, "veto_ok", False)
    ):
        usages["veto_ok"] = True
        reasons["veto_ok"] = f"U3:role={feature_role.value},fam_ok"
    else:
        reasons["veto_ok"] = f"U3:role={feature_role.value},fam_blocked"

    return usages, reasons


def assign_roles_and_governance_with_reasons(
    df: pd.DataFrame,
    *,
    family_meta: Mapping[str, FamilyMeta],
    requested_families: Optional[Sequence[str]],
    overrides: Mapping[str, FeatureRole],
) -> Tuple[
    Dict[str, FeatureRole],
    Dict[str, str],
    Dict[str, str],
    Dict[str, bool],
    Dict[str, str],
    Dict[str, bool],
    Dict[str, bool],
    Dict[str, bool],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, bool],
    Dict[str, str],
    Dict[str, float],
    Dict[str, str],
]:
    """Return roles + column-level governance maps.

    Includes the legacy `executable` map (alpha_ok) and new allowed-usage fields.
    """

    roles, fams, role_reasons, executable_map, executable_reasons = assign_roles_and_executability_with_reasons(
        df,
        family_meta=family_meta,
        requested_families=requested_families,
        overrides=overrides,
    )

    risk_scale_ok_map: Dict[str, bool] = {}
    gating_ok_map: Dict[str, bool] = {}
    veto_ok_map: Dict[str, bool] = {}

    risk_scale_reason_map: Dict[str, str] = {}
    gating_reason_map: Dict[str, str] = {}
    veto_reason_map: Dict[str, str] = {}

    requires_pit_map: Dict[str, bool] = {}
    latency_map: Dict[str, str] = {}
    half_life_map: Dict[str, float] = {}
    sparsity_map: Dict[str, str] = {}

    for col, role in roles.items():
        fam = str(fams.get(col, ""))
        meta = family_meta.get(fam, FamilyMeta(FeatureRole.PREDICTIVE))
        usages, usage_reasons = infer_allowed_usages_with_reasons(feature_role=role, family_meta=meta)
        risk_scale_ok_map[col] = bool(usages.get("risk_scale_ok", False))
        gating_ok_map[col] = bool(usages.get("gating_ok", False))
        veto_ok_map[col] = bool(usages.get("veto_ok", False))
        risk_scale_reason_map[col] = str(usage_reasons.get("risk_scale_ok", ""))
        gating_reason_map[col] = str(usage_reasons.get("gating_ok", ""))
        veto_reason_map[col] = str(usage_reasons.get("veto_ok", ""))

        requires_pit_map[col] = bool(getattr(meta, "requires_point_in_time", False))
        latency_map[col] = str(getattr(meta, "update_latency_class", "unknown") or "unknown")
        try:
            half_life_map[col] = float(getattr(meta, "decay_half_life_days", 0.0) or 0.0)
        except Exception:
            half_life_map[col] = 0.0
        sparsity_map[col] = str(getattr(meta, "expected_sparsity", "unknown") or "unknown")

    return (
        roles,
        fams,
        role_reasons,
        executable_map,
        executable_reasons,
        risk_scale_ok_map,
        gating_ok_map,
        veto_ok_map,
        risk_scale_reason_map,
        gating_reason_map,
        veto_reason_map,
        requires_pit_map,
        latency_map,
        half_life_map,
        sparsity_map,
    )


# -----------------------------------------------------------------------------
# Feature-level ordered rules (hard, deterministic).
# -----------------------------------------------------------------------------
HYGIENE_TOKENS = (
    "has_data",
    "coverage",
    "is_valid",
    "is_active",
    "activity",
    "eligible",
    "stale",
    "delist",
    "listing_status",
    "days_since_update",
    "days_since_event",
    "missing",
    "availability",
)

REGIME_TOKENS = (
    "regime",
    "state",
    "prob",
    "probability",
    "risk_on",
    "risk_off",
    "trend_flag",
    "event_window",
    "pre_event",
    "post_event",
    "calendar",
    "is_fomc",
    "is_cpi",
    "is_earnings_week",
    "seasonality",
    "month_end",
    "quarter_end",
)

RISK_TOKENS = (
    "vol",
    "volatility",
    "variance",
    "sigma",
    "std",
    "dispersion",
    "correlation",
    "corr",
    "beta",
    "drawdown",
    "liquidity",
    "illiquidity",
    "spread",
    "skew_abs",
    "kurtosis",
    "tail",
    "var_",
    "cvar",
    "iv_level",
    "iv_rank",
    "gamma_exposure",
)

PREDICTIVE_TOKENS = (
    "return",
    "ret_",
    "alpha",
    "residual",
    "spread_z",
    "momentum",
    "mean_reversion",
    "sentiment",
    "surprise",
    "score",
    "zscore",
    "forecast",
    "expected",
    "edge",
    "mispricing",
    "carry",
    "relative",
)


# ----------------------------------------------------------------------------
# Alternative-signals authoritative overrides (Jan 2026 policy)
# ----------------------------------------------------------------------------
_ALT_SIGNALS_PREFIX = "alternative_signals_"

_ALT_SIGNALS_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
}

_ALT_SIGNALS_RISK_SUFFIXES = {
    # Beta / correlation
    "beta_20d",
    "beta_change_rate",
    "beta_vix_interaction",
    "spy_correlation_20d",
    "qqq_correlation_20d",
    "sector_etf_correlation_20d",
    # Volatility
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_ratio_5_20",
    "rv_z_20",
    "close_to_close_volatility",
    "open_to_close_volatility",
    "high_low_volatility_ratio",
    "intraday_volatility_ratio",
    # Liquidity / cost
    "liquidity_stress_pct",
}

_ALT_SIGNALS_REGIME_SUFFIXES = {
    # Earnings regime / calendar
    "days_since_last_earnings",
    "days_to_next_earnings",
    "turn_of_month_flag",
    # Attention levels (policy)
    "news_volume_count",
    "google_trends_score",
}

_ALT_SIGNALS_PREDICTIVE_SUFFIXES = {
    # Intraday / overnight
    "intraday_range_pct",
    "intraday_range_z",
    "opening_reversal",
    "closing_ramp",
    "overnight_return",
    "overnight_return_z",
    "gap_up_pct",
    "gap_down_pct",
    "gap_vs_vix_interaction",
    # Volume
    "relative_volume_20d",
    "volume_z_20d",
    "volume_trend_10d",
    "opening_volume_surge",
    "buy_volume_proxy",
    "volume_price_divergence",
    # Earnings drift
    "earnings_runup_10d",
    "post_earnings_drift_5d",
    # News / sentiment shocks
    "news_volume_change",
    "news_volume_z",
    # Derived signals
    "trend_acceleration",
    "mean_reversion_signal",
}

# ----------------------------------------------------------------------------
# ARIMA forecast authoritative overrides (Jan 2026 policy)
# ----------------------------------------------------------------------------
_ARIMA_PREFIX = "arima_forecast_"

_ARIMA_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
    "arima_log_likelihood",
    "confidence",
}

_ARIMA_RISK_SUFFIXES = {
    "arima_abs_residual",
    "arima_uncertainty_proxy",
    "arima_innovation",
}

_ARIMA_REGIME_SUFFIXES = {
    "arima_persistence",
}

_ARIMA_PREDICTIVE_SUFFIXES = {
    "arima_forecast_1d",
    "arima_forecast_5d",
    "arima_residual_zscore",
    "arima_residual_t",
    "arima_momentum_indicator",
}

# ----------------------------------------------------------------------------
# Quantile forecast + calibration + online learning overrides (Jan 2026 policy)
# ----------------------------------------------------------------------------
_QUANTILE_PREFIX = "quantile_forecast_"
_CALIB_PREFIX = "calibration_"
_ONLINE_PREFIX = "online_learning_"
_CANDLE_PREFIX = "candle_mechanics_"

# ----------------------------------------------------------------------------
# CBOE term structure authoritative overrides (Jan 2026 policy)
# All cboe_term columns route to portfolio parquet (RISK/REGIME), NOT Mamba.
# This family provides market stress context for overlays and policy state.
# ----------------------------------------------------------------------------
_CBOE_TERM_PREFIX = "cboe_term_"

_CBOE_TERM_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
}

_CBOE_TERM_RISK_SUFFIXES = {
    # Term slopes (z-scored)
    "vxst_vix_term_slope",
    "vix_vxv_term_slope",
    "vix_vxmt_term_slope",
    # Advanced normalized features
    "normalized_term_slope",
    "vix_term_curvature",
    "front_back_spread",
    "panic_premium",
    # Continuous regime indicators (risk-adjacent)
    "vix_roll_yield",
    "vix_ratio_term",
    "vix_contango_strength",
    "vol_risk_premium",
    "vol_risk_premium_pct",
    "vol_risk_premium_z",
}

_CBOE_TERM_REGIME_SUFFIXES = {
    # Change/shock features - regime transition indicators
    "vix_term_slope_change_1d",
    "vix_term_slope_change_5d",
    "vix_curvature_change",
    "panic_premium_change",
}

# ----------------------------------------------------------------------------
# Corp actions splits authoritative overrides (Jan 2026 policy)
# Critical: days_since=9999 corrupts role-aware averaging. Use bounded recency.
# ----------------------------------------------------------------------------
_CORP_SPLITS_PREFIX = "corp_actions_splits_"

_CORP_SPLITS_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
}

_CORP_SPLITS_REGIME_SUFFIXES = {
    # Event detection (bounded and safe for averaging)
    "flag",
    "post_5d",
    "post_20d",
    "log_ratio",
    "count_5y",
    "recency",  # Bounded [0,1] recency intensity (replaces raw days_since)
}

_CORP_SPLITS_RISK_SUFFIXES = {
    # These affect position sizing via risk_scale
    "ratio",  # Split magnitude can affect microstructure stress
}

# ----------------------------------------------------------------------------
# Correlation family authoritative overrides (Jan 2026 policy)
# Routing: predictive spillovers to Mamba; exposure/regime/instability to portfolio
# ----------------------------------------------------------------------------
_CORRELATION_PREFIX = "correlation_"

_CORRELATION_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
}

_CORRELATION_MAMBA_PREDICTIVE_SUFFIXES = {
    # Spillover / propagation (predictive, goes to Mamba)
    "lag_corr_1_spy",
    "lag_corr_5_spy",
    "lag_corr_1_vxx",
    "lag_corr_2_vxx",
    # VIX lag correlations
    "corr_20_vix_lag1",
    "corr_60_vix_lag1",
    "corr_20_vix_lag2",
    "corr_60_vix_lag2",
    # Serial structure in returns (autocorrelation)
    "acf_ret_1",
    "acf_ret_5",
}

_CORRELATION_MAMBA_OPTIONAL = {
    # Optional context (only if you explicitly want conditional alpha)
    "corr_decoupling_z",
    "corr_spread_20_60_spy",
}

_CORRELATION_PORTFOLIO_RISK_SUFFIXES = {
    # Systematic exposure / hedge effectiveness
    "corr_20_spy",
    "corr_60_spy",
    "corr_20_qqq",
    "corr_20_vxx",
    "corr_20_sector",
    # Correlation volatility
    "corr_20_spy_vol",
    "corr_20_vxx_vol",
}

_CORRELATION_PORTFOLIO_REGIME_SUFFIXES = {
    # Regime shifts / instability
    "corr_decoupling_z",
    "corr_spread_20_60_spy",
    "corr_20_spy_trend",
    "corr_20_qqq_trend",
    "corr_20_vxx_trend",
    "corr_spread_20_60_qqq",
    # Cross-feature regime diagnostics
    "corr_return_vol_20",
    "corr_return_range_10",
    "corr_vol_volatility_20",
    # Vol clustering
    "acf_absret_1",
    "acf_vol_1",
}

# ----------------------------------------------------------------------------
# Cross-asset family authoritative overrides (Jan 2026 policy)
# Routing: lead/lag predictors to Mamba; everything else to portfolio
# CRITICAL: regime-change fields are ABSOLUTE MAGNITUDE for stress aggregation
# ----------------------------------------------------------------------------
_CROSS_ASSET_PREFIX = "cross_asset_"

_CROSS_ASSET_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
}

_CROSS_ASSET_MAMBA_PREDICTIVE_SUFFIXES = {
    # Lead/lag analysis: per-asset timing/leadership patterns (goes to Mamba)
    "spy_leads_stock_5d",
    "stock_leads_spy_5d",
}

_CROSS_ASSET_RISK_SUFFIXES = {
    # ETF correlations - systematic exposure
    "spy_corr_20d",
    "qqq_corr_20d",
    "sector_etf_corr_20d",
    # Beta exposure
    "beta_20d",
    "beta_volatility_20d",
    # Volatility relationships
    "vix_corr_20d",
    "vix_spread_indicator",
    "realized_vol_vs_spy_corr",
    # Rate sensitivity (levels)
    "tnx_corr_20d",
    "irx_corr_20d",
    # Credit/FX
    "credit_spread_level",
    "asset_corr_hyg_60",
    "asset_corr_uup_60",
    # One-sided stress for overlays
    "risk_offness",
}

_CROSS_ASSET_REGIME_SUFFIXES = {
    # Regime-break detectors (ABSOLUTE MAGNITUDE - non-negative)
    "beta_change_rate",         # abs(zscore) of beta change
    "beta_volatility_change",   # abs(pct_change) of beta stability
    "beta_sign_flip_flag",      # Binary regime shift
    "tnx_corr_change_5d",       # abs(change) in rate correlation
    "irx_corr_change_5d",       # abs(change) in rate correlation
    "cross_asset_coupling_change",  # abs(change) in coupling factor
    # Composite (directional, for policy state only)
    "risk_onoff_factor",
}

# ----------------------------------------------------------------------------
# DCF family authoritative overrides (Jan 2026 policy)
# Routing: momentum/z-score predictors to Mamba; anchors/stress to portfolio
# CRITICAL: overextension and downside_skew_stress are one-sided for overlays
# ----------------------------------------------------------------------------
_DCF_PREFIX = "dcf_"

_DCF_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
}

_DCF_MAMBA_PREDICTIVE_SUFFIXES = {
    # Momentum (alpha signals)
    "mom_1m",
    "mom_3m",
    "mom_12m",
    "mom_vol_adjusted",
    "mom_sharped",
    # Mean-reversion z-scores
    "zscore_1m",
    "zscore_3m",
    # Value-momentum mix
    "value_momentum_ratio",
    # Directional alpha signals (NOT stress)
    "undervaluation",          # One-sided: more negative = more undervalued (opportunity)
    "scenario_skew",           # Directional: +/- indicates upside/downside asymmetry
}

_DCF_RISK_SUFFIXES = {
    # Volatility-adjusted value
    "vol_adjusted_value",
    # Scenario uncertainty
    "scenario_spread",         # Width of valuation uncertainty
    # One-sided stress for overlays
    "overextension",           # clip(log_p2fv_1y, 0, 1): high = overextended
    "downside_skew_stress",    # clip(-scenario_skew, 0, 1): high = downside risk
}

_DCF_REGIME_SUFFIXES = {
    # Anchor ratios (raw and log)
    "price_to_fairvalue_1y",
    "price_to_fairvalue_3m",
    "price_regime",
    "log_p2fv_1y",
    "log_p2fv_3m",
    "log_price_regime",
    # Trend quality
    "trend_slope_1m",
    "trend_stability",
    # Scenario positioning
    "scenario_position",
    # Horizon structure
    "terminal_value_pct",
}

_QUANTILE_HYGIENE_SUFFIXES = {
    "has_data",
    "activity",
    "days_since_update",
    "hf_conf",
}

_QUANTILE_PREDICTIVE_SUFFIXES = {
    "q50",
    "q_median_50",
    "q_skewness_proxy",
    "q_tilt_direction",
    "skew",
    "hf_score",
}

_QUANTILE_RISK_SUFFIXES = {
    "q_spread_95_5",
    "q_vol_forecast",
    "width",
    "uncertainty",
}


def _quantile_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    name = str(column or "")
    if not name.lower().startswith(_QUANTILE_PREFIX):
        return None
    suffix = name[len(_QUANTILE_PREFIX):].lower()
    if suffix in _QUANTILE_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "QF:governance"
    if suffix in _QUANTILE_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "QF:predictive"
    if suffix in _QUANTILE_RISK_SUFFIXES:
        return FeatureRole.RISK, "QF:risk"
    # Default: keep quantile outputs as policy/regime
    return FeatureRole.REGIME, "QF:policy"


def _calibration_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route calibration columns: freshness→HYGIENE, quality→RISK, status→REGIME.
    
    Critical: RoleAwareContext only auto-gates on has_data/days_since_update for HYGIENE.
    Quality metrics (overall_score, calibration_error, etc.) must be RISK to affect
    risk_scale = 1/(1+risk_agg). Keeping them as HYGIENE does nothing useful.
    """
    name = str(column or "")
    if not name.lower().startswith(_CALIB_PREFIX):
        return None
    suffix = name[len(_CALIB_PREFIX):].lower()
    
    # HYGIENE: freshness/availability gates (veto on NaN or stale)
    if suffix in {"has_data", "activity", "days_since_update"}:
        return FeatureRole.HYGIENE, "CAL:governance"
    
    # RISK: quality metrics that should shrink exposure when degraded
    # These affect risk_scale = 1/(1+risk_agg) in RoleAwareContext
    if suffix in {
        "mean_calibration_error",
        "overall_score",
        "interval_error",
        "quantile_error",
        "sharpness",
        "reliability",
        "calibration_slope",
        "calibration_intercept",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "maximum_calibration_error",
    }:
        return FeatureRole.RISK, "CAL:quality_risk"
    
    # REGIME: binary/categorical status flags that modulate regime_multiplier
    if suffix in {
        "requires_recalibration",
        "recalibration_flag",
        "model_stale",
        "quality_regime",
    }:
        return FeatureRole.REGIME, "CAL:status_regime"
    
    # Default: treat unknown calibration columns as RISK (conservative)
    return FeatureRole.RISK, "CAL:risk_default"


def _online_learning_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route online_learning columns: freshness→HYGIENE, trust/accuracy→RISK, drift→REGIME.
    
    Critical: These are meta-performance signals. They should NEVER go to Mamba.
    Trust/accuracy metrics affect risk_scale; drift flags affect regime_multiplier.
    """
    name = str(column or "")
    if not name.lower().startswith(_ONLINE_PREFIX):
        return None
    suffix = name[len(_ONLINE_PREFIX):].lower()
    
    # HYGIENE: freshness/availability gates
    if suffix in {"has_data", "activity", "days_since_update"}:
        return FeatureRole.HYGIENE, "OL:governance"
    
    # RISK: trust/accuracy/quality metrics - shrink exposure when model is uncertain
    if suffix in {
        "trust_score",
        "model_confidence",
        "direction_accuracy",
        "sign_accuracy",
        "hit_rate",
        "accuracy",
        "quantile_accuracy",
        "mse",
        "mae",
        "sharpe_estimate",
        "information_ratio",
        "uncertainty_estimate",
        "prediction_variance",
    }:
        return FeatureRole.RISK, "OL:trust_risk"
    
    # REGIME: drift/retrain flags - modulate regime_multiplier
    if suffix in {
        "drift_flag",
        "partial_retrain_flag",
        "full_retrain_flag",
        "uncertainty_compression_alert",
        "regime_shift_detected",
        "distribution_drift",
        "concept_drift",
        "covariate_drift",
    }:
        return FeatureRole.REGIME, "OL:drift_regime"
    
    # REGIME: regime probabilities (continuous modulation)
    if suffix in {
        "regime_prob_bull",
        "regime_prob_bear",
        "regime_prob_neutral",
        "regime_probability",
    }:
        return FeatureRole.REGIME, "OL:regime_prob"
    
    # Default: treat unknown online_learning columns as RISK (conservative)
    return FeatureRole.RISK, "OL:risk_default"


def _cboe_term_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route cboe_term columns: ALL to portfolio parquet (RISK/REGIME), NONE to Mamba.
    
    Critical: cboe_term provides market stress context for overlays and policy state.
    It should NOT be used as direct alpha in Mamba - tends to become a global risk proxy
    that the model misuses. Keep it purely for:
    - risk_scale modulation (RISK columns)
    - regime_multiplier modulation (REGIME columns)
    - policy controller state features
    """
    name = str(column or "")
    # Support both prefixed and unprefixed cboe_term column names
    name_lower = name.lower()
    
    # Handle prefixed columns
    if name_lower.startswith(_CBOE_TERM_PREFIX):
        suffix = name_lower[len(_CBOE_TERM_PREFIX):]
    elif name_lower.startswith("cboe_term_"):
        suffix = name_lower[len("cboe_term_"):]
    else:
        # Also handle raw cboe column names (from cboe_term.py output)
        known_cboe_cols = (
            _CBOE_TERM_HYGIENE_SUFFIXES
            | _CBOE_TERM_RISK_SUFFIXES
            | _CBOE_TERM_REGIME_SUFFIXES
        )
        if name_lower in known_cboe_cols:
            suffix = name_lower
        else:
            return None
    
    # HYGIENE: freshness/availability gates
    if suffix in _CBOE_TERM_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "CBOE:governance"
    
    # RISK: term structure levels, slopes, ratios - affect risk_scale
    if suffix in _CBOE_TERM_RISK_SUFFIXES:
        return FeatureRole.RISK, "CBOE:risk"
    
    # REGIME: change/shock features - affect regime_multiplier
    if suffix in _CBOE_TERM_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "CBOE:regime"
    
    # Default: unknown cboe_term columns go to RISK (conservative, not Mamba)
    return FeatureRole.RISK, "CBOE:risk_default"


def _corp_actions_splits_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route corp_actions_splits columns with proper bounded recency.
    
    Critical: The raw days_since=9999 value corrupts role-aware regime averaging.
    The feature builder should transform to bounded recency in [0,1]:
        recency = exp(-min(days_since, 252)/20)
    
    Routing:
    - HYGIENE: has_data, activity, days_since_update (freshness gating)
    - REGIME: flag, post_5d, post_20d, log_ratio, count_5y, recency (regime modulation)
    - RISK: ratio (microstructure stress affects position sizing)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_CORP_SPLITS_PREFIX):
        return None
    
    suffix = name_lower[len(_CORP_SPLITS_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _CORP_SPLITS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "SPLITS:governance"
    
    # RISK: split magnitude affects microstructure stress
    if suffix in _CORP_SPLITS_RISK_SUFFIXES:
        return FeatureRole.RISK, "SPLITS:risk"
    
    # REGIME: event detection and post-event windows
    if suffix in _CORP_SPLITS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "SPLITS:regime"
    
    # Handle the problematic days_since column
    # This should be transformed to recency, but if raw, treat as REGIME
    if suffix == "days_since":
        # WARNING: Raw days_since=9999 will corrupt averaging!
        # Feature builder should transform to bounded recency.
        return FeatureRole.REGIME, "SPLITS:days_since_raw_WARNING"
    
    # Default: unknown splits columns go to REGIME (conservative)
    return FeatureRole.REGIME, "SPLITS:regime_default"


def _correlation_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route correlation columns: predictive spillovers to Mamba, exposure/regime to portfolio.
    
    Routing strategy:
    - Mamba (PREDICTIVE): lag correlations, VIX lag, return autocorrelations
    - Portfolio (RISK): systematic exposure, correlation levels
    - Portfolio (REGIME): decoupling, trends, cross-feature diagnostics, vol clustering
    """
    name = str(column or "")
    name_lower = name.lower()
    
    # Handle both prefixed and unprefixed column names
    if name_lower.startswith(_CORRELATION_PREFIX):
        suffix = name_lower[len(_CORRELATION_PREFIX):]
    elif name_lower.startswith("corr_") or name_lower.startswith("acf_") or name_lower.startswith("lag_corr_"):
        suffix = name_lower
    else:
        return None
    
    # HYGIENE: freshness/availability gates
    if suffix in _CORRELATION_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "CORR:governance"
    
    # Check for Mamba predictive spillovers (these go to Mamba parquet)
    if suffix in _CORRELATION_MAMBA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "CORR:mamba_spillover"
    
    # Check for Mamba optional context
    # Note: these are also in REGIME, but if explicitly enabled, mark as PREDICTIVE
    # Environment variable CORRELATION_MAMBA_OPTIONAL can enable these
    import os
    corr_opt_raw = os.getenv("CORRELATION_MAMBA_OPTIONAL", "").strip().lower()
    corr_opt_tokens = {t.strip() for t in corr_opt_raw.split(",") if t.strip()}
    if suffix in _CORRELATION_MAMBA_OPTIONAL and suffix in corr_opt_tokens:
        return FeatureRole.PREDICTIVE, "CORR:mamba_optional"
    
    # RISK: systematic exposure, correlation levels
    if suffix in _CORRELATION_PORTFOLIO_RISK_SUFFIXES:
        return FeatureRole.RISK, "CORR:risk_exposure"
    
    # REGIME: regime shifts, instability, cross-feature diagnostics
    if suffix in _CORRELATION_PORTFOLIO_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "CORR:regime_stability"
    
    # Default: treat unknown correlation columns as RISK (conservative, to portfolio)
    return FeatureRole.RISK, "CORR:risk_default"


def _cross_asset_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route cross_asset columns: lead/lag predictors to Mamba, everything else to portfolio.
    
    CRITICAL: Regime-change fields are ABSOLUTE MAGNITUDE for proper stress aggregation.
    RoleAwareContext treats large values as "stress"; signed values confuse the aggregator.
    
    Routing:
    - Mamba (PREDICTIVE): spy_leads_stock_5d, stock_leads_spy_5d (per-asset timing alpha)
    - Portfolio (RISK): ETF correlations, beta, vol relationships, credit/FX, risk_offness
    - Portfolio (REGIME): beta_change_rate, beta_volatility_change, rate correlation changes,
      coupling_change, risk_onoff_factor (for policy state)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    # Handle both prefixed and unprefixed column names
    if name_lower.startswith(_CROSS_ASSET_PREFIX):
        suffix = name_lower[len(_CROSS_ASSET_PREFIX):]
    elif name_lower.startswith("cross_asset_"):
        suffix = name_lower[len("cross_asset_"):]
    else:
        # Check for known unprefixed cross_asset column names
        known_cols = (
            _CROSS_ASSET_HYGIENE_SUFFIXES
            | _CROSS_ASSET_MAMBA_PREDICTIVE_SUFFIXES
            | _CROSS_ASSET_RISK_SUFFIXES
            | _CROSS_ASSET_REGIME_SUFFIXES
        )
        if name_lower in known_cols:
            suffix = name_lower
        else:
            return None
    
    # HYGIENE: freshness/availability gates
    if suffix in _CROSS_ASSET_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "XASSET:governance"
    
    # Mamba predictive: lead/lag analysis (per-asset timing alpha)
    if suffix in _CROSS_ASSET_MAMBA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "XASSET:mamba_leadlag"
    
    # RISK: systematic exposure, one-sided stress signals
    if suffix in _CROSS_ASSET_RISK_SUFFIXES:
        return FeatureRole.RISK, "XASSET:risk"
    
    # REGIME: regime-break detectors (absolute magnitude) and policy state
    if suffix in _CROSS_ASSET_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "XASSET:regime"
    
    # Default: unknown cross_asset columns go to RISK (conservative, to portfolio)
    return FeatureRole.RISK, "XASSET:risk_default"


def _dcf_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route dcf columns: momentum/z-score predictors to Mamba, anchors/stress to portfolio.
    
    CRITICAL: One-sided stress signals for portfolio overlays:
    - overextension: clip(log_p2fv_1y, 0, 1) — only penalize "too expensive"
    - downside_skew_stress: clip(-scenario_skew, 0, 1) — only penalize downside asymmetry
    - undervaluation: clip(log_p2fv_1y, -1, 0) — directional alpha, NOT stress
    
    Routing:
    - Mamba (PREDICTIVE): momentum, z-scores, value_momentum_ratio, undervaluation, scenario_skew
    - Portfolio (RISK): vol_adjusted_value, scenario_spread, overextension, downside_skew_stress
    - Portfolio (REGIME): anchor ratios (raw + log), trend quality, scenario_position, terminal_value_pct
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_DCF_PREFIX):
        return None
    
    suffix = name_lower[len(_DCF_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _DCF_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "DCF:governance"
    
    # Mamba predictive: momentum, z-scores, directional alpha
    if suffix in _DCF_MAMBA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "DCF:mamba_alpha"
    
    # RISK: one-sided stress signals for overlays
    if suffix in _DCF_RISK_SUFFIXES:
        return FeatureRole.RISK, "DCF:risk_stress"
    
    # REGIME: anchors, trend quality, scenario positioning
    if suffix in _DCF_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "DCF:regime"
    
    # Default: unknown dcf columns go to REGIME (conservative, to portfolio)
    return FeatureRole.REGIME, "DCF:regime_default"


# ============================================================================
# Dividends family role overrides (Jan 2026)
# ============================================================================
_DIVIDENDS_PREFIX = "dividends_"

_DIVIDENDS_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

_DIVIDENDS_PREDICTIVE_SUFFIXES = frozenset({
    # Only if labels are dividend-adjusted AND shifted
    "dividend_event_intensity",
})

_DIVIDENDS_RISK_SUFFIXES = frozenset({
    # Yield metrics
    "dividend_yield_est",
    "dividend_yield_zscore",
    # One-sided event stress
    "dividend_event_stress",
})

_DIVIDENDS_REGIME_SUFFIXES = frozenset({
    # Event proximity
    "days_to_ex_dividend",
    "ex_dividend_window_strength",
    # Structural
    "dividend_frequency",
    "dividend_amount",
})


def _dividends_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route dividends columns: event_intensity to Mamba (conditional), rest to portfolio.
    
    CRITICAL: dividend_event_stress is one-sided RISK for overlays:
    - dividend_event_stress = window_strength × clip(|amount/price|, 0, 0.10) × 10 → [0,1]
    
    Routing:
    - Mamba (PREDICTIVE): dividend_event_intensity (only if labels are adjusted + shifted)
    - Portfolio (RISK): yield metrics + dividend_event_stress
    - Portfolio (REGIME): event proximity + structural classifiers
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_DIVIDENDS_PREFIX):
        return None
    
    suffix = name_lower[len(_DIVIDENDS_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _DIVIDENDS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "DIVIDENDS:governance"
    
    # Mamba predictive (conditional on label pipeline)
    if suffix in _DIVIDENDS_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "DIVIDENDS:mamba_conditional"
    
    # RISK: yield metrics + one-sided event stress
    if suffix in _DIVIDENDS_RISK_SUFFIXES:
        return FeatureRole.RISK, "DIVIDENDS:risk"
    
    # REGIME: event proximity + structural
    if suffix in _DIVIDENDS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "DIVIDENDS:regime"
    
    # Default: unknown dividends columns go to REGIME (conservative, to portfolio)
    return FeatureRole.REGIME, "DIVIDENDS:regime_default"


# ============================================================================
# Earnings family role overrides (Jan 2026)
# ============================================================================
_EARNINGS_PREFIX = "earnings_"

_EARNINGS_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

_EARNINGS_MAMBA_PREDICTIVE_SUFFIXES = frozenset({
    # Core alpha signals (SHIFTED to prevent leakage)
    "eps_surprise_pct",
    "revenue_surprise_pct",
    # Growth trends
    "eps_growth_qoq",
    "eps_growth_yoy",
    "revenue_growth_qoq",
    "revenue_growth_yoy",
    # Beat patterns (HIGH ALPHA)
    "beat_streak",
    "beat_rate_3y",
    # Revisions (CRITICAL ALPHA)
    "revision_breadth",
    # Event decay (PEAD capture)
    "event_decay",
})

_EARNINGS_RISK_SUFFIXES = frozenset({
    # Uncertainty / volatility
    "miss_streak",
    "surprise_volatility",
    "estimate_dispersion",
    # One-sided event stress (magnitude-based)
    "pre_event_stress",
    "post_event_stress",
})

_EARNINGS_REGIME_SUFFIXES = frozenset({
    # Event timing (also useful for policy state)
    "days_since_earnings",
    "days_to_next_earnings",
})


def _earnings_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route earnings columns: surprises/growth/revisions to Mamba, stress/dispersion to portfolio.
    
    CRITICAL: Leakage prevention
    - Surprise features (eps_surprise_pct, revenue_surprise_pct) are SHIFTED by 1 day
    - If day t bar includes earnings reaction, surprise is only known after release
    
    CRITICAL: One-sided stress signals for portfolio overlays:
    - pre_event_stress = exp(-days_to_next / tau_pre) — reduces size before uncertainty
    - post_event_stress = exp(-days_since / tau_post) — suppresses post-event over-sizing
    
    Routing:
    - Mamba (PREDICTIVE): surprises, growth, beat patterns, revisions, event_decay
    - Portfolio (RISK): miss_streak, surprise_volatility, estimate_dispersion, event stress
    - Portfolio (REGIME): event timing (days_since, days_to_next)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_EARNINGS_PREFIX):
        return None
    
    suffix = name_lower[len(_EARNINGS_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _EARNINGS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "EARNINGS:governance"
    
    # Mamba predictive: surprises, growth, beat patterns, revisions
    if suffix in _EARNINGS_MAMBA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "EARNINGS:mamba_alpha"
    
    # RISK: uncertainty + one-sided event stress
    if suffix in _EARNINGS_RISK_SUFFIXES:
        return FeatureRole.RISK, "EARNINGS:risk"
    
    # REGIME: event timing
    if suffix in _EARNINGS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "EARNINGS:regime"
    
    # Default: unknown earnings columns go to RISK (conservative, to portfolio)
    return FeatureRole.RISK, "EARNINGS:risk_default"


# ============================================================================
# Econ events calendar family role overrides (Jan 2026)
# ============================================================================
_ECON_EVENTS_PREFIX = "econ_events_calendar_"

_ECON_EVENTS_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

# Per-event HYGIENE (surprise_has_forecast_*)
_ECON_EVENTS_HYGIENE_PATTERNS = frozenset({
    "surprise_has_forecast_",
})

# Mamba compact context (shifted, leak-safe)
_ECON_EVENTS_MAMBA_PREDICTIVE_SUFFIXES = frozenset({
    # Shifted pulse surprises (signed, for directional context)
    "pulse_surprise_cpi",
    "pulse_surprise_fomc",
    "pulse_surprise_nfp",
    "pulse_surprise_gdp",
    "pulse_surprise_pce",
    "pulse_surprise_unemployment",
    "pulse_surprise_retail_sales",
    "pulse_surprise_ism",
    "pulse_surprise_core_cpi",
    # Pulse strength (magnitude only)
    "pulse_strength_cpi",
    "pulse_strength_fomc",
    "pulse_strength_nfp",
    "pulse_strength_gdp",
    "pulse_strength_pce",
    "pulse_strength_unemployment",
    "pulse_strength_retail_sales",
    "pulse_strength_ism",
    "pulse_strength_core_cpi",
    # Bounded proximity (for event imminence)
    "prox_next_cpi",
    "prox_next_fomc",
    "prox_next_nfp",
    "prox_next_gdp",
    "prox_next_pce",
    "prox_next_unemployment",
    "prox_next_retail_sales",
    "prox_next_ism",
    "prox_next_core_cpi",
})

# Portfolio RISK (stress aggregation, magnitude only)
_ECON_EVENTS_RISK_SUFFIXES = frozenset({
    # Composite macro stress for overlays
    "macro_shock_major",
    "macro_upcoming_major",
})

# Portfolio REGIME (regime/policy state)
_ECON_EVENTS_REGIME_SUFFIXES = frozenset({
    # Pre/post windows
    "pre_window_3d_cpi",
    "pre_window_3d_fomc",
    "pre_window_3d_nfp",
    "pre_window_3d_gdp",
    "pre_window_3d_pce",
    "pre_window_3d_unemployment",
    "pre_window_3d_retail_sales",
    "pre_window_3d_ism",
    "pre_window_3d_core_cpi",
    "post_window_3d_cpi",
    "post_window_3d_fomc",
    "post_window_3d_nfp",
    "post_window_3d_gdp",
    "post_window_3d_pce",
    "post_window_3d_unemployment",
    "post_window_3d_retail_sales",
    "post_window_3d_ism",
    "post_window_3d_core_cpi",
    # Pulse occurrence (binary)
    "pulse_occurrence_cpi",
    "pulse_occurrence_fomc",
    "pulse_occurrence_nfp",
    "pulse_occurrence_gdp",
    "pulse_occurrence_pce",
    "pulse_occurrence_unemployment",
    "pulse_occurrence_retail_sales",
    "pulse_occurrence_ism",
    "pulse_occurrence_core_cpi",
    # Event occurrence (binary)
    "event_occurrence_cpi",
    "event_occurrence_fomc",
    "event_occurrence_nfp",
    "event_occurrence_gdp",
    "event_occurrence_pce",
    "event_occurrence_unemployment",
    "event_occurrence_retail_sales",
    "event_occurrence_ism",
    "event_occurrence_core_cpi",
    # Recency (bounded, for regime modulation)
    "recency_last_cpi",
    "recency_last_fomc",
    "recency_last_nfp",
    "recency_last_gdp",
    "recency_last_pce",
    "recency_last_unemployment",
    "recency_last_retail_sales",
    "recency_last_ism",
    "recency_last_core_cpi",
    # Composite signed (directional, for policy state)
    "macro_surprise_signed",
})


def _econ_events_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route econ_events_calendar columns: compact context to Mamba, regime/risk to portfolio.
    
    CRITICAL: 9999 sentinel replacement
    - Raw days_to_next_* and days_since_last_* columns are DEPRECATED
    - Use bounded prox_next_* and recency_last_* instead (safe for aggregation)
    
    Routing:
    - Mamba (PREDICTIVE): pulse_surprise_*, pulse_strength_*, prox_next_* (compact context)
    - Portfolio (RISK): macro_shock_major, macro_upcoming_major (stress aggregation)
    - Portfolio (REGIME): windows, occurrences, recency, macro_surprise_signed
    - DEPRECATED (keep but exclude from Mamba): days_to_next_*, days_since_last_*, surprise_z_*
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_ECON_EVENTS_PREFIX):
        return None
    
    suffix = name_lower[len(_ECON_EVENTS_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _ECON_EVENTS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "ECON:governance"
    
    # HYGIENE: per-event forecast presence
    for pattern in _ECON_EVENTS_HYGIENE_PATTERNS:
        if suffix.startswith(pattern):
            return FeatureRole.HYGIENE, "ECON:forecast_flag"
    
    # Mamba predictive: pulse surprises, pulse strength, bounded proximity
    if suffix in _ECON_EVENTS_MAMBA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "ECON:mamba_context"
    
    # RISK: composite stress signals
    if suffix in _ECON_EVENTS_RISK_SUFFIXES:
        return FeatureRole.RISK, "ECON:risk_stress"
    
    # REGIME: windows, occurrences, recency, composites
    if suffix in _ECON_EVENTS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "ECON:regime"
    
    # Raw days_to_next / days_since_last (DEPRECATED but keep for backward compat)
    if suffix.startswith("days_to_next_") or suffix.startswith("days_since_last_"):
        return FeatureRole.REGIME, "ECON:deprecated_days_sentinel"
    
    # Surprise z-scores (keep in portfolio, not Mamba by default)
    if suffix.startswith("surprise_z_") or suffix.startswith("surprise_w_z_"):
        return FeatureRole.PREDICTIVE, "ECON:surprise_zscore"
    
    # Default: unknown econ_events columns go to REGIME (to portfolio)
    return FeatureRole.REGIME, "ECON:regime_default"


# ============================================================================
# Exchange calendar family role overrides (Jan 2026)
# ============================================================================
_EXCHANGE_CALENDAR_PREFIX = "exchange_calendar_"

_EXCHANGE_CALENDAR_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

_EXCHANGE_CALENDAR_RISK_SUFFIXES = frozenset({
    # Composite liquidity stress for position sizing / turnover control
    "liquidity_stress",
})

_EXCHANGE_CALENDAR_REGIME_SUFFIXES = frozenset({
    # Event flags
    "is_holiday_adjacent",
    "is_half_day",
    # Bounded proximity/recency (replace 9999 sentinels)
    "holiday_prox",
    "holiday_recency",
    # Raw days (DEPRECATED but kept for backward compat)
    "days_to_holiday",
    "days_since_holiday",
})


def _exchange_calendar_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route exchange_calendar columns: ALL to portfolio, NONE to Mamba.
    
    This family provides execution/microstructure regime context. It should influence:
    - Position sizing conservatism (via liquidity_stress → risk_scale)
    - Turnover/participation behavior (via policy controller)
    - Transaction cost assumptions (spreads widen around holidays/half-days)
    
    It should NOT be a primary alpha source for Mamba (teaches model mechanical
    liquidity effects rather than returns).
    
    CRITICAL: 9999 sentinel replacement
    - Raw days_to_holiday / days_since_holiday are DEPRECATED
    - Use holiday_prox / holiday_recency instead (bounded [0,1])
    - Use liquidity_stress composite for execution overlays
    
    Routing:
    - Portfolio (RISK): liquidity_stress (affects position sizing via risk_scale)
    - Portfolio (REGIME): all other columns (event flags, bounded proximity/recency)
    - Mamba: NONE by default (execution context, not alpha)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_EXCHANGE_CALENDAR_PREFIX):
        return None
    
    suffix = name_lower[len(_EXCHANGE_CALENDAR_PREFIX):]
    
    # HYGIENE: freshness/availability gates
    if suffix in _EXCHANGE_CALENDAR_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "XCAL:governance"
    
    # RISK: liquidity stress for position sizing
    if suffix in _EXCHANGE_CALENDAR_RISK_SUFFIXES:
        return FeatureRole.RISK, "XCAL:liquidity_risk"
    
    # REGIME: event flags, bounded proximity/recency
    if suffix in _EXCHANGE_CALENDAR_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "XCAL:regime"
    
    # Default: unknown exchange_calendar columns go to REGIME (conservative)
    return FeatureRole.REGIME, "XCAL:regime_default"


# ============================================================================
# FIN_G1 (Liquidity) family role overrides (Jan 2026)
# ============================================================================
_FIN_G1_PREFIX = "fin_g1_"

_FIN_G1_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G1_RISK_SUFFIXES = frozenset({
    # STRESS features: higher = worse (safe for risk aggregation)
    "liquidity_stress",
    "quick_stress",
    "cash_stress",
    "liquidity_trend_stress",
    # Z-score (can indicate stress when negative)
    "liquidity_zscore_5y",
})

_FIN_G1_REGIME_SUFFIXES = frozenset({
    # Trend direction (regime awareness)
    "liquidity_trend_3y",
})

# Raw ratios: PREDICTIVE but only for long-horizon Mamba (optional)
_FIN_G1_RAW_RATIO_SUFFIXES = frozenset({
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
})


def _fin_g1_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g1 (liquidity) columns: stress features to portfolio, raw ratios optional for Mamba.
    
    CRITICAL: Raw ratios are HIGHER = SAFER, which inverts RoleAwareContext risk aggregation.
    Use *_stress features for portfolio risk overlays (higher = worse).
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): liquidity_stress, quick_stress, cash_stress, liquidity_trend_stress, liquidity_zscore_5y
    - Portfolio (REGIME): liquidity_trend_3y
    - Mamba (PREDICTIVE, optional): current_ratio, quick_ratio, cash_ratio (only if horizon >= 21d)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G1_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G1_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G1_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G1:governance"
    
    # RISK: stress features (already inverted, higher = worse)
    if suffix in _FIN_G1_RISK_SUFFIXES:
        return FeatureRole.RISK, "G1:liquidity_stress"
    
    # REGIME
    if suffix in _FIN_G1_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G1:regime"
    
    # Raw ratios: PREDICTIVE (for long-horizon Mamba, optional)
    # Note: routing decision is in prep_families.py; here we mark intent
    if suffix in _FIN_G1_RAW_RATIO_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G1:raw_ratio_optional"
    
    return FeatureRole.REGIME, "G1:regime_default"


# ============================================================================
# FIN_G2 (Leverage) family role overrides (Jan 2026)
# ============================================================================
_FIN_G2_PREFIX = "fin_g2_"

_FIN_G2_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G2_RISK_SUFFIXES = frozenset({
    # STRESS features: higher = worse (safe for risk aggregation)
    "interest_coverage_stress",
    # Raw leverage ratios (already higher = worse direction)
    "debt_to_equity",
    "debt_to_assets",
    "equity_multiplier",
    "net_debt_to_ebitda",
    "net_debt_to_fcf",
    "interest_burden",
    # Robust transforms
    "debt_to_equity_robust",
    "net_debt_to_ebitda_robust",
    # Z-score (can indicate stress when extreme)
    "leverage_zscore_5y",
})

_FIN_G2_REGIME_SUFFIXES = frozenset({
    # Trend direction (regime awareness)
    "leverage_trend_3y",
})

# Raw coverage: PREDICTIVE but needs inversion for portfolio
_FIN_G2_COVERAGE_SUFFIXES = frozenset({
    "interest_coverage",  # Higher = safer; use interest_coverage_stress for portfolio
})


def _fin_g2_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g2 (leverage) columns: leverage stress to portfolio, coverage inverted.
    
    CRITICAL: interest_coverage is HIGHER = SAFER. Use interest_coverage_stress for portfolio.
    Debt ratios are already HIGHER = WORSE, so they're safe for risk aggregation.
    Use robust transforms (signed_log1p) for exploding ratios.
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): interest_coverage_stress, debt_to_*, leverage_zscore_5y, robust transforms
    - Portfolio (REGIME): leverage_trend_3y
    - Mamba (PREDICTIVE, optional): raw ratios + interest_coverage (only if horizon >= 21d)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G2_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G2_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G2_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G2:governance"
    
    # RISK: leverage stress features
    if suffix in _FIN_G2_RISK_SUFFIXES:
        return FeatureRole.RISK, "G2:leverage_stress"
    
    # REGIME
    if suffix in _FIN_G2_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G2:regime"
    
    # Coverage: PREDICTIVE (but use _stress variant for portfolio risk)
    if suffix in _FIN_G2_COVERAGE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G2:coverage_optional"
    
    # Dollar amounts
    if suffix in {"total_debt", "long_term_debt"}:
        return FeatureRole.REGIME, "G2:debt_levels"
    
    return FeatureRole.REGIME, "G2:regime_default"


# ============================================================================
# FIN_G3 (Efficiency) family role overrides (Jan 2026)
# ============================================================================
_FIN_G3_PREFIX = "fin_g3_"

_FIN_G3_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G3_RISK_SUFFIXES = frozenset({
    # STRESS features: higher = worse (safe for risk aggregation)
    "ccc_stress",
    # Turnover volatility (higher = more instability = worse)
    "turnover_volatility_3y",
})

_FIN_G3_REGIME_SUFFIXES = frozenset({
    # Days outstanding (regime/ops-conditioners)
    "dsos",
    "dios",
    "dpos",
    # Cash conversion cycle (raw)
    "ccc",
})

# Raw turnover ratios: industry-structural, need sector normalization for Mamba
_FIN_G3_TURNOVER_SUFFIXES = frozenset({
    "asset_turnover",
    "inventory_turnover",
    "receivables_turnover",
    "payables_turnover",
})


def _fin_g3_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g3 (efficiency) columns: stress to portfolio, turnover ratios optional for Mamba.
    
    CRITICAL: Efficiency ratios are INDUSTRY-STRUCTURAL. If fed raw, the model may learn
    "industry ID" rather than alpha. Prefer sector-normalized versions or keep out of Mamba.
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): ccc_stress, turnover_volatility_3y
    - Portfolio (REGIME): dsos, dios, dpos, ccc
    - Mamba (PREDICTIVE, optional): sector-normalized turnover ratios only
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G3_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G3_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G3_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G3:governance"
    
    # RISK: stress features
    if suffix in _FIN_G3_RISK_SUFFIXES:
        return FeatureRole.RISK, "G3:efficiency_stress"
    
    # REGIME: days outstanding
    if suffix in _FIN_G3_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G3:regime"
    
    # Raw turnover ratios: PREDICTIVE (but industry-structural, use caution)
    if suffix in _FIN_G3_TURNOVER_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G3:turnover_optional"
    
    return FeatureRole.REGIME, "G3:regime_default"


def _candle_mechanics_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    name = str(column or "")
    if not name.lower().startswith(_CANDLE_PREFIX):
        return None
    suffix = name[len(_CANDLE_PREFIX):].lower()

    # Governance
    if suffix in {"has_data", "activity", "days_since_update", "confidence"}:
        return FeatureRole.HYGIENE, "CM:governance"

    # Policy / risk / regime
    if suffix in {
        "range_atr14",
        "atr_ratio_14_60",
        "rv_5",
        "rv_20",
        "dir_change",
        "inside_bar",
        "outside_bar",
        "ret_1d",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "moy_1",
        "moy_2",
        "moy_3",
        "moy_4",
        "moy_5",
        "moy_6",
        "moy_7",
        "moy_8",
        "moy_9",
        "moy_10",
        "moy_11",
        "moy_12",
    }:
        return FeatureRole.REGIME, "CM:policy"

    # Predictive default
    return FeatureRole.PREDICTIVE, "CM:predictive"


def _alt_signals_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    name = str(column or "")
    if not name.lower().startswith(_ALT_SIGNALS_PREFIX):
        return None
    suffix = name[len(_ALT_SIGNALS_PREFIX):].lower()
    if suffix in _ALT_SIGNALS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "ALT:governance"
    if suffix in _ALT_SIGNALS_RISK_SUFFIXES:
        return FeatureRole.RISK, "ALT:risk"
    if suffix in _ALT_SIGNALS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "ALT:regime"
    if suffix in _ALT_SIGNALS_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "ALT:predictive"
    # Unknown alt-signals columns default to family intent (defer to later rules).
    return None


def _arima_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    name = str(column or "")
    if not name.lower().startswith(_ARIMA_PREFIX):
        return None
    suffix = name[len(_ARIMA_PREFIX):].lower()
    if suffix in _ARIMA_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "ARIMA:governance"
    if suffix in _ARIMA_RISK_SUFFIXES:
        return FeatureRole.RISK, "ARIMA:risk"
    if suffix in _ARIMA_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "ARIMA:regime"
    if suffix in _ARIMA_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "ARIMA:predictive"
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _is_boolish(series: pd.Series) -> bool:
    if series.dtype == bool:
        return True
    s = _safe_numeric(series).dropna()
    if s.empty:
        return False
    # Avoid np.unique on ExtensionArray-backed Series; keep it pandas-native.
    return bool(((s == 0.0) | (s == 1.0)).all())


def _is_probabilistic(series: pd.Series) -> bool:
    s = _safe_numeric(series)
    s = s.dropna()
    if s.empty:
        return False
    mn = float(s.min())
    mx = float(s.max())
    return mn >= -1e-9 and mx <= 1.0 + 1e-9


def _is_strictly_nonneg(series: pd.Series) -> bool:
    s = _safe_numeric(series)
    s = s.dropna()
    if s.empty:
        return False
    return float(s.min()) >= -1e-12


def _family_from_column(col: str, family_names: Sequence[str]) -> str:
    # Prefer longest matching prefix to support families with underscores.
    candidates = [f for f in family_names if col == f or col.startswith(f + "_")]
    if candidates:
        return max(candidates, key=len)
    # Heuristic attribution for a few known families that emit un-prefixed columns.
    # These keep provenance + allow_direct_alpha policies correct.
    if "macro_tst_hf" in set(family_names):
        if col.startswith(("l1_", "l2_", "l3_", "derived_", "derived_interact", "macro_")):
            return "macro_tst_hf"

    if "doc_embedding_novelty_hf" in set(family_names):
        if col in {"n_events", "n_articles", "theme_weight", "baseline_mean", "top_theme_numeric"}:
            return "doc_embedding_novelty_hf"
        if "novelty" in col:
            return "doc_embedding_novelty_hf"
        if col.startswith(("theme_", "baseline_", "top_theme_")):
            return "doc_embedding_novelty_hf"

    return "<unknown>"


def load_overrides(path: Optional[Path]) -> Dict[str, FeatureRole]:
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    out: Dict[str, FeatureRole] = {}
    if isinstance(payload, dict):
        for k, v in cast(Mapping[object, object], payload).items():
            try:
                out[str(k)] = FeatureRole(str(v))
            except Exception:
                continue
    return out


def infer_feature_role(
    *,
    column: str,
    series: pd.Series,
    family_primary_intent: FeatureRole,
    overrides: Mapping[str, FeatureRole],
) -> FeatureRole:
    """Assign feature role by first matching ordered rule."""

    role, _reason = infer_feature_role_with_reason(
        column=column,
        series=series,
        family_primary_intent=family_primary_intent,
        overrides=overrides,
    )
    return role


def infer_feature_role_with_reason(
    *,
    column: str,
    series: pd.Series,
    family_primary_intent: FeatureRole,
    overrides: Mapping[str, FeatureRole],
) -> Tuple[FeatureRole, str]:
    """Like infer_feature_role, but also returns a short rule/reason string.

    Reason strings are intended for provenance/debugging, not as a stable API.
    """

    if column in overrides:
        return overrides[column], "override"

    alt_override = _alt_signals_role_override(column)
    if alt_override is not None:
        return alt_override

    arima_override = _arima_role_override(column)
    if arima_override is not None:
        return arima_override

    quantile_override = _quantile_role_override(column)
    if quantile_override is not None:
        return quantile_override

    calib_override = _calibration_role_override(column)
    if calib_override is not None:
        return calib_override

    online_override = _online_learning_role_override(column)
    if online_override is not None:
        return online_override

    candle_override = _candle_mechanics_role_override(column)
    if candle_override is not None:
        return candle_override

    # CBOE term structure - all columns route to portfolio (RISK/REGIME)
    cboe_override = _cboe_term_role_override(column)
    if cboe_override is not None:
        return cboe_override

    # Corp actions splits - bounded recency, regime/risk overlays
    splits_override = _corp_actions_splits_role_override(column)
    if splits_override is not None:
        return splits_override

    # Correlation - predictive spillovers to Mamba, exposure/regime to portfolio
    corr_override = _correlation_role_override(column)
    if corr_override is not None:
        return corr_override

    # Cross-asset - lead/lag to Mamba, everything else (abs magnitude regime-breaks) to portfolio
    xasset_override = _cross_asset_role_override(column)
    if xasset_override is not None:
        return xasset_override

    # DCF - momentum/z-scores to Mamba, anchors/stress (one-sided) to portfolio
    dcf_override = _dcf_role_override(column)
    if dcf_override is not None:
        return dcf_override

    # Dividends - event_intensity to Mamba (conditional), yield/stress to portfolio
    dividends_override = _dividends_role_override(column)
    if dividends_override is not None:
        return dividends_override

    # Earnings - surprises/growth to Mamba, stress/dispersion to portfolio
    earnings_override = _earnings_role_override(column)
    if earnings_override is not None:
        return earnings_override

    # Econ events calendar - compact context to Mamba, regime/risk to portfolio
    econ_override = _econ_events_role_override(column)
    if econ_override is not None:
        return econ_override

    # Exchange calendar - ALL to portfolio (execution/microstructure context)
    xcal_override = _exchange_calendar_role_override(column)
    if xcal_override is not None:
        return xcal_override

    # FIN_G1 (Liquidity) - stress to portfolio, raw ratios optional for long-horizon Mamba
    fin_g1_override = _fin_g1_role_override(column)
    if fin_g1_override is not None:
        return fin_g1_override

    # FIN_G2 (Leverage) - stress to portfolio, interest_coverage inverted
    fin_g2_override = _fin_g2_role_override(column)
    if fin_g2_override is not None:
        return fin_g2_override

    # FIN_G3 (Efficiency) - stress to portfolio, turnover ratios industry-structural
    fin_g3_override = _fin_g3_role_override(column)
    if fin_g3_override is not None:
        return fin_g3_override

    name = str(column).lower()

    # Explicit governance suffixes (structural, not keyword heuristics)
    if name.endswith("_has_data") or name.endswith("_activity") or name.endswith("_days_since_update") or name.endswith("__days_since_update"):
        return FeatureRole.HYGIENE, "A1:governance_suffix"

    # Fallback to family intent only.
    return family_primary_intent, f"A5:family_intent:{family_primary_intent.value}"


def infer_executable_with_reason(
    *,
    feature_role: FeatureRole,
    family_allow_direct_alpha: str,
) -> Tuple[bool, str]:
    """STEP 3: Enforce allow_direct_alpha at the end.

    We do NOT change the feature role here; we only decide whether a predictive
    feature is allowed to act as standalone/executable alpha.
    """

    if feature_role != FeatureRole.PREDICTIVE:
        return False, "A6:non_predictive_role"

    policy = _coerce_allow_policy(family_allow_direct_alpha)
    if policy == "true":
        return True, "A6:alpha_allowed"
    if policy == "conditional":
        return False, "A6:alpha_conditional_requires_gating"
    return False, "A6:alpha_blocked"


def assign_roles(
    df: pd.DataFrame,
    *,
    family_meta: Mapping[str, FamilyMeta],
    requested_families: Optional[Sequence[str]],
    overrides: Mapping[str, FeatureRole],
) -> Tuple[Dict[str, FeatureRole], Dict[str, str]]:
    """Return per-column role map and per-column family map."""

    if requested_families is None:
        # Best-effort: infer candidate families from the first token of each column.
        inferred: list[str] = []
        seen: set[str] = set()
        for c in df.columns:
            c_s = str(c)
            if c_s == "date":
                continue
            fam = c_s.split("_", 1)[0] if "_" in c_s else c_s
            if fam not in seen:
                inferred.append(fam)
                seen.add(fam)
        family_names = inferred
    else:
        family_names = [str(f) for f in requested_families]
    roles: Dict[str, FeatureRole] = {}
    fam_map: Dict[str, str] = {}

    for col in df.columns:
        col_s = str(col)
        if col_s == "date":
            continue
        family = _family_from_column(col_s, family_names)
        fam_map[col_s] = family
        meta = family_meta.get(str(family), FamilyMeta(FeatureRole.PREDICTIVE))
        series = df[col_s] if col_s in df.columns else pd.Series([], dtype=float)
        roles[col_s] = infer_feature_role(
            column=col_s,
            series=series,
            family_primary_intent=meta.primary_intent,
            overrides=overrides,
        )

    return roles, fam_map


def assign_roles_with_reasons(
    df: pd.DataFrame,
    *,
    family_meta: Mapping[str, FamilyMeta],
    requested_families: Optional[Sequence[str]],
    overrides: Mapping[str, FeatureRole],
) -> Tuple[Dict[str, FeatureRole], Dict[str, str], Dict[str, str]]:
    """Return (roles, family_map, reason_map) for provenance/debugging."""

    roles, fams = assign_roles(
        df,
        family_meta=family_meta,
        requested_families=requested_families,
        overrides=overrides,
    )

    reasons: Dict[str, str] = {}
    for col in roles:
        family = fams.get(col, "")
        meta = family_meta.get(str(family), FamilyMeta(FeatureRole.PREDICTIVE))
        series = df[col] if col in df.columns else pd.Series([], dtype=float)
        _role2, reason = infer_feature_role_with_reason(
            column=col,
            series=series,
            family_primary_intent=meta.primary_intent,
            overrides=overrides,
        )
        reasons[col] = reason
    return roles, fams, reasons


def assign_roles_and_executability_with_reasons(
    df: pd.DataFrame,
    *,
    family_meta: Mapping[str, FamilyMeta],
    requested_families: Optional[Sequence[str]],
    overrides: Mapping[str, FeatureRole],
) -> Tuple[Dict[str, FeatureRole], Dict[str, str], Dict[str, str], Dict[str, bool], Dict[str, str]]:
    """Return (roles, family_map, role_reason_map, executable_map, executable_reason_map)."""

    roles, fams, role_reasons = assign_roles_with_reasons(
        df,
        family_meta=family_meta,
        requested_families=requested_families,
        overrides=overrides,
    )

    executable_map: Dict[str, bool] = {}
    executable_reasons: Dict[str, str] = {}
    for col, role in roles.items():
        family = fams.get(col, "")
        meta = family_meta.get(str(family), FamilyMeta(FeatureRole.PREDICTIVE))
        is_exec, exec_reason = infer_executable_with_reason(
            feature_role=role,
            family_allow_direct_alpha=getattr(meta, "allow_direct_alpha", "false"),
        )
        executable_map[col] = bool(is_exec)
        executable_reasons[col] = exec_reason

    return roles, fams, role_reasons, executable_map, executable_reasons


def normalize_by_role(
    df: pd.DataFrame,
    *,
    roles: Mapping[str, FeatureRole],
    pred_window: int = 252,
    risk_window: int = 252,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Apply role-specific transforms in-place on a copy and return it."""

    out = df.copy()
    eps = 1e-6

    for col, role in roles.items():
        if col not in out.columns:
            continue
        if col == "date":
            continue

        if role == FeatureRole.HYGIENE:
            continue

        s = _safe_numeric(out[col])
        if s.isna().all():
            out[col] = 0.0
            continue

        if role == FeatureRole.REGIME:
            if _is_boolish(out[col]):
                out[col] = _safe_numeric(out[col]).fillna(0.0).clip(0.0, 1.0)
                continue
            # Keep in [0,1] if already probabilistic; otherwise min-max into [0,1].
            s2 = _safe_numeric(out[col])
            s2 = s2.replace([np.inf, -np.inf], np.nan)
            if _is_probabilistic(s2):
                out[col] = s2.fillna(0.0).clip(0.0, 1.0)
            else:
                valid = s2.dropna()
                if valid.empty:
                    out[col] = 0.0
                else:
                    mn = float(valid.min())
                    mx = float(valid.max())
                    if abs(mx - mn) < eps:
                        out[col] = 0.0
                    else:
                        out[col] = ((s2 - mn) / (mx - mn)).fillna(0.0).clip(0.0, 1.0)
            continue

        if role == FeatureRole.RISK:
            name = str(col).lower()
            s2 = s
            # bounded correlation/beta -> map to [0,1]
            if any(tok in name for tok in ("corr", "correlation", "beta")):
                s2 = s2.clip(-1.0, 1.0)
                out[col] = ((s2 + 1.0) / 2.0).fillna(0.0).clip(0.0, 1.0)
                continue

            if _is_strictly_nonneg(s2):
                # Keep this explicitly as a pandas Series (np.log1p() is typed as ndarray).
                base = s2.clip(lower=0.0).to_numpy(dtype=float, na_value=np.nan)
                t = pd.Series(np.log1p(base), index=s2.index)
            else:
                # Risk magnitudes should generally be non-negative.
                t = s2.abs()

            # Percentile scaling: map rolling distribution into [0,1] using
            # robust quantiles (approx percentile). This is more stable than
            # min-max when outliers exist.
            q_lo = float(os.getenv("PREP_FAMILIES_ROLE_RISK_Q_LO", "0.05"))
            q_hi = float(os.getenv("PREP_FAMILIES_ROLE_RISK_Q_HI", "0.95"))
            q_lo = min(max(q_lo, 0.0), 0.49)
            q_hi = max(min(q_hi, 1.0), 0.51)

            roll = t.rolling(int(risk_window), min_periods=int(min_periods))
            lo = roll.quantile(q_lo)
            hi = roll.quantile(q_hi)
            denom = (hi - lo).abs() + eps
            scaled = pd.Series((t - lo) / denom, index=t.index)
            out[col] = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
            continue

        if role == FeatureRole.PREDICTIVE:
            roll = s.rolling(int(pred_window), min_periods=int(min_periods))
            mu = roll.mean()
            sd = roll.std(ddof=0)
            z = (s - mu) / (sd + eps)
            out[col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)
            continue

    return out


def apply_timing_rules(
    df: pd.DataFrame,
    *,
    roles: Mapping[str, FeatureRole],
    family_map: Mapping[str, str],
    family_meta: Mapping[str, FamilyMeta],
    shift_days: int = 1,
    enable_ffill: bool = True,
    enable_shift: bool = True,
    enable_decay: bool = True,
) -> pd.DataFrame:
    """STEP 2: Apply timing rules (ffill/shift/decay) before role normalization.

    Notes:
    - This function must keep the dataframe numeric-only (no metadata strings).
    - allow_direct_alpha is NOT used here (tag-only policy).
    - All operations are causal (use past only).
    """

    out = df.copy()
    if "date" in out.columns:
        try:
            out = out.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    def _decay_halflife(profile: str) -> float:
        p = str(profile or "").strip().lower()
        if p == "fast":
            return 3.0
        if p == "medium":
            return 10.0
        if p == "slow":
            return 30.0
        if p == "two_phase":
            return 6.0
        return 0.0

    def _is_gate_or_conf_column(column: str) -> bool:
        """Return True for stable-schema gate/confidence scalars.

        These columns should generally be forward-filled but NOT impulse-differenced
        or aggressively shifted/decayed, since that can collapse them to zeros
        (e.g. constant confidence=1 -> diff() == 0).
        """

        name = str(column or "").strip().lower()
        if not name:
            return False
        if "has_data" in name or "days_since_update" in name:
            return True
        # Be careful: avoid matching tokens like 'conflict'. Only match suffix/pattern.
        if name.endswith("_conf") or name.endswith("_confidence"):
            return True
        if name.endswith("_conf_raw") or name.endswith("_confidence_raw"):
            return True
        return False

    for col, role in roles.items():
        if col == "date" or col not in out.columns:
            continue

        fam = str(family_map.get(col, ""))
        meta = family_meta.get(fam, FamilyMeta(primary_intent=FeatureRole.PREDICTIVE))
        cadence = str(getattr(meta, "update_cadence", "unknown") or "unknown").lower()
        decay_profile = str(getattr(meta, "decay", "unknown") or "unknown").lower()
        requires_pit = bool(getattr(meta, "requires_point_in_time", False))
        latency = str(getattr(meta, "update_latency_class", "unknown") or "unknown").lower()
        half_life_override = 0.0
        try:
            half_life_override = float(getattr(meta, "decay_half_life_days", 0.0) or 0.0)
        except Exception:
            half_life_override = 0.0

        gate_or_conf = _is_gate_or_conf_column(col)

        s = _safe_numeric(out[col])

        # Forward-fill: primarily for stateful / infrequent updates.
        if enable_ffill:
            should_ffill = (
                role in {FeatureRole.HYGIENE, FeatureRole.REGIME}
                or cadence in {"weekly", "monthly", "quarterly", "irregular", "event", "snapshot"}
            )
            if should_ffill:
                s = s.ffill()

        # Shift: avoid same-session leakage for event/irregular style signals.
        if enable_shift and shift_days and int(shift_days) > 0:
            should_shift = (
                role != FeatureRole.HYGIENE
                and (
                    cadence in {"weekly", "monthly", "quarterly", "irregular", "event", "snapshot"}
                    or requires_pit
                    or latency in {"t_plus_1", "t_plus_2_plus", "delayed"}
                )
            )
            if gate_or_conf:
                should_shift = False
            if should_shift:
                s = s.shift(int(shift_days))

        # Decay: smooth event impulses / staleness effects using causal EWMA.
        if enable_decay:
            hl = float(half_life_override) if half_life_override and float(half_life_override) > 0.0 else _decay_halflife(decay_profile)
            if hl > 0:
                if cadence in {"event", "irregular"}:
                    if gate_or_conf:
                        # Keep confidence/gates as a level (do not diff into an impulse).
                        # If a decay profile is requested, apply it directly to the level.
                        if decay_profile == "two_phase":
                            fast = s.ewm(halflife=3.0, adjust=False, min_periods=1).mean()
                            slow = s.ewm(halflife=20.0, adjust=False, min_periods=1).mean()
                            s = 0.7 * fast + 0.3 * slow
                        else:
                            s = s.ewm(halflife=float(hl), adjust=False, min_periods=1).mean()
                    else:
                        impulse = s.diff().fillna(0.0)
                        if decay_profile == "two_phase":
                            fast = impulse.ewm(halflife=3.0, adjust=False, min_periods=1).mean()
                            slow = impulse.ewm(halflife=20.0, adjust=False, min_periods=1).mean()
                            s = 0.7 * fast + 0.3 * slow
                        else:
                            s = impulse.ewm(halflife=float(hl), adjust=False, min_periods=1).mean()
                elif cadence in {"weekly", "monthly", "quarterly", "snapshot"}:
                    if decay_profile == "two_phase":
                        fast = s.ewm(halflife=3.0, adjust=False, min_periods=1).mean()
                        slow = s.ewm(halflife=20.0, adjust=False, min_periods=1).mean()
                        s = 0.7 * fast + 0.3 * slow
                    else:
                        s = s.ewm(halflife=float(hl), adjust=False, min_periods=1).mean()

        out[col] = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out
