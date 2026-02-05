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

    # Event risk management (portfolio-only)
    "event_risk_hf": FamilyMeta(FeatureRole.RISK, update_cadence="daily", decay="none", allow_direct_alpha="false"),

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

    reg_path = path or (_repo_root() / "src" / "features" / "family_metadata_registry.json")
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
    "confidence",  # CRITICAL: any *_confidence must be HYGIENE, never alpha
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

_ALT_SIGNALS_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_ALT_SIGNALS_RISK_SUFFIXES = frozenset({
    # Beta / correlation (risk overlay → Portfolio)
    "beta_20d",
    "beta_change_rate",
    "beta_vix_interaction",
    "spy_correlation_20d",
    "qqq_correlation_20d",
    "sector_etf_correlation_20d",
    # Volatility block (risk overlay → Portfolio)
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "rv_ratio_5_20",
    "rv_z_20",
    "close_to_close_volatility",
    "open_to_close_volatility",
    "high_low_volatility_ratio",
    "intraday_volatility_ratio",
    # Liquidity stress (risk → Portfolio)
    "liquidity_stress_pct",
})

_ALT_SIGNALS_REGIME_SUFFIXES = frozenset({
    # Earnings timing (calendar/context → Portfolio)
    "days_since_last_earnings",
    "days_to_next_earnings",
    # Seasonality flag
    "turn_of_month_flag",
})

_ALT_SIGNALS_PREDICTIVE_SUFFIXES = frozenset({
    # Intraday / overnight (alpha → Mamba)
    "intraday_range_pct",
    "intraday_range_z",
    "opening_reversal",
    "closing_ramp",
    "overnight_return",
    "overnight_return_z",
    "gap_up_pct",
    "gap_down_pct",
    "gap_vs_vix_interaction",
    # Volume (mostly alpha → Mamba; liquidity_stress_pct is RISK)
    "relative_volume_20d",
    "volume_z_20d",
    "volume_trend_10d",
    "opening_volume_surge",
    "buy_volume_proxy",
    "volume_price_divergence",
    # Earnings drift (alpha → Mamba)
    "earnings_runup_10d",
    "post_earnings_drift_5d",
    # News / attention (alpha → Mamba)
    "news_volume_count",
    "news_volume_change",
    "news_volume_z",
    "google_trends_score",
    # Final signals (alpha → Mamba)
    "trend_acceleration",
    "mean_reversion_signal",
})

# ----------------------------------------------------------------------------
# ARIMA forecast authoritative overrides (Jan 2026 policy)
# ----------------------------------------------------------------------------
_ARIMA_PREFIX = "arima_forecast_"

_ARIMA_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "arima_log_likelihood",
    "confidence",
})

_ARIMA_RISK_SUFFIXES = frozenset({
    # Magnitude/variance → Portfolio for risk sizing
    "arima_abs_residual",
    "arima_uncertainty_proxy",
    "arima_innovation",
})

_ARIMA_REGIME_SUFFIXES = frozenset({
    "arima_persistence",
})

_ARIMA_PREDICTIVE_SUFFIXES = frozenset({
    # Forecast outputs (alpha → Mamba)
    "1d",
    "5d",
    # Signed residuals (alpha → Mamba)
    "arima_residual_t",
    "arima_residual_zscore",
    # Momentum indicator (alpha → Mamba)
    "arima_momentum_indicator",
})

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
    "coupling_change",          # abs(change) in coupling factor (FIXED: was cross_asset_coupling_change)
    # Composite policy state factor
    # NOTE: risk_onoff_factor must be normalized to [0,1] centered at 0.5 before use.
    # The derived risk_offness = clip((0.5 - risk_onoff_factor) / 0.5, 0, 1) assumes this.
    # Normalization: logistic(z-score) or min-max over 252d rolling window.
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

_QUANTILE_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "hf_conf",  # HF transformer confidence → gating only, not alpha
})

_QUANTILE_PREDICTIVE_SUFFIXES = frozenset({
    # Raw quantiles (8) → Mamba: full predictive distribution
    "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
    # Derived aliases (5) → Mamba: same as raw quantiles
    "q_low_5", "q_low_25", "q_median_50", "q_high_75", "q_high_95",
    # Directional skew signals (3) → Mamba: alpha
    "q_skewness_proxy", "q_tilt_direction", "skew",
    # HF transformer score → Mamba: directional alpha
    "hf_score",
})

_QUANTILE_RISK_SUFFIXES = frozenset({
    # Uncertainty/spread metrics → Portfolio: risk sizing only
    "q_spread_95_5",  # 95-5 spread for position sizing
    "q_vol_forecast", # Vol forecast for risk sizing
    "width",          # Distribution width risk
    "uncertainty",    # Normalized uncertainty
})


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
    """Route calibration columns: ALL 27 columns → HYGIENE → Portfolio parquet only.
    
    Critical: Calibration columns are model-quality diagnostics (hygiene).
    Feeding them into Mamba creates circular behaviour (model learns "when it is good"
    rather than learning market structure). These should gate/weight alpha, not become alpha.
    
    All columns route to Portfolio parquet:
    - Governance: has_data, activity, days_since_update
    - Coverage: q05_coverage..q99_coverage (8 quantiles)
    - Error: q05_error..q99_error (8 quantiles)
    - Interval: interval_coverage, interval_expected, interval_error
    - Quality: mean_calibration_error, overall_score
    - Status: requires_recalibration, sample_size, confidence
    """
    name = str(column or "")
    if not name.lower().startswith(_CALIB_PREFIX):
        return None
    suffix = name[len(_CALIB_PREFIX):].lower()
    
    # ALL calibration columns → HYGIENE (Portfolio parquet only)
    # Governance gates
    if suffix in {"has_data", "activity", "days_since_update"}:
        return FeatureRole.HYGIENE, "CAL:governance"
    
    # Coverage tracking (8 quantile coverage cols)
    if suffix.startswith("q") and suffix.endswith("_coverage"):
        return FeatureRole.HYGIENE, "CAL:coverage"
    
    # Error tracking (8 quantile error cols)
    if suffix.startswith("q") and suffix.endswith("_error"):
        return FeatureRole.HYGIENE, "CAL:error"
    
    # Interval reliability metrics
    if suffix in {"interval_coverage", "interval_expected", "interval_error"}:
        return FeatureRole.HYGIENE, "CAL:interval"
    
    # Quality scalars for weighting
    if suffix in {"mean_calibration_error", "overall_score"}:
        return FeatureRole.HYGIENE, "CAL:quality"
    
    # Status/control flags
    if suffix in {"requires_recalibration", "sample_size", "confidence"}:
        return FeatureRole.HYGIENE, "CAL:status"
    
    # Default: ALL calibration → HYGIENE (never to Mamba)
    return FeatureRole.HYGIENE, "CAL:hygiene_default"


def _online_learning_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route online_learning columns: ALL 35 columns → Portfolio parquet only.
    
    Critical: These are model/learner telemetry, NOT market alpha features.
    The 7 columns previously marked PREDICTIVE (direction_accuracy, model_confidence,
    trust_score, recalibration_strength, quantile_spread_shock, asymmetric_calibration_ratio,
    prediction_skewness) are reclassified to HYGIENE to prevent circular behaviour.
    
    Routing summary:
    - HYGIENE: governance, quality metrics, counters, meta-signals (25 cols)
    - REGIME: drift/update/compression flags, regime probabilities (10 cols)
    - RISK: quantile drift, calibration asymmetry, uncertainty (0 → moved to HYGIENE)
    
    Net: 0 columns to Mamba, 35 columns to Portfolio.
    """
    name = str(column or "")
    if not name.lower().startswith(_ONLINE_PREFIX):
        return None
    suffix = name[len(_ONLINE_PREFIX):].lower()
    
    # HYGIENE: governance/freshness gates
    if suffix in {"has_data", "activity", "days_since_update"}:
        return FeatureRole.HYGIENE, "OL:governance"
    
    # HYGIENE: quality metrics (gate/weight signals, not alpha)
    if suffix in {"mae", "rmse", "mape", "samples"}:
        return FeatureRole.HYGIENE, "OL:quality"
    
    # HYGIENE: maintenance counters
    if suffix in {"partial_retrains", "incremental_updates", "recalibration_needed"}:
        return FeatureRole.HYGIENE, "OL:maintenance"
    
    # HYGIENE: former PREDICTIVE meta-signals → reclassified to prevent circular behaviour
    # These describe "how good the model is" not "where the market is going"
    if suffix in {
        "direction_accuracy",
        "model_confidence",
        "trust_score",
        "recalibration_strength",
        "quantile_spread_shock",
        "asymmetric_calibration_ratio",
        "prediction_skewness",
    }:
        return FeatureRole.HYGIENE, "OL:meta_hygiene"
    
    # REGIME: drift/retrain/update event flags
    if suffix in {
        "drift_flag",
        "drift_events",
        "partial_retrain_flag",
        "model_updated_flag",
        "uncertainty_compression_alert",
    }:
        return FeatureRole.REGIME, "OL:drift_regime"
    
    # REGIME: regime probabilities (6 cols: trend +/-/neutral, vol high/low/normal)
    if suffix in {
        "regime_trend_positive",
        "regime_trend_negative",
        "regime_trend_neutral",
        "regime_vol_high",
        "regime_vol_low",
        "regime_vol_normal",
    }:
        return FeatureRole.REGIME, "OL:regime_prob"
    
    # RISK: distribution instability metrics (route to Portfolio for risk sizing)
    if suffix in {
        "quantile_drift_avg",
        "quantile_drift_q50",
        "quantile_drift_std",
        "quantile_spread_current",
        "upside_calibration_error",
        "downside_calibration_error",
        "uncertainty_compression_ratio",
    }:
        return FeatureRole.RISK, "OL:distribution_risk"
    
    # Default: ALL online_learning → HYGIENE (never to Mamba)
    return FeatureRole.HYGIENE, "OL:hygiene_default"


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
    
    TIMING CONVENTION (Option A - canonical):
    - flag=1.0 is emitted on the split effective date (close of that session)
    - Global shift(1) downstream makes it tradeable next session
    - Do NOT shift inside the family builder; this family participates in global shift
    
    Critical: The raw days_since=9999 value corrupts role-aware regime averaging.
    The feature builder should transform to bounded recency in [0,1]:
        recency = exp(-min(days_since, 252)/20)
    
    TOKEN INFERENCE WARNING:
    - 'days_since' suffix can be misclassified as HYGIENE by token-based inference
    - This override provides explicit per-column roles to prevent misclassification
    - Always use this family's explicit schema, never rely on token inference
    
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
    
    COLUMN STATUS NOTE:
    - corr_20_qqq_trend: Included in routing config (REGIME). If not generated by your
      feature builder, add it or remove from _CORRELATION_PORTFOLIO_REGIME_SUFFIXES.
    
    TOKEN INFERENCE WARNING:
    - '*_z' suffixes (e.g., corr_decoupling_z) can be misclassified as PREDICTIVE
      by token-based inference since 'z' often indicates z-scored alpha features.
    - This override provides explicit per-column roles: corr_decoupling_z → REGIME.
    - Always use this family's explicit schema, never rely on token inference.
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


# ============================================================================
# FIN_G4 (Cash Flow) family role overrides (Jan 2026)
# ============================================================================
_FIN_G4_PREFIX = "fin_g4_"

_FIN_G4_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G4_RISK_SUFFIXES = frozenset({
    # STRESS features: higher = worse (safe for risk aggregation)
    "cash_flow_stress",
    "earnings_quality_stress",
    # Accruals ratio (higher = lower earnings quality = worse)
    "accruals_ratio",
})

_FIN_G4_PREDICTIVE_SUFFIXES = frozenset({
    # Scale-free quality metrics for Mamba (slow alpha)
    "cfo_to_net_income",
    "fcf_to_net_income",
    "cfo_margin",
    "fcf_margin",
    "fcf_to_revenue",
    # Normalized metrics
    "cfo_to_assets",
    "fcf_to_assets",
    # Z-scores (PREDICTIVE: value tilt / mean reversion)
    "cfo_to_assets_zscore_3y",
    "fcf_to_assets_zscore_3y",
    "cfo_margin_zscore_3y",
    "fcf_margin_zscore_3y",
})

_FIN_G4_REGIME_SUFFIXES = frozenset({
    # Capital intensity (regime/industry structural)
    "capex_to_revenue",
})

# Raw dollar amounts: NOT scale-free, exclude from cross-sectional models
_FIN_G4_RAW_DOLLAR_SUFFIXES = frozenset({
    "operating_cash_flow",
    "free_cash_flow",
})


def _fin_g4_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g4 (cash flow) columns: stress to portfolio, quality metrics to Mamba.
    
    CRITICAL: Raw OCF/FCF are in dollars and NOT cross-sectionally comparable.
    Use normalized versions (cfo_to_assets, fcf_margin, z-scores) for Mamba.
    Use stress features (cash_flow_stress, earnings_quality_stress) for portfolio risk.
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): cash_flow_stress, earnings_quality_stress, accruals_ratio
    - Portfolio (REGIME): capex_to_revenue (industry structural)
    - Mamba (PREDICTIVE): scale-free quality metrics, z-scores
    - EXCLUDED: raw dollar amounts (not scale-free)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G4_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G4_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G4_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G4:governance"
    
    # RISK: stress features
    if suffix in _FIN_G4_RISK_SUFFIXES:
        return FeatureRole.RISK, "G4:cash_flow_stress"
    
    # PREDICTIVE: quality metrics and z-scores
    if suffix in _FIN_G4_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G4:quality_metrics"
    
    # REGIME: industry structural
    if suffix in _FIN_G4_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G4:regime"
    
    # Raw dollar amounts: REGIME (exclude from cross-sectional ranking)
    if suffix in _FIN_G4_RAW_DOLLAR_SUFFIXES:
        return FeatureRole.REGIME, "G4:dollar_not_scalefree"
    
    return FeatureRole.REGIME, "G4:regime_default"


# ============================================================================
# FIN_G5 (Growth) family role overrides (Jan 2026)
# ============================================================================
_FIN_G5_PREFIX = "fin_g5_"

_FIN_G5_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G5_RISK_SUFFIXES = frozenset({
    # Growth volatility (higher = unstable growers = worse)
    "growth_volatility_3y",
    "eps_growth_volatility_3y",
})

_FIN_G5_PREDICTIVE_SUFFIXES = frozenset({
    # YoY growth metrics: PREDICTIVE (medium-horizon drivers)
    "revenue_growth_yoy",
    "ebitda_growth_yoy",
    "cf_growth_yoy",
    "eps_growth_yoy",
    # Margin dynamics: PREDICTIVE (leads price re-rating)
    "margin_expansion",
})

_FIN_G5_REGIME_SUFFIXES = frozenset({
    # Multi-year CAGR: REGIME (slow style classification, growth vs value)
    "revenue_cagr_3y",
    "eps_cagr_3y",
})


def _fin_g5_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g5 (growth) columns: YoY growth to Mamba (optional), CAGR to regime.
    
    CRITICAL ROUTING:
    - YoY growth + margin_expansion: PREDICTIVE (medium-horizon alpha, optional for Mamba)
    - Multi-year CAGR: REGIME (slow style tilt, not short-term alpha)
    - Growth volatility: RISK (unstable growers get downweighted)
    
    Initial recommendation: Do NOT feed fin_g5 to Mamba until portfolio loop is stable.
    If/when enabling: include only YoY growth + margin_expansion (clipped/winsorized).
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): growth_volatility_3y (downweight unstable growers)
    - Portfolio (REGIME): revenue_cagr_3y, eps_cagr_3y (slow style)
    - Mamba (PREDICTIVE, optional): *_growth_yoy, margin_expansion
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G5_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G5_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G5_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G5:governance"
    
    # RISK: growth volatility
    if suffix in _FIN_G5_RISK_SUFFIXES:
        return FeatureRole.RISK, "G5:growth_risk"
    
    # PREDICTIVE: YoY growth + margin dynamics
    if suffix in _FIN_G5_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G5:growth_alpha"
    
    # REGIME: multi-year CAGR (slow style)
    if suffix in _FIN_G5_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G5:style_regime"
    
    return FeatureRole.REGIME, "G5:regime_default"


# ============================================================================
# FIN_G6 (Valuation) family role overrides (Jan 2026)
# ============================================================================
_FIN_G6_PREFIX = "fin_g6_"

_FIN_G6_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
})

_FIN_G6_RISK_SUFFIXES = frozenset({
    # Raw multiples: RISK (pathological values, use for risk/style not alpha)
    # PE has negative EPS issues, all multiples are industry-structural
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_ebitda",
    "dividend_yield",
    # STRESS features: valuation_richness = max(0, zscore)
    # Expensive stocks are fragile: more downside risk if sentiment shifts
    "valuation_richness",
    "pe_ratio_richness",
    "pb_ratio_richness",
    "ev_ebitda_richness",
})

_FIN_G6_PREDICTIVE_SUFFIXES = frozenset({
    # Z-scores: PREDICTIVE (value tilt / mean reversion for Mamba)
    # This is the USEFUL version - scale-controlled and learnable
    # Positive z = expensive vs history = potential short candidate or reduce
    # Negative z = cheap vs history = potential value opportunity
    "pe_ratio_zscore_5y",
    "pb_ratio_zscore_5y",
    "ev_ebitda_zscore_5y",
    "ps_ratio_zscore_5y",
})


def _fin_g6_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g6 (valuation) columns: z-scores to Mamba, raw multiples + richness to portfolio.
    
    CRITICAL ROLE SPLIT:
    - Raw multiples (pe_ratio, pb_ratio, etc.): RISK for portfolio style/risk.
      DO NOT feed to Mamba - pathological values (negative EPS), industry-structural.
    - Z-scores: PREDICTIVE value factor for Mamba (slow alpha on 21-252d horizons).
      These are the USEFUL version - scale-controlled and actually learnable.
    - Richness: RISK for portfolio risk scaling.
      Expensive stocks are fragile: higher position risk if sentiment shifts.
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update
    - Portfolio (RISK): raw multiples + valuation_richness, *_richness
    - Mamba (PREDICTIVE): ONLY z-scores (value factor alpha)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G6_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G6_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G6_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G6:governance"
    
    # RISK: raw multiples + valuation richness
    if suffix in _FIN_G6_RISK_SUFFIXES:
        return FeatureRole.RISK, "G6:valuation_risk"
    
    # PREDICTIVE: z-scores for value factor (ONLY these go to Mamba)
    if suffix in _FIN_G6_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G6:value_factor"
    
    return FeatureRole.RISK, "G6:risk_default"


# ============================================================================
# FIN_G7 (Dividend Policy / Shareholder Yield) family role overrides (Jan 2026)
# ============================================================================
_FIN_G7_PREFIX = "fin_g7_"

_FIN_G7_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    # shares_outstanding is raw scale, should not be ML input
    "shares_outstanding",
})

_FIN_G7_RISK_SUFFIXES = frozenset({
    # Payout sustainability / cut risk
    "payout_ratio",
    # Dividend exposure, not short-horizon alpha
    "dividend_yield_proxy",
    # Stability penalty/bonus
    "dividend_policy_stability",
    # Dilution is a quality negative
    "share_dilution_3y",
})

_FIN_G7_REGIME_SUFFIXES = frozenset({
    # Capital return regime (structural)
    "buyback_consistency",
})

_FIN_G7_PREDICTIVE_SUFFIXES = frozenset({
    # The ML-safe valuation-like transform (z-score)
    "yield_zscore_5y",
})


def _fin_g7_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route fin_g7 (dividend/shareholder yield) columns.
    
    CRITICAL ROUTING:
    - payout_ratio, dividend_yield_proxy, dividend_policy_stability, share_dilution_3y: RISK
      These are sustainability/cut risk inputs, NOT short-horizon alpha.
    - buyback_consistency: REGIME (capital return regime signal)
    - yield_zscore_5y: PREDICTIVE (ML-safe z-score, especially for longer horizons)
    - shares_outstanding: HYGIENE (raw scale, never ML input)
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update, shares_outstanding
    - Portfolio (RISK): payout_ratio, dividend_yield_proxy, policy_stability, dilution
    - Portfolio (REGIME): buyback_consistency
    - Mamba (PREDICTIVE): yield_zscore_5y only
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FIN_G7_PREFIX):
        return None
    
    suffix = name_lower[len(_FIN_G7_PREFIX):]
    
    # HYGIENE
    if suffix in _FIN_G7_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "G7:governance"
    
    # RISK: sustainability/cut risk metrics
    if suffix in _FIN_G7_RISK_SUFFIXES:
        return FeatureRole.RISK, "G7:dividend_risk"
    
    # REGIME: capital return regime
    if suffix in _FIN_G7_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "G7:capital_return_regime"
    
    # PREDICTIVE: z-score (ML-safe)
    if suffix in _FIN_G7_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "G7:yield_alpha"
    
    return FeatureRole.RISK, "G7:risk_default"


# ============================================================================
# FINBERT (Sentiment) family role overrides (Jan 2026)
# ============================================================================
_FINBERT_PREFIX = "finbert_"

_FINBERT_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    # CRITICAL: confidence must be HYGIENE globally, never alpha
    "confidence",
})

_FINBERT_PREDICTIVE_SUFFIXES = frozenset({
    # Direct alpha input
    "score",
})

_FINBERT_REGIME_SUFFIXES = frozenset({
    # High neutral = low informational content
    # Use to downweight sentiment impact, not as direct alpha
    "neutral",
})


def _finbert_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route finbert (sentiment) columns.
    
    CRITICAL ROUTING:
    - finbert_score: PREDICTIVE (direct alpha input)
    - finbert_neutral: REGIME (high neutral = low info, use to downweight sentiment)
    - finbert_confidence: HYGIENE (must NEVER be alpha input)
    
    Routing:
    - Portfolio (HYGIENE): has_data, activity, days_since_update, confidence
    - Mamba (PREDICTIVE): score only
    - Portfolio (REGIME): neutral (downweight sentiment when high)
    """
    name = str(column or "")
    name_lower = name.lower()
    
    if not name_lower.startswith(_FINBERT_PREFIX):
        return None
    
    suffix = name_lower[len(_FINBERT_PREFIX):]
    
    # HYGIENE
    if suffix in _FINBERT_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "FINBERT:governance"
    
    # PREDICTIVE: sentiment score
    if suffix in _FINBERT_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "FINBERT:sentiment_alpha"
    
    # REGIME: neutral probability
    if suffix in _FINBERT_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "FINBERT:info_quality"
    
    return FeatureRole.REGIME, "FINBERT:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# GARCH_IV FAMILY — Risk/Regime diagnostics, NOT Mamba input
# ─────────────────────────────────────────────────────────────────────────────
# This is risk + regime diagnostics. Primary consumers: Portfolio parquet,
# Policy Controller state, risk overlays, signal scaling.
# Only vol_momentum is OPTIONAL for Mamba (if normalized/clipped).
# ─────────────────────────────────────────────────────────────────────────────

_GARCH_IV_PREFIX = "garch_iv_"

# Governance + model diagnostics (HYGIENE)
_GARCH_IV_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity", 
    "days_since_update",
    "confidence",
    "garch_log_likelihood",       # Model fit quality, not tradable
    "garch_standardized_residual",  # Single-point noise, REMOVE from Mamba
})

# RISK: vol targeting, execution, risk scaling
_GARCH_IV_RISK_SUFFIXES = frozenset({
    "garch30_minus_garch180",     # Legacy RV spread
    "skew_proxy_downside_minus_upside",  # Tail risk proxy
    "garch_1d",                   # Vol targeting
    "garch_5d",                   # Policy state
    "garch_20d",                  # Regime baseline
    "garch_ratio_1d_20d",         # Shock detector
    "garch_zscore",               # Vol extremeness
    "garch_residual_vol",         # Model uncertainty
    "garch_short_long_ratio",     # Deviation from steady state
    "garch_vol_norm_20d",         # Model vs actual calibration
})

# REGIME: slow-moving, policy controller state
_GARCH_IV_REGIME_SUFFIXES = frozenset({
    "garch_persistence",          # Near 1.0 = persistent regime
    "garch_long_run_variance",    # Steady-state volatility
    "garch_vol_of_vol",           # Regime instability (NOT predictive!)
    "garch_vol_momentum",         # Vol trend (Policy state, optional Mamba)
    "garch_spike_flag",           # Binary shock detector → should be continuous
    "garch_shock_indicator",      # Binary outlier → should be continuous
})


def _garch_iv_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """GARCH_IV: Risk/regime diagnostics for Portfolio + Policy Controller.
    
    NOT Mamba input except optional vol_momentum (normalized/clipped).
    Binaries (spike_flag, shock_indicator) should be converted to continuous.
    standardized_residual should be REMOVED (single-point noise).
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_GARCH_IV_PREFIX):
        return None
    
    suffix = name_lower[len(_GARCH_IV_PREFIX):]
    
    # HYGIENE: governance + diagnostics
    if suffix in _GARCH_IV_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "GARCH_IV:governance_or_diagnostic"
    
    # RISK: vol targeting, execution, risk scaling
    if suffix in _GARCH_IV_RISK_SUFFIXES:
        return FeatureRole.RISK, "GARCH_IV:risk_scaling"
    
    # REGIME: policy controller state
    if suffix in _GARCH_IV_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "GARCH_IV:policy_state"
    
    # Default to RISK (conservative - not Mamba input)
    return FeatureRole.RISK, "GARCH_IV:risk_default"


# ─────────────────────────────────────────────────────────────────────────────
# INDEX_CONSTITUENTS FAMILY — Flow/Liquidity/Institutional behavior
# ─────────────────────────────────────────────────────────────────────────────
# This is flow + liquidity + institutional behavior, NOT sequence learning.
# Primary consumers: Portfolio parquet, Policy Controller.
# Binary pulses (added_*, removed_*) should be replaced with decaying pulses.
# ─────────────────────────────────────────────────────────────────────────────

_INDEX_CONST_PREFIX = "index_constituents_"

# Governance + availability (HYGIENE)
_INDEX_CONST_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
    "has_hist_gspc",              # Data availability
    "has_hist_dji",               # Data availability
})

# RISK: liquidity scaling, position sizing
_INDEX_CONST_RISK_SUFFIXES = frozenset({
    "weight_gspc",                # Current snapshot weight (liquidity scaling)
    "weight_dji",                 # Current snapshot weight
})

# REGIME: membership state, slow style factors, flow regime
_INDEX_CONST_REGIME_SUFFIXES = frozenset({
    "member_gspc",                # Large-cap status
    "member_dji",                 # Blue-chip status
    "num_indices",                # Institutional ownership proxy
    "days_since_add_gspc",        # Post-inclusion state
    "days_since_add_dji",
    "days_since_remove_gspc",     # Post-removal state
    "days_since_remove_dji",
    "add_flow_20d",               # Slow rebalancing wave (not short-term predictive)
    "remove_flow_20d",            # Slow removal wave
    # Binary pulses - should be decaying, route to REGIME for now
    "added_gspc",                 # Binary pulse → should be exp(-days/τ)
    "added_dji",
    "removed_gspc",
    "removed_dji",
})

# PREDICTIVE: short-term flow signals (Portfolio use, NOT Mamba)
_INDEX_CONST_PREDICTIVE_SUFFIXES = frozenset({
    "add_flow_5d",                # Short-term addition wave
    "remove_flow_5d",             # Short-term removal wave
})


def _index_constituents_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """INDEX_CONSTITUENTS: Flow/liquidity signals for Portfolio + Policy.
    
    NOT Mamba input. Binary pulses should be converted to decaying pulses.
    Weights are current snapshots, NOT historical time series.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_INDEX_CONST_PREFIX):
        return None
    
    suffix = name_lower[len(_INDEX_CONST_PREFIX):]
    
    # HYGIENE: governance + data availability
    if suffix in _INDEX_CONST_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "INDEX_CONST:governance"
    
    # RISK: liquidity scaling
    if suffix in _INDEX_CONST_RISK_SUFFIXES:
        return FeatureRole.RISK, "INDEX_CONST:liquidity_scaling"
    
    # REGIME: membership state, flow regime
    if suffix in _INDEX_CONST_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "INDEX_CONST:membership_or_flow_regime"
    
    # PREDICTIVE: short-term flows (for Portfolio, not Mamba)
    if suffix in _INDEX_CONST_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "INDEX_CONST:short_term_flow"
    
    # Default to REGIME (conservative)
    return FeatureRole.REGIME, "INDEX_CONST:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# MARKETCAP_HISTORY FAMILY — Size/Liquidity/Institutional regime
# ─────────────────────────────────────────────────────────────────────────────
# This is size, liquidity, and institutional regime, NOT pattern learning.
# Primary consumers: Portfolio parquet (caps, costs), Policy Controller.
# NEVER feed into Mamba — level ≠ alpha.
# ─────────────────────────────────────────────────────────────────────────────

_MCAP_HISTORY_PREFIX = "marketcap_history_"

# Governance (HYGIENE)
_MCAP_HISTORY_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

# RISK: risk scaling, turnover penalties
_MCAP_HISTORY_RISK_SUFFIXES = frozenset({
    "mcap_chg_20d",               # Short-term size change (dilution/M&A detector)
    "mcap_vol_63d",               # Size volatility (penalize unstable names)
    "float_turnover",             # Portfolio costs + turnover cap
})

# REGIME: size determines caps, leverage, max_name, gross exposure
_MCAP_HISTORY_REGIME_SUFFIXES = frozenset({
    "mcap",                       # Raw market cap level
    "log_mcap",                   # Log market cap (Policy state)
    "mcap_chg_63d",               # Medium-term size change (Policy state)
    "mcap_chg_252d",              # Long-term size change (Policy state)
    "turnover_z_252d",            # Liquidity regime (Policy state)
})


def _marketcap_history_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """MARKETCAP_HISTORY: Size/liquidity/institutional regime for Portfolio + Policy.
    
    NEVER Mamba input. Size level determines:
    - Liquidity caps
    - Turnover tolerance
    - Max name concentration
    - Gross exposure limits
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_MCAP_HISTORY_PREFIX):
        return None
    
    suffix = name_lower[len(_MCAP_HISTORY_PREFIX):]
    
    # HYGIENE: governance
    if suffix in _MCAP_HISTORY_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "MCAP_HISTORY:governance"
    
    # RISK: volatility, turnover penalties
    if suffix in _MCAP_HISTORY_RISK_SUFFIXES:
        return FeatureRole.RISK, "MCAP_HISTORY:risk_scaling"
    
    # REGIME: size state, liquidity regime
    if suffix in _MCAP_HISTORY_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "MCAP_HISTORY:size_regime"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "MCAP_HISTORY:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# MICROSTRUCTURE FAMILY — Primary Mamba input family (surgically clean)
# ─────────────────────────────────────────────────────────────────────────────
# Most important family for Mamba. Must be surgically clean:
# - Candle geometry, volume surprises, order-flow, overnight gaps → Mamba
# - Impact, spreads, liquidity, volatility → Portfolio
# - Vol-of-vol, liquidity regime → Policy Controller
# ─────────────────────────────────────────────────────────────────────────────

_MICRO_PREFIX = "microstructure_"

# Governance + binary flags (HYGIENE only, never learning)
_MICRO_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
    "stale_tick",                 # Binary gating flag only
    "zero_range_flag",            # REMOVE (duplicate) - HYGIENE to gate
    "low_liquidity_flag",         # Binary gating flag only
    # Quote data availability
    "quote_has_data",
    "bid",                        # Quote level - last row only, not historical
    "ask",
    "bid_size",
    "ask_size",
})

# RISK: volatility, spreads, impact → Portfolio parquet
_MICRO_RISK_SUFFIXES = frozenset({
    # Range & volatility ratios - NOT Mamba
    "true_range",
    "atr_ratio",
    "range_pct",
    "range_scaled",
    # Volume & liquidity proxies
    "volume_liquidity",
    "turnover",
    "amihud",
    # Volatility & price impact
    "impact_ratio",
    "impact_volatility",
    "spread_proxy",
    "intraday_vol_proxy",
    # Overnight volatility
    "overnight_vol",
    "intraday_vs_overnight_vol",
    # Quote spreads
    "spread_abs",
    "spread_bps",
})

# REGIME: liquidity regime, vol-of-vol → Policy Controller
_MICRO_REGIME_SUFFIXES = frozenset({
    "hl_volume_corr",             # Volume-range correlation regime
    "vol_of_vol",                 # Volatility regime instability
    "gap_direction",              # Optional: {-1,0,1} embedding
})

# PREDICTIVE: Primary Mamba inputs - candle geometry, order-flow, gaps
_MICRO_PREDICTIVE_SUFFIXES = frozenset({
    # Candle geometry → Mamba gold
    "body_pct",
    "wick_top",
    "wick_bottom",
    "shadow_ratio",
    # Volume surprises → Mamba
    "volume_zscore",
    "volume_surge",
    # Order-flow proxies → Mamba gold
    "ofi_proxy",
    "signed_volume",
    "pressure_proxy",
    "demand_supply_ratio",
    "liquidity_imbalance",
    # Overnight gap → Mamba
    "overnight_gap",
    # Quote imbalance → Mamba (but time-gate carefully)
    "quote_imbalance",
})


def _microstructure_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """MICROSTRUCTURE: Primary Mamba input family (surgically clean).
    
    - Candle geometry, volume surprises, order-flow, overnight gaps → Mamba
    - Impact, spreads, liquidity, volatility → Portfolio
    - Vol-of-vol, liquidity regime → Policy Controller
    
    Binary flags (stale_tick, zero_range_flag, low_liquidity_flag) are HYGIENE
    only - they gate, they don't learn.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_MICRO_PREFIX):
        return None
    
    suffix = name_lower[len(_MICRO_PREFIX):]
    
    # HYGIENE: governance + binary flags + quote levels
    if suffix in _MICRO_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "MICRO:governance_or_gating"
    
    # RISK: volatility, spreads, impact → Portfolio
    if suffix in _MICRO_RISK_SUFFIXES:
        return FeatureRole.RISK, "MICRO:portfolio_risk"
    
    # REGIME: vol-of-vol, liquidity regime → Policy
    if suffix in _MICRO_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "MICRO:policy_state"
    
    # PREDICTIVE: candle geometry, order-flow, gaps → Mamba
    if suffix in _MICRO_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "MICRO:mamba_input"
    
    # Default to PREDICTIVE for unknown microstructure features (conservative for this family)
    return FeatureRole.PREDICTIVE, "MICRO:predictive_default"


# ─────────────────────────────────────────────────────────────────────────────
# MULTIASSET FAMILY — Systematic exposure measurement
# ─────────────────────────────────────────────────────────────────────────────
# This measures systematic exposure (betas, correlations), NOT short-horizon 
# price formation. Primary consumers: Portfolio Parquet + Policy Controller.
# NO Mamba input (except optional clipped spread_spy_20).
# ─────────────────────────────────────────────────────────────────────────────

_MULTIASSET_PREFIX = "multiasset_"

# Governance (HYGIENE)
_MULTIASSET_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
})

# RISK: systematic exposure for portfolio construction
_MULTIASSET_RISK_SUFFIXES = frozenset({
    # SPY (S&P 500)
    "corr_spy_120",               # 120-day correlation with market
    "beta_spy_120",               # Systematic risk exposure
    # QQQ (Nasdaq 100 - growth tilt)
    "corr_qqq_120",
    "beta_qqq_120",
    # IWM (Russell 2000 - size tilt)
    "corr_iwm_120",
    "beta_iwm_120",
    # ACWI (Global exposure)
    "corr_acwi_120",
    "beta_acwi_120",
    # PCA market factor
    "equity_factor_market",       # PC1: overall market exposure
})

# REGIME: style factors for Policy Controller
_MULTIASSET_REGIME_SUFFIXES = frozenset({
    "equity_factor_growth_value", # PC2: growth vs value tilt
    "equity_factor_size",         # PC3: large vs small cap
})

# PREDICTIVE: only spread_spy_20 (relative strength, portfolio use)
# NOTE: This is PREDICTIVE but should NOT go to Mamba (portfolio allocator only)
_MULTIASSET_PREDICTIVE_SUFFIXES = frozenset({
    "spread_spy_20",              # Relative strength vs SPY (clip if using in Mamba)
})


def _multiasset_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """MULTIASSET: Systematic exposure for Portfolio + Policy Controller.
    
    NOT Mamba input (except optional clipped spread_spy_20).
    Betas/correlations are slow-moving and break sequence stationarity.
    PCA factors are structural, not predictive.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_MULTIASSET_PREFIX):
        return None
    
    suffix = name_lower[len(_MULTIASSET_PREFIX):]
    
    # HYGIENE: governance
    if suffix in _MULTIASSET_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "MULTIASSET:governance"
    
    # RISK: systematic exposure
    if suffix in _MULTIASSET_RISK_SUFFIXES:
        return FeatureRole.RISK, "MULTIASSET:systematic_exposure"
    
    # REGIME: style factors
    if suffix in _MULTIASSET_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "MULTIASSET:style_factor"
    
    # PREDICTIVE: relative strength (portfolio use only)
    if suffix in _MULTIASSET_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "MULTIASSET:relative_strength"
    
    # Default to RISK (conservative - not Mamba input)
    return FeatureRole.RISK, "MULTIASSET:risk_default"


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS FAMILY — Snapshot-only options chain metrics
# ─────────────────────────────────────────────────────────────────────────────
# Snapshot architecture: only snapshot date has has_data=1, all other dates=0.
# Primary use: Policy conditioning + Portfolio constraints.
# NO Mamba input — snapshot data with zero-fill destroys temporal learning.
# ─────────────────────────────────────────────────────────────────────────────

_OPTIONS_PREFIX = "options_"

# Governance + chain health diagnostics (HYGIENE)
_OPTIONS_HYGIENE_SUFFIXES = frozenset({
    "has_data",
    "activity",
    "days_since_update",
    "confidence",
    # Chain availability
    "strikes_available",          # Total number of strikes (chain health)
    "expiries_available",         # Number of expiries (chain health)
    # Pricing features - NOT signals, just chain diagnostics
    "call_bid_avg",
    "call_ask_avg",
    "call_last_avg",
    "put_bid_avg",
    "put_ask_avg",
    "put_last_avg",
})

# RISK: IV for portfolio risk scaling, bid-ask for liquidity
_OPTIONS_RISK_SUFFIXES = frozenset({
    "atm_iv",                     # At-the-money IV → portfolio risk scaling
    "call_iv_avg",                # Average call IV
    "put_iv_avg",                 # Average put IV
    "iv_spread",                  # Put-call IV skew → fear premium
    "bid_ask_spread_pct",         # Liquidity proxy
})

# REGIME: sentiment, positioning, event proximity → Policy Controller
_OPTIONS_REGIME_SUFFIXES = frozenset({
    # Volume ratios (sentiment)
    "call_volume",
    "put_volume",
    "put_call_volume_ratio",
    # Open interest ratios (structural positioning)
    "call_oi",
    "put_oi",
    "put_call_oi_ratio",
    # Moneyness (skew structure)
    "otm_call_pct",
    "otm_put_pct",
    # Term structure (event proximity)
    "nearest_expiry_days",
})


def _options_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """OPTIONS: Snapshot-only options metrics for Policy + Portfolio.
    
    NO Mamba input — snapshot architecture with zero-fill destroys temporal learning.
    IV → portfolio risk scaling
    Volume/OI ratios → policy sentiment
    Pricing features → HYGIENE chain diagnostics only
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_OPTIONS_PREFIX):
        return None
    
    suffix = name_lower[len(_OPTIONS_PREFIX):]
    
    # HYGIENE: governance + chain health + pricing diagnostics
    if suffix in _OPTIONS_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "OPTIONS:governance_or_chain_health"
    
    # RISK: IV for portfolio risk scaling
    if suffix in _OPTIONS_RISK_SUFFIXES:
        return FeatureRole.RISK, "OPTIONS:portfolio_risk_scaling"
    
    # REGIME: sentiment, positioning → Policy
    if suffix in _OPTIONS_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "OPTIONS:policy_sentiment"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "OPTIONS:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS_ANCHORING FAMILY — Institutional options anchoring with decay weighting
# ─────────────────────────────────────────────────────────────────────────────
# Daily time-series (leakage-safe, continuous, decay-weighted).
# Mamba: YES (10 columns) — IV regime, skew, expected move, positioning.
# Portfolio: YES (all 12) — risk mgmt + gating + monitoring.
# Key correction: IV anchoring/skew columns are REGIME (not RISK).
# ─────────────────────────────────────────────────────────────────────────────

_OPTIONS_ANCHORING_PREFIX = "options_anchoring_"

# Governance / data quality — HYGIENE (freshness gating, NOT alpha)
_OPTIONS_ANCHORING_HYGIENE_SUFFIXES = frozenset({
    "days_since_update",          # Freshness gating
    "confidence",                 # Data quality (can also serve as mask)
})

# REGIME: IV anchoring (z-scores, percentiles describe IV environment)
# CORRECTED: these are regime descriptors, not "risk" in portfolio-only sense
_OPTIONS_ANCHORING_IV_REGIME_SUFFIXES = frozenset({
    "iv_anchor_pct",              # Z-score of ATM IV (high/low IV environment)
    "iv_percentile_30d",          # Percentile rank → regime context
    "iv_percentile_1yr",          # Slow regime context
})

# REGIME: Skew anchoring (sentiment/fear regime variables)
_OPTIONS_ANCHORING_SKEW_REGIME_SUFFIXES = frozenset({
    "iv_skew_anchor",             # Skew = sentiment/fear regime
    "iv_skew_zscore",             # Normalized skew state
    "risk_reversal_25d",          # Institutional skew proxy → regime
})

# REGIME: Positioning (flow and structural positioning)
_OPTIONS_ANCHORING_POSITIONING_REGIME_SUFFIXES = frozenset({
    "put_call_vol_ratio_anchor",  # Flow regime (short memory)
    "put_call_oi_ratio_anchor",   # Structural positioning regime (slow memory)
    "em_vs_real_vol_ratio",       # Forward vs backward vol regime
})

# PREDICTIVE: Expected move (forward-looking uncertainty for Mamba)
_OPTIONS_ANCHORING_PREDICTIVE_SUFFIXES = frozenset({
    "expected_move_pct",          # Straddle-implied move → Mamba conditioning
})


def _options_anchoring_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """OPTIONS_ANCHORING: Institutional options anchoring with decay weighting.
    
    Mamba sees all except HYGIENE (10 columns):
      - IV anchoring (regime descriptors)
      - Skew anchoring (sentiment regime)
      - Expected move (forward uncertainty)
      - Positioning (flow + structural regime)
    
    Portfolio sees all 12 (risk mgmt + gating + monitoring).
    
    Key correction: iv_anchor_pct, iv_percentile_*, iv_skew_*, risk_reversal_25d
    are REGIME (not RISK) — they describe IV environment, not portfolio risk scaling.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_OPTIONS_ANCHORING_PREFIX):
        return None
    
    suffix = name_lower[len(_OPTIONS_ANCHORING_PREFIX):]
    
    # HYGIENE: governance/freshness (NOT alpha, NOT Mamba input)
    if suffix in _OPTIONS_ANCHORING_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "OPTIONS_ANCHORING:governance"
    
    # REGIME: IV anchoring (regime descriptors) → BOTH (Mamba + Portfolio)
    if suffix in _OPTIONS_ANCHORING_IV_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "OPTIONS_ANCHORING:iv_regime"
    
    # REGIME: Skew anchoring (sentiment regime) → BOTH
    if suffix in _OPTIONS_ANCHORING_SKEW_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "OPTIONS_ANCHORING:skew_regime"
    
    # REGIME: Positioning (flow + structural) → BOTH
    if suffix in _OPTIONS_ANCHORING_POSITIONING_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "OPTIONS_ANCHORING:positioning_regime"
    
    # PREDICTIVE: Expected move → BOTH (Mamba can learn conditional drift)
    if suffix in _OPTIONS_ANCHORING_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "OPTIONS_ANCHORING:expected_move"
    
    # Default to REGIME (conservative - regime conditioning)
    return FeatureRole.REGIME, "OPTIONS_ANCHORING:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# REGIME FAMILY — Institutional-grade regime classification
# ─────────────────────────────────────────────────────────────────────────────
# Model conditioning + portfolio gating. Most columns go to BOTH.
# Mamba: YES (8-9 columns) — probabilities, duration, trend metrics.
# Portfolio: YES (all 9) — gating, risk scaling, regime awareness.
# Recommendation: exclude regime_label from Mamba (prefer probabilities).
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_PREFIX = "regime_"

# REGIME: Probabilities (continuous, best for Mamba)
_REGIME_PROBABILITY_SUFFIXES = frozenset({
    "bull_probability",           # Bull regime probability (0-1)
    "bear_probability",           # Bear regime probability (0-1)
    "neutral_probability",        # Neutral regime probability (0-1)
})

# REGIME: Label and temporal context
_REGIME_CONTEXT_SUFFIXES = frozenset({
    "label",                      # Discrete label (0/1/2) — portfolio-first
    "duration",                   # Days in current regime
    "change_flag",                # Transition risk indicator
})

# PREDICTIVE: Trend metrics (continuous trend strength, extreme detector)
_REGIME_PREDICTIVE_SUFFIXES = frozenset({
    "trend_ratio",                # Fast/slow MA ratio → Mamba conditioning
    "trend_ratio_zscore",         # Extreme detector → Mamba conditioning
})

# RISK: Volatility for sizing control
_REGIME_RISK_SUFFIXES = frozenset({
    "volatility_20d",             # Sizing control + Mamba conditioning
})


def _regime_family_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """REGIME: Institutional-grade regime classification.
    
    Mamba sees (8 columns, recommend exclude label):
      - bull_probability, bear_probability, neutral_probability
      - duration, change_flag
      - trend_ratio, trend_ratio_zscore
      - volatility_20d
    
    Portfolio sees all 9 (gating + risk scaling + regime awareness).
    
    Note: regime_label introduces discontinuities; prefer probabilities for Mamba.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_REGIME_PREFIX):
        return None
    
    suffix = name_lower[len(_REGIME_PREFIX):]
    
    # REGIME: Probabilities → BOTH (best Mamba input, continuous)
    if suffix in _REGIME_PROBABILITY_SUFFIXES:
        return FeatureRole.REGIME, "REGIME:probability"
    
    # REGIME: Label + temporal context → BOTH (label optional for Mamba)
    if suffix in _REGIME_CONTEXT_SUFFIXES:
        return FeatureRole.REGIME, "REGIME:context"
    
    # PREDICTIVE: Trend metrics → BOTH
    if suffix in _REGIME_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "REGIME:trend_metric"
    
    # RISK: Volatility → BOTH (sizing + conditioning)
    if suffix in _REGIME_RISK_SUFFIXES:
        return FeatureRole.RISK, "REGIME:volatility"
    
    # Default to REGIME (family intent)
    return FeatureRole.REGIME, "REGIME:default"


# ─────────────────────────────────────────────────────────────────────────────
# SHORT_INTEREST FAMILY — Short interest tracking with squeeze analysis
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (6 columns) — change/momentum/squeeze forecasting only.
# Portfolio: YES (all) — risk constraints, regime gating.
# Key principle: Bi-monthly data is forward-filled; avoid letting Mamba learn
# data-availability patterns. Core metrics are RISK, not PREDICTIVE.
# ─────────────────────────────────────────────────────────────────────────────

_SHORT_INTEREST_PREFIX = "short_interest_"

# HYGIENE: Governance + quality scoring (never Mamba)
# Note: conf is alias of confidence, score_raw/score are diagnostic
_SHORT_INTEREST_HYGIENE_SUFFIXES = frozenset({
    "has_data",                   # Data availability flag
    "confidence",                 # Data quality score
    "conf",                       # Alias of confidence (WARN: consider dropping)
    "score_raw",                  # Raw composite score (diagnostic)
    "score",                      # Normalized composite score (diagnostic)
})

# RISK: Core short metrics for portfolio constraints (not Mamba)
# These are positioning/risk pressure descriptors, not alpha
_SHORT_INTEREST_RISK_SUFFIXES = frozenset({
    "percent",                    # Short % of shares outstanding
    "ratio",                      # Days to cover (short interest / ADV)
    "days_to_cover",              # Alias of ratio (WARN: consider dropping)
    "float_short_pct",            # Short % of float (primary metric)
    "shares_on_loan_pct",         # Utilization (borrow demand)
    "borrow_rate",                # Annual borrow rate (cost to short)
})

# PREDICTIVE: Change/momentum/squeeze forecasting → Mamba
# These describe how positioning is evolving (directional descriptors)
_SHORT_INTEREST_PREDICTIVE_SUFFIXES = frozenset({
    "change_1m",                  # MoM % change in short interest
    "change_3m",                  # 3-month % change
    "momentum",                   # Rolling trend in short interest
    "squeeze_risk_flag",          # Binary: squeeze conditions met
    "squeeze_risk_score",         # Composite squeeze risk (0-100)
    "squeeze_probability",        # Probability of squeeze event (0-1)
})

# REGIME: Statistical context for gating (not Mamba)
# Crowdedness regime descriptors; avoid learning forward-fill patterns
_SHORT_INTEREST_REGIME_SUFFIXES = frozenset({
    "zscore_1y",                  # Z-score vs 1-year history
    "pct_zscore_3y",              # Z-score of float_short_pct vs 3-year
    "short_to_oi_ratio",          # Short vs options OI
    "short_vs_institutional",     # Short vs institutional ownership
    "borrow_rate_zscore_3y",      # Z-score of borrow rate vs 3-year
})


def _short_interest_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """SHORT_INTEREST: Short interest tracking with squeeze analysis.
    
    Mamba sees (6 columns):
      - change_1m, change_3m, momentum (positioning evolution)
      - squeeze_risk_flag, squeeze_risk_score, squeeze_probability
    
    Portfolio sees all (risk constraints + regime gating):
      - Core metrics for sizing constraints (high borrow = reduce size)
      - Regime z-scores for crowdedness gating
    
    Key concern: Bi-monthly data is forward-filled. Avoid letting Mamba
    learn data-availability patterns from statistical context columns.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_SHORT_INTEREST_PREFIX):
        return None
    
    suffix = name_lower[len(_SHORT_INTEREST_PREFIX):]
    
    # HYGIENE: Governance + quality (never Mamba)
    if suffix in _SHORT_INTEREST_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "SHORT_INTEREST:governance"
    
    # RISK: Core short metrics (Portfolio constraints, not Mamba)
    if suffix in _SHORT_INTEREST_RISK_SUFFIXES:
        return FeatureRole.RISK, "SHORT_INTEREST:risk_constraint"
    
    # PREDICTIVE: Change/momentum/squeeze → Mamba
    if suffix in _SHORT_INTEREST_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "SHORT_INTEREST:squeeze_forecast"
    
    # REGIME: Statistical context for gating (not Mamba)
    if suffix in _SHORT_INTEREST_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "SHORT_INTEREST:crowdedness_regime"
    
    # Default to RISK (conservative - not Mamba input)
    return FeatureRole.RISK, "SHORT_INTEREST:risk_default"


# ─────────────────────────────────────────────────────────────────────────────
# SUBSIDIARY FAMILY — Organizational complexity from quarterly OpEx/R&D
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: NO (recommend REGIME for all, slow-moving structural descriptors).
# Portfolio: YES (all) — complexity regime, operational scale context.
# Key principle: Quarterly data is publication-lagged and slow-moving.
# Growth/intensity features are REGIME-like, not short-horizon alpha.
# ─────────────────────────────────────────────────────────────────────────────

_SUBSIDIARY_PREFIX = "subsidiary_"

# HYGIENE: Governance (never Mamba)
_SUBSIDIARY_HYGIENE_SUFFIXES = frozenset({
    "has_data",                   # Data availability flag
    "activity",                   # Activity indicator
    "days_since_update",          # Freshness (quarterly, often high)
    "confidence",                 # Data quality score
})

# REGIME: Core scale + structure (slow-moving, sector/size encoders)
_SUBSIDIARY_STRUCTURE_REGIME_SUFFIXES = frozenset({
    "rd_spending",                # R&D spending (absolute USD)
    "opex_spending",              # Total operating expenses
    "complexity_score",           # Log(1 + OpEx) - scale-invariant proxy
})

# REGIME: Growth/intensity (treated as REGIME, not PREDICTIVE)
# Quarterly financials are publication-lagged and slow-moving
# Better as context than daily alpha inputs
_SUBSIDIARY_GROWTH_REGIME_SUFFIXES = frozenset({
    "rd_intensity",               # R&D as % of OpEx (innovation focus)
    "rd_qoq_growth",              # QoQ % change in R&D
    "opex_qoq_growth",            # QoQ % change in OpEx
    "rd_yoy_growth",              # YoY % change in R&D
    "opex_yoy_growth",            # YoY % change in OpEx
})


def _subsidiary_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """SUBSIDIARY: Organizational complexity from quarterly OpEx/R&D.
    
    Mamba sees: NOTHING (all columns are slow-moving structural descriptors)
    
    Portfolio sees all (complexity regime + operational scale context):
      - Core metrics for normalization (don't compare across complexity)
      - Growth rates for regime detection
    
    Key concern: Quarterly financials are publication-lagged (30-45 days).
    Even when lag-safe, they move slowly — better as context than alpha.
    
    Exception: If you have a long-horizon head (30-252d targets), move
    growth/intensity features to PREDICTIVE for that head only.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_SUBSIDIARY_PREFIX):
        return None
    
    suffix = name_lower[len(_SUBSIDIARY_PREFIX):]
    
    # HYGIENE: Governance (never Mamba)
    if suffix in _SUBSIDIARY_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "SUBSIDIARY:governance"
    
    # REGIME: Core scale + structure (slow-moving, not Mamba)
    if suffix in _SUBSIDIARY_STRUCTURE_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "SUBSIDIARY:operational_scale"
    
    # REGIME: Growth/intensity (quarterly, slow-moving, not Mamba)
    if suffix in _SUBSIDIARY_GROWTH_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "SUBSIDIARY:growth_regime"
    
    # Default to REGIME (family intent - slow-moving structural)
    return FeatureRole.REGIME, "SUBSIDIARY:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# TFT_FEATURES FAMILY — Multi-scale temporal patterns for forecasting
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (4 columns) — trend strength/signal-to-noise only.
# Portfolio: YES (13 columns) — volatility, seasonality, stability, confidence.
# Key principle: Volatility/stability/seasonality are regime/risk descriptors,
# not short-horizon alpha. Only trend metrics go to Mamba.
# ─────────────────────────────────────────────────────────────────────────────

_TFT_FEATURES_PREFIX = "tft_features_"

# PREDICTIVE: Trend metrics → Mamba
_TFT_FEATURES_PREDICTIVE_SUFFIXES = frozenset({
    "trend_short_strength",       # 5-10d momentum bursts
    "trend_long_strength",        # 20-30d structural alignment
    "trend_signal_to_noise",      # |trend| / volatility
    "trend_importance",           # Blended trend importance
})

# REGIME: Trend consistency, seasonality, stability (NOT Mamba)
# These describe market state, not predictive signals
_TFT_FEATURES_REGIME_SUFFIXES = frozenset({
    "trend_consistency",          # Fraction positive days in 20d (regime)
    "vol_regime_zscore",          # High/low vol regime detector
    "weekday_effect",             # Rolling weekday return pattern
    "month_phase",                # Day-of-month normalized
    "seasonality_importance",     # Seasonal pattern strength
    "stability_short",            # 5-10d stability
    "stability_long",             # 20-30d stability
    "stability_score",            # Alias of stability_long
})

# RISK: Volatility metrics for sizing (NOT Mamba)
_TFT_FEATURES_RISK_SUFFIXES = frozenset({
    "vol_short",                  # 5-10d realized vol
    "vol_long",                   # 20-30d baseline vol
    "vol_trend",                  # Is vol rising or calming?
    "volatility_importance",      # Alias of vol_long
})

# HYGIENE: Confidence score
_TFT_FEATURES_HYGIENE_SUFFIXES = frozenset({
    "confidence",                 # Dynamic confidence score
})


def _tft_features_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """TFT_FEATURES: Multi-scale temporal patterns for forecasting.
    
    Mamba sees (4 columns):
      - trend_short_strength, trend_long_strength
      - trend_signal_to_noise, trend_importance
    
    Portfolio sees all (vol + seasonality + stability + confidence):
      - Volatility metrics for sizing
      - Seasonality/stability for regime awareness
    
    Key risk: Names like trend_consistency, weekday_effect, stability_*
    lack "risk/regime" tokens and may default to PREDICTIVE incorrectly.
    This override ensures correct routing.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_TFT_FEATURES_PREFIX):
        return None
    
    suffix = name_lower[len(_TFT_FEATURES_PREFIX):]
    
    # HYGIENE: Confidence (never Mamba)
    if suffix in _TFT_FEATURES_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "TFT_FEATURES:confidence"
    
    # PREDICTIVE: Trend metrics → Mamba
    if suffix in _TFT_FEATURES_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "TFT_FEATURES:trend"
    
    # REGIME: Consistency, seasonality, stability (not Mamba)
    if suffix in _TFT_FEATURES_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "TFT_FEATURES:regime_context"
    
    # RISK: Volatility (not Mamba)
    if suffix in _TFT_FEATURES_RISK_SUFFIXES:
        return FeatureRole.RISK, "TFT_FEATURES:volatility"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "TFT_FEATURES:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# PEER_SCREENER_CONTEXT FAMILY — Cross-sectional peer/sector/industry context
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (10 columns) — cheap_pct, universe z-scores/percentiles.
# Portfolio: YES (14 columns) — percentile ranks, peer counts, options/short z.
# Key risk: *_pct columns often default to PREDICTIVE; *_peer_count to PREDICTIVE.
# This override ensures correct routing.
# ─────────────────────────────────────────────────────────────────────────────

_PEER_SCREENER_CONTEXT_PREFIX = "peer_screener_context_"

# HYGIENE: Governance + peer counts (data quality, not alpha)
_PEER_SCREENER_CONTEXT_HYGIENE_SUFFIXES = frozenset({
    "has_data",                   # Data availability flag
    "sector_peer_count",          # Number of sector peers (data quality)
    "industry_peer_count",        # Number of industry peers (data quality)
})

# PREDICTIVE: Cheap percentiles + universe momentum/valuation → Mamba
# These are cross-sectional relative value signals
_PEER_SCREENER_CONTEXT_PREDICTIVE_SUFFIXES = frozenset({
    # Sector cheap percentiles
    "sector_pe_ratio_cheap_pct",
    "sector_pb_ratio_cheap_pct",
    "sector_ev_ebitda_cheap_pct",
    # Industry cheap percentiles
    "industry_pe_ratio_cheap_pct",
    "industry_pb_ratio_cheap_pct",
    "industry_ev_ebitda_cheap_pct",
    # Universe momentum/valuation (z-scores and percentiles)
    "universe_momentum_z",
    "universe_valuation_z",
    "universe_momentum_pct",
    "universe_valuation_pct",
})

# REGIME: Percentile ranks within sector/industry (context, not alpha)
_PEER_SCREENER_CONTEXT_REGIME_SUFFIXES = frozenset({
    "sector_pe_ratio_pct",        # Sector PE percentile
    "sector_pb_ratio_pct",        # Sector PB percentile
    "sector_ev_ebitda_pct",       # Sector EV/EBITDA percentile
    "industry_pe_ratio_pct",      # Industry PE percentile
    "industry_pb_ratio_pct",      # Industry PB percentile
    "industry_ev_ebitda_pct",     # Industry EV/EBITDA percentile
})

# RISK: Universe options/short interest z-scores (portfolio constraints)
_PEER_SCREENER_CONTEXT_RISK_SUFFIXES = frozenset({
    "universe_options_iv_z",      # Universe IV z-score
    "universe_options_skew_z",    # Universe skew z-score
    "universe_short_interest_z",  # Universe short interest z-score
    "universe_options_iv_pct",    # Universe IV percentile
    "universe_options_skew_pct",  # Universe skew percentile
})


def _peer_screener_context_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """PEER_SCREENER_CONTEXT: Cross-sectional peer/sector/industry context.
    
    Mamba sees (10 columns):
      - sector/industry *_cheap_pct (relative value signals)
      - universe_momentum_z, universe_valuation_z
      - universe_momentum_pct, universe_valuation_pct
    
    Portfolio sees (14 columns):
      - *_pct percentile ranks (context, not alpha)
      - peer_count (data quality)
      - options/short interest z-scores (risk constraints)
    
    Key risk: *_pct columns often default to PREDICTIVE incorrectly.
    *_peer_count often defaults to PREDICTIVE. This override fixes that.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_PEER_SCREENER_CONTEXT_PREFIX):
        return None
    
    suffix = name_lower[len(_PEER_SCREENER_CONTEXT_PREFIX):]
    
    # HYGIENE: Governance + peer counts (data quality)
    if suffix in _PEER_SCREENER_CONTEXT_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "PEER_SCREENER_CONTEXT:governance"
    
    # PREDICTIVE: Cheap percentiles + universe momentum/valuation → Mamba
    if suffix in _PEER_SCREENER_CONTEXT_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "PEER_SCREENER_CONTEXT:relative_value"
    
    # REGIME: Percentile ranks (context, not alpha)
    if suffix in _PEER_SCREENER_CONTEXT_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "PEER_SCREENER_CONTEXT:peer_context"
    
    # RISK: Options/short interest z-scores (portfolio constraints)
    if suffix in _PEER_SCREENER_CONTEXT_RISK_SUFFIXES:
        return FeatureRole.RISK, "PEER_SCREENER_CONTEXT:risk_constraint"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "PEER_SCREENER_CONTEXT:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL_GRAPH_CONTEXT FAMILY — Cross-symbol graph embeddings + neighbor stats
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (all columns) — embeddings, neighbor aggregates, centrality
# Portfolio: NO — these are pure alpha signals, not risk/regime context
# Key insight: Graph structure encodes cross-symbol relationships for relational
#              learning without changing dataset shape.
# ─────────────────────────────────────────────────────────────────────────────

_SYMBOL_GRAPH_CONTEXT_PREFIX = "sgc_"

# PREDICTIVE: All sgc_* columns are designed for Mamba alpha input
# - sgc_emb_* : SVD embeddings from adjacency (captures graph structure)
# - sgc_neighbor_* : Weighted neighbor aggregates (momentum, vol, drawdown, lags)
# - sgc_degree, sgc_strength_topk, sgc_pagerank : Centrality metrics
_SYMBOL_GRAPH_CONTEXT_PREDICTIVE_SUFFIXES = frozenset({
    # Embeddings (8 dims by default)
    "emb_0", "emb_1", "emb_2", "emb_3", "emb_4", "emb_5", "emb_6", "emb_7",
    # Centrality
    "degree",
    "strength_topk",
    "pagerank",
    # Neighbor aggregates (current + lagged)
    "neighbor_momentum",
    "neighbor_vol",
    "neighbor_drawdown",
    "neighbor_momentum_lag1",
    "neighbor_momentum_lag5",
})


def _symbol_graph_context_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """SYMBOL_GRAPH_CONTEXT: Cross-symbol graph embeddings + neighbor stats.
    
    Mamba sees (ALL columns):
      - sgc_emb_0..7 : Graph embeddings (weekly refresh)
      - sgc_degree, sgc_strength_topk, sgc_pagerank : Centrality
      - sgc_neighbor_* : Neighbor-aggregated momentum/vol/drawdown
    
    Portfolio sees: NOTHING (pure alpha signals)
    
    All columns are PREDICTIVE by design. This family encodes cross-symbol
    relationships via learned adjacency (correlation + sector) without
    changing dataset shape.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_SYMBOL_GRAPH_CONTEXT_PREFIX):
        return None
    
    suffix = name_lower[len(_SYMBOL_GRAPH_CONTEXT_PREFIX):]
    
    # All sgc_* columns are PREDICTIVE (Mamba input)
    # Check against known suffixes for explicit match
    if suffix in _SYMBOL_GRAPH_CONTEXT_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "SYMBOL_GRAPH_CONTEXT:graph_feature"
    
    # Handle dynamic embedding dims (sgc_emb_8, sgc_emb_9, etc.)
    if suffix.startswith("emb_"):
        return FeatureRole.PREDICTIVE, "SYMBOL_GRAPH_CONTEXT:embedding"
    
    # Handle any neighbor aggregates with different windows
    if suffix.startswith("neighbor_"):
        return FeatureRole.PREDICTIVE, "SYMBOL_GRAPH_CONTEXT:neighbor_aggregate"
    
    # Default: still PREDICTIVE (all graph features go to Mamba)
    return FeatureRole.PREDICTIVE, "SYMBOL_GRAPH_CONTEXT:graph_default"


# ─────────────────────────────────────────────────────────────────────────────
# DOC_EMBEDDING_NOVELTY_HF FAMILY — GDELT global event novelty detection
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (1 column ONLY) — `score` (PREDICTIVE) reaches Mamba under current rules.
# Portfolio: YES (14 columns) — all REGIME/RISK/HYGIENE columns.
# Key issue: Cluster novelty metrics are REGIME → Portfolio-only by default.
# If you want Mamba to see cluster novelty, add prefix to forced-mamba list.
# ─────────────────────────────────────────────────────────────────────────────

_DOC_EMBEDDING_NOVELTY_HF_PREFIX = "doc_embedding_novelty_hf_"

# HYGIENE: Governance + metadata (not features, consider moving to attrs)
_DOC_EMBEDDING_NOVELTY_HF_HYGIENE_SUFFIXES = frozenset({
    "has_data",                   # Data availability
    "n_events",                   # Event count (data quality)
    "n_articles",                 # Article count (data quality)
    "top_theme",                  # String - will be coerced to 0.0 downstream
    "theme_weight",               # Metadata weight
    "baseline_mean",              # Not scalar - move to attrs instead
})

# PREDICTIVE: Overall novelty score → Mamba (only column that reaches Mamba)
_DOC_EMBEDDING_NOVELTY_HF_PREDICTIVE_SUFFIXES = frozenset({
    "score",                      # Overall novelty (0-1) - ONLY Mamba column
})

# RISK: Confidence score (portfolio constraint)
_DOC_EMBEDDING_NOVELTY_HF_RISK_SUFFIXES = frozenset({
    "conf",                       # Confidence based on event count/coherence
})

# REGIME: Cluster-specific novelty + spike detection (Portfolio-only by default)
_DOC_EMBEDDING_NOVELTY_HF_REGIME_SUFFIXES = frozenset({
    "novelty_macro",              # Macro/economic cluster
    "novelty_geopolitical",       # Geopolitical cluster
    "novelty_regulatory",         # Regulatory/policy cluster
    "novelty_energy",             # Energy/commodities cluster
    "novelty_conflict",           # Conflict/crisis cluster
    "novelty_tech",               # Technology/innovation cluster
    "novelty_spike_flag",         # >2σ novelty spike
    "novelty_persistence_5d",     # Sustained high novelty
})


def _doc_embedding_novelty_hf_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """DOC_EMBEDDING_NOVELTY_HF: GDELT global event novelty detection.
    
    Mamba sees (1 column ONLY by default):
      - `score` (overall novelty) — the only PREDICTIVE column
    
    Portfolio sees (14 columns):
      - Cluster novelty metrics (REGIME)
      - Confidence (RISK)
      - Metadata/governance (HYGIENE)
    
    Key issue: If you want Mamba to see cluster novelty, either:
      - Add doc_embedding_novelty_hf_ to forced-mamba prefix list, OR
      - Re-role cluster columns as PREDICTIVE (not recommended)
    
    WARNING: top_theme is string, baseline_mean may not be scalar.
    Consider moving these to attrs instead of feature columns.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_DOC_EMBEDDING_NOVELTY_HF_PREFIX):
        return None
    
    suffix = name_lower[len(_DOC_EMBEDDING_NOVELTY_HF_PREFIX):]
    
    # HYGIENE: Governance + metadata
    if suffix in _DOC_EMBEDDING_NOVELTY_HF_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "DOC_EMBEDDING_NOVELTY_HF:governance"
    
    # PREDICTIVE: Overall novelty → Mamba
    if suffix in _DOC_EMBEDDING_NOVELTY_HF_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "DOC_EMBEDDING_NOVELTY_HF:novelty_score"
    
    # RISK: Confidence (portfolio constraint)
    if suffix in _DOC_EMBEDDING_NOVELTY_HF_RISK_SUFFIXES:
        return FeatureRole.RISK, "DOC_EMBEDDING_NOVELTY_HF:confidence"
    
    # REGIME: Cluster novelty + spike detection (Portfolio-only by default)
    if suffix in _DOC_EMBEDDING_NOVELTY_HF_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "DOC_EMBEDDING_NOVELTY_HF:cluster_novelty"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "DOC_EMBEDDING_NOVELTY_HF:regime_default"


# ─────────────────────────────────────────────────────────────────────────────
# EARNINGS_TRANSCRIPT_HF FAMILY — Earnings call sentiment analysis
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: YES (ALL columns) — This family IS in forced-mamba prefix list.
# Portfolio: NO — Portfolio does not see these signals by default!
# Key issue: If portfolio needs transcript uncertainty/risk/confidence,
# you must mirror columns to portfolio parquet or change prefix override.
# ─────────────────────────────────────────────────────────────────────────────

_EARNINGS_TRANSCRIPT_HF_PREFIX = "earnings_transcript_hf_"

# HYGIENE: Governance (but still goes to Mamba due to prefix forcing)
_EARNINGS_TRANSCRIPT_HF_HYGIENE_SUFFIXES = frozenset({
    "has_data",                   # Data availability
})

# PREDICTIVE: Sentiment + deltas + interactions → Mamba (forced by prefix)
_EARNINGS_TRANSCRIPT_HF_PREDICTIVE_SUFFIXES = frozenset({
    "score",                      # Overall sentiment (-1 to 1)
    "score_delta_qoq",            # QoQ sentiment change (HIGH ALPHA)
    "score_delta_yoy",            # YoY sentiment change
    "sentiment_divergence",       # Prepared - QA (management defensiveness)
    "sentiment_shock",            # Delta × confidence
})

# RISK: Confidence + tone dimensions (forced to Mamba by prefix)
_EARNINGS_TRANSCRIPT_HF_RISK_SUFFIXES = frozenset({
    "conf",                       # Intensity × volume × dispersion
    "uncertainty_score",          # Information asymmetry (vol amplifier)
    "risk_score",                 # Explicit downside language
})

# REGIME: Sectional sentiment (forced to Mamba by prefix)
_EARNINGS_TRANSCRIPT_HF_REGIME_SUFFIXES = frozenset({
    "score_prepared",             # Prepared remarks (often upward biased)
    "score_qa",                   # Q&A section (reveals stress)
})


def _earnings_transcript_hf_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """EARNINGS_TRANSCRIPT_HF: Earnings call sentiment analysis.
    
    Mamba sees ALL columns (forced by prefix rule):
      - score, score_delta_qoq, score_delta_yoy (PREDICTIVE)
      - sentiment_divergence, sentiment_shock (PREDICTIVE)
      - conf, uncertainty_score, risk_score (RISK but forced to Mamba)
      - score_prepared, score_qa (REGIME but forced to Mamba)
      - has_data (HYGIENE but forced to Mamba)
    
    Portfolio sees NOTHING by default!
    
    Key issue: If portfolio needs transcript uncertainty/risk/confidence:
      - Mirror selected columns into portfolio parquet, OR
      - Have portfolio read from mamba parquet too, OR
      - Change prefix override to split by role instead of forcing all to Mamba
    
    Role assignments here are for semantic correctness; routing is by prefix.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_EARNINGS_TRANSCRIPT_HF_PREFIX):
        return None
    
    suffix = name_lower[len(_EARNINGS_TRANSCRIPT_HF_PREFIX):]
    
    # HYGIENE: Governance (still forced to Mamba by prefix)
    if suffix in _EARNINGS_TRANSCRIPT_HF_HYGIENE_SUFFIXES:
        return FeatureRole.HYGIENE, "EARNINGS_TRANSCRIPT_HF:governance"
    
    # PREDICTIVE: Sentiment + deltas + interactions
    if suffix in _EARNINGS_TRANSCRIPT_HF_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "EARNINGS_TRANSCRIPT_HF:sentiment"
    
    # RISK: Confidence + tone dimensions
    if suffix in _EARNINGS_TRANSCRIPT_HF_RISK_SUFFIXES:
        return FeatureRole.RISK, "EARNINGS_TRANSCRIPT_HF:risk_tone"
    
    # REGIME: Sectional sentiment
    if suffix in _EARNINGS_TRANSCRIPT_HF_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "EARNINGS_TRANSCRIPT_HF:sectional"
    
    # Handle event-prefixed columns (forced to Mamba by prefix)
    if suffix.startswith("event_"):
        return FeatureRole.REGIME, "EARNINGS_TRANSCRIPT_HF:event_metadata"
    
    # Default to PREDICTIVE (family intent - sentiment signals)
    return FeatureRole.PREDICTIVE, "EARNINGS_TRANSCRIPT_HF:predictive_default"


# ─────────────────────────────────────────────────────────────────────────────
# MACRO_TST_HF FAMILY — Macro TST daily (no governance columns)
# ─────────────────────────────────────────────────────────────────────────────
# Mamba: Only 3 PREDICTIVE interaction terms (symbol-specific exposures)
# Portfolio: Everything else (REGIME + RISK)
# Key issue: Almost everything is REGIME/RISK → Portfolio. Only the
# symbol-specific interaction columns are PREDICTIVE → Mamba.
# ─────────────────────────────────────────────────────────────────────────────

_MACRO_TST_HF_PREFIX = "macro_tst_hf_"

# REGIME: Core macro levels + changes + economic momentum
_MACRO_TST_HF_REGIME_SUFFIXES = frozenset({
    # Layer 1: Core regime (levels + changes)
    "tnx_level",
    "irx_level",
    "2y_level",
    "tnx_change_1d",
    "tnx_change_5d",
    "curve_slope",
    "curve_slope_change",
    "vix_spike_flag",
    "oil_change",
    "gold_change",
    # Layer 2: Economic momentum (derived)
    "derived_inflation_accel",
    "derived_gdp_growth_accel",
    "derived_unemployment_change",
    "derived_real_rate_change",
    "derived_debt_to_gdp_change",
    "derived_trade_balance_change",
    # Layer 3: Rate shock columns
    "rates_2y_change_1d",
    "rates_2y_change_5d",
})

# RISK: VIX + credit spreads
_MACRO_TST_HF_RISK_SUFFIXES = frozenset({
    # Layer 1: VIX metrics
    "vix_level",
    "vix_return_1d",
    "vix_return_5d",
    # Layer 1: Credit spreads
    "credit_spread_level",
    "credit_spread_change",
    "credit_spread_z",
    # Layer 4: VIX beta (risk overlay)
    "vix_beta_sensitivity",
})

# PREDICTIVE: Symbol-specific interactions ONLY (Mamba input)
_MACRO_TST_HF_PREDICTIVE_SUFFIXES = frozenset({
    # Layer 4: Symbol-specific macro interactions
    "real_rate_growth_beta",      # Sensitivity to real rate changes
    "oil_energy_exposure",        # Energy sector exposure
    "curve_bank_exposure",        # Banks/financials curve exposure
})


def _macro_tst_hf_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """MACRO_TST_HF: Macro TST daily (no governance columns).
    
    CRITICAL ROUTING (almost everything → Portfolio):
    
    Mamba sees (3 columns ONLY):
      - real_rate_growth_beta: Symbol-specific real rate sensitivity
      - oil_energy_exposure: Symbol-specific energy exposure
      - curve_bank_exposure: Symbol-specific curve exposure
    
    Portfolio sees (ALL other columns):
      - REGIME: Levels, changes, economic momentum, shocks
      - RISK: VIX, credit spreads, vix_beta_sensitivity
    
    Key issue: No governance columns in this family. Routing is purely by role.
    """
    name_lower = str(column or "").lower()
    
    if not name_lower.startswith(_MACRO_TST_HF_PREFIX):
        return None
    
    suffix = name_lower[len(_MACRO_TST_HF_PREFIX):]
    
    # PREDICTIVE: Symbol-specific interactions → Mamba
    if suffix in _MACRO_TST_HF_PREDICTIVE_SUFFIXES:
        return FeatureRole.PREDICTIVE, "MACRO_TST_HF:symbol_interaction"
    
    # RISK: VIX + credit spreads → Portfolio
    if suffix in _MACRO_TST_HF_RISK_SUFFIXES:
        return FeatureRole.RISK, "MACRO_TST_HF:risk_overlay"
    
    # REGIME: Levels + changes + economic momentum → Portfolio
    if suffix in _MACRO_TST_HF_REGIME_SUFFIXES:
        return FeatureRole.REGIME, "MACRO_TST_HF:macro_regime"
    
    # Default to REGIME (conservative - not Mamba input)
    return FeatureRole.REGIME, "MACRO_TST_HF:regime_default"


def _candle_mechanics_role_override(column: str) -> Optional[Tuple[FeatureRole, str]]:
    """Route candle_mechanics columns per repo truth (Jan 2026 policy).
    
    A) Default Mamba (PREDICTIVE) - 21 alpha-suitable columns
    B) Optional Mamba - 5 columns controlled by CANDLE_MECHANICS_MAMBA_OPTIONAL env var
       If not enabled, these go to Portfolio.
    C) Portfolio only - governance, risk/vol, calendar encodings
    
    The optional toggle is handled at the parquet split layer (prep_families.py),
    not here. This function returns the "ideal" role; the split layer may override.
    """
    name = str(column or "")
    if not name.lower().startswith(_CANDLE_PREFIX):
        return None
    suffix = name[len(_CANDLE_PREFIX):].lower()

    # === C) PORTFOLIO ONLY ===
    
    # Governance / hygiene
    if suffix in {"has_data", "activity", "days_since_update", "confidence"}:
        return FeatureRole.HYGIENE, "CM:governance"
    
    # Risk/vol + supporting stats (Portfolio for position sizing)
    if suffix in {
        "range_pct",
        "atr14",
        "atr60",
        "body_atr14",
        "gap_atr14",
        "close_vs_prev_close_atr14",
        "high_vs_prev_close_atr14",
        "low_vs_prev_close_atr14",
        "ret_1d",
        "ret_3d",
        "rv_5",
        "rv_20",
        "dist_to_high_20_atr14",
        "dist_to_low_20_atr14",
        "dist_to_mean_20_atr14",
    }:
        return FeatureRole.RISK, "CM:risk"
    
    # Calendar encodings (one-hot style → REGIME for policy)
    if suffix in {
        "dow_1", "dow_2", "dow_3", "dow_4", "dow_5",
        "moy_1", "moy_2", "moy_3", "moy_4", "moy_5", "moy_6",
        "moy_7", "moy_8", "moy_9", "moy_10", "moy_11", "moy_12",
    }:
        return FeatureRole.REGIME, "CM:calendar"
    
    # === B) OPTIONAL MAMBA (default to REGIME so split layer can promote) ===
    # These are REGIME here; prep_families.py promotes to Mamba if env var enabled
    if suffix in {
        "inside_bar",
        "outside_bar",
        "range_atr14",
        "sign_sum_3",
        "sign_sum_5",
    }:
        return FeatureRole.REGIME, "CM:optional"
    
    # === A) DEFAULT MAMBA (PREDICTIVE) ===
    # 21 alpha-suitable columns
    if suffix in {
        "close_pos",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "wick_imbalance",
        "gap_pct",
        "close_vs_prev_close_pct",
        "high_vs_prev_close_pct",
        "low_vs_prev_close_pct",
        "logret_1d",
        "ret_5d",
        "rng_z_20",
        "trend_3",
        "trend_5",
        "body_z_20",
        "dir_change",
        "vol_ratio_20",
        "vol_log_chg",
        "up_day",
        "down_day",
        "atr_ratio_14_60",
    }:
        return FeatureRole.PREDICTIVE, "CM:predictive"

    # Default: unknown candle columns → RISK (conservative, Portfolio)
    return FeatureRole.RISK, "CM:risk_default"


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

    # FIN_G4 (Cash Flow) - stress to portfolio, quality metrics to Mamba
    fin_g4_override = _fin_g4_role_override(column)
    if fin_g4_override is not None:
        return fin_g4_override

    # FIN_G5 (Growth) - YoY growth optional for Mamba, CAGR to regime, volatility to risk
    fin_g5_override = _fin_g5_role_override(column)
    if fin_g5_override is not None:
        return fin_g5_override

    # FIN_G6 (Valuation) - raw multiples to RISK, z-scores to Mamba PREDICTIVE
    fin_g6_override = _fin_g6_role_override(column)
    if fin_g6_override is not None:
        return fin_g6_override

    # FIN_G7 (Dividend Policy) - payout/dilution to RISK, yield_zscore to Mamba
    fin_g7_override = _fin_g7_role_override(column)
    if fin_g7_override is not None:
        return fin_g7_override

    # FINBERT (Sentiment) - score to Mamba, neutral to regime, confidence to HYGIENE
    finbert_override = _finbert_role_override(column)
    if finbert_override is not None:
        return finbert_override

    # GARCH_IV (Volatility) - Risk/regime diagnostics, NOT Mamba input
    # Portfolio parquet + Policy Controller state + risk overlays
    garch_iv_override = _garch_iv_role_override(column)
    if garch_iv_override is not None:
        return garch_iv_override

    # INDEX_CONSTITUENTS (Flows) - Flow/liquidity, NOT Mamba input
    # Portfolio parquet + Policy Controller + turnover logic
    index_const_override = _index_constituents_role_override(column)
    if index_const_override is not None:
        return index_const_override

    # MARKETCAP_HISTORY (Size) - Size/liquidity/institutional regime, NOT Mamba input
    # Portfolio parquet (caps, costs) + Policy Controller
    mcap_history_override = _marketcap_history_role_override(column)
    if mcap_history_override is not None:
        return mcap_history_override

    # MICROSTRUCTURE - Primary Mamba input family (surgically clean)
    # Candle geometry, order-flow, gaps → Mamba; Impact, spreads → Portfolio
    micro_override = _microstructure_role_override(column)
    if micro_override is not None:
        return micro_override

    # MULTIASSET - Systematic exposure, NOT Mamba input (except optional spread_spy_20)
    # Portfolio parquet + Policy Controller (betas, correlations, style factors)
    multiasset_override = _multiasset_role_override(column)
    if multiasset_override is not None:
        return multiasset_override

    # OPTIONS - Snapshot-only options metrics, NO Mamba input
    # Policy Controller (sentiment, positioning) + Portfolio (IV scaling)
    options_override = _options_role_override(column)
    if options_override is not None:
        return options_override

    # OPTIONS_ANCHORING - Institutional decay-weighted options, BOTH Mamba + Portfolio
    # IV regime, skew regime, expected move, positioning → Mamba conditioning
    options_anchoring_override = _options_anchoring_role_override(column)
    if options_anchoring_override is not None:
        return options_anchoring_override

    # REGIME - Institutional regime classification, BOTH Mamba + Portfolio
    # Probabilities, duration, trend metrics → Mamba conditioning + gating
    regime_override = _regime_family_role_override(column)
    if regime_override is not None:
        return regime_override

    # SHORT_INTEREST - Change/momentum/squeeze to Mamba, core metrics to Portfolio
    # Bi-monthly data, forward-filled; avoid learning data-availability patterns
    short_interest_override = _short_interest_role_override(column)
    if short_interest_override is not None:
        return short_interest_override

    # SUBSIDIARY - Organizational complexity, NO Mamba input
    # Quarterly financials are slow-moving; better as regime context
    subsidiary_override = _subsidiary_role_override(column)
    if subsidiary_override is not None:
        return subsidiary_override

    # TFT_FEATURES - Trend metrics to Mamba, vol/seasonality/stability to Portfolio
    # High risk of misclassification: trend_consistency, weekday_effect, stability_*
    tft_features_override = _tft_features_role_override(column)
    if tft_features_override is not None:
        return tft_features_override

    # PEER_SCREENER_CONTEXT - Cheap percentiles + universe z to Mamba
    # High risk of misclassification: *_pct defaults, *_peer_count, options/short z
    peer_screener_override = _peer_screener_context_role_override(column)
    if peer_screener_override is not None:
        return peer_screener_override

    # SYMBOL_GRAPH_CONTEXT - Cross-symbol graph embeddings + neighbor stats to Mamba
    # All columns are PREDICTIVE by design (graph embeddings, neighbor aggregates)
    symbol_graph_context_override = _symbol_graph_context_role_override(column)
    if symbol_graph_context_override is not None:
        return symbol_graph_context_override

    # DOC_EMBEDDING_NOVELTY_HF - Only `score` to Mamba, cluster novelty to Portfolio
    # Key issue: If you want cluster novelty in Mamba, add to forced-mamba prefix list
    doc_embedding_novelty_override = _doc_embedding_novelty_hf_role_override(column)
    if doc_embedding_novelty_override is not None:
        return doc_embedding_novelty_override

    # EARNINGS_TRANSCRIPT_HF - ALL columns to Mamba (forced-mamba prefix)
    # Key issue: Portfolio sees NOTHING; mirror columns if portfolio needs them
    earnings_transcript_override = _earnings_transcript_hf_role_override(column)
    if earnings_transcript_override is not None:
        return earnings_transcript_override

    # MACRO_TST_HF - Only symbol interactions to Mamba; everything else to Portfolio
    # Key issue: Almost ALL columns are REGIME/RISK → Portfolio. Only 3 PREDICTIVE.
    macro_tst_hf_override = _macro_tst_hf_role_override(column)
    if macro_tst_hf_override is not None:
        return macro_tst_hf_override

    name = str(column).lower()

    # Explicit governance suffixes (structural, not keyword heuristics)
    # CRITICAL: confidence must be HYGIENE globally
    if name.endswith("_has_data") or name.endswith("_activity") or name.endswith("_days_since_update") or name.endswith("__days_since_update") or name.endswith("_confidence"):
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
    family_map: Optional[Mapping[str, str]] = None,
    family_meta: Optional[Mapping[str, FamilyMeta]] = None,
) -> pd.DataFrame:
    """Apply role-specific transforms in-place on a copy and return it."""

    out = df.copy()
    eps = 1e-6

    gov_suffixes = ("_has_data", "_activity", "_days_since_update", "_confidence", "_conf")

    def _cadence_defaults(cadence: str) -> Tuple[int, int, int, bool]:
        token = str(cadence or "").strip().lower()
        if token == "daily":
            return 252, 252, 20, False
        if token == "weekly":
            return 104, 104, 12, False
        if token == "monthly":
            return 60, 60, 8, False
        if token == "quarterly":
            return 0, 0, 8, True
        if token == "event":
            return 40, 40, 20, True
        if token == "intraday":
            bars = int(os.getenv("BARS_PER_DAY", "390"))
            z_win = max(100, bars * 10)
            return z_win, z_win, max(20, int(z_win * 0.2)), False
        return pred_window, risk_window, min_periods, False

    def _update_mask_for_family(fam: str) -> Optional[pd.Series]:
        if fam and f"{fam}_days_since_update" in out.columns:
            ds = pd.to_numeric(out[f"{fam}_days_since_update"], errors="coerce")
            return ds.fillna(1.0) <= 0.0
        if fam and f"{fam}_has_data" in out.columns:
            hd = pd.to_numeric(out[f"{fam}_has_data"], errors="coerce")
            return hd.fillna(0.0) > 0.0
        return None

    def _rolling_stat(
        series: pd.Series,
        *,
        window: int,
        min_p: int,
        update_mask: Optional[pd.Series],
        stat: str,
        quantile: Optional[float] = None,
    ) -> pd.Series:
        s = series
        if update_mask is not None:
            mask = update_mask.reindex(s.index).fillna(False)
            s_evt = s[mask]
            if s_evt.empty:
                return pd.Series(index=s.index, dtype=float)
            if window and window > 0:
                roll = s_evt.rolling(int(window), min_periods=int(min_p))
            else:
                roll = s_evt.expanding(min_periods=int(min_p))
            if stat == "mean":
                out_stat = roll.mean().shift(1)
            elif stat == "std":
                out_stat = roll.std(ddof=0).shift(1)
            elif stat == "min":
                out_stat = roll.min().shift(1)
            elif stat == "max":
                out_stat = roll.max().shift(1)
            elif stat == "quantile" and quantile is not None:
                out_stat = roll.quantile(float(quantile)).shift(1)
            else:
                out_stat = roll.mean().shift(1)
            return out_stat.reindex(s.index).ffill()

        roll = s.rolling(int(window), min_periods=int(min_p)) if window and window > 0 else s.expanding(min_periods=int(min_p))
        if stat == "mean":
            return roll.mean().shift(1)
        if stat == "std":
            return roll.std(ddof=0).shift(1)
        if stat == "min":
            return roll.min().shift(1)
        if stat == "max":
            return roll.max().shift(1)
        if stat == "quantile" and quantile is not None:
            return roll.quantile(float(quantile)).shift(1)
        return roll.mean().shift(1)

    for col, role in roles.items():
        if col not in out.columns:
            continue
        if col == "date":
            continue
        if str(col).endswith(gov_suffixes):
            continue

        if role == FeatureRole.HYGIENE:
            continue

        s = _safe_numeric(out[col])
        if s.isna().all():
            out[col] = 0.0
            continue

        fam = ""
        if family_map is not None:
            fam = str(family_map.get(col, ""))
        meta = family_meta.get(fam) if family_meta is not None else None
        cadence_token = str(getattr(meta, "update_cadence", "")) if meta is not None else ""
        cad_pred_win, cad_risk_win, cad_min_p, update_points_only = _cadence_defaults(cadence_token)
        update_mask = _update_mask_for_family(fam) if update_points_only else None

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
                win = int(cad_pred_win) if cad_pred_win else 0
                mp = int(cad_min_p) if cad_min_p else int(min_periods)
                mn = _rolling_stat(s2, window=win, min_p=mp, update_mask=update_mask, stat="min")
                mx = _rolling_stat(s2, window=win, min_p=mp, update_mask=update_mask, stat="max")
                denom = (mx - mn).abs() + eps
                out[col] = ((s2 - mn) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
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

            win = int(cad_risk_win) if cad_risk_win else 0
            mp = int(cad_min_p) if cad_min_p else int(min_periods)
            lo = _rolling_stat(t, window=win, min_p=mp, update_mask=update_mask, stat="quantile", quantile=q_lo)
            hi = _rolling_stat(t, window=win, min_p=mp, update_mask=update_mask, stat="quantile", quantile=q_hi)
            denom = (hi - lo).abs() + eps
            scaled = pd.Series((t - lo) / denom, index=t.index)
            out[col] = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
            continue

        if role == FeatureRole.PREDICTIVE:
            name = str(col).lower()
            if _is_boolish(out[col]) or _is_probabilistic(s):
                out[col] = _safe_numeric(out[col]).fillna(0.0).clip(0.0, 1.0)
                continue
            if "zscore" in name or name.endswith("_z") or "_z_" in name:
                out[col] = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
                continue
            win = int(cad_pred_win) if cad_pred_win else 0
            mp = int(cad_min_p) if cad_min_p else int(min_periods)
            mu = _rolling_stat(s, window=win, min_p=mp, update_mask=update_mask, stat="mean")
            sd = _rolling_stat(s, window=win, min_p=mp, update_mask=update_mask, stat="std")
            z = (s - mu) / (sd + eps)
            out[col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
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
