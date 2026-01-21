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

    name = str(column).lower()

    # A1: Hygiene / meta (never normalize)
    if any(tok in name for tok in HYGIENE_TOKENS):
        return FeatureRole.HYGIENE, "A1:name"

    # A2: Regime / context (probabilities/booleans)
    if any(tok in name for tok in REGIME_TOKENS):
        return FeatureRole.REGIME, "A2:name"

    # A3: Risk / uncertainty (note: avoid misclassifying spread_z)
    risk_name_match = any(tok in name for tok in RISK_TOKENS)
    if "spread_z" in name:
        risk_name_match = False
    if risk_name_match:
        return FeatureRole.RISK, "A3:name"

    # A4: Predictive
    if any(tok in name for tok in PREDICTIVE_TOKENS):
        return FeatureRole.PREDICTIVE, "A4:name"

    # A2b: Boolish heuristic (high confidence) for columns with no matching tokens.
    # Keep this after keyword matching to avoid hijacking e.g. iv_level-like risk signals.
    if _is_boolish(series):
        return FeatureRole.REGIME, "A2:boolish"

    # A5: Fallback to family intent (only for ambiguous features)
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
