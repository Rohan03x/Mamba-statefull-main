"""Canonical family metadata registry (Phase-2 / Stage-B).

This module defines the family-level metadata schema requested for production-like
feature governance.

Key principle: this metadata should be conservative by default. If a family is
missing or unspecified, downstream systems should treat it as `mixed` intent and
`allow_direct_alpha=false`.

The registry is designed to be editable via JSON (checked into the repo) and
loaded at runtime by tools/pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, cast


class PrimaryIntent(str, Enum):
    PREDICTIVE = "predictive"
    RISK = "risk"
    REGIME = "regime"
    HYGIENE = "hygiene"
    MIXED = "mixed"


class UpdateFrequency(str, Enum):
    DAILY = "daily"
    INTRADAY = "intraday"
    EVENT = "event"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"


class DecayProfile(str, Enum):
    NONE = "none"
    FAST = "fast"
    SLOW = "slow"
    TWO_PHASE = "two_phase"


class UpdateLatencyClass(str, Enum):
    """When the feature becomes safely tradable relative to the session label."""

    UNKNOWN = "unknown"
    INTRADAY = "intraday"
    SAME_DAY_CLOSE = "same_day_close"
    T_PLUS_1 = "t_plus_1"
    T_PLUS_2_PLUS = "t_plus_2_plus"


class ExpectedSparsity(str, Enum):
    UNKNOWN = "unknown"
    DENSE = "dense"
    SPARSE = "sparse"
    EVENT = "event"


@dataclass(frozen=True)
class FamilyMetadata:
    # B1. Canonical family metadata schema
    family_id: str
    primary_intent: PrimaryIntent
    update_frequency: UpdateFrequency
    decay_profile: DecayProfile
    is_event_driven: bool
    # Cross-sectional scope policy.
    # - "false": primarily time-series/per-symbol features
    # - "true": explicitly cross-sectional features
    # - "conditional": mixed; some features are cross-sectional and some are not
    is_cross_sectional: str
    # Alpha usability policy.
    # - "false": do not use as standalone alpha
    # - "true": can be used as standalone alpha
    # - "conditional": can be alpha with gating / strict leakage controls
    allow_direct_alpha: str

    # B2. Allowed-usage defaults (portfolio governance).
    # These are family-level defaults; per-column inference may further restrict.
    risk_scale_ok: bool = False
    gating_ok: bool = False
    veto_ok: bool = False

    # B3. Point-in-time + latency governance.
    requires_point_in_time: bool = False
    update_latency_class: UpdateLatencyClass = UpdateLatencyClass.UNKNOWN

    # B4. Decay + sparsity hints (for modeling + hygiene audits).
    decay_half_life_days: float = 0.0
    expected_sparsity: ExpectedSparsity = ExpectedSparsity.UNKNOWN
    notes: str = ""


def conservative_default(family_id: str) -> FamilyMetadata:
    """Conservative default for unknown/missing families.

    We intentionally default to MIXED + IRREGULAR and disable direct alpha.
    """

    return FamilyMetadata(
        family_id=str(family_id),
        primary_intent=PrimaryIntent.MIXED,
        update_frequency=UpdateFrequency.IRREGULAR,
        decay_profile=DecayProfile.NONE,
        is_event_driven=False,
        is_cross_sectional="false",
        allow_direct_alpha="false",
        risk_scale_ok=False,
        gating_ok=False,
        veto_ok=False,
        requires_point_in_time=False,
        update_latency_class=UpdateLatencyClass.UNKNOWN,
        decay_half_life_days=0.0,
        expected_sparsity=ExpectedSparsity.UNKNOWN,
        notes="",
    )


def _coerce_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    token = str(v or "").strip().lower()
    if token in {"1", "true", "yes", "y", "t"}:
        return True
    if token in {"0", "false", "no", "n", "f"}:
        return False
    return default


def _coerce_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _coerce_latency(v: object) -> UpdateLatencyClass:
    token = str(v or "").strip().lower()
    if not token:
        return UpdateLatencyClass.UNKNOWN
    try:
        return UpdateLatencyClass(token)
    except Exception:
        return UpdateLatencyClass.UNKNOWN


def _coerce_sparsity(v: object) -> ExpectedSparsity:
    token = str(v or "").strip().lower()
    if not token:
        return ExpectedSparsity.UNKNOWN
    try:
        return ExpectedSparsity(token)
    except Exception:
        return ExpectedSparsity.UNKNOWN


def _coerce_allow_direct_alpha(v: object) -> str:
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


def _coerce_cross_sectional(v: object) -> str:
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


def _coerce_record(obj: Mapping[str, object]) -> Optional[FamilyMetadata]:
    try:
        family_id = str(obj.get("family_id") or "").strip()
        if not family_id:
            return None
        return FamilyMetadata(
            family_id=family_id,
            primary_intent=PrimaryIntent(str(obj.get("primary_intent"))),
            update_frequency=UpdateFrequency(str(obj.get("update_frequency"))),
            decay_profile=DecayProfile(str(obj.get("decay_profile"))),
            is_event_driven=bool(obj.get("is_event_driven")),
            is_cross_sectional=_coerce_cross_sectional(obj.get("is_cross_sectional")),
            allow_direct_alpha=_coerce_allow_direct_alpha(obj.get("allow_direct_alpha")),

            risk_scale_ok=_coerce_bool(obj.get("risk_scale_ok"), default=False),
            gating_ok=_coerce_bool(obj.get("gating_ok"), default=False),
            veto_ok=_coerce_bool(obj.get("veto_ok"), default=False),
            requires_point_in_time=_coerce_bool(obj.get("requires_point_in_time"), default=False),
            update_latency_class=_coerce_latency(obj.get("update_latency_class")),
            decay_half_life_days=_coerce_float(obj.get("decay_half_life_days"), default=0.0),
            expected_sparsity=_coerce_sparsity(obj.get("expected_sparsity")),

            notes=str(obj.get("notes") or ""),
        )
    except Exception:
        return None


def load_family_metadata(path: Optional[Path]) -> Dict[str, FamilyMetadata]:
    """Load metadata registry from JSON.

    Supported formats:
    - List[record]
    - Dict[family_id -> record]
    """

    if path is None:
        return {}

    try:
        payload: Any = json.loads(Path(path).read_text())
    except Exception:
        return {}

    records: List[Mapping[str, object]] = []
    if isinstance(payload, list):
        payload_list = cast(List[object], payload)
        for item in payload_list:
            if isinstance(item, dict):
                records.append(cast(Mapping[str, object], item))
    elif isinstance(payload, dict):
        payload_dict = cast(Dict[object, object], payload)
        # allow both {family_id: {..}} and full record dicts
        for k, v in payload_dict.items():
            if isinstance(v, dict):
                v_dict = cast(Dict[object, object], v)
                rec_dict: Dict[str, object] = {str(kk): vv for kk, vv in v_dict.items()}
                rec_dict.setdefault("family_id", str(k))
                records.append(rec_dict)
    else:
        return {}

    out: Dict[str, FamilyMetadata] = {}
    for r in records:
        meta = _coerce_record(r)
        if meta is None:
            continue
        out[meta.family_id] = meta
    return out


def validate_family_metadata(
    *,
    registry: Mapping[str, FamilyMetadata],
    known_families: Iterable[str],
) -> Tuple[bool, List[str]]:
    """Validate that all known families have a metadata record."""

    issues: List[str] = []
    known = [str(f) for f in known_families]

    missing = sorted([f for f in known if f not in registry])
    if missing:
        issues.append(f"missing metadata for {len(missing)} families: {', '.join(missing)}")

    # Basic sanity: unique family_id
    ids = list(registry.keys())
    if len(set(ids)) != len(ids):
        issues.append("duplicate family_id entries detected")

    ok = len(issues) == 0
    return ok, issues


def dump_registry(registry: Mapping[str, FamilyMetadata]) -> List[Dict[str, object]]:
    """Dump registry to JSON-serializable list of dicts (sorted by family_id)."""

    out: List[Dict[str, object]] = []
    for fam in sorted(registry.keys()):
        out.append(cast(Dict[str, object], asdict(registry[fam])))
    return out
